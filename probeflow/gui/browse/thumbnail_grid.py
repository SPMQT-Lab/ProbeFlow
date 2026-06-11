"""Thumbnail grid and folder navigation for browse mode."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QCursor, QImage, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from probeflow.gui.models import FolderEntry, SxmFile, VertFile, browse_entry_key
from probeflow.gui.rendering import (
    DEFAULT_CMAP_KEY,
    THUMBNAIL_CHANNEL_DEFAULT,
    THUMBNAIL_CHANNEL_OPTIONS,
    resolve_thumbnail_plane_index,
)
from probeflow.gui.workers import (
    FolderIndexLoader,
    FolderThumbnailLoader,
    SpecThumbnailLoader,
    ThumbnailLoader,
)

from .breadcrumbs import _BreadcrumbBar
from .cards import FolderCard, ScanCard, SpecCard, _BrowseCard
from .helpers import _browse_attr, _CARD_SIZE_PRESETS, _is_deleted_qt_runtime_error

# Thumbnail renders share QThreadPool.globalInstance() with interactive loads
# (ViewerLoader, ChannelPreviewLoader, _ScanLoadWorker). Queue them below the
# default priority (0) so opening a viewer never waits behind a screenful of
# queued thumbnail reads on a slow network folder. Already-running thumbnail
# workers still finish, but everything queued yields.
_THUMBNAIL_PRIORITY = -1


# ── ThumbnailGrid ─────────────────────────────────────────────────────────────
class ThumbnailGrid(QWidget):
    """
    Browse panel: folder toolbar + thumbnail grid.

    - All images share a global thumbnail appearance.
    - Click = single-select; Ctrl+click = multi-select toggle.
    - Double-click = open full-size image viewer.
    """
    entry_selected    = Signal(object)   # primary SxmFile for sidebar
    selection_changed = Signal(int)      # count of selected items
    view_requested    = Signal(object)   # SxmFile to open in full-size viewer
    card_context_action = Signal(object, str)  # entry, action key — re-emitted from cards
    folder_changed    = Signal(object)   # current folder Path (after navigation)
    root_changed      = Signal(object)   # root folder Path (after open-folder dialog)

    GAP = 10

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self._t    = t
        self._pool = QThreadPool.globalInstance()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Breadcrumb bar (back/up + segments) ─────────────────────────────
        self._breadcrumb = _BreadcrumbBar(t)
        self._breadcrumb.segment_clicked.connect(self._on_breadcrumb_clicked)
        self._breadcrumb.back_requested.connect(self._on_back_requested)
        self._breadcrumb.up_requested.connect(self._on_up_requested)
        outer.addWidget(self._breadcrumb)

        # ── Path strip (folder name + count) ────────────────────────────────
        self._toolbar = QWidget()
        self._toolbar.setFixedHeight(28)
        tb_lay = QHBoxLayout(self._toolbar)
        tb_lay.setContentsMargins(10, 4, 8, 4)
        tb_lay.setSpacing(0)

        self._path_lbl = QLabel("No folder open")
        self._path_lbl.setFont(ui_font(10))
        self._path_lbl.setStyleSheet("background: transparent;")

        self._refresh_btn = QPushButton("⟳")
        self._refresh_btn.setFixedSize(24, 20)
        self._refresh_btn.setFont(ui_font(11))
        self._refresh_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._refresh_btn.setToolTip(
            "Rescan the current folder for new files (F5).\n"
            "Use this when the STM has saved new scans while ProbeFlow is open.")
        self._refresh_btn.setEnabled(False)   # enabled once a folder is open
        self._refresh_btn.clicked.connect(self.refresh)

        tb_lay.addWidget(self._path_lbl, 1)
        tb_lay.addWidget(self._refresh_btn)
        outer.addWidget(self._toolbar)

        # ── Scroll area with grid ────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QFrame.NoFrame)

        self._content = QWidget()
        self._grid    = QGridLayout(self._content)
        self._grid.setSpacing(self.GAP)
        self._grid.setContentsMargins(self.GAP, self.GAP, self.GAP, self.GAP)
        self._grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll, 1)

        # state
        self._cards:          dict[str, Union[ScanCard, SpecCard, FolderCard]] = {}
        self._entries:        list[Union[SxmFile, VertFile, FolderEntry]]      = []
        self._selected:       set[str]                         = set()
        self._primary:        Optional[str]                    = None
        self._thumbnail_colormap: str                          = DEFAULT_CMAP_KEY
        self._thumbnail_processing: dict                       = {}
        self._per_scan_processing: dict                        = {}  # path_str → processing dict
        self._thumbnail_clip: tuple[float, float]              = (1.0, 99.0)
        self._thumbnail_channel: str                           = THUMBNAIL_CHANNEL_DEFAULT
        self._load_token                                       = object()
        self._nav_token                                        = object()
        self._current_cols: int                                = 1
        self._filter_mode: str                                 = "all"
        self._thumbnail_size_name: str                         = "large"
        self._thumbnail_pending: dict[str, Union[SxmFile, VertFile, FolderEntry]] = {}
        self._background_thumbnail_batch_size: int             = 2
        # Timer-sliced card construction state (see _build_card_batch).
        # The build has its own generation token: _load_token is replaced by
        # appearance changes (_rerender_scan_thumbnails) to drop in-flight
        # thumbnail renders, and reusing it for the build cancelled the
        # remaining slices — the tail cards of a large folder were never
        # constructed. _card_build_token is reset only when the grid
        # re-renders (navigation / refresh / load), the one case where the
        # queued slices really are stale.
        self._card_build_queue: list                           = []
        self._card_build_token                                 = object()
        self._next_grid_index: int                             = 0

        self._visible_thumb_timer = QTimer(self)
        self._visible_thumb_timer.setSingleShot(True)
        self._visible_thumb_timer.timeout.connect(self._queue_visible_thumbnails)
        self._thumbnail_bg_timer = QTimer(self)
        self._thumbnail_bg_timer.setSingleShot(True)
        self._thumbnail_bg_timer.timeout.connect(self._queue_background_thumbnail_batch)
        self._scroll.verticalScrollBar().valueChanged.connect(
            self._schedule_visible_thumbnail_refresh)

        # navigation state
        self._root:        Optional[Path] = None
        self._current_dir: Optional[Path] = None
        self._history:     list[Path]     = []  # back stack of previous dirs
        # The folder whose entries the grid is actually showing. _current_dir
        # is optimistic (set at navigation intent, before the off-thread index
        # lands); when an index fails the two diverge and _current_dir is
        # restored from this so refresh / history / card events stay
        # consistent with what is on screen.
        self._rendered_dir: Optional[Path] = None

        # empty-state placeholder
        self._empty_lbl = QLabel("Open a folder to browse scans and spectra")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setFont(ui_font(12))
        self._grid.addWidget(self._empty_lbl, 0, 0)

    # ── Public API ────────────────────────────────────────────────────────────
    def set_root(self, path: Path):
        """Set a new browse root (called by 'Open folder…') and navigate to it.

        Resets navigation history and clears any cached selection.
        """
        path = Path(path)
        self._root = path
        self._history = []
        self.root_changed.emit(path)
        self._navigate(path)

    def navigate_to(self, path: Path):
        """Navigate to *path*, pushing the current folder onto the history."""
        path = Path(path)
        if self._current_dir is not None and path != self._current_dir:
            self._history.append(self._current_dir)
        self._navigate(path)

    def current_dir(self) -> Optional[Path]:
        return self._current_dir

    def root(self) -> Optional[Path]:
        return self._root

    def refresh(self) -> None:
        """Rescan the current folder and update the grid with any new files.

        Does not push the current folder onto the navigation history, so
        Back / breadcrumb state is unchanged. Safe to call at any time;
        a no-op if no folder is open yet.
        """
        if self._current_dir is not None:
            self._navigate(self._current_dir)

    def _navigate(self, path: Path):
        """Index *path* shallowly off-thread, then rebuild the grid.

        The breadcrumb and path strip update immediately (navigation intent);
        the current grid stays interactive until the index arrives.  Cold
        network folders previously froze the GUI here for the whole index.
        """
        path = Path(path)
        self._current_dir = path
        if self._root is None:
            self._root = path
        self._breadcrumb.set_state(
            self._root, self._current_dir,
            can_go_back=bool(self._history),
        )
        self._refresh_btn.setEnabled(False)  # re-enabled when the index lands
        self._path_lbl.setText(f"Indexing {path.name}…")
        self._nav_token = object()
        loader = FolderIndexLoader(path, self._nav_token)
        loader.signals.indexed.connect(self._on_folder_indexed)
        loader.signals.failed.connect(self._on_folder_index_failed)
        # Default (interactive) priority — must not queue behind thumbnails.
        self._pool.start(loader)

    @Slot(object, object, object)
    def _on_folder_indexed(self, path, index, token):
        if token is not getattr(self, "_nav_token", None):
            return
        from probeflow.gui.models import (
            FolderEntry as _FE,
            _scan_items_to_sxm,
            _spec_items_to_vert,
        )

        scan_items   = [it for it in index.files if it.item_type == "scan"]
        spec_items   = [it for it in index.files if it.item_type == "spectrum"]
        sxm_entries  = _scan_items_to_sxm(scan_items)
        vert_entries = _spec_items_to_vert(spec_items)

        folder_entries = [_FE.from_index(s) for s in index.subfolders]
        # Folders sort alphabetically already (from indexing layer); files
        # follow, sorted by stem like the legacy view.
        file_entries = sorted(sxm_entries + vert_entries, key=lambda e: e.stem)
        entries: list[Union[SxmFile, VertFile, FolderEntry]] = list(folder_entries) + list(file_entries)

        self._refresh_btn.setEnabled(True)
        self._rendered_dir = Path(path)
        self._render_entries(entries)
        self.folder_changed.emit(path)

    @Slot(object, str, object)
    def _on_folder_index_failed(self, path, message, token):
        if token is not getattr(self, "_nav_token", None):
            return
        self._refresh_btn.setEnabled(True)
        self._path_lbl.setText(f"Could not open {Path(path).name}: {message}")
        # The grid still shows the previously rendered folder, but
        # _current_dir was optimistically set to the failed path: refresh()
        # would retarget the failed folder and the next navigation would push
        # it onto the Back history. Restore the displayed folder as current
        # (only within the same root — a failed set_root has nothing
        # consistent to restore to).
        rendered = self._rendered_dir
        if (
            rendered is not None
            and rendered != self._current_dir
            and self._root is not None
            and rendered.is_relative_to(self._root)
        ):
            self._current_dir = rendered
            # Drop the history entry the failed navigation pushed (it is the
            # folder we just restored as current), so Back returns to where
            # the user actually was before that.
            if self._history and self._history[-1] == rendered:
                self._history.pop()
            self._breadcrumb.set_state(
                self._root, self._current_dir,
                can_go_back=bool(self._history),
            )

    def load(self, entries: list, folder_path: str = ""):
        """Legacy entry point: render a flat list of entries (no navigation).

        Kept for backwards compatibility with code paths that already have a
        prebuilt entry list and don't want shallow folder discovery.
        """
        # Reset navigation since this isn't a tree-aware load.
        if folder_path:
            p = Path(folder_path)
            self._root = p
            self._current_dir = p
            self._rendered_dir = p
            self._history = []
            self._breadcrumb.set_state(p, p, can_go_back=False)
        self._render_entries(entries)

    def _render_entries(self, entries: list):
        self._entries    = entries
        had_selection    = bool(self._selected)
        self._selected   = set()
        self._primary    = None
        self._load_token = object()
        # Re-rendering drops the selection; announce it so selection-driven
        # UI (hints, menus) resets instead of going stale.
        if had_selection:
            self.selection_changed.emit(0)

        n_folders = sum(1 for e in entries if isinstance(e, FolderEntry))
        n_sxm     = sum(1 for e in entries if isinstance(e, SxmFile))
        n_vert    = sum(1 for e in entries if isinstance(e, VertFile))
        if self._current_dir is not None:
            parts = []
            if n_folders:
                parts.append(f"{n_folders} folder{'s' if n_folders != 1 else ''}")
            if n_sxm:
                parts.append(f"{n_sxm} scan{'s' if n_sxm != 1 else ''}")
            if n_vert:
                parts.append(f"{n_vert} spec{'s' if n_vert != 1 else ''}")
            self._path_lbl.setText(
                f"{self._current_dir.name}  "
                f"({', '.join(parts) if parts else '0 items'})"
            )

        # clear grid (gridded widgets), then any filtered-out cards that were
        # never re-added to the layout — those are parentless and would
        # otherwise linger until garbage collection.
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for card in self._cards.values():
            if card.parent() is None:
                card.deleteLater()
        self._cards = {}
        self._card_build_queue = []
        self._next_grid_index = 0

        if not entries:
            self._empty_lbl = QLabel("No scans, spectra, or subfolders here")
            self._empty_lbl.setAlignment(Qt.AlignCenter)
            self._empty_lbl.setFont(ui_font(12))
            self._grid.addWidget(self._empty_lbl, 0, 0)
            return

        self._current_cols = self._calc_cols()
        # Build the first batch synchronously so the folder appears instantly,
        # then drain the rest in zero-delay timer slices: constructing ~1000
        # card widgets in one loop froze the GUI for seconds, independent of
        # disk speed. The build token ties the slices to this render; it is
        # deliberately not _load_token, which appearance changes replace
        # mid-build (see __init__).
        self._card_build_queue = list(entries)
        self._card_build_token = object()
        self._build_card_batch(self._card_build_token)

        # Queue all thumbnails now; scheduling skips entries whose cards are
        # not built yet and picks them up on later refresh ticks.
        self._prepare_thumbnail_queue(entries)

    # How many cards to construct per slice. ~120 keeps each slice well under
    # a frame budget on typical hardware while finishing 1000 entries in <10
    # event-loop turns.
    _CARD_BUILD_BATCH = 120

    def _build_card_batch(self, token) -> None:
        if token is not self._card_build_token or not self._card_build_queue:
            return
        batch = self._card_build_queue[: self._CARD_BUILD_BATCH]
        del self._card_build_queue[: self._CARD_BUILD_BATCH]
        cols = self._current_cols
        for entry in batch:
            key = self._key_for(entry)
            if isinstance(entry, FolderEntry):
                card = FolderCard(entry, self._t)
                card.folder_activated.connect(self._on_folder_activated)
            elif isinstance(entry, VertFile):
                card = SpecCard(entry, self._t)
            else:
                card = ScanCard(entry, self._t)
                card.context_action_requested.connect(self.card_context_action)
            card.clicked.connect(self._on_card_click)
            card.double_clicked.connect(self._on_card_dbl)
            if self._thumbnail_size_name == "small":
                card.set_compact_mode(True)
            self._cards[key] = card
            # Append in entry order honouring the current filter; a filter or
            # column change mid-build triggers _relayout_filtered, which
            # re-places built cards and resets _next_grid_index.
            if self._is_entry_visible(entry):
                row, col = divmod(self._next_grid_index, cols)
                self._grid.addWidget(card, row, col, Qt.AlignTop | Qt.AlignLeft)
                card.setVisible(True)
                self._next_grid_index += 1
            else:
                card.setVisible(False)
        if self._card_build_queue:
            QTimer.singleShot(
                0, self, lambda tok=token: self._build_card_batch(tok)
            )
            self._schedule_visible_thumbnail_refresh(delay_ms=0)

    @staticmethod
    def _key_for(entry) -> str:
        """Stable per-entry key for the _cards dict.

        File cards include entry type and path so same-stem scan/spectrum files
        can never collide.
        """
        return browse_entry_key(entry)

    def _on_breadcrumb_clicked(self, path: Path):
        if self._current_dir is not None and path != self._current_dir:
            self._history.append(self._current_dir)
            self._navigate(path)

    def _on_back_requested(self):
        if not self._history:
            return
        previous = self._history.pop()
        self._navigate(previous)

    def _on_up_requested(self):
        if self._current_dir is None or self._root is None:
            return
        if self._current_dir == self._root:
            return
        parent = self._current_dir.parent
        if self._current_dir != parent:
            self._history.append(self._current_dir)
            self._navigate(parent)

    def _on_folder_activated(self, path):
        self.navigate_to(Path(path))

    def _make_thumbnail_loader(self, entry: SxmFile, token) -> ThumbnailLoader:
        clip_low, clip_high = self._thumbnail_clip
        # Merge global thumbnail processing with any per-scan override.
        # Per-scan processing (saved from the viewer) takes precedence.
        per_scan = self._per_scan_processing.get(str(entry.path), {})
        if per_scan:
            proc = {**self._thumbnail_processing, **per_scan}
        else:
            proc = self._thumbnail_processing or None
        Loader = _browse_attr("ThumbnailLoader", ThumbnailLoader)
        return Loader(entry, self._thumbnail_colormap, token,
                               ScanCard.IMG_W, ScanCard.IMG_H,
                               clip_low, clip_high,
                               processing=proc,
                               thumbnail_channel=self._thumbnail_channel)

    def _entry_needs_thumbnail(self, entry) -> bool:
        if isinstance(entry, FolderEntry):
            return bool(entry.sample_scan_paths)
        return isinstance(entry, (SxmFile, VertFile))

    def _prepare_thumbnail_queue(self, entries: list) -> None:
        self._thumbnail_bg_timer.stop()
        self._visible_thumb_timer.stop()
        self._thumbnail_pending = {
            self._key_for(entry): entry
            for entry in entries
            if self._entry_needs_thumbnail(entry)
        }
        if not self._thumbnail_pending:
            return
        self._schedule_visible_thumbnail_refresh(delay_ms=0)
        self._schedule_background_thumbnail_batch(delay_ms=300)

    def _schedule_visible_thumbnail_refresh(self, *_args, delay_ms: int = 50) -> None:
        if not self._thumbnail_pending:
            return
        self._visible_thumb_timer.start(max(0, int(delay_ms)))

    def _schedule_background_thumbnail_batch(self, delay_ms: int = 120) -> None:
        if not self._thumbnail_pending:
            self._thumbnail_bg_timer.stop()
            return
        self._thumbnail_bg_timer.start(max(0, int(delay_ms)))

    def _visible_thumbnail_keys(self) -> list[str]:
        if not self._thumbnail_pending:
            return []
        self._grid.activate()
        bar = self._scroll.verticalScrollBar()
        top = bar.value()
        bottom = top + self._scroll.viewport().height()
        preload = max(_BrowseCard.CARD_H * 2, 320)
        keys: list[str] = []
        for entry in self._entries:
            key = self._key_for(entry)
            if key not in self._thumbnail_pending:
                continue
            card = self._cards.get(key)
            if card is None or card.isHidden():
                continue
            geom = card.geometry()
            if geom.bottom() >= top - preload and geom.top() <= bottom + preload:
                keys.append(key)
        return keys

    def _start_thumbnail_for_key(self, key: str) -> bool:
        entry = self._thumbnail_pending.get(key)
        if entry is None:
            return False
        # Card not constructed yet (timer-sliced build still draining): keep
        # the entry pending so the result has somewhere to land; a later
        # refresh tick retries.
        if key not in self._cards:
            return False
        self._thumbnail_pending.pop(key, None)
        token = self._load_token
        if isinstance(entry, FolderEntry):
            Loader = _browse_attr("FolderThumbnailLoader", FolderThumbnailLoader)
            loader = Loader(
                str(entry.path),
                list(entry.sample_scan_paths),
                self._thumbnail_colormap,
                token,
                FolderCard.THUMB_W, FolderCard.THUMB_H,
                self._thumbnail_clip[0], self._thumbnail_clip[1],
                thumbnail_channel=self._thumbnail_channel,
            )
            loader.signals.loaded.connect(self._on_folder_thumbs)
            self._pool.start(loader, _THUMBNAIL_PRIORITY)
            return True
        if isinstance(entry, VertFile):
            Loader = _browse_attr("SpecThumbnailLoader", SpecThumbnailLoader)
            loader = Loader(entry, token, SpecCard.IMG_W, SpecCard.IMG_H)
            loader.signals.loaded.connect(self._on_thumb)
            self._pool.start(loader, _THUMBNAIL_PRIORITY)
            return True
        if isinstance(entry, SxmFile):
            loader = self._make_thumbnail_loader(entry, token)
            loader.signals.loaded.connect(self._on_thumb)
            self._pool.start(loader, _THUMBNAIL_PRIORITY)
            return True
        return False

    def _queue_visible_thumbnails(self) -> None:
        for key in self._visible_thumbnail_keys():
            self._start_thumbnail_for_key(key)
        if self._thumbnail_pending:
            self._schedule_background_thumbnail_batch()

    def _background_thumbnail_slots(self) -> int:
        active_fn = getattr(self._pool, "activeThreadCount", None)
        max_fn = getattr(self._pool, "maxThreadCount", None)
        if not callable(active_fn) or not callable(max_fn):
            return self._background_thumbnail_batch_size
        try:
            active = int(active_fn())
            max_threads = max(1, int(max_fn()))
        except Exception:
            return self._background_thumbnail_batch_size
        target_active = max(1, min(4, max_threads // 2))
        return max(0, min(self._background_thumbnail_batch_size, target_active - active))

    def _queue_background_thumbnail_batch(self) -> None:
        if not self._thumbnail_pending:
            return
        for key in self._visible_thumbnail_keys():
            self._start_thumbnail_for_key(key)
        slots = self._background_thumbnail_slots()
        if slots <= 0:
            self._schedule_background_thumbnail_batch(delay_ms=180)
            return
        started = 0
        for key in list(self._thumbnail_pending):
            if started >= slots:
                break
            if self._start_thumbnail_for_key(key):
                started += 1
        self._schedule_background_thumbnail_batch()

    def _rerender_scan_thumbnails(self) -> int:
        # Replacing _load_token drops in-flight renders that still carry the
        # old appearance. The card build (own token) is unaffected.
        self._load_token = object()
        # Include entries whose cards are not built yet (timer-sliced build
        # still draining): _start_thumbnail_for_key keeps them pending until
        # their card exists, so they render with the new appearance once
        # built. Excluding them here left tail cards permanently blank.
        entries: list[Union[SxmFile, FolderEntry]] = [
            entry for entry in self._entries
            if isinstance(entry, SxmFile)
            or (isinstance(entry, FolderEntry) and entry.sample_scan_paths)
        ]
        self._prepare_thumbnail_queue(entries)
        return sum(1 for entry in entries if isinstance(entry, SxmFile))

    def set_thumbnail_colormap(self, colormap_key: str) -> int:
        """Set the global browse thumbnail colormap and re-render scan cards."""
        self._thumbnail_colormap = colormap_key or DEFAULT_CMAP_KEY
        return self._rerender_scan_thumbnails()

    def set_thumbnail_channel(self, channel: str) -> int:
        """Set the global browse thumbnail channel and re-render scan cards."""
        if channel not in THUMBNAIL_CHANNEL_OPTIONS:
            channel = THUMBNAIL_CHANNEL_DEFAULT
        self._thumbnail_channel = channel
        return self._rerender_scan_thumbnails()

    def set_thumbnail_align_rows(self, mode: str | None) -> int:
        """Set the global browse thumbnail row-alignment preview mode."""
        value = (mode or "").strip().lower()
        if value in ("median", "mean"):
            self._thumbnail_processing = {"align_rows": value}
        else:
            self._thumbnail_processing = {}
        return self._rerender_scan_thumbnails()

    def thumbnail_channel(self) -> str:
        return self._thumbnail_channel

    def thumbnail_colormap(self) -> str:
        return self._thumbnail_colormap

    def thumbnail_processing(self) -> dict:
        return dict(self._thumbnail_processing)

    def set_entry_processing(self, path_str: str, proc: dict) -> None:
        """Save per-scan processing and re-render that scan's thumbnail.

        Called by ProbeFlowWindow when an ImageViewerDialog closes with a
        non-empty processing state, so the Browse thumbnail reflects the
        viewer processing the user applied.
        """
        if proc:
            self._per_scan_processing[path_str] = dict(proc)
        else:
            self._per_scan_processing.pop(path_str, None)
        # Find the entry and re-render just its thumbnail.
        token = self._load_token
        for entry in self._entries:
            if isinstance(entry, SxmFile) and str(entry.path) == path_str:
                if self._key_for(entry) in self._cards:
                    self._thumbnail_pending.pop(self._key_for(entry), None)
                    loader = self._make_thumbnail_loader(entry, token)
                    loader.signals.loaded.connect(self._on_thumb)
                    self._pool.start(loader, _THUMBNAIL_PRIORITY)
                break

    def thumbnail_plane_index_for_entry(self, entry: SxmFile) -> int:
        # Header-only metadata read: this runs on the GUI thread when a viewer
        # is opened, and a full load_scan here meant transferring the entire
        # file (then the viewer transferred it again) before the window showed.
        try:
            from probeflow.core.metadata import read_scan_metadata
            names = read_scan_metadata(entry.path).plane_names
            return resolve_thumbnail_plane_index(list(names), self._thumbnail_channel)
        except Exception:
            return 0

    def get_card_state(self, entry_or_key) -> tuple[str, tuple[float, float], dict]:
        """Return viewer-opening state for a browse entry.

        Browse align-row correction is only a thumbnail preview aid, so the
        full viewer opens raw unless the user applies processing there.
        """
        return self._thumbnail_colormap, self._thumbnail_clip, {}

    def get_entries(self) -> list[Union[SxmFile, VertFile]]:
        return self._entries

    def get_selected(self) -> set[str]:
        return self._selected.copy()

    def get_primary(self) -> Optional[str]:
        return self._primary

    def get_primary_entry(self):
        if self._primary is None:
            return None
        return next(
            (e for e in self._entries
             if not isinstance(e, FolderEntry) and self._key_for(e) == self._primary),
            None,
        )

    def get_selected_entries(self) -> list[Union[SxmFile, VertFile]]:
        return [
            e for e in self._entries
            if not isinstance(e, FolderEntry) and self._key_for(e) in self._selected
        ]

    def apply_theme(self, t: dict):
        self._t = t
        self._content.setStyleSheet(f"background-color: {t['main_bg']};")
        self._toolbar.setStyleSheet(f"background-color: {t['main_bg']};")
        self._path_lbl.setStyleSheet(f"color: {t['sub_fg']}; background: transparent;")
        self._breadcrumb.apply_theme(t)
        stale_keys = []
        for key, card in list(self._cards.items()):
            try:
                card.apply_theme(t)
            except RuntimeError as exc:
                if _is_deleted_qt_runtime_error(exc):
                    stale_keys.append(key)
                else:
                    raise
        for key in stale_keys:
            self._cards.pop(key, None)

    # ── Slots ──────────────────────────────────────────────────────────────────
    # Workers emit QImage (QPixmap is GUI-thread-only); convert here.
    @Slot(str, QImage, object)
    def _on_thumb(self, key: str, image: QImage, token):
        if token is not self._load_token:
            return
        card = self._cards.get(key)
        if card is None:
            return
        if image is None or image.isNull():
            # Render failed — show a visible placeholder instead of silently
            # leaving the card blank/stale.
            img_lbl = getattr(card, "img_lbl", None)
            if img_lbl is not None:
                img_lbl.setText("render\nfailed")
            return
        card.set_pixmap(QPixmap.fromImage(image))

    @Slot(str, list, object)
    def _on_folder_thumbs(self, folder_key: str, images: list, token):
        if token is not self._load_token:
            return
        key = f"folder:{folder_key}"
        card = self._cards.get(key)
        if isinstance(card, FolderCard):
            card.set_thumbnails([
                None if img is None or img.isNull() else QPixmap.fromImage(img)
                for img in images
            ])

    def _on_card_click(self, entry, ctrl: bool):
        # Folders are not selectable — selecting one would have no effect on
        # the info panel and would just confuse the file selection state.
        if isinstance(entry, FolderEntry):
            return
        key = self._key_for(entry)
        if ctrl:
            # toggle this card in/out of selection
            if key in self._selected:
                self._selected.discard(key)
                self._cards[key].set_selected(False)
                self._primary = next(iter(self._selected), None) if self._selected else None
            else:
                self._selected.add(key)
                self._cards[key].set_selected(True)
                self._primary = key
        else:
            # single select: deselect all others
            for s in list(self._selected):
                c = self._cards.get(s)
                if c:
                    c.set_selected(False)
            self._selected = {key}
            self._primary  = key
            self._cards[key].set_selected(True)

        self.selection_changed.emit(len(self._selected))
        if self._primary:
            primary_entry = self.get_primary_entry()
            if primary_entry:
                self.entry_selected.emit(primary_entry)

    def _on_card_dbl(self, entry):
        # FolderCard emits its own folder_activated signal; the generic
        # double_clicked path is only for files.
        if isinstance(entry, FolderEntry):
            return
        self.view_requested.emit(entry)

    # ── Layout helpers ─────────────────────────────────────────────────────────
    def _calc_cols(self) -> int:
        vw = self._scroll.viewport().width()
        if vw < 10:
            vw = 880
        return max(1, (vw - self.GAP) // (ScanCard.CARD_W + self.GAP))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._entries:
            # Receiver-aware overload — cancelled if this widget is deleted
            # before the timer fires.
            QTimer.singleShot(60, self, self._relayout)
            self._schedule_visible_thumbnail_refresh(delay_ms=80)

    def _relayout(self):
        if not self._entries:
            return
        new_cols = self._calc_cols()
        if new_cols == self._current_cols:
            return
        self._current_cols = new_cols
        self._relayout_filtered()

    def apply_filter(self, mode: str):
        """Switch between showing all entries, only images, or only spectra.

        Does not clear the selection or re-scan the folder; it only re-lays
        out the grid so hidden cards don't leave empty slots.
        """
        if mode not in ("all", "images", "spectra"):
            mode = "all"
        self._filter_mode = mode
        self._relayout_filtered()
        self._schedule_visible_thumbnail_refresh(delay_ms=0)

    def _is_entry_visible(self, entry) -> bool:
        # Folders are navigation aids; never hide them based on a file filter.
        if isinstance(entry, FolderEntry):
            return True
        mode = self._filter_mode
        if mode == "images":
            return isinstance(entry, SxmFile)
        if mode == "spectra":
            return isinstance(entry, VertFile)
        return True  # "all"

    def _relayout_filtered(self):
        """Re-populate the grid with only cards matching the current filter.

        Selections are preserved on the cards themselves; we merely remove
        all widgets from the QGridLayout and re-add visible ones in row/col
        order, which avoids gaps caused by ``setVisible(False)``.
        """
        if not self._entries:
            return
        # Remove every card from the layout (do not delete the widgets).
        for card in self._cards.values():
            self._grid.removeWidget(card)
            card.setVisible(False)

        cols = self._calc_cols()
        self._current_cols = cols

        i = 0
        for entry in self._entries:
            card = self._cards.get(self._key_for(entry))
            if not card or not self._is_entry_visible(entry):
                continue
            row, col = divmod(i, cols)
            self._grid.addWidget(card, row, col, Qt.AlignTop | Qt.AlignLeft)
            card.setVisible(True)
            i += 1
        # Keep incremental (timer-sliced) card placement appending after the
        # cards this full pass just laid out.
        self._next_grid_index = i

    def set_thumbnail_size(self, name: str) -> None:
        """Switch all cards to a size preset ("large" or "small")."""
        sizes = _CARD_SIZE_PRESETS.get(name)
        if sizes is None:
            return
        self._thumbnail_size_name = name
        compact = (name == "small")
        _BrowseCard.CARD_W = sizes["CARD_W"]
        _BrowseCard.CARD_H = sizes["CARD_H"]
        _BrowseCard.IMG_W  = sizes["IMG_W"]
        _BrowseCard.IMG_H  = sizes["IMG_H"]
        for card in self._cards.values():
            card.resize_to(sizes["CARD_W"], sizes["CARD_H"],
                           sizes["IMG_W"], sizes["IMG_H"])
            card.set_compact_mode(compact)
        self._relayout_filtered()
        self._schedule_visible_thumbnail_refresh(delay_ms=0)
