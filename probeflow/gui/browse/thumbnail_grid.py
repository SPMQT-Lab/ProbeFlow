"""Thumbnail grid and folder navigation for browse mode."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from probeflow.core.scan_loader import load_scan as _default_load_scan
from probeflow.gui.models import FolderEntry, SxmFile, VertFile
from probeflow.gui.rendering import (
    DEFAULT_CMAP_KEY,
    THUMBNAIL_CHANNEL_DEFAULT,
    THUMBNAIL_CHANNEL_OPTIONS,
    resolve_thumbnail_plane_index,
)
from probeflow.gui.workers import FolderThumbnailLoader, SpecThumbnailLoader, ThumbnailLoader

from .breadcrumbs import _BreadcrumbBar
from .cards import FolderCard, ScanCard, SpecCard, _BrowseCard
from .helpers import _browse_attr, _CARD_SIZE_PRESETS, _is_deleted_qt_runtime_error

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
        self._path_lbl.setFont(QFont("Helvetica", 10))
        self._path_lbl.setStyleSheet("background: transparent;")

        tb_lay.addWidget(self._path_lbl, 1)
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
        self._thumbnail_clip: tuple[float, float]              = (1.0, 99.0)
        self._thumbnail_channel: str                           = THUMBNAIL_CHANNEL_DEFAULT
        self._load_token                                       = object()
        self._current_cols: int                                = 1
        self._filter_mode: str                                 = "all"
        self._thumbnail_size_name: str                         = "large"

        # navigation state
        self._root:        Optional[Path] = None
        self._current_dir: Optional[Path] = None
        self._history:     list[Path]     = []  # back stack of previous dirs

        # empty-state placeholder
        self._empty_lbl = QLabel("Open a folder to browse SXM scans")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setFont(QFont("Helvetica", 12))
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

    def _navigate(self, path: Path):
        """Index *path* shallowly and rebuild the grid + breadcrumb."""
        from probeflow.core.indexing import index_folder_shallow
        from probeflow.gui.models import (
            FolderEntry as _FE,
            _scan_items_to_sxm,
            _spec_items_to_vert,
        )

        index = index_folder_shallow(path, include_errors=True)
        scan_items   = [it for it in index.files if it.item_type == "scan"]
        spec_items   = [it for it in index.files if it.item_type == "spectrum"]
        sxm_entries  = _scan_items_to_sxm(scan_items)
        vert_entries = _spec_items_to_vert(spec_items)

        folder_entries = [_FE.from_index(s) for s in index.subfolders]
        # Folders sort alphabetically already (from indexing layer); files
        # follow, sorted by stem like the legacy view.
        file_entries = sorted(sxm_entries + vert_entries, key=lambda e: e.stem)
        entries: list[Union[SxmFile, VertFile, FolderEntry]] = list(folder_entries) + list(file_entries)

        self._current_dir = path
        if self._root is None:
            self._root = path
        self._breadcrumb.set_state(
            self._root, self._current_dir,
            can_go_back=bool(self._history),
        )
        self._render_entries(entries)
        self.folder_changed.emit(path)

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
            self._history = []
            self._breadcrumb.set_state(p, p, can_go_back=False)
        self._render_entries(entries)

    def _render_entries(self, entries: list):
        self._entries    = entries
        self._selected   = set()
        self._primary    = None
        self._load_token = object()

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

        # clear grid
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards = {}

        if not entries:
            self._empty_lbl = QLabel("No scans, spectra, or subfolders here")
            self._empty_lbl.setAlignment(Qt.AlignCenter)
            self._empty_lbl.setFont(QFont("Helvetica", 12))
            self._grid.addWidget(self._empty_lbl, 0, 0)
            return

        cols = self._calc_cols()
        self._current_cols = cols
        for entry in entries:
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

        # Populate the grid honouring the current filter.
        self._relayout_filtered()

        # Queue async thumbnail rendering for every card.
        token = self._load_token
        for entry in entries:
            if isinstance(entry, FolderEntry):
                if entry.sample_scan_paths:
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
                    self._pool.start(loader)
            elif isinstance(entry, VertFile):
                Loader = _browse_attr("SpecThumbnailLoader", SpecThumbnailLoader)
                loader = Loader(entry, token,
                                             SpecCard.IMG_W, SpecCard.IMG_H)
                loader.signals.loaded.connect(self._on_thumb)
                self._pool.start(loader)
            else:
                loader = self._make_thumbnail_loader(entry, token)
                loader.signals.loaded.connect(self._on_thumb)
                self._pool.start(loader)

    @staticmethod
    def _key_for(entry) -> str:
        """Stable per-entry key for the _cards dict.

        Folders use ``folder:<path>`` so a subfolder named the same as a sibling
        scan stem can never collide.
        """
        if isinstance(entry, FolderEntry):
            return f"folder:{entry.path}"
        return entry.stem

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
        Loader = _browse_attr("ThumbnailLoader", ThumbnailLoader)
        return Loader(entry, self._thumbnail_colormap, token,
                               ScanCard.IMG_W, ScanCard.IMG_H,
                               clip_low, clip_high,
                               processing=self._thumbnail_processing or None,
                               thumbnail_channel=self._thumbnail_channel)

    def _rerender_scan_thumbnails(self) -> int:
        token = self._load_token
        count = 0
        for entry in self._entries:
            if isinstance(entry, SxmFile):
                if entry.stem not in self._cards:
                    continue
                loader = self._make_thumbnail_loader(entry, token)
                loader.signals.loaded.connect(self._on_thumb)
                self._pool.start(loader)
                count += 1
            elif isinstance(entry, FolderEntry) and entry.sample_scan_paths:
                key = self._key_for(entry)
                if key not in self._cards:
                    continue
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
                self._pool.start(loader)
        return count

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

    def thumbnail_plane_index_for_entry(self, entry: SxmFile) -> int:
        try:
            scan = _browse_attr("load_scan", _default_load_scan)(entry.path)
            return resolve_thumbnail_plane_index(scan.plane_names, self._thumbnail_channel)
        except Exception:
            return 0

    def get_card_state(self, stem: str) -> tuple[str, tuple[float, float], dict]:
        """Return viewer-opening state for a stem.

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
    @Slot(str, QPixmap, object)
    def _on_thumb(self, stem: str, pixmap: QPixmap, token):
        if token is not self._load_token:
            return
        card = self._cards.get(stem)
        if card:
            card.set_pixmap(pixmap)

    @Slot(str, list, object)
    def _on_folder_thumbs(self, folder_key: str, pixmaps: list, token):
        if token is not self._load_token:
            return
        key = f"folder:{folder_key}"
        card = self._cards.get(key)
        if isinstance(card, FolderCard):
            card.set_thumbnails(pixmaps)

    def _on_card_click(self, entry, ctrl: bool):
        # Folders are not selectable — selecting one would have no effect on
        # the info panel and would just confuse the file selection state.
        if isinstance(entry, FolderEntry):
            return
        stem = entry.stem
        if ctrl:
            # toggle this card in/out of selection
            if stem in self._selected:
                self._selected.discard(stem)
                self._cards[stem].set_selected(False)
                self._primary = next(iter(self._selected), None) if self._selected else None
            else:
                self._selected.add(stem)
                self._cards[stem].set_selected(True)
                self._primary = stem
        else:
            # single select: deselect all others
            for s in list(self._selected):
                c = self._cards.get(s)
                if c:
                    c.set_selected(False)
            self._selected = {stem}
            self._primary  = stem
            self._cards[stem].set_selected(True)

        self.selection_changed.emit(len(self._selected))
        if self._primary:
            primary_entry = next(
                (e for e in self._entries
                 if not isinstance(e, FolderEntry) and e.stem == self._primary),
                None)
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
            QTimer.singleShot(60, self._relayout)

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
