"""Browse-grid cards and thumbnail grid widgets for the ProbeFlow GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QCursor, QFont, QPixmap
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QMenu, QScrollArea,
    QVBoxLayout, QWidget,
)

from probeflow.gui.models import SxmFile, VertFile, _card_meta_str
from probeflow.gui.rendering import (
    DEFAULT_CMAP_KEY,
    THUMBNAIL_CHANNEL_DEFAULT,
    THUMBNAIL_CHANNEL_OPTIONS,
    resolve_thumbnail_plane_index,
)
from probeflow.gui.workers import SpecThumbnailLoader, ThumbnailLoader
from probeflow.core.scan_loader import load_scan

# ── Browse cards ──────────────────────────────────────────────────────────────
class _BrowseCard(QFrame):
    """Shared thumbnail-card behavior for image and spectroscopy entries."""

    clicked        = Signal(object, bool)  # SxmFile, ctrl_held
    double_clicked = Signal(object)

    CARD_W = 200
    CARD_H = 220
    IMG_W  = 180
    IMG_H  = 150

    def __init__(self, entry, t: dict, meta_text: str, parent=None):
        super().__init__(parent)
        self.entry     = entry
        self._t        = t
        self._sel      = False

        self.setFixedSize(self.CARD_W, self.CARD_H)
        self.setCursor(QCursor(Qt.PointingHandCursor))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 6)
        lay.setSpacing(3)

        self.img_lbl = QLabel()
        self.img_lbl.setFixedSize(self.IMG_W, self.IMG_H)
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setText("…")

        lbl_text = entry.stem if len(entry.stem) <= 22 else entry.stem[:20] + ".."
        self.name_lbl = QLabel(lbl_text)
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.name_lbl.setFont(QFont("Helvetica", 10))

        self.meta_lbl = QLabel(meta_text)
        self.meta_lbl.setAlignment(Qt.AlignCenter)
        self.meta_lbl.setFont(QFont("Helvetica", 9))

        lay.addWidget(self.img_lbl)
        lay.addWidget(self.name_lbl)
        lay.addWidget(self.meta_lbl)
        self._refresh_style()

    def set_pixmap(self, pixmap: QPixmap):
        self.img_lbl.setPixmap(
            pixmap.scaled(self.IMG_W, self.IMG_H,
                          Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.img_lbl.setText("")

    def set_selected(self, val: bool):
        self._sel = val
        self._refresh_style()

    def apply_theme(self, t: dict):
        self._t = t
        self._refresh_style()

    def _refresh_style(self):
        t = self._t
        if self._sel:
            bg, border, bw, fg = t["card_sel"], t["accent_bg"], 3, t["accent_bg"]
        else:
            bg, border, bw, fg = t["card_bg"], t["sep"], 1, t["card_fg"]
        selector = self.__class__.__name__
        self.setStyleSheet(f"""
            {selector} {{
                background-color: {bg};
                border: {bw}px solid {border};
                border-radius: 6px;
            }}
            {selector}:hover {{
                border: {bw}px solid {t["accent_bg"]};
            }}
        """)
        self.name_lbl.setStyleSheet(f"color: {fg}; background: transparent;")
        self.meta_lbl.setStyleSheet(f"color: {t['sub_fg']}; background: transparent;")
        self.img_lbl.setStyleSheet(f"color: {t['sub_fg']}; background: transparent;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            ctrl = bool(event.modifiers() & Qt.ControlModifier)
            self.clicked.emit(self.entry, ctrl)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self.entry)
        super().mouseDoubleClickEvent(event)


class ScanCard(_BrowseCard):
    """Single image thumbnail card."""

    context_action_requested = Signal(object, str)  # SxmFile, action key

    def __init__(self, entry: SxmFile, t: dict, parent=None):
        super().__init__(entry, t, _card_meta_str(entry), parent=parent)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        a_features = QAction("Send to FeatureCounting", self)
        a_features.triggered.connect(
            lambda: self.context_action_requested.emit(self.entry, "features"))
        menu.addAction(a_features)

        a_meta_csv = QAction("Export metadata as CSV…", self)
        a_meta_csv.triggered.connect(
            lambda: self.context_action_requested.emit(self.entry, "export_metadata_csv"))
        menu.addAction(a_meta_csv)

        a_meta_show = QAction("Show full metadata", self)
        a_meta_show.triggered.connect(
            lambda: self.context_action_requested.emit(self.entry, "show_metadata"))
        menu.addAction(a_meta_show)

        menu.exec(event.globalPos())


# ── SpecCard ──────────────────────────────────────────────────────────────────
class SpecCard(_BrowseCard):
    """Thumbnail card for a .VERT spectroscopy file."""

    def __init__(self, entry: VertFile, t: dict, parent=None):
        sweep = entry.sweep_type.replace("_", " ") if entry.sweep_type != "unknown" else "VERT"
        pts   = f"{entry.n_points} pts" if entry.n_points else ""
        meta  = "  |  ".join(filter(None, [entry.measurement_label, sweep, pts]))
        super().__init__(entry, t, meta, parent=parent)


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

    GAP = 10

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self._t    = t
        self._pool = QThreadPool.globalInstance()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

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
        self._cards:          dict[str, Union[ScanCard, SpecCard]] = {}
        self._entries:        list[Union[SxmFile, VertFile]]       = []
        self._selected:       set[str]                         = set()
        self._primary:        Optional[str]                    = None
        self._thumbnail_colormap: str                          = DEFAULT_CMAP_KEY
        self._thumbnail_processing: dict                       = {}
        self._thumbnail_clip: tuple[float, float]              = (1.0, 99.0)
        self._thumbnail_channel: str                           = THUMBNAIL_CHANNEL_DEFAULT
        self._load_token                                       = object()
        self._current_cols: int                                = 1
        self._filter_mode: str                                 = "all"

        # empty-state placeholder
        self._empty_lbl = QLabel("Open a folder to browse SXM scans")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._empty_lbl.setFont(QFont("Helvetica", 12))
        self._grid.addWidget(self._empty_lbl, 0, 0)

    # ── Public API ────────────────────────────────────────────────────────────
    def load(self, entries: list[SxmFile], folder_path: str = ""):
        self._entries         = entries
        self._selected        = set()
        self._primary         = None
        self._load_token      = object()

        if folder_path:
            p = Path(folder_path)
            n_sxm  = sum(1 for e in entries if isinstance(e, SxmFile))
            n_vert = sum(1 for e in entries if isinstance(e, VertFile))
            parts = []
            if n_sxm:
                parts.append(f"{n_sxm} scan{'s' if n_sxm != 1 else ''}")
            if n_vert:
                parts.append(f"{n_vert} spec{'s' if n_vert != 1 else ''}")
            self._path_lbl.setText(f"{p.name}  ({', '.join(parts) if parts else '0 files'})")

        # clear grid
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards = {}

        if not entries:
            self._empty_lbl = QLabel("No .sxm or .VERT files found in this folder")
            self._empty_lbl.setAlignment(Qt.AlignCenter)
            self._empty_lbl.setFont(QFont("Helvetica", 12))
            self._grid.addWidget(self._empty_lbl, 0, 0)
            return

        cols = self._calc_cols()
        self._current_cols = cols
        dark = True  # will be updated via apply_theme if needed
        for entry in entries:
            if isinstance(entry, VertFile):
                card = SpecCard(entry, self._t)
            else:
                card = ScanCard(entry, self._t)
                card.context_action_requested.connect(self.card_context_action)
            card.clicked.connect(self._on_card_click)
            card.double_clicked.connect(self._on_card_dbl)
            self._cards[entry.stem] = card

        # Populate the grid honouring the current filter (fresh loads default
        # to the filter previously set via apply_filter()).
        self._relayout_filtered()

        # load all thumbnails
        token = self._load_token
        for entry in entries:
            if isinstance(entry, VertFile):
                loader = SpecThumbnailLoader(entry, token,
                                             SpecCard.IMG_W, SpecCard.IMG_H)
            else:
                loader = self._make_thumbnail_loader(entry, token)
            loader.signals.loaded.connect(self._on_thumb)
            self._pool.start(loader)

    def _make_thumbnail_loader(self, entry: SxmFile, token) -> ThumbnailLoader:
        clip_low, clip_high = self._thumbnail_clip
        return ThumbnailLoader(entry, self._thumbnail_colormap, token,
                               ScanCard.IMG_W, ScanCard.IMG_H,
                               clip_low, clip_high,
                               processing=self._thumbnail_processing or None,
                               thumbnail_channel=self._thumbnail_channel)

    def _rerender_scan_thumbnails(self) -> int:
        token = self._load_token
        count = 0
        for entry in self._entries:
            if not isinstance(entry, SxmFile) or entry.stem not in self._cards:
                continue
            loader = self._make_thumbnail_loader(entry, token)
            loader.signals.loaded.connect(self._on_thumb)
            self._pool.start(loader)
            count += 1
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
            scan = load_scan(entry.path)
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
        for card in self._cards.values():
            card.apply_theme(t)

    # ── Slots ──────────────────────────────────────────────────────────────────
    @Slot(str, QPixmap, object)
    def _on_thumb(self, stem: str, pixmap: QPixmap, token):
        if token is not self._load_token:
            return
        card = self._cards.get(stem)
        if card:
            card.set_pixmap(pixmap)

    def _on_card_click(self, entry: SxmFile, ctrl: bool):
        if ctrl:
            # toggle this card in/out of selection
            if entry.stem in self._selected:
                self._selected.discard(entry.stem)
                self._cards[entry.stem].set_selected(False)
                self._primary = next(iter(self._selected), None) if self._selected else None
            else:
                self._selected.add(entry.stem)
                self._cards[entry.stem].set_selected(True)
                self._primary = entry.stem
        else:
            # single select: deselect all others
            for stem in list(self._selected):
                c = self._cards.get(stem)
                if c:
                    c.set_selected(False)
            self._selected = {entry.stem}
            self._primary  = entry.stem
            self._cards[entry.stem].set_selected(True)

        self.selection_changed.emit(len(self._selected))
        if self._primary:
            primary_entry = next(
                (e for e in self._entries if e.stem == self._primary), None)
            if primary_entry:
                self.entry_selected.emit(primary_entry)

    def _on_card_dbl(self, entry: SxmFile):
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
            card = self._cards.get(entry.stem)
            if not card or not self._is_entry_visible(entry):
                continue
            row, col = divmod(i, cols)
            self._grid.addWidget(card, row, col, Qt.AlignTop | Qt.AlignLeft)
            card.setVisible(True)
            i += 1
