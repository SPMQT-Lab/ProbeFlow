"""Thumbnail cards for browse grid entries."""

from __future__ import annotations

from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QCursor, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QVBoxLayout, QWidget

from probeflow.gui.models import FolderEntry, SxmFile, VertFile, _card_meta_str

from .helpers import _card_compact_meta_str

# ── Browse cards ──────────────────────────────────────────────────────────────
class _BrowseCard(QFrame):
    """Shared thumbnail-card behavior for image and spectroscopy entries."""

    clicked        = Signal(object, bool)  # SxmFile, ctrl_held
    double_clicked = Signal(object)

    CARD_W = 200
    CARD_H = 220
    IMG_W  = 180
    IMG_H  = 150

    def __init__(self, entry, t: dict, meta_text: str,
                 compact_meta_text: str = "", parent=None):
        super().__init__(parent)
        self.entry     = entry
        self._t        = t
        self._sel      = False
        self._orig_pixmap: QPixmap | None = None
        self._full_meta    = meta_text
        self._compact_meta = compact_meta_text or meta_text
        self._compact_mode = False

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
        self.name_lbl.setFont(ui_font(10))
        # Don't reserve space for name_lbl when it's hidden in compact mode.
        _sp = self.name_lbl.sizePolicy()
        _sp.setRetainSizeWhenHidden(False)
        self.name_lbl.setSizePolicy(_sp)

        self.meta_lbl = QLabel(meta_text)
        self.meta_lbl.setAlignment(Qt.AlignCenter)
        self.meta_lbl.setFont(ui_font(9))

        lay.addWidget(self.img_lbl)
        lay.addWidget(self.name_lbl)
        lay.addWidget(self.meta_lbl)
        self._refresh_style()

    def set_pixmap(self, pixmap: QPixmap):
        if getattr(self, "img_lbl", None) is None:
            return
        self._orig_pixmap = pixmap
        self.img_lbl.setPixmap(
            pixmap.scaled(self.IMG_W, self.IMG_H,
                          Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.img_lbl.setText("")

    def resize_to(self, card_w: int, card_h: int, img_w: int, img_h: int) -> None:
        self.setFixedSize(card_w, card_h)
        if getattr(self, "img_lbl", None) is not None:
            self.img_lbl.setFixedSize(img_w, img_h)
            if self._orig_pixmap is not None:
                self.img_lbl.setPixmap(
                    self._orig_pixmap.scaled(img_w, img_h,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_compact_mode(self, compact: bool) -> None:
        """In compact mode: hide the name label and show only V/I meta; filename as tooltip."""
        if self._compact_mode == compact:
            return
        self._compact_mode = compact
        if compact:
            self.name_lbl.setVisible(False)
            self.meta_lbl.setText(self._compact_meta)
            self.setToolTip(self.entry.stem)
        else:
            self.name_lbl.setVisible(True)
            self.meta_lbl.setText(self._full_meta)
            self.setToolTip("")

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
        if getattr(self, "img_lbl", None) is not None:
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
        super().__init__(entry, t, _card_meta_str(entry),
                         compact_meta_text=_card_compact_meta_str(entry),
                         parent=parent)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
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
        super().__init__(entry, t, meta,
                         compact_meta_text=sweep,
                         parent=parent)


# ── FolderCard ────────────────────────────────────────────────────────────────
class FolderCard(_BrowseCard):
    """Card representing a navigable subfolder, with up to 3 preview thumbnails.

    Double-click navigates into the subfolder rather than opening a viewer.
    """

    folder_activated = Signal(object)  # Path

    NUM_THUMBS = 3
    THUMB_W    = 56
    THUMB_H    = 56

    def __init__(self, entry: FolderEntry, t: dict, parent=None):
        # A "+" marks counts that hit the indexing peek's file budget — they
        # are lower bounds, not exact totals (big network trees).
        plus = "+" if getattr(entry, "counts_capped", False) else ""
        meta_parts = []
        if entry.n_scans:
            meta_parts.append(f"{entry.n_scans}{plus} scan{'s' if entry.n_scans != 1 else ''}")
        if entry.n_specs:
            meta_parts.append(f"{entry.n_specs}{plus} spec{'s' if entry.n_specs != 1 else ''}")
        meta = "  |  ".join(meta_parts) if meta_parts else "(empty)"
        super().__init__(entry, t, meta, parent=parent)

        # Tag the folder name with a leading icon glyph so users immediately
        # recognise it as a folder rather than a file in the same grid.
        self.name_lbl.setText(f"📁  {self.name_lbl.text()}")

        # Replace the single img_lbl with a horizontal strip of small previews.
        outer_lay = self.layout()
        idx = outer_lay.indexOf(self.img_lbl)
        outer_lay.removeWidget(self.img_lbl)
        self.img_lbl.deleteLater()
        self.img_lbl = None

        strip = QWidget()
        strip.setFixedSize(self.IMG_W, self.IMG_H)
        strip_lay = QHBoxLayout(strip)
        strip_lay.setContentsMargins(0, 0, 0, 0)
        strip_lay.setSpacing(6)
        strip_lay.setAlignment(Qt.AlignCenter)

        self._thumb_lbls: list[QLabel] = []
        for _ in range(self.NUM_THUMBS):
            lbl = QLabel()
            lbl.setFixedSize(self.THUMB_W, self.THUMB_H)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setText("")
            strip_lay.addWidget(lbl)
            self._thumb_lbls.append(lbl)

        outer_lay.insertWidget(idx, strip, alignment=Qt.AlignHCenter)
        self._strip = strip
        self._refresh_style()

    # FolderCards do not use the single-pixmap API.
    def set_pixmap(self, pixmap: QPixmap):  # pragma: no cover - intentional no-op
        return

    def set_thumbnails(self, pixmaps: list):
        for lbl, pix in zip(self._thumb_lbls, pixmaps):
            if pix is None:
                lbl.setText("·")
                continue
            lbl.setPixmap(pix.scaled(self.THUMB_W, self.THUMB_H,
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation))
            lbl.setText("")

    def _refresh_style(self):
        super()._refresh_style()
        # Make the inner thumbnail labels visually distinct from the card bg.
        t = self._t
        for lbl in getattr(self, "_thumb_lbls", []):
            lbl.setStyleSheet(
                f"background-color: {t['main_bg']};"
                f"color: {t['sub_fg']};"
                f"border-radius: 3px;"
            )

    def mouseDoubleClickEvent(self, event):
        # Diverge from _BrowseCard: we want navigation, not viewer-open.
        if event.button() == Qt.LeftButton:
            self.folder_activated.emit(self.entry.path)
        # Skip super().mouseDoubleClickEvent to avoid emitting double_clicked.

    def contextMenuEvent(self, event):
        # No context actions for folders for now.
        return
