"""Browse-grid cards and thumbnail grid widgets for the ProbeFlow GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QCursor, QFont, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMenu, QPushButton,
    QScrollArea, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from probeflow.gui.models import FolderEntry, PLANE_NAMES, SxmFile, VertFile, _card_meta_str
from probeflow.gui.rendering import (
    CMAP_KEY,
    CMAP_NAMES,
    DEFAULT_CMAP_KEY,
    DEFAULT_CMAP_LABEL,
    THUMBNAIL_CHANNEL_DEFAULT,
    THUMBNAIL_CHANNEL_OPTIONS,
    resolve_thumbnail_plane_index,
)
from probeflow.gui.workers import (
    ChannelLoader,
    ChannelSignals,
    FolderThumbnailLoader,
    SpecThumbnailLoader,
    ThumbnailLoader,
)
from probeflow.core.scan_loader import load_scan


def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


class BrowseToolPanel(QWidget):
    """Left-side control panel for browsing and live thumbnail appearance."""
    open_folder_requested      = Signal()
    colormap_changed           = Signal(str)
    thumbnail_align_changed    = Signal(str)
    map_spectra_requested      = Signal()
    filter_changed             = Signal(str)   # "all" | "images" | "spectra"
    thumbnail_channel_changed  = Signal(str)

    def __init__(self, t: dict, cfg: dict, parent=None):
        super().__init__(parent)
        self._t            = t
        self._filter_mode  = cfg.get("browse_filter", "all")
        self._build(cfg)

    def _build(self, cfg: dict):
        # Wrap everything in a scroll area so nothing gets clipped on small screens
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 6)
        lay.setSpacing(6)

        # ── Open folder button ─────────────────────────────────────────────────
        open_btn = QPushButton("Open folder…")
        open_btn.setFont(QFont("Helvetica", 9))
        open_btn.setFixedHeight(30)
        open_btn.setCursor(QCursor(Qt.PointingHandCursor))
        open_btn.setObjectName("accentBtn")
        open_btn.clicked.connect(self.open_folder_requested.emit)
        lay.addWidget(open_btn)

        # ── Filter toggle (All / Images / Spectra) ─────────────────────────────
        filter_row = QWidget()
        filter_lay = QHBoxLayout(filter_row)
        filter_lay.setContentsMargins(0, 0, 0, 0)
        filter_lay.setSpacing(0)

        self._filter_group = QButtonGroup(self)
        self._filter_group.setExclusive(True)
        self._filter_btns: dict[str, QPushButton] = {}
        _modes = [("All", "all"), ("Images", "images"), ("Spectra", "spectra")]
        for i, (label, mode) in enumerate(_modes):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFont(QFont("Helvetica", 9))
            btn.setFixedHeight(26)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            # Segmented-control shape: rounded corners only on outer edges
            if i == 0:
                btn.setObjectName("segBtnLeft")
            elif i == len(_modes) - 1:
                btn.setObjectName("segBtnRight")
            else:
                btn.setObjectName("segBtnMid")
            btn.clicked.connect(lambda _c=False, m=mode: self._on_filter_click(m))
            self._filter_group.addButton(btn)
            filter_lay.addWidget(btn, 1)
            self._filter_btns[mode] = btn

        # Set the initially checked button from config.
        initial = self._filter_mode if self._filter_mode in self._filter_btns else "all"
        self._filter_btns[initial].setChecked(True)
        self._filter_mode = initial

        lay.addWidget(filter_row)
        lay.addWidget(_sep())

        # ── Thumbnail appearance ──────────────────────────────────────────────
        appearance_lbl = QLabel("Thumbnail appearance")
        appearance_lbl.setFont(QFont("Helvetica", 11, QFont.Bold))
        lay.addWidget(appearance_lbl)

        cm_lbl = QLabel("Colormap")
        cm_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        lay.addWidget(cm_lbl)

        self.cmap_cb = QComboBox()
        self.cmap_cb.addItems(CMAP_NAMES)
        self.cmap_cb.setCurrentText(cfg.get("colormap", DEFAULT_CMAP_LABEL))
        self.cmap_cb.setFont(QFont("Helvetica", 10))
        self.cmap_cb.currentTextChanged.connect(self._on_colormap_changed)
        lay.addWidget(self.cmap_cb)

        thumb_lbl = QLabel("Thumbnail channel")
        thumb_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        lay.addWidget(thumb_lbl)

        self.thumbnail_channel_cb = QComboBox()
        self.thumbnail_channel_cb.addItems(THUMBNAIL_CHANNEL_OPTIONS)
        self.thumbnail_channel_cb.setCurrentText(THUMBNAIL_CHANNEL_DEFAULT)
        self.thumbnail_channel_cb.setFont(QFont("Helvetica", 10))
        self.thumbnail_channel_cb.setToolTip(
            "Choose which forward scan channel is used for browse thumbnails. "
            "Files without that channel fall back to Z."
        )
        self.thumbnail_channel_cb.currentTextChanged.connect(
            self.thumbnail_channel_changed.emit)
        lay.addWidget(self.thumbnail_channel_cb)

        align_lbl = QLabel("Align rows")
        align_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        lay.addWidget(align_lbl)
        self.align_rows_cb = QComboBox()
        self.align_rows_cb.addItems(["None", "Median", "Mean"])
        self.align_rows_cb.setCurrentText("None")
        self.align_rows_cb.setFont(QFont("Helvetica", 10))
        self.align_rows_cb.setToolTip(
            "Preview-only thumbnail row alignment. Full-size viewer data opens raw."
        )
        self.align_rows_cb.currentTextChanged.connect(self._on_align_changed)
        lay.addWidget(self.align_rows_cb)
        lay.addWidget(_sep())

        self._map_spectra_btn = QPushButton("Map spectra to images\u2026")
        self._map_spectra_btn.setFont(QFont("Helvetica", 9))
        self._map_spectra_btn.setFixedHeight(28)
        self._map_spectra_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._map_spectra_btn.setToolTip(
            "Pick the parent image for each .VERT spectrum in the current "
            "folder. Spectra without a mapping show no marker on any image. "
            "You can also map per-image inside the viewer.")
        self._map_spectra_btn.clicked.connect(self.map_spectra_requested.emit)
        lay.addWidget(self._map_spectra_btn)

        lay.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # ── Slots ──────────────────────────────────────────────────────────────────
    def _on_colormap_changed(self):
        cmap_key = CMAP_KEY.get(self.cmap_cb.currentText(), DEFAULT_CMAP_KEY)
        self.colormap_changed.emit(cmap_key)

    def _on_align_changed(self, text: str):
        self.thumbnail_align_changed.emit(text)

    def _on_filter_click(self, mode: str):
        self._filter_mode = mode
        self.filter_changed.emit(mode)

    # ── Public API ─────────────────────────────────────────────────────────────
    def get_filter_mode(self) -> str:
        return self._filter_mode

    def set_filter_mode(self, mode: str) -> None:
        if mode not in self._filter_btns:
            mode = "all"
        self._filter_mode = mode
        btn = self._filter_btns[mode]
        if not btn.isChecked():
            btn.setChecked(True)

    def update_selection_hint(self, n: int):
        if n == 0:
            return
        elif n == 1:
            return
        else:
            return

    def apply_theme(self, t: dict):
        self._t = t


# ── Browse info panel (RIGHT) ─────────────────────────────────────────────────
class BrowseInfoPanel(QWidget):
    """Right-side info panel: selected file name, channel thumbnails, metadata."""

    def __init__(self, t: dict, cfg: dict, parent=None):
        super().__init__(parent)
        self._t         = t
        self._pool      = QThreadPool.globalInstance()
        self._ch_token  = object()
        self._meta_rows: list[tuple[str, str]] = []
        self._clip_low  = 1.0
        self._clip_high = 99.0
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        self._main_lay = lay
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(3)

        summary = QWidget()
        summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        summary_lay = QVBoxLayout(summary)
        summary_lay.setContentsMargins(0, 0, 0, 0)
        summary_lay.setSpacing(3)

        self.name_lbl = QLabel("No scan selected")
        self.name_lbl.setFont(QFont("Helvetica", 10, QFont.Bold))
        self.name_lbl.setWordWrap(True)
        self.name_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        summary_lay.addWidget(self.name_lbl)

        # Compact key scan summary. Keep this tight so channels sit high.
        qi_widget = QWidget()
        qi_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        qi_grid = QGridLayout()
        qi_grid.setHorizontalSpacing(8)
        qi_grid.setVerticalSpacing(2)
        qi_grid.setContentsMargins(0, 0, 0, 0)
        qi_widget.setLayout(qi_grid)
        self._qi: dict[str, QLabel] = {}
        _QI_ROWS = [("Pixels", "pixels"), ("Size", "size"),
                    ("Bias",   "bias"),   ("Setp.", "setp")]
        for i, (title, key) in enumerate(_QI_ROWS):
            r, c = divmod(i, 2)
            t_lbl = QLabel(title + ":")
            t_lbl.setFont(QFont("Helvetica", 8))
            v_lbl = QLabel("—")
            v_lbl.setFont(QFont("Helvetica", 10, QFont.Bold))
            qi_grid.addWidget(t_lbl, r, c * 2)
            qi_grid.addWidget(v_lbl, r, c * 2 + 1)
            self._qi[key] = v_lbl
        summary_lay.addWidget(qi_widget)
        summary_lay.addWidget(_sep())

        ch_hdr = QLabel("Channels")
        ch_hdr.setFont(QFont("Helvetica", 11, QFont.Bold))
        summary_lay.addWidget(ch_hdr)

        self._ch_widget = QWidget()
        self._ch_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self._ch_grid = QGridLayout()
        self._ch_grid.setSpacing(8)
        self._ch_grid.setContentsMargins(0, 0, 0, 0)
        self._ch_widget.setLayout(self._ch_grid)
        self._ch_cells: list[QWidget] = []
        self._ch_img_lbls:  list[QLabel] = []
        self._ch_name_lbls: list[QLabel] = []
        self._set_channel_preview_slots(PLANE_NAMES)
        summary_lay.addWidget(self._ch_widget)
        summary_lay.addWidget(_sep())

        # Full metadata is hidden behind a toggle. The quick-info grid above
        # (Pixels / Size / Bias / Setpoint) is what users want at a glance;
        # the full header table is dense and only useful occasionally.
        self._meta_toggle = QPushButton("[+] Show all metadata")
        self._meta_toggle.setFont(QFont("Helvetica", 9, QFont.Bold))
        self._meta_toggle.setFixedHeight(24)
        self._meta_toggle.setCursor(QCursor(Qt.PointingHandCursor))
        self._meta_toggle.setToolTip(
            "Expand to show the full scan header (also accessible via "
            "right-click → Show full metadata).")
        self._meta_toggle.clicked.connect(self._toggle_meta)
        summary_lay.addWidget(self._meta_toggle)
        lay.addWidget(summary)

        self._meta_widget = QWidget()
        meta_lay = QVBoxLayout(self._meta_widget)
        meta_lay.setContentsMargins(0, 4, 0, 0)
        meta_lay.setSpacing(4)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search…")
        self.search_box.setFont(QFont("Helvetica", 10))
        self.search_box.setFixedHeight(28)
        self.search_box.textChanged.connect(self._filter_meta)
        meta_lay.addWidget(self.search_box)

        self.meta_table = QTableWidget(0, 2)
        self.meta_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.meta_table.setWordWrap(True)
        self.meta_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.meta_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.meta_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.meta_table.setColumnWidth(0, 92)
        self.meta_table.verticalHeader().setVisible(False)
        self.meta_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.meta_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.meta_table.setAlternatingRowColors(True)
        self.meta_table.setFont(QFont("Helvetica", 10))
        self.meta_table.verticalHeader().setDefaultSectionSize(22)
        self.meta_table.setShowGrid(False)
        meta_lay.addWidget(self.meta_table, 1)
        self._meta_widget.setVisible(False)
        lay.addWidget(self._meta_widget, 0)
        self._bottom_spacer = QWidget()
        self._bottom_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.addWidget(self._bottom_spacer, 1)

    def _toggle_meta(self):
        vis = not self._meta_widget.isVisible()
        self._meta_widget.setVisible(vis)
        self._meta_toggle.setText(
            "[-] Hide all metadata" if vis else "[+] Show all metadata")
        self._main_lay.setStretchFactor(self._meta_widget, 1 if vis else 0)
        self._main_lay.setStretchFactor(self._bottom_spacer, 0 if vis else 1)
        self._bottom_spacer.setVisible(not vis)
        if vis:
            self.meta_table.resizeRowsToContents()

    # ── Public API ─────────────────────────────────────────────────────────────
    def show_entry(self, entry: SxmFile, colormap_key: str,
                    processing: dict = None):
        self.name_lbl.setText(entry.stem)
        self._qi["pixels"].setText(f"{entry.Nx} × {entry.Ny}")
        self._qi["size"].setText(f"{entry.scan_nm:.1f} nm" if entry.scan_nm is not None else "—")
        self._qi["bias"].setText(f"{entry.bias_mv:.0f} mV" if entry.bias_mv is not None else "—")
        self._qi["setp"].setText(f"{entry.current_pa:.1f} pA" if entry.current_pa is not None else "—")
        self.load_channels(entry, colormap_key, processing=None)
        self._load_metadata(entry)

    def show_vert_entry(self, entry: VertFile):
        self.name_lbl.setText(entry.stem)
        sweep = entry.sweep_type.replace("_", " ") if entry.sweep_type != "unknown" else "—"
        self._qi["pixels"].setText(sweep)
        self._qi["size"].setText(f"{entry.n_points} pts" if entry.n_points else "—")
        self._qi["bias"].setText(f"{entry.bias_mv:.0f} mV" if entry.bias_mv is not None else "—")
        freq = entry.spec_freq_hz
        self._qi["setp"].setText(f"{freq:.0f} Hz" if freq is not None else "—")
        for lbl in self._ch_img_lbls:
            lbl.clear()
            lbl.setText("—")
        self._load_vert_metadata(entry)

    def _load_vert_metadata(self, entry: VertFile):
        from probeflow.io.spectroscopy import parse_spec_header
        try:
            hdr = parse_spec_header(entry.path)
        except Exception:
            hdr = {}
        rows: list[tuple[str, str]] = [
            ("Sweep type", entry.sweep_type.replace("_", " ")),
            ("Points", str(entry.n_points)),
        ]
        if entry.bias_mv is not None:
            rows.append(("Bias", f"{entry.bias_mv:.1f} mV"))
        if entry.spec_freq_hz is not None:
            rows.append(("Freq", f"{entry.spec_freq_hz:.0f} Hz"))
        seen = {"sweep_type", "n_points", "bias_mv", "spec_freq_hz"}
        for k, v in hdr.items():
            if k not in seen and v.strip():
                rows.append((k, v.strip()))
        self._meta_rows = rows
        self._filter_meta()

    def clear(self):
        self.name_lbl.setText("No scan selected")
        for v in self._qi.values():
            v.setText("—")
        for lbl in self._ch_img_lbls:
            lbl.clear()
        self._meta_rows = []
        self.meta_table.setRowCount(0)

    def apply_theme(self, t: dict):
        self._t = t
        self._filter_meta()

    # ── Public ─────────────────────────────────────────────────────────────────
    def load_channels(self, entry: SxmFile, colormap_key: str,
                       processing: dict = None):
        self._ch_token = object()
        sigs = ChannelSignals()
        sigs.loaded.connect(self._on_ch_loaded)
        self._ch_sigs = sigs
        planes = []
        try:
            scan = load_scan(entry.path)
            plane_names = list(scan.plane_names)
            n_planes = scan.n_planes
            planes = list(getattr(scan, "planes", []) or [])
        except Exception:
            plane_names = list(PLANE_NAMES)
            n_planes = len(plane_names)
        self._set_channel_preview_slots(plane_names)
        for i in range(n_planes):
            arr = planes[i] if i < len(planes) else None
            loader = ChannelLoader(entry, i, colormap_key,
                                   self._ch_token, 124, 98, sigs,
                                   self._clip_low, self._clip_high,
                                   processing=processing,
                                   arr=arr)
            self._pool.start(loader)

    # Back-compat alias used internally
    _load_channels = load_channels

    @Slot(int, QPixmap, object)
    def _on_ch_loaded(self, idx: int, pixmap: QPixmap, token):
        if token is not self._ch_token:
            return
        if idx >= len(self._ch_img_lbls):
            return
        lbl = self._ch_img_lbls[idx]
        lbl.setPixmap(pixmap.scaled(lbl.width(), lbl.height(),
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _set_channel_preview_slots(self, names: list[str]) -> None:
        names = names or PLANE_NAMES
        if [lbl.text() for lbl in self._ch_name_lbls] == names:
            for lbl in self._ch_img_lbls:
                lbl.clear()
                lbl.setText("—")
            return
        for cell in self._ch_cells:
            self._ch_grid.removeWidget(cell)
            cell.deleteLater()
        self._ch_cells.clear()
        self._ch_img_lbls.clear()
        self._ch_name_lbls.clear()
        for i, name in enumerate(names):
            r, c = divmod(i, 2)
            cell = QWidget()
            cell_lay = QVBoxLayout(cell)
            cell_lay.setContentsMargins(0, 0, 0, 0)
            cell_lay.setSpacing(2)
            img_lbl = QLabel()
            img_lbl.setFixedSize(128, 102)
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setFrameShape(QFrame.StyledPanel)
            img_lbl.setText("—")
            nm_lbl = QLabel(name)
            nm_lbl.setFont(QFont("Helvetica", 9))
            nm_lbl.setAlignment(Qt.AlignCenter)
            nm_lbl.setWordWrap(True)
            cell_lay.addWidget(img_lbl)
            cell_lay.addWidget(nm_lbl)
            self._ch_grid.addWidget(cell, r, c)
            self._ch_cells.append(cell)
            self._ch_img_lbls.append(img_lbl)
            self._ch_name_lbls.append(nm_lbl)

    def _load_metadata(self, entry: SxmFile):
        try:
            hdr = load_scan(entry.path).header
        except Exception:
            hdr = {}
        priority = [
            "REC_DATE", "REC_TIME", "SCAN_PIXELS", "SCAN_RANGE",
            "SCAN_OFFSET", "SCAN_ANGLE", "SCAN_DIR", "BIAS",
            "REC_TEMP", "ACQ_TIME", "SCAN_TIME", "COMMENT",
        ]
        rows: list[tuple[str, str]] = []
        seen: set[str]              = set()
        for k in priority:
            v = hdr.get(k)
            if isinstance(v, str) and v.strip():
                rows.append((k, v.strip()))
                seen.add(k)
        for k, v in hdr.items():
            if k in seen:
                continue
            if isinstance(v, str) and v.strip():
                rows.append((k, v.strip()))
            elif v is not None and not isinstance(v, (bytes, bytearray)):
                s = str(v).strip()
                if s:
                    rows.append((k, s))
        self._meta_rows = rows
        self._filter_meta()

    def _filter_meta(self):
        query = self.search_box.text().lower()
        self.meta_table.setRowCount(0)
        t = self._t
        for param, value in self._meta_rows:
            if not query or query in param.lower() or query in value.lower():
                row    = self.meta_table.rowCount()
                self.meta_table.insertRow(row)
                p_item = QTableWidgetItem(param)
                p_item.setForeground(QColor(t["accent_bg"]))
                v_item = QTableWidgetItem(value)
                v_item.setForeground(QColor(t["fg"]))
                self.meta_table.setItem(row, 0, p_item)
                self.meta_table.setItem(row, 1, v_item)
        self.meta_table.resizeRowsToContents()


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
        meta_parts = []
        if entry.n_scans:
            meta_parts.append(f"{entry.n_scans} scan{'s' if entry.n_scans != 1 else ''}")
        if entry.n_specs:
            meta_parts.append(f"{entry.n_specs} spec{'s' if entry.n_specs != 1 else ''}")
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


# ── BreadcrumbBar ─────────────────────────────────────────────────────────────
class _BreadcrumbBar(QWidget):
    """Path strip with clickable segments + back/up buttons.

    Segments are clickable and emit ``segment_clicked(Path)``. Back/up buttons
    emit their own signals so the grid can decide whether they're enabled.
    """

    segment_clicked = Signal(object)  # Path
    back_requested  = Signal()
    up_requested    = Signal()

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self._t = t
        self._root: Optional[Path] = None
        self._current: Optional[Path] = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(4)

        self._back_btn = QPushButton("←")
        self._back_btn.setFixedSize(24, 24)
        self._back_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._back_btn.setEnabled(False)
        self._back_btn.clicked.connect(self.back_requested)
        lay.addWidget(self._back_btn)

        self._up_btn = QPushButton("↑")
        self._up_btn.setFixedSize(24, 24)
        self._up_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._up_btn.setEnabled(False)
        self._up_btn.clicked.connect(self.up_requested)
        lay.addWidget(self._up_btn)

        self._segments_host = QWidget()
        self._segments_lay = QHBoxLayout(self._segments_host)
        self._segments_lay.setContentsMargins(6, 0, 0, 0)
        self._segments_lay.setSpacing(4)
        self._segments_lay.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._segments_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self._segments_host, 1)

        self.setFixedHeight(36)
        self.apply_theme(t)

    def apply_theme(self, t: dict):
        self._t = t
        self.setStyleSheet(f"background-color: {t['main_bg']};")
        for btn in (self._back_btn, self._up_btn):
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {t['card_bg']}; "
                f"color: {t['fg']}; border: 1px solid {t['sep']}; "
                f"border-radius: 3px; }}"
                f"QPushButton:hover:enabled {{ border: 1px solid {t['accent_bg']}; }}"
                f"QPushButton:disabled {{ color: {t['sub_fg']}; }}"
            )
        self._restyle_segments()

    def _restyle_segments(self):
        t = self._t
        for i in range(self._segments_lay.count()):
            w = self._segments_lay.itemAt(i).widget()
            if isinstance(w, QPushButton):
                w.setStyleSheet(
                    f"QPushButton {{ background: transparent; color: {t['fg']}; "
                    f"border: none; padding: 2px 6px; }}"
                    f"QPushButton:hover {{ color: {t['accent_bg']}; "
                    f"text-decoration: underline; }}"
                )
            elif isinstance(w, QLabel):
                w.setStyleSheet(f"color: {t['sub_fg']}; background: transparent;")

    def set_state(self, root: Optional[Path], current: Optional[Path],
                  *, can_go_back: bool):
        self._root = root
        self._current = current
        self._back_btn.setEnabled(can_go_back)
        self._up_btn.setEnabled(
            current is not None and root is not None and current != root
        )
        self._rebuild_segments()

    def _clear_segments(self):
        while self._segments_lay.count():
            item = self._segments_lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _rebuild_segments(self):
        self._clear_segments()
        if self._root is None or self._current is None:
            placeholder = QLabel("No folder open")
            placeholder.setFont(QFont("Helvetica", 10))
            self._segments_lay.addWidget(placeholder)
            self._restyle_segments()
            return

        # Build relative segment chain: root, then each subdir down to current.
        try:
            rel = self._current.relative_to(self._root)
            tail = [] if str(rel) == "." else list(rel.parts)
        except ValueError:
            tail = []

        # Root segment uses the folder name, full chain uses parts.
        segments: list[tuple[str, Path]] = [(self._root.name or str(self._root), self._root)]
        cum = self._root
        for part in tail:
            cum = cum / part
            segments.append((part, cum))

        for i, (name, path) in enumerate(segments):
            if i:
                sep = QLabel("›")
                sep.setFont(QFont("Helvetica", 11))
                self._segments_lay.addWidget(sep)
            btn = QPushButton(name)
            btn.setFont(QFont("Helvetica", 10, QFont.Bold if i == len(segments) - 1 else QFont.Normal))
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.setFlat(True)
            btn.clicked.connect(lambda _=False, p=path: self.segment_clicked.emit(p))
            self._segments_lay.addWidget(btn)
        self._segments_lay.addStretch(1)
        self._restyle_segments()


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
            self._cards[key] = card

        # Populate the grid honouring the current filter.
        self._relayout_filtered()

        # Queue async thumbnail rendering for every card.
        token = self._load_token
        for entry in entries:
            if isinstance(entry, FolderEntry):
                if entry.sample_scan_paths:
                    loader = FolderThumbnailLoader(
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
                loader = SpecThumbnailLoader(entry, token,
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
        return ThumbnailLoader(entry, self._thumbnail_colormap, token,
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
                loader = FolderThumbnailLoader(
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
        self._breadcrumb.apply_theme(t)
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
