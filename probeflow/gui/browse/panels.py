"""Browse side panels for controls and selected-file metadata."""

from __future__ import annotations

import json

from probeflow.core.browse_filters import FolderFilterState
from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from probeflow.gui.models import PLANE_NAMES, SxmFile, VertFile
from probeflow.gui.rendering import (
    CMAP_KEY,
    CMAP_NAMES,
    DEFAULT_CMAP_KEY,
    DEFAULT_CMAP_LABEL,
    THUMBNAIL_CHANNEL_DEFAULT,
    THUMBNAIL_CHANNEL_OPTIONS,
)
from probeflow.gui.workers import ChannelPreviewLoader

from .helpers import _browse_attr, _sep


def _setp_display(entry: SxmFile) -> str:
    """Header-strip setpoint value: tunnel current for STM, Δf for AFM."""
    if entry.current_pa is not None:
        return f"{entry.current_pa:.1f} pA"
    if entry.feedback_setpoint is not None:
        unit = f" {entry.feedback_setpoint_unit}" if entry.feedback_setpoint_unit else ""
        return f"{entry.feedback_setpoint:.4g}{unit}"
    return "—"


class BrowseToolPanel(QWidget):
    """Left-side control panel for browsing and live thumbnail appearance."""
    open_folder_requested      = Signal()
    colormap_changed           = Signal(str)
    thumbnail_align_changed    = Signal(str)
    map_spectra_requested      = Signal()
    overlay_spectra_requested  = Signal()
    filter_changed             = Signal(str)   # "all" | "images" | "spectra"
    folder_filter_changed      = Signal(object)
    export_filtered_requested  = Signal()
    thumbnail_channel_changed  = Signal(str)
    thumbnail_size_changed     = Signal(str)   # "large" | "small"
    open_fc_window_requested   = Signal()      # floating Feature Counting window

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
        open_btn.setFont(ui_font(9))
        open_btn.setFixedHeight(30)
        open_btn.setCursor(QCursor(Qt.PointingHandCursor))
        open_btn.setObjectName("accentBtn")
        open_btn.clicked.connect(self.open_folder_requested.emit)
        lay.addWidget(open_btn)

        self._filter_toggle_btn = QPushButton("[+] Filter folder")
        self._filter_toggle_btn.setFont(ui_font(9, weight=QFont.Bold))
        self._filter_toggle_btn.setFixedHeight(26)
        self._filter_toggle_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._filter_toggle_btn.clicked.connect(self._toggle_folder_filters)
        lay.addWidget(self._filter_toggle_btn)

        self._folder_filter_box = QWidget()
        filter_box_lay = QVBoxLayout(self._folder_filter_box)
        filter_box_lay.setContentsMargins(0, 0, 0, 0)
        filter_box_lay.setSpacing(6)

        self._size_filter_btn = self._make_filter_toggle("Size (nm)")
        filter_box_lay.addWidget(self._size_filter_btn)
        self._size_width_nm = self._make_filter_spinbox(0.0, 1_000_000.0, 1.0, " nm")
        self._size_height_nm = self._make_filter_spinbox(0.0, 1_000_000.0, 1.0, " nm")
        filter_box_lay.addWidget(self._labeled_spin_row("Max width", self._size_width_nm))
        filter_box_lay.addWidget(self._labeled_spin_row("Max height", self._size_height_nm))

        self._completion_filter_btn = self._make_filter_toggle("Completion (%)")
        filter_box_lay.addWidget(self._completion_filter_btn)
        self._completion_min_pct = self._make_filter_spinbox(0.0, 100.0, 1.0, " %")
        self._completion_min_pct.setValue(50.0)
        filter_box_lay.addWidget(self._labeled_spin_row("Min completion", self._completion_min_pct))

        self._bias_filter_btn = self._make_filter_toggle("Bias (mV)")
        filter_box_lay.addWidget(self._bias_filter_btn)
        self._bias_min_mv = self._make_filter_spinbox(-10000.0, 10000.0, 1.0, " mV")
        self._bias_max_mv = self._make_filter_spinbox(-10000.0, 10000.0, 1.0, " mV")
        self._bias_min_mv.setValue(-500.0)
        self._bias_max_mv.setValue(500.0)
        filter_box_lay.addWidget(self._labeled_spin_row("Min bias", self._bias_min_mv))
        filter_box_lay.addWidget(self._labeled_spin_row("Max bias", self._bias_max_mv))

        self._export_filtered_btn = QPushButton("Export filtered folder")
        self._export_filtered_btn.setFont(ui_font(9))
        self._export_filtered_btn.setFixedHeight(28)
        self._export_filtered_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._export_filtered_btn.clicked.connect(self.export_filtered_requested.emit)
        filter_box_lay.addWidget(self._export_filtered_btn)

        self._folder_filter_box.setVisible(False)
        lay.addWidget(self._folder_filter_box)

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
            btn.setFont(ui_font(9))
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
        appearance_lbl.setFont(ui_font(11, weight=QFont.Bold))
        lay.addWidget(appearance_lbl)

        cm_lbl = QLabel("Colormap")
        cm_lbl.setFont(ui_font(9, weight=QFont.Bold))
        lay.addWidget(cm_lbl)

        self.cmap_cb = QComboBox()
        self.cmap_cb.addItems(CMAP_NAMES)
        self.cmap_cb.setCurrentText(cfg.get("colormap", DEFAULT_CMAP_LABEL))
        self.cmap_cb.setFont(ui_font(10))
        self.cmap_cb.currentTextChanged.connect(self._on_colormap_changed)
        lay.addWidget(self.cmap_cb)

        thumb_lbl = QLabel("Thumbnail channel")
        thumb_lbl.setFont(ui_font(9, weight=QFont.Bold))
        lay.addWidget(thumb_lbl)

        self.thumbnail_channel_cb = QComboBox()
        self.thumbnail_channel_cb.addItems(THUMBNAIL_CHANNEL_OPTIONS)
        self.thumbnail_channel_cb.setCurrentText(THUMBNAIL_CHANNEL_DEFAULT)
        self.thumbnail_channel_cb.setFont(ui_font(10))
        self.thumbnail_channel_cb.setToolTip(
            "Choose which forward scan channel is used for browse thumbnails. "
            "Files without that channel fall back to Z."
        )
        self.thumbnail_channel_cb.currentTextChanged.connect(
            self.thumbnail_channel_changed.emit)
        lay.addWidget(self.thumbnail_channel_cb)

        align_lbl = QLabel("Align rows")
        align_lbl.setFont(ui_font(9, weight=QFont.Bold))
        lay.addWidget(align_lbl)
        self.align_rows_cb = QComboBox()
        self.align_rows_cb.addItems(["None", "Median", "Mean"])
        self.align_rows_cb.setCurrentText("None")
        self.align_rows_cb.setFont(ui_font(10))
        self.align_rows_cb.setToolTip(
            "Preview-only thumbnail row alignment. Full-size viewer data opens raw."
        )
        self.align_rows_cb.currentTextChanged.connect(self._on_align_changed)
        lay.addWidget(self.align_rows_cb)

        size_lbl = QLabel("Thumbnail size")
        size_lbl.setFont(ui_font(9, weight=QFont.Bold))
        lay.addWidget(size_lbl)
        self.size_cb = QComboBox()
        self.size_cb.addItems(["Large", "Small"])
        self.size_cb.setCurrentText(cfg.get("thumbnail_size", "large").capitalize())
        self.size_cb.setFont(ui_font(10))
        self.size_cb.currentTextChanged.connect(
            lambda t: self.thumbnail_size_changed.emit(t.lower()))
        lay.addWidget(self.size_cb)
        lay.addWidget(_sep())

        self._map_spectra_btn = QPushButton("Map spectra to images\u2026")
        self._map_spectra_btn.setFont(ui_font(9))
        self._map_spectra_btn.setFixedHeight(28)
        self._map_spectra_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._map_spectra_btn.setToolTip(
            "Pick the parent image for each .VERT spectrum in the current "
            "folder. Spectra without a mapping show no marker on any image. "
            "You can also map per-image inside the viewer.")
        self._map_spectra_btn.clicked.connect(self.map_spectra_requested.emit)
        lay.addWidget(self._map_spectra_btn)

        self._overlay_spectra_btn = QPushButton("Overlay selected spectra…")
        self._overlay_spectra_btn.setFont(ui_font(9))
        self._overlay_spectra_btn.setFixedHeight(28)
        self._overlay_spectra_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._overlay_spectra_btn.setToolTip(
            "Ctrl-click two or more spectroscopy cards, then overlay or "
            "waterfall one common signal channel.")
        self._overlay_spectra_btn.clicked.connect(self.overlay_spectra_requested.emit)
        lay.addWidget(self._overlay_spectra_btn)

        lay.addWidget(_sep())

        fc_btn = QPushButton("🔬  Feature Counting")
        fc_btn.setFont(ui_font(10, weight=QFont.Bold))
        fc_btn.setFixedHeight(34)
        fc_btn.setCursor(QCursor(Qt.PointingHandCursor))
        fc_btn.setObjectName("accentBtn")
        fc_btn.setToolTip(
            "Open the Feature Counting window.\n"
            "Stays open alongside Browse so you can keep browsing\n"
            "thumbnails while counting molecules.\n"
            "Shortcut: Ctrl+3")
        fc_btn.clicked.connect(self.open_fc_window_requested.emit)
        lay.addWidget(fc_btn)

        lay.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self._sync_folder_filter_inputs()
        self.apply_theme(self._t)

    # ── Slots ──────────────────────────────────────────────────────────────────
    def _on_colormap_changed(self):
        cmap_key = CMAP_KEY.get(self.cmap_cb.currentText(), DEFAULT_CMAP_KEY)
        self.colormap_changed.emit(cmap_key)

    def _on_align_changed(self, text: str):
        self.thumbnail_align_changed.emit(text)

    def _on_filter_click(self, mode: str):
        self._filter_mode = mode
        self.filter_changed.emit(mode)

    def _make_filter_toggle(self, text: str) -> QCheckBox:
        btn = QCheckBox(text)
        btn.setFont(ui_font(9, weight=QFont.Bold))
        btn.setCursor(QCursor(Qt.PointingHandCursor))
        btn.setObjectName("folderFilterToggle")
        btn.toggled.connect(self._sync_folder_filter_inputs)
        btn.toggled.connect(self._emit_folder_filter_state)
        return btn

    def _make_filter_spinbox(
        self,
        minimum: float,
        maximum: float,
        step: float,
        suffix: str,
    ) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(1)
        box.setRange(minimum, maximum)
        box.setSingleStep(step)
        box.setSuffix(suffix)
        box.setFont(ui_font(9))
        box.valueChanged.connect(self._emit_folder_filter_state)
        return box

    def _labeled_spin_row(self, label: str, box: QDoubleSpinBox) -> QWidget:
        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(6)
        lbl = QLabel(label)
        lbl.setFont(ui_font(8))
        row_lay.addWidget(lbl)
        row_lay.addWidget(box, 1)
        return row

    def _toggle_folder_filters(self) -> None:
        visible = not self._folder_filter_box.isVisible()
        self._folder_filter_box.setVisible(visible)
        self._filter_toggle_btn.setText(
            "[-] Filter folder" if visible else "[+] Filter folder"
        )

    def _sync_folder_filter_inputs(self) -> None:
        # Filters are gated by their checkbox state, but their numeric values
        # stay editable at all times so users can prepare thresholds before
        # turning a filter on.
        return

    def _emit_folder_filter_state(self, *_args) -> None:
        if not all(
            hasattr(self, name)
            for name in (
                "_size_filter_btn",
                "_completion_filter_btn",
                "_bias_filter_btn",
                "_size_width_nm",
                "_size_height_nm",
                "_completion_min_pct",
                "_bias_min_mv",
                "_bias_max_mv",
            )
        ):
            return
        self.folder_filter_changed.emit(self.get_folder_filter_state())

    # ── Public API ─────────────────────────────────────────────────────────────
    def get_filter_mode(self) -> str:
        return self._filter_mode

    def get_folder_filter_state(self) -> FolderFilterState:
        return FolderFilterState(
            size_enabled=self._size_filter_btn.isChecked(),
            max_width_nm=float(self._size_width_nm.value()),
            max_height_nm=float(self._size_height_nm.value()),
            completion_enabled=self._completion_filter_btn.isChecked(),
            min_completion_pct=float(self._completion_min_pct.value()),
            bias_enabled=self._bias_filter_btn.isChecked(),
            min_bias_mv=float(self._bias_min_mv.value()),
            max_bias_mv=float(self._bias_max_mv.value()),
        )

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
        toggle_style = (
            f"QCheckBox#folderFilterToggle {{"
            f"color: {t['sub_fg']};"
            f"spacing: 8px;"
            f"padding: 2px 0px;"
            f"}}"
            f"QCheckBox#folderFilterToggle:hover {{ color: {t['fg']}; }}"
            f"QCheckBox#folderFilterToggle:checked {{ color: {t['accent_bg']}; }}"
            f"QCheckBox#folderFilterToggle::indicator {{"
            f"width: 14px;"
            f"height: 14px;"
            f"border: 1px solid {t['sep']};"
            f"border-radius: 3px;"
            f"background-color: {t.get('bg', '#1e2128')};"
            f"}}"
            f"QCheckBox#folderFilterToggle::indicator:hover {{"
            f"border-color: {t['accent_bg']};"
            f"}}"
            f"QCheckBox#folderFilterToggle::indicator:checked {{"
            f"background-color: {t['accent_bg']};"
            f"border-color: {t['accent_bg']};"
            f"}}"
        )
        for btn in (
            getattr(self, "_size_filter_btn", None),
            getattr(self, "_completion_filter_btn", None),
            getattr(self, "_bias_filter_btn", None),
        ):
            if btn is not None:
                btn.setStyleSheet(toggle_style)


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
        self.name_lbl.setFont(ui_font(10, weight=QFont.Bold))
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
            t_lbl.setFont(ui_font(8))
            v_lbl = QLabel("—")
            v_lbl.setFont(ui_font(10, weight=QFont.Bold))
            qi_grid.addWidget(t_lbl, r, c * 2)
            qi_grid.addWidget(v_lbl, r, c * 2 + 1)
            self._qi[key] = v_lbl
        summary_lay.addWidget(qi_widget)
        summary_lay.addWidget(_sep())

        ch_hdr = QLabel("Channels")
        ch_hdr.setFont(ui_font(11, weight=QFont.Bold))
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
        self._meta_toggle.setFont(ui_font(9, weight=QFont.Bold))
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
        self.search_box.setFont(ui_font(10))
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
        self.meta_table.setFont(ui_font(10))
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
        self._qi["setp"].setText(_setp_display(entry))
        # One worker load serves both the channel previews and the metadata
        # table (meta_ready → _populate_metadata_rows).
        self.load_channels(entry, colormap_key, processing=None)

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
        # Invalidate any in-flight preview load so late results don't
        # repopulate a panel the user just cleared.
        self._ch_token = object()
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
        """Kick off the off-thread preview + header load for *entry*.

        The scan is read exactly once, on a pool worker (previously this did a
        synchronous full ``load_scan`` here for the previews and a second one
        in the metadata loader — two full file transfers on the GUI thread per
        card click, the main network-drive freeze).  ``meta_ready`` populates
        the slot layout and metadata table; ``loaded`` fills in each preview.
        Stale deliveries are filtered by the token check in each slot.
        """
        self._ch_token = object()
        self._ch_entry = entry
        for lbl in self._ch_img_lbls:
            lbl.clear()
            lbl.setText("…")
        Loader = _browse_attr("ChannelPreviewLoader", ChannelPreviewLoader)
        loader = Loader(entry, colormap_key, self._ch_token, 124, 98,
                        self._clip_low, self._clip_high,
                        processing=processing)
        loader.signals.meta_ready.connect(self._on_ch_meta)
        loader.signals.loaded.connect(self._on_ch_loaded)
        loader.signals.failed.connect(self._on_ch_failed)
        self._pool.start(loader)

    # Back-compat alias used internally
    _load_channels = load_channels

    @Slot(list, dict, object)
    def _on_ch_meta(self, names: list, header: dict, token) -> None:
        if token is not self._ch_token:
            return
        self._set_channel_preview_slots(list(names))
        entry = getattr(self, "_ch_entry", None)
        if entry is not None:
            self._populate_metadata_rows(entry, dict(header))

    @Slot(str, object)
    def _on_ch_failed(self, message: str, token) -> None:
        if token is not self._ch_token:
            return
        for lbl in self._ch_img_lbls:
            lbl.clear()
            lbl.setText("load failed")
        self._meta_rows = []
        self._filter_meta()

    @Slot(int, QImage, object)
    def _on_ch_loaded(self, idx: int, image: QImage, token):
        # Workers emit QImage (QPixmap is GUI-thread-only); convert here.
        if token is not self._ch_token:
            return
        if idx >= len(self._ch_img_lbls):
            return
        lbl = self._ch_img_lbls[idx]
        if image is None or image.isNull():
            lbl.setText("render failed")
            return
        pixmap = QPixmap.fromImage(image)
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
            nm_lbl.setFont(ui_font(9))
            nm_lbl.setAlignment(Qt.AlignCenter)
            nm_lbl.setWordWrap(True)
            cell_lay.addWidget(img_lbl)
            cell_lay.addWidget(nm_lbl)
            self._ch_grid.addWidget(cell, r, c)
            self._ch_cells.append(cell)
            self._ch_img_lbls.append(img_lbl)
            self._ch_name_lbls.append(nm_lbl)

    def _populate_metadata_rows(self, entry: SxmFile, hdr: dict) -> None:
        """Fill the metadata table from an already-loaded header dict.

        The header arrives via ``ChannelPreviewLoader.meta_ready`` so the file
        is read once, off the GUI thread (this used to be a second synchronous
        full ``load_scan`` per card click).
        """
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

        scanflow = dict(entry.scanflow_acquisition or {})
        if scanflow:
            rows.append(("SCANFLOW_SCHEMA", str(scanflow.get("schema") or "")))
            rows.append(("SCANFLOW_RECORD_TYPE", str(scanflow.get("record_type") or "")))
            session = scanflow.get("session") or {}
            if isinstance(session, dict):
                for key in ("session_id", "routine"):
                    value = session.get(key)
                    if value not in (None, ""):
                        rows.append((f"SCANFLOW_SESSION_{key.upper()}", str(value)))
            for key in ("step", "motion", "drift", "quality", "safety", "scan_parameters", "position"):
                value = scanflow.get(key)
                if value not in (None, {}, []):
                    if isinstance(value, dict):
                        rows.append((f"SCANFLOW_{key.upper()}", json.dumps(value, sort_keys=True, default=str)))
                    else:
                        rows.append((f"SCANFLOW_{key.upper()}", str(value)))
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
