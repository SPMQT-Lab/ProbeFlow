from __future__ import annotations

import math
import weakref

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

from probeflow.analysis.lattice_correction_workflow import (
    lattice_correction_matrix_px,
    lattice_correction_operation_params,
)
from probeflow.analysis.lattice_distortion import (
    IdealLattice,
    LatticeCorrection,
    MeasuredLattice,
    compute_correction,
)
from probeflow.analysis.lattice_grid import direct_lattice_vectors_from_reciprocal_grid
from probeflow.gui.no_wheel import install_no_wheel_spinboxes


class FFTViewerDialog(QDialog):
    """Side-by-side real-space / FFT inspection window."""

    def __init__(
        self,
        arr: np.ndarray,
        scan_range_m: tuple,
        colormap: str = "gray",
        theme: dict | None = None,
        channel_unit: tuple | None = None,
        get_image_fn=None,
        apply_correction_fn=None,
        preview_image_fn=None,
        clear_preview_fn=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.setWindowTitle("FFT Viewer")
        self.resize(1180, 820)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._arr = arr.astype(np.float64, copy=True)
        self._scan_range_m = scan_range_m
        self._colormap = colormap
        self._theme = theme or {}
        self._ch_unit = channel_unit  # (scale, unit, axis_label) or None
        self._get_image_fn = get_image_fn
        self._apply_correction_fn = apply_correction_fn
        self._preview_image_fn = preview_image_fn
        self._clear_preview_fn = clear_preview_fn
        self._fft_correction: LatticeCorrection | None = None
        self._fft_preview_active = False
        self._updating_fft_ideal = False

        self._fft_mag: np.ndarray | None = None
        self._qx: np.ndarray | None = None
        self._qy: np.ndarray | None = None
        self._fft_xlim: tuple = (0.0, 1.0)
        self._fft_ylim: tuple = (1.0, 0.0)
        self._scale_mode = "log"
        self._window_mode = "hann"
        self._dc_mode = "keep"
        self._pan_anchor: tuple | None = None
        self._fft_lattice_drag_active: bool = False
        self._disp_range: tuple = (0.0, 1.0)
        self._vmin_frac: float = 0.0
        self._vmax_frac: float = 1.0
        self._hist_drag: str | None = None
        self._vmin_line = None
        self._vmax_line = None
        self._fft_im = None
        self._fft_lattice_dock = None
        self._fft_lattice_panel = None
        self._focus_fft_active = False
        self._fft_cmap = "gray"
        self._cmap_options = ["gray", "gray_r", "inferno", "hot", "viridis", "plasma", "turbo"]
        self._bragg_artists: list = []
        self._calib_picks: list = []   # (qx_nm, qy_nm) in nm⁻¹

        self._build()
        self._recompute_fft()
        self._update_info_panel()
        self._redraw()
        self._refresh_fft_correction_ui()

    # ── layout helpers (shared with subclasses) ────────────────────────────────

    def _build_toolbar_row(self) -> QHBoxLayout:
        """Create toolbar with Scale/LUT/Window/DC controls and zoom/export buttons."""
        tb = QHBoxLayout()
        tb.setSpacing(6)

        def _lbl(text):
            lb = QLabel(text)
            lb.setFont(QFont("Helvetica", 9))
            return lb

        def _combo(items, min_width=80):
            c = QComboBox()
            c.addItems(items)
            c.setFont(QFont("Helvetica", 9))
            c.setFixedHeight(24)
            c.setMinimumWidth(min_width)
            return c

        tb.addWidget(_lbl("Scale:"))
        self._scale_combo = _combo(["Log", "Linear"], 80)
        self._scale_combo.currentIndexChanged.connect(self._on_scale_changed)
        tb.addWidget(self._scale_combo)

        tb.addWidget(_lbl("LUT:"))
        self._cmap_combo = _combo(
            ["Gray", "Gray (inv.)", "Inferno", "Hot", "Viridis", "Plasma", "Turbo"], 96
        )
        self._cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        tb.addWidget(self._cmap_combo)

        tb.addWidget(_lbl("Window:"))
        self._window_combo = _combo(["Hann", "None", "Tukey"], 82)
        self._window_combo.currentIndexChanged.connect(self._on_window_changed)
        tb.addWidget(self._window_combo)

        tb.addWidget(_lbl("DC:"))
        self._dc_combo = _combo(["Zero DC", "Keep DC", "Mask DC"], 95)
        self._dc_combo.setCurrentIndex(1)
        self._dc_combo.currentIndexChanged.connect(self._on_dc_changed)
        tb.addWidget(self._dc_combo)

        tb.addStretch(1)

        focus_btn = QPushButton("Focus FFT")
        focus_btn.setFont(QFont("Helvetica", 9))
        focus_btn.setFixedHeight(24)
        focus_btn.setMinimumWidth(86)
        focus_btn.setCheckable(True)
        focus_btn.setToolTip("Hide the real-space reference and lower tools for a larger FFT view")
        focus_btn.toggled.connect(self._on_focus_fft_toggled)
        tb.addWidget(focus_btn)
        self._focus_fft_btn = focus_btn

        for label, tip, slot in [
            ("Fit",   "Zoom to fit full FFT extent",    self._zoom_fit),
            ("Ctr",   "Zoom to centre (quarter range)", self._zoom_centre),
            ("  +  ", "Zoom in",                        lambda: self._zoom_by(0.5)),
            ("  −  ", "Zoom out",                       lambda: self._zoom_by(2.0)),
        ]:
            btn = QPushButton(label)
            btn.setFont(QFont("Helvetica", 9))
            btn.setFixedHeight(24)
            btn.setMinimumWidth(44)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            tb.addWidget(btn)

        tb.addSpacing(8)
        exp_btn = QPushButton("Export PNG…")
        exp_btn.setFont(QFont("Helvetica", 9))
        exp_btn.setFixedHeight(24)
        exp_btn.setMinimumWidth(96)
        exp_btn.clicked.connect(self._on_export)
        tb.addWidget(exp_btn)

        tb.addSpacing(8)
        grid_btn = QPushButton("Grid")
        grid_btn.setFont(QFont("Helvetica", 9))
        grid_btn.setFixedHeight(24)
        grid_btn.setMinimumWidth(68)
        grid_btn.setToolTip("Show reciprocal-space lattice grid controls")
        grid_btn.clicked.connect(lambda _checked=False: self._on_open_fft_lattice(select_advanced=True))
        tb.addWidget(grid_btn)
        self._grid_lattice_btn = grid_btn

        clear_btn = QPushButton("Clear Grid")
        clear_btn.setFont(QFont("Helvetica", 9))
        clear_btn.setFixedHeight(24)
        clear_btn.setMinimumWidth(80)
        clear_btn.setToolTip("Remove the reciprocal-space lattice overlay")
        clear_btn.setEnabled(False)
        clear_btn.clicked.connect(self._on_clear_fft_lattice)
        tb.addWidget(clear_btn)
        self._clear_grid_btn = clear_btn

        return tb

    def _build_fft_column(self) -> QVBoxLayout:
        """Create the FFT workspace and guided correction tools."""
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")

        right_col = QVBoxLayout()
        right_col.setSpacing(6)

        self._fig_fft = Figure(figsize=(6.0, 5.4), dpi=90)
        self._fig_fft.patch.set_facecolor(bg)
        self._canvas_fft = FigureCanvasQTAgg(self._fig_fft)
        self._canvas_fft.setMinimumSize(520, 360)
        self._canvas_fft.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._ax_fft = self._fig_fft.add_subplot(111)
        self._ax_fft.set_facecolor(bg)
        for sp in self._ax_fft.spines.values():
            sp.set_color(fg)
        self._ax_fft.tick_params(colors=fg, labelsize=9)
        self._fig_fft.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.14)

        fft_top = QWidget()
        fft_top_lay = QHBoxLayout(fft_top)
        fft_top_lay.setContentsMargins(0, 0, 0, 0)
        fft_top_lay.setSpacing(8)
        fft_top_lay.addWidget(self._canvas_fft, 1)

        side_panel = QFrame()
        side_panel.setMinimumWidth(190)
        side_panel.setMaximumWidth(240)
        side_lay = QVBoxLayout(side_panel)
        side_lay.setContentsMargins(8, 8, 8, 8)
        side_lay.setSpacing(8)
        cursor_title = QLabel("Cursor")
        cursor_title.setFont(QFont("Helvetica", 9, QFont.Bold))
        self._cursor_readout_lbl = QLabel("Move over the FFT")
        self._cursor_readout_lbl.setFont(QFont("Courier", 8))
        self._cursor_readout_lbl.setWordWrap(True)
        corr_title = QLabel("Correction")
        corr_title.setFont(QFont("Helvetica", 9, QFont.Bold))
        self._fft_correction_status_lbl = QLabel("No reciprocal grid yet")
        self._fft_correction_status_lbl.setFont(QFont("Helvetica", 8))
        self._fft_correction_status_lbl.setWordWrap(True)
        side_lay.addWidget(cursor_title)
        side_lay.addWidget(self._cursor_readout_lbl)
        side_lay.addSpacing(8)
        side_lay.addWidget(corr_title)
        side_lay.addWidget(self._fft_correction_status_lbl)
        side_lay.addStretch(1)
        fft_top_lay.addWidget(side_panel)

        self._tab_widget = QTabWidget()
        self._tab_widget.setMinimumHeight(220)
        self._tab_widget.setFont(QFont("Helvetica", 9))

        # ── Guided Correction tab ────────────────────────────────────────────
        corr_scroll = QScrollArea()
        corr_scroll.setWidgetResizable(True)
        corr_scroll.setFrameShape(QFrame.NoFrame)
        corr_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        corr_inner = QWidget()
        corr_lay = QVBoxLayout(corr_inner)
        corr_lay.setSpacing(8)
        corr_lay.setContentsMargins(8, 8, 8, 6)

        ref_grp = QGroupBox("1. Reference lattice")
        ref_grp.setFont(QFont("Helvetica", 9))
        ref_grid = QGridLayout(ref_grp)
        ref_grid.setContentsMargins(8, 8, 8, 6)
        ref_grid.setHorizontalSpacing(8)
        ref_grid.setVerticalSpacing(4)

        self._bragg_enable_cb = QCheckBox("Show predicted shells")
        self._bragg_enable_cb.setFont(QFont("Helvetica", 9))
        self._bragg_enable_cb.setChecked(True)
        self._bragg_enable_cb.setToolTip("Overlay expected low-index Bragg shell radii on the FFT.")
        self._bragg_enable_cb.toggled.connect(self._on_bragg_changed)
        ref_grid.addWidget(self._bragg_enable_cb, 0, 0, 1, 2)

        self._bragg_sym_combo = QComboBox()
        self._bragg_sym_combo.addItems(["Square", "Hexagonal"])
        self._bragg_sym_combo.setFont(QFont("Helvetica", 9))
        self._bragg_sym_combo.setFixedHeight(24)
        self._bragg_sym_combo.setToolTip("Crystal symmetry used for predicted FFT shell spacing.")
        self._bragg_sym_combo.currentIndexChanged.connect(self._on_bragg_symmetry_changed)
        self._bragg_a_spin = QDoubleSpinBox()
        self._bragg_a_spin.setRange(0.001, 999.0)
        self._bragg_a_spin.setValue(2.46)
        self._bragg_a_spin.setDecimals(3)
        self._bragg_a_spin.setFont(QFont("Helvetica", 9))
        self._bragg_a_spin.setFixedHeight(24)
        self._bragg_a_spin.setToolTip("Approximate real-space lattice spacing for predicted Bragg shells.")
        self._bragg_a_spin.valueChanged.connect(self._on_bragg_changed)
        self._bragg_unit_combo = QComboBox()
        self._bragg_unit_combo.addItems(["Å", "nm"])
        self._bragg_unit_combo.setFont(QFont("Helvetica", 9))
        self._bragg_unit_combo.setFixedHeight(24)
        self._bragg_unit_combo.setToolTip("Unit for the reference lattice spacing.")
        self._bragg_unit_combo.currentIndexChanged.connect(self._on_bragg_changed)
        self._bragg_max_shells_spin = QSpinBox()
        self._bragg_max_shells_spin.setRange(1, 12)
        self._bragg_max_shells_spin.setValue(5)
        self._bragg_max_shells_spin.setFont(QFont("Helvetica", 9))
        self._bragg_max_shells_spin.setFixedHeight(24)
        self._bragg_max_shells_spin.setToolTip("Maximum number of visible predicted shell families.")
        self._bragg_max_shells_spin.valueChanged.connect(self._on_bragg_changed)

        a_value_row = QHBoxLayout()
        a_value_row.setSpacing(4)
        a_value_row.addWidget(self._bragg_a_spin, 1)
        a_value_row.addWidget(self._bragg_unit_combo)
        ref_grid.addWidget(QLabel("Symmetry:"), 1, 0)
        ref_grid.addWidget(self._bragg_sym_combo, 1, 1)
        ref_grid.addWidget(QLabel("Lattice a:"), 1, 2)
        ref_grid.addLayout(a_value_row, 1, 3)
        ref_grid.addWidget(QLabel("Max shells:"), 2, 0)
        ref_grid.addWidget(self._bragg_max_shells_spin, 2, 1)
        ref_grid.setColumnStretch(1, 1)
        ref_grid.setColumnStretch(3, 1)
        self._bragg_radius_lbl = QLabel("Radius: —")
        self._bragg_radius_lbl.setFont(QFont("Courier", 9))
        self._bragg_radius_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ref_grid.addWidget(self._bragg_radius_lbl, 3, 0, 1, 4)
        corr_lay.addWidget(ref_grp)

        meas_grp = QGroupBox("2. Measure reciprocal lattice")
        meas_grp.setFont(QFont("Helvetica", 9))
        meas_lay = QVBoxLayout(meas_grp)
        meas_lay.setContentsMargins(8, 8, 8, 6)
        meas_lay.setSpacing(5)
        meas_btn_row = QHBoxLayout()
        self._edit_grid_btn = QPushButton("Edit reciprocal grid")
        self._edit_grid_btn.setFont(QFont("Helvetica", 9))
        self._edit_grid_btn.setFixedHeight(24)
        self._edit_grid_btn.setToolTip("Create/select the FFT grid overlay and adjust g1/g2 handles on the FFT.")
        self._edit_grid_btn.clicked.connect(self._on_edit_reciprocal_grid)
        meas_btn_row.addWidget(self._edit_grid_btn)
        self._bragg_detect_btn = QPushButton("Detect peaks")
        self._bragg_detect_btn.setFont(QFont("Helvetica", 9))
        self._bragg_detect_btn.setFixedHeight(24)
        self._bragg_detect_btn.setToolTip("Auto-detect compact first-shell Bragg peaks as visual picks.")
        self._bragg_detect_btn.clicked.connect(self._detect_bragg_peaks)
        meas_btn_row.addWidget(self._bragg_detect_btn)
        self._bragg_clear_btn = QPushButton("Clear picks")
        self._bragg_clear_btn.setFont(QFont("Helvetica", 9))
        self._bragg_clear_btn.setFixedHeight(24)
        self._bragg_clear_btn.clicked.connect(self._clear_bragg_picks)
        meas_btn_row.addWidget(self._bragg_clear_btn)
        meas_btn_row.addStretch(1)
        meas_lay.addLayout(meas_btn_row)

        pick_row = QHBoxLayout()
        self._bragg_pick_cb = QCheckBox("Pick peaks")
        self._bragg_pick_cb.setFont(QFont("Helvetica", 9))
        self._bragg_pick_cb.setToolTip("When enabled, clicks on the FFT add/remove Bragg calibration picks.")
        self._bragg_pick_cb.toggled.connect(self._update_calib_ui)
        pick_row.addWidget(self._bragg_pick_cb)
        self._bragg_snap_cb = QCheckBox("Snap picks to compact peak")
        self._bragg_snap_cb.setFont(QFont("Helvetica", 9))
        self._bragg_snap_cb.setChecked(True)
        self._bragg_snap_cb.setToolTip("Move manual picks to nearby compact local maxima rather than streak pixels.")
        pick_row.addWidget(self._bragg_snap_cb)
        pick_row.addStretch(1)
        meas_lay.addLayout(pick_row)
        self._bragg_picks_lbl = QLabel("Picks: 0  (enable Pick peaks to edit)")
        self._bragg_picks_lbl.setFont(QFont("Helvetica", 9))
        meas_lay.addWidget(self._bragg_picks_lbl)
        self._fft_measured_lbl = QLabel("Create or edit a reciprocal grid to measure the direct lattice.")
        self._fft_measured_lbl.setFont(QFont("Courier", 8))
        self._fft_measured_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        meas_lay.addWidget(self._fft_measured_lbl)
        corr_lay.addWidget(meas_grp)

        apply_grp = QGroupBox("3. Preview/apply correction")
        apply_grp.setFont(QFont("Helvetica", 9))
        apply_lay = QVBoxLayout(apply_grp)
        apply_lay.setContentsMargins(8, 8, 8, 6)
        apply_lay.setSpacing(5)
        ideal_grid = QGridLayout()
        ideal_grid.setHorizontalSpacing(8)
        ideal_grid.setVerticalSpacing(4)
        self._fft_ideal_combo = QComboBox()
        self._fft_ideal_combo.addItems(["Match measured", "Square", "Rectangular", "Hexagonal", "Custom"])
        self._fft_ideal_combo.setCurrentText("Square")
        self._fft_ideal_combo.setFont(QFont("Helvetica", 9))
        self._fft_ideal_combo.setFixedHeight(24)
        self._fft_ideal_combo.setToolTip("Ideal direct lattice to map the measured FFT-derived lattice onto.")
        self._fft_ideal_combo.currentIndexChanged.connect(self._on_fft_ideal_changed)
        self._fft_ideal_a_spin = QDoubleSpinBox()
        self._fft_ideal_a_spin.setRange(0.001, 999.0)
        self._fft_ideal_a_spin.setDecimals(4)
        self._fft_ideal_a_spin.setSuffix(" nm")
        self._fft_ideal_a_spin.setFixedHeight(24)
        self._fft_ideal_a_spin.valueChanged.connect(self._on_fft_ideal_changed)
        self._fft_ideal_b_spin = QDoubleSpinBox()
        self._fft_ideal_b_spin.setRange(0.001, 999.0)
        self._fft_ideal_b_spin.setDecimals(4)
        self._fft_ideal_b_spin.setSuffix(" nm")
        self._fft_ideal_b_spin.setFixedHeight(24)
        self._fft_ideal_b_spin.valueChanged.connect(self._on_fft_ideal_changed)
        self._fft_ideal_angle_spin = QDoubleSpinBox()
        self._fft_ideal_angle_spin.setRange(1.0, 179.0)
        self._fft_ideal_angle_spin.setDecimals(2)
        self._fft_ideal_angle_spin.setSuffix(" °")
        self._fft_ideal_angle_spin.setFixedHeight(24)
        self._fft_ideal_angle_spin.valueChanged.connect(self._on_fft_ideal_changed)
        ideal_grid.addWidget(QLabel("Ideal:"), 0, 0)
        ideal_grid.addWidget(self._fft_ideal_combo, 0, 1)
        ideal_grid.addWidget(QLabel("|a|:"), 0, 2)
        ideal_grid.addWidget(self._fft_ideal_a_spin, 0, 3)
        ideal_grid.addWidget(QLabel("|b|:"), 1, 0)
        ideal_grid.addWidget(self._fft_ideal_b_spin, 1, 1)
        ideal_grid.addWidget(QLabel("Angle:"), 1, 2)
        ideal_grid.addWidget(self._fft_ideal_angle_spin, 1, 3)
        ideal_grid.setColumnStretch(1, 1)
        ideal_grid.setColumnStretch(3, 1)
        apply_lay.addLayout(ideal_grid)
        self._fft_preserve_orientation_cb = QCheckBox("Preserve image orientation")
        self._fft_preserve_orientation_cb.setFont(QFont("Helvetica", 9))
        self._fft_preserve_orientation_cb.setChecked(True)
        self._fft_preserve_orientation_cb.setToolTip("Apply only stretch/shear and remove the fitted rigid rotation.")
        self._fft_preserve_orientation_cb.toggled.connect(self._on_fft_ideal_changed)
        apply_lay.addWidget(self._fft_preserve_orientation_cb)

        opts_grp = QGroupBox("Advanced correction options")
        opts_grp.setFont(QFont("Helvetica", 9))
        opts_grp.setCheckable(True)
        opts_grp.setChecked(False)
        opts_lay = QGridLayout(opts_grp)
        opts_lay.setContentsMargins(8, 8, 8, 6)
        self._fft_expand_cb = QCheckBox("Expand canvas")
        self._fft_expand_cb.setChecked(True)
        self._fft_interp_combo = QComboBox()
        self._fft_interp_combo.addItems(["Bilinear", "Nearest", "Bicubic"])
        self._fft_fill_combo = QComboBox()
        self._fft_fill_combo.addItems(["NaN", "Background", "Zero"])
        opts_lay.addWidget(self._fft_expand_cb, 0, 0)
        opts_lay.addWidget(QLabel("Interpolation:"), 0, 1)
        opts_lay.addWidget(self._fft_interp_combo, 0, 2)
        opts_lay.addWidget(QLabel("Fill:"), 0, 3)
        opts_lay.addWidget(self._fft_fill_combo, 0, 4)
        apply_lay.addWidget(opts_grp)

        self._fft_correction_lbl = QLabel("Create a reciprocal grid to compute correction.")
        self._fft_correction_lbl.setFont(QFont("Courier", 8))
        self._fft_correction_lbl.setWordWrap(True)
        self._fft_correction_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        apply_lay.addWidget(self._fft_correction_lbl)
        action_row = QHBoxLayout()
        self._fft_preview_btn = QPushButton("Preview correction")
        self._fft_preview_btn.setFont(QFont("Helvetica", 9))
        self._fft_preview_btn.setFixedHeight(24)
        self._fft_preview_btn.setEnabled(False)
        self._fft_preview_btn.setToolTip("Temporarily show the corrected image in the parent viewer.")
        self._fft_preview_btn.clicked.connect(self._on_fft_preview_correction)
        self._fft_clear_preview_btn = QPushButton("Clear preview")
        self._fft_clear_preview_btn.setFont(QFont("Helvetica", 9))
        self._fft_clear_preview_btn.setFixedHeight(24)
        self._fft_clear_preview_btn.setEnabled(False)
        self._fft_clear_preview_btn.clicked.connect(self._on_fft_clear_preview)
        self._fft_apply_btn = QPushButton("Apply correction")
        self._fft_apply_btn.setObjectName("accentBtn")
        self._fft_apply_btn.setFont(QFont("Helvetica", 9))
        self._fft_apply_btn.setFixedHeight(24)
        self._fft_apply_btn.setEnabled(False)
        self._fft_apply_btn.setToolTip("Append this FFT-derived affine lattice correction to the image processing history.")
        self._fft_apply_btn.clicked.connect(self._on_fft_apply_correction)
        action_row.addWidget(self._fft_preview_btn)
        action_row.addWidget(self._fft_clear_preview_btn)
        action_row.addStretch(1)
        action_row.addWidget(self._fft_apply_btn)
        apply_lay.addLayout(action_row)
        corr_lay.addWidget(apply_grp)
        corr_lay.addStretch(1)
        corr_scroll.setWidget(corr_inner)
        self._tab_widget.addTab(corr_scroll, "Correction")

        # ── Inspect tab ─────────────────────────────────────────────────────
        inspect_tab = QWidget()
        inspect_lay = QHBoxLayout(inspect_tab)
        inspect_lay.setContentsMargins(6, 6, 6, 6)
        inspect_lay.setSpacing(8)
        intensity_grp = QGroupBox("Intensity")
        intensity_grp.setFont(QFont("Helvetica", 9))
        int_lay = QVBoxLayout(intensity_grp)
        self._hist_fig = Figure(figsize=(1, 1), dpi=90)
        self._hist_fig.patch.set_facecolor(bg)
        self._hist_canvas = FigureCanvasQTAgg(self._hist_fig)
        self._hist_ax = self._hist_fig.add_subplot(111)
        self._hist_ax.set_facecolor(bg)
        int_lay.addWidget(self._hist_canvas, 1)
        reset_intensity_btn = QPushButton("Reset range")
        reset_intensity_btn.setFont(QFont("Helvetica", 9))
        reset_intensity_btn.setFixedHeight(22)
        reset_intensity_btn.clicked.connect(self._reset_intensity)
        int_lay.addWidget(reset_intensity_btn)
        self._hist_canvas.mpl_connect("button_press_event", self._on_hist_press)
        self._hist_canvas.mpl_connect("motion_notify_event", self._on_hist_motion)
        self._hist_canvas.mpl_connect("button_release_event", self._on_hist_release)
        inspect_lay.addWidget(intensity_grp, 1)
        radial_grp = QGroupBox("Radial profile")
        radial_grp.setFont(QFont("Helvetica", 9))
        rad_lay = QVBoxLayout(radial_grp)
        self._radial_fig = Figure(figsize=(1, 1), dpi=90)
        self._radial_fig.patch.set_facecolor(bg)
        self._radial_canvas = FigureCanvasQTAgg(self._radial_fig)
        self._radial_ax = self._radial_fig.add_axes([0.11, 0.24, 0.85, 0.66])
        self._radial_ax.set_facecolor(bg)
        rad_lay.addWidget(self._radial_canvas, 1)
        inspect_lay.addWidget(radial_grp, 1)
        self._tab_widget.addTab(inspect_tab, "Inspect")

        # ── Advanced tab ────────────────────────────────────────────────────
        advanced_scroll = QScrollArea()
        advanced_scroll.setWidgetResizable(True)
        advanced_scroll.setFrameShape(QFrame.NoFrame)
        advanced_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        advanced_inner = QWidget()
        advanced_lay = QVBoxLayout(advanced_inner)
        advanced_lay.setContentsMargins(8, 8, 8, 6)
        advanced_lay.setSpacing(8)
        grid_grp = QGroupBox("Reciprocal grid details")
        grid_grp.setFont(QFont("Helvetica", 9))
        self._grid_tab_lay = QVBoxLayout(grid_grp)
        self._grid_tab_lay.setContentsMargins(8, 8, 8, 6)
        self._grid_placeholder_lbl = QLabel("Click Edit reciprocal grid to create controls.")
        self._grid_placeholder_lbl.setFont(QFont("Helvetica", 9))
        self._grid_placeholder_lbl.setAlignment(Qt.AlignCenter)
        self._grid_tab_lay.addWidget(self._grid_placeholder_lbl)
        advanced_lay.addWidget(grid_grp)

        piezo_grp = QGroupBox("Piezo calibration from picked peaks")
        piezo_grp.setFont(QFont("Helvetica", 9))
        piezo_lay = QVBoxLayout(piezo_grp)
        piezo_lay.setContentsMargins(8, 8, 8, 6)
        piezo_row = QHBoxLayout()
        piezo_row.addWidget(QLabel("Piezo X:"))
        self._bragg_cx_edit = QLineEdit("96.52")
        self._bragg_cx_edit.setFixedWidth(72)
        piezo_row.addWidget(self._bragg_cx_edit)
        piezo_row.addWidget(QLabel("Piezo Y:"))
        self._bragg_cy_edit = QLineEdit("96.52")
        self._bragg_cy_edit.setFixedWidth(72)
        piezo_row.addWidget(self._bragg_cy_edit)
        self._bragg_compute_btn = QPushButton("Compute piezo constants")
        self._bragg_compute_btn.setFont(QFont("Helvetica", 9))
        self._bragg_compute_btn.setFixedHeight(24)
        self._bragg_compute_btn.setEnabled(False)
        self._bragg_compute_btn.clicked.connect(self._compute_bragg_correction)
        piezo_row.addWidget(self._bragg_compute_btn)
        piezo_row.addStretch(1)
        piezo_lay.addLayout(piezo_row)
        self._bragg_results_txt = QPlainTextEdit()
        self._bragg_results_txt.setReadOnly(True)
        self._bragg_results_txt.setFont(QFont("Courier", 9))
        self._bragg_results_txt.setFixedHeight(76)
        self._bragg_results_txt.setPlaceholderText("Piezo results will appear here.")
        piezo_lay.addWidget(self._bragg_results_txt)
        self._bragg_copy_btn = QPushButton("Copy piezo results")
        self._bragg_copy_btn.setFont(QFont("Helvetica", 9))
        self._bragg_copy_btn.setFixedHeight(22)
        self._bragg_copy_btn.clicked.connect(self._copy_bragg_results)
        piezo_lay.addWidget(self._bragg_copy_btn)
        advanced_lay.addWidget(piezo_grp)
        advanced_lay.addStretch(1)
        advanced_scroll.setWidget(advanced_inner)
        self._grid_tab_index = self._tab_widget.addTab(advanced_scroll, "Advanced")

        self._fft_splitter = QSplitter(Qt.Vertical)
        self._fft_splitter.addWidget(fft_top)
        self._fft_splitter.addWidget(self._tab_widget)
        self._fft_splitter.setStretchFactor(0, 1)
        self._fft_splitter.setStretchFactor(1, 0)
        self._fft_splitter.setSizes([450, 330])
        right_col.addWidget(self._fft_splitter, 1)
        install_no_wheel_spinboxes(self._tab_widget)

        return right_col

    def _connect_canvas_events(self) -> None:
        """Wire mpl events for the FFT canvas and radial panel."""
        self._canvas_fft.mpl_connect("scroll_event",         self._on_scroll)
        self._canvas_fft.mpl_connect("button_press_event",   self._on_press)
        self._canvas_fft.mpl_connect("button_release_event", self._on_release)
        self._canvas_fft.mpl_connect("motion_notify_event",  self._on_motion)
        self._canvas_fft.mpl_connect("draw_event",           self._sync_tab_width)
        self._radial_canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _build(self):
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")

        lay = QVBoxLayout(self)
        lay.setSpacing(4)
        lay.setContentsMargins(6, 6, 6, 4)

        lay.addLayout(self._build_toolbar_row())

        # ── body row: compact real-space reference | FFT workspace ───────────
        body_row = QHBoxLayout()
        body_row.setSpacing(8)

        # ── left rail: real-space thumbnail + image info ─────────────────────
        left_panel = QWidget()
        self._left_panel = left_panel
        left_panel.setFixedWidth(260)
        left_col = QVBoxLayout(left_panel)
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(2)

        self._fig_real = Figure(figsize=(2.7, 2.5), dpi=90)
        self._fig_real.patch.set_facecolor(bg)
        self._canvas_real = FigureCanvasQTAgg(self._fig_real)
        self._canvas_real.setFixedHeight(245)
        self._ax_real = self._fig_real.add_subplot(111)
        self._ax_real.set_facecolor(bg)
        for sp in self._ax_real.spines.values():
            sp.set_color(fg)
        self._ax_real.tick_params(colors=fg, labelsize=8)
        self._fig_real.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
        left_col.addWidget(self._canvas_real)

        info_frame = QFrame()
        info_lay = QVBoxLayout(info_frame)
        info_lay.setContentsMargins(8, 6, 8, 4)
        info_lay.setSpacing(2)
        self._info_lbl = QLabel("")
        self._info_lbl.setFont(QFont("Courier", 9))
        self._info_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_lay.addWidget(self._info_lbl)
        info_lay.addStretch(1)
        left_col.addWidget(info_frame, 1)

        body_row.addWidget(left_panel)
        body_row.addLayout(self._build_fft_column(), 1)
        lay.addLayout(body_row, 1)

        # ── status bar ────────────────────────────────────────────────────────
        self._status_lbl = QLabel("")
        self._status_lbl.setFont(QFont("Helvetica", 8))
        self._status_lbl.setAlignment(Qt.AlignLeft)
        lay.addWidget(self._status_lbl)

        # ── canvas event connections ──────────────────────────────────────────
        self._connect_canvas_events()
        self._canvas_real.mpl_connect("motion_notify_event", self._on_motion)

    # ── FFT computation ────────────────────────────────────────────────────────

    @staticmethod
    def _tukey_1d(N: int, alpha: float = 0.5) -> np.ndarray:
        w = np.ones(N, dtype=np.float64)
        ramp = int(alpha * N / 2)
        if ramp > 0:
            t = np.linspace(0, 1, ramp)
            w[:ramp] = 0.5 * (1 - np.cos(np.pi * t))
            w[N - ramp:] = w[:ramp][::-1]
        return w

    def _make_window(self) -> np.ndarray:
        Ny, Nx = self._arr.shape
        if self._window_mode == "hann":
            return np.outer(np.hanning(Ny), np.hanning(Nx))
        if self._window_mode == "tukey":
            return np.outer(self._tukey_1d(Ny), self._tukey_1d(Nx))
        return np.ones((Ny, Nx), dtype=np.float64)

    def _recompute_fft(self):
        arr = self._arr.copy()
        finite = arr[np.isfinite(arr)]
        arr[~np.isfinite(arr)] = float(np.nanmedian(finite)) if finite.size > 0 else 0.0
        arr -= arr.mean()
        arr *= self._make_window()

        F = np.fft.fftshift(np.fft.fft2(arr))
        self._fft_mag = np.abs(F)

        Ny, Nx = arr.shape
        try:
            w_nm = float(self._scan_range_m[0]) * 1e9
            h_nm = float(self._scan_range_m[1]) * 1e9
            dx_nm = w_nm / Nx if Nx > 0 else 1.0
            dy_nm = h_nm / Ny if Ny > 0 else 1.0
        except (TypeError, ValueError, IndexError):
            dx_nm, dy_nm = 1.0, 1.0

        self._qx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx_nm))
        self._qy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy_nm))
        self._fft_xlim = (float(self._qx[0]),  float(self._qx[-1]))
        self._fft_ylim = (float(self._qy[-1]), float(self._qy[0]))

    def _compute_display_fft(self) -> np.ndarray:
        mag = self._fft_mag.copy()
        Ny, Nx = mag.shape
        cy, cx = Ny // 2, Nx // 2
        r = max(1, min(Ny, Nx) // 60)
        y0, y1 = max(0, cy - r), min(Ny, cy + r + 1)
        x0, x1 = max(0, cx - r), min(Nx, cx + r + 1)
        if self._dc_mode == "zero":
            mag[y0:y1, x0:x1] = 0.0
        elif self._dc_mode == "mask":
            mag = mag.astype(np.float64)
            mag[y0:y1, x0:x1] = np.nan
        if self._scale_mode == "log":
            # log10 gives meaningful contrast regardless of the absolute scale of
            # the input data (log1p degenerates to linear when magnitudes are << 1,
            # which is always the case for SI-unit STM height data).
            with np.errstate(divide="ignore", invalid="ignore"):
                log_mag = np.log10(mag)
            mag = np.where(np.isfinite(log_mag), log_mag, np.nan)
        finite = mag[np.isfinite(mag)]
        self._disp_range = (
            (float(finite.min()), float(finite.max())) if finite.size > 0 else (0.0, 1.0)
        )
        return mag

    # ── drawing ────────────────────────────────────────────────────────────────

    def _redraw(self):
        self._redraw_real_panel()
        self._redraw_fft_panel()

    def _redraw_real_panel(self):
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")
        ax = self._ax_real
        ax.cla()
        ax.set_facecolor(bg)
        try:
            w_nm = float(self._scan_range_m[0]) * 1e9
            h_nm = float(self._scan_range_m[1]) * 1e9
        except Exception:
            w_nm, h_nm = float(self._arr.shape[1]), float(self._arr.shape[0])
        ax.imshow(
            self._arr, cmap=self._colormap, origin="upper",
            extent=[0, w_nm, h_nm, 0], aspect="equal",
        )
        ax.set_title("Real space", fontsize=9, color=fg)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(colors=fg, labelsize=7, length=0)
        for spine in ax.spines.values():
            spine.set_color(fg)
        self._canvas_real.draw_idle()

    def _redraw_fft_panel(self):
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")
        ax = self._ax_fft
        ax.cla()
        ax.set_facecolor(bg)
        disp = self._compute_display_fft()
        lo, hi = self._disp_range
        vmin_val = lo + self._vmin_frac * (hi - lo)
        vmax_val = lo + self._vmax_frac * (hi - lo)
        extent_q = [
            float(self._qx[0]), float(self._qx[-1]),
            float(self._qy[-1]), float(self._qy[0]),
        ]
        self._fft_im = ax.imshow(
            disp, cmap=self._fft_cmap, origin="upper",
            extent=extent_q, aspect="equal",
            vmin=vmin_val, vmax=vmax_val,
        )
        ax.set_xlim(*self._fft_xlim)
        ax.set_ylim(*self._fft_ylim)
        scale_lbl = "log₁₀|FFT|" if self._scale_mode == "log" else "|FFT|"
        ax.set_title(f"FFT  ({scale_lbl})", fontsize=10, color=fg)
        ax.set_xlabel("q_x  (nm⁻¹)", fontsize=9, color=fg)
        ax.set_ylabel("q_y  (nm⁻¹)", fontsize=9, color=fg)
        ax.tick_params(colors=fg, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(fg)
        ax.axhline(0, color=fg, lw=0.4, alpha=0.35)
        ax.axvline(0, color=fg, lw=0.4, alpha=0.35)
        self._bragg_artists = []
        self._draw_bragg_overlay()
        self._update_histogram(disp)
        self._update_radial_profile(disp)
        self._canvas_fft.draw_idle()

    # ── tab-width sync ────────────────────────────────────────────────────────

    def _sync_tab_width(self, event=None):
        """Compatibility hook for subclasses that still sync auxiliary controls."""
        return

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_tab_width()

    def _on_focus_fft_toggled(self, checked: bool) -> None:
        self._focus_fft_active = bool(checked)
        left = getattr(self, "_left_panel", None)
        if left is not None:
            left.setVisible(not checked)
        tabs = getattr(self, "_tab_widget", None)
        if tabs is not None:
            tabs.setVisible(not checked)
        btn = getattr(self, "_focus_fft_btn", None)
        if btn is not None:
            btn.setText("Show tools" if checked else "Focus FFT")

    def _set_status_text(self, text: str) -> None:
        self._status_lbl.setText(text)
        lbl = getattr(self, "_cursor_readout_lbl", None)
        if lbl is not None:
            lbl.setText(text or "Move over the FFT")

    # ── zoom / pan ─────────────────────────────────────────────────────────────

    def _zoom_fit(self):
        self._fft_xlim = (float(self._qx[0]),  float(self._qx[-1]))
        self._fft_ylim = (float(self._qy[-1]), float(self._qy[0]))
        self._ax_fft.set_xlim(*self._fft_xlim)
        self._ax_fft.set_ylim(*self._fft_ylim)
        self._canvas_fft.draw_idle()

    def _zoom_centre(self):
        qx_half = (float(self._qx[-1]) - float(self._qx[0])) * 0.25
        qy_half = (float(self._qy[-1]) - float(self._qy[0])) * 0.25
        self._fft_xlim = (-qx_half, qx_half)
        self._fft_ylim = (qy_half, -qy_half)
        self._ax_fft.set_xlim(*self._fft_xlim)
        self._ax_fft.set_ylim(*self._fft_ylim)
        self._canvas_fft.draw_idle()

    def _zoom_by(self, factor: float, cx: float | None = None, cy: float | None = None):
        xl, xr = self._fft_xlim
        yb, yt = self._fft_ylim
        xc = cx if cx is not None else (xl + xr) / 2
        yc = cy if cy is not None else (yb + yt) / 2
        min_x, min_y = self._minimum_fft_spans()
        self._fft_xlim = self._interval_with_min_span(
            xc + (xl - xc) * factor,
            xc + (xr - xc) * factor,
            min_x,
        )
        self._fft_ylim = self._interval_with_min_span(
            yc + (yb - yc) * factor,
            yc + (yt - yc) * factor,
            min_y,
        )
        self._ax_fft.set_xlim(*self._fft_xlim)
        self._ax_fft.set_ylim(*self._fft_ylim)
        self._canvas_fft.draw_idle()

    def _minimum_fft_spans(self) -> tuple[float, float]:
        """Return zoom-span floors so trackpad bursts cannot collapse the FFT view.

        Two floors are taken and the larger is used:
          • 1e-4 × full span  — absolute safety floor (prevents floating-point collapse)
          • 4 px / N × full span — keeps at least 4 reciprocal-lattice pixels visible
            so the user always sees meaningful structure rather than a single point.
        """
        Ny, Nx = self._arr.shape[:2]
        full_x = abs(float(self._qx[-1]) - float(self._qx[0]))
        full_y = abs(float(self._qy[-1]) - float(self._qy[0]))
        return (
            max(full_x * 1e-4, full_x * 4.0 / max(1, Nx)),
            max(full_y * 1e-4, full_y * 4.0 / max(1, Ny)),
        )

    @staticmethod
    def _interval_with_min_span(a: float, b: float, minimum: float) -> tuple[float, float]:
        span = b - a
        if abs(span) >= minimum:
            return (a, b)
        centre = (a + b) * 0.5
        sign = 1.0 if span >= 0 else -1.0
        half = minimum * 0.5
        return (centre - sign * half, centre + sign * half)

    @staticmethod
    def _scroll_has_zoom_modifier(event) -> bool:
        key = str(getattr(event, "key", "") or "").lower()
        if any(token in key for token in ("ctrl", "control", "cmd", "command", "meta", "super")):
            return True
        modifiers = QApplication.keyboardModifiers()
        return bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier))

    def _fft_lattice_overlay_wants_event(self, event) -> bool:
        """Return True if the lattice overlay wants to handle this canvas event."""
        overlay = getattr(self, "_fft_lattice_overlay", None)
        if overlay is None:
            return False
        return bool(overlay.hit_test_event(event))

    def _on_fft_lattice_drag_state_changed(self, dragging: bool) -> None:
        self._fft_lattice_drag_active = bool(dragging)
        if dragging:
            self._pan_anchor = None

    # ── canvas events ──────────────────────────────────────────────────────────

    def _on_scroll(self, event):
        if (
            event.inaxes is not self._ax_fft
            or self._fft_lattice_drag_active
            or not self._scroll_has_zoom_modifier(event)
        ):
            return
        # 0.88 per notch ≈ 12 % zoom step — smoother than the previous 0.65 (35 %)
        # and closer to native macOS trackpad zoom feel.
        factor = 0.88 if event.step > 0 else 1.0 / 0.88
        self._zoom_by(factor, event.xdata, event.ydata)

    def _on_press(self, event):
        if (
            event.inaxes is self._ax_fft
            and event.button == 1
            and event.xdata is not None
            and event.ydata is not None
        ):
            if self._bragg_pick_mode_active():
                self._on_bragg_pick_click(event)
                return
            if not self._fft_lattice_overlay_wants_event(event):
                self._pan_anchor = (
                    event.xdata, event.ydata,
                    self._fft_xlim, self._fft_ylim,
                )

    def _on_release(self, event):
        self._pan_anchor = None

    def _on_motion(self, event):
        if self._fft_lattice_drag_active:
            self._pan_anchor = None
            return
        if self._pan_anchor is not None and event.inaxes is self._ax_fft:
            if event.xdata is None or event.ydata is None:
                return
            x0, y0, xlim0, ylim0 = self._pan_anchor
            dx = x0 - event.xdata
            dy = y0 - event.ydata
            self._fft_xlim = (xlim0[0] + dx, xlim0[1] + dx)
            self._fft_ylim = (ylim0[0] + dy, ylim0[1] + dy)
            self._ax_fft.set_xlim(*self._fft_xlim)
            self._ax_fft.set_ylim(*self._fft_ylim)
            self._canvas_fft.draw_idle()

        if event.inaxes is self._ax_fft and event.xdata is not None:
            qx, qy = event.xdata, event.ydata
            q = np.hypot(qx, qy)
            if q > 0:
                d_nm = 1.0 / q
                d_str = f"{d_nm:.2f} nm" if d_nm >= 1.0 else f"{d_nm * 10:.2f} Å"
            else:
                d_str = "∞"
            theta = np.degrees(np.arctan2(qy, qx))
            self._set_status_text(
                f"q_x={qx:+.3f}  q_y={qy:+.3f}  |q|={q:.3f} nm⁻¹  "
                f"d={d_str}  θ={theta:.1f}°"
            )
        elif event.inaxes is self._radial_ax and event.xdata is not None:
            q = event.xdata
            val = event.ydata
            scale_lbl = "log₁₀|FFT|" if self._scale_mode == "log" else "|FFT|"
            if q > 0:
                d_nm = 1.0 / q
                d_str = f"{d_nm:.2f} nm" if d_nm >= 1.0 else f"{d_nm * 10:.2f} Å"
            else:
                d_str = "∞"
            self._set_status_text(
                f"q={q:.3f} nm⁻¹  d={d_str}  ⟨{scale_lbl}⟩={val:.4g}"
            )
        elif event.inaxes is getattr(self, "_ax_real", None) and event.xdata is not None:
            self._set_status_text(
                f"x={event.xdata:.2f} nm  y={event.ydata:.2f} nm"
            )
        else:
            self._set_status_text("")

    # ── toolbar callbacks ──────────────────────────────────────────────────────

    def _on_scale_changed(self, idx: int):
        self._scale_mode = "log" if idx == 0 else "linear"
        self._redraw()

    def _on_window_changed(self, idx: int):
        self._window_mode = ["hann", "none", "tukey"][idx]
        self._recompute_fft()
        self._redraw()

    def _on_dc_changed(self, idx: int):
        self._dc_mode = ["zero", "keep", "mask"][idx]
        self._redraw()

    def _on_cmap_changed(self, idx: int):
        self._fft_cmap = self._cmap_options[idx]
        if self._fft_im is not None:
            self._fft_im.set_cmap(self._fft_cmap)
            self._canvas_fft.draw_idle()

    def _update_histogram(self, disp: np.ndarray):
        fg = self._theme.get("fg", "#dddddd")
        bg = self._theme.get("bg", "#1e1e1e")
        ax = self._hist_ax
        ax.cla()
        ax.set_facecolor(bg)
        self._vmin_line = None
        self._vmax_line = None
        finite = disp[np.isfinite(disp)]
        if finite.size > 0:
            lo, hi = self._disp_range
            ax.hist(finite.ravel(), bins=256, range=(lo, hi),
                    color="#6699bb", alpha=0.9, linewidth=0, log=True)
            vmin_val = lo + self._vmin_frac * (hi - lo)
            vmax_val = lo + self._vmax_frac * (hi - lo)
            ylo, yhi = ax.get_ylim()
            self._vmin_line = ax.axvline(vmin_val, color="#ff6060", lw=2.0, zorder=5)
            self._vmax_line = ax.axvline(vmax_val, color="#50ee70", lw=2.0, zorder=5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(ylo, yhi)
        ax.set_yticks([])
        ax.tick_params(colors=fg, labelsize=8, length=3)
        for sp in ax.spines.values():
            sp.set_color(fg)
        self._hist_fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.22)
        self._hist_canvas.draw_idle()

    def _apply_intensity(self):
        """Fast path: update FFT clim and histogram markers without full redraw."""
        if self._fft_im is None:
            return
        lo, hi = self._disp_range
        vmin_val = lo + self._vmin_frac * (hi - lo)
        vmax_val = lo + self._vmax_frac * (hi - lo)
        self._fft_im.set_clim(vmin_val, vmax_val)
        self._canvas_fft.draw_idle()
        self._update_histogram_markers()

    def _update_histogram_markers(self):
        if self._vmin_line is None or self._vmax_line is None:
            return
        lo, hi = self._disp_range
        vmin_val = lo + self._vmin_frac * (hi - lo)
        vmax_val = lo + self._vmax_frac * (hi - lo)
        self._vmin_line.set_xdata([vmin_val, vmin_val])
        self._vmax_line.set_xdata([vmax_val, vmax_val])
        self._hist_canvas.draw_idle()

    def _on_hist_press(self, event):
        if event.inaxes is not self._hist_ax or event.xdata is None:
            return
        lo, hi = self._disp_range
        if hi <= lo:
            return
        x = event.xdata
        span = hi - lo
        tol = span * 0.04
        vmin_val = lo + self._vmin_frac * span
        vmax_val = lo + self._vmax_frac * span
        d_min = abs(x - vmin_val)
        d_max = abs(x - vmax_val)
        if d_min <= tol or d_max <= tol:
            self._hist_drag = "vmin" if d_min <= d_max else "vmax"

    def _on_hist_motion(self, event):
        if self._hist_drag is None or event.inaxes is not self._hist_ax or event.xdata is None:
            return
        lo, hi = self._disp_range
        if hi <= lo:
            return
        frac = max(0.0, min(1.0, (event.xdata - lo) / (hi - lo)))
        if self._hist_drag == "vmin":
            self._vmin_frac = min(frac, self._vmax_frac)
        else:
            self._vmax_frac = max(frac, self._vmin_frac)
        self._apply_intensity()

    def _on_hist_release(self, event):
        self._hist_drag = None

    def _reset_intensity(self):
        self._vmin_frac = 0.0
        self._vmax_frac = 1.0
        self._apply_intensity()

    def _update_info_panel(self):
        Ny, Nx = self._arr.shape
        try:
            w_nm = float(self._scan_range_m[0]) * 1e9
            h_nm = float(self._scan_range_m[1]) * 1e9
            dx_nm = w_nm / Nx if Nx > 0 else 1.0
        except Exception:
            w_nm = h_nm = float(Nx)
            dx_nm = 1.0
        dqx = 1.0 / w_nm if w_nm > 0 else 0.0
        q_ny = 1.0 / (2.0 * dx_nm) if dx_nm > 0 else 0.0
        self._info_lbl.setText(
            f"Image:    {Nx} × {Ny} px\n"
            f"Size:     {w_nm:.3g} × {h_nm:.3g} nm\n"
            f"px size:  {dx_nm:.4g} nm\n"
            f"q-res:    {dqx:.4g} nm⁻¹\n"
            f"Nyquist:  {q_ny:.3g} nm⁻¹"
        )

    def _update_radial_profile(self, disp: np.ndarray):
        fg = self._theme.get("fg", "#dddddd")
        bg = self._theme.get("bg", "#1e1e1e")
        ax = self._radial_ax
        ax.cla()
        ax.set_facecolor(bg)
        if self._qx is None or self._qy is None:
            self._radial_canvas.draw_idle()
            return
        qx_2d = self._qx[np.newaxis, :]
        qy_2d = self._qy[:, np.newaxis]
        q_map = np.sqrt(qx_2d ** 2 + qy_2d ** 2)
        Ny, Nx = disp.shape
        n_bins = min(Nx, Ny) // 2
        q_max = float(q_map.max())
        q_bins = np.linspace(0.0, q_max, n_bins + 1)
        q_centers = 0.5 * (q_bins[:-1] + q_bins[1:])
        flat_q = q_map.ravel()
        flat_d = disp.ravel()
        valid = np.isfinite(flat_d)
        idx = np.clip(np.digitize(flat_q, q_bins) - 1, 0, n_bins - 1)
        profile = np.zeros(n_bins)
        counts  = np.zeros(n_bins, dtype=np.int64)
        np.add.at(profile, idx[valid], flat_d[valid])
        np.add.at(counts,  idx[valid], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            profile = np.where(counts > 0, profile / counts, np.nan)
        good = np.isfinite(profile)
        if good.any():
            ax.plot(q_centers[good], profile[good], color="#88bbee", lw=1.0)
        scale_lbl = "log₁₀|FFT|" if self._scale_mode == "log" else "|FFT|"
        ax.set_xlabel("q  (nm⁻¹)", fontsize=8, color=fg)
        ax.set_ylabel(f"⟨{scale_lbl}⟩", fontsize=8, color=fg)
        ax.tick_params(colors=fg, labelsize=7, length=3)
        for sp in ax.spines.values():
            sp.set_color(fg)
        self._radial_fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.26)
        self._radial_canvas.draw_idle()

    # ── FFT-derived lattice correction ───────────────────────────────────────

    def _on_bragg_symmetry_changed(self, _idx: int) -> None:
        self._on_bragg_changed()
        self._sync_fft_ideal_from_symmetry()
        self._refresh_fft_correction_ui()

    def _on_fft_grid_changed(self, _grid=None) -> None:
        self._refresh_fft_correction_ui()

    def _fft_pixel_sizes_m(self) -> tuple[float, float] | None:
        Ny, Nx = self._arr.shape[:2]
        try:
            w_m = float(self._scan_range_m[0])
            h_m = float(self._scan_range_m[1])
        except Exception:
            return None
        if Nx <= 0 or Ny <= 0 or w_m <= 0 or h_m <= 0:
            return None
        return (w_m / Nx, h_m / Ny)

    def _fft_measured_direct_vectors_nm(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        overlay = getattr(self, "_fft_lattice_overlay", None)
        if overlay is None:
            return None
        grid = overlay.grid()
        panel = getattr(self, "_fft_lattice_panel", None)
        cal = getattr(panel, "_cal", None)
        if grid is None or cal is None:
            return None
        return direct_lattice_vectors_from_reciprocal_grid(grid, cal)

    @staticmethod
    def _vec_len_nm(vec: tuple[float, float]) -> float:
        return math.hypot(float(vec[0]), float(vec[1]))

    @staticmethod
    def _vec_angle_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
        ax, ay = a
        bx, by = b
        la = math.hypot(ax, ay)
        lb = math.hypot(bx, by)
        if la <= 0 or lb <= 0:
            return float("nan")
        cosang = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
        return math.degrees(math.acos(cosang))

    def _sync_fft_ideal_from_symmetry(self) -> None:
        combo = getattr(self, "_fft_ideal_combo", None)
        if combo is None or combo.currentText() == "Custom":
            return
        symmetry = "Square" if self._bragg_sym_combo.currentIndex() == 0 else "Hexagonal"
        if combo.currentText() != symmetry:
            self._updating_fft_ideal = True
            try:
                combo.setCurrentText(symmetry)
            finally:
                self._updating_fft_ideal = False

    def _reference_lattice_a_nm(self) -> float:
        value = float(self._bragg_a_spin.value())
        return value * 0.1 if self._bragg_unit_combo.currentText() == "Å" else value

    def _sync_fft_ideal_values(self, a_nm: float, b_nm: float, angle: float) -> None:
        combo = self._fft_ideal_combo.currentText()
        if combo == "Custom":
            return
        ref_a_nm = self._reference_lattice_a_nm()
        if combo == "Square":
            side = ref_a_nm
            values = (side, side, 90.0)
        elif combo == "Rectangular":
            values = (ref_a_nm, b_nm, 90.0)
        elif combo == "Hexagonal":
            side = ref_a_nm
            values = (side, side, 120.0 if angle >= 90.0 else 60.0)
        else:
            values = (a_nm, b_nm, angle)
        self._updating_fft_ideal = True
        try:
            self._fft_ideal_a_spin.setValue(max(self._fft_ideal_a_spin.minimum(), values[0]))
            self._fft_ideal_b_spin.setValue(max(self._fft_ideal_b_spin.minimum(), values[1]))
            self._fft_ideal_angle_spin.setValue(min(179.0, max(1.0, values[2])))
            self._fft_ideal_a_spin.setEnabled(combo not in {"Match measured"})
            self._fft_ideal_b_spin.setEnabled(combo in {"Rectangular", "Custom"})
            self._fft_ideal_angle_spin.setEnabled(combo == "Custom")
        finally:
            self._updating_fft_ideal = False

    def _on_fft_ideal_changed(self, *_args) -> None:
        if self._updating_fft_ideal:
            return
        self._clear_fft_preview_if_active()
        self._refresh_fft_correction_ui()

    def _refresh_fft_correction_ui(self) -> None:
        measured_lbl = getattr(self, "_fft_measured_lbl", None)
        corr_lbl = getattr(self, "_fft_correction_lbl", None)
        status_lbl = getattr(self, "_fft_correction_status_lbl", None)
        if measured_lbl is None or corr_lbl is None:
            return
        self._fft_correction = None
        for btn_name in ("_fft_preview_btn", "_fft_apply_btn"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(False)

        vectors = self._fft_measured_direct_vectors_nm()
        if vectors is None:
            measured_lbl.setText("Create or edit a reciprocal grid to measure the direct lattice.")
            corr_lbl.setText("Create a reciprocal grid to compute correction.")
            if status_lbl is not None:
                status_lbl.setText("No reciprocal grid yet")
            return

        a_vec, b_vec = vectors
        a_len = self._vec_len_nm(a_vec)
        b_len = self._vec_len_nm(b_vec)
        angle = self._vec_angle_deg(a_vec, b_vec)
        measured_lbl.setText(
            f"direct |a| = {a_len:.4g} nm\n"
            f"direct |b| = {b_len:.4g} nm\n"
            f"direct angle = {angle:.2f}°"
        )
        self._sync_fft_ideal_values(a_len, b_len, angle)

        ideal = IdealLattice(
            a_nm=self._fft_ideal_a_spin.value(),
            b_nm=self._fft_ideal_b_spin.value(),
            angle_deg=self._fft_ideal_angle_spin.value(),
        )
        result = compute_correction(MeasuredLattice(a_nm=a_vec, b_nm=b_vec), ideal)
        if isinstance(result, str):
            corr_lbl.setText(f"Cannot compute correction:\n{result}")
            if status_lbl is not None:
                status_lbl.setText("Correction unavailable")
            return

        self._fft_correction = result
        y_scale = result.x_scale * result.y_over_x
        rot_state = "removed" if self._fft_preserve_orientation_cb.isChecked() else "applied"
        lines = [
            f"X ×{result.x_scale:.5f}   Y ×{y_scale:.5f}",
            f"shear {result.shear:.5f}   rigid rotation {result.polar_rotation_deg:.3f}° ({rot_state})",
        ]
        corr_lbl.setText("\n".join(lines))
        if status_lbl is not None:
            status_lbl.setText("FFT-derived affine correction ready")
        if self._get_image_fn is not None and self._preview_image_fn is not None:
            self._fft_preview_btn.setEnabled(True)
        if self._apply_correction_fn is not None:
            self._fft_apply_btn.setEnabled(True)

    def _fft_correction_options(self) -> dict:
        interp_map = {"Bilinear": "bilinear", "Nearest": "nearest", "Bicubic": "bicubic"}
        fill_map = {"NaN": "nan", "Background": "background", "Zero": "zero"}
        return {
            "interpolation": interp_map.get(self._fft_interp_combo.currentText(), "bilinear"),
            "fill_mode": fill_map.get(self._fft_fill_combo.currentText(), "nan"),
            "expand_canvas": self._fft_expand_cb.isChecked(),
            "preserve_orientation": self._fft_preserve_orientation_cb.isChecked(),
        }

    def _fft_correction_matrix_px(self) -> np.ndarray | None:
        if self._fft_correction is None:
            return None
        px = self._fft_pixel_sizes_m()
        if px is None:
            return None
        opts = self._fft_correction_options()
        return lattice_correction_matrix_px(
            self._fft_correction,
            pixel_size_x_m=px[0],
            pixel_size_y_m=px[1],
            preserve_orientation=opts["preserve_orientation"],
        )

    def _clear_fft_preview_if_active(self) -> None:
        if self._fft_preview_active and self._clear_preview_fn is not None:
            self._clear_preview_fn()
            self._fft_preview_active = False
            if getattr(self, "_fft_clear_preview_btn", None) is not None:
                self._fft_clear_preview_btn.setEnabled(False)

    def _on_fft_preview_correction(self) -> None:
        if self._fft_correction is None or self._get_image_fn is None:
            return
        T_px = self._fft_correction_matrix_px()
        if T_px is None:
            return
        arr = self._get_image_fn()
        if arr is None:
            return
        from probeflow.processing.image import affine_lattice_correction
        opts = self._fft_correction_options()
        try:
            corrected = affine_lattice_correction(
                arr,
                T_px,
                expand_canvas=opts["expand_canvas"],
                interpolation=opts["interpolation"],
                fill_mode=opts["fill_mode"],
            )
        except Exception as exc:
            self._fft_correction_lbl.setText(f"Preview failed: {exc}")
            return
        if self._preview_image_fn is not None:
            self._preview_image_fn(corrected)
            self._fft_preview_active = True
            self._fft_clear_preview_btn.setEnabled(True)
            self._fft_correction_status_lbl.setText("Previewing FFT-derived correction")

    def _on_fft_clear_preview(self) -> None:
        if self._clear_preview_fn is not None:
            self._clear_preview_fn()
        self._fft_preview_active = False
        self._fft_clear_preview_btn.setEnabled(False)
        self._refresh_fft_correction_ui()

    def _on_fft_apply_correction(self) -> None:
        if self._apply_correction_fn is None or self._fft_correction is None:
            return
        px = self._fft_pixel_sizes_m()
        if px is None or self._fft_correction_matrix_px() is None:
            return
        if self._fft_preview_active and self._clear_preview_fn is not None:
            self._clear_preview_fn()
            self._fft_preview_active = False
        opts = self._fft_correction_options()
        op_params = lattice_correction_operation_params(
            self._fft_correction,
            pixel_size_x_m=px[0],
            pixel_size_y_m=px[1],
            expand_canvas=opts["expand_canvas"],
            interpolation=opts["interpolation"],
            fill_mode=opts["fill_mode"],
            preserve_orientation=opts["preserve_orientation"],
        )
        if op_params is None:
            return
        op_params["source"] = "fft_reciprocal_grid"
        self._apply_correction_fn("affine_lattice_correction", op_params)
        self._fft_correction_lbl.setText(
            "Correction applied.\n"
            "FFT grid remains visible for reference."
        )
        self._fft_correction_status_lbl.setText("Correction applied")
        self._fft_preview_btn.setEnabled(False)
        self._fft_clear_preview_btn.setEnabled(False)
        self._fft_apply_btn.setEnabled(False)

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export FFT view", "fft_view.png",
            "PNG image (*.png);;All files (*)"
        )
        if path:
            self._fig_fft.savefig(path, dpi=150, bbox_inches="tight")

    def _clear_grid_tab(self) -> None:
        """Reset the embedded reciprocal-grid tab to its empty state."""
        lay = getattr(self, "_grid_tab_lay", None)
        if lay is None:
            return
        while lay.count():
            item = lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        lay.addWidget(self._grid_placeholder_lbl)

    def _set_grid_tab_panel(self, panel: QWidget) -> None:
        """Install the reciprocal-grid controls into the embedded Grid tab."""
        lay = getattr(self, "_grid_tab_lay", None)
        if lay is None:
            return
        while lay.count():
            item = lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        panel.setParent(self)
        lay.addWidget(panel)

    def _on_clear_fft_lattice(self):
        """Remove the FFT lattice overlay and reset its embedded controls."""
        self._clear_fft_preview_if_active()
        overlay = getattr(self, "_fft_lattice_overlay", None)
        if overlay is not None:
            overlay.clear()
            self._fft_lattice_overlay = None
        panel = getattr(self, "_fft_lattice_panel", None)
        if panel is not None:
            panel.setParent(None)
            panel.deleteLater()
            self._fft_lattice_panel = None
        self._fft_lattice_dock = None
        self._clear_grid_tab()
        if getattr(self, "_clear_grid_btn", None) is not None:
            self._clear_grid_btn.setEnabled(False)
        self._refresh_fft_correction_ui()

    def _on_edit_reciprocal_grid(self) -> None:
        self._on_open_fft_lattice(select_advanced=False)

    def _on_open_fft_lattice(self, select_advanced: bool = True):
        from probeflow.gui.lattice_grid import open_fft_tool

        existing = getattr(self, "_fft_lattice_overlay", None)
        existing_panel = getattr(self, "_fft_lattice_panel", None)
        if existing is not None and existing_panel is not None:
            if select_advanced:
                self._tab_widget.setCurrentIndex(self._grid_tab_index)
            return
        if self._qx is None or self._qy is None:
            return

        Ny, Nx = self._arr.shape[:2]
        overlay, panel = open_fft_tool(
            self._ax_fft, self._canvas_fft,
            self._qx, self._qy,
            (Ny, Nx), parent=self,
            on_change=self._on_fft_grid_changed,
        )
        self._fft_lattice_overlay = overlay
        self._fft_lattice_panel = panel
        self._fft_lattice_dock = None
        if hasattr(overlay, "set_drag_state_callback"):
            # Use WeakMethod so the overlay does not prevent this dialog from being
            # garbage-collected after it is closed. The overlay's callback wrapper
            # must handle the case where the weak reference has expired (returns None).
            _cb_ref = weakref.WeakMethod(self._on_fft_lattice_drag_state_changed)

            def _drag_state_cb(dragging: bool, _ref: weakref.ref = _cb_ref) -> None:
                cb = _ref()
                if cb is not None:
                    cb(dragging)

        overlay.set_drag_state_callback(_drag_state_cb)

        self._set_grid_tab_panel(panel)
        if select_advanced:
            self._tab_widget.setCurrentIndex(self._grid_tab_index)
        if getattr(self, "_clear_grid_btn", None) is not None:
            self._clear_grid_btn.setEnabled(True)
        self._refresh_fft_correction_ui()

    # ── Bragg ring overlay ─────────────────────────────────────────────────────

    def _on_bragg_changed(self):
        """Fast-path update: remove old ring artists, draw new ones, redraw."""
        for art in self._bragg_artists:
            try:
                art.remove()
            except ValueError:
                pass
        self._bragg_artists = []
        self._draw_bragg_overlay()
        self._canvas_fft.draw_idle()
        self._refresh_fft_correction_ui()

    def _draw_bragg_overlay(self):
        """Add predicted Bragg shell circle(s) to the FFT axes if enabled.

        Called both from ``_on_bragg_changed`` (fast path) and at the end of
        ``_redraw_fft_panel`` (after ``ax.cla()`` wipes the canvas).
        """
        if not getattr(self, "_bragg_enable_cb", None):
            return
        if not self._bragg_enable_cb.isChecked():
            self._bragg_radius_lbl.setText("Radius: —")
            self._draw_calib_pick_artists()
            return
        if self._qx is None or self._qy is None:
            self._draw_calib_pick_artists()
            return

        # ── user inputs ──────────────────────────────────────────────────────
        symmetry = "square" if self._bragg_sym_combo.currentIndex() == 0 else "hex"
        a_val = self._bragg_a_spin.value()
        unit = self._bragg_unit_combo.currentText()
        a_m = a_val * 1e-10 if unit == "Å" else a_val * 1e-9

        # ── compute low-index shell radii in nm⁻¹ ────────────────────────────
        from probeflow.processing.filters import bragg_shells, first_bragg_q
        try:
            q1 = first_bragg_q(a_m, symmetry) * 1e-9
        except ValueError:
            self._bragg_radius_lbl.setText("Radius: invalid input")
            return

        q_nyquist = 0.95 * min(
            float(np.nanmax(np.abs(self._qx))),
            float(np.nanmax(np.abs(self._qy))),
        )
        max_factor = q_nyquist / q1 if q1 > 0 else None
        shells = bragg_shells(
            symmetry,
            max_shells=self._bragg_max_shells_spin.value(),
            max_factor=max_factor,
        )
        theta = np.linspace(0.0, 2.0 * np.pi, 360)
        colours = ["#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#89b4fa", "#cba6f7"]
        styles = ["--", ":", "-.", (0, (5, 2, 1, 2))]
        label_lines: list[str] = []
        for idx, shell in enumerate(shells, start=1):
            q_shell = q1 * shell.factor
            art, = self._ax_fft.plot(
                q_shell * np.cos(theta), q_shell * np.sin(theta),
                color=colours[(idx - 1) % len(colours)],
                lw=1.2 if idx == 1 else 0.95,
                linestyle=styles[(idx - 1) % len(styles)],
                alpha=0.9 if idx == 1 else 0.75,
                zorder=7,
            )
            self._bragg_artists.append(art)
            d_angstrom = 1.0 / q_shell * 10.0 if q_shell > 0 else float("inf")
            label_lines.append(
                f"{idx}: {shell.label}  q={q_shell:.4f} nm⁻¹  "
                f"(plane d={d_angstrom:.3g} Å)"
            )
        if not shells:
            label_lines.append("No shells within FFT q-range")

        self._bragg_radius_lbl.setText("\n".join(label_lines))
        self._draw_calib_pick_artists()

    # ── calibration helpers ────────────────────────────────────────────────────

    def _bragg_pick_mode_active(self) -> bool:
        """True when clicks on the FFT canvas should add/remove calibration picks."""
        cb = getattr(self, "_bragg_enable_cb", None)
        pick_cb = getattr(self, "_bragg_pick_cb", None)
        return bool(
            cb is not None and cb.isChecked()
            and pick_cb is not None and pick_cb.isChecked()
        )

    def _draw_calib_pick_artists(self) -> None:
        """Draw calibration pick dots on the FFT axes (always, if any picks exist)."""
        for qx, qy in self._calib_picks:
            art, = self._ax_fft.plot(
                qx, qy, "o",
                color="#a6e3a1", markerfacecolor="none",
                markersize=9, markeredgewidth=1.8, zorder=8,
            )
            self._bragg_artists.append(art)

    def _update_calib_ui(self, *_args) -> None:
        """Refresh the picks label and the compute-button enabled state."""
        n = len(self._calib_picks)
        mode = "click FFT to add/remove" if self._bragg_pick_mode_active() else "enable Pick peaks to edit"
        self._bragg_picks_lbl.setText(f"Picks: {n}  ({mode})")
        self._bragg_compute_btn.setEnabled(n >= 3)
        self._bragg_compute_btn.setToolTip(
            "Fit ellipse and compute piezo correction"
            if n >= 3
            else f"Need at least 3 picks ({n} so far)"
        )

    def _on_bragg_pick_click(self, event) -> None:
        """Add or remove a calibration pick at the clicked FFT position."""
        qx_click, qy_click = float(event.xdata), float(event.ydata)

        # Check if clicking near an existing pick (10 screen pixels = hit)
        click_disp = self._ax_fft.transData.transform((qx_click, qy_click))
        for i, (qx, qy) in enumerate(self._calib_picks):
            pt_disp = self._ax_fft.transData.transform((qx, qy))
            dist = float(np.hypot(click_disp[0] - pt_disp[0],
                                  click_disp[1] - pt_disp[1]))
            if dist < 10.0:
                self._calib_picks.pop(i)
                self._update_calib_ui()
                self._on_bragg_changed()
                return

        if getattr(self, "_bragg_snap_cb", None) is not None and self._bragg_snap_cb.isChecked():
            try:
                from probeflow.processing.filters import snap_to_compact_peak_q
                snapped = snap_to_compact_peak_q(
                    self._fft_mag, self._qx, self._qy, qx_click, qy_click,
                )
                if snapped is not None:
                    qx_click, qy_click = snapped
            except Exception:
                pass

        # No existing pick nearby — add a new one
        self._calib_picks.append((qx_click, qy_click))
        self._update_calib_ui()
        self._on_bragg_changed()

    def _detect_bragg_peaks(self) -> None:
        """Auto-detect first-order Bragg peaks and store them as calibration picks."""
        from probeflow.processing.filters import (
            find_bragg_peaks_in_q_annulus,
            first_bragg_q,
        )
        if self._fft_mag is None or self._qx is None or self._qy is None:
            return

        symmetry = "square" if self._bragg_sym_combo.currentIndex() == 0 else "hex"
        a_val = self._bragg_a_spin.value()
        unit = self._bragg_unit_combo.currentText()
        a_m = a_val * 1e-10 if unit == "Å" else a_val * 1e-9

        try:
            q_predicted = first_bragg_q(a_m, symmetry) * 1e-9
        except ValueError:
            return

        expected = 4 if symmetry == "square" else 6
        peaks_q = find_bragg_peaks_in_q_annulus(
            self._fft_mag, self._qx, self._qy, q_predicted, expected_count=expected,
        )

        if peaks_q.size == 0:
            self._bragg_picks_lbl.setText("Peaks: none found in annulus")
            return

        self._calib_picks = [(float(qx), float(qy)) for qx, qy in peaks_q]
        self._update_calib_ui()
        if len(self._calib_picks) < expected:
            self._bragg_picks_lbl.setText(
                f"Picks: {len(self._calib_picks)}  "
                "(some sectors skipped; add missing picks manually)"
            )
        self._on_bragg_changed()

    def _clear_bragg_picks(self) -> None:
        self._calib_picks = []
        self._update_calib_ui()
        self._on_bragg_changed()

    def _compute_bragg_correction(self) -> None:
        """Fit an axis-aligned ellipse to the picks and report piezo corrections."""
        from probeflow.processing.filters import (
            fit_axis_aligned_ellipse,
            first_bragg_q,
            piezo_correction,
        )
        if len(self._calib_picks) < 3:
            return

        symmetry = "square" if self._bragg_sym_combo.currentIndex() == 0 else "hex"
        a_val = self._bragg_a_spin.value()
        unit = self._bragg_unit_combo.currentText()
        a_m = a_val * 1e-10 if unit == "Å" else a_val * 1e-9

        try:
            r_predicted_nm = first_bragg_q(a_m, symmetry) * 1e-9
        except ValueError as exc:
            self._bragg_results_txt.setPlainText(f"Error: {exc}")
            return

        # ── parse piezo values, tracking user precision ──────────────────────
        cx_str = self._bragg_cx_edit.text().strip()
        cy_str = self._bragg_cy_edit.text().strip()

        def _count_dec(s: str) -> int:
            return len(s.split(".")[1]) if "." in s else 0

        try:
            cx_old = float(cx_str)
            cy_old = float(cy_str)
        except ValueError:
            self._bragg_results_txt.setPlainText("Error: invalid piezo values")
            return
        if cx_old <= 0 or cy_old <= 0:
            self._bragg_results_txt.setPlainText("Error: piezo values must be positive")
            return
        cx_dec = _count_dec(cx_str)
        cy_dec = _count_dec(cy_str)

        # ── fit ellipse in nm⁻¹ ──────────────────────────────────────────────
        pts = np.array(self._calib_picks, dtype=np.float64)
        try:
            r_x_nm, r_y_nm, rms_nm = fit_axis_aligned_ellipse(pts)
        except ValueError as exc:
            self._bragg_results_txt.setPlainText(f"Fit error: {exc}")
            return

        # ── compute corrections ──────────────────────────────────────────────
        try:
            cx_new, cy_new = piezo_correction(
                r_x_nm, r_y_nm, r_predicted_nm, cx_old, cy_old,
            )
        except ValueError as exc:
            self._bragg_results_txt.setPlainText(f"Correction error: {exc}")
            return

        # ── format results ───────────────────────────────────────────────────
        lines = [
            f"r_predicted:  {r_predicted_nm:.4f} nm⁻¹",
            f"r_x_obs:      {r_x_nm:.4f} nm⁻¹",
            f"r_y_obs:      {r_y_nm:.4f} nm⁻¹",
            f"fit RMS:      {rms_nm:.4f} nm⁻¹",
            "",
            f"Piezo X:  {cx_old:.{cx_dec}f}  →  {cx_new:.{cx_dec}f}",
            f"Piezo Y:  {cy_old:.{cy_dec}f}  →  {cy_new:.{cy_dec}f}",
        ]
        self._bragg_results_txt.setPlainText("\n".join(lines))

    def _copy_bragg_results(self) -> None:
        text = self._bragg_results_txt.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
