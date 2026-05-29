from __future__ import annotations

import math
import weakref

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QInputDialog, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
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
from probeflow.gui.lattice_correction_ui import (
    KnownStructure,
    correction_main_lines,
    delete_structure,
    ideal_lattice_from_structure,
    load_known_structures,
    save_known_structures,
    structure_display_value_nm,
    upsert_structure,
)
from probeflow.gui.no_wheel import install_no_wheel_spinboxes
from probeflow.gui.viewer.display_range import DisplayRangeController
from probeflow.gui.viewer.display_sliders import DisplaySliderController
from probeflow.gui.viewer.histogram import HistogramPanel


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
        self._updating_structure = False
        self._known_structures = load_known_structures()
        self._active_known_structure = self._known_structures[0]

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
        self._last_fft_disp: np.ndarray | None = None
        self._fft_im = None
        self._fft_lattice_dock = None
        self._fft_lattice_panel = None
        self._focus_fft_active = False
        self._fft_cmap = "gray"
        self._cmap_options = ["gray", "gray_r", "inferno", "hot", "viridis", "plasma", "turbo"]
        self._bragg_artists: list = []
        self._calib_picks: list = []   # (qx_nm, qy_nm) in nm⁻¹

        self._fft_drs = DisplayRangeController(clip_low=0.0, clip_high=100.0, parent=self)

        self._build()

        self._display_slider_ctrl = DisplaySliderController(
            self._fft_drs,
            self._hist_panel,
            lambda: self._last_fft_disp,
            lambda: (1.0, "", "Intensity"),
        )
        self._fft_drs.rangeChanged.connect(self._apply_intensity_from_drs)
        self._hist_panel.minReleased.connect(
            lambda v: self._display_slider_ctrl.on_min_changed(v))
        self._hist_panel.maxReleased.connect(
            lambda v: self._display_slider_ctrl.on_max_changed(v))
        self._hist_panel.brightnessReleased.connect(
            lambda v: self._display_slider_ctrl.on_brightness_changed(v))
        self._hist_panel.contrastReleased.connect(
            lambda v: self._display_slider_ctrl.on_contrast_changed(v))
        self._hist_panel.rangeReleased.connect(self._on_fft_hist_range_released)
        self._hist_panel.resetRequested.connect(self._reset_intensity)
        self._hist_panel.autoClipRequested.connect(self._reset_intensity)

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
        self._scale_combo.setToolTip(
            "FFT display scale.  Log makes weak peaks visible; "
            "Linear shows raw amplitude ratios."
        )
        self._scale_combo.currentIndexChanged.connect(self._on_scale_changed)
        tb.addWidget(self._scale_combo)

        tb.addWidget(_lbl("LUT:"))
        self._cmap_combo = _combo(
            ["Gray", "Gray (inv.)", "Inferno", "Hot", "Viridis", "Plasma", "Turbo"], 96
        )
        self._cmap_combo.setToolTip("Colour map for the FFT intensity display.")
        self._cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        tb.addWidget(self._cmap_combo)

        tb.addWidget(_lbl("Window:"))
        self._window_combo = _combo(["Hann", "None", "Tukey"], 82)
        self._window_combo.setToolTip(
            "Apodisation window applied before the FFT to reduce ringing at image edges. "
            "Hann is recommended for most images."
        )
        self._window_combo.currentIndexChanged.connect(self._on_window_changed)
        tb.addWidget(self._window_combo)

        tb.addWidget(_lbl("DC:"))
        self._dc_combo = _combo(["Zero DC", "Keep DC", "Mask DC"], 95)
        self._dc_combo.setCurrentIndex(1)
        self._dc_combo.setToolTip(
            "How the zero-frequency (DC) component is treated.  "
            "'Keep DC' shows the bright central peak; "
            "'Zero DC' removes it; "
            "'Mask DC' hides it without removing it from the data."
        )
        self._dc_combo.currentIndexChanged.connect(self._on_dc_changed)
        tb.addWidget(self._dc_combo)

        tb.addStretch(1)

        show_tools_btn = QPushButton("Show tools")
        show_tools_btn.setFont(QFont("Helvetica", 9))
        show_tools_btn.setFixedHeight(24)
        show_tools_btn.setMinimumWidth(86)
        show_tools_btn.setCheckable(True)
        show_tools_btn.setToolTip("Show or hide the cursor details side panel")
        show_tools_btn.toggled.connect(self._on_show_tools_toggled)
        tb.addWidget(show_tools_btn)
        self._show_tools_btn = show_tools_btn

        focus_btn = QPushButton("Focus FFT")
        focus_btn.setFont(QFont("Helvetica", 9))
        focus_btn.setFixedHeight(24)
        focus_btn.setMinimumWidth(86)
        focus_btn.setCheckable(True)
        focus_btn.setToolTip(
            "Hide the real-space reference, side panel, and lower tools "
            "for a larger FFT view. Click again to exit."
        )
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
        exp_btn.setToolTip("Save the current FFT view as a PNG file.")
        exp_btn.clicked.connect(self._on_export)
        tb.addWidget(exp_btn)

        tb.addSpacing(8)
        grid_btn = QPushButton("Grid")
        grid_btn.setFont(QFont("Helvetica", 9))
        grid_btn.setFixedHeight(24)
        grid_btn.setMinimumWidth(68)
        grid_btn.setToolTip("Switch to the Grid tab (creates a lattice overlay if none exists)")
        grid_btn.clicked.connect(lambda _checked=False: self._on_open_fft_lattice(select_advanced=True))
        tb.addWidget(grid_btn)
        self._grid_lattice_btn = grid_btn

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
        self._fig_fft.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.09)

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
        side_lay.addWidget(cursor_title)
        side_lay.addWidget(self._cursor_readout_lbl)
        side_lay.addStretch(1)
        fft_top_lay.addWidget(side_panel)
        self._side_panel = side_panel
        side_panel.hide()  # hidden by default; "Show tools" in toolbar reveals it

        self._tab_widget = QTabWidget()
        self._tab_widget.setMinimumHeight(300)
        self._tab_widget.setFont(QFont("Helvetica", 9))

        # ── Tab 0: Inspect ───────────────────────────────────────────────────
        inspect_tab = QWidget()
        inspect_lay = QHBoxLayout(inspect_tab)
        inspect_lay.setContentsMargins(6, 6, 6, 6)
        inspect_lay.setSpacing(8)

        intensity_grp = QGroupBox("Intensity")
        intensity_grp.setFont(QFont("Helvetica", 9))
        int_lay = QVBoxLayout(intensity_grp)
        int_lay.setContentsMargins(6, 6, 6, 4)
        self._hist_panel = HistogramPanel(parent=intensity_grp)
        int_lay.addWidget(self._hist_panel, 1)
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

        # ── Tab 1: Grid ──────────────────────────────────────────────────────
        grid_scroll = QScrollArea()
        grid_scroll.setWidgetResizable(True)
        grid_scroll.setFrameShape(QFrame.NoFrame)
        grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        grid_inner = QWidget()
        grid_outer_lay = QVBoxLayout(grid_inner)
        grid_outer_lay.setContentsMargins(8, 8, 8, 6)
        grid_outer_lay.setSpacing(6)

        # Measurement summary — always visible at the top
        self._grid_measure_lbl = QLabel("No grid — click Draw Grid to start")
        self._grid_measure_lbl.setFont(QFont("Courier", 9))
        self._grid_measure_lbl.setWordWrap(True)
        self._grid_measure_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid_outer_lay.addWidget(self._grid_measure_lbl)

        # Grid extent control row (lives here, used by panel)
        extent_row = QHBoxLayout()
        extent_row.addWidget(QLabel("Grid orders ±:"))
        self._grid_extent_spin = QSpinBox()
        self._grid_extent_spin.setRange(1, 200)
        self._grid_extent_spin.setValue(12)
        self._grid_extent_spin.setFont(QFont("Helvetica", 9))
        self._grid_extent_spin.setFixedHeight(24)
        self._grid_extent_spin.setToolTip("How many reciprocal-lattice repeats to draw in each direction.")
        self._grid_extent_spin.valueChanged.connect(self._on_grid_extent_changed)
        extent_row.addWidget(self._grid_extent_spin)
        extent_row.addStretch(1)
        grid_outer_lay.addLayout(extent_row)

        # Panel container: initially shows placeholder + Draw Grid button;
        # replaced by FFTLatticePanel when a grid is created.
        grid_panel_container = QWidget()
        self._grid_tab_lay = QVBoxLayout(grid_panel_container)
        self._grid_tab_lay.setContentsMargins(0, 0, 0, 0)
        self._grid_placeholder_lbl = QLabel(
            "Click Draw Grid to overlay a reciprocal lattice.\n"
            "Drag the g₁/g₂ handles to align with Bragg peaks."
        )
        self._grid_placeholder_lbl.setFont(QFont("Helvetica", 9))
        self._grid_placeholder_lbl.setAlignment(Qt.AlignCenter)
        self._grid_placeholder_lbl.setWordWrap(True)
        self._grid_draw_btn = QPushButton("Draw Grid")
        self._grid_draw_btn.setFont(QFont("Helvetica", 9))
        self._grid_draw_btn.setFixedHeight(26)
        self._grid_draw_btn.setToolTip(
            "Create a hexagonal reciprocal-lattice overlay on the FFT. "
            "Drag the handles to align g₁/g₂ with Bragg peaks."
        )
        self._grid_draw_btn.clicked.connect(lambda: self._on_open_fft_lattice())
        self._grid_tab_lay.addWidget(self._grid_placeholder_lbl)
        self._grid_tab_lay.addWidget(self._grid_draw_btn)
        grid_outer_lay.addWidget(grid_panel_container)

        # Known structure section (moved here from Correction tab)
        ref_grp = QGroupBox("Known structure")
        ref_grp.setFont(QFont("Helvetica", 9))
        ref_grp.setMinimumHeight(120)
        ref_grid = QGridLayout(ref_grp)
        ref_grid.setContentsMargins(8, 7, 8, 4)
        ref_grid.setHorizontalSpacing(8)
        ref_grid.setVerticalSpacing(2)

        structure_row = QHBoxLayout()
        self._structure_combo = QComboBox()
        self._structure_combo.setFont(QFont("Helvetica", 9))
        self._structure_combo.setFixedHeight(24)
        self._structure_combo.setToolTip("Known surface lattice used for shell guides and undistortion target.")
        self._structure_combo.currentIndexChanged.connect(self._on_structure_selected)
        self._structure_save_btn = QPushButton("Save")
        self._structure_update_btn = QPushButton("Update")
        self._structure_delete_btn = QPushButton("Delete")
        for btn in (
            self._structure_save_btn,
            self._structure_update_btn,
            self._structure_delete_btn,
        ):
            btn.setFont(QFont("Helvetica", 8))
            btn.setFixedHeight(23)
            structure_row.addWidget(btn)
        self._structure_save_btn.clicked.connect(self._on_save_structure)
        self._structure_update_btn.clicked.connect(self._on_update_structure)
        self._structure_delete_btn.clicked.connect(self._on_delete_structure)

        self._bragg_enable_cb = QCheckBox("Show shell rings")
        self._bragg_enable_cb.setFont(QFont("Helvetica", 9))
        self._bragg_enable_cb.setChecked(True)
        self._bragg_enable_cb.setToolTip("Overlay the expected Bragg-shell radii from the known lattice.")
        self._bragg_enable_cb.toggled.connect(self._on_bragg_changed)

        self._bragg_sym_combo = QComboBox()
        self._bragg_sym_combo.addItems(["Square", "Hexagonal"])
        self._bragg_sym_combo.setFont(QFont("Helvetica", 9))
        self._bragg_sym_combo.setFixedHeight(24)
        self._bragg_sym_combo.setToolTip("Surface symmetry used for the predicted reciprocal-lattice shells.")
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
        ref_grid.addWidget(QLabel("Structure:"), 0, 0)
        ref_grid.addWidget(self._structure_combo, 0, 1, 1, 2)
        ref_grid.addLayout(structure_row, 0, 3)
        ref_grid.addWidget(QLabel("Symmetry:"), 1, 0)
        ref_grid.addWidget(self._bragg_sym_combo, 1, 1)
        ref_grid.addWidget(QLabel("Lattice a:"), 1, 2)
        ref_grid.addLayout(a_value_row, 1, 3)
        ref_grid.addWidget(QLabel("Shells:"), 2, 0)
        ref_grid.addWidget(self._bragg_max_shells_spin, 2, 1)
        ref_grid.addWidget(self._bragg_enable_cb, 2, 2, 1, 2)
        ref_grid.setColumnStretch(1, 1)
        ref_grid.setColumnStretch(3, 1)
        self._bragg_radius_lbl = QLabel("Shells: —")
        self._bragg_radius_lbl.setFont(QFont("Courier", 8))
        self._bragg_radius_lbl.setWordWrap(True)
        self._bragg_radius_lbl.setMaximumHeight(34)
        self._bragg_radius_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ref_grid.addWidget(self._bragg_radius_lbl, 3, 0, 1, 4)
        self._refresh_structure_combo(self._active_known_structure.name)
        self._apply_known_structure_to_fft(self._active_known_structure, refresh=False)
        grid_outer_lay.addWidget(ref_grp)

        # Compare section
        compare_grp = QGroupBox("Compare with known structure")
        compare_grp.setFont(QFont("Helvetica", 9))
        compare_lay = QVBoxLayout(compare_grp)
        compare_lay.setContentsMargins(8, 7, 8, 4)
        self._fft_measured_lbl = QLabel(
            "Draw a grid and select a known structure to see the comparison."
        )
        self._fft_measured_lbl.setFont(QFont("Courier", 8))
        self._fft_measured_lbl.setWordWrap(True)
        self._fft_measured_lbl.setMinimumHeight(32)
        self._fft_measured_lbl.setMaximumHeight(54)
        self._fft_measured_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        compare_lay.addWidget(self._fft_measured_lbl)
        grid_outer_lay.addWidget(compare_grp)

        # Clear Grid button at the bottom of the Grid tab
        self._clear_grid_btn = QPushButton("Clear Grid")
        self._clear_grid_btn.setFont(QFont("Helvetica", 9))
        self._clear_grid_btn.setFixedHeight(24)
        self._clear_grid_btn.setToolTip("Remove the reciprocal-space lattice overlay")
        self._clear_grid_btn.setEnabled(False)
        self._clear_grid_btn.clicked.connect(self._on_clear_fft_lattice)
        grid_outer_lay.addWidget(self._clear_grid_btn)
        grid_outer_lay.addStretch(1)
        grid_scroll.setWidget(grid_inner)
        self._grid_tab_index = self._tab_widget.addTab(grid_scroll, "Grid")

        # ── Tab 2: Correction ────────────────────────────────────────────────
        corr_tab = QWidget()
        corr_lay = QVBoxLayout(corr_tab)
        corr_lay.setSpacing(6)
        corr_lay.setContentsMargins(8, 8, 8, 6)

        # Ideal-lattice target controls (read-only in this tab; editable in Expert)
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
        self._fft_preserve_orientation_cb = QCheckBox("Preserve image orientation")
        self._fft_preserve_orientation_cb.setFont(QFont("Helvetica", 9))
        self._fft_preserve_orientation_cb.setChecked(True)
        self._fft_preserve_orientation_cb.setToolTip("Apply only stretch/shear and remove the fitted rigid rotation.")
        self._fft_preserve_orientation_cb.toggled.connect(self._on_fft_ideal_changed)
        self._fft_expand_cb = QCheckBox("Expand canvas")
        self._fft_expand_cb.setChecked(True)
        self._fft_expand_cb.setToolTip(
            "Grow the output canvas so no image content is clipped when "
            "the correction involves rotation."
        )
        self._fft_expand_cb.toggled.connect(self._on_fft_ideal_changed)
        self._fft_interp_combo = QComboBox()
        self._fft_interp_combo.addItems(["Bilinear", "Nearest", "Bicubic"])
        self._fft_fill_combo = QComboBox()
        self._fft_fill_combo.addItems(["NaN", "Background", "Zero"])

        self._fft_correction_lbl = QLabel("Align a reciprocal grid to compute correction factors.")
        self._fft_correction_lbl.setFont(QFont("Courier", 8))
        self._fft_correction_lbl.setWordWrap(True)
        self._fft_correction_lbl.setMinimumHeight(34)
        self._fft_correction_lbl.setMaximumHeight(60)
        self._fft_correction_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        corr_lay.addWidget(self._fft_correction_lbl)

        self._fft_correction_status_lbl = QLabel("No reciprocal grid yet")
        self._fft_correction_status_lbl.setFont(QFont("Helvetica", 8))
        self._fft_correction_status_lbl.setWordWrap(True)
        corr_lay.addWidget(self._fft_correction_status_lbl)

        opts_row = QHBoxLayout()
        opts_row.addWidget(self._fft_preserve_orientation_cb)
        opts_row.addSpacing(8)
        opts_row.addWidget(self._fft_expand_cb)
        opts_row.addStretch(1)
        corr_lay.addLayout(opts_row)

        action_row = QHBoxLayout()
        self._fft_preview_btn = QPushButton("Preview corrected image")
        self._fft_preview_btn.setFont(QFont("Helvetica", 9))
        self._fft_preview_btn.setFixedHeight(24)
        self._fft_preview_btn.setEnabled(False)
        self._fft_preview_btn.setToolTip("Show the affine-corrected real-space image in the left preview rail.")
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
        corr_lay.addLayout(action_row)
        self._apply_known_structure_to_fft(self._active_known_structure, refresh=False)
        corr_lay.addStretch(1)
        self._tab_widget.addTab(corr_tab, "Correction")

        # ── Tab 3: Expert ────────────────────────────────────────────────────
        advanced_scroll = QScrollArea()
        advanced_scroll.setWidgetResizable(True)
        advanced_scroll.setFrameShape(QFrame.NoFrame)
        advanced_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        advanced_inner = QWidget()
        advanced_lay = QVBoxLayout(advanced_inner)
        advanced_lay.setContentsMargins(8, 8, 8, 6)
        advanced_lay.setSpacing(8)

        def _collapsible_group(title: str, checked: bool = False):
            grp = QGroupBox(title)
            grp.setFont(QFont("Helvetica", 9))
            grp.setCheckable(True)
            grp.setChecked(checked)
            outer = QVBoxLayout(grp)
            outer.setContentsMargins(8, 8, 8, 6)
            content = QWidget()
            content_lay = QVBoxLayout(content)
            content_lay.setContentsMargins(0, 0, 0, 0)
            content_lay.setSpacing(6)
            outer.addWidget(content)
            content.setVisible(checked)
            grp.toggled.connect(content.setVisible)
            return grp, content_lay

        correction_opts_grp, opts_lay_outer = _collapsible_group(
            "Advanced correction options", checked=False,
        )
        target_grid = QGridLayout()
        target_grid.setHorizontalSpacing(8)
        target_grid.setVerticalSpacing(4)
        target_grid.addWidget(QLabel("Ideal:"), 0, 0)
        target_grid.addWidget(self._fft_ideal_combo, 0, 1)
        target_grid.addWidget(QLabel("|a|:"), 0, 2)
        target_grid.addWidget(self._fft_ideal_a_spin, 0, 3)
        target_grid.addWidget(QLabel("|b|:"), 1, 0)
        target_grid.addWidget(self._fft_ideal_b_spin, 1, 1)
        target_grid.addWidget(QLabel("Angle:"), 1, 2)
        target_grid.addWidget(self._fft_ideal_angle_spin, 1, 3)
        target_grid.setColumnStretch(1, 1)
        target_grid.setColumnStretch(3, 1)
        opts_lay_outer.addLayout(target_grid)
        opts_adv_row = QHBoxLayout()
        opts_adv_row.addWidget(QLabel("Interpolation:"))
        self._fft_interp_combo.setToolTip(
            "Resampling method used when remapping pixels.  "
            "Bilinear is a good default."
        )
        opts_adv_row.addWidget(self._fft_interp_combo)
        opts_adv_row.addSpacing(8)
        opts_adv_row.addWidget(QLabel("Fill:"))
        self._fft_fill_combo.setToolTip(
            "Value assigned to pixels outside the original image boundary "
            "after transformation."
        )
        opts_adv_row.addWidget(self._fft_fill_combo)
        opts_adv_row.addStretch(1)
        opts_lay_outer.addLayout(opts_adv_row)
        advanced_lay.addWidget(correction_opts_grp)

        piezo_grp, piezo_lay = _collapsible_group(
            "Scanner calibration (expert)", checked=False,
        )
        piezo_info_lbl = QLabel(
            "Measures piezo scan-speed constants from Bragg picks.\n"
            "This is for scanner calibration — not needed for image undistortion."
        )
        piezo_info_lbl.setFont(QFont("Helvetica", 8))
        piezo_info_lbl.setWordWrap(True)
        piezo_info_lbl.setStyleSheet("color: gray; font-style: italic;")
        piezo_lay.addWidget(piezo_info_lbl)
        peak_btn_row = QHBoxLayout()
        self._bragg_detect_btn = QPushButton("Detect peaks")
        self._bragg_detect_btn.setFont(QFont("Helvetica", 9))
        self._bragg_detect_btn.setFixedHeight(24)
        self._bragg_detect_btn.setToolTip("Auto-detect compact first-shell Bragg peaks as visual picks.")
        self._bragg_detect_btn.clicked.connect(self._detect_bragg_peaks)
        peak_btn_row.addWidget(self._bragg_detect_btn)
        self._bragg_clear_btn = QPushButton("Clear picks")
        self._bragg_clear_btn.setFont(QFont("Helvetica", 9))
        self._bragg_clear_btn.setFixedHeight(24)
        self._bragg_clear_btn.clicked.connect(self._clear_bragg_picks)
        peak_btn_row.addWidget(self._bragg_clear_btn)
        peak_btn_row.addStretch(1)
        piezo_lay.addLayout(peak_btn_row)

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
        piezo_lay.addLayout(pick_row)
        self._bragg_picks_lbl = QLabel("Picks: 0  (enable Pick peaks to edit)")
        self._bragg_picks_lbl.setFont(QFont("Helvetica", 9))
        piezo_lay.addWidget(self._bragg_picks_lbl)
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

        display_grp = QGroupBox("Display")
        display_grp.setFont(QFont("Helvetica", 9))
        display_grp_lay = QVBoxLayout(display_grp)
        display_grp_lay.setContentsMargins(8, 8, 8, 6)
        self._fft_equal_aspect_cb = QCheckBox("Preserve equal q_x/q_y aspect while zoomed")
        self._fft_equal_aspect_cb.setFont(QFont("Helvetica", 9))
        self._fft_equal_aspect_cb.setChecked(False)
        self._fft_equal_aspect_cb.setToolTip(
            "Off (default): zoom uses the full canvas width — the visible q-window "
            "matches the canvas shape so no space is wasted. "
            "On: q_x and q_y zoom symmetrically (equal q/pixel in both directions, "
            "but may leave blank space on a wide canvas)."
        )
        display_grp_lay.addWidget(self._fft_equal_aspect_cb)
        advanced_lay.addWidget(display_grp)
        advanced_lay.addStretch(1)
        advanced_scroll.setWidget(advanced_inner)
        self._tab_widget.addTab(advanced_scroll, "⚙ Expert")

        self._fft_splitter = QSplitter(Qt.Vertical)
        self._fft_splitter.addWidget(fft_top)
        self._fft_splitter.addWidget(self._tab_widget)
        self._fft_splitter.setStretchFactor(0, 1)
        self._fft_splitter.setStretchFactor(1, 0)
        self._fft_splitter.setSizes([420, 360])
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
        left_col.addWidget(info_frame)

        self._fft_preview_frame = QFrame()
        preview_lay = QVBoxLayout(self._fft_preview_frame)
        preview_lay.setContentsMargins(0, 4, 0, 0)
        preview_lay.setSpacing(2)
        preview_title = QLabel("Corrected preview")
        preview_title.setFont(QFont("Helvetica", 9, QFont.Bold))
        preview_lay.addWidget(preview_title)
        self._fig_preview = Figure(figsize=(2.7, 2.1), dpi=90)
        self._fig_preview.patch.set_facecolor(bg)
        self._canvas_preview = FigureCanvasQTAgg(self._fig_preview)
        self._canvas_preview.setFixedHeight(205)
        self._ax_preview = self._fig_preview.add_subplot(111)
        self._ax_preview.set_facecolor(bg)
        self._fig_preview.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        preview_lay.addWidget(self._canvas_preview)
        self._fft_preview_frame.setVisible(False)
        left_col.addWidget(self._fft_preview_frame)
        left_col.addStretch(1)

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
        self._last_fft_disp = mag
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

    def _show_fft_preview(self, arr: np.ndarray) -> None:
        """Render the corrected real-space preview in the left rail."""
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")
        ax = self._ax_preview
        ax.cla()
        ax.set_facecolor(bg)
        h, w = arr.shape[:2]
        px = self._fft_pixel_sizes_m()
        if px is not None:
            w_nm = w * px[0] * 1e9
            h_nm = h * px[1] * 1e9
        else:
            w_nm = float(w)
            h_nm = float(h)
        ax.imshow(
            arr, cmap=self._colormap, origin="upper",
            extent=[0, w_nm, h_nm, 0], aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(colors=fg, labelsize=7, length=0)
        for spine in ax.spines.values():
            spine.set_color(fg)
        self._fft_preview_frame.setVisible(True)
        self._canvas_preview.draw_idle()

    def _hide_fft_preview(self) -> None:
        frame = getattr(self, "_fft_preview_frame", None)
        if frame is not None:
            frame.setVisible(False)
        ax = getattr(self, "_ax_preview", None)
        if ax is not None:
            ax.cla()
        canvas = getattr(self, "_canvas_preview", None)
        if canvas is not None:
            canvas.draw_idle()
        self._fft_preview_active = False
        btn = getattr(self, "_fft_clear_preview_btn", None)
        if btn is not None:
            btn.setEnabled(False)

    def _redraw_fft_panel(self):
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")
        ax = self._ax_fft
        ax.cla()
        ax.set_facecolor(bg)
        disp = self._compute_display_fft()
        lo, hi = self._disp_range
        vmin_val, vmax_val = self._fft_drs.resolve(disp)
        if vmin_val is None:
            vmin_val, vmax_val = lo, hi
        extent_q = [
            float(self._qx[0]), float(self._qx[-1]),
            float(self._qy[-1]), float(self._qy[0]),
        ]
        self._fft_im = ax.imshow(
            disp, cmap=self._fft_cmap, origin="upper",
            extent=extent_q, aspect="auto",
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
        if not getattr(self, "_initial_fit_done", False):
            self._initial_fit_done = True
            QTimer.singleShot(0, self._zoom_fit)

    def _on_focus_fft_toggled(self, checked: bool) -> None:
        self._focus_fft_active = bool(checked)
        left = getattr(self, "_left_panel", None)
        if left is not None:
            left.setVisible(not checked)
        tabs = getattr(self, "_tab_widget", None)
        if tabs is not None:
            tabs.setVisible(not checked)
        # Side panel: hidden in focus mode; otherwise follow the tools toggle
        side = getattr(self, "_side_panel", None)
        if side is not None:
            tools_btn = getattr(self, "_show_tools_btn", None)
            tools_on = tools_btn is not None and tools_btn.isChecked()
            side.setVisible(not checked and tools_on)
        btn = getattr(self, "_focus_fft_btn", None)
        if btn is not None:
            btn.setText("Exit Focus" if checked else "Focus FFT")
        QTimer.singleShot(0, self._adapt_zoom_to_canvas)

    def _on_show_tools_toggled(self, checked: bool) -> None:
        side = getattr(self, "_side_panel", None)
        if side is not None:
            focus_active = getattr(self, "_focus_fft_active", False)
            side.setVisible(checked and not focus_active)
        btn = getattr(self, "_show_tools_btn", None)
        if btn is not None:
            btn.setText("Hide tools" if checked else "Show tools")
        QTimer.singleShot(0, self._adapt_zoom_to_canvas)

    def _set_status_text(self, text: str) -> None:
        self._status_lbl.setText(text)
        lbl = getattr(self, "_cursor_readout_lbl", None)
        if lbl is not None:
            lbl.setText(text or "Move over the FFT")

    # ── zoom / pan ─────────────────────────────────────────────────────────────

    def _axes_aspect(self) -> float:
        """Width/height pixel ratio of the FFT axes (accounts for subplot margins)."""
        cw = self._canvas_fft.width()
        ch = self._canvas_fft.height()
        if cw < 1 or ch < 1:
            return 1.0
        # Must match the subplots_adjust call: left=0.07, right=0.99, top=0.95, bottom=0.09
        w = cw * (0.99 - 0.07)
        h = ch * (0.95 - 0.09)
        return w / h if h > 1 else 1.0

    def _adapt_zoom_to_canvas(self) -> None:
        """Re-derive q_x limits from the current canvas aspect, preserving q_y zoom.

        Called after any layout change that resizes the canvas (Focus FFT toggle,
        Show tools toggle) so circles in q-space remain circular on screen.
        """
        if self._use_equal_aspect():
            return
        yb, yt = self._fft_ylim   # yb > yt (inverted y axis)
        y_half = abs(yb - yt) / 2
        yc = (yb + yt) / 2
        xc = (self._fft_xlim[0] + self._fft_xlim[1]) / 2
        x_half = y_half * self._axes_aspect()
        self._fft_xlim = (xc - x_half, xc + x_half)
        self._ax_fft.set_xlim(*self._fft_xlim)
        self._canvas_fft.draw_idle()

    def _use_equal_aspect(self) -> bool:
        cb = getattr(self, "_fft_equal_aspect_cb", None)
        return cb is not None and cb.isChecked()

    def _zoom_fit(self):
        qx_center = (float(self._qx[0]) + float(self._qx[-1])) / 2
        qy_center = (float(self._qy[0]) + float(self._qy[-1])) / 2
        qy_half = (float(self._qy[-1]) - float(self._qy[0])) / 2
        if self._use_equal_aspect():
            qx_half = (float(self._qx[-1]) - float(self._qx[0])) / 2
        else:
            qx_half = qy_half * self._axes_aspect()
        self._fft_xlim = (qx_center - qx_half, qx_center + qx_half)
        self._fft_ylim = (qy_center + qy_half, qy_center - qy_half)
        self._ax_fft.set_xlim(*self._fft_xlim)
        self._ax_fft.set_ylim(*self._fft_ylim)
        self._canvas_fft.draw_idle()

    def _zoom_centre(self):
        # Show centre quarter-range in y; derive x from canvas aspect.
        qy_half = (float(self._qy[-1]) - float(self._qy[0])) * 0.25
        if self._use_equal_aspect():
            qx_half = (float(self._qx[-1]) - float(self._qx[0])) * 0.25
        else:
            qx_half = qy_half * self._axes_aspect()
        self._fft_xlim = (-qx_half, qx_half)
        self._fft_ylim = (qy_half, -qy_half)
        self._ax_fft.set_xlim(*self._fft_xlim)
        self._ax_fft.set_ylim(*self._fft_ylim)
        self._canvas_fft.draw_idle()

    def _zoom_by(self, factor: float, cx: float | None = None, cy: float | None = None):
        xl, xr = self._fft_xlim
        yb, yt = self._fft_ylim   # yb > yt (inverted y axis)
        xc = cx if cx is not None else (xl + xr) / 2
        yc = cy if cy is not None else (yb + yt) / 2
        if self._use_equal_aspect():
            # Symmetric: scale both axes by the same factor.
            min_x, min_y = self._minimum_fft_spans()
            self._fft_xlim = self._interval_with_min_span(
                xc + (xl - xc) * factor, xc + (xr - xc) * factor, min_x,
            )
            self._fft_ylim = self._interval_with_min_span(
                yc + (yb - yc) * factor, yc + (yt - yc) * factor, min_y,
            )
        else:
            # Aspect-aware: scale y by factor, derive x so q/pixel is equal.
            y_half = abs(yb - yt) / 2 * factor
            x_half = y_half * self._axes_aspect()
            _, min_y = self._minimum_fft_spans()
            min_x = min_y * self._axes_aspect()
            y_half = max(y_half, min_y / 2)
            x_half = max(x_half, min_x / 2)
            self._fft_xlim = (xc - x_half, xc + x_half)
            self._fft_ylim = (yc + y_half, yc - y_half)
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
            xl, xr = self._fft_xlim
            yb, yt = self._fft_ylim   # yb > yt (inverted)
            self._set_status_text(
                f"q_x: {xl:.3g} to {xr:.3g} nm⁻¹    "
                f"q_y: {yt:.3g} to {yb:.3g} nm⁻¹"
            )

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
        finite = disp[np.isfinite(disp)]
        if finite.size == 0:
            self._hist_panel.clear(self._theme)
            return
        lo, hi = self._disp_range
        vmin, vmax = self._fft_drs.resolve(disp)
        if vmin is None:
            vmin, vmax = lo, hi
        self._hist_panel.render(
            flat_phys=finite.ravel(),
            lo_phys=float(vmin),
            hi_phys=float(vmax),
            unit="",
            axis_label="Intensity",
            theme=self._theme,
            scale=1.0,
            data_min_phys=lo,
            data_max_phys=hi,
        )
        self._display_slider_ctrl.update()

    def _apply_intensity_from_drs(self) -> None:
        """Fast path: update FFT clim and histogram markers from display-range controller."""
        if self._fft_im is None:
            return
        disp = self._last_fft_disp
        if disp is None:
            return
        vmin, vmax = self._fft_drs.resolve(disp)
        if vmin is None:
            return
        self._fft_im.set_clim(vmin, vmax)
        self._canvas_fft.draw_idle()
        self._hist_panel.update_drag_lines(float(vmin), float(vmax))
        self._display_slider_ctrl.update()

    def _apply_intensity(self) -> None:
        """Convenience alias for _apply_intensity_from_drs."""
        self._apply_intensity_from_drs()

    def _on_fft_hist_range_released(self, lo_phys: float, hi_phys: float) -> None:
        self._fft_drs.set_manual(lo_phys, hi_phys)

    def _reset_intensity(self) -> None:
        self._fft_drs.reset(0.0, 100.0)

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

    def _refresh_structure_combo(self, selected_name: str | None = None) -> None:
        combo = getattr(self, "_structure_combo", None)
        if combo is None:
            return
        self._updating_structure = True
        try:
            combo.clear()
            for structure in self._known_structures:
                combo.addItem(structure.name, structure)
            if selected_name:
                for idx, structure in enumerate(self._known_structures):
                    if structure.name == selected_name:
                        combo.setCurrentIndex(idx)
                        break
        finally:
            self._updating_structure = False

    def _on_structure_selected(self, idx: int) -> None:
        if self._updating_structure:
            return
        combo = getattr(self, "_structure_combo", None)
        if combo is None or idx < 0:
            return
        structure = combo.itemData(idx)
        if not isinstance(structure, KnownStructure):
            return
        self._apply_known_structure_to_fft(structure)

    def _apply_known_structure_to_fft(
        self,
        structure: KnownStructure,
        *,
        refresh: bool = True,
    ) -> None:
        self._active_known_structure = structure
        self._updating_structure = True
        try:
            if structure.symmetry in {"square", "hexagonal"}:
                self._bragg_sym_combo.setCurrentIndex(0 if structure.symmetry == "square" else 1)
                if not self._bragg_enable_cb.isChecked():
                    self._bragg_enable_cb.setChecked(True)
            else:
                self._bragg_enable_cb.setChecked(False)
            unit = "Å" if structure.unit == "Å" else "nm"
            self._bragg_unit_combo.setCurrentText(unit)
            self._bragg_a_spin.setValue(structure_display_value_nm(structure))
        finally:
            self._updating_structure = False

        if hasattr(self, "_fft_ideal_combo"):
            target = {
                "square": "Square",
                "rectangular": "Rectangular",
                "hexagonal": "Hexagonal",
            }.get(structure.symmetry, "Custom")
            self._updating_fft_ideal = True
            try:
                self._fft_ideal_combo.setCurrentText(target)
                self._fft_ideal_a_spin.setValue(max(self._fft_ideal_a_spin.minimum(), structure.a_nm))
                self._fft_ideal_b_spin.setValue(max(self._fft_ideal_b_spin.minimum(), structure.b_nm))
                self._fft_ideal_angle_spin.setValue(
                    min(179.0, max(1.0, structure.angle_deg))
                )
            finally:
                self._updating_fft_ideal = False
        if refresh:
            self._on_bragg_changed()
            self._refresh_fft_correction_ui()

    def _structure_from_fft_controls(self, name: str) -> KnownStructure:
        active = getattr(self, "_active_known_structure", None)
        symmetry = active.symmetry if isinstance(active, KnownStructure) else "hexagonal"
        if symmetry in {"square", "hexagonal"}:
            symmetry = "square" if self._bragg_sym_combo.currentIndex() == 0 else "hexagonal"
        a_nm = self._reference_lattice_a_nm()
        if symmetry == "square":
            return KnownStructure(name, "square", a_nm, a_nm, 90.0, self._bragg_unit_combo.currentText())
        if symmetry == "hexagonal":
            return KnownStructure(name, "hexagonal", a_nm, a_nm, 60.0, self._bragg_unit_combo.currentText())
        b_nm = getattr(self, "_fft_ideal_b_spin", None)
        angle = getattr(self, "_fft_ideal_angle_spin", None)
        return KnownStructure(
            name,
            symmetry,
            a_nm,
            float(b_nm.value()) if b_nm is not None else a_nm,
            float(angle.value()) if angle is not None else 90.0,
            self._bragg_unit_combo.currentText(),
        )

    def _persist_known_structures(self, selected_name: str) -> None:
        save_known_structures(self._known_structures)
        self._refresh_structure_combo(selected_name)

    def _on_save_structure(self) -> None:
        default = getattr(self, "_active_known_structure", self._known_structures[0]).name
        name, ok = QInputDialog.getText(self, "Save known structure", "Structure name:", text=default)
        if not ok or not name.strip():
            return
        structure = self._structure_from_fft_controls(name.strip())
        self._known_structures = upsert_structure(self._known_structures, structure)
        self._persist_known_structures(structure.name)
        self._apply_known_structure_to_fft(structure)

    def _on_update_structure(self) -> None:
        combo = getattr(self, "_structure_combo", None)
        if combo is None or combo.currentIndex() < 0:
            return
        name = combo.currentText().strip()
        if not name:
            return
        structure = self._structure_from_fft_controls(name)
        self._known_structures = upsert_structure(self._known_structures, structure)
        self._persist_known_structures(structure.name)
        self._apply_known_structure_to_fft(structure)

    def _on_delete_structure(self) -> None:
        combo = getattr(self, "_structure_combo", None)
        if combo is None or combo.currentIndex() < 0:
            return
        name = combo.currentText().strip()
        self._known_structures = delete_structure(self._known_structures, name)
        selected = self._known_structures[0]
        self._persist_known_structures(selected.name)
        self._apply_known_structure_to_fft(selected)

    def _on_bragg_symmetry_changed(self, _idx: int) -> None:
        if self._updating_structure:
            return
        self._on_bragg_changed()
        self._sync_fft_ideal_from_symmetry()
        self._refresh_fft_correction_ui()

    def _on_fft_grid_changed(self, _grid=None) -> None:
        self._clear_fft_preview_if_active()
        self._refresh_grid_measure_lbl()
        self._refresh_fft_correction_ui()

    def _refresh_grid_measure_lbl(self) -> None:
        lbl = getattr(self, "_grid_measure_lbl", None)
        if lbl is None:
            return
        panel = getattr(self, "_fft_lattice_panel", None)
        if panel is None:
            lbl.setText("No grid — click Draw Grid to start")
            return
        try:
            from probeflow.analysis.lattice_grid import format_reciprocal_measurements
            grid = panel._overlay.grid()
            if grid is None:
                return
            d = format_reciprocal_measurements(grid, panel._cal)
            lbl.setText(f"{d['g1']}    {d['g2']}    ∠ {d['angle']}")
        except Exception:
            pass

    def _on_grid_extent_changed(self, value: int) -> None:
        overlay = getattr(self, "_fft_lattice_overlay", None)
        if overlay is not None:
            overlay.set_cells(value)
        panel = getattr(self, "_fft_lattice_panel", None)
        spin = getattr(panel, "_cells_spin", None)
        if spin is not None and spin.value() != value:
            blocked = spin.blockSignals(True)
            try:
                spin.setValue(value)
            finally:
                spin.blockSignals(blocked)

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
        structure = getattr(self, "_active_known_structure", None)
        structure_combo = {
            "square": "Square",
            "rectangular": "Rectangular",
            "hexagonal": "Hexagonal",
            "custom": "Custom",
        }.get(structure.symmetry if isinstance(structure, KnownStructure) else "", "")
        if isinstance(structure, KnownStructure) and combo == structure_combo:
            ideal = ideal_lattice_from_structure(structure, measured_angle_deg=angle)
            values = (ideal.a_nm, ideal.b_nm, ideal.angle_deg)
        elif combo == "Custom":
            return
        elif combo == "Square":
            ref_a_nm = self._reference_lattice_a_nm()
            side = ref_a_nm
            values = (side, side, 90.0)
        elif combo == "Rectangular":
            ref_a_nm = self._reference_lattice_a_nm()
            values = (ref_a_nm, b_nm, 90.0)
        elif combo == "Hexagonal":
            ref_a_nm = self._reference_lattice_a_nm()
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
            measured_lbl.setText(
                "Create a reciprocal grid, then drag g1/g2 handles until the grid "
                "tracks the visible Bragg peaks."
            )
            corr_lbl.setText(
                "Step 2: Click 'Create/Edit reciprocal grid' below and drag the "
                "g₁/g₂ handles onto two Bragg peaks in the FFT."
            )
            if status_lbl is not None:
                status_lbl.setText("No reciprocal grid yet")
            return

        a_vec, b_vec = vectors
        a_len = self._vec_len_nm(a_vec)
        b_len = self._vec_len_nm(b_vec)
        angle = self._vec_angle_deg(a_vec, b_vec)
        self._sync_fft_ideal_values(a_len, b_len, angle)

        ideal = IdealLattice(
            a_nm=self._fft_ideal_a_spin.value(),
            b_nm=self._fft_ideal_b_spin.value(),
            angle_deg=self._fft_ideal_angle_spin.value(),
        )
        da_pct = 100.0 * (a_len / ideal.a_nm - 1.0) if ideal.a_nm > 0 else float("nan")
        db_pct = 100.0 * (b_len / ideal.b_nm - 1.0) if ideal.b_nm > 0 else float("nan")
        measured_lbl.setText(
            f"Measured |a|={a_len:.4g} nm |b|={b_len:.4g} nm angle={angle:.2f}°\n"
            f"Target |a|={ideal.a_nm:.4g} nm |b|={ideal.b_nm:.4g} nm "
            f"angle={ideal.angle_deg:.2f}°   "
            f"da {da_pct:+.2f}% db {db_pct:+.2f}% dAngle {angle - ideal.angle_deg:+.2f}°"
        )
        result = compute_correction(MeasuredLattice(a_nm=a_vec, b_nm=b_vec), ideal)
        if isinstance(result, str):
            corr_lbl.setText(f"Cannot compute correction:\n{result}")
            if status_lbl is not None:
                status_lbl.setText("Correction unavailable")
            return

        self._fft_correction = result
        correction_lines = correction_main_lines(
            result,
            preserve_orientation=self._fft_preserve_orientation_cb.isChecked(),
        )
        corr_lbl.setText(
            "\n".join(correction_lines)
            + "\n→ Click 'Preview corrected image' to verify."
        )
        if status_lbl is not None:
            status_lbl.setText("Correction ready")
        if self._get_image_fn is not None:
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
        if self._fft_preview_active:
            self._hide_fft_preview()

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
        self._show_fft_preview(corrected)
        self._fft_preview_active = True
        self._fft_clear_preview_btn.setEnabled(True)
        self._fft_correction_status_lbl.setText("Preview shown")

    def _on_fft_clear_preview(self) -> None:
        self._hide_fft_preview()
        self._refresh_fft_correction_ui()

    def _on_fft_apply_correction(self) -> None:
        if self._apply_correction_fn is None or self._fft_correction is None:
            return
        px = self._fft_pixel_sizes_m()
        if px is None or self._fft_correction_matrix_px() is None:
            return
        if self._fft_preview_active:
            self._hide_fft_preview()
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
        structure = getattr(self, "_active_known_structure", None)
        if isinstance(structure, KnownStructure):
            op_params["known_structure"] = structure.as_dict()
        self._apply_correction_fn("affine_lattice_correction", op_params)
        # Recompute FFT from the now-corrected image so the display reflects the
        # undistorted real-space data.  _apply_correction_fn calls
        # _refresh_processing_display() synchronously, so _get_image_fn() already
        # returns the corrected array by the time we reach here.
        if self._get_image_fn is not None:
            updated = self._get_image_fn()
            if updated is not None and np.asarray(updated).ndim == 2:
                # Preserve original pixel sizes across the canvas-size change.
                # Canvas expansion increases row/col count without changing the
                # physical size of each pixel.  If we kept _scan_range_m the same
                # while Nx/Ny grew, _recompute_fft would divide the same physical
                # range over more pixels, computing a smaller dx/dy and a
                # spuriously higher Nyquist frequency in the expanded direction.
                orig_ny, orig_nx = self._arr.shape
                px_x_nm = self._scan_range_m[0] * 1e9 / orig_nx if orig_nx > 0 else 1.0
                px_y_nm = self._scan_range_m[1] * 1e9 / orig_ny if orig_ny > 0 else 1.0
                self._arr = np.asarray(updated, dtype=np.float64)
                new_ny, new_nx = self._arr.shape
                self._scan_range_m = (px_x_nm * new_nx * 1e-9,
                                      px_y_nm * new_ny * 1e-9)
                self._recompute_fft()
                self._redraw()
                self._update_info_panel()
        # Clear the stale grid overlay — it was fitted on the pre-correction FFT
        # and no longer aligns with the corrected diffraction pattern.
        self._on_clear_fft_lattice()
        self._fft_correction_lbl.setText(
            "Correction applied. FFT recomputed from corrected image.\n"
            "Bragg peaks should now lie on the inner shell ring."
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
        """Reset the embedded Grid tab panel area to its placeholder state."""
        lay = getattr(self, "_grid_tab_lay", None)
        if lay is None:
            return
        while lay.count():
            item = lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        lay.addWidget(self._grid_placeholder_lbl)
        draw_btn = getattr(self, "_grid_draw_btn", None)
        if draw_btn is not None:
            lay.addWidget(draw_btn)
        lbl = getattr(self, "_grid_measure_lbl", None)
        if lbl is not None:
            lbl.setText("No grid — click Draw Grid to start")

    def _set_grid_tab_panel(self, panel: QWidget) -> None:
        """Install the reciprocal-grid controls into the Grid tab panel area."""
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
        self._on_grid_extent_changed(self._grid_extent_spin.value())
        if select_advanced:
            self._tab_widget.setCurrentIndex(self._grid_tab_index)
        if getattr(self, "_clear_grid_btn", None) is not None:
            self._clear_grid_btn.setEnabled(True)
        self._refresh_grid_measure_lbl()
        self._refresh_fft_correction_ui()

    # ── Bragg ring overlay ─────────────────────────────────────────────────────

    def _on_bragg_changed(self):
        """Fast-path update: remove old ring artists, draw new ones, redraw."""
        self._clear_fft_preview_if_active()
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
            self._bragg_radius_lbl.setText("Shells hidden")
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
            self._bragg_radius_lbl.setText("Shells: invalid input")
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
        label_bits: list[str] = []
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
            label_bits.append(f"{idx}:{shell.label} q={q_shell:.3g}")
        if not shells:
            label_bits.append("No shells within FFT q-range")

        suffix = " nm^-1" if shells else ""
        self._bragg_radius_lbl.setText("Shells: " + "; ".join(label_bits) + suffix)
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
