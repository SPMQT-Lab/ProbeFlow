from __future__ import annotations


import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from probeflow.gui.typography import mono_font, ui_font
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMenuBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from probeflow.analysis.lattice_distortion import LatticeCorrection
from probeflow.gui._tooltips import tip as _tip
from probeflow.gui.lattice_correction_ui import load_known_structures
from probeflow.gui.dialogs.fft_viewer_lattice_mixin import FFTViewerLatticeMixin
from probeflow.gui.dialogs.fft_viewer_mains_mixin import FFTViewerMainsMixin
from probeflow.gui.dialogs.fft_viewer_reconstruct_mixin import FFTViewerReconstructMixin
from probeflow.gui.no_wheel import install_no_wheel_spinboxes
from probeflow.gui.viewer.display_range import DisplayRangeController
from probeflow.gui.viewer.display_sliders import DisplaySliderController
from probeflow.gui.viewer.histogram import HistogramPanel


def crop_to_bounds(
    arr: np.ndarray,
    bounds: tuple[int, int, int, int],
    scan_range_m: tuple[float, float],
) -> tuple[np.ndarray, tuple[float, float]]:
    """Crop *arr* to inclusive pixel *bounds* and scale the scan range to match.

    Parameters
    ----------
    arr
        2-D image, shape ``(Ny, Nx)`` = (rows, cols).
    bounds
        ``(row_min, row_max, col_min, col_max)`` inclusive, as returned by
        :meth:`probeflow.core.roi.ROI.bounds` / ``active_area_roi_bounds``.
    scan_range_m
        ``(width_x_m, height_y_m)`` physical extent of the full image.

    Returns
    -------
    (cropped_arr, cropped_scan_range_m)
        The cropped sub-array and its physical extent. The scan range is scaled
        by the crop's pixel-count ratio so the **pixel size is preserved**
        (``range / count`` is unchanged on each axis). This is what keeps the
        reciprocal-space q-grid correct when the FFT is recomputed on the crop:
        q-resolution scales with the smaller extent, Nyquist is unchanged.

    The bounds are clipped to the array; an empty or degenerate crop falls back
    to the full array and range so callers never produce a zero-size FFT.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError("crop_to_bounds requires a 2-D array")
    ny, nx = a.shape
    r0, r1, c0, c1 = (int(v) for v in bounds)
    # Clip inclusive bounds to valid index range.
    r0 = max(0, min(r0, ny - 1))
    r1 = max(0, min(r1, ny - 1))
    c0 = max(0, min(c0, nx - 1))
    c1 = max(0, min(c1, nx - 1))
    if r1 < r0 or c1 < c0:
        return a, (float(scan_range_m[0]), float(scan_range_m[1]))

    cropped = a[r0:r1 + 1, c0:c1 + 1]
    crop_ny, crop_nx = cropped.shape
    width_m, height_m = float(scan_range_m[0]), float(scan_range_m[1])
    # Preserve pixel size: new_range = full_range * (crop_count / full_count).
    new_width_m = width_m * (crop_nx / nx) if nx > 0 else width_m
    new_height_m = height_m * (crop_ny / ny) if ny > 0 else height_m
    return cropped, (new_width_m, new_height_m)


class FFTViewerDialog(
    FFTViewerLatticeMixin,
    FFTViewerMainsMixin,
    FFTViewerReconstructMixin,
    QDialog,
):
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
        roi_bounds_px: tuple[int, int, int, int] | None = None,
        roi_id: str | None = None,
        roi_name: str | None = None,
        scan_speed_m_per_s: float | None = None,
        new_image_fn=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.setWindowTitle("FFT Viewer")
        self.resize(1180, 820)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        # Full image is the canonical source; _arr/_scan_range_m are the *working*
        # source that every downstream consumer reads. They equal the full image
        # unless the user selects the ROI source, in which case _resolve_source_array
        # crops to the ROI bbox (pixel size preserved → q-grid stays correct).
        self._full_arr = arr.astype(np.float64, copy=True)
        self._full_scan_range_m = scan_range_m
        self._roi_bounds_px = roi_bounds_px
        self._roi_id = roi_id
        self._roi_name = roi_name
        self._fft_source = "whole_image"  # default; ROI is opt-in via the selector
        self._arr, self._scan_range_m = self._resolve_source_array()
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
        self._fft_phase: np.ndarray | None = None   # angle(F), for the phase view
        self._qx: np.ndarray | None = None
        self._qy: np.ndarray | None = None
        self._fft_xlim: tuple = (0.0, 1.0)
        self._fft_ylim: tuple = (1.0, 0.0)
        self._scale_mode = "log"
        self._fft_display_mode = "magnitude"         # "magnitude" | "phase"
        self._phase_cmap = "twilight"                # cyclic map for the phase view
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
        # Mains-pickup diagnostic/removal (⚡ Mains tab).
        self._scan_speed_m_per_s = scan_speed_m_per_s
        self._mains_artists: list = []
        self._mains_preview_active = False
        self._mains_fast_axis = "x"
        # User-placed streak pairs (positive |q| in nm⁻¹) and drag state.
        self._mains_custom_q: list = []
        self._mains_drag_idx = None
        self._mains_tab_index = None
        # Inverse FFT / Fourier reconstruction (Reconstruct tab).
        self._fft_selection_overlay = None
        self._reconstruct_tab_index = -1
        self._reconstruct_preview_active = False
        self._new_image_fn = new_image_fn

        self._fft_drs = DisplayRangeController(clip_low=0.0, clip_high=100.0, parent=self)

        self._build()

        self._display_slider_ctrl = DisplaySliderController(
            self._fft_drs,
            self._hist_panel,
            lambda: self._last_fft_disp,
            lambda: (1.0, "", "Intensity"),
        )
        self._fft_drs.rangeChanged.connect(self._apply_intensity_from_drs)
        self._hist_panel.minReleased.connect(self._on_fft_hist_min_released)
        self._hist_panel.maxReleased.connect(self._on_fft_hist_max_released)
        self._hist_panel.brightnessReleased.connect(self._on_fft_hist_brightness_released)
        self._hist_panel.contrastReleased.connect(self._on_fft_hist_contrast_released)
        self._hist_panel.rangeReleased.connect(self._on_fft_hist_range_released)
        self._hist_panel.resetRequested.connect(self._reset_intensity)
        self._hist_panel.autoClipRequested.connect(self._reset_intensity)

        self._recompute_fft()
        self._update_info_panel()
        self._redraw()
        self._refresh_fft_correction_ui()

    # ── layout helpers (shared with subclasses) ────────────────────────────────

    def _build_toolbar_row(self) -> QHBoxLayout:
        """Essentials-only toolbar.

        Appearance/preprocessing controls (component, colour map, window, DC
        handling) and Export live in the menu bar; the combos below are kept as
        hidden state-holders that the menu actions drive, so all existing
        handlers and state-restore paths keep working unchanged.
        """
        tb = QHBoxLayout()
        tb.setSpacing(6)

        def _lbl(text):
            lb = QLabel(text)
            lb.setFont(ui_font(9))
            return lb

        def _combo(items, min_width=80, *, hidden=False):
            c = QComboBox(self)
            c.addItems(items)
            c.setFont(ui_font(9))
            c.setFixedHeight(24)
            c.setMinimumWidth(min_width)
            if hidden:
                c.hide()
            return c

        # ── hidden state-holders (driven by the menu bar) ────────────────────
        self._fft_view_combo = _combo(["Magnitude", "Phase"], 96, hidden=True)
        self._fft_view_combo.setToolTip(_tip(
            "Show the FFT magnitude (default) or phase. Phase is in radians "
            "(−π…π) on a cyclic colour map; most analysis uses magnitude."))
        self._fft_view_combo.currentIndexChanged.connect(self._on_fft_view_changed)

        # Scale is essentially set-and-forget (log by default) → View menu.
        self._scale_combo = _combo(["Log", "Linear"], 80, hidden=True)
        self._scale_combo.currentIndexChanged.connect(self._on_scale_changed)

        self._window_combo = _combo(["Hann", "None", "Tukey"], 82, hidden=True)
        self._window_combo.currentIndexChanged.connect(self._on_window_changed)

        self._dc_combo = _combo(["Zero DC", "Keep DC", "Mask DC"], 95, hidden=True)
        self._dc_combo.setCurrentIndex(1)
        self._dc_combo.currentIndexChanged.connect(self._on_dc_changed)

        # Equal-aspect display toggle (driven by the View menu; was the "Expert" tab).
        self._fft_equal_aspect_cb = QCheckBox("Equal q_x/q_y aspect")
        self._fft_equal_aspect_cb.setChecked(False)
        self._fft_equal_aspect_cb.hide()
        self._fft_equal_aspect_cb.toggled.connect(self._on_equal_aspect_toggled)

        # ── essentials kept on the toolbar ───────────────────────────────────
        tb.addWidget(_lbl("Colour:"))
        self._cmap_combo = _combo(
            ["Gray", "Gray (inv.)", "Inferno", "Hot", "Viridis", "Plasma", "Turbo"], 96
        )
        self._cmap_combo.setToolTip("Colour map for the FFT intensity display.")
        self._cmap_combo.currentIndexChanged.connect(self._on_cmap_changed)
        tb.addWidget(self._cmap_combo)

        tb.addWidget(_lbl("Source:"))
        self._fft_source_combo = _combo(["Whole image", "Active ROI"], 104)
        self._fft_source_combo.setToolTip(
            "Region the FFT is computed from. 'Whole image' uses the full scan; "
            "'Active ROI' uses the active area ROI's bounding box (q-resolution "
            "then reflects the smaller region)."
        )
        if not self._has_roi_source():
            # No ROI was passed — disable the ROI entry so the control is honest.
            self._fft_source_combo.model().item(1).setEnabled(False)
            self._fft_source_combo.setToolTip(
                "Activate an area ROI before opening the FFT to enable ROI-sourced FFTs."
            )
        self._fft_source_combo.setCurrentIndex(
            1 if self._fft_source == "active_roi" else 0
        )
        self._fft_source_combo.currentIndexChanged.connect(self._on_fft_source_changed)
        tb.addWidget(self._fft_source_combo)

        tb.addStretch(1)

        show_tools_btn = QPushButton("Show tools")
        show_tools_btn.setFont(ui_font(9))
        show_tools_btn.setFixedHeight(24)
        show_tools_btn.setMinimumWidth(86)
        show_tools_btn.setCheckable(True)
        show_tools_btn.setToolTip("Show or hide the cursor details side panel")
        show_tools_btn.toggled.connect(self._on_show_tools_toggled)
        tb.addWidget(show_tools_btn)
        self._show_tools_btn = show_tools_btn

        focus_btn = QPushButton("Focus FFT")
        focus_btn.setFont(ui_font(9))
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
            ("Fit",   "Zoom to fit full FFT extent", self._zoom_fit),
            ("  +  ", "Zoom in",                     lambda: self._zoom_by(0.5)),
            ("  −  ", "Zoom out",                    lambda: self._zoom_by(2.0)),
        ]:
            btn = QPushButton(label)
            btn.setFont(ui_font(9))
            btn.setFixedHeight(24)
            btn.setMinimumWidth(44)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            tb.addWidget(btn)

        return tb

    def _build_menu_bar(self) -> QMenuBar:
        """Menu bar hosting the appearance/preprocessing controls demoted from
        the toolbar (component, colour map, window, DC handling), zoom, panel
        toggles, the Grid jump, and Export."""
        mb = QMenuBar(self)
        mb.setFont(ui_font(9))

        def _radio_menu(parent_menu, title, items, combo, *, tip=None):
            """Add a submenu of mutually-exclusive actions that drive `combo`."""
            menu = parent_menu.addMenu(title)
            if tip:
                menu.setToolTip(tip)
            group = QActionGroup(menu)
            group.setExclusive(True)
            actions = []
            for i, label in enumerate(items):
                act = QAction(label, menu, checkable=True)
                act.setChecked(combo.currentIndex() == i)
                act.triggered.connect(lambda _c=False, idx=i: combo.setCurrentIndex(idx))
                group.addAction(act)
                menu.addAction(act)
                actions.append(act)
            # Keep the menu's check-state in sync if the combo changes elsewhere.
            combo.currentIndexChanged.connect(
                lambda idx, acts=actions: (
                    acts[idx].setChecked(True) if 0 <= idx < len(acts) else None
                )
            )
            return menu

        # ── View menu ────────────────────────────────────────────────────────
        view_menu = mb.addMenu("&View")
        _radio_menu(
            view_menu, "Component", ["Magnitude", "Phase"], self._fft_view_combo,
            tip="Magnitude (default) or phase. Phase is in radians on a cyclic map.",
        )
        self._scale_menu = _radio_menu(
            view_menu, "Scale", ["Log", "Linear"], self._scale_combo,
            tip="Log makes weak peaks visible; Linear shows raw amplitude ratios.",
        )
        view_menu.addSeparator()
        zoom_menu = view_menu.addMenu("Zoom")
        for label, slot in [
            ("Fit to extent", self._zoom_fit),
            ("Centre (quarter range)", self._zoom_centre),
            ("Zoom in", lambda: self._zoom_by(0.5)),
            ("Zoom out", lambda: self._zoom_by(2.0)),
        ]:
            act = QAction(label, zoom_menu)
            act.triggered.connect(lambda _c=False, s=slot: s())
            zoom_menu.addAction(act)
        view_menu.addSeparator()
        self._show_tools_act = QAction("Show cursor tools", view_menu, checkable=True)
        self._show_tools_act.triggered.connect(
            lambda checked: self._show_tools_btn.setChecked(checked)
        )
        view_menu.addAction(self._show_tools_act)
        self._focus_fft_act = QAction("Focus FFT", view_menu, checkable=True)
        self._focus_fft_act.triggered.connect(
            lambda checked: self._focus_fft_btn.setChecked(checked)
        )
        view_menu.addAction(self._focus_fft_act)
        self._equal_aspect_act = QAction("Equal q_x/q_y aspect", view_menu, checkable=True)
        self._equal_aspect_act.setChecked(self._fft_equal_aspect_cb.isChecked())
        self._equal_aspect_act.setToolTip(
            "Zoom q_x and q_y symmetrically (equal q per pixel in both directions; "
            "may leave blank space on a wide canvas). Off uses the full canvas width."
        )
        self._equal_aspect_act.triggered.connect(
            lambda checked: self._fft_equal_aspect_cb.setChecked(checked)
        )
        self._fft_equal_aspect_cb.toggled.connect(
            lambda checked: (
                self._equal_aspect_act.setChecked(checked)
                if self._equal_aspect_act.isChecked() != checked else None
            )
        )
        view_menu.addAction(self._equal_aspect_act)

        # ── FFT (compute) menu ────────────────────────────────────────────────
        fft_menu = mb.addMenu("&FFT")
        _radio_menu(
            fft_menu, "Window", ["Hann", "None", "Tukey"], self._window_combo,
            tip="Apodisation window applied before the FFT to reduce edge ringing.",
        )
        _radio_menu(
            fft_menu, "DC component", ["Zero DC", "Keep DC", "Mask DC"], self._dc_combo,
            tip="How the zero-frequency (central) component is treated.",
        )
        fft_menu.addSeparator()
        grid_act = QAction("Go to Grid tab", fft_menu)
        grid_act.setToolTip(
            "Switch to the Grid tab (creates a lattice overlay if none exists)"
        )
        grid_act.triggered.connect(
            lambda _c=False: self._on_open_fft_lattice(select_advanced=True)
        )
        fft_menu.addAction(grid_act)

        # ── Export menu ───────────────────────────────────────────────────────
        export_menu = mb.addMenu("&Export")
        exp_act = QAction("Export PNG…", export_menu)
        exp_act.setToolTip("Save the current FFT view as a PNG file.")
        exp_act.triggered.connect(lambda _c=False: self._on_export())
        export_menu.addAction(exp_act)

        return mb

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
        cursor_title.setFont(ui_font(9, weight=QFont.Bold))
        self._cursor_readout_lbl = QLabel("Move over the FFT")
        self._cursor_readout_lbl.setFont(mono_font(8))
        self._cursor_readout_lbl.setWordWrap(True)
        side_lay.addWidget(cursor_title)
        side_lay.addWidget(self._cursor_readout_lbl)
        side_lay.addStretch(1)
        fft_top_lay.addWidget(side_panel)
        self._side_panel = side_panel
        side_panel.hide()  # hidden by default; "Show tools" in toolbar reveals it

        self._tab_widget = QTabWidget()
        self._tab_widget.setMinimumHeight(300)
        self._tab_widget.setFont(ui_font(9))

        # ── Tab 0: Inspect ───────────────────────────────────────────────────
        inspect_tab = QWidget()
        inspect_lay = QHBoxLayout(inspect_tab)
        inspect_lay.setContentsMargins(6, 6, 6, 6)
        inspect_lay.setSpacing(8)

        intensity_grp = QGroupBox("Intensity")
        intensity_grp.setFont(ui_font(9))
        int_lay = QVBoxLayout(intensity_grp)
        int_lay.setContentsMargins(6, 6, 6, 4)
        self._hist_panel = HistogramPanel(parent=intensity_grp)
        int_lay.addWidget(self._hist_panel, 1)
        inspect_lay.addWidget(intensity_grp, 1)

        radial_grp = QGroupBox("Radial profile")
        radial_grp.setFont(ui_font(9))
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
        self._grid_measure_lbl.setFont(mono_font(9))
        self._grid_measure_lbl.setWordWrap(True)
        self._grid_measure_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid_outer_lay.addWidget(self._grid_measure_lbl)

        # Grid extent (mesh size) — created here, placed inside the Known
        # structure group below so it isn't a floating row.
        self._grid_extent_spin = QSpinBox()
        self._grid_extent_spin.setRange(1, 200)
        self._grid_extent_spin.setValue(12)
        self._grid_extent_spin.setFont(ui_font(9))
        self._grid_extent_spin.setFixedHeight(24)
        self._grid_extent_spin.setMaximumWidth(72)
        self._grid_extent_spin.setToolTip("How many reciprocal-lattice repeats to draw in each direction.")
        self._grid_extent_spin.valueChanged.connect(self._on_grid_extent_changed)

        # Draw + Clear Grid share a single row.
        self._grid_draw_btn = QPushButton("Draw Grid")
        self._grid_draw_btn.setFont(ui_font(9))
        self._grid_draw_btn.setFixedHeight(26)
        self._grid_draw_btn.setToolTip(
            "Create a hexagonal reciprocal-lattice overlay on the FFT. "
            "Drag the handles to align g₁/g₂ with Bragg peaks."
        )
        self._grid_draw_btn.clicked.connect(lambda: self._on_open_fft_lattice())
        self._clear_grid_btn = QPushButton("Clear Grid")
        self._clear_grid_btn.setFont(ui_font(9))
        self._clear_grid_btn.setFixedHeight(26)
        self._clear_grid_btn.setToolTip("Remove the reciprocal-space lattice overlay")
        self._clear_grid_btn.setEnabled(False)
        self._clear_grid_btn.clicked.connect(self._on_clear_fft_lattice)
        grid_btn_row = QHBoxLayout()
        grid_btn_row.addWidget(self._grid_draw_btn)
        grid_btn_row.addWidget(self._clear_grid_btn)
        grid_outer_lay.addLayout(grid_btn_row)

        # Panel container: initially shows the placeholder hint; replaced by
        # FFTLatticePanel when a grid is created.
        grid_panel_container = QWidget()
        self._grid_tab_lay = QVBoxLayout(grid_panel_container)
        self._grid_tab_lay.setContentsMargins(0, 0, 0, 0)
        self._grid_placeholder_lbl = QLabel(
            "Click Draw Grid to overlay a reciprocal lattice.\n"
            "Drag the g₁/g₂ handles to align with Bragg peaks."
        )
        self._grid_placeholder_lbl.setFont(ui_font(9))
        self._grid_placeholder_lbl.setAlignment(Qt.AlignCenter)
        self._grid_placeholder_lbl.setWordWrap(True)
        self._grid_tab_lay.addWidget(self._grid_placeholder_lbl)
        grid_outer_lay.addWidget(grid_panel_container)

        # Known structure section (moved here from Correction tab)
        ref_grp = QGroupBox("Known structure")
        ref_grp.setFont(ui_font(9))
        ref_grp.setMinimumHeight(120)
        ref_grid = QGridLayout(ref_grp)
        ref_grid.setContentsMargins(8, 7, 8, 4)
        ref_grid.setHorizontalSpacing(8)
        ref_grid.setVerticalSpacing(2)

        structure_row = QHBoxLayout()
        self._structure_combo = QComboBox()
        self._structure_combo.setFont(ui_font(9))
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
            btn.setFont(ui_font(8))
            btn.setFixedHeight(23)
            btn.setMaximumWidth(72)
            structure_row.addWidget(btn)
        structure_row.addStretch(1)
        self._structure_save_btn.clicked.connect(self._on_save_structure)
        self._structure_update_btn.clicked.connect(self._on_update_structure)
        self._structure_delete_btn.clicked.connect(self._on_delete_structure)

        self._bragg_enable_cb = QCheckBox("Show shell rings")
        self._bragg_enable_cb.setFont(ui_font(9))
        self._bragg_enable_cb.setChecked(True)
        self._bragg_enable_cb.setToolTip("Overlay the expected Bragg-shell radii from the known lattice.")
        self._bragg_enable_cb.toggled.connect(self._on_bragg_changed)

        self._bragg_sym_combo = QComboBox()
        self._bragg_sym_combo.addItems(["Square", "Hexagonal"])
        self._bragg_sym_combo.setFont(ui_font(9))
        self._bragg_sym_combo.setFixedHeight(24)
        self._bragg_sym_combo.setMaximumWidth(120)
        self._bragg_sym_combo.setToolTip("Surface symmetry used for the predicted reciprocal-lattice shells.")
        self._bragg_sym_combo.currentIndexChanged.connect(self._on_bragg_symmetry_changed)
        self._bragg_a_spin = QDoubleSpinBox()
        self._bragg_a_spin.setRange(0.001, 999.0)
        self._bragg_a_spin.setValue(2.46)
        self._bragg_a_spin.setDecimals(3)
        self._bragg_a_spin.setFont(ui_font(9))
        self._bragg_a_spin.setFixedHeight(24)
        self._bragg_a_spin.setToolTip("Approximate real-space lattice spacing for predicted Bragg shells.")
        self._bragg_a_spin.valueChanged.connect(self._on_bragg_changed)
        self._bragg_a_spin.setMaximumWidth(84)
        self._bragg_unit_combo = QComboBox()
        self._bragg_unit_combo.addItems(["Å", "nm"])
        self._bragg_unit_combo.setFont(ui_font(9))
        self._bragg_unit_combo.setFixedHeight(24)
        self._bragg_unit_combo.setMaximumWidth(58)
        self._bragg_unit_combo.setToolTip("Unit for the reference lattice spacing.")
        self._bragg_unit_combo.currentIndexChanged.connect(self._on_bragg_changed)
        self._bragg_max_shells_spin = QSpinBox()
        self._bragg_max_shells_spin.setRange(1, 12)
        self._bragg_max_shells_spin.setValue(5)
        self._bragg_max_shells_spin.setFont(ui_font(9))
        self._bragg_max_shells_spin.setFixedHeight(24)
        self._bragg_max_shells_spin.setToolTip("Maximum number of visible predicted shell families.")
        self._bragg_max_shells_spin.valueChanged.connect(self._on_bragg_changed)
        self._bragg_max_shells_spin.setMaximumWidth(64)

        a_value_row = QHBoxLayout()
        a_value_row.setSpacing(4)
        a_value_row.addWidget(self._bragg_a_spin)
        a_value_row.addWidget(self._bragg_unit_combo)
        a_value_row.addStretch(1)
        ref_grid.addWidget(QLabel("Structure:"), 0, 0)
        ref_grid.addWidget(self._structure_combo, 0, 1, 1, 2)
        ref_grid.addLayout(structure_row, 0, 3)
        ref_grid.addWidget(QLabel("Symmetry:"), 1, 0)
        ref_grid.addWidget(self._bragg_sym_combo, 1, 1)
        ref_grid.addWidget(QLabel("Lattice a:"), 1, 2)
        ref_grid.addLayout(a_value_row, 1, 3)
        ref_grid.addWidget(QLabel("Shells:"), 2, 0)
        ref_grid.addWidget(self._bragg_max_shells_spin, 2, 1)
        ref_grid.addWidget(QLabel("Grid orders ±:"), 2, 2)
        ref_grid.addWidget(self._grid_extent_spin, 2, 3)
        ref_grid.addWidget(self._bragg_enable_cb, 3, 0, 1, 4)
        ref_grid.setColumnStretch(1, 1)
        ref_grid.setColumnStretch(3, 1)
        self._bragg_radius_lbl = QLabel("Shells: —")
        self._bragg_radius_lbl.setFont(mono_font(8))
        self._bragg_radius_lbl.setWordWrap(True)
        self._bragg_radius_lbl.setMaximumHeight(34)
        self._bragg_radius_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ref_grid.addWidget(self._bragg_radius_lbl, 4, 0, 1, 4)
        self._refresh_structure_combo(self._active_known_structure.name)
        self._apply_known_structure_to_fft(self._active_known_structure, refresh=False)
        grid_outer_lay.addWidget(ref_grp)

        # Compare section
        compare_grp = QGroupBox("Compare with known structure")
        compare_grp.setFont(ui_font(9))
        compare_lay = QVBoxLayout(compare_grp)
        compare_lay.setContentsMargins(8, 7, 8, 4)
        self._fft_measured_lbl = QLabel(
            "Draw a grid and select a known structure to see the comparison."
        )
        self._fft_measured_lbl.setFont(mono_font(8))
        self._fft_measured_lbl.setWordWrap(True)
        self._fft_measured_lbl.setMinimumHeight(32)
        self._fft_measured_lbl.setMaximumHeight(54)
        self._fft_measured_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        compare_lay.addWidget(self._fft_measured_lbl)
        grid_outer_lay.addWidget(compare_grp)

        # (Clear Grid now lives in the Draw/Clear row at the top of the tab.)
        grid_outer_lay.addStretch(1)
        grid_scroll.setWidget(grid_inner)
        self._grid_tab_index = self._tab_widget.addTab(grid_scroll, "Grid")

        # ── Tab 2: Correction ────────────────────────────────────────────────
        corr_tab = QWidget()
        corr_lay = QVBoxLayout(corr_tab)
        corr_lay.setSpacing(6)
        corr_lay.setContentsMargins(8, 8, 8, 6)

        # Ideal-lattice target controls for the lattice-correction workflow.
        self._fft_ideal_combo = QComboBox()
        self._fft_ideal_combo.addItems(["Match measured", "Square", "Rectangular", "Hexagonal", "Custom"])
        self._fft_ideal_combo.setCurrentText("Square")
        self._fft_ideal_combo.setFont(ui_font(9))
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
        self._fft_preserve_orientation_cb.setFont(ui_font(9))
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
        self._fft_correction_lbl.setFont(mono_font(8))
        self._fft_correction_lbl.setWordWrap(True)
        self._fft_correction_lbl.setMinimumHeight(34)
        self._fft_correction_lbl.setMaximumHeight(60)
        self._fft_correction_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        corr_lay.addWidget(self._fft_correction_lbl)

        self._fft_correction_status_lbl = QLabel("No reciprocal grid yet")
        self._fft_correction_status_lbl.setFont(ui_font(8))
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
        self._fft_preview_btn.setFont(ui_font(9))
        self._fft_preview_btn.setFixedHeight(24)
        self._fft_preview_btn.setEnabled(False)
        self._fft_preview_btn.setToolTip("Show the affine-corrected real-space image in the left preview rail.")
        self._fft_preview_btn.clicked.connect(self._on_fft_preview_correction)
        self._fft_clear_preview_btn = QPushButton("Clear preview")
        self._fft_clear_preview_btn.setFont(ui_font(9))
        self._fft_clear_preview_btn.setFixedHeight(24)
        self._fft_clear_preview_btn.setEnabled(False)
        self._fft_clear_preview_btn.clicked.connect(self._on_fft_clear_preview)
        self._fft_apply_btn = QPushButton("Apply correction")
        self._fft_apply_btn.setObjectName("accentBtn")
        self._fft_apply_btn.setFont(ui_font(9))
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

        # The former "Expert" tab only duplicated the Grid tool (ideal target +
        # interpolation/fill) plus one display option, so it was removed. The
        # ideal-target / interpolation / fill controls stay alive as hidden
        # state-holders — the ideal target follows the measured/known lattice and
        # interp/fill keep sensible defaults (bilinear / NaN) — and the
        # equal-aspect display option moved to the View menu.
        self._fft_param_holder = QWidget(self)
        _param_lay = QVBoxLayout(self._fft_param_holder)
        _param_lay.setContentsMargins(0, 0, 0, 0)
        for _w in (
            self._fft_ideal_combo,
            self._fft_ideal_a_spin,
            self._fft_ideal_b_spin,
            self._fft_ideal_angle_spin,
            self._fft_interp_combo,
            self._fft_fill_combo,
        ):
            _param_lay.addWidget(_w)
        self._fft_param_holder.hide()

        # Append the Mains tab last so existing tab indices (e.g. _grid_tab_index)
        # are unaffected.
        self._mains_tab_index = self._tab_widget.addTab(
            self._build_mains_tab(), "⚡ Mains")
        self._reconstruct_tab_index = self._tab_widget.addTab(
            self._build_reconstruct_tab(), "Inverse FFT")

        self._fft_splitter = QSplitter(Qt.Vertical)
        self._fft_splitter.addWidget(fft_top)
        self._fft_splitter.addWidget(self._tab_widget)
        self._fft_splitter.setStretchFactor(0, 1)
        self._fft_splitter.setStretchFactor(1, 0)
        self._fft_splitter.setSizes([420, 360])
        right_col.addWidget(self._fft_splitter, 1)
        install_no_wheel_spinboxes(self._tab_widget)
        self._compact_fft_tab_fields()

        return right_col

    def _compact_fft_tab_fields(self) -> None:
        """Stop spin boxes / combos in the tabs from stretching across the row.

        Several tab layouts add fields with a stretch factor, so a short value
        sits next to its label on the far left while the spin arrows / dropdown
        are pushed to the far right edge.  Capping each field at its content
        width (``QSizePolicy.Maximum``) keeps the arrows next to the value; the
        spare space goes to the right of the field.
        """
        from PySide6.QtWidgets import QAbstractSpinBox, QComboBox, QSizePolicy
        for cls in (QAbstractSpinBox, QComboBox):
            for w in self._tab_widget.findChildren(cls):
                w.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

    def _connect_canvas_events(self) -> None:
        """Wire mpl events for the FFT canvas and radial panel."""
        self._canvas_fft.mpl_connect("scroll_event",         self._on_scroll)
        self._canvas_fft.mpl_connect("button_press_event",   self._on_press)
        self._canvas_fft.mpl_connect("button_release_event", self._on_release)
        self._canvas_fft.mpl_connect("motion_notify_event",  self._on_motion)
        self._canvas_fft.mpl_connect("draw_event",           self._sync_tab_width)
        self._canvas_fft.mpl_connect("resize_event",         self._adapt_zoom_to_canvas)
        self._radial_canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _build(self):
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")

        lay = QVBoxLayout(self)
        lay.setSpacing(4)
        lay.setContentsMargins(6, 6, 6, 4)

        # Toolbar must be built first: it creates the combos the menu drives.
        toolbar_row = self._build_toolbar_row()
        lay.setMenuBar(self._build_menu_bar())
        lay.addLayout(toolbar_row)

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
        self._info_lbl.setFont(mono_font(9))
        self._info_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_lay.addWidget(self._info_lbl)
        left_col.addWidget(info_frame)

        self._fft_preview_frame = QFrame()
        preview_lay = QVBoxLayout(self._fft_preview_frame)
        preview_lay.setContentsMargins(0, 4, 0, 0)
        preview_lay.setSpacing(2)
        preview_title = QLabel("Corrected preview")
        preview_title.setFont(ui_font(9, weight=QFont.Bold))
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
        self._status_lbl.setFont(ui_font(8))
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

    # ── FFT source (whole image vs active ROI) ──────────────────────────────────

    def _has_roi_source(self) -> bool:
        return self._roi_bounds_px is not None

    def _resolve_source_array(self) -> tuple[np.ndarray, tuple[float, float]]:
        """Return the working (array, scan_range_m) for the current FFT source.

        Whole image → the full array/range. Active ROI → the ROI bbox crop with a
        proportionally-scaled scan range (pixel size preserved), via
        :func:`crop_to_bounds`.
        """
        if self._fft_source == "active_roi" and self._roi_bounds_px is not None:
            return crop_to_bounds(
                self._full_arr, self._roi_bounds_px, self._full_scan_range_m,
            )
        return (
            self._full_arr,
            (float(self._full_scan_range_m[0]), float(self._full_scan_range_m[1])),
        )

    def _on_fft_source_changed(self, *_args) -> None:
        source = "active_roi" if self._fft_source_combo.currentIndex() == 1 else "whole_image"
        if source == self._fft_source:
            return
        self._fft_source = source
        self._arr, self._scan_range_m = self._resolve_source_array()
        self._recompute_fft()
        self._update_info_panel()
        self._redraw()

    def _recompute_fft(self, reset_view: bool = True):
        """Recompute the FFT magnitude and reciprocal-space axes.

        ``reset_view`` controls the q-window:
          • True  — reset to the full reciprocal-space range (image first
            loaded, or after a correction changed the canvas/q-grid).
          • False — preserve the current zoom (e.g. switching the apodisation
            window, which leaves the image shape and q-grid unchanged).
            ``_redraw_fft_panel`` then re-fits q_x to the canvas aspect, so the
            preserved view stays undistorted.
        """
        arr = self._arr.copy()
        finite = arr[np.isfinite(arr)]
        arr[~np.isfinite(arr)] = float(np.nanmedian(finite)) if finite.size > 0 else 0.0
        arr -= arr.mean()
        arr *= self._make_window()

        F = np.fft.fftshift(np.fft.fft2(arr))
        self._fft_mag = np.abs(F)
        self._fft_phase = np.angle(F)   # radians, −π…π (same windowed F as the magnitude)

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
        if reset_view:
            self._fft_xlim = (float(self._qx[0]),  float(self._qx[-1]))
            self._fft_ylim = (float(self._qy[-1]), float(self._qy[0]))

    def _compute_display_fft(self) -> np.ndarray:
        Ny, Nx = self._fft_mag.shape
        cy, cx = Ny // 2, Nx // 2
        r = max(1, min(Ny, Nx) // 60)
        y0, y1 = max(0, cy - r), min(Ny, cy + r + 1)
        x0, x1 = max(0, cx - r), min(Nx, cx + r + 1)

        if self._fft_display_mode == "phase":
            # Phase view: radians in [−π, π], no log / no percentile range.
            phase = self._fft_phase.astype(np.float64, copy=True)
            if self._dc_mode in {"zero", "mask"}:
                # DC phase is meaningless.  Magnitude can distinguish "zero"
                # from "mask"; phase should hide the same central patch for both
                # non-keep modes instead of painting arbitrary zero-radian color.
                phase[y0:y1, x0:x1] = np.nan
            self._disp_range = (-np.pi, np.pi)
            self._last_fft_disp = phase
            return phase

        mag = self._fft_mag.copy()
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
        title = "Real space (ROI)" if self._fft_source == "active_roi" else "Real space"
        ax.set_title(title, fontsize=9, color=fg)
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
        phase_view = self._fft_display_mode == "phase"
        if phase_view:
            # Fixed −π…π range on a cyclic colour map; ignore the magnitude-only
            # percentile range controller.
            cmap_name = self._phase_cmap
            vmin_val, vmax_val = -np.pi, np.pi
        else:
            cmap_name = self._fft_cmap
            lo, hi = self._disp_range
            vmin_val, vmax_val = self._fft_drs.resolve(disp)
            if vmin_val is None:
                vmin_val, vmax_val = lo, hi
        extent_q = [
            float(self._qx[0]), float(self._qx[-1]),
            float(self._qy[-1]), float(self._qy[0]),
        ]
        self._fft_im = ax.imshow(
            disp, cmap=cmap_name, origin="upper",
            extent=extent_q, aspect="auto",
            vmin=vmin_val, vmax=vmax_val,
        )
        # Enforce the display-aspect invariant on every draw. In fill-canvas
        # (default) mode the axes stretch to fill the canvas and q_x is re-fitted
        # to the canvas aspect, preserving the current q_y zoom; in equal-aspect
        # mode the axes box is squared (matplotlib then adds side margins).
        if self._use_equal_aspect():
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_aspect("auto")
            self._aspect_correct_xlim()
        ax.set_xlim(*self._fft_xlim)
        ax.set_ylim(*self._fft_ylim)
        if phase_view:
            title_lbl = "phase, rad"
        else:
            title_lbl = "log₁₀|FFT|" if self._scale_mode == "log" else "|FFT|"
        ax.set_title(f"FFT  ({title_lbl})", fontsize=10, color=fg)
        ax.set_xlabel("q_x  (nm⁻¹)", fontsize=9, color=fg)
        ax.set_ylabel("q_y  (nm⁻¹)", fontsize=9, color=fg)
        ax.tick_params(colors=fg, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(fg)
        ax.axhline(0, color=fg, lw=0.4, alpha=0.35)
        ax.axvline(0, color=fg, lw=0.4, alpha=0.35)
        self._bragg_artists = []
        self._draw_bragg_overlay()
        self._draw_mains_overlay()
        self._draw_selection_overlay()
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
            # Receiver-aware overload: the timer is cancelled if this dialog's
            # C++ object is deleted before it fires (same below).
            QTimer.singleShot(0, self, self._zoom_fit)

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
        act = getattr(self, "_focus_fft_act", None)
        if act is not None and act.isChecked() != checked:
            act.setChecked(checked)
        QTimer.singleShot(0, self, self._adapt_zoom_to_canvas)

    def _on_show_tools_toggled(self, checked: bool) -> None:
        side = getattr(self, "_side_panel", None)
        if side is not None:
            focus_active = getattr(self, "_focus_fft_active", False)
            side.setVisible(checked and not focus_active)
        btn = getattr(self, "_show_tools_btn", None)
        if btn is not None:
            btn.setText("Hide tools" if checked else "Show tools")
        act = getattr(self, "_show_tools_act", None)
        if act is not None and act.isChecked() != checked:
            act.setChecked(checked)
        QTimer.singleShot(0, self, self._adapt_zoom_to_canvas)

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

    def _aspect_correct_xlim(self) -> None:
        """Recompute _fft_xlim from _fft_ylim and the canvas aspect (no redraw).

        Enforces the core display invariant — in fill-canvas (non-equal) mode the
        visible q_x span must equal the q_y span × canvas aspect, so q-space
        circles render as circles. Preserves the current q_y zoom and the x
        centre. No-op in equal-aspect mode (there the axes box is squared instead).
        """
        if self._use_equal_aspect():
            return
        yb, yt = self._fft_ylim   # yb > yt (inverted y axis)
        y_half = abs(yb - yt) / 2
        xc = (self._fft_xlim[0] + self._fft_xlim[1]) / 2
        x_half = y_half * self._axes_aspect()
        self._fft_xlim = (xc - x_half, xc + x_half)

    def _on_equal_aspect_toggled(self, *_args) -> None:
        """Re-fit and redraw when the equal-aspect display mode is switched."""
        self._zoom_fit()
        self._redraw()

    def _adapt_zoom_to_canvas(self, *_args) -> None:
        """Re-fit q_x to the current canvas aspect and redraw, preserving q_y zoom.

        The single chokepoint for every canvas-size change — window resize,
        splitter drag, Focus FFT / Show tools toggles — wired to the matplotlib
        ``resize_event`` so circles stay circular no matter how the size changed.
        """
        if self._use_equal_aspect():
            return
        self._aspect_correct_xlim()
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
            # Aspect-aware: scale q_y by factor, derive q_x so q/pixel is equal,
            # and keep the point (xc, yc) fixed under the cursor (so scroll-zoom
            # toward the mouse doesn't recentre the view).
            _, min_y = self._minimum_fft_spans()
            span_y = max(abs(yb - yt) * factor, min_y)
            span_x = span_y * self._axes_aspect()
            fx = (xc - xl) / (xr - xl) if xr != xl else 0.5
            fy = (yc - yt) / (yb - yt) if yb != yt else 0.5
            new_xl = xc - fx * span_x
            new_yt = yc - fy * span_y
            self._fft_xlim = (new_xl, new_xl + span_x)
            self._fft_ylim = (new_yt + span_y, new_yt)
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
        # On the Reconstruct tab, Fourier selections take priority over panning.
        if (self._reconstruct_active() and event.inaxes is self._ax_fft
                and event.button == 1 and self._fft_selection_overlay is not None):
            if self._fft_selection_overlay.on_press(event):
                return
        # On the Mains tab, grabbing a custom streak line beats panning.
        if self._mains_handle_press(event):
            return
        if (
            event.inaxes is self._ax_fft
            and event.button == 1
            and event.xdata is not None
            and event.ydata is not None
        ):
            if not self._fft_lattice_overlay_wants_event(event):
                self._pan_anchor = (
                    event.xdata, event.ydata,
                    self._fft_xlim, self._fft_ylim,
                )

    def _on_release(self, event):
        self._pan_anchor = None
        self._mains_handle_release(event)
        if self._fft_selection_overlay is not None:
            self._fft_selection_overlay.on_release(event)

    def _on_motion(self, event):
        if self._mains_handle_motion(event):
            return
        if (self._fft_selection_overlay is not None
                and self._fft_selection_overlay.is_dragging()):
            self._fft_selection_overlay.on_motion(event)
            return
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
            status = (
                f"q_x={qx:+.3f}  q_y={qy:+.3f}  |q|={q:.3f} nm⁻¹  "
                f"d={d_str}  θ={theta:.1f}°"
            )
            # Equivalent time-domain frequency along the fast axis (mains check):
            # f = q_fast · v. Only meaningful when the scan speed is known.
            if self._scan_speed_m_per_s and self._scan_speed_m_per_s > 0:
                from probeflow.processing.mains_pickup import equivalent_frequency_hz
                fast_q = qx if getattr(self, "_mains_fast_axis", "x") == "x" else qy
                f_hz = equivalent_frequency_hz(abs(fast_q), self._scan_speed_m_per_s)
                if f_hz is not None:
                    status += f"   ≈ {f_hz:.1f} Hz"
            # In the phase view, report the phase under the cursor.
            if self._fft_display_mode == "phase" and self._fft_phase is not None:
                col = int(np.argmin(np.abs(self._qx - qx)))
                row = int(np.argmin(np.abs(self._qy - qy)))
                status += f"   φ={self._fft_phase[row, col]:+.2f} rad"
            self._set_status_text(status)
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

    def _on_fft_view_changed(self, idx: int):
        self._fft_display_mode = "phase" if idx == 1 else "magnitude"
        # Scale / colour map are magnitude-only; disable them in the phase view.
        magnitude = self._fft_display_mode == "magnitude"
        for combo in (getattr(self, "_scale_combo", None), getattr(self, "_cmap_combo", None)):
            if combo is not None:
                combo.setEnabled(magnitude)
        # Scale lives in the View menu now; grey it out in phase view too.
        scale_menu = getattr(self, "_scale_menu", None)
        if scale_menu is not None:
            scale_menu.setEnabled(magnitude)
        if getattr(self, "_hist_panel", None) is not None:
            self._hist_panel.setEnabled(magnitude)
        self._redraw()   # phase is already stored; no FFT recompute needed

    def _on_scale_changed(self, idx: int):
        self._scale_mode = "log" if idx == 0 else "linear"
        self._redraw()

    def _on_window_changed(self, idx: int):
        self._window_mode = ["hann", "none", "tukey"][idx]
        # Apodisation change leaves the image shape and q-grid unchanged, so keep
        # the user's current zoom rather than snapping back to the full FFT.
        self._recompute_fft(reset_view=False)
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
        if self._fft_display_mode == "phase":
            self._hist_panel.render(
                flat_phys=finite.ravel(),
                lo_phys=-np.pi,
                hi_phys=np.pi,
                unit="rad",
                axis_label="Phase",
                theme=self._theme,
                scale=1.0,
                data_min_phys=-np.pi,
                data_max_phys=np.pi,
            )
            self._hist_panel.set_slider_positions(0, 1000, 500, 0)
            self._hist_panel.set_slider_labels(
                f"{-np.pi:.3g} rad",
                f"{np.pi:.3g} rad",
                "0 rad",
                f"{2 * np.pi:.3g} rad",
            )
            return
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
        if self._fft_display_mode == "phase":
            self._fft_im.set_clim(-np.pi, np.pi)
            self._canvas_fft.draw_idle()
            self._hist_panel.update_drag_lines(-np.pi, np.pi)
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

    def _fft_histogram_is_adjustable(self) -> bool:
        return self._fft_display_mode != "phase"

    def _on_fft_hist_min_released(self, value: int) -> None:
        if self._fft_histogram_is_adjustable():
            self._display_slider_ctrl.on_min_changed(value)

    def _on_fft_hist_max_released(self, value: int) -> None:
        if self._fft_histogram_is_adjustable():
            self._display_slider_ctrl.on_max_changed(value)

    def _on_fft_hist_brightness_released(self, value: int) -> None:
        if self._fft_histogram_is_adjustable():
            self._display_slider_ctrl.on_brightness_changed(value)

    def _on_fft_hist_contrast_released(self, value: int) -> None:
        if self._fft_histogram_is_adjustable():
            self._display_slider_ctrl.on_contrast_changed(value)

    def _on_fft_hist_range_released(self, lo_phys: float, hi_phys: float) -> None:
        if not self._fft_histogram_is_adjustable():
            return
        self._fft_drs.set_manual(lo_phys, hi_phys)

    # Auto-contrast presets for the (log-scaled) FFT display. A single
    # idempotent reset to 0–100 % meant repeated Auto clicks visibly did
    # nothing; cycling through progressively tighter percentile windows
    # gives each click an effect and returns to the full range.
    _FFT_AUTO_PRESETS = (
        (0.0, 100.0, "full range"),
        (1.0, 99.5, "1–99.5 %"),
        (5.0, 98.0, "5–98 %"),
    )

    def _reset_intensity(self) -> None:
        if not self._fft_histogram_is_adjustable():
            return
        idx = (getattr(self, "_fft_auto_idx", -1) + 1) % len(self._FFT_AUTO_PRESETS)
        self._fft_auto_idx = idx
        lo, hi, label = self._FFT_AUTO_PRESETS[idx]
        self._fft_drs.reset(lo, hi)
        status = getattr(self, "_mains_status_lbl", None)
        if self._mains_tab_active() and status is not None:
            status.setText(f"FFT auto contrast: {label}.")

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
        if self._fft_source == "active_roi":
            label = self._roi_name or "active ROI"
            source_line = f"Source:   ROI ({label})\n"
        else:
            source_line = "Source:   Whole image\n"
        self._info_lbl.setText(
            f"{source_line}"
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
        if self._fft_display_mode == "phase":
            # Azimuthal averaging of phase is not meaningful.
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "Radial profile:\nmagnitude only",
                    ha="center", va="center", color=fg, fontsize=8,
                    transform=ax.transAxes)
            for sp in ax.spines.values():
                sp.set_color(fg)
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
            return
        if self._qx is None or self._qy is None:
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
