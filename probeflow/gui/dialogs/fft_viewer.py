from __future__ import annotations

import weakref

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)


class FFTViewerDialog(QDialog):
    """Side-by-side real-space / FFT inspection window."""

    def __init__(
        self,
        arr: np.ndarray,
        scan_range_m: tuple,
        colormap: str = "gray",
        theme: dict | None = None,
        channel_unit: tuple | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.setWindowTitle("FFT Viewer")
        self.resize(1000, 700)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._arr = arr.astype(np.float64, copy=True)
        self._scan_range_m = scan_range_m
        self._colormap = colormap
        self._theme = theme or {}
        self._ch_unit = channel_unit  # (scale, unit, axis_label) or None

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
        self._fft_cmap = "gray"
        self._cmap_options = ["gray", "gray_r", "inferno", "hot", "viridis", "plasma", "turbo"]
        self._bragg_artists: list = []
        self._calib_picks: list = []   # (qx_nm, qy_nm) in nm⁻¹

        self._build()
        self._recompute_fft()
        self._update_info_panel()
        self._redraw()

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
        grid_btn = QPushButton("Grid/Lattice…")
        grid_btn.setFont(QFont("Helvetica", 9))
        grid_btn.setFixedHeight(24)
        grid_btn.setMinimumWidth(100)
        grid_btn.setToolTip("Add a reciprocal-space lattice grid overlay to the FFT")
        grid_btn.clicked.connect(self._on_open_fft_lattice)
        tb.addWidget(grid_btn)

        return tb

    def _build_fft_column(self) -> QVBoxLayout:
        """Create FFT canvas + intensity/radial tab panel; set instance attrs."""
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")

        right_col = QVBoxLayout()
        right_col.setSpacing(2)

        self._fig_fft = Figure(figsize=(4.8, 4.5), dpi=90)
        self._fig_fft.patch.set_facecolor(bg)
        self._canvas_fft = FigureCanvasQTAgg(self._fig_fft)
        self._ax_fft = self._fig_fft.add_subplot(111)
        self._ax_fft.set_facecolor(bg)
        for sp in self._ax_fft.spines.values():
            sp.set_color(fg)
        self._ax_fft.tick_params(colors=fg, labelsize=9)
        self._fig_fft.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.14)
        right_col.addWidget(self._canvas_fft, 1)

        self._tab_widget = QTabWidget()
        self._tab_widget.setFixedHeight(250)
        self._tab_widget.setFont(QFont("Helvetica", 9))

        # ── Intensity tab ─────────────────────────────────────────────────────
        intensity_tab = QWidget()
        int_lay = QVBoxLayout(intensity_tab)
        int_lay.setSpacing(2)
        int_lay.setContentsMargins(4, 2, 4, 2)

        self._hist_fig = Figure(figsize=(1, 1), dpi=90)
        self._hist_fig.patch.set_facecolor(bg)
        self._hist_canvas = FigureCanvasQTAgg(self._hist_fig)
        self._hist_ax = self._hist_fig.add_subplot(111)
        self._hist_ax.set_facecolor(bg)
        int_lay.addWidget(self._hist_canvas, 1)

        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        reset_intensity_btn = QPushButton("Reset range")
        reset_intensity_btn.setFont(QFont("Helvetica", 9))
        reset_intensity_btn.setFixedHeight(22)
        reset_intensity_btn.setMinimumWidth(88)
        reset_intensity_btn.setToolTip("Reset intensity to full range")
        reset_intensity_btn.clicked.connect(self._reset_intensity)
        reset_row.addWidget(reset_intensity_btn)
        int_lay.addLayout(reset_row)

        self._hist_canvas.mpl_connect("button_press_event",   self._on_hist_press)
        self._hist_canvas.mpl_connect("motion_notify_event",  self._on_hist_motion)
        self._hist_canvas.mpl_connect("button_release_event", self._on_hist_release)

        self._tab_widget.addTab(intensity_tab, "Intensity")

        # ── Radial profile tab ────────────────────────────────────────────────
        radial_tab = QWidget()
        rad_lay = QVBoxLayout(radial_tab)
        rad_lay.setSpacing(0)
        rad_lay.setContentsMargins(2, 2, 2, 2)

        self._radial_fig = Figure(figsize=(1, 1), dpi=90)
        self._radial_fig.patch.set_facecolor(bg)
        self._radial_canvas = FigureCanvasQTAgg(self._radial_fig)
        self._radial_ax = self._radial_fig.add_axes([0.11, 0.24, 0.85, 0.66])
        self._radial_ax.set_facecolor(bg)
        rad_lay.addWidget(self._radial_canvas)

        self._tab_widget.addTab(radial_tab, "Radial profile")

        # ── Predicted Lattice tab (scrollable, so the 250 px tab height is safe) ─
        lat_scroll = QScrollArea()
        lat_scroll.setWidgetResizable(True)
        lat_scroll.setFrameShape(QFrame.NoFrame)
        lat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lat_inner = QWidget()
        lat_lay = QVBoxLayout(lat_inner)
        lat_lay.setSpacing(4)
        lat_lay.setContentsMargins(6, 6, 6, 4)

        enable_row = QHBoxLayout()
        self._bragg_enable_cb = QCheckBox("Show predicted Bragg ring")
        self._bragg_enable_cb.setFont(QFont("Helvetica", 9))
        self._bragg_enable_cb.toggled.connect(self._on_bragg_changed)
        enable_row.addWidget(self._bragg_enable_cb)
        enable_row.addStretch(1)
        lat_lay.addLayout(enable_row)

        sym_row = QHBoxLayout()
        sym_lbl = QLabel("Symmetry:")
        sym_lbl.setFont(QFont("Helvetica", 9))
        sym_row.addWidget(sym_lbl)
        self._bragg_sym_combo = QComboBox()
        self._bragg_sym_combo.addItems(["Square", "Hexagonal"])
        self._bragg_sym_combo.setFont(QFont("Helvetica", 9))
        self._bragg_sym_combo.setFixedHeight(24)
        self._bragg_sym_combo.currentIndexChanged.connect(self._on_bragg_changed)
        sym_row.addWidget(self._bragg_sym_combo)
        sym_row.addStretch(1)
        lat_lay.addLayout(sym_row)

        a_row = QHBoxLayout()
        a_lbl = QLabel("Lattice a:")
        a_lbl.setFont(QFont("Helvetica", 9))
        a_row.addWidget(a_lbl)
        self._bragg_a_spin = QDoubleSpinBox()
        self._bragg_a_spin.setRange(0.001, 999.0)
        self._bragg_a_spin.setValue(2.46)
        self._bragg_a_spin.setDecimals(3)
        self._bragg_a_spin.setFont(QFont("Helvetica", 9))
        self._bragg_a_spin.setFixedHeight(24)
        self._bragg_a_spin.valueChanged.connect(self._on_bragg_changed)
        a_row.addWidget(self._bragg_a_spin)
        self._bragg_unit_combo = QComboBox()
        self._bragg_unit_combo.addItems(["Å", "nm"])
        self._bragg_unit_combo.setFont(QFont("Helvetica", 9))
        self._bragg_unit_combo.setFixedHeight(24)
        self._bragg_unit_combo.currentIndexChanged.connect(self._on_bragg_changed)
        a_row.addWidget(self._bragg_unit_combo)
        a_row.addStretch(1)
        lat_lay.addLayout(a_row)

        order2_row = QHBoxLayout()
        self._bragg_order2_cb = QCheckBox("Show 2nd-order ring")
        self._bragg_order2_cb.setFont(QFont("Helvetica", 9))
        self._bragg_order2_cb.toggled.connect(self._on_bragg_changed)
        order2_row.addWidget(self._bragg_order2_cb)
        order2_row.addStretch(1)
        lat_lay.addLayout(order2_row)

        self._bragg_radius_lbl = QLabel("Radius: —")
        self._bragg_radius_lbl.setFont(QFont("Courier", 9))
        lat_lay.addWidget(self._bragg_radius_lbl)

        # ── calibration section ──────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        lat_lay.addWidget(sep)

        det_row = QHBoxLayout()
        self._bragg_detect_btn = QPushButton("Detect peaks")
        self._bragg_detect_btn.setFont(QFont("Helvetica", 9))
        self._bragg_detect_btn.setFixedHeight(24)
        self._bragg_detect_btn.setToolTip(
            "Auto-detect first-order Bragg peaks inside the predicted annulus"
        )
        self._bragg_detect_btn.clicked.connect(self._detect_bragg_peaks)
        det_row.addWidget(self._bragg_detect_btn)
        self._bragg_clear_btn = QPushButton("Clear")
        self._bragg_clear_btn.setFont(QFont("Helvetica", 9))
        self._bragg_clear_btn.setFixedHeight(24)
        self._bragg_clear_btn.clicked.connect(self._clear_bragg_picks)
        det_row.addWidget(self._bragg_clear_btn)
        det_row.addStretch(1)
        lat_lay.addLayout(det_row)

        self._bragg_picks_lbl = QLabel("Picks: 0  (click FFT to add/remove)")
        self._bragg_picks_lbl.setFont(QFont("Helvetica", 9))
        lat_lay.addWidget(self._bragg_picks_lbl)

        piezo_row = QHBoxLayout()
        px_lbl = QLabel("Piezo X:")
        px_lbl.setFont(QFont("Helvetica", 9))
        piezo_row.addWidget(px_lbl)
        self._bragg_cx_edit = QLineEdit("96.52")
        self._bragg_cx_edit.setFont(QFont("Helvetica", 9))
        self._bragg_cx_edit.setFixedHeight(24)
        self._bragg_cx_edit.setFixedWidth(72)
        piezo_row.addWidget(self._bragg_cx_edit)
        piezo_row.addSpacing(8)
        py_lbl = QLabel("Piezo Y:")
        py_lbl.setFont(QFont("Helvetica", 9))
        piezo_row.addWidget(py_lbl)
        self._bragg_cy_edit = QLineEdit("96.52")
        self._bragg_cy_edit.setFont(QFont("Helvetica", 9))
        self._bragg_cy_edit.setFixedHeight(24)
        self._bragg_cy_edit.setFixedWidth(72)
        piezo_row.addWidget(self._bragg_cy_edit)
        piezo_row.addStretch(1)
        lat_lay.addLayout(piezo_row)

        compute_row = QHBoxLayout()
        self._bragg_compute_btn = QPushButton("Compute correction")
        self._bragg_compute_btn.setFont(QFont("Helvetica", 9))
        self._bragg_compute_btn.setFixedHeight(24)
        self._bragg_compute_btn.setEnabled(False)
        self._bragg_compute_btn.setToolTip("Need at least 3 picks")
        self._bragg_compute_btn.clicked.connect(self._compute_bragg_correction)
        compute_row.addWidget(self._bragg_compute_btn)
        compute_row.addStretch(1)
        lat_lay.addLayout(compute_row)

        self._bragg_results_txt = QPlainTextEdit()
        self._bragg_results_txt.setReadOnly(True)
        self._bragg_results_txt.setFont(QFont("Courier", 9))
        self._bragg_results_txt.setFixedHeight(88)
        self._bragg_results_txt.setPlaceholderText(
            "Results will appear here after computing."
        )
        lat_lay.addWidget(self._bragg_results_txt)

        copy_row = QHBoxLayout()
        self._bragg_copy_btn = QPushButton("Copy")
        self._bragg_copy_btn.setFont(QFont("Helvetica", 9))
        self._bragg_copy_btn.setFixedHeight(22)
        self._bragg_copy_btn.setMinimumWidth(56)
        self._bragg_copy_btn.setToolTip("Copy results to clipboard")
        self._bragg_copy_btn.clicked.connect(self._copy_bragg_results)
        copy_row.addWidget(self._bragg_copy_btn)
        copy_row.addStretch(1)
        lat_lay.addLayout(copy_row)

        lat_lay.addStretch(1)
        lat_scroll.setWidget(lat_inner)
        self._tab_widget.addTab(lat_scroll, "Predicted Lattice")

        # wrap tab_widget in a row with spacers so its width tracks the FFT axes area
        self._tab_left_spacer = QWidget()
        self._tab_left_spacer.setFixedWidth(0)
        self._tab_right_spacer = QWidget()
        self._tab_right_spacer.setFixedWidth(0)
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(0, 0, 0, 0)
        tab_row.setSpacing(0)
        tab_row.addWidget(self._tab_left_spacer)
        tab_row.addWidget(self._tab_widget, 1)
        tab_row.addWidget(self._tab_right_spacer)
        right_col.addLayout(tab_row)

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

        # ── body row: left column | right column ──────────────────────────────
        body_row = QHBoxLayout()
        body_row.setSpacing(4)

        # ── left column: real-space image + info panel ────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(2)

        self._fig_real = Figure(figsize=(4.8, 4.5), dpi=90)
        self._fig_real.patch.set_facecolor(bg)
        self._canvas_real = FigureCanvasQTAgg(self._fig_real)
        self._ax_real = self._fig_real.add_subplot(111)
        self._ax_real.set_facecolor(bg)
        for sp in self._ax_real.spines.values():
            sp.set_color(fg)
        self._ax_real.tick_params(colors=fg, labelsize=9)
        self._fig_real.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.14)
        left_col.addWidget(self._canvas_real, 1)

        info_frame = QFrame()
        info_frame.setFixedHeight(250)
        info_lay = QVBoxLayout(info_frame)
        info_lay.setContentsMargins(8, 6, 8, 4)
        info_lay.setSpacing(2)
        self._info_lbl = QLabel("")
        self._info_lbl.setFont(QFont("Courier", 9))
        self._info_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_lay.addWidget(self._info_lbl)
        info_lay.addStretch(1)
        left_col.addWidget(info_frame)

        body_row.addLayout(left_col, 1)
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
        ax.set_title("Real space", fontsize=10, color=fg)
        ax.set_xlabel("nm", fontsize=9, color=fg)
        ax.set_ylabel("nm", fontsize=9, color=fg)
        ax.tick_params(colors=fg, labelsize=8)
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
        """Keep tab panel width equal to the FFT axes data area."""
        bbox = self._ax_fft.get_position()  # figure-fraction Bbox
        w = self._canvas_fft.width()
        self._tab_left_spacer.setFixedWidth(max(0, int(bbox.x0 * w)))
        self._tab_right_spacer.setFixedWidth(max(0, int((1.0 - bbox.x1) * w)))

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_tab_width()

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
            self._status_lbl.setText(
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
            self._status_lbl.setText(
                f"q={q:.3f} nm⁻¹  d={d_str}  ⟨{scale_lbl}⟩={val:.4g}"
            )
        elif event.inaxes is getattr(self, "_ax_real", None) and event.xdata is not None:
            self._status_lbl.setText(
                f"x={event.xdata:.2f} nm  y={event.ydata:.2f} nm"
            )
        else:
            self._status_lbl.setText("")

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

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export FFT view", "fft_view.png",
            "PNG image (*.png);;All files (*)"
        )
        if path:
            self._fig_fft.savefig(path, dpi=150, bbox_inches="tight")

    def _on_open_fft_lattice(self):
        from probeflow.gui.lattice_grid import open_fft_tool
        if self._qx is None or self._qy is None:
            return
        Ny, Nx = self._arr.shape[:2]
        overlay, panel = open_fft_tool(
            self._ax_fft, self._canvas_fft,
            self._qx, self._qy,
            (Ny, Nx), parent=self,
        )
        self._fft_lattice_overlay = overlay
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

        # Use a parentless QWidget (Qt.Window) rather than a QDialog.
        #
        # Both QDockWidget(setFloating) and QDialog cause macOS menu-bar
        # blocking in different ways:
        #   - floating QDockWidget: Z-order undefined when parent is a QDialog
        #   - QDialog (any window flags): Qt still creates an NSPanel at an
        #     elevated window level when the class is QDialog and its parent
        #     chain contains other dialogs; the elevated level causes macOS to
        #     grey-out and block the application menu bar (File, View, …)
        #
        # A plain QWidget with no parent and Qt.Window is a fully independent
        # NSWindow at the normal window level. It cannot block menus, cannot
        # create modal relationships, and participates in ordinary Z-order.
        # We tag it with _probeflow_tool_window so window_menu.py can include
        # it in the Window menu even though it has no Qt parent.
        dlg = QWidget(None, Qt.Window)
        dlg.setWindowTitle("Reciprocal Grid")
        dlg.setAttribute(Qt.WA_DeleteOnClose)
        dlg._probeflow_tool_window = True   # recognised by window_menu.py
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setContentsMargins(0, 0, 0, 0)
        dlg_layout.setSpacing(0)
        dlg_layout.addWidget(panel)
        dlg.setMinimumWidth(240)
        dlg.adjustSize()
        self._fft_lattice_dock = dlg

        # No Qt parent means Qt won't auto-close the panel when the FFT viewer
        # closes; do it explicitly.
        self.finished.connect(lambda _result, _d=dlg: _d.close())

        def _clear_dock_ref():
            if getattr(self, "_fft_lattice_dock", None) is dlg:
                self._fft_lattice_dock = None

        dlg.destroyed.connect(_clear_dock_ref)
        dlg.show()

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

    def _draw_bragg_overlay(self):
        """Add predicted Bragg ring circle(s) to the FFT axes if enabled.

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

        # ── scan metadata ────────────────────────────────────────────────────
        Ny, Nx = self._arr.shape
        try:
            w_m = float(self._scan_range_m[0])
            h_m = float(self._scan_range_m[1])
        except Exception:
            return

        is_nonsquare = abs(w_m - h_m) / max(w_m, h_m, 1e-30) > 0.01
        scan_m = (w_m * h_m) ** 0.5   # geometric mean for non-square scans
        n_px = max(Nx, Ny)

        # ── user inputs ──────────────────────────────────────────────────────
        symmetry = "square" if self._bragg_sym_combo.currentIndex() == 0 else "hex"
        a_val = self._bragg_a_spin.value()
        unit = self._bragg_unit_combo.currentText()
        a_m = a_val * 1e-10 if unit == "Å" else a_val * 1e-9

        # ── compute radii in nm⁻¹ ────────────────────────────────────────────
        from probeflow.processing.filters import predicted_bragg_radius
        try:
            r_px_1 = predicted_bragg_radius(a_m, symmetry, scan_m, n_px, order=1)
        except ValueError:
            self._bragg_radius_lbl.setText("Radius: invalid input")
            return

        # Convert FFT pixel units → nm⁻¹:  q = r_pixels / (scan_m * 1e9)
        q1 = r_px_1 / (scan_m * 1e9)

        # ── draw first-order ring ─────────────────────────────────────────────
        theta = np.linspace(0.0, 2.0 * np.pi, 360)
        art1, = self._ax_fft.plot(
            q1 * np.cos(theta), q1 * np.sin(theta),
            color="#f38ba8", lw=1.2, linestyle="--", alpha=0.9, zorder=7,
        )
        self._bragg_artists.append(art1)

        label_lines = [f"1st:  {q1:.4f} nm⁻¹  (d = {1.0/q1*10:.3f} Å)"]

        # ── optional second-order ring ────────────────────────────────────────
        if self._bragg_order2_cb.isChecked():
            r_px_2 = predicted_bragg_radius(a_m, symmetry, scan_m, n_px, order=2)
            q2 = r_px_2 / (scan_m * 1e9)
            art2, = self._ax_fft.plot(
                q2 * np.cos(theta), q2 * np.sin(theta),
                color="#fab387", lw=1.0, linestyle=":", alpha=0.8, zorder=7,
            )
            self._bragg_artists.append(art2)
            label_lines.append(f"2nd:  {q2:.4f} nm⁻¹  (d = {1.0/q2*10:.3f} Å)")

        if is_nonsquare:
            label_lines.append("(non-square scan: geom. mean)")

        self._bragg_radius_lbl.setText("\n".join(label_lines))
        self._draw_calib_pick_artists()

    # ── calibration helpers ────────────────────────────────────────────────────

    def _bragg_pick_mode_active(self) -> bool:
        """True when clicks on the FFT canvas should add/remove calibration picks."""
        cb = getattr(self, "_bragg_enable_cb", None)
        return bool(cb is not None and cb.isChecked())

    def _draw_calib_pick_artists(self) -> None:
        """Draw calibration pick dots on the FFT axes (always, if any picks exist)."""
        for qx, qy in self._calib_picks:
            art, = self._ax_fft.plot(
                qx, qy, "o",
                color="#a6e3a1", markerfacecolor="none",
                markersize=9, markeredgewidth=1.8, zorder=8,
            )
            self._bragg_artists.append(art)

    def _update_calib_ui(self) -> None:
        """Refresh the picks label and the compute-button enabled state."""
        n = len(self._calib_picks)
        self._bragg_picks_lbl.setText(
            f"Picks: {n}  (click FFT to add/remove)"
        )
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

        # No existing pick nearby — add a new one
        self._calib_picks.append((qx_click, qy_click))
        self._update_calib_ui()
        self._on_bragg_changed()

    def _detect_bragg_peaks(self) -> None:
        """Auto-detect first-order Bragg peaks and store them as calibration picks."""
        from probeflow.processing.filters import (
            find_bragg_peaks_in_annulus,
            predicted_bragg_radius,
        )
        if self._fft_mag is None or self._qx is None:
            return

        Ny, Nx = self._arr.shape
        try:
            w_m = float(self._scan_range_m[0])
            h_m = float(self._scan_range_m[1])
        except Exception:
            return
        scan_m = (w_m * h_m) ** 0.5
        n_px = max(Nx, Ny)

        symmetry = "square" if self._bragg_sym_combo.currentIndex() == 0 else "hex"
        a_val = self._bragg_a_spin.value()
        unit = self._bragg_unit_combo.currentText()
        a_m = a_val * 1e-10 if unit == "Å" else a_val * 1e-9

        try:
            r_predicted_px = predicted_bragg_radius(a_m, symmetry, scan_m, n_px, order=1)
        except ValueError:
            return

        expected = 4 if symmetry == "square" else 6
        peaks_px = find_bragg_peaks_in_annulus(
            self._fft_mag, r_predicted_px, expected_count=expected,
        )

        if peaks_px.size == 0:
            self._bragg_picks_lbl.setText("Peaks: none found in annulus")
            return

        # Convert (x_px, y_px) offsets from centre → nm⁻¹
        scan_w_nm = w_m * 1e9
        scan_h_nm = h_m * 1e9
        self._calib_picks = [
            (x_px / scan_w_nm, y_px / scan_h_nm)
            for x_px, y_px in peaks_px
        ]
        self._update_calib_ui()
        self._on_bragg_changed()

    def _clear_bragg_picks(self) -> None:
        self._calib_picks = []
        self._update_calib_ui()
        self._on_bragg_changed()

    def _compute_bragg_correction(self) -> None:
        """Fit an axis-aligned ellipse to the picks and report piezo corrections."""
        from probeflow.processing.filters import (
            fit_axis_aligned_ellipse,
            piezo_correction,
            predicted_bragg_radius,
        )
        if len(self._calib_picks) < 3:
            return

        # ── gather scan metadata ─────────────────────────────────────────────
        Ny, Nx = self._arr.shape
        try:
            w_m = float(self._scan_range_m[0])
            h_m = float(self._scan_range_m[1])
        except Exception:
            self._bragg_results_txt.setPlainText("Error: scan metadata unavailable")
            return
        scan_m = (w_m * h_m) ** 0.5

        symmetry = "square" if self._bragg_sym_combo.currentIndex() == 0 else "hex"
        a_val = self._bragg_a_spin.value()
        unit = self._bragg_unit_combo.currentText()
        a_m = a_val * 1e-10 if unit == "Å" else a_val * 1e-9

        try:
            r_px_1 = predicted_bragg_radius(a_m, symmetry, scan_m, max(Nx, Ny), order=1)
        except ValueError as exc:
            self._bragg_results_txt.setPlainText(f"Error: {exc}")
            return
        r_predicted_nm = r_px_1 / (scan_m * 1e9)

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
