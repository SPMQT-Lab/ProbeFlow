from __future__ import annotations

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QVBoxLayout, QWidget,
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
        self._disp_range: tuple = (0.0, 1.0)
        self._vmin_frac: float = 0.0
        self._vmax_frac: float = 1.0
        self._hist_drag: str | None = None
        self._vmin_line = None
        self._vmax_line = None
        self._fft_im = None
        self._fft_cmap = "gray"
        self._cmap_options = ["gray", "gray_r", "inferno", "hot", "viridis", "plasma", "turbo"]

        self._build()
        self._recompute_fft()
        self._update_info_panel()
        self._redraw()

    # ── layout ─────────────────────────────────────────────────────────────────

    def _build(self):
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")

        lay = QVBoxLayout(self)
        lay.setSpacing(4)
        lay.setContentsMargins(6, 6, 6, 4)

        # ── toolbar ───────────────────────────────────────────────────────────
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
            ("Fit",  "Zoom to fit full FFT extent",    self._zoom_fit),
            ("Ctr",  "Zoom to centre (quarter range)", self._zoom_centre),
            ("  +  ", "Zoom in",                       lambda: self._zoom_by(0.5)),
            ("  −  ", "Zoom out",                      lambda: self._zoom_by(2.0)),
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

        lay.addLayout(tb)

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

        # ── right column: FFT image + tab panel ───────────────────────────────
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

        # tab panel (fixed height, aligned with info_frame)
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

        body_row.addLayout(right_col, 1)

        lay.addLayout(body_row, 1)

        # ── status bar ────────────────────────────────────────────────────────
        self._status_lbl = QLabel("")
        self._status_lbl.setFont(QFont("Helvetica", 8))
        self._status_lbl.setAlignment(Qt.AlignLeft)
        lay.addWidget(self._status_lbl)

        # ── canvas event connections ──────────────────────────────────────────
        self._canvas_fft.mpl_connect("scroll_event",         self._on_scroll)
        self._canvas_fft.mpl_connect("button_press_event",   self._on_press)
        self._canvas_fft.mpl_connect("button_release_event", self._on_release)
        self._canvas_fft.mpl_connect("motion_notify_event",  self._on_motion)
        self._canvas_fft.mpl_connect("draw_event",           self._sync_tab_width)
        self._canvas_real.mpl_connect("motion_notify_event", self._on_motion)
        self._radial_canvas.mpl_connect("motion_notify_event", self._on_motion)

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
            mag = np.log1p(mag)
        finite = mag[np.isfinite(mag)]
        self._disp_range = (
            (float(finite.min()), float(finite.max())) if finite.size > 0 else (0.0, 1.0)
        )
        return mag

    # ── drawing ────────────────────────────────────────────────────────────────

    def _redraw(self):
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
        scale_lbl = "log|FFT|" if self._scale_mode == "log" else "|FFT|"
        ax.set_title(f"FFT  ({scale_lbl})", fontsize=10, color=fg)
        ax.set_xlabel("q_x  (nm⁻¹)", fontsize=9, color=fg)
        ax.set_ylabel("q_y  (nm⁻¹)", fontsize=9, color=fg)
        ax.tick_params(colors=fg, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(fg)
        ax.axhline(0, color=fg, lw=0.4, alpha=0.35)
        ax.axvline(0, color=fg, lw=0.4, alpha=0.35)

        self._update_histogram(disp)
        self._update_radial_profile(disp)
        self._canvas_real.draw_idle()
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
        self._fft_xlim = (xc + (xl - xc) * factor, xc + (xr - xc) * factor)
        self._fft_ylim = (yc + (yb - yc) * factor, yc + (yt - yc) * factor)
        self._ax_fft.set_xlim(*self._fft_xlim)
        self._ax_fft.set_ylim(*self._fft_ylim)
        self._canvas_fft.draw_idle()

    # ── canvas events ──────────────────────────────────────────────────────────

    def _on_scroll(self, event):
        if event.inaxes is not self._ax_fft:
            return
        factor = 0.65 if event.step > 0 else 1.0 / 0.65
        self._zoom_by(factor, event.xdata, event.ydata)

    def _on_press(self, event):
        if event.inaxes is self._ax_fft and event.button == 1:
            self._pan_anchor = (
                event.xdata, event.ydata,
                self._fft_xlim, self._fft_ylim,
            )

    def _on_release(self, event):
        self._pan_anchor = None

    def _on_motion(self, event):
        if self._pan_anchor is not None and event.inaxes is self._ax_fft:
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
            scale_lbl = "log|FFT|" if self._scale_mode == "log" else "|FFT|"
            if q > 0:
                d_nm = 1.0 / q
                d_str = f"{d_nm:.2f} nm" if d_nm >= 1.0 else f"{d_nm * 10:.2f} Å"
            else:
                d_str = "∞"
            self._status_lbl.setText(
                f"q={q:.3f} nm⁻¹  d={d_str}  ⟨{scale_lbl}⟩={val:.4g}"
            )
        elif event.inaxes is self._ax_real and event.xdata is not None:
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
        scale_lbl = "log|FFT|" if self._scale_mode == "log" else "|FFT|"
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
