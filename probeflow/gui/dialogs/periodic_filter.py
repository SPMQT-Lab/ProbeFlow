from __future__ import annotations

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from probeflow.gui.typography import mono_font, ui_font
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from probeflow.processing.image import periodic_notch_filter
from .fft_viewer import FFTViewerDialog


class PeriodicFilterDialog(FFTViewerDialog):
    """Interactive FFT peak picker for periodic notch filtering.

    Layout mirrors FFTViewerDialog: filtered real-space preview + image info
    on the left; FFT canvas with zoom / LUT / histogram / radial-profile
    controls on the right, with the notch-radius slider and peaks list
    constrained to the FFT axes width (same spacer technique as the histogram).
    """

    def __init__(
        self,
        arr: np.ndarray,
        peaks=None,
        radius_px: float = 3.0,
        scan_range_m: tuple | None = None,
        colormap: str = "gray",
        theme: dict | None = None,
        parent=None,
    ):
        self._peaks: list[tuple[int, int]] = [
            (int(p[0]), int(p[1])) for p in (peaks or [])
        ]
        self._radius_px_init = radius_px
        self._peak_artists: list = []

        if scan_range_m is None:
            scan_range_m = (float(arr.shape[1]) * 1e-9, float(arr.shape[0]) * 1e-9)

        super().__init__(arr, scan_range_m, colormap=colormap, theme=theme, parent=parent)
        self.setWindowTitle("Periodic FFT filter")
        self.resize(1050, 760)
        if hasattr(self, "_tab_widget"):
            self._tab_widget.setTabVisible(0, False)  # FFT correction workflow
            self._tab_widget.setTabVisible(2, False)  # reciprocal-grid advanced controls
            self._tab_widget.setCurrentIndex(1)

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self):
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")

        lay = QVBoxLayout(self)
        lay.setSpacing(4)
        lay.setContentsMargins(6, 6, 6, 4)

        help_lbl = QLabel(
            "Click bright periodic peaks in the FFT power spectrum. "
            "Each click suppresses that peak and its conjugate; "
            "the left panel shows a live preview of the filtered image."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setFont(ui_font(9))
        lay.addWidget(help_lbl)

        lay.addLayout(self._build_toolbar_row())

        # ── body row ──────────────────────────────────────────────────────────
        body_row = QHBoxLayout()
        body_row.setSpacing(4)

        # ── left column: filtered preview + info ──────────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(2)

        self._fig_preview = Figure(figsize=(4.8, 4.5), dpi=90)
        self._fig_preview.patch.set_facecolor(bg)
        self._canvas_preview = FigureCanvasQTAgg(self._fig_preview)
        self._ax_preview = self._fig_preview.add_subplot(111)
        self._ax_preview.set_facecolor(bg)
        for sp in self._ax_preview.spines.values():
            sp.set_color(fg)
        self._ax_preview.tick_params(colors=fg, labelsize=9)
        self._fig_preview.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.14)
        left_col.addWidget(self._canvas_preview, 1)

        info_frame = QFrame()
        info_frame.setFixedHeight(310)
        info_lay = QVBoxLayout(info_frame)
        info_lay.setContentsMargins(8, 6, 8, 4)
        info_lay.setSpacing(2)
        self._info_lbl = QLabel("")
        self._info_lbl.setFont(mono_font(9))
        self._info_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_lay.addWidget(self._info_lbl)
        info_lay.addStretch(1)
        left_col.addWidget(info_frame)

        body_row.addLayout(left_col, 1)

        # ── right column: FFT canvas + tabs + notch controls ──────────────────
        fft_col = self._build_fft_column()

        # Notch controls — constrained to FFT axes width using the same
        # spacer technique as the tab widget / histogram.
        self._notch_left_spacer = QWidget()
        self._notch_left_spacer.setFixedWidth(0)
        self._notch_right_spacer = QWidget()
        self._notch_right_spacer.setFixedWidth(0)

        notch_inner = QVBoxLayout()
        notch_inner.setSpacing(3)
        notch_inner.setContentsMargins(0, 6, 0, 0)

        radius_row = QHBoxLayout()
        radius_lbl = QLabel("Notch radius:")
        radius_lbl.setFont(ui_font(8))
        self._radius_sl = QSlider(Qt.Horizontal)
        self._radius_sl.setRange(1, 20)
        self._radius_sl.setValue(max(1, min(20, int(round(self._radius_px_init)))))
        self._radius_val = QLabel(f"{self._radius_sl.value()} px")
        self._radius_val.setFont(ui_font(8))
        self._radius_val.setMinimumWidth(30)
        self._radius_sl.valueChanged.connect(self._on_radius_changed)
        radius_row.addWidget(radius_lbl)
        radius_row.addWidget(self._radius_sl, 1)
        radius_row.addWidget(self._radius_val)
        notch_inner.addLayout(radius_row)

        self._selected_lbl = QLabel("Selected peaks: none")
        self._selected_lbl.setWordWrap(True)
        self._selected_lbl.setFont(ui_font(8))
        notch_inner.addWidget(self._selected_lbl)

        notch_row = QHBoxLayout()
        notch_row.setContentsMargins(0, 0, 0, 0)
        notch_row.setSpacing(0)
        notch_row.addWidget(self._notch_left_spacer)
        notch_row.addLayout(notch_inner, 1)
        notch_row.addWidget(self._notch_right_spacer)
        fft_col.addLayout(notch_row)

        body_row.addLayout(fft_col, 1)
        lay.addLayout(body_row, 1)

        # ── button row ────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear peaks")
        clear_btn.clicked.connect(self._clear)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("Use selected peaks")
        apply_btn.setObjectName("accentBtn")
        apply_btn.clicked.connect(self.accept)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(apply_btn)
        lay.addLayout(btn_row)

        self._status_lbl = QLabel("")
        self._status_lbl.setFont(ui_font(8))
        self._status_lbl.setAlignment(Qt.AlignLeft)
        lay.addWidget(self._status_lbl)

        self._connect_canvas_events()

    # ── tab + notch width sync ────────────────────────────────────────────────

    def _sync_tab_width(self, event=None):
        super()._sync_tab_width(event)
        bbox = self._ax_fft.get_position()
        w = self._canvas_fft.width()
        left_w  = max(0, int(bbox.x0 * w))
        right_w = max(0, int((1.0 - bbox.x1) * w))
        self._notch_left_spacer.setFixedWidth(left_w)
        self._notch_right_spacer.setFixedWidth(right_w)

    # ── drawing ────────────────────────────────────────────────────────────────

    def _redraw(self):
        self._redraw_fft_panel()
        self._draw_peaks()   # also calls _redraw_preview

    def _redraw_preview(self):
        """Recompute and display the filtered image in the left preview panel."""
        bg = self._theme.get("bg", "#1e1e1e")
        fg = self._theme.get("fg", "#dddddd")
        ax = self._ax_preview
        ax.cla()
        ax.set_facecolor(bg)

        radius = float(self._radius_sl.value()) if hasattr(self, "_radius_sl") else self._radius_px_init
        if self._peaks:
            preview = periodic_notch_filter(self._arr, self._peaks, radius_px=radius)
            title = "Preview (filtered)"
        else:
            preview = self._arr
            title = "Original"

        try:
            w_nm = float(self._scan_range_m[0]) * 1e9
            h_nm = float(self._scan_range_m[1]) * 1e9
        except Exception:
            w_nm, h_nm = float(self._arr.shape[1]), float(self._arr.shape[0])

        ax.imshow(
            preview, cmap=self._colormap, origin="upper",
            extent=[0, w_nm, h_nm, 0], aspect="equal",
        )
        ax.set_title(title, fontsize=10, color=fg)
        ax.set_xlabel("nm", fontsize=9, color=fg)
        ax.set_ylabel("nm", fontsize=9, color=fg)
        ax.tick_params(colors=fg, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(fg)
        self._canvas_preview.draw_idle()

    # ── peak picking (overrides pan) ──────────────────────────────────────────

    def _on_press(self, event):
        if event.inaxes is not self._ax_fft or event.xdata is None or self._qx is None:
            return
        cx_idx = len(self._qx) // 2
        cy_idx = len(self._qy) // 2
        dx = int(np.argmin(np.abs(self._qx - event.xdata))) - cx_idx
        dy = int(np.argmin(np.abs(self._qy - event.ydata))) - cy_idx
        if dx == 0 and dy == 0:
            return
        canonical = (dx, dy)
        conjugate = (-dx, -dy)
        if conjugate in self._peaks:
            canonical = conjugate
        if canonical in self._peaks:
            self._peaks.remove(canonical)
        else:
            self._peaks.append(canonical)
        self._draw_peaks()

    def _on_release(self, event):
        pass  # no panning

    def _on_radius_changed(self, v: int):
        self._radius_val.setText(f"{v} px")
        self._redraw_preview()

    def _on_motion(self, event):
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
        else:
            self._status_lbl.setText("")

    # ── peak markers ──────────────────────────────────────────────────────────

    def _draw_peaks(self):
        """Overlay selected peak circles on the FFT axes and refresh the preview."""
        for artist in self._peak_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._peak_artists = []

        if self._qx is None or self._qy is None:
            self._canvas_fft.draw_idle()
            return

        cx_idx = len(self._qx) // 2
        cy_idx = len(self._qy) // 2

        for dx, dy in self._peaks:
            for sdx, sdy in ((dx, dy), (-dx, -dy)):
                xi = cx_idx + sdx
                yi = cy_idx + sdy
                if 0 <= xi < len(self._qx) and 0 <= yi < len(self._qy):
                    art, = self._ax_fft.plot(
                        float(self._qx[xi]), float(self._qy[yi]),
                        "o", color="#89b4fa", markerfacecolor="none",
                        markersize=9, markeredgewidth=1.8,
                    )
                    self._peak_artists.append(art)

        self._canvas_fft.draw_idle()
        self._update_peaks_label()
        self._redraw_preview()

    def _update_peaks_label(self):
        if self._peaks:
            text = ", ".join(f"({dx:+d}, {dy:+d})" for dx, dy in self._peaks)
            self._selected_lbl.setText(f"Selected peaks: {text}")
        else:
            self._selected_lbl.setText("Selected peaks: none")

    def _clear(self):
        self._peaks.clear()
        self._draw_peaks()

    # ── public API ────────────────────────────────────────────────────────────

    def selected_peaks(self) -> list[tuple[int, int]]:
        return list(self._peaks)

    def radius_px(self) -> float:
        return float(self._radius_sl.value())
