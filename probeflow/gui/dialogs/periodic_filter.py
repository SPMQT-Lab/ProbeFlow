from __future__ import annotations

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout,
)

from .fft_viewer import FFTViewerDialog


class PeriodicFilterDialog(FFTViewerDialog):
    """Interactive FFT peak picker for periodic notch filtering.

    Inherits all FFT display infrastructure from FFTViewerDialog: colormap
    picker, zoom buttons (Fit/Ctr/+/-), scroll-to-zoom, pan, histogram with
    draggable intensity slider, and radial power spectrum tab.  Click adds /
    removes a peak; each selection suppresses that peak and its conjugate.
    """

    def __init__(
        self,
        arr: np.ndarray,
        peaks=None,
        radius_px: float = 3.0,
        scan_range_m: tuple | None = None,
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

        super().__init__(arr, scan_range_m, theme=theme, parent=parent)
        self.setWindowTitle("Periodic FFT filter")
        self.resize(750, 760)

    # ── layout (overrides parent) ──────────────────────────────────────────────

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(4)
        lay.setContentsMargins(6, 6, 6, 4)

        help_lbl = QLabel(
            "Click bright periodic peaks in the FFT power spectrum. "
            "Each click suppresses that peak and its conjugate in the processed image."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setFont(QFont("Helvetica", 9))
        lay.addWidget(help_lbl)

        lay.addLayout(self._build_toolbar_row())
        lay.addLayout(self._build_fft_column(), 1)

        # Notch radius
        radius_row = QHBoxLayout()
        radius_lbl = QLabel("Notch radius:")
        radius_lbl.setFont(QFont("Helvetica", 8))
        self._radius_sl = QSlider(Qt.Horizontal)
        self._radius_sl.setRange(1, 20)
        self._radius_sl.setValue(max(1, min(20, int(round(self._radius_px_init)))))
        self._radius_val = QLabel(f"{self._radius_sl.value()} px")
        self._radius_val.setFont(QFont("Helvetica", 8))
        self._radius_sl.valueChanged.connect(lambda v: self._radius_val.setText(f"{v} px"))
        radius_row.addWidget(radius_lbl)
        radius_row.addWidget(self._radius_sl, 1)
        radius_row.addWidget(self._radius_val)
        lay.addLayout(radius_row)

        self._selected_lbl = QLabel("Selected peaks: none")
        self._selected_lbl.setWordWrap(True)
        self._selected_lbl.setFont(QFont("Helvetica", 8))
        lay.addWidget(self._selected_lbl)

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
        self._status_lbl.setFont(QFont("Helvetica", 8))
        self._status_lbl.setAlignment(Qt.AlignLeft)
        lay.addWidget(self._status_lbl)

        self._connect_canvas_events()

    # ── no real-space panel ────────────────────────────────────────────────────

    def _update_info_panel(self):
        pass

    def _redraw(self):
        self._redraw_fft_panel()
        self._draw_peaks()

    # ── peak picking (overrides pan behaviour) ─────────────────────────────────

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
            scale_lbl = "log|FFT|" if self._scale_mode == "log" else "|FFT|"
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

    # ── peak markers ───────────────────────────────────────────────────────────

    def _draw_peaks(self):
        """Overlay selected peak circles on the FFT axes without a full redraw."""
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

    def _update_peaks_label(self):
        if self._peaks:
            text = ", ".join(f"({dx:+d}, {dy:+d})" for dx, dy in self._peaks)
            self._selected_lbl.setText(f"Selected peaks: {text}")
        else:
            self._selected_lbl.setText("Selected peaks: none")

    def _clear(self):
        self._peaks.clear()
        self._draw_peaks()

    # ── public API (unchanged from original) ───────────────────────────────────

    def selected_peaks(self) -> list[tuple[int, int]]:
        return list(self._peaks)

    def radius_px(self) -> float:
        return float(self._radius_sl.value())
