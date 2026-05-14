"""Small viewer for FFTs computed from detected feature-point masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from probeflow.measurements.fft_points import PointFFTResult


class PointMaskFFTDialog(QDialog):
    """Display a derived point mask and its FFT magnitude."""

    def __init__(
        self,
        mask: np.ndarray,
        fft_result: PointFFTResult,
        *,
        title: str = "Point-mask FFT",
        theme: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 520)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self._mask = np.asarray(mask, dtype=bool)
        self._result = fft_result
        self._theme = theme or {}
        self._build()
        self._draw()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self._info_lbl = QLabel(self._info_text())
        self._info_lbl.setWordWrap(True)
        lay.addWidget(self._info_lbl)

        self._fig = Figure(figsize=(8.5, 4.2), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._fig)
        lay.addWidget(self._canvas, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        export_btn = QPushButton("Export PNG")
        export_btn.setDefault(False)
        export_btn.setAutoDefault(False)
        export_btn.clicked.connect(self._export_png)
        close_btn = QPushButton("Close")
        close_btn.setDefault(False)
        close_btn.setAutoDefault(False)
        close_btn.clicked.connect(self.close)
        row.addWidget(export_btn)
        row.addWidget(close_btn)
        lay.addLayout(row)

    def _draw(self) -> None:
        self._fig.clear()
        bg = self._theme.get("bg", "#ffffff")
        fg = self._theme.get("fg", "#111111")
        self._fig.patch.set_facecolor(bg)
        ax_mask, ax_fft = self._fig.subplots(1, 2)
        for ax in (ax_mask, ax_fft):
            ax.set_facecolor(bg)
            ax.tick_params(colors=fg, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(fg)

        ax_mask.imshow(self._mask, cmap="gray_r", origin="upper", interpolation="nearest")
        ax_mask.set_title("Binary point mask", color=fg, fontsize=10)
        ax_mask.set_xlabel("x pixel", color=fg, fontsize=9)
        ax_mask.set_ylabel("y pixel", color=fg, fontsize=9)

        disp = np.log1p(np.asarray(self._result.fft_magnitude, dtype=np.float64))
        qx = self._result.qx
        qy = self._result.qy
        if qx is not None and qy is not None:
            extent = [float(qx[0]), float(qx[-1]), float(qy[-1]), float(qy[0])]
            ax_fft.imshow(disp, cmap="magma", origin="upper", extent=extent, aspect="auto")
            unit = self._result.units or "frequency units"
            ax_fft.set_xlabel(f"q_x ({unit})", color=fg, fontsize=9)
            ax_fft.set_ylabel(f"q_y ({unit})", color=fg, fontsize=9)
        else:
            ax_fft.imshow(disp, cmap="magma", origin="upper", aspect="auto")
            ax_fft.set_xlabel("q_x", color=fg, fontsize=9)
            ax_fft.set_ylabel("q_y", color=fg, fontsize=9)
        ax_fft.set_title("log(1 + |FFT(mask)|)", color=fg, fontsize=10)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _info_text(self) -> str:
        unit = self._result.units or "cycles/pixel"
        return (
            f"Derived binary point mask: {self._result.n_points} detected points, "
            f"radius {self._result.radius_px} px, FFT axes in {unit}."
        )

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export point-mask FFT PNG",
            str(Path.home() / "probeflow_point_mask_fft.png"),
            "PNG files (*.png)",
        )
        if path:
            self._fig.savefig(path, dpi=160, bbox_inches="tight")
