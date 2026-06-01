"""Small modeless viewer for generated 2-D arrays.

This is intentionally lighter than :class:`ImageViewerDialog`: generated arrays
do not always have a backing file or browse entry, but users still need a real
window for inspecting and exporting results such as ROI inverse-FFT crops.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

from probeflow.gui.image_canvas import ImageCanvas
from probeflow.gui.rendering import pil_to_pixmap, render_scan_image


class ArrayImageDialog(QDialog):
    """Display and export a generated floating-point image array."""

    def __init__(
        self,
        arr: np.ndarray,
        *,
        scan_range_m: tuple[float, float] | None = None,
        title: str = "Generated image",
        colormap: str = "gray",
        theme: dict | None = None,
        provenance: dict | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError("ArrayImageDialog expects a 2-D array")
        self._arr = a.copy()
        self._scan_range_m = (
            (float(scan_range_m[0]), float(scan_range_m[1]))
            if scan_range_m is not None else None
        )
        self._colormap = colormap
        self._theme = theme or {}
        self._provenance = dict(provenance or {})
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.setWindowTitle(title)
        self.resize(900, 720)
        self._build(title)

    def _build(self, title: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        heading = QLabel(title)
        heading.setFont(QFont("Helvetica", 11, QFont.Bold))
        heading.setAlignment(Qt.AlignCenter)
        root.addWidget(heading)

        info = QLabel(self._info_text())
        info.setFont(QFont("Helvetica", 9))
        info.setAlignment(Qt.AlignCenter)
        root.addWidget(info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignCenter)
        self._canvas = ImageCanvas()
        self._canvas.set_source(self._pixmap(), reset_zoom=True)
        scroll.setWidget(self._canvas)
        root.addWidget(scroll, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        export_btn = QPushButton("Export data...")
        export_btn.clicked.connect(self._on_export)
        row.addWidget(export_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        row.addWidget(close_btn)
        root.addLayout(row)

    def _info_text(self) -> str:
        ny, nx = self._arr.shape
        parts = [f"Image: {nx} x {ny} px"]
        if self._scan_range_m is not None:
            w_nm = self._scan_range_m[0] * 1e9
            h_nm = self._scan_range_m[1] * 1e9
            parts.append(f"Size: {w_nm:.4g} x {h_nm:.4g} nm")
        op = self._provenance.get("op")
        if op:
            parts.append(f"Source: {op}")
        return "    ".join(parts)

    def _pixmap(self):
        img = render_scan_image(
            arr=self._arr,
            colormap=self._colormap,
            clip_low=1.0,
            clip_high=99.0,
            size=None,
        )
        if img is None:
            raise ValueError("Could not render generated image")
        return pil_to_pixmap(img)

    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export generated image",
            str(Path.home() / "probeflow_generated_image.csv"),
            "CSV (*.csv);;NumPy (*.npy)",
        )
        if not path:
            return
        if path.lower().endswith(".npy"):
            np.save(path, self._arr)
        else:
            np.savetxt(path, np.nan_to_num(self._arr), delimiter=",")
