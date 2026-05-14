"""Compact controls for image feature-maxima detection."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class FeatureDetectionPanel(QWidget):
    """Small, reusable panel for ROI-scoped maxima detection and point export."""

    detectRequested = Signal()
    copyPointsRequested = Signal()
    exportCsvRequested = Signal()
    exportJsonRequested = Signal()
    clearRequested = Signal()

    _MODE_LABELS = [
        ("Percentile", "percentile"),
        ("Absolute", "absolute"),
        ("Mean + offset", "mean_offset"),
        ("Median + offset", "median_offset"),
        ("Mean + n sigma", "mean_std"),
        ("Median + n sigma", "median_std"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self.set_points_count(0)

    def settings(self) -> dict[str, object]:
        """Return current detection settings as plain values."""
        max_peaks = int(self._max_peaks_spin.value())
        smoothing_sigma = float(self._smooth_spin.value())
        return {
            "threshold_mode": self._mode_cb.currentData() or "percentile",
            "threshold_value": float(self._threshold_spin.value()),
            "min_distance_px": int(self._distance_spin.value()),
            "smoothing_sigma": smoothing_sigma if smoothing_sigma > 0.0 else None,
            "max_peaks": max_peaks if max_peaks > 0 else None,
            "exclude_border": int(self._border_spin.value()),
        }

    def set_points_count(self, count: int, *, roi_name: str | None = None) -> None:
        """Update status text and export-button state."""
        n = int(max(0, count))
        label = f"Detected points: {n}"
        if roi_name:
            label += f" ({roi_name})"
        self._status_lbl.setText(label)
        has_points = n > 0
        self._copy_btn.setEnabled(has_points)
        self._csv_btn.setEnabled(has_points)
        self._json_btn.setEnabled(has_points)
        self._clear_btn.setEnabled(has_points)

    def show_message(self, message: str) -> None:
        self._status_lbl.setText(str(message))

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        title = QLabel("Feature maxima")
        title.setStyleSheet("font-weight: 600;")
        lay.addWidget(title)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(3)

        self._mode_cb = QComboBox()
        for label, value in self._MODE_LABELS:
            self._mode_cb.addItem(label, value)
        self._mode_cb.setCurrentText("Percentile")
        self._mode_cb.setToolTip("How the local-maxima height threshold is calculated.")

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(-1.0e12, 1.0e12)
        self._threshold_spin.setDecimals(3)
        self._threshold_spin.setValue(95.0)
        self._threshold_spin.setToolTip("Threshold value for the selected mode.")

        self._distance_spin = QSpinBox()
        self._distance_spin.setRange(1, 500)
        self._distance_spin.setValue(2)
        self._distance_spin.setSuffix(" px")
        self._distance_spin.setToolTip("Minimum spacing between accepted maxima.")

        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setRange(0.0, 100.0)
        self._smooth_spin.setDecimals(2)
        self._smooth_spin.setSingleStep(0.25)
        self._smooth_spin.setValue(0.0)
        self._smooth_spin.setToolTip("Optional Gaussian sigma used on a copy for detection only.")

        self._max_peaks_spin = QSpinBox()
        self._max_peaks_spin.setRange(0, 1_000_000)
        self._max_peaks_spin.setValue(0)
        self._max_peaks_spin.setToolTip("0 keeps all accepted maxima.")

        self._border_spin = QSpinBox()
        self._border_spin.setRange(0, 10_000)
        self._border_spin.setValue(0)
        self._border_spin.setSuffix(" px")
        self._border_spin.setToolTip("Ignore maxima this many pixels from the image border.")

        for row, (label, widget) in enumerate([
            ("Mode", self._mode_cb),
            ("Value", self._threshold_spin),
            ("Spacing", self._distance_spin),
            ("Smooth", self._smooth_spin),
            ("Max", self._max_peaks_spin),
            ("Border", self._border_spin),
        ]):
            lbl = QLabel(label)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(widget, row, 1)
        lay.addLayout(grid)

        self._detect_btn = QPushButton("Detect maxima")
        self._detect_btn.setDefault(False)
        self._detect_btn.setAutoDefault(False)
        self._detect_btn.clicked.connect(self.detectRequested)
        lay.addWidget(self._detect_btn)

        export_row = QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        self._copy_btn = QPushButton("Copy points")
        self._csv_btn = QPushButton("Export CSV")
        self._json_btn = QPushButton("Export JSON")
        self._clear_btn = QPushButton("Clear")
        for button, signal in [
            (self._copy_btn, self.copyPointsRequested),
            (self._csv_btn, self.exportCsvRequested),
            (self._json_btn, self.exportJsonRequested),
            (self._clear_btn, self.clearRequested),
        ]:
            button.setDefault(False)
            button.setAutoDefault(False)
            button.clicked.connect(signal)
            export_row.addWidget(button)
        lay.addLayout(export_row)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        lay.addWidget(self._status_lbl)


class PointMaskFFTPanel(QWidget):
    """Controls for derived point masks and FFTs from detected maxima."""

    exportMaskCsvRequested = Signal()
    computeFftRequested = Signal()
    exportFftCsvRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self.set_points_available(False)

    def mask_settings(self) -> dict[str, object]:
        """Return point-mask settings as plain values."""
        return {
            "radius_px": int(self._mask_radius_spin.value()),
            "shape_mode": self._mask_shape_cb.currentData() or "disk",
        }

    def set_points_available(self, available: bool) -> None:
        has_points = bool(available)
        self._mask_csv_btn.setEnabled(has_points)
        self._fft_btn.setEnabled(has_points)
        self._fft_csv_btn.setEnabled(has_points)
        self._status_lbl.setText(
            "Uses the latest detected maxima." if has_points
            else "Detect feature maxima before creating a point mask or FFT."
        )

    def show_message(self, message: str) -> None:
        self._status_lbl.setText(str(message))

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        title = QLabel("Point mask / FFT")
        title.setStyleSheet("font-weight: 600;")
        lay.addWidget(title)

        intro = QLabel("Create a binary mask from detected maxima and inspect its FFT.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        mask_grid = QGridLayout()
        mask_grid.setContentsMargins(0, 0, 0, 0)
        mask_grid.setHorizontalSpacing(4)
        mask_grid.setVerticalSpacing(3)

        self._mask_radius_spin = QSpinBox()
        self._mask_radius_spin.setRange(0, 100)
        self._mask_radius_spin.setValue(0)
        self._mask_radius_spin.setSuffix(" px")
        self._mask_radius_spin.setToolTip(
            "Dilation radius used only for the derived binary point mask."
        )

        self._mask_shape_cb = QComboBox()
        self._mask_shape_cb.addItem("Disk", "disk")
        self._mask_shape_cb.addItem("Square", "square")
        self._mask_shape_cb.setToolTip("Shape used when expanding detected points in the mask.")

        for row, (label, widget) in enumerate([
            ("Radius", self._mask_radius_spin),
            ("Shape", self._mask_shape_cb),
        ]):
            mask_grid.addWidget(QLabel(label), row, 0)
            mask_grid.addWidget(widget, row, 1)
        lay.addLayout(mask_grid)

        mask_row = QHBoxLayout()
        mask_row.setContentsMargins(0, 0, 0, 0)
        self._mask_csv_btn = QPushButton("Export mask")
        self._fft_btn = QPushButton("FFT mask")
        self._fft_csv_btn = QPushButton("Export FFT")
        for button, signal in [
            (self._mask_csv_btn, self.exportMaskCsvRequested),
            (self._fft_btn, self.computeFftRequested),
            (self._fft_csv_btn, self.exportFftCsvRequested),
        ]:
            button.setDefault(False)
            button.setAutoDefault(False)
            button.clicked.connect(signal)
            mask_row.addWidget(button)
        lay.addLayout(mask_row)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        lay.addWidget(self._status_lbl)
