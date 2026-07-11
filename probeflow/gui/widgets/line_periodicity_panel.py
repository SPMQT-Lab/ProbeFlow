"""Controls panel for the line-profile periodicity tool."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LinePeriodicityPanel(QWidget):
    """Settings, action button, and result display for periodicity estimation."""

    findPeriodicityRequested = Signal()
    copyResultRequested = Signal()
    exportProfileCsvRequested = Signal()
    saveStructureRequested = Signal()

    _METHODS = [
        ("Autocorrelation", "autocorrelation"),
        ("Peak spacing", "peak_spacing"),
        ("FFT", "fft"),
    ]
    _BACKGROUNDS = [
        ("Linear", "linear"),
        ("None", "none"),
        ("Polynomial 2", "polynomial_2"),
        ("Moving average", "moving_average"),
    ]
    _SMOOTHINGS = [
        ("Light Gaussian", "light_gaussian"),
        ("None", "none"),
        ("Savitzky-Golay", "savitzky_golay"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def settings(self) -> dict[str, object]:
        """Return current UI settings as plain values."""
        min_val = self._min_period_spin.value()
        max_val = self._max_period_spin.value()
        return {
            "method": self._method_cb.currentData() or "autocorrelation",
            "background": self._background_cb.currentData() or "linear",
            "smoothing": self._smoothing_cb.currentData() or "light_gaussian",
            "width_px": float(self._width_spin.value()),
            "min_period_m": min_val * 1e-10 if min_val > 0 else None,
            "max_period_m": max_val * 1e-10 if max_val > 0 else None,
        }

    def set_result(self, result) -> None:
        """Update the result display after a successful run."""
        from probeflow.analysis.line_periodicity import format_period

        if math.isnan(result.period_m):
            period_text = "N/A"
        else:
            val, unit = format_period(result.period_m)
            if result.uncertainty_m is not None and not math.isnan(result.uncertainty_m):
                scale = 1e10 if unit == "Å" else 1e9
                unc = result.uncertainty_m * scale
                period_text = f"{val} ± {unc:.2g} {unit}"
            else:
                period_text = f"{val} {unit}"

        length_nm = result.line_length_m * 1e9
        lines = [
            f"Period: {period_text}",
            f"Line length: {length_nm:.3g} nm",
            f"Periods sampled: {result.n_periods:.1f}",
            f"Method: {result.method}",
            f"Quality: {result.quality}",
        ]
        self._result_lbl.setText("\n".join(lines))
        has_result = not math.isnan(result.period_m)
        self._copy_btn.setEnabled(has_result)
        self._export_btn.setEnabled(has_result)
        self._structure_btn.setEnabled(has_result)

    def show_message(self, message: str) -> None:
        self._result_lbl.setText(str(message))
        if hasattr(self, "_structure_btn"):
            self._structure_btn.setEnabled(False)

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        title = QLabel("Line periodicity")
        title.setStyleSheet("font-weight: 600;")
        lay.addWidget(title)

        desc = QLabel(
            "Extract the repeat distance from the active line ROI profile "
            "using autocorrelation, peak spacing, or FFT."
        )
        desc.setWordWrap(True)
        lay.addWidget(desc)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)

        self._method_cb = QComboBox()
        for label, key in self._METHODS:
            self._method_cb.addItem(label, key)
        form.addRow("Method:", self._method_cb)

        self._background_cb = QComboBox()
        for label, key in self._BACKGROUNDS:
            self._background_cb.addItem(label, key)
        form.addRow("Background:", self._background_cb)

        self._smoothing_cb = QComboBox()
        for label, key in self._SMOOTHINGS:
            self._smoothing_cb.addItem(label, key)
        form.addRow("Smoothing:", self._smoothing_cb)

        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(1.0, 100.0)
        self._width_spin.setDecimals(0)
        self._width_spin.setSingleStep(1.0)
        self._width_spin.setValue(1.0)
        self._width_spin.setSuffix(" px")
        form.addRow("Profile width:", self._width_spin)

        self._min_period_spin = QDoubleSpinBox()
        self._min_period_spin.setRange(0.0, 10000.0)
        self._min_period_spin.setDecimals(2)
        self._min_period_spin.setValue(0.0)
        self._min_period_spin.setSuffix(" Å")
        self._min_period_spin.setSpecialValueText("Auto")
        form.addRow("Min period:", self._min_period_spin)

        self._max_period_spin = QDoubleSpinBox()
        self._max_period_spin.setRange(0.0, 10000.0)
        self._max_period_spin.setDecimals(2)
        self._max_period_spin.setValue(0.0)
        self._max_period_spin.setSuffix(" Å")
        self._max_period_spin.setSpecialValueText("Auto")
        form.addRow("Max period:", self._max_period_spin)

        lay.addLayout(form)

        find_btn = QPushButton("Find periodicity")
        find_btn.setDefault(False)
        find_btn.setAutoDefault(False)
        find_btn.clicked.connect(self.findPeriodicityRequested.emit)
        lay.addWidget(find_btn)

        self._result_lbl = QLabel("")
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setTextInteractionFlags(
            self._result_lbl.textInteractionFlags()
            | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        lay.addWidget(self._result_lbl)

        action_row = QHBoxLayout()
        self._copy_btn = QPushButton("Copy result")
        self._copy_btn.setEnabled(False)
        self._copy_btn.setDefault(False)
        self._copy_btn.setAutoDefault(False)
        self._copy_btn.clicked.connect(self.copyResultRequested.emit)
        self._export_btn = QPushButton("Export profile…")
        self._export_btn.setEnabled(False)
        self._export_btn.setDefault(False)
        self._export_btn.setAutoDefault(False)
        self._export_btn.clicked.connect(self.exportProfileCsvRequested.emit)
        self._structure_btn = QPushButton("Save as structure…")
        self._structure_btn.setEnabled(False)
        self._structure_btn.setDefault(False)
        self._structure_btn.setAutoDefault(False)
        self._structure_btn.setToolTip("Save this period as a reusable known lattice spacing.")
        self._structure_btn.clicked.connect(self.saveStructureRequested.emit)
        action_row.addWidget(self._copy_btn)
        action_row.addWidget(self._export_btn)
        action_row.addWidget(self._structure_btn)
        lay.addLayout(action_row)

        lay.addStretch(1)
