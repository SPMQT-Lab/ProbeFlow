"""Threshold dialog — clip or binarize a scan plane by height value."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class ThresholdDialog(QDialog):
    """Modeless dialog to apply a data threshold to the current image.

    Modes
    -----
    *Clip*: values outside ``[lower, upper]`` become NaN.
    *Binarize*: pixels inside the range become 1.0, outside 0.0.

    Signals
    -------
    applied(dict):
        Emitted when the user clicks **Apply**.  The dict has keys
        ``"mode"`` (str), and optionally ``"lower"`` and ``"upper"`` (float).
    """

    applied: Signal = Signal(dict)

    def __init__(
        self,
        arr: np.ndarray,
        *,
        preview_fn: "Callable | None" = None,
        clear_preview_fn: "Callable | None" = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Threshold")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMinimumWidth(340)

        self._arr = arr
        self._preview_fn = preview_fn
        self._clear_preview_fn = clear_preview_fn

        # Compute data range for informational label
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            vmin, vmax = float(finite.min()), float(finite.max())
        else:
            vmin, vmax = 0.0, 1.0

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        # Data range info
        range_lbl = QLabel(f"{vmin:.5g}  …  {vmax:.5g}")
        range_lbl.setToolTip("Finite data range of the current image")
        form.addRow("Data range:", range_lbl)

        # Mode selector
        self._mode_cb = QComboBox()
        self._mode_cb.addItem("Clip (set out-of-range to NaN)", "clip")
        self._mode_cb.addItem("Binarize (0 outside, 1 inside)", "binarize")
        form.addRow("Mode:", self._mode_cb)

        # Lower bound
        lower_row = QHBoxLayout()
        self._lower_check = QCheckBox("Enable")
        self._lower_spin = QDoubleSpinBox()
        self._lower_spin.setRange(-1e15, 1e15)
        self._lower_spin.setDecimals(6)
        self._lower_spin.setValue(vmin)
        self._lower_spin.setEnabled(False)
        self._lower_check.toggled.connect(self._lower_spin.setEnabled)
        lower_row.addWidget(self._lower_check)
        lower_row.addWidget(self._lower_spin, 1)
        form.addRow("Lower bound:", lower_row)

        # Upper bound
        upper_row = QHBoxLayout()
        self._upper_check = QCheckBox("Enable")
        self._upper_spin = QDoubleSpinBox()
        self._upper_spin.setRange(-1e15, 1e15)
        self._upper_spin.setDecimals(6)
        self._upper_spin.setValue(vmax)
        self._upper_spin.setEnabled(True)
        self._upper_check.setChecked(True)
        self._upper_check.toggled.connect(self._upper_spin.setEnabled)
        upper_row.addWidget(self._upper_check)
        upper_row.addWidget(self._upper_spin, 1)
        form.addRow("Upper bound:", upper_row)

        # Buttons
        btn_row = QHBoxLayout()
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._do_preview)
        self._preview_btn.setEnabled(preview_fn is not None)
        apply_btn = QPushButton("Apply")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._do_apply)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(self._preview_btn)
        btn_row.addStretch()
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(close_btn)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addLayout(btn_row)

    def _current_params(self) -> dict:
        params: dict = {
            "mode": self._mode_cb.currentData(),
        }
        if self._lower_check.isChecked():
            params["lower"] = self._lower_spin.value()
        if self._upper_check.isChecked():
            params["upper"] = self._upper_spin.value()
        return params

    def _do_preview(self) -> None:
        if self._preview_fn is None:
            return
        from probeflow.processing.geometry import threshold_image
        params = self._current_params()
        result = threshold_image(
            self._arr,
            lower=params.get("lower"),
            upper=params.get("upper"),
            mode=params.get("mode", "clip"),
        )
        self._preview_fn(result)

    def _do_apply(self) -> None:
        params = self._current_params()
        self.applied.emit(params)
        self.close()

    def closeEvent(self, event) -> None:
        if self._clear_preview_fn is not None:
            self._clear_preview_fn()
        super().closeEvent(event)
