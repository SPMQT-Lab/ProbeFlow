"""Shear dialog — apply a 2-component shear transform to the current image."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class ShearDialog(QDialog):
    """Modal dialog to apply a 2-component shear correction.

    Applies the shear matrix ``[[1, shear_x], [shear_y, 1]]`` via affine
    resampling with canvas expansion.

    Signals
    -------
    applied(dict):
        Emitted when the user clicks **Apply** with keys
        ``"shear_x"`` (float), ``"shear_y"`` (float), and
        ``"interpolation"`` (str).
    """

    applied: Signal = Signal(dict)

    def __init__(self, *, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Shear")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMinimumWidth(320)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        info_lbl = QLabel(
            "Applies the shear matrix [[1, shear_x], [shear_y, 1]] via affine "
            "resampling.  The canvas expands to avoid clipping."
        )
        info_lbl.setWordWrap(True)

        def _spin() -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(-5.0, 5.0)
            s.setSingleStep(0.01)
            s.setDecimals(4)
            s.setValue(0.0)
            return s

        self._shear_x_spin = _spin()
        self._shear_y_spin = _spin()
        form.addRow("Shear X (horizontal):", self._shear_x_spin)
        form.addRow("Shear Y (vertical):", self._shear_y_spin)

        self._interp_cb = QComboBox()
        self._interp_cb.addItem("Bilinear (recommended)", "bilinear")
        self._interp_cb.addItem("Nearest", "nearest")
        self._interp_cb.addItem("Bicubic", "bicubic")
        form.addRow("Interpolation:", self._interp_cb)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        apply_btn = QPushButton("Apply")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._do_apply)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(close_btn)

        root = QVBoxLayout(self)
        root.addWidget(info_lbl)
        root.addLayout(form)
        root.addLayout(btn_row)

    def _do_apply(self) -> None:
        params = {
            "shear_x": self._shear_x_spin.value(),
            "shear_y": self._shear_y_spin.value(),
            "interpolation": self._interp_cb.currentData(),
        }
        self.applied.emit(params)
        self.accept()
