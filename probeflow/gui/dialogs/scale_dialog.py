"""Scale dialog — resample the image to new pixel dimensions."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class ScaleDialog(QDialog):
    """Modal dialog to rescale the image to new pixel dimensions.

    The physical scan range is unchanged; only the pixel count (and therefore
    pixel size) changes.

    Signals
    -------
    applied(dict):
        Emitted when the user clicks **Apply** with keys
        ``"new_width"`` (int), ``"new_height"`` (int), and ``"order"`` (int).
    """

    applied: Signal = Signal(dict)

    def __init__(
        self,
        current_shape: tuple[int, int],
        *,
        scan_range_m: tuple[float, float] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scale Image")
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setMinimumWidth(320)

        Ny, Nx = current_shape
        self._orig_Ny = Ny
        self._orig_Nx = Nx
        self._updating = False

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)

        # Current size info
        curr_lbl = QLabel(f"{Nx} × {Ny} px")
        form.addRow("Current size:", curr_lbl)

        if scan_range_m is not None and Nx > 0 and Ny > 0:
            dx_pm = (scan_range_m[0] / Nx) * 1e12
            dy_pm = (scan_range_m[1] / Ny) * 1e12
            px_lbl = QLabel(f"{dx_pm:.3g} × {dy_pm:.3g} pm/px")
            form.addRow("Pixel size:", px_lbl)

        # Width / height spinboxes
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 16384)
        self._width_spin.setValue(Nx)
        self._height_spin = QSpinBox()
        self._height_spin.setRange(1, 16384)
        self._height_spin.setValue(Ny)

        form.addRow("Width (px):", self._width_spin)
        form.addRow("Height (px):", self._height_spin)

        # Lock aspect ratio
        self._lock_cb = QCheckBox("Lock aspect ratio")
        self._lock_cb.setChecked(True)
        form.addRow("", self._lock_cb)

        self._width_spin.valueChanged.connect(self._on_width_changed)
        self._height_spin.valueChanged.connect(self._on_height_changed)

        # Interpolation
        self._interp_cb = QComboBox()
        self._interp_cb.addItem("Bilinear (recommended)", 1)
        self._interp_cb.addItem("Nearest neighbour", 0)
        self._interp_cb.addItem("Bicubic", 3)
        form.addRow("Interpolation:", self._interp_cb)

        # Buttons
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
        root.addLayout(form)
        root.addLayout(btn_row)

    def _on_width_changed(self, new_w: int) -> None:
        if self._updating or not self._lock_cb.isChecked():
            return
        if self._orig_Nx <= 0:
            return
        self._updating = True
        try:
            ratio = self._orig_Ny / self._orig_Nx
            self._height_spin.setValue(max(1, round(new_w * ratio)))
        finally:
            self._updating = False

    def _on_height_changed(self, new_h: int) -> None:
        if self._updating or not self._lock_cb.isChecked():
            return
        if self._orig_Ny <= 0:
            return
        self._updating = True
        try:
            ratio = self._orig_Nx / self._orig_Ny
            self._width_spin.setValue(max(1, round(new_h * ratio)))
        finally:
            self._updating = False

    def _do_apply(self) -> None:
        params = {
            "new_width": self._width_spin.value(),
            "new_height": self._height_spin.value(),
            "order": self._interp_cb.currentData(),
        }
        self.applied.emit(params)
        self.accept()
