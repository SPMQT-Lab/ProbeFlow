"""Image arithmetic dialog for the image viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from probeflow.core.scan_loader import load_scan


class ImageArithmeticDialog(QDialog):
    """Collect a first-pass image arithmetic operation specification."""

    WHOLE_IMAGE = "whole_image"
    ACTIVE_AREA_ROI = "active_area_roi"

    def __init__(
        self,
        entries: list[Any] | tuple[Any, ...],
        *,
        current_entry_index: int,
        current_plane_idx: int,
        current_shape: tuple[int, int] | None,
        current_scan_range_m: tuple[float, float] | None,
        display_scale: float,
        display_unit: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Image Operations")
        self.setModal(True)
        self.resize(460, 260)

        self._entries = list(entries or ())
        self._current_shape = tuple(current_shape) if current_shape is not None else None
        self._current_scan_range_m = current_scan_range_m
        self._display_scale = float(display_scale) if display_scale else 1.0
        self._display_unit = str(display_unit or "SI")
        self._scan_cache: dict[int, Any] = {}
        self._accepted_spec: dict[str, Any] | None = None
        self._accepted_scope = self.WHOLE_IMAGE

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel("Image Operations")
        title.setStyleSheet("font-weight: 700; font-size: 16px;")
        root.addWidget(title)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self._operand_type_combo = QComboBox()
        self._operand_type_combo.addItem("Constant", "constant")
        self._operand_type_combo.addItem("Image", "image")
        form.addRow("Operand:", self._operand_type_combo)

        self._operation_combo = QComboBox()
        form.addRow("Operation:", self._operation_combo)

        self._scope_combo = QComboBox()
        self._scope_combo.addItem("Whole image", self.WHOLE_IMAGE)
        self._scope_combo.addItem("Active area ROI only", self.ACTIVE_AREA_ROI)
        form.addRow("Scope:", self._scope_combo)
        root.addLayout(form)

        self._operand_stack = QStackedWidget()
        self._operand_stack.addWidget(self._constant_page())
        self._operand_stack.addWidget(self._image_page(current_entry_index, current_plane_idx))
        root.addWidget(self._operand_stack)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._apply_button = buttons.button(QDialogButtonBox.Ok)
        self._apply_button.setText("Apply")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._operand_type_combo.currentIndexChanged.connect(self._refresh_ui)
        self._operation_combo.currentIndexChanged.connect(self._refresh_ui)
        self._constant_spin.valueChanged.connect(self._refresh_ui)
        self._entry_combo.currentIndexChanged.connect(self._on_source_entry_changed)
        self._plane_combo.currentIndexChanged.connect(self._on_source_plane_changed)

        self._refresh_ui()

    def _constant_page(self) -> QWidget:
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        self._constant_spin = QDoubleSpinBox()
        self._constant_spin.setDecimals(8)
        self._constant_spin.setRange(-1.0e12, 1.0e12)
        self._constant_spin.setSingleStep(1.0)
        self._constant_spin.setValue(0.0)
        self._constant_label = QLabel("")
        row.addWidget(self._constant_spin, 1)
        row.addWidget(self._constant_label)
        return page

    def _image_page(self, current_entry_index: int, current_plane_idx: int) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)

        self._entry_combo = QComboBox()
        for idx, entry in enumerate(self._entries):
            label = getattr(entry, "stem", None) or Path(getattr(entry, "path", "")).stem
            self._entry_combo.addItem(str(label), idx)
        if self._entries:
            self._entry_combo.setCurrentIndex(
                max(0, min(int(current_entry_index), len(self._entries) - 1))
            )
        form.addRow("Image:", self._entry_combo)

        self._plane_combo = QComboBox()
        self._pending_plane_idx = max(0, int(current_plane_idx))
        form.addRow("Channel:", self._plane_combo)
        return page

    def _set_status(self, message: str, *, warning: bool = False, error: bool = False) -> None:
        color = "#5a5a5a"
        if warning:
            color = "#8a5a00"
        if error:
            color = "#9b1c1c"
        self._status_label.setStyleSheet(f"color: {color};")
        self._status_label.setText(message)

    def _operand_type(self) -> str:
        return str(self._operand_type_combo.currentData() or "constant")

    def _operation(self) -> str:
        return str(self._operation_combo.currentData() or "add")

    def _scope(self) -> str:
        return str(self._scope_combo.currentData() or self.WHOLE_IMAGE)

    def _refresh_operation_choices(self) -> None:
        operand_type = self._operand_type()
        current = self._operation()
        operations = [
            ("Add", "add"),
            ("Subtract", "subtract"),
        ]
        if operand_type == "constant":
            operations.extend([
                ("Multiply", "multiply"),
                ("Divide", "divide"),
            ])

        if [self._operation_combo.itemData(i) for i in range(self._operation_combo.count())] == [
            value for _, value in operations
        ]:
            return

        self._operation_combo.blockSignals(True)
        self._operation_combo.clear()
        for label, value in operations:
            self._operation_combo.addItem(label, value)
        index = next((i for i, (_, value) in enumerate(operations) if value == current), 0)
        self._operation_combo.setCurrentIndex(index)
        self._operation_combo.blockSignals(False)

    def _refresh_ui(self) -> None:
        self._refresh_operation_choices()
        operand_type = self._operand_type()
        self._operand_stack.setCurrentIndex(0 if operand_type == "constant" else 1)
        op = self._operation()
        if operand_type == "constant":
            if op in {"add", "subtract"}:
                self._constant_label.setText(self._display_unit)
            else:
                self._constant_label.setText("factor")
            self._refresh_constant_status()
        else:
            self._refresh_source_controls()

    def _refresh_constant_status(self) -> None:
        if self._operation() == "divide" and float(self._constant_spin.value()) == 0.0:
            self._apply_button.setEnabled(False)
            self._set_status("Divide by zero is not allowed.", error=True)
            return
        self._apply_button.setEnabled(True)
        if self._operation() in {"add", "subtract"}:
            self._set_status(
                f"Value will be stored in native SI units from {self._display_unit} input."
            )
        else:
            self._set_status("Factor is dimensionless.")

    def _selected_entry_index(self) -> int | None:
        if self._entry_combo.count() == 0:
            return None
        try:
            return int(self._entry_combo.currentData())
        except (TypeError, ValueError):
            return None

    def _selected_entry(self) -> Any | None:
        idx = self._selected_entry_index()
        if idx is None or idx < 0 or idx >= len(self._entries):
            return None
        return self._entries[idx]

    def _load_source_scan(self, entry_index: int) -> Any:
        if entry_index not in self._scan_cache:
            entry = self._entries[entry_index]
            self._scan_cache[entry_index] = load_scan(Path(entry.path))
        return self._scan_cache[entry_index]

    def _on_source_entry_changed(self) -> None:
        self._pending_plane_idx = 0
        self._refresh_source_controls()

    def _on_source_plane_changed(self) -> None:
        self._pending_plane_idx = int(self._plane_combo.currentData() or 0)
        self._refresh_source_controls()

    def _refresh_source_controls(self) -> None:
        entry_index = self._selected_entry_index()
        if entry_index is None:
            self._apply_button.setEnabled(False)
            self._set_status("No source images are available in this viewer.", error=True)
            return

        try:
            scan = self._load_source_scan(entry_index)
        except Exception as exc:
            self._apply_button.setEnabled(False)
            self._plane_combo.clear()
            self._set_status(f"Could not load source image: {exc}", error=True)
            return

        names = list(scan.plane_names) if scan.plane_names else [
            f"Channel {i + 1}" for i in range(len(scan.planes))
        ]
        existing = [
            (self._plane_combo.itemText(i), self._plane_combo.itemData(i))
            for i in range(self._plane_combo.count())
        ]
        desired = [(str(name), idx) for idx, name in enumerate(names)]
        if existing != desired:
            self._plane_combo.blockSignals(True)
            self._plane_combo.clear()
            for label, idx in desired:
                self._plane_combo.addItem(label, idx)
            if names:
                self._plane_combo.setCurrentIndex(
                    max(0, min(self._pending_plane_idx, len(names) - 1))
                )
            self._plane_combo.blockSignals(False)
        self._pending_plane_idx = self._plane_combo.currentIndex()

        plane = self._selected_source_plane()
        if plane is None:
            self._apply_button.setEnabled(False)
            self._set_status("Source image has no selectable channel.", error=True)
            return
        if self._current_shape is not None and tuple(plane.shape) != self._current_shape:
            self._apply_button.setEnabled(False)
            self._set_status(
                f"Shape mismatch: source {tuple(plane.shape)} vs current {self._current_shape}.",
                error=True,
            )
            return

        self._apply_button.setEnabled(True)
        if not self._scan_ranges_match(getattr(scan, "scan_range_m", None)):
            self._set_status(
                "Warning: source scan range differs from the current image or is missing.",
                warning=True,
            )
            return
        self._set_status("Source image shape matches the current image.")

    def _selected_source_plane(self) -> np.ndarray | None:
        entry_index = self._selected_entry_index()
        if entry_index is None:
            return None
        try:
            scan = self._load_source_scan(entry_index)
        except Exception:
            return None
        plane_idx = int(self._plane_combo.currentData() or 0)
        if plane_idx < 0 or plane_idx >= len(scan.planes):
            return None
        return np.asarray(scan.planes[plane_idx], dtype=np.float64)

    def _scan_ranges_match(self, source_range: Any) -> bool:
        if self._current_scan_range_m is None or source_range is None:
            return False
        try:
            return bool(np.allclose(
                np.asarray(source_range, dtype=float),
                np.asarray(self._current_scan_range_m, dtype=float),
                rtol=1e-6,
                atol=0.0,
            ))
        except (TypeError, ValueError):
            return False

    def _constant_params(self) -> dict[str, Any]:
        op = self._operation()
        if op in {"add", "subtract"}:
            display_value = float(self._constant_spin.value())
            return {
                "operation": op,
                "operand_type": "constant",
                "value_si": display_value / self._display_scale,
                "display_value": display_value,
                "display_unit": self._display_unit,
            }
        return {
            "operation": op,
            "operand_type": "constant",
            "factor": float(self._constant_spin.value()),
        }

    def _image_params(self) -> dict[str, Any]:
        entry = self._selected_entry()
        if entry is None:
            raise ValueError("No source image selected.")
        plane_idx = int(self._plane_combo.currentData() or 0)
        return {
            "operation": self._operation(),
            "operand_type": "image",
            "source_path": str(Path(entry.path).resolve()),
            "source_label": str(getattr(entry, "stem", Path(entry.path).stem)),
            "plane_idx": plane_idx,
            "plane_label": self._plane_combo.currentText(),
        }

    def operation_spec(self) -> dict[str, Any] | None:
        return self._accepted_spec

    def scope(self) -> str:
        return self._accepted_scope

    def accept(self) -> None:
        self._refresh_ui()
        if not self._apply_button.isEnabled():
            return
        try:
            params = (
                self._constant_params()
                if self._operand_type() == "constant" else self._image_params()
            )
        except ValueError as exc:
            self._set_status(str(exc), error=True)
            return
        self._accepted_spec = {"op": "arithmetic", "params": params}
        self._accepted_scope = self._scope()
        super().accept()
