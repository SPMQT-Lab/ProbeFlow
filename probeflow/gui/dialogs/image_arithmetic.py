"""Image arithmetic dialog for the image viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from probeflow.core.scan_loader import load_scan


class _ScanLoaderSignals(QObject):
    """Signals for ScanLoaderWorker (must be a QObject subclass)."""
    finished = Signal(int, object)   # (entry_index, scan_or_None)
    failed   = Signal(int, str)      # (entry_index, error_message)


class _ScanLoaderWorker(QRunnable):
    """Load a scan file off the GUI thread."""

    def __init__(self, entry_index: int, path: Path) -> None:
        super().__init__()
        self.setAutoDelete(True)
        self._entry_index = entry_index
        self._path = path
        self.signals = _ScanLoaderSignals()

    def run(self) -> None:
        try:
            scan = load_scan(self._path)
            self.signals.finished.emit(self._entry_index, scan)
        except Exception as exc:
            self.signals.failed.emit(self._entry_index, str(exc))


class ImageArithmeticDialog(QDialog):
    """Collect a first-pass image arithmetic operation specification."""

    WHOLE_IMAGE = "whole_image"
    ACTIVE_AREA_ROI = "active_area_roi"
    _LOADING = object()  # sentinel: scan load is in progress

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
        self._operand_type_combo.addItem("Generated pattern", "generated")
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
        self._operand_stack.addWidget(self._generated_page())
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
        self._pattern_combo.currentIndexChanged.connect(self._refresh_ui)
        self._amplitude_spin.valueChanged.connect(self._refresh_ui)
        self._period_spin.valueChanged.connect(self._refresh_ui)
        self._seed_spin.valueChanged.connect(self._refresh_ui)

        self._refresh_ui()

        # Pre-warm the cache for the currently selected entry so the plane combo
        # is populated quickly when the user first switches to the "image" operand.
        if self._entries:
            current_entry_idx = max(0, min(int(current_entry_index), len(self._entries) - 1))
            self._request_scan_load(current_entry_idx)

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

    def _generated_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)

        self._pattern_combo = QComboBox()
        for label, value in (
            ("Checkerboard", "checkerboard"),
            ("Ramp X", "ramp_x"),
            ("Ramp Y", "ramp_y"),
            ("Speckle noise", "speckle"),
            ("Impulse grid", "impulse_grid"),
        ):
            self._pattern_combo.addItem(label, value)
        form.addRow("Pattern:", self._pattern_combo)

        amp_row = QHBoxLayout()
        amp_row.setContentsMargins(0, 0, 0, 0)
        self._amplitude_spin = QDoubleSpinBox()
        self._amplitude_spin.setDecimals(8)
        self._amplitude_spin.setRange(0.0, 1.0e12)
        self._amplitude_spin.setSingleStep(1.0)
        self._amplitude_spin.setValue(1.0)
        self._amplitude_label = QLabel(self._display_unit)
        amp_row.addWidget(self._amplitude_spin, 1)
        amp_row.addWidget(self._amplitude_label)
        form.addRow("Amplitude:", amp_row)

        self._period_spin = QSpinBox()
        self._period_spin.setRange(1, 100000)
        self._period_spin.setValue(16)
        form.addRow("Period / spacing px:", self._period_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2_147_483_647)
        self._seed_spin.setValue(1)
        form.addRow("Noise seed:", self._seed_spin)
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
        self._operand_stack.setCurrentIndex({
            "constant": 0,
            "image": 1,
            "generated": 2,
        }.get(operand_type, 0))
        op = self._operation()
        if operand_type == "constant":
            if op in {"add", "subtract"}:
                self._constant_label.setText(self._display_unit)
            else:
                self._constant_label.setText("factor")
            self._refresh_constant_status()
        elif operand_type == "image":
            self._refresh_source_controls()
        else:
            self._refresh_generated_status()

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

    def _refresh_generated_status(self) -> None:
        pattern = str(self._pattern_combo.currentData() or "checkerboard")
        self._amplitude_label.setText(self._display_unit)
        self._period_spin.setEnabled(pattern in {"checkerboard", "impulse_grid"})
        self._seed_spin.setEnabled(pattern == "speckle")
        self._apply_button.setEnabled(True)
        self._set_status(
            "Generated patterns are virtual operands and will be replayed from "
            "processing history."
        )

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

    def _request_scan_load(self, entry_index: int) -> None:
        """Start a background load for entry_index if not already cached or in-flight."""
        if entry_index in self._scan_cache:
            return  # already loaded or already loading
        entry = self._entries[entry_index]
        path = Path(entry.path)
        self._scan_cache[entry_index] = self._LOADING  # mark in-flight
        worker = _ScanLoaderWorker(entry_index, path)
        worker.signals.finished.connect(self._on_scan_loaded)
        worker.signals.failed.connect(self._on_scan_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_scan_loaded(self, entry_index: int, scan: Any) -> None:
        self._scan_cache[entry_index] = scan
        # Only refresh if this is still the selected entry
        if self._selected_entry_index() == entry_index:
            self._refresh_source_controls()

    def _on_scan_failed(self, entry_index: int, error_msg: str) -> None:
        self._scan_cache.pop(entry_index, None)  # remove in-flight marker
        if self._selected_entry_index() == entry_index:
            self._apply_button.setEnabled(False)
            self._set_status(f"Could not load source image: {error_msg}", error=True)

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

        # Check cache — may be loaded, in-flight, or missing
        cached = self._scan_cache.get(entry_index)
        if cached is None:
            # Not yet requested — start async load
            self._request_scan_load(entry_index)
            self._apply_button.setEnabled(False)
            self._set_status("Loading source image…")
            return
        if cached is self._LOADING:
            # Already loading — show spinner message and wait
            self._apply_button.setEnabled(False)
            self._set_status("Loading source image…")
            return
        scan = cached

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
        cached = self._scan_cache.get(entry_index)
        if cached is None or cached is self._LOADING:
            return None
        scan = cached
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

    def _generated_params(self) -> dict[str, Any]:
        pattern = str(self._pattern_combo.currentData() or "checkerboard")
        display_amplitude = float(self._amplitude_spin.value())
        params: dict[str, Any] = {
            "operation": self._operation(),
            "operand_type": "generated",
            "pattern": pattern,
            "amplitude_si": display_amplitude / self._display_scale,
            "display_amplitude": display_amplitude,
            "display_unit": self._display_unit,
        }
        if pattern in {"checkerboard", "impulse_grid"}:
            params["period_px"] = int(self._period_spin.value())
        if pattern == "speckle":
            params["seed"] = int(self._seed_spin.value())
        return params

    def operation_spec(self) -> dict[str, Any] | None:
        return self._accepted_spec

    def scope(self) -> str:
        return self._accepted_scope

    def accept(self) -> None:
        self._refresh_ui()
        if not self._apply_button.isEnabled():
            return
        try:
            operand_type = self._operand_type()
            if operand_type == "constant":
                params = self._constant_params()
            elif operand_type == "image":
                params = self._image_params()
            else:
                params = self._generated_params()
        except ValueError as exc:
            self._set_status(str(exc), error=True)
            return
        self._accepted_spec = {"op": "arithmetic", "params": params}
        self._accepted_scope = self._scope()
        super().accept()
