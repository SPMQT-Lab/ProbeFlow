"""Reusable measurement results table widget."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from probeflow.measurements.export import (
    measurements_to_csv,
    measurements_to_tsv,
    measurements_to_json,
)
from probeflow.measurements.models import MeasurementResult, measurement_main_value

KIND_LABELS: dict[str, str] = {
    "feature_maxima": "Feature maxima",
    "point_fft": "Point mask FFT",
    "roi_stats": "ROI statistics",
    "step_height": "Step height",
    "line_profile": "Line profile",
    "spectrum_delta": "Spectrum Δ",
}

VALUE_LABELS: dict[str, str] = {
    "mean_height": "Mean height",
    "median_height": "Median height",
    "std_height": "Std height",
    "rms_roughness": "RMS roughness",
    "height_difference": "Height difference",
    "length": "Length",
    "n_points": "Number of points",
    "dominant_frequency": "Dominant frequency",
    "dx": "Δx",
    "dy": "Δy",
}


class MeasurementResultsTable(QWidget):
    """Small table for measured values that can be copied or exported."""

    _HEADERS = ["ID", "Kind", "Source", "Channel", "Main value", "Units", "Notes"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: list[MeasurementResult] = []
        self._next_index = 1
        self._build()

    def next_measurement_id(self) -> str:
        """Return the next stable measurement identifier for this table."""
        measurement_id = f"M{self._next_index:04d}"
        self._next_index += 1
        return measurement_id

    def add_result(self, result: MeasurementResult) -> None:
        """Append one measurement result."""
        self._results.append(result)
        self._append_row(result)

    def results(self) -> list[MeasurementResult]:
        """Return all table results."""
        return list(self._results)

    def selected_results(self) -> list[MeasurementResult]:
        """Return selected table results."""
        rows = sorted({index.row() for index in self._table.selectedIndexes()})
        return [self._results[row] for row in rows if 0 <= row < len(self._results)]

    def clear_results(self) -> None:
        """Clear all rows."""
        self._results.clear()
        self._table.setRowCount(0)
        self._details.clear()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._table = QTableWidget(0, len(self._HEADERS))
        self._table.setHorizontalHeaderLabels(self._HEADERS)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.ExtendedSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(110)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        lay.addWidget(self._table)

        self._details = QTextEdit()
        self._details.setReadOnly(True)
        self._details.setMinimumHeight(80)
        self._details.setMaximumHeight(160)
        self._details.setPlaceholderText("Select a row to see full details.")
        lay.addWidget(self._details)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        for idx, (label, callback) in enumerate([
            ("Copy selected", self.copy_selected),
            ("Copy all", self.copy_all),
            ("Export CSV", self.export_csv),
            ("Export JSON", self.export_json),
            ("Clear selected", self.clear_selected),
            ("Clear all", self.clear_results),
        ]):
            button = QPushButton(label)
            button.setDefault(False)
            button.setAutoDefault(False)
            button.clicked.connect(callback)
            grid.addWidget(button, idx // 3, idx % 3)
        lay.addLayout(grid)

    def _on_selection_changed(self) -> None:
        rows = sorted({index.row() for index in self._table.selectedIndexes()})
        if len(rows) == 1 and 0 <= rows[0] < len(self._results):
            self._details.setPlainText(_format_details(self._results[rows[0]]))
        else:
            self._details.clear()

    def _append_row(self, result: MeasurementResult) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        key, value, unit = measurement_main_value(result)
        main_value = "" if value is None else f"{VALUE_LABELS.get(key, key)}={_fmt_value(value)}"
        values = [
            result.measurement_id,
            KIND_LABELS.get(result.kind, result.kind),
            result.source_label,
            result.channel or "",
            main_value,
            unit or result.z_unit or result.y_unit or result.x_unit or "",
            result.notes,
        ]
        for col, text in enumerate(values):
            item = QTableWidgetItem(str(text))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()

    def copy_selected(self) -> None:
        """Copy selected rows as TSV."""
        self._copy_results(self.selected_results())

    def copy_all(self) -> None:
        """Copy all rows as TSV."""
        self._copy_results(self._results)

    def export_csv(self) -> None:
        """Export all measurements as CSV."""
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export measurements CSV",
            str(Path.home() / "probeflow_measurements.csv"),
            "CSV files (*.csv)",
        )
        if path:
            measurements_to_csv(self._results, path)

    def export_json(self) -> None:
        """Export all measurements as JSON."""
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export measurements JSON",
            str(Path.home() / "probeflow_measurements.json"),
            "JSON files (*.json)",
        )
        if path:
            measurements_to_json(self._results, path)

    def clear_selected(self) -> None:
        """Remove selected rows."""
        rows = sorted({index.row() for index in self._table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for row in rows:
            if 0 <= row < len(self._results):
                del self._results[row]
                self._table.removeRow(row)

    def _copy_results(self, results: list[MeasurementResult]) -> None:
        if not results:
            return
        QApplication.clipboard().setText(measurements_to_tsv(results))


def _format_details(result: MeasurementResult) -> str:
    lines: list[str] = []
    lines.append(f"{result.measurement_id}  —  {KIND_LABELS.get(result.kind, result.kind)}")
    if result.source_label:
        lines.append(f"Source: {result.source_label}")
    if result.source_path and result.source_path != result.source_label:
        lines.append(f"File:   {result.source_path}")
    if result.channel:
        lines.append(f"Channel: {result.channel}")
    unit_parts = []
    if result.x_unit:
        unit_parts.append(f"x={result.x_unit}")
    if result.y_unit and result.y_unit != result.x_unit:
        unit_parts.append(f"y={result.y_unit}")
    if result.z_unit:
        unit_parts.append(f"z={result.z_unit}")
    if unit_parts:
        lines.append("Units:  " + "  ".join(unit_parts))
    if result.values:
        lines.append("Values:")
        for k, v in result.values.items():
            label = VALUE_LABELS.get(k, k)
            lines.append(f"  {label}: {_fmt_value(v)}")
    if result.context:
        ctx_items = [f"{k}={v}" for k, v in result.context.items() if v is not None]
        if ctx_items:
            lines.append("Context: " + "  ".join(ctx_items))
    if result.notes:
        lines.append(f"Notes: {result.notes}")
    return "\n".join(lines)


def _fmt_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
