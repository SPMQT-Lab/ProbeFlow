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
    QVBoxLayout,
    QWidget,
)

from probeflow.measurements.export import (
    measurements_to_csv,
    measurements_to_tsv,
    measurements_to_json,
)
from probeflow.measurements.models import MeasurementResult, measurement_main_value


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
        lay.addWidget(self._table)

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

    def _append_row(self, result: MeasurementResult) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        key, value, unit = measurement_main_value(result)
        main_value = "" if value is None else f"{key}={_fmt_value(value)}"
        values = [
            result.measurement_id,
            result.kind,
            result.source_label,
            result.channel or "",
            main_value,
            unit or result.y_unit or result.x_unit or "",
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


def _fmt_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
