"""Compact inline measurement results table for the Measure tab sidebar."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from probeflow.analysis.measurements import MeasurementResult, result_to_text, results_to_csv


class MeasurementResultsPanel(QWidget):
    """Compact table of MeasurementResult records.

    Designed to live inline in the Measure tab sidebar. Accepts results from
    any tool that produces a probeflow.analysis.measurements.MeasurementResult.
    """

    _HEADERS = ["ID", "Kind", "Source", "Summary", "Notes"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: list[MeasurementResult] = []
        self._build()

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_result(self, result: MeasurementResult) -> None:
        """Append one measurement result row."""
        self._results.append(result)
        self._append_row(result)
        self._table.scrollToBottom()

    def result_count(self) -> int:
        return len(self._results)

    def clear_all(self) -> None:
        self._results.clear()
        self._table.setRowCount(0)

    def selected_results(self) -> list[MeasurementResult]:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        return [self._results[r] for r in rows if 0 <= r < len(self._results)]

    def clear_selected(self) -> None:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self._results):
                del self._results[r]
                self._table.removeRow(r)

    def copy_selected(self) -> None:
        sel = self.selected_results()
        if sel:
            QApplication.clipboard().setText("\n\n".join(result_to_text(r) for r in sel))

    def export_csv(self) -> None:
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export measurements CSV",
            str(Path.home() / "probeflow_measurements.csv"),
            "CSV files (*.csv)",
        )
        if path:
            Path(path).write_text(results_to_csv(self._results), encoding="utf-8")

    # ── Build ──────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        header_lbl = QLabel("Results")
        header_lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(header_lbl)

        self._table = QTableWidget(0, len(self._HEADERS))
        self._table.setHorizontalHeaderLabels(self._HEADERS)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.ExtendedSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(100)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        lay.addWidget(self._table, 1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(4)
        for label, callback in [
            ("Copy", self.copy_selected),
            ("Export CSV", self.export_csv),
            ("Clear sel.", self.clear_selected),
            ("Clear all", self.clear_all),
        ]:
            btn = QPushButton(label)
            btn.setDefault(False)
            btn.setAutoDefault(False)
            btn.clicked.connect(callback)
            btn_row.addWidget(btn)
        lay.addLayout(btn_row)

    def _append_row(self, result: MeasurementResult) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        for col, text in enumerate([
            result.id,
            result.kind,
            result.source,
            result.summary,
            result.notes,
        ]):
            item = QTableWidgetItem(str(text))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, col, item)
        self._table.resizeColumnToContents(0)
