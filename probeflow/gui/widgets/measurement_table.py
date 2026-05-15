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
    "line_profile_delta": "Line profile Δ",
    "spectrum_delta": "Spectrum Δ",
}

VALUE_LABELS: dict[str, str] = {
    "mean_height": "Mean height",
    "median_height": "Median height",
    "std_height": "Std height",
    "rms_roughness": "RMS roughness",
    "height_difference": "Height difference",
    "length": "Length",
    "length_px": "Length (px)",
    "n_points": "N points",
    "dominant_frequency": "Dominant frequency",
    "dx": "Δx",
    "dy": "Δy",
    "delta_x": "Δx",
    "delta_y": "Δy",
}


class MeasurementResultsTable(QWidget):
    """Small table for measured values that can be copied or exported."""

    _HEADERS = ["ID", "Kind", "Channel", "Main value", "Units", "Notes"]

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
        main_value = "" if value is None else _fmt_value(value)
        values = [
            result.measurement_id,
            KIND_LABELS.get(result.kind, result.kind),
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
    lines.append("")

    x_unit = result.x_unit or ""
    y_unit = result.z_unit or result.y_unit or ""

    if result.kind == "line_profile_delta":
        dy = result.values.get("delta_y")
        dx = result.values.get("delta_x")
        if dy is not None:
            lines.append(f"Main value: Δy = {_fmt_value(dy)} {y_unit}".rstrip())
        if dx is not None:
            lines.append(f"Δx = {_fmt_value(dx)} {x_unit}".rstrip())
        if result.channel:
            lines.append(f"Channel: {result.channel}")
        roi_name = result.context.get("roi_name")
        if roi_name:
            lines.append(f"Line ROI: {roi_name}")
        p1d = result.values.get("p1_distance")
        p1h = result.values.get("p1_height")
        p2d = result.values.get("p2_distance")
        p2h = result.values.get("p2_height")
        if p1d is not None and p2d is not None:
            lines.append(
                f"P1: {_fmt_value(p1d)} {x_unit}, {_fmt_value(p1h)} {y_unit}".rstrip()
            )
            lines.append(
                f"P2: {_fmt_value(p2d)} {x_unit}, {_fmt_value(p2h)} {y_unit}".rstrip()
            )
    elif result.kind == "line_profile":
        length = result.values.get("length")
        hdiff = result.values.get("height_difference")
        n_pts = result.values.get("n_points")
        length_px = result.values.get("length_px")
        x1 = result.values.get("x1")
        y1 = result.values.get("y1")
        x2 = result.values.get("x2")
        y2 = result.values.get("y2")
        if length is not None:
            lines.append(f"Length: {_fmt_value(length)} {x_unit}".rstrip())
        if hdiff is not None:
            lines.append(f"Height difference: {_fmt_value(hdiff)} {y_unit}".rstrip())
        if n_pts is not None:
            lines.append(f"N sampled points: {int(n_pts)}")
        if x1 is not None and y1 is not None:
            lpx_str = f"  ({_fmt_value(length_px)} px)" if length_px is not None else ""
            lines.append(f"Start pixel: x={int(x1)}, y={int(y1)}{lpx_str}")
        if x2 is not None and y2 is not None:
            lines.append(f"End pixel:   x={int(x2)}, y={int(y2)}")
        if result.channel:
            lines.append(f"Channel: {result.channel}")
        roi_name = result.context.get("roi_name")
        if roi_name:
            lines.append(f"Line ROI: {roi_name}")
    elif result.kind == "spectrum_delta":
        dy = result.values.get("dy")
        dx = result.values.get("dx")
        slope = result.values.get("slope")
        slope_unit = (
            f"{y_unit}/{x_unit}".strip("/") if (x_unit or y_unit) else ""
        )
        if dy is not None:
            lines.append(f"Main value: Δy = {_fmt_value(dy)} {y_unit}".rstrip())
        if dx is not None:
            lines.append(f"Δx = {_fmt_value(dx)} {x_unit}".rstrip())
        if slope is not None:
            lines.append(f"Slope = {_fmt_value(slope)} {slope_unit}".rstrip())
        if result.channel:
            lines.append(f"Channel: {result.channel}")
    else:
        if result.channel:
            lines.append(f"Channel: {result.channel}")
        unit_parts = []
        if x_unit:
            unit_parts.append(f"x={x_unit}")
        if y_unit and y_unit != x_unit:
            unit_parts.append(f"y={y_unit}")
        if result.z_unit and result.z_unit not in (x_unit, y_unit):
            unit_parts.append(f"z={result.z_unit}")
        if unit_parts:
            lines.append("Units:  " + "  ".join(unit_parts))
        if result.values:
            lines.append("Values:")
            for k, v in result.values.items():
                label = VALUE_LABELS.get(k, k)
                lines.append(f"  {label}: {_fmt_value(v)}")

    if result.notes:
        lines.append(f"Notes: {result.notes}")

    lines.append("")
    lines.append("── Technical metadata ──")
    if result.source_label:
        lines.append(f"Source: {result.source_label}")
    if result.source_path and result.source_path != result.source_label:
        lines.append(f"File:   {result.source_path}")
    ctx_tech = {
        k: v for k, v in result.context.items()
        if v is not None and k not in ("roi_name",)
    }
    if ctx_tech:
        for k, v in ctx_tech.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)


def _fmt_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
