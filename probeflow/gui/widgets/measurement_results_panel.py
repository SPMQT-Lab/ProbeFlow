"""Compatibility wrapper for the canonical measurement results table."""

from __future__ import annotations

from probeflow.gui.widgets.measurement_table import MeasurementResultsTable
from probeflow.measurements.models import MeasurementResult


class MeasurementResultsPanel(MeasurementResultsTable):
    """Backward-compatible name for the canonical measurement table."""

    def add_result(self, result: MeasurementResult) -> None:
        super().add_result(result)

    def result_count(self) -> int:
        return len(self.results())

    def clear_all(self) -> None:
        self.clear_results()
