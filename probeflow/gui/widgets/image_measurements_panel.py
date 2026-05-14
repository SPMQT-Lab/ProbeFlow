"""Composite image measurement panel for the viewer dock."""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from probeflow.gui.widgets.feature_detection_panel import FeatureDetectionPanel
from probeflow.gui.widgets.measurement_table import MeasurementResultsTable


class ImageMeasurementsPanel(QWidget):
    """Container for feature detection controls and measurement results."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.feature_panel = FeatureDetectionPanel(self)
        self.table = MeasurementResultsTable(self)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)
        lay.addWidget(self.feature_panel)
        lay.addWidget(self.table, 1)
