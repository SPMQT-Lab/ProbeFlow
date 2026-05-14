"""Composite image measurement panel for the viewer dock."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from probeflow.gui.widgets.feature_detection_panel import (
    FeatureDetectionPanel,
    PointMaskFFTPanel,
)
from probeflow.gui.widgets.measurement_table import MeasurementResultsTable


class ImageMeasurementsPanel(QWidget):
    """Mode-based container for image measurement setup and results."""

    roiStatsRequested = Signal()
    stepHeightRequested = Signal()
    lineProfileRequested = Signal()

    _MODES = [
        ("Feature maxima", "feature_maxima"),
        ("Point mask / FFT", "point_fft"),
        ("ROI statistics", "roi_stats"),
        ("Step height", "step_height"),
        ("Line profile", "line_profile"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.feature_panel = FeatureDetectionPanel(self)
        self.point_mask_panel = PointMaskFFTPanel(self)
        self.table = MeasurementResultsTable(self)
        self._mode_pages: dict[str, int] = {}
        self._action_buttons: dict[str, QPushButton] = {}
        self._action_status: dict[str, QLabel] = {}
        self._build()

    def measurement_type(self) -> str:
        """Return the current measurement mode key."""
        return self._type_cb.currentData() or "feature_maxima"

    def set_measurement_type(self, key: str) -> None:
        """Select a measurement mode by stable key."""
        for idx in range(self._type_cb.count()):
            if self._type_cb.itemData(idx) == key:
                self._type_cb.setCurrentIndex(idx)
                return

    def set_points_count(self, count: int, *, roi_name: str | None = None) -> None:
        """Update point-dependent child panels together."""
        self.feature_panel.set_points_count(count, roi_name=roi_name)
        self.point_mask_panel.set_points_available(int(count) > 0)

    def show_message(self, message: str) -> None:
        """Forward a status message to the active child panel when useful."""
        self.feature_panel.show_message(message)
        self.point_mask_panel.show_message(message)

    def set_action_available(self, mode: str, available: bool, *, message: str = "") -> None:
        """Enable or disable the action button for a mode and show a status message."""
        if mode in self._action_buttons:
            self._action_buttons[mode].setEnabled(available)
        if mode in self._action_status:
            self._action_status[mode].setText(message)

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        type_lbl = QLabel("Measurement type")
        type_lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(type_lbl)
        self._type_cb = QComboBox()
        for label, key in self._MODES:
            self._type_cb.addItem(label, key)
        self._type_cb.currentIndexChanged.connect(self._sync_mode_page)
        lay.addWidget(self._type_cb)

        self._setup_stack = QStackedWidget()
        self._setup_stack.addWidget(self.feature_panel)
        self._mode_pages["feature_maxima"] = 0
        self._setup_stack.addWidget(self.point_mask_panel)
        self._mode_pages["point_fft"] = 1
        self._setup_stack.addWidget(self._action_page(
            "roi_stats",
            "ROI statistics",
            "Calculate mean, median, roughness, area, extrema, and finite-pixel counts for the active area ROI.",
            "Add active ROI statistics",
            self.roiStatsRequested,
        ))
        self._mode_pages["roi_stats"] = 2
        self._setup_stack.addWidget(self._action_page(
            "step_height",
            "Step height",
            "Select two area ROIs, then calculate the mean-height difference between them.",
            "Add step height from selected ROIs",
            self.stepHeightRequested,
        ))
        self._mode_pages["step_height"] = 3
        self._setup_stack.addWidget(self._action_page(
            "line_profile",
            "Line profile",
            "Use the active line ROI to add a profile summary to the measurement table.",
            "Add current line profile",
            self.lineProfileRequested,
        ))
        self._mode_pages["line_profile"] = 4
        lay.addWidget(self._setup_stack)

        results_lbl = QLabel("Results")
        results_lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(results_lbl)
        lay.addWidget(self.table, 1)
        self._sync_mode_page()

    def _sync_mode_page(self) -> None:
        key = self.measurement_type()
        self._setup_stack.setCurrentIndex(self._mode_pages.get(key, 0))

    def _action_page(
        self,
        mode_key: str,
        title: str,
        description: str,
        button_text: str,
        signal: Any,
    ) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(title_lbl)
        desc_lbl = QLabel(description)
        desc_lbl.setWordWrap(True)
        lay.addWidget(desc_lbl)
        button = QPushButton(button_text)
        button.setDefault(False)
        button.setAutoDefault(False)
        button.clicked.connect(signal.emit)
        lay.addWidget(button)
        status_lbl = QLabel("")
        status_lbl.setWordWrap(True)
        status_lbl.setStyleSheet("color: palette(mid);")
        lay.addWidget(status_lbl)
        lay.addStretch(1)
        self._action_buttons[mode_key] = button
        self._action_status[mode_key] = status_lbl
        return page
