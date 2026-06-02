"""Composite image measurement panel for the viewer Measure tab.

The panel is a **master → detail** surface:

* the **menu** lists the measurement tools, grouped, plus a "Results (N)" entry;
* picking a tool opens a **detail** view that shows *only* that tool's controls and
  the shared results table, with a "‹ Tools" button back to the menu.

This keeps a chosen tool's output (e.g. an angle) front-and-centre instead of buried
under every other option.  The panel holds no analysis logic — tools either drive the
reusable setup widgets or emit a request signal the viewer connects to its handlers.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
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
from probeflow.gui.widgets.line_periodicity_panel import LinePeriodicityPanel
from probeflow.gui.widgets.measurement_table import MeasurementResultsTable


class ImageMeasurementsPanel(QWidget):
    """Master/detail container for image measurement tools and results."""

    # Tools the viewer fulfils (it connects each to an existing handler).
    roiStatsRequested = Signal()
    stepHeightRequested = Signal()
    lineProfileRequested = Signal()
    distanceRequested = Signal()
    angleRequested = Signal()
    updateAngleRequested = Signal()
    featureFinderRequested = Signal()
    pairCorrelationRequested = Signal()
    featureToLatticeRequested = Signal()
    latticeGridRequested = Signal()

    # Setup-style modes that map to a page of ``_setup_stack``.
    _MODES = [
        ("Feature maxima", "feature_maxima"),
        ("Point mask / FFT", "point_fft"),
        ("ROI statistics", "roi_stats"),
        ("Step height", "step_height"),
        ("Line profile", "line_profile"),
        ("Line periodicity", "line_periodicity"),
    ]

    # Menu layout: (group title, [(label, key, kind), …]); kind ∈ setup|oneshot|dialog.
    _TOOL_GROUPS = [
        ("Quick measurements", [
            ("Distance", "distance", "oneshot"),
            ("Angle", "angle", "oneshot"),
        ]),
        ("ROI measurements", [
            ("ROI statistics", "roi_stats", "setup"),
            ("Step height", "step_height", "setup"),
        ]),
        ("Profiles", [
            ("Line profile", "line_profile", "setup"),
            ("Line periodicity", "line_periodicity", "setup"),
        ]),
        ("Feature detection", [
            ("Feature maxima", "feature_maxima", "setup"),
            ("Point mask / FFT", "point_fft", "setup"),
        ]),
        ("Tools", [
            ("Feature finder…", "feature_finder", "dialog"),
            ("Pair correlation…", "pair_correlation", "dialog"),
            ("Feature-to-lattice…", "feature_to_lattice", "dialog"),
            ("Lattice grid…", "lattice_grid", "dialog"),
        ]),
    ]

    _DIALOG_SIGNALS = {
        "feature_finder": "featureFinderRequested",
        "pair_correlation": "pairCorrelationRequested",
        "feature_to_lattice": "featureToLatticeRequested",
        "lattice_grid": "latticeGridRequested",
    }
    _ONESHOT_SIGNALS = {
        "distance": "distanceRequested",
        "angle": "angleRequested",
    }
    _ONESHOT_INFO = {
        "distance": "Select a line ROI on the image; its length and angle are added "
                    "to the results below.",
        "angle": "Click P1, P2 (vertex), then P3 on the image to measure an angle. "
                 "Drag the handles to adjust, then use “Update angle”.",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.feature_panel = FeatureDetectionPanel(self)
        self.point_mask_panel = PointMaskFFTPanel(self)
        self.line_periodicity_panel = LinePeriodicityPanel(self)
        self.table = MeasurementResultsTable(self)
        self._mode_pages: dict[str, int] = {}
        self._action_buttons: dict[str, QPushButton] = {}
        self._action_status: dict[str, QLabel] = {}
        self._current_mode = "feature_maxima"
        self._titles = self._collect_titles()
        self._build()

    # ── public API (unchanged for callers) ────────────────────────────────────

    def measurement_type(self) -> str:
        """Return the current setup-mode key."""
        return self._current_mode

    def set_measurement_type(self, key: str) -> None:
        """Select a setup mode by stable key and focus its detail view."""
        if key in self._mode_pages:
            self._open_tool(key)

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

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        self._nav = QStackedWidget()
        outer.addWidget(self._nav)
        self._nav.addWidget(self._build_menu_page())     # index 0 — menu
        self._nav.addWidget(self._build_detail_page())   # index 1 — detail
        self._nav.setCurrentIndex(0)

        self.table.resultsChanged.connect(self._on_results_changed)

    def _build_menu_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        for group, tools in self._TOOL_GROUPS:
            lay.addWidget(self._group_label(group))
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(4)
            grid.setVerticalSpacing(4)
            n = len(tools)
            for i, (label, key, _kind) in enumerate(tools):
                btn = QPushButton(label)
                btn.setFixedHeight(26)
                btn.setDefault(False)
                btn.setAutoDefault(False)
                btn.clicked.connect(lambda _c=False, k=key: self._open_tool(k))
                r, c = divmod(i, 2)
                if i == n - 1 and n % 2 == 1:
                    grid.addWidget(btn, r, 0, 1, 2)
                else:
                    grid.addWidget(btn, r, c)
            lay.addLayout(grid)

        lay.addWidget(self._hline())
        self._results_btn = QPushButton("Results (0)")
        self._results_btn.setFixedHeight(26)
        self._results_btn.setDefault(False)
        self._results_btn.setAutoDefault(False)
        self._results_btn.clicked.connect(self._open_results)
        lay.addWidget(self._results_btn)
        lay.addStretch(1)
        return page

    def _build_detail_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        header = QHBoxLayout()
        self._back_btn = QPushButton("‹  Tools")
        self._back_btn.setObjectName("ghostBtn")
        self._back_btn.setCursor(Qt.PointingHandCursor)
        self._back_btn.setDefault(False)
        self._back_btn.setAutoDefault(False)
        self._back_btn.clicked.connect(self._open_menu)
        header.addWidget(self._back_btn)
        self._detail_title = QLabel("")
        self._detail_title.setStyleSheet("font-weight: 700;")
        header.addWidget(self._detail_title, 1)
        lay.addLayout(header)

        self._detail_info = QLabel("")
        self._detail_info.setWordWrap(True)
        self._detail_info.setStyleSheet("color: palette(mid);")
        self._detail_info.hide()
        lay.addWidget(self._detail_info)

        # Reusable per-mode setup pages (shown only for setup tools).
        self._setup_stack = QStackedWidget()
        self._setup_stack.addWidget(self.feature_panel)
        self._mode_pages["feature_maxima"] = 0
        self._setup_stack.addWidget(self.point_mask_panel)
        self._mode_pages["point_fft"] = 1
        self._setup_stack.addWidget(self._action_page(
            "roi_stats", "ROI statistics",
            "Calculate mean, median, roughness, area, extrema, and finite-pixel "
            "counts for the active area ROI.",
            "Add active ROI statistics", self.roiStatsRequested,
        ))
        self._mode_pages["roi_stats"] = 2
        self._setup_stack.addWidget(self._action_page(
            "step_height", "Step height",
            "Select two area ROIs, then calculate the mean-height difference "
            "between them.",
            "Add step height from selected ROIs", self.stepHeightRequested,
        ))
        self._mode_pages["step_height"] = 3
        self._setup_stack.addWidget(self._action_page(
            "line_profile", "Line profile",
            "Use the active line ROI to add a profile summary to the measurement "
            "table.",
            "Add current line profile", self.lineProfileRequested,
        ))
        self._mode_pages["line_profile"] = 4
        self._setup_stack.addWidget(self.line_periodicity_panel)
        self._mode_pages["line_periodicity"] = 5
        lay.addWidget(self._setup_stack)

        # Extra controls for one-shot tools (currently the Angle “Update” action).
        self._detail_extra = QWidget()
        extra_lay = QVBoxLayout(self._detail_extra)
        extra_lay.setContentsMargins(0, 0, 0, 0)
        extra_lay.setSpacing(4)
        self._update_angle_btn = QPushButton("Update angle measurement")
        self._update_angle_btn.setToolTip(
            "After dragging the angle handles, rewrite the current angle measurement "
            "with the adjusted value."
        )
        self._update_angle_btn.setDefault(False)
        self._update_angle_btn.setAutoDefault(False)
        self._update_angle_btn.clicked.connect(self.updateAngleRequested.emit)
        extra_lay.addWidget(self._update_angle_btn)
        self._detail_extra.setVisible(False)
        lay.addWidget(self._detail_extra)

        results_lbl = QLabel("Results")
        results_lbl.setStyleSheet("font-weight: 600;")
        lay.addWidget(results_lbl)
        lay.addWidget(self.table, 1)
        return page

    # ── navigation ─────────────────────────────────────────────────────────────

    def _open_menu(self) -> None:
        self._nav.setCurrentIndex(0)

    def _open_tool(self, key: str) -> None:
        # Dialog tools pop their own overlay/dock; the menu stays put.
        if key in self._DIALOG_SIGNALS:
            getattr(self, self._DIALOG_SIGNALS[key]).emit()
            return

        self._detail_title.setText(self._titles.get(key, key))
        if key in self._ONESHOT_SIGNALS:
            self._setup_stack.hide()
            self._detail_info.setText(self._ONESHOT_INFO.get(key, ""))
            self._detail_info.setVisible(bool(self._ONESHOT_INFO.get(key)))
            self._detail_extra.setVisible(key == "angle")
            self._nav.setCurrentIndex(1)
            getattr(self, self._ONESHOT_SIGNALS[key]).emit()
        else:  # setup tool
            self._select_mode(key)
            self._setup_stack.show()
            self._detail_info.hide()
            self._detail_extra.setVisible(False)
            self._nav.setCurrentIndex(1)

    def _open_results(self) -> None:
        self._detail_title.setText("Results")
        self._setup_stack.hide()
        self._detail_info.hide()
        self._detail_extra.setVisible(False)
        self._nav.setCurrentIndex(1)

    def _on_results_changed(self, count: int, added: bool) -> None:
        self._results_btn.setText(f"Results ({count})")
        # Surface a freshly produced result even if it came from outside the menu.
        if added and self._nav.currentIndex() == 0:
            self._open_results()

    def _select_mode(self, key: str) -> None:
        if key in self._mode_pages:
            self._current_mode = key
            self._setup_stack.setCurrentIndex(self._mode_pages[key])

    # ── helpers ────────────────────────────────────────────────────────────────

    def _collect_titles(self) -> dict[str, str]:
        titles: dict[str, str] = {}
        for _group, tools in self._TOOL_GROUPS:
            for label, key, _kind in tools:
                titles[key] = label.rstrip("…")
        return titles

    @staticmethod
    def _group_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: 600; color: palette(mid);")
        return lbl

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line

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
