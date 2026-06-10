"""
ROI manager for ImageViewerDialog.

``ROIManagerPanel`` is a plain ``QWidget`` that lists the ROIs of the active
scan and provides:

  • rename, delete, set active via toolbar / context menu
  • invert (single ROI) and combine (multi-ROI, mode dropdown)
  • operation context menu: background subtract (fit / exclude), FFT, histogram, line profile

It is embedded directly in the viewer's ROI tab.  ``ROIManagerDock`` is a thin
``QDockWidget`` wrapper kept for backward compatibility and optional floating use;
it simply hosts an ``ROIManagerPanel`` and forwards the public API.

The panel communicates via a shared ROISet object and a callback API.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDockWidget, QGridLayout, QHBoxLayout,
    QInputDialog, QListWidget, QListWidgetItem, QMenu, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from probeflow.core import AREA_ROI_KINDS

_KIND_PREFIX = {
    "rectangle":    "▭ ",
    "ellipse":      "◯ ",
    "polygon":      "⬠ ",
    "freehand":     "⬠ ",
    "multipolygon": "⬡ ",
    "line":         "— ",
    "point":        "• ",
}


class ROIManagerPanel(QWidget):
    """Widget that lists ROIs and provides editing actions."""

    def __init__(self, roi_set_getter: Callable, callbacks: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("roiManagerPanel")
        self._roi_set_getter = roi_set_getter
        self._cb = callbacks

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        self._rename_btn = QPushButton("Rename")
        self._rename_btn.setFixedHeight(26)
        self._rename_btn.setEnabled(False)
        self._rename_btn.clicked.connect(self._on_rename)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setFixedHeight(26)
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._on_delete)

        self._active_btn = QPushButton("Set active")
        self._active_btn.setFixedHeight(26)
        self._active_btn.setEnabled(False)
        self._active_btn.clicked.connect(self._on_set_active)

        self._invert_btn = QPushButton("Invert")
        self._invert_btn.setFixedHeight(26)
        self._invert_btn.setEnabled(False)
        self._invert_btn.clicked.connect(self._on_invert)

        action_grid = QGridLayout()
        action_grid.setContentsMargins(0, 0, 0, 0)
        action_grid.setHorizontalSpacing(3)
        action_grid.setVerticalSpacing(3)
        action_grid.addWidget(self._rename_btn, 0, 0)
        action_grid.addWidget(self._delete_btn, 0, 1)
        action_grid.addWidget(self._active_btn, 1, 0)
        action_grid.addWidget(self._invert_btn, 1, 1)
        lay.addLayout(action_grid)

        combine_row = QWidget()
        combine_lay = QHBoxLayout(combine_row)
        combine_lay.setContentsMargins(0, 0, 0, 0)
        combine_lay.setSpacing(2)
        self._combine_btn = QPushButton("Combine")
        self._combine_btn.setFixedHeight(26)
        self._combine_btn.setEnabled(False)
        self._combine_btn.clicked.connect(self._on_combine)
        self._combine_mode = QComboBox()
        self._combine_mode.addItems(["union", "intersection", "difference", "xor"])
        self._combine_mode.setFixedHeight(26)
        combine_lay.addWidget(self._combine_btn)
        combine_lay.addWidget(self._combine_mode)
        lay.addWidget(combine_row)

        # ── list widget ───────────────────────────────────────────────────────
        self._list = QListWidget()
        self._list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.itemDoubleClicked.connect(lambda _: self._on_rename())
        self._list.itemSelectionChanged.connect(self._on_item_selection_changed)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.installEventFilter(self)
        lay.addWidget(self._list)

    # ── event filter ─────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        if obj is self._list and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                if self._delete_btn.isEnabled():
                    self._on_delete()
                    return True
        return super().eventFilter(obj, event)

    # ── public ───────────────────────────────────────────────────────────────

    def refresh(self, roi_set) -> None:
        self._list.blockSignals(True)
        try:
            self._list.clear()
            if roi_set is not None:
                for roi in roi_set.rois:
                    prefix = _KIND_PREFIX.get(roi.kind, "? ")
                    item = QListWidgetItem(prefix + roi.name)
                    item.setData(Qt.UserRole, roi.id)
                    self._list.addItem(item)
        finally:
            self._list.blockSignals(False)
        self._on_item_selection_changed()

    # ── selection helpers ─────────────────────────────────────────────────────

    def _selected_roi_ids(self) -> list[str]:
        return [
            item.data(Qt.UserRole)
            for item in self._list.selectedItems()
            if item.data(Qt.UserRole) is not None
        ]

    def selected_roi_ids(self) -> list[str]:
        """Return selected ROI IDs for viewer-level actions."""
        return self._selected_roi_ids()

    def _selected_roi_id(self) -> "str | None":
        ids = self._selected_roi_ids()
        return ids[0] if ids else None

    # ── action slots ─────────────────────────────────────────────────────────

    def _on_rename(self) -> None:
        roi_id = self._selected_roi_id()
        roi_set = self._roi_set_getter()
        if roi_id is None or roi_set is None:
            return
        roi = roi_set.get(roi_id)
        if roi is None:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename ROI", "New name:", text=roi.name
        )
        if ok and new_name.strip():
            roi.name = new_name.strip()
            self._cb.get("on_roi_set_changed", lambda: None)()
            self.refresh(roi_set)

    def _on_delete(self) -> None:
        roi_ids = self._selected_roi_ids()
        roi_set = self._roi_set_getter()
        if not roi_ids or roi_set is None:
            return
        for roi_id in roi_ids:
            roi_set.remove(roi_id)
        self._cb.get("on_roi_set_changed", lambda: None)()
        self.refresh(roi_set)

    def _on_set_active(self) -> None:
        roi_id = self._selected_roi_id()
        roi_set = self._roi_set_getter()
        if roi_id is None or roi_set is None:
            return
        roi_set.set_active(roi_id)
        self._cb.get("on_roi_set_changed", lambda: None)()

    def _on_invert(self) -> None:
        roi_id = self._selected_roi_id()
        roi_set = self._roi_set_getter()
        if roi_id is None or roi_set is None:
            return
        roi = roi_set.get(roi_id)
        if roi is None:
            return
        get_shape = self._cb.get("get_image_shape")
        image_shape = get_shape() if get_shape else None
        if image_shape is None:
            return
        from probeflow.core import roi as _roi_module
        inverted = _roi_module.invert(roi, image_shape)
        roi_set.add(inverted)
        self._cb.get("on_roi_set_changed", lambda: None)()
        self.refresh(roi_set)

    def _on_combine(self) -> None:
        roi_ids = self._selected_roi_ids()
        roi_set = self._roi_set_getter()
        if len(roi_ids) < 2 or roi_set is None:
            return
        rois = [roi_set.get(rid) for rid in roi_ids]
        rois = [r for r in rois if r is not None]
        if len(rois) < 2:
            return
        mode = self._combine_mode.currentText()
        from probeflow.core import roi as _roi_module
        combined = _roi_module.combine(rois, mode)
        roi_set.add(combined)
        self._cb.get("on_roi_set_changed", lambda: None)()
        self.refresh(roi_set)

    def _on_item_selection_changed(self) -> None:
        ids = self._selected_roi_ids()
        n = len(ids)
        self._rename_btn.setEnabled(n == 1)
        self._delete_btn.setEnabled(n >= 1)
        self._active_btn.setEnabled(n == 1)
        self._invert_btn.setEnabled(n == 1)
        self._combine_btn.setEnabled(n >= 2)
        self._cb.get("on_roi_selection_changed", lambda: None)()

    # ── context menu ─────────────────────────────────────────────────────────

    def _show_context_menu(self, pos) -> None:
        roi_id = self._selected_roi_id()
        roi_ids = self._selected_roi_ids()
        roi_set = self._roi_set_getter()
        roi = roi_set.get(roi_id) if (roi_set and roi_id) else None
        is_area = roi is not None and roi.kind in AREA_ROI_KINDS
        is_line = roi is not None and roi.kind == "line"
        selected_area_pair = False
        if roi_set is not None and len(roi_ids) == 2:
            selected = [roi_set.get(rid) for rid in roi_ids]
            selected_area_pair = all(r is not None and r.kind in AREA_ROI_KINDS for r in selected)

        menu = QMenu(self)

        rename_act = menu.addAction("Rename")
        rename_act.setEnabled(roi is not None)
        rename_act.triggered.connect(self._on_rename)

        delete_act = menu.addAction("Delete")
        delete_act.setEnabled(roi is not None)
        delete_act.triggered.connect(self._on_delete)

        set_active_act = menu.addAction("Set Active")
        set_active_act.setEnabled(roi is not None)
        set_active_act.triggered.connect(self._on_set_active)

        invert_act = menu.addAction("Invert")
        invert_act.setEnabled(roi is not None)
        invert_act.triggered.connect(self._on_invert)

        menu.addSeparator()

        stm_bg_act = menu.addAction("STM Background fit from ROI...")
        stm_bg_act.setEnabled(is_area)
        stm_bg_act.triggered.connect(
            lambda: self._cb.get("on_stm_background_roi", lambda _: None)(roi_id)
        )

        fft_act = menu.addAction("FFT this region")
        fft_act.setEnabled(is_area)
        fft_act.triggered.connect(
            lambda: self._cb.get("on_fft_roi", lambda _: None)(roi_id)
        )

        hist_act = menu.addAction("Histogram of this region")
        hist_act.setEnabled(is_area)
        hist_act.triggered.connect(
            lambda: self._cb.get("on_histogram_roi", lambda _: None)(roi_id)
        )

        stats_act = menu.addAction("Add ROI statistics to measurements")
        stats_act.setEnabled(is_area)
        stats_act.triggered.connect(
            lambda: self._cb.get("on_roi_stats_measurement", lambda _: None)(roi_id)
        )

        maxima_act = menu.addAction("Detect maxima in this region")
        maxima_act.setEnabled(is_area)
        maxima_act.triggered.connect(
            lambda: self._cb.get("on_feature_maxima_roi", lambda _: None)(roi_id)
        )

        step_act = menu.addAction("Add step height from selected ROIs")
        step_act.setEnabled(selected_area_pair)
        step_act.triggered.connect(
            lambda: self._cb.get("on_step_height_measurement", lambda _: None)(roi_ids)
        )

        profile_act = menu.addAction("Line profile")
        profile_act.setEnabled(is_line)
        profile_act.triggered.connect(
            lambda: self._cb.get("on_line_profile_roi", lambda _: None)(roi_id)
        )

        profile_measure_act = menu.addAction("Add line profile measurement")
        profile_measure_act.setEnabled(is_line)
        profile_measure_act.triggered.connect(
            lambda: self._cb.get("on_line_profile_measurement", lambda _: None)(roi_id)
        )

        menu.exec(self._list.mapToGlobal(pos))


class ROIManagerDock(QDockWidget):
    """Thin dock wrapper around :class:`ROIManagerPanel` (optional floating use)."""

    def __init__(self, roi_set_getter: Callable, callbacks: dict, parent=None):
        super().__init__("ROI Manager", parent)
        self.setObjectName("roiManagerDock")
        self.setFeatures(
            QDockWidget.DockWidgetClosable
            | QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
        )
        self.setMinimumWidth(160)
        self.setMaximumWidth(280)
        self.resize(200, self.height())

        self.panel = ROIManagerPanel(roi_set_getter, callbacks, parent=self)
        self.setWidget(self.panel)

    # ── forwarded API ──────────────────────────────────────────────────────────

    def refresh(self, roi_set) -> None:
        self.panel.refresh(roi_set)

    def selected_roi_ids(self) -> list[str]:
        return self.panel.selected_roi_ids()
