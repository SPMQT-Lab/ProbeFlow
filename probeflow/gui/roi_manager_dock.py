"""
ROI manager dock widget for ImageViewerDialog.

Wraps in a QDockWidget for use inside a QMainWindow that is embedded in
the viewer dialog.  Displays the ROIs of the active scan and provides:

  • rename, delete, set active via toolbar / context menu
  • invert (single ROI) and combine (multi-ROI, mode dropdown)
  • operation context menu: background subtract (fit / exclude), FFT, histogram, line profile

The dock communicates via a shared ROISet object and a callback API.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDockWidget, QHBoxLayout, QInputDialog,
    QLabel, QListWidget, QListWidgetItem, QMenu, QPushButton, QToolBar,
    QVBoxLayout, QWidget,
)

_KIND_PREFIX = {
    "rectangle":    "▭ ",
    "ellipse":      "◯ ",
    "polygon":      "⬠ ",
    "freehand":     "⬠ ",
    "multipolygon": "⬡ ",
    "line":         "— ",
    "point":        "• ",
}

_AREA_KINDS = {"rectangle", "ellipse", "polygon", "freehand", "multipolygon"}


class ROIManagerDock(QDockWidget):
    """Dock widget that lists ROIs and provides editing actions."""

    def __init__(self, roi_set_getter: Callable, callbacks: dict, parent=None):
        super().__init__("ROI Manager", parent)
        self._roi_set_getter = roi_set_getter
        self._cb = callbacks
        self.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )

        contents = QWidget()
        lay = QVBoxLayout(contents)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        # ── toolbar ──────────────────────────────────────────────────────────
        bar = QToolBar()
        bar.setMovable(False)

        self._rename_btn = QPushButton("Rename")
        self._rename_btn.setFixedHeight(22)
        self._rename_btn.setEnabled(False)
        self._rename_btn.clicked.connect(self._on_rename)
        bar.addWidget(self._rename_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setFixedHeight(22)
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._on_delete)
        bar.addWidget(self._delete_btn)

        self._active_btn = QPushButton("Set Active")
        self._active_btn.setFixedHeight(22)
        self._active_btn.setEnabled(False)
        self._active_btn.clicked.connect(self._on_set_active)
        bar.addWidget(self._active_btn)

        self._invert_btn = QPushButton("Invert")
        self._invert_btn.setFixedHeight(22)
        self._invert_btn.setEnabled(False)
        self._invert_btn.clicked.connect(self._on_invert)
        bar.addWidget(self._invert_btn)

        combine_row = QWidget()
        combine_lay = QHBoxLayout(combine_row)
        combine_lay.setContentsMargins(0, 0, 0, 0)
        combine_lay.setSpacing(2)
        self._combine_btn = QPushButton("Combine")
        self._combine_btn.setFixedHeight(22)
        self._combine_btn.setEnabled(False)
        self._combine_btn.clicked.connect(self._on_combine)
        self._combine_mode = QComboBox()
        self._combine_mode.addItems(["union", "intersection", "difference", "xor"])
        self._combine_mode.setFixedHeight(22)
        combine_lay.addWidget(self._combine_btn)
        combine_lay.addWidget(self._combine_mode)
        bar.addWidget(combine_row)

        lay.addWidget(bar)

        # ── list widget ───────────────────────────────────────────────────────
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.itemDoubleClicked.connect(lambda _: self._on_rename())
        self._list.itemSelectionChanged.connect(self._on_item_selection_changed)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        lay.addWidget(self._list)

        self.setWidget(contents)

    # ── public ───────────────────────────────────────────────────────────────

    def refresh(self, roi_set) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        if roi_set is not None:
            for roi in roi_set.rois:
                prefix = _KIND_PREFIX.get(roi.kind, "? ")
                item = QListWidgetItem(prefix + roi.name)
                item.setData(Qt.UserRole, roi.id)
                self._list.addItem(item)
        self._list.blockSignals(False)
        self._on_item_selection_changed()

    # ── selection helpers ─────────────────────────────────────────────────────

    def _selected_roi_ids(self) -> list[str]:
        return [
            item.data(Qt.UserRole)
            for item in self._list.selectedItems()
            if item.data(Qt.UserRole) is not None
        ]

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

    # ── context menu ─────────────────────────────────────────────────────────

    def _show_context_menu(self, pos) -> None:
        roi_id = self._selected_roi_id()
        roi_set = self._roi_set_getter()
        roi = roi_set.get(roi_id) if (roi_set and roi_id) else None
        is_area = roi is not None and roi.kind in _AREA_KINDS
        is_line = roi is not None and roi.kind == "line"

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

        bg_fit_act = menu.addAction("Background subtract (fit region)")
        bg_fit_act.setEnabled(is_area)
        bg_fit_act.triggered.connect(
            lambda: self._cb.get("on_bg_subtract_fit", lambda _: None)(roi_id)
        )

        bg_exc_act = menu.addAction("Background subtract (exclude region)")
        bg_exc_act.setEnabled(is_area)
        bg_exc_act.triggered.connect(
            lambda: self._cb.get("on_bg_subtract_exclude", lambda _: None)(roi_id)
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

        profile_act = menu.addAction("Line profile")
        profile_act.setEnabled(is_line)
        profile_act.triggered.connect(
            lambda: self._cb.get("on_line_profile_roi", lambda _: None)(roi_id)
        )

        menu.exec(self._list.mapToGlobal(pos))
