"""Compact manager widget for the active-mask layer.

Sibling of :class:`probeflow.gui.roi_manager_dock.ROIManagerPanel`: lists the
masks owned by the viewer's :class:`~probeflow.core.mask.MaskSet`, sets the
active mask, runs morphological cleanup, and bridges to ROI conversion and
mask-restricted statistics.  All edits go through the supplied callbacks so the
viewer can persist the sidecar and refresh the overlay.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from probeflow.processing import mask_ops

# Cleanup ops: label → (function, kwargs).  Radii/sizes in pixels.
_CLEANUP_OPS: dict[str, tuple] = {
    "Remove small objects": (mask_ops.remove_small_objects, {"min_size": 16}),
    "Fill holes": (mask_ops.fill_holes, {}),
    "Dilate": (mask_ops.dilate, {"radius": 1}),
    "Erode": (mask_ops.erode, {"radius": 1}),
    "Open": (mask_ops.binary_open, {"radius": 1}),
    "Close": (mask_ops.binary_close, {"radius": 1}),
    "Skeletonize": (mask_ops.skeletonize, {}),
    "Remove border objects": (mask_ops.remove_border_objects, {}),
}


class MaskManagerPanel(QWidget):
    """Widget that lists masks and provides activation / cleanup / conversion."""

    def __init__(self, mask_set_getter: Callable, callbacks: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("maskManagerPanel")
        self._mask_set_getter = mask_set_getter
        self._cb = callbacks

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        self._active_btn = _btn("Set active", self._on_set_active)
        self._rename_btn = _btn("Rename", self._on_rename)
        self._delete_btn = _btn("Delete", self._on_delete)
        self._invert_btn = _btn("Invert", self._on_invert)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(3)
        grid.setVerticalSpacing(3)
        grid.addWidget(self._active_btn, 0, 0)
        grid.addWidget(self._rename_btn, 0, 1)
        grid.addWidget(self._delete_btn, 1, 0)
        grid.addWidget(self._invert_btn, 1, 1)
        lay.addLayout(grid)

        cleanup_row = QWidget()
        crow = QHBoxLayout(cleanup_row)
        crow.setContentsMargins(0, 0, 0, 0)
        crow.setSpacing(2)
        self._cleanup_combo = QComboBox()
        self._cleanup_combo.addItems(list(_CLEANUP_OPS))
        self._cleanup_btn = _btn("Apply cleanup", self._on_cleanup)
        crow.addWidget(self._cleanup_combo, 1)
        crow.addWidget(self._cleanup_btn)
        lay.addWidget(cleanup_row)

        convert_row = QWidget()
        vrow = QHBoxLayout(convert_row)
        vrow.setContentsMargins(0, 0, 0, 0)
        vrow.setSpacing(2)
        self._roi_btn = _btn("To ROI(s)", self._on_convert_roi)
        self._stats_btn = _btn("Statistics", self._on_stats)
        self._export_btn = _btn("Export…", self._on_export)
        vrow.addWidget(self._roi_btn)
        vrow.addWidget(self._stats_btn)
        vrow.addWidget(self._export_btn)
        lay.addWidget(convert_row)

        self._list = QListWidget()
        self._list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.itemDoubleClicked.connect(lambda _: self._on_set_active())
        self._list.itemSelectionChanged.connect(self._sync_buttons)
        lay.addWidget(self._list)

        self.refresh(self._mask_set_getter())

    # ── public ───────────────────────────────────────────────────────────────

    def refresh(self, mask_set) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        if mask_set is not None:
            for mask in mask_set.masks:
                active = mask.id == mask_set.active_mask_id
                label = ("● " if active else "○ ") + f"{mask.name}  ({mask.count()} px)"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, mask.id)
                self._list.addItem(item)
        self._list.blockSignals(False)
        self._sync_buttons()

    # ── selection helpers ──────────────────────────────────────────────────────

    def _selected_id(self) -> "str | None":
        items = self._list.selectedItems()
        return items[0].data(Qt.UserRole) if items else None

    def _sync_buttons(self) -> None:
        has = self._selected_id() is not None
        for b in (self._active_btn, self._rename_btn, self._delete_btn,
                  self._invert_btn, self._cleanup_btn, self._roi_btn,
                  self._stats_btn, self._export_btn):
            b.setEnabled(has)

    def _changed(self) -> None:
        self._cb.get("on_mask_set_changed", lambda: None)()
        self.refresh(self._mask_set_getter())

    # ── action slots ────────────────────────────────────────────────────────────

    def _on_set_active(self) -> None:
        mask_id = self._selected_id()
        mask_set = self._mask_set_getter()
        if mask_id is None or mask_set is None:
            return
        mask_set.set_active(mask_id)
        self._changed()

    def _on_rename(self) -> None:
        mask_id = self._selected_id()
        mask_set = self._mask_set_getter()
        if mask_id is None or mask_set is None:
            return
        mask = mask_set.get(mask_id)
        if mask is None:
            return
        new_name, ok = QInputDialog.getText(self, "Rename mask", "New name:", text=mask.name)
        if ok and new_name.strip():
            mask.name = new_name.strip()
            self._changed()

    def _on_delete(self) -> None:
        mask_id = self._selected_id()
        mask_set = self._mask_set_getter()
        if mask_id is None or mask_set is None:
            return
        mask_set.remove(mask_id)
        self._changed()

    def _on_invert(self) -> None:
        self._apply_op(mask_ops.invert, {})

    def _on_cleanup(self) -> None:
        fn, kwargs = _CLEANUP_OPS[self._cleanup_combo.currentText()]
        self._apply_op(fn, kwargs)

    def _apply_op(self, fn, kwargs) -> None:
        mask_id = self._selected_id()
        mask_set = self._mask_set_getter()
        if mask_id is None or mask_set is None:
            return
        mask = mask_set.get(mask_id)
        if mask is None:
            return
        mask_set.replace(mask_id, fn(mask.data, **kwargs))
        self._changed()

    def _on_convert_roi(self) -> None:
        mask_id = self._selected_id()
        if mask_id is not None:
            self._cb.get("convert_to_roi", lambda _id: None)(mask_id)

    def _on_stats(self) -> None:
        mask_id = self._selected_id()
        mask_set = self._mask_set_getter()
        if mask_id is None or mask_set is None:
            return
        # Statistics run over the active mask, so activate the selection first.
        mask_set.set_active(mask_id)
        self._changed()
        self._cb.get("add_mask_stats", lambda: None)()

    def _on_export(self) -> None:
        mask_id = self._selected_id()
        if mask_id is not None:
            self._cb.get("export_mask", lambda _id: None)(mask_id)


def _btn(text: str, slot) -> QPushButton:
    b = QPushButton(text)
    b.setFixedHeight(26)
    b.setEnabled(False)
    b.clicked.connect(slot)
    return b
