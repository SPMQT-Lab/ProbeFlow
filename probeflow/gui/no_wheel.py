"""Shared Qt helpers for scrollable parameter panels."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QAbstractSpinBox, QScrollArea, QWidget


class _NoWheelSpinBoxFilter(QObject):
    """Prevents mouse-wheel scrolling over spinboxes from changing values."""

    def eventFilter(self, obj, event):
        if isinstance(obj, QAbstractSpinBox) and event.type() == QEvent.Wheel:
            scroll = _parent_scroll_area(obj)
            if scroll is not None:
                delta_y = event.angleDelta().y()
                if delta_y:
                    bar = scroll.verticalScrollBar()
                    bar.setValue(bar.value() - delta_y)
            event.accept()
            return True
        return False


_NO_WHEEL_FILTER: _NoWheelSpinBoxFilter | None = None


def _no_wheel_filter() -> _NoWheelSpinBoxFilter:
    global _NO_WHEEL_FILTER
    if _NO_WHEEL_FILTER is None:
        _NO_WHEEL_FILTER = _NoWheelSpinBoxFilter()
    return _NO_WHEEL_FILTER


def _parent_scroll_area(widget: QWidget) -> QScrollArea | None:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QScrollArea):
            return parent
        parent = parent.parentWidget()
    return None


def install_no_wheel_spinboxes(widget: QWidget) -> None:
    """Install a shared no-wheel filter on all spinboxes under *widget*."""
    filt = _no_wheel_filter()
    targets = []
    if isinstance(widget, QAbstractSpinBox):
        targets.append(widget)
    targets.extend(widget.findChildren(QAbstractSpinBox))
    for spin in targets:
        spin.installEventFilter(filt)
