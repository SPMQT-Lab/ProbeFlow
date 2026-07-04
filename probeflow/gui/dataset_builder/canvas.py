"""Dataset Builder-specific image canvas."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal

from probeflow.gui.image_canvas import ImageCanvas


class DatasetBuilderCanvas(ImageCanvas):
    """Image canvas that can emit mask-paint stamps without changing ImageCanvas."""

    paint_stroke_started = Signal()
    paint_stroke_finished = Signal()
    mask_painted = Signal(int, int)
    quickseg_click_requested = Signal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paint_enabled = False
        self._quickseg_enabled = False
        self._rgba_overlay_item = None

    def set_paint_enabled(self, enabled: bool) -> None:
        self._paint_enabled = bool(enabled)
        self.setCursor(Qt.CrossCursor if self._paint_enabled else Qt.ArrowCursor)

    def set_quickseg_enabled(self, enabled: bool) -> None:
        self._quickseg_enabled = bool(enabled)
        if self._quickseg_enabled:
            self.setCursor(Qt.CrossCursor)
        elif not self._paint_enabled:
            self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event) -> None:
        if getattr(self, "_set_zero_mode", False):
            super().mousePressEvent(event)
            return
        if self._quickseg_enabled and event.button() == Qt.LeftButton:
            image_size = getattr(self, "_image_size", None)
            if image_size is not None:
                width, height = image_size
                pos = self.mapToScene(event.pos())
                x = int(round(pos.x()))
                y = int(round(pos.y()))
                if 0 <= x < int(width) and 0 <= y < int(height):
                    self.quickseg_click_requested.emit(x, y, int(event.modifiers()))
                    event.accept()
                    return
        if self._paint_enabled and event.button() == Qt.LeftButton:
            self.paint_stroke_started.emit()
            self._emit_paint(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if getattr(self, "_set_zero_mode", False):
            super().mouseMoveEvent(event)
            return
        if self._paint_enabled and event.buttons() & Qt.LeftButton:
            self._emit_paint(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if getattr(self, "_set_zero_mode", False):
            super().mouseReleaseEvent(event)
            return
        if self._paint_enabled and event.button() == Qt.LeftButton:
            self.paint_stroke_finished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _emit_paint(self, event) -> None:
        image_size = getattr(self, "_image_size", None)
        if image_size is None:
            return
        width, height = image_size
        pos = self.mapToScene(event.pos())
        x = int(round(pos.x()))
        y = int(round(pos.y()))
        if 0 <= x < int(width) and 0 <= y < int(height):
            self.mask_painted.emit(x, y)

    def set_rgba_overlay(self, rgba) -> None:
        self.clear_rgba_overlay()
        if rgba is None:
            return
        import numpy as np
        from PySide6.QtGui import QImage, QPixmap

        arr = np.asarray(rgba, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 4 or not arr.any():
            return
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, arr.strides[0], QImage.Format_RGBA8888).copy()
        item = self.scene().addPixmap(QPixmap.fromImage(qimg))
        item.setZValue(20)
        item.setAcceptedMouseButtons(Qt.NoButton)
        self._rgba_overlay_item = item

    def clear_rgba_overlay(self) -> None:
        if self._rgba_overlay_item is not None:
            self.scene().removeItem(self._rgba_overlay_item)
            self._rgba_overlay_item = None
