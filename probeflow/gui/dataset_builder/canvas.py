"""Dataset Builder-specific image canvas."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal

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
        self._quickseg_click_pending: tuple[int, int, int] | None = None
        self._quickseg_click_press_pos = None
        self._right_pan_start = None

    def set_paint_enabled(self, enabled: bool) -> None:
        self._paint_enabled = bool(enabled)
        self.setCursor(Qt.CrossCursor if self._paint_enabled else Qt.ArrowCursor)

    def set_quickseg_enabled(self, enabled: bool) -> None:
        self._quickseg_enabled = bool(enabled)
        if not self._quickseg_enabled:
            self._quickseg_click_pending = None
            self._quickseg_click_press_pos = None
        if self._quickseg_enabled:
            self.setCursor(Qt.CrossCursor)
        elif not self._paint_enabled:
            self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.MouseFocusReason)
        if getattr(self, "_set_zero_mode", False):
            super().mousePressEvent(event)
            return
        if event.button() == Qt.RightButton:
            self._right_pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self._quickseg_enabled and event.button() == Qt.LeftButton:
            image_size = getattr(self, "_image_size", None)
            if image_size is not None:
                width, height = image_size
                pos = self.mapToScene(event.pos())
                x = int(round(pos.x()))
                y = int(round(pos.y()))
                if 0 <= x < int(width) and 0 <= y < int(height):
                    self._quickseg_click_press_pos = event.pos()
                    mods = getattr(event.modifiers(), "value", event.modifiers())
                    self._quickseg_click_pending = (x, y, int(mods))
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
        if self._right_pan_start is not None and event.buttons() & Qt.RightButton:
            delta = event.pos() - self._right_pan_start
            self._right_pan_start = event.pos()
            self._scroll_by(-delta.x(), -delta.y())
            event.accept()
            return
        if self._quickseg_click_pending is not None and event.buttons() & Qt.LeftButton:
            press_pos = self._quickseg_click_press_pos
            if press_pos is not None:
                delta = event.pos() - press_pos
                if abs(delta.x()) + abs(delta.y()) > 4:
                    self._quickseg_click_pending = None
                    self._quickseg_click_press_pos = None
        if self._paint_enabled and event.buttons() & Qt.LeftButton:
            self._emit_paint(event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if getattr(self, "_set_zero_mode", False):
            super().mouseReleaseEvent(event)
            return
        if event.button() == Qt.RightButton and self._right_pan_start is not None:
            self._right_pan_start = None
            self.setCursor(Qt.CrossCursor if self._quickseg_enabled or self._paint_enabled else Qt.ArrowCursor)
            event.accept()
            return
        if self._paint_enabled and event.button() == Qt.LeftButton:
            self.paint_stroke_finished.emit()
            event.accept()
            return
        if self._quickseg_enabled and event.button() == Qt.LeftButton:
            pending = self._quickseg_click_pending
            self._quickseg_click_pending = None
            self._quickseg_click_press_pos = None
            if pending is not None:
                self.quickseg_click_requested.emit(*pending)
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def viewportEvent(self, event) -> bool:
        if event.type() == QEvent.Type.Wheel:
            self._zoom_from_wheel(event)
            return True
        return super().viewportEvent(event)

    def wheelEvent(self, event) -> None:
        self._zoom_from_wheel(event)

    def _zoom_from_wheel(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        self.setFocus(Qt.MouseFocusReason)
        self.zoom_by(1.12 if delta > 0 else 1 / 1.12)
        event.accept()

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
