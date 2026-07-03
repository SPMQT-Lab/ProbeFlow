"""Dataset Builder-specific image canvas."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal

from probeflow.gui.image_canvas import ImageCanvas


class DatasetBuilderCanvas(ImageCanvas):
    """Image canvas that can emit mask-paint stamps without changing ImageCanvas."""

    mask_painted = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paint_enabled = False

    def set_paint_enabled(self, enabled: bool) -> None:
        self._paint_enabled = bool(enabled)
        self.setCursor(Qt.CrossCursor if self._paint_enabled else Qt.ArrowCursor)

    def mousePressEvent(self, event) -> None:
        if getattr(self, "_set_zero_mode", False):
            super().mousePressEvent(event)
            return
        if self._paint_enabled and event.button() == Qt.LeftButton:
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
