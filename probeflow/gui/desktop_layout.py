"""Desktop layout persistence helpers for ProbeFlow Qt windows."""

from __future__ import annotations

import base64

from PySide6.QtCore import QByteArray
from PySide6.QtGui import QGuiApplication


def qbytearray_to_b64(data: QByteArray) -> str:
    return base64.b64encode(bytes(data)).decode("ascii")


def b64_to_qbytearray(text: str) -> QByteArray:
    return QByteArray(base64.b64decode(text.encode("ascii")))


def apply_screen_fraction_geometry(widget, scale: float) -> None:
    """Center *widget* on the active screen using a fraction of available space."""
    screen = widget.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        return
    available = screen.availableGeometry()
    w = max(widget.minimumWidth(), int(available.width() * scale))
    h = max(widget.minimumHeight(), int(available.height() * scale))
    w = min(w, available.width())
    h = min(h, available.height())
    x = available.x() + int((available.width() - w) / 2)
    y = available.y() + int((available.height() - h) / 2)
    widget.setGeometry(x, y, w, h)


def geometry_is_visible(widget) -> bool:
    frame = widget.frameGeometry()
    for screen in QGuiApplication.screens():
        if screen.availableGeometry().intersects(frame):
            return True
    return False


def restore_geometry_or_default(widget, geometry_b64: str | None, scale: float) -> bool:
    """Restore saved geometry, falling back to a centered screen-fraction size."""
    restored = False
    if geometry_b64:
        try:
            restored = bool(widget.restoreGeometry(b64_to_qbytearray(geometry_b64)))
        except Exception:
            restored = False
    if not restored or not geometry_is_visible(widget):
        apply_screen_fraction_geometry(widget, scale)
        return False
    return True
