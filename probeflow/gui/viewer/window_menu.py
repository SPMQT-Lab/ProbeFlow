"""Window-menu helpers for the image viewer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QRect
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QMenu,
    QWidget,
)


@dataclass(frozen=True)
class ViewerWindowItem:
    """A visible ProbeFlow-owned window or floating tool."""

    label: str
    widget: QWidget


def _safe_visible(widget: QWidget) -> bool:
    try:
        return bool(widget.isVisible())
    except RuntimeError:
        return False


def _safe_title(widget: QWidget) -> str:
    try:
        return str(widget.windowTitle()).strip()
    except RuntimeError:
        return ""


def _is_owned_by_viewer(widget: QWidget, viewer: QWidget) -> bool:
    current = widget
    while current is not None:
        if current is viewer:
            return True
        try:
            parent = current.parent()
        except RuntimeError:
            return False
        current = parent if isinstance(parent, QWidget) else None
    return False


def _is_candidate_tool_window(widget: QWidget, viewer: QWidget) -> bool:
    if widget is viewer:
        return True
    if not _safe_visible(widget):
        return False
    if not _is_owned_by_viewer(widget, viewer):
        return False
    if isinstance(widget, QFileDialog | QMenu):
        return False
    return isinstance(widget, QDialog | QDockWidget)


def _is_arrangeable_window(widget: QWidget, viewer: QWidget) -> bool:
    if widget is viewer:
        return True
    if isinstance(widget, QDockWidget):
        return widget.isFloating()
    return widget.isWindow()


def _item_label(widget: QWidget, viewer: QWidget) -> str:
    title = _safe_title(widget)
    if widget is viewer:
        return f"Image viewer: {title}" if title else "Image viewer"
    if isinstance(widget, QDockWidget):
        return title or "Floating dock"
    return title or widget.__class__.__name__


def owned_viewer_windows(viewer: QWidget) -> list[ViewerWindowItem]:
    """Return visible top-level windows/tools owned by an image viewer."""
    widgets: list[QWidget] = [viewer]
    app = QApplication.instance()
    if app is not None:
        widgets.extend(widget for widget in app.topLevelWidgets() if isinstance(widget, QWidget))

    # Some floating docks are not consistently returned as top-level widgets
    # across Qt platforms, so include known viewer-owned dock attributes too.
    for attr in ("_lattice_grid_dock", "_roi_dock", "_measurement_dock"):
        dock = getattr(viewer, attr, None)
        if isinstance(dock, QDockWidget):
            widgets.append(dock)
    for dialog in viewer.findChildren(QDialog):
        dock = getattr(dialog, "_fft_lattice_dock", None)
        if isinstance(dock, QDockWidget):
            widgets.append(dock)

    seen: set[int] = set()
    items: list[ViewerWindowItem] = []
    for widget in widgets:
        try:
            marker = id(widget)
        except RuntimeError:
            continue
        if marker in seen or not _is_candidate_tool_window(widget, viewer):
            continue
        seen.add(marker)
        if widget is not viewer and isinstance(widget, QDockWidget) and not widget.isFloating():
            continue
        items.append(ViewerWindowItem(_item_label(widget, viewer), widget))

    if items:
        head = [item for item in items if item.widget is viewer]
        tail = sorted(
            (item for item in items if item.widget is not viewer),
            key=lambda item: item.label.casefold(),
        )
        return head + tail
    return []


def focus_window(widget: QWidget) -> None:
    """Show and raise a Qt window/tool if it is still alive."""
    try:
        if widget.isMinimized():
            widget.showNormal()
        else:
            widget.show()
        widget.raise_()
        widget.activateWindow()
    except RuntimeError:
        return


def bring_all_to_front(viewer: QWidget) -> None:
    for item in owned_viewer_windows(viewer):
        focus_window(item.widget)


def minimize_tool_windows(viewer: QWidget) -> None:
    for item in owned_viewer_windows(viewer):
        widget = item.widget
        if widget is viewer or not _is_arrangeable_window(widget, viewer):
            continue
        try:
            widget.showMinimized()
        except RuntimeError:
            continue


def restore_tool_windows(viewer: QWidget) -> None:
    for item in owned_viewer_windows(viewer):
        widget = item.widget
        if widget is viewer or not _is_arrangeable_window(widget, viewer):
            continue
        focus_window(widget)


def _arrangeable_windows(viewer: QWidget) -> list[QWidget]:
    return [
        item.widget for item in owned_viewer_windows(viewer)
        if _safe_visible(item.widget) and _is_arrangeable_window(item.widget, viewer)
    ]


def _available_geometry(viewer: QWidget) -> QRect:
    screen = viewer.screen() or QApplication.primaryScreen()
    if screen is None:
        return QRect(40, 40, 1200, 800)
    return screen.availableGeometry()


def tile_viewer_windows(viewer: QWidget) -> None:
    windows = _arrangeable_windows(viewer)
    if not windows:
        return
    geometry = _available_geometry(viewer)
    count = len(windows)
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))
    gap = 8
    cell_w = max(280, int((geometry.width() - gap * (cols - 1)) / cols))
    cell_h = max(220, int((geometry.height() - gap * (rows - 1)) / rows))

    for idx, widget in enumerate(windows):
        row, col = divmod(idx, cols)
        rect = QRect(
            geometry.x() + col * (cell_w + gap),
            geometry.y() + row * (cell_h + gap),
            cell_w,
            cell_h,
        )
        try:
            widget.showNormal()
            widget.setGeometry(rect)
            widget.raise_()
        except RuntimeError:
            continue


def cascade_viewer_windows(viewer: QWidget) -> None:
    windows = _arrangeable_windows(viewer)
    if not windows:
        return
    geometry = _available_geometry(viewer)
    offset = 32
    width = min(max(420, int(geometry.width() * 0.62)), max(320, geometry.width() - 24))
    height = min(max(320, int(geometry.height() * 0.70)), max(240, geometry.height() - 24))

    for idx, widget in enumerate(windows):
        wrapped = idx % max(1, min(8, len(windows)))
        x = geometry.x() + 12 + wrapped * offset
        y = geometry.y() + 12 + wrapped * offset
        if x + width > geometry.right():
            x = geometry.x() + 12
        if y + height > geometry.bottom():
            y = geometry.y() + 12
        try:
            widget.showNormal()
            widget.setGeometry(x, y, width, height)
            widget.raise_()
        except RuntimeError:
            continue


def populate_window_menu(menu: QMenu, viewer: QWidget) -> None:
    """Rebuild the image-viewer Window menu."""
    menu.clear()

    menu.addAction("Bring All to Front", lambda: bring_all_to_front(viewer))
    menu.addAction("Minimize This Viewer", viewer.showMinimized)
    menu.addAction("Minimize Tool Windows", lambda: minimize_tool_windows(viewer))
    menu.addAction("Restore Tool Windows", lambda: restore_tool_windows(viewer))
    menu.addSeparator()
    menu.addAction("Tile Visible Viewer Windows", lambda: tile_viewer_windows(viewer))
    menu.addAction("Cascade Visible Viewer Windows", lambda: cascade_viewer_windows(viewer))
    menu.addSeparator()

    header = menu.addAction("Open Windows and Tools")
    header.setEnabled(False)

    items = owned_viewer_windows(viewer)
    if not items:
        empty = menu.addAction("No open viewer windows")
        empty.setEnabled(False)
        return

    for item in items:
        action = menu.addAction(item.label)
        action.triggered.connect(
            lambda _checked=False, widget=item.widget: focus_window(widget)
        )
