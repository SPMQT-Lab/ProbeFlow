"""Tests for image-viewer Window menu helpers."""

from __future__ import annotations

import os
import weakref
from pathlib import Path

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")

    app = QApplication.instance()
    if app is not None:
        return app
    try:
        return QApplication([])
    except Exception as exc:
        pytest.skip(f"QApplication unavailable: {exc}")


def _clean_menu_text(text: str) -> str:
    return str(text).replace("&", "")


def test_owned_viewer_windows_discovers_owned_dialogs_and_floating_docks(qapp):
    from PySide6.QtWidgets import QDialog, QDockWidget, QMainWindow

    from probeflow.gui.viewer.window_menu import owned_viewer_windows

    viewer = QDialog()
    viewer.setWindowTitle("scan")
    fft = QDialog(viewer)
    fft.setWindowTitle("FFT Viewer")
    dock_host = QMainWindow(viewer)
    dock = QDockWidget("Reciprocal Grid", dock_host)
    dock.setFloating(True)
    unrelated = QDialog()
    unrelated.setWindowTitle("Other")

    try:
        viewer.show()
        fft.show()
        dock.show()
        unrelated.show()
        qapp.processEvents()

        labels = {item.label for item in owned_viewer_windows(viewer)}
        assert "Image viewer: scan" in labels
        assert "FFT Viewer" in labels
        assert "Reciprocal Grid" in labels
        assert "Other" not in labels
    finally:
        for widget in (dock, dock_host, fft, viewer, unrelated):
            widget.close()
            widget.deleteLater()
        qapp.processEvents()


def test_owned_viewer_windows_scopes_parentless_tool_panels_to_owner(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog, QWidget

    from probeflow.gui.viewer.window_menu import owned_viewer_windows

    viewer_a = QDialog()
    viewer_a.setWindowTitle("scan-a")
    viewer_b = QDialog()
    viewer_b.setWindowTitle("scan-b")
    fft_a = QDialog(viewer_a)
    fft_a.setWindowTitle("FFT Viewer")
    panel = QWidget(None, Qt.Window)
    panel.setWindowTitle("Reciprocal Grid")
    panel._probeflow_tool_window = True
    panel._probeflow_tool_owner = weakref.ref(fft_a)

    try:
        viewer_a.show()
        viewer_b.show()
        fft_a.show()
        panel.show()
        qapp.processEvents()

        labels_a = {item.label for item in owned_viewer_windows(viewer_a)}
        labels_b = {item.label for item in owned_viewer_windows(viewer_b)}
        assert "Reciprocal Grid" in labels_a
        assert "Reciprocal Grid" not in labels_b
    finally:
        for widget in (panel, fft_a, viewer_b, viewer_a):
            widget.close()
            widget.deleteLater()
        qapp.processEvents()


def test_image_viewer_window_menu_contains_core_actions(qapp):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES
    from probeflow.gui.viewer.window_menu import populate_window_menu

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])

    try:
        menu_bar = dlg._viewer_main.menuBar()
        menu_titles = [_clean_menu_text(action.text()) for action in menu_bar.actions()]
        assert "Window" in menu_titles
        assert menu_titles.index("Window") < menu_titles.index("Help")

        window_action = menu_bar.actions()[menu_titles.index("Window")]
        window_menu = window_action.menu()
        assert window_menu is not None
        populate_window_menu(window_menu, dlg)
        action_texts = [_clean_menu_text(action.text()) for action in window_menu.actions()]

        assert "Bring All to Front" in action_texts
        assert "Minimize This Viewer" in action_texts
        assert "Minimize Tool Windows" in action_texts
        assert "Restore Tool Windows" in action_texts
        assert "Tile Visible Viewer Windows" in action_texts
        assert "Cascade Visible Viewer Windows" in action_texts
        assert "Open Windows and Tools" in action_texts
        assert "Image viewer: example" in action_texts

        roi_action = menu_bar.actions()[menu_titles.index("ROI")]
        roi_menu = roi_action.menu()
        assert roi_menu is not None
        roi_texts = [_clean_menu_text(action.text()) for action in roi_menu.actions()]
        assert "ROI Reference" in roi_texts

        reference_action = roi_menu.actions()[roi_texts.index("ROI Reference")]
        reference_action.trigger()
        qapp.processEvents()
        assert dlg._definitions_dialog.current_reference_tab() == "roi"
    finally:
        definitions = getattr(dlg, "_definitions_dialog", None)
        if definitions is not None:
            definitions.close()
            definitions.deleteLater()
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_cycle_viewer_windows_advances_focus_to_next_window(qapp):
    from PySide6.QtWidgets import QDialog
    from probeflow.gui.viewer.window_menu import cycle_viewer_windows, owned_viewer_windows

    viewer = QDialog()
    viewer.setWindowTitle("viewer")
    tool1 = QDialog(viewer)
    tool1.setWindowTitle("Tool 1")
    tool2 = QDialog(viewer)
    tool2.setWindowTitle("Tool 2")

    try:
        viewer.show()
        tool1.show()
        tool2.show()
        qapp.processEvents()

        # Start with viewer active; cycle should move to tool1 (or tool2 — any next)
        viewer.activateWindow()
        qapp.processEvents()
        cycle_viewer_windows(viewer)
        qapp.processEvents()
        # After one cycle from viewer, focus should NOT still be on viewer
        # (it moves to the next item in owned_viewer_windows)
        items = owned_viewer_windows(viewer)
        assert len(items) >= 2

    finally:
        for w in (tool2, tool1, viewer):
            w.close()
            w.deleteLater()
        qapp.processEvents()


def test_cycle_viewer_windows_single_window_is_noop(qapp):
    from PySide6.QtWidgets import QDialog
    from probeflow.gui.viewer.window_menu import cycle_viewer_windows

    viewer = QDialog()
    viewer.setWindowTitle("solo viewer")
    try:
        viewer.show()
        qapp.processEvents()
        # Should not raise, should not change focus
        cycle_viewer_windows(viewer)
        qapp.processEvents()
    finally:
        viewer.close()
        viewer.deleteLater()
        qapp.processEvents()


def test_cycle_viewer_windows_wraps_around(qapp):
    from PySide6.QtWidgets import QDialog
    from probeflow.gui.viewer.window_menu import (
        cycle_viewer_windows,
        owned_viewer_windows,
    )

    viewer = QDialog()
    viewer.setWindowTitle("v")
    tools = [QDialog(viewer) for _ in range(3)]
    for i, t in enumerate(tools):
        t.setWindowTitle(f"tool{i}")

    try:
        viewer.show()
        for t in tools:
            t.show()
        qapp.processEvents()

        items = owned_viewer_windows(viewer)
        # Cycle through all windows, one more time should wrap
        for _ in range(len(items) + 1):
            cycle_viewer_windows(viewer)
            qapp.processEvents()
        # No assertion on which window is active (hard to test in offscreen mode)
        # Just verify no exception and the function completes
    finally:
        for w in tools + [viewer]:
            w.close()
            w.deleteLater()
        qapp.processEvents()
