"""Tests for image-viewer Window menu helpers."""

from __future__ import annotations

import os
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
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()
