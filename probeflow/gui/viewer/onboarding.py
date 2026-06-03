"""Lightweight image-viewer help dialogs."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

from probeflow.gui.viewer.shortcuts import (
    display_shortcuts_for_all_platforms,
    viewer_command,
)


def _keys(command_id: str) -> str:
    return display_shortcuts_for_all_platforms(viewer_command(command_id).shortcuts)


def _shortcut(shortcut: str) -> str:
    return display_shortcuts_for_all_platforms((shortcut,))


class ImageViewerShortcutsDialog(QDialog):
    """Compact mouse and keyboard reference for the image viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image viewer shortcuts")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)

        text = QLabel(
            "<b>Image navigation</b><br>"
            "Drag image: pan · click an ROI to select<br>"
            "⌘+scroll / Ctrl+scroll: zoom<br>"
            f"<b>{_keys('viewer.command_finder')}: command finder</b> — search &amp; run any action<br>"
            f"{_keys('view.fit')}: fit image<br>"
            f"{_keys('view.one_to_one')}: 1:1 view<br><br>"
            "<b>Panels</b><br>"
            f"View: {_keys('panel.view')}; Process: {_keys('panel.process')}; "
            f"ROI: {_keys('panel.roi')}; Measure: {_keys('panel.measure')}; "
            f"Export: {_keys('panel.export')}: switch sidebar tabs<br><br>"
            "<b>Processing</b><br>"
            f"{_keys('view.auto_contrast')}: auto contrast<br>"
            f"{_keys('processing.apply')}: apply processing<br>"
            f"{_keys('processing.reset')}: reset processing<br>"
            f"{_keys('processing.plane_background')}: plane/background subtraction<br>"
            f"{_keys('processing.stm_background')}: STM background<br>"
            f"{_keys('processing.bad_lines')}: bad scan-line correction<br><br>"
            "<b>ROIs</b><br>"
            "Click ROI: select<br>"
            "Drag active ROI: move<br>"
            "Right-click ROI: object actions<br>"
            "Delete/Backspace: delete active ROI<br>"
            f"Copy ROI: {_shortcut('Ctrl+C')}; paste ROI: {_shortcut('Ctrl+V')}<br><br>"
            "<b>Drawing tools</b> (press the key, then draw)<br>"
            "R rectangle · E ellipse · L line · P point · G polygon · F freehand<br>"
            "Rectangle/Ellipse/Line: drag to draw; Polygon: click vertices, "
            "Enter/double-click to finish<br>"
            "Esc: cancel drawing and return to the cursor<br><br>"
            "<b>Measurements</b><br>"
            f"{_keys('measure.distance')}: distance/ruler<br>"
            f"{_keys('measure.roi_stats')}: ROI statistics<br>"
            f"{_keys('measure.line_profile')}: add line profile<br>"
            f"{_keys('measure.line_periodicity')}: estimate spacing and save known structures<br>"
            f"{_keys('fft.open')}: fit FFT Bragg peaks and undistort<br>"
            "Select a line ROI for profile and spacing tools.<br>"
            "Select an area ROI for masking, ROI statistics, histogram, and FFT.<br><br>"
            "<b>Export</b><br>"
            f"{_keys('export.save_png')}: save PNG copy<br>"
            f"{_keys('export.save_processed')}: save processed image"
        )
        text.setWordWrap(True)
        text.setTextFormat(Qt.RichText)
        layout.addWidget(text)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(close)
