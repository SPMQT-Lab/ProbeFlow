"""Lightweight image-viewer help dialogs."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

from probeflow.gui.viewer.shortcuts import viewer_command


def _keys(command_id: str) -> str:
    return " / ".join(viewer_command(command_id).shortcuts)


class ImageViewerShortcutsDialog(QDialog):
    """Compact mouse and keyboard reference for the image viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image viewer shortcuts")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)

        text = QLabel(
            "<b>Image navigation</b><br>"
            "Drag blank image: pan<br>"
            "Ctrl+scroll: zoom<br>"
            f"{_keys('view.fit')}: fit image<br>"
            f"{_keys('view.one_to_one')}: 1:1 view<br><br>"
            "<b>Panels and docks</b><br>"
            f"{_keys('panel.view')} / {_keys('panel.process')} / {_keys('panel.measure')}: switch sidebar tabs<br>"
            f"{_keys('dock.roi_manager')}: ROI Manager dock<br>"
            f"{_keys('dock.measurements')}: Measurements dock<br><br>"
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
            "Ctrl+C / Ctrl+V: copy and paste ROI<br><br>"
            "<b>Drawing tools</b><br>"
            "Rectangle/Ellipse/Line: drag to draw<br>"
            "Polygon: click vertices, double-click or Enter to finish<br>"
            "Esc: cancel current drawing tool<br><br>"
            "<b>Measurements</b><br>"
            f"{_keys('measure.distance')}: distance/ruler<br>"
            f"{_keys('measure.roi_stats')}: ROI statistics<br>"
            f"{_keys('measure.line_profile')}: add line profile<br>"
            f"{_keys('measure.line_periodicity')}: estimate periodicity<br>"
            "Select a line ROI for profile and periodicity tools.<br>"
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
