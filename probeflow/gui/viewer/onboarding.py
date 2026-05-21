"""Lightweight image-viewer help dialogs."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout


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
            "Fit / 1:1 buttons: reset view<br><br>"
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
            "Select a line ROI for profile and periodicity tools.<br>"
            "Select an area ROI for masking, ROI statistics, histogram, and FFT."
        )
        text.setWordWrap(True)
        text.setTextFormat(Qt.RichText)
        layout.addWidget(text)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(close)
