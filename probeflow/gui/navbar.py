"""Top navigation bar widget."""

from __future__ import annotations

from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QCursor, QFont, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from probeflow.gui.config import (
    GITHUB_URL,
    GUI_FONT_DEFAULT,
    LOGO_NAV_PATH,
    normalise_gui_font_size,
)
from probeflow.gui.styling import NAVBAR_DARK_BG, NAVBAR_LIGHT_BG
from probeflow.gui.utils import _open_url


class Navbar(QWidget):
    theme_toggle_clicked = Signal()
    font_size_changed    = Signal(str)
    about_clicked        = Signal()

    def __init__(self, dark: bool, font_size_label: str = GUI_FONT_DEFAULT, parent=None):
        super().__init__(parent)
        self._dark            = dark
        self._font_size_label = normalise_gui_font_size(font_size_label)
        self._btns:           list[QPushButton] = []
        self.setFixedHeight(50)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(6)

        if LOGO_NAV_PATH.exists():
            self._logo_lbl = QLabel()
            self._logo_lbl.setStyleSheet("background: transparent;")
            pix = QPixmap(str(LOGO_NAV_PATH))
            self._logo_lbl.setPixmap(
                pix.scaled(9999, 46, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self._logo_lbl.setCursor(QCursor(Qt.PointingHandCursor))
            self._logo_lbl.mousePressEvent = lambda e: _open_url(GITHUB_URL)
            lay.addWidget(self._logo_lbl)

        title_lbl = QLabel("ProbeFlow")
        title_lbl.setFont(ui_font(12, weight=QFont.Bold))
        title_lbl.setStyleSheet("background: transparent;")
        lay.addWidget(title_lbl)
        lay.addStretch()
        self._font_size_actions: dict[str, QAction] = {}

        self._apply_nav_theme()

    def set_dark(self, dark: bool):
        self._dark = dark
        self._apply_nav_theme()

    def set_font_size(self, label: str):
        label = normalise_gui_font_size(label)
        if label == self._font_size_label:
            self._sync_font_size_button()
            return
        self._font_size_label = label
        self._sync_font_size_button()
        self.font_size_changed.emit(label)

    def _sync_font_size_button(self):
        for label, action in self._font_size_actions.items():
            action.setChecked(label == self._font_size_label)

    def _apply_nav_theme(self):
        if self._dark:
            self.setStyleSheet(
                f"background-color: {NAVBAR_DARK_BG};"
            )
            btn_qss = """
                QPushButton {
                    color: #ffffff;
                    background-color: transparent;
                    border: 2px solid rgba(255,255,255,0.6);
                    border-radius: 4px;
                    padding: 4px 14px;
                }
                QPushButton:hover {
                    background-color: rgba(255,255,255,0.18);
                }
            """
        else:
            self.setStyleSheet(
                f"background-color: {NAVBAR_LIGHT_BG};"
                "border-bottom: 2px solid #b0bec5;"
            )
            btn_qss = """
                QPushButton {
                    color: #1e1e2e;
                    background-color: #f0f2f5;
                    border: 2px solid #b0bec5;
                    border-radius: 4px;
                    padding: 4px 14px;
                }
                QPushButton:hover {
                    background-color: #e4edf8;
                    border-color: #3273dc;
                }
            """
        if hasattr(self, "_logo_lbl"):
            self._logo_lbl.setStyleSheet("background: transparent;")
        for btn in self._btns:
            btn.setStyleSheet(btn_qss)


__all__ = ["Navbar"]
