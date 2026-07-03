"""Collapsible View tray for Dataset Builder."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QFont
from PySide6.QtWidgets import QVBoxLayout, QPushButton, QWidget

from probeflow.gui.typography import ui_font
from probeflow.gui.viewer.histogram import HistogramPanel


class DatasetBuilderViewTray(QWidget):
    """Collapsible display-controls tray for Dataset Builder."""

    flatten_toggled = Signal(bool)

    def __init__(self, theme: dict, parent=None):
        super().__init__(parent)
        self._theme = dict(theme)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._toggle_btn = QPushButton("> View")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.setFont(ui_font(9, weight=QFont.Bold))
        self._toggle_btn.setFixedHeight(28)
        self._toggle_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._toggle_btn.setObjectName("accentBtn")
        self._toggle_btn.toggled.connect(self._on_toggled)
        lay.addWidget(self._toggle_btn)

        self._body = QWidget()
        body_lay = QVBoxLayout(self._body)
        body_lay.setContentsMargins(2, 2, 0, 2)
        body_lay.setSpacing(6)

        self._hist_panel = HistogramPanel(parent=self)
        body_lay.addWidget(self._hist_panel)

        self._flatten_btn = QPushButton("Global flatten")
        self._flatten_btn.setCheckable(True)
        self._flatten_btn.setFont(ui_font(9, weight=QFont.Bold))
        self._flatten_btn.setFixedHeight(28)
        self._flatten_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._flatten_btn.toggled.connect(lambda checked: self.flatten_toggled.emit(checked))
        body_lay.addWidget(self._flatten_btn)

        lay.addWidget(self._body)
        self._body.setVisible(False)

        self.apply_theme(theme)

    @property
    def hist_panel(self) -> HistogramPanel:
        return self._hist_panel

    def is_flatten_enabled(self) -> bool:
        return self._flatten_btn.isChecked()

    def set_flatten_enabled(self, enabled: bool) -> None:
        self._flatten_btn.blockSignals(True)
        try:
            self._flatten_btn.setChecked(bool(enabled))
        finally:
            self._flatten_btn.blockSignals(False)
        self._sync_flatten_style()

    def is_expanded(self) -> bool:
        return self._toggle_btn.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self._toggle_btn.blockSignals(True)
        try:
            self._toggle_btn.setChecked(bool(expanded))
        finally:
            self._toggle_btn.blockSignals(False)
        self._on_toggled(bool(expanded))

    def update_histogram(self, arr, vmin: float, vmax: float, theme: dict) -> None:
        if arr is None:
            self._hist_panel.clear(theme)
            return
        import numpy as np

        finite = np.asarray(arr, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            self._hist_panel.clear(theme)
            return
        self._hist_panel.render(finite, vmin, vmax, "", "Value", theme, scale=1.0)

    def clear_histogram(self, theme: dict) -> None:
        self._hist_panel.clear(theme)

    def apply_theme(self, theme: dict) -> None:
        self._theme = dict(theme)
        accent = theme.get("accent_bg", "#4d8dff")
        accent_fg = theme.get("accent_fg", "#0c0e12")
        bg = theme.get("btn_bg", theme.get("main_bg", "#16181d"))
        fg = theme.get("sub_fg", "#9aa1ab")
        border = theme.get("border", "#3a414c")
        hover = theme.get("hover", bg)

        self._toggle_btn.setStyleSheet(
            "QPushButton {"
            f" background-color: {accent}; color: {accent_fg};"
            f" border: 1px solid {accent}; border-radius: 6px;"
            " padding: 4px 10px; font-weight: 700;"
            "}"
            f"QPushButton:hover {{ background-color: {accent}; }}"
        )
        self._sync_flatten_style()
        self._body.setStyleSheet("background: transparent;")
        self._hist_panel.setStyleSheet(f"background: transparent; color: {theme.get('fg', '#e6e8eb')};")

    def _on_toggled(self, checked: bool) -> None:
        self._body.setVisible(bool(checked))
        self._toggle_btn.setText("v View" if checked else "> View")

    def _sync_flatten_style(self) -> None:
        theme = self._theme
        accent = theme.get("accent_bg", "#4d8dff")
        accent_fg = theme.get("accent_fg", "#0c0e12")
        bg = theme.get("btn_bg", theme.get("main_bg", "#16181d"))
        fg = theme.get("sub_fg", "#9aa1ab")
        border = theme.get("border", "#3a414c")
        hover = theme.get("hover", bg)
        self._flatten_btn.setStyleSheet(
            "QPushButton {"
            f" background-color: {bg}; color: {fg};"
            f" border: 1px solid {border}; border-radius: 6px;"
            " padding: 4px 10px;"
            "}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
            f"QPushButton:checked {{ background-color: {accent}; color: {accent_fg}; border-color: {accent}; font-weight: 700; }}"
        )
