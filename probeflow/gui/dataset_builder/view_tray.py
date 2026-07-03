"""Collapsible view trays for Dataset Builder."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QFont
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from probeflow.gui.dataset_builder.display import percentile_value, value_to_percentile
from probeflow.gui.typography import ui_font
from probeflow.gui.viewer.histogram import HistogramPanel


class _BaseViewTray(QWidget):
    """Shared collapsible tray shell."""

    def __init__(self, theme: dict, title: str, parent=None):
        super().__init__(parent)
        self._theme = dict(theme)
        self._title = title

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._toggle_btn = QPushButton(f"> {title}")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.setFont(ui_font(9, weight=QFont.Bold))
        self._toggle_btn.setFixedHeight(28)
        self._toggle_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._toggle_btn.setObjectName("accentBtn")
        self._toggle_btn.toggled.connect(self._on_toggled)
        lay.addWidget(self._toggle_btn)

        self._body = QWidget()
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(2, 2, 0, 2)
        self._body_lay.setSpacing(6)
        lay.addWidget(self._body)
        self._body.setVisible(False)

    def is_expanded(self) -> bool:
        return self._toggle_btn.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self._toggle_btn.blockSignals(True)
        try:
            self._toggle_btn.setChecked(bool(expanded))
        finally:
            self._toggle_btn.blockSignals(False)
        self._on_toggled(bool(expanded))

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
        self._body.setStyleSheet("background: transparent;")
        self._apply_body_theme(theme, bg, fg, border, hover)

    def _apply_body_theme(self, theme: dict, bg: str, fg: str, border: str, hover: str) -> None:
        raise NotImplementedError

    def _on_toggled(self, checked: bool) -> None:
        self._body.setVisible(bool(checked))
        self._toggle_btn.setText(f"v {self._title}" if checked else f"> {self._title}")


class DatasetBuilderGlobalViewTray(_BaseViewTray):
    """Collapsible global display-controls tray for Dataset Builder."""

    flatten_toggled = Signal(bool)
    percentiles_changed = Signal(float, float)

    def __init__(self, theme: dict, parent=None):
        super().__init__(theme, "Global image view settings", parent)
        self._last_array = None

        self._hist_panel = HistogramPanel(parent=self)
        self._hist_panel.set_clip_controls_visible(False)
        self._hist_panel.set_threshold_mode(True)
        self._hist_panel.rangeReleased.connect(self._on_hist_range_released)
        self._body_lay.addWidget(self._hist_panel)

        pct_form = QFormLayout()
        pct_form.setContentsMargins(0, 0, 0, 0)
        pct_form.setSpacing(4)
        self._pct_min = self._make_percent_spinbox()
        self._pct_max = self._make_percent_spinbox()
        self._pct_min.setValue(1.0)
        self._pct_max.setValue(99.0)
        self._pct_min.valueChanged.connect(lambda _v: self._emit_percentiles())
        self._pct_max.valueChanged.connect(lambda _v: self._emit_percentiles())
        pct_form.addRow("Percentile min", self._pct_min)
        pct_form.addRow("Percentile max", self._pct_max)
        self._body_lay.addLayout(pct_form)

        self._flatten_btn = QPushButton("Global flatten")
        self._flatten_btn.setCheckable(True)
        self._flatten_btn.setFont(ui_font(9, weight=QFont.Bold))
        self._flatten_btn.setFixedHeight(28)
        self._flatten_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._flatten_btn.toggled.connect(lambda checked: self.flatten_toggled.emit(bool(checked)))
        self._body_lay.addWidget(self._flatten_btn)

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

    def percentile_bounds(self) -> tuple[float, float]:
        return float(self._pct_min.value()), float(self._pct_max.value())

    def set_percentile_bounds(self, lo: float, hi: float) -> None:
        self._pct_min.blockSignals(True)
        self._pct_max.blockSignals(True)
        try:
            self._pct_min.setValue(float(lo))
            self._pct_max.setValue(float(hi))
        finally:
            self._pct_min.blockSignals(False)
            self._pct_max.blockSignals(False)
        self._update_histogram()

    def set_array(self, arr) -> None:
        self._last_array = arr
        self._update_histogram()

    def clear_histogram(self, theme: dict) -> None:
        self._last_array = None
        self._hist_panel.clear(theme)

    def _make_percent_spinbox(self) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(0.0, 100.0)
        sb.setDecimals(1)
        sb.setSingleStep(1.0)
        sb.setSuffix(" %")
        sb.setFont(ui_font(9))
        return sb

    def _update_histogram(self) -> None:
        arr = self._last_array
        if arr is None:
            self._hist_panel.clear(self._theme)
            return
        lo, hi = self.percentile_bounds()
        try:
            value_lo = percentile_value(arr, lo)
            value_hi = percentile_value(arr, hi)
        except Exception:
            self._hist_panel.clear(self._theme)
            return
        self._hist_panel.render(
            arr,
            value_lo,
            value_hi,
            "a.u.",
            "Value",
            self._theme,
            scale=1.0,
        )
        self._hist_panel.set_clip_text(f"{lo:.1f}%  \u2192  {hi:.1f}%")

    def _emit_percentiles(self) -> None:
        lo = float(self._pct_min.value())
        hi = float(self._pct_max.value())
        if hi < lo:
            if self.sender() is self._pct_min:
                hi = lo
                self._pct_max.blockSignals(True)
                try:
                    self._pct_max.setValue(hi)
                finally:
                    self._pct_max.blockSignals(False)
            else:
                lo = hi
                self._pct_min.blockSignals(True)
                try:
                    self._pct_min.setValue(lo)
                finally:
                    self._pct_min.blockSignals(False)
        self._update_histogram()
        self.percentiles_changed.emit(lo, hi)

    def _on_hist_range_released(self, lo_value: float, hi_value: float) -> None:
        arr = self._last_array
        if arr is None:
            return
        try:
            lo = value_to_percentile(arr, lo_value)
            hi = value_to_percentile(arr, hi_value)
        except Exception:
            return
        lo = max(0.0, min(100.0, lo))
        hi = max(0.0, min(100.0, hi))
        if hi < lo:
            lo, hi = hi, lo
        self.set_percentile_bounds(lo, hi)
        self._update_histogram()
        self.percentiles_changed.emit(lo, hi)

    def _apply_body_theme(self, theme: dict, bg: str, fg: str, border: str, hover: str) -> None:
        self._hist_panel.setStyleSheet(f"background: transparent; color: {theme.get('fg', '#e6e8eb')};")
        self._pct_min.setStyleSheet(
            f"QDoubleSpinBox {{ background-color: {bg}; color: {fg}; border: 1px solid {border}; border-radius: 6px; padding: 3px 8px; }}"
            f"QDoubleSpinBox:hover {{ border-color: {hover}; }}"
        )
        self._pct_max.setStyleSheet(
            f"QDoubleSpinBox {{ background-color: {bg}; color: {fg}; border: 1px solid {border}; border-radius: 6px; padding: 3px 8px; }}"
            f"QDoubleSpinBox:hover {{ border-color: {hover}; }}"
        )
        self._sync_flatten_style()

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


class DatasetBuilderCurrentViewTray(_BaseViewTray):
    """Collapsible current-image display tools tray for Dataset Builder."""

    flatten_requested = Signal(bool)
    clear_requested = Signal()

    def __init__(self, theme: dict, parent=None):
        super().__init__(theme, "Current image view settings", parent)

        self._flatten_btn = QPushButton("3 point flatten")
        self._flatten_btn.setCheckable(True)
        self._flatten_btn.setFont(ui_font(9, weight=QFont.Bold))
        self._flatten_btn.setFixedHeight(28)
        self._flatten_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._flatten_btn.toggled.connect(lambda checked: self.flatten_requested.emit(bool(checked)))

        self._clear_btn = QPushButton("Clear current flatten")
        self._clear_btn.setFont(ui_font(9))
        self._clear_btn.setFixedHeight(28)
        self._clear_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._clear_btn.clicked.connect(lambda *_: self.clear_requested.emit())

        self._body_lay.addWidget(self._flatten_btn)
        self._body_lay.addWidget(self._clear_btn)

        self.apply_theme(theme)

    def is_flatten_armed(self) -> bool:
        return self._flatten_btn.isChecked()

    def set_flatten_armed(self, armed: bool) -> None:
        self._flatten_btn.blockSignals(True)
        try:
            self._flatten_btn.setChecked(bool(armed))
        finally:
            self._flatten_btn.blockSignals(False)
        self.apply_theme(self._theme)

    def _apply_body_theme(self, theme: dict, bg: str, fg: str, border: str, hover: str) -> None:
        accent = theme.get("accent_bg", "#4d8dff")
        accent_fg = theme.get("accent_fg", "#0c0e12")
        self._flatten_btn.setStyleSheet(
            "QPushButton {"
            f" background-color: {bg}; color: {fg};"
            f" border: 1px solid {border}; border-radius: 6px;"
            " padding: 4px 10px;"
            "}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
            f"QPushButton:checked {{ background-color: {accent}; color: {accent_fg}; border-color: {accent}; font-weight: 700; }}"
        )
        self._clear_btn.setStyleSheet(
            f"QPushButton {{ background-color: {bg}; color: {fg}; border: 1px solid {border}; border-radius: 6px; padding: 4px 10px; }}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
        )


DatasetBuilderViewTray = DatasetBuilderGlobalViewTray
