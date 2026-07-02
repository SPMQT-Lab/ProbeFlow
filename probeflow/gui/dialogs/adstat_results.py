"""Qt renderer for AdStat result view specifications."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QLabel,
    QScrollArea,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


_CURVE_OBSERVED_COLOR = "#ff9f1c"
_CURVE_MODEL_COLOR = "#7cc7ff"
_CURVE_BAND_COLOR = "#5ea3ff"


class AdStatResultView(QWidget):
    """Reusable renderer for an AdStat ``ResultViewSpec``."""

    def __init__(
        self,
        view_spec: Any,
        *,
        source_label: str = "",
        theme: dict | None = None,
        data_mode: str = "real",
        show_banner: bool = True,
        show_panels: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._spec = view_spec
        self._source_label = str(source_label or "")
        self._theme = theme or {}
        self._data_mode = _normalise_data_mode(data_mode)
        self._show_banner = bool(show_banner)
        self._show_panels = bool(show_panels)
        self._show_technical_details = True
        self._title: QLabel | None = None
        self._banner: QLabel | None = None
        self._tabs = QTabWidget(self)
        self._build()

    @property
    def data_mode(self) -> str:
        """Either ``real`` or ``sandbox``."""

        return self._data_mode

    @property
    def tab_count(self) -> int:
        """Number of tabs currently rendered, exposed for GUI tests."""

        return self._tabs.count()

    @property
    def tab_titles(self) -> tuple[str, ...]:
        """Rendered tab titles, exposed for contract-style GUI tests."""

        return tuple(self._tabs.tabText(index) for index in range(self._tabs.count()))

    @property
    def banner_text(self) -> str:
        """Visible generated-data banner text, exposed for GUI tests."""

        if self._banner is None or not self._show_banner or self._data_mode != "sandbox":
            return ""
        return self._banner.text()

    @property
    def technical_details_visible(self) -> bool:
        """Whether the diagnostics table tab is currently allowed to render."""

        return bool(self._show_technical_details)

    def set_technical_details_visible(self, visible: bool) -> None:
        """Show or hide the diagnostics tab without changing the result spec."""

        visible = bool(visible)
        if self._show_technical_details == visible:
            return
        self._show_technical_details = visible
        self._refresh()

    def set_view_spec(
        self,
        view_spec: Any,
        *,
        source_label: str | None = None,
        data_mode: str | None = None,
    ) -> None:
        """Replace the rendered spec without rebuilding the owning dialog."""

        self._spec = view_spec
        if source_label is not None:
            self._source_label = str(source_label or "")
        if data_mode is not None:
            self._data_mode = _normalise_data_mode(data_mode)
        self._refresh()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._banner = QLabel("TEST MODE - GENERATED DATA")
        self._banner.setObjectName("adstatSandboxBanner")
        self._banner.setAlignment(Qt.AlignCenter)
        self._banner.setStyleSheet(
            "background: #f59f00; color: #1f1300; font-weight: 800; "
            "padding: 6px; border: 1px solid #b36b00;"
        )
        layout.addWidget(self._banner)

        self._title = QLabel()
        self._title.setObjectName("dialogTitle")
        self._title.setStyleSheet("font-weight: 700;")
        layout.addWidget(self._title)

        self._tabs.setDocumentMode(True)
        layout.addWidget(self._tabs, 1)
        self._refresh()

    def _refresh(self) -> None:
        if self._banner is not None:
            self._banner.setVisible(self._show_banner and self._data_mode == "sandbox")
        if self._title is not None:
            self._title.setText(self._title_text())

        self._clear_tabs()
        self._tabs.addTab(self._summary_tab(), "Summary")
        if self._show_technical_details and tuple(_field(self._spec, "verdict_rows", ()) or ()):
            self._tabs.addTab(self._technical_details_tab(), "Technical details")
        # Embedded in the Particle Statistics dialog the per-statistic plots and the
        # point-pattern panel duplicate the always-visible top panel and left field, so
        # they are suppressed there (show_panels=False); the standalone results dialog
        # still renders them.
        if self._show_panels:
            for index, panel in enumerate(tuple(_field(self._spec, "panels", ()) or ())):
                tab_title = _short_tab_title(_field(panel, "title", "") or f"Panel {index + 1}")
                self._tabs.addTab(self._panel_tab(panel), tab_title)

    def _clear_tabs(self) -> None:
        while self._tabs.count():
            widget = self._tabs.widget(0)
            self._tabs.removeTab(0)
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _title_text(self) -> str:
        if self._data_mode == "sandbox":
            if self._source_label:
                return f"Generated examples - Particle Statistics results - {self._source_label}"
            return "Generated examples - Particle Statistics results"
        if self._source_label:
            return f"Particle Statistics results - {self._source_label}"
        return "Particle Statistics results"

    def _summary_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        rows = tuple(_field(self._spec, "verdict_rows", ()) or ())
        if rows:
            layout.addWidget(_section_label("Model summary"))
            layout.addWidget(_model_summary_widget(rows))
            note = QLabel(
                "The same model can appear several times because Particle Statistics "
                "tests one null model with several statistics. Read the group as one "
                "model assumption checked from multiple angles."
            )
            note.setWordWrap(True)
            note.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(note)

        status_lines = tuple(_field(self._spec, "status_lines", ()) or ())
        if status_lines:
            layout.addWidget(_section_label("Diagnostics"))
            status = QLabel("\n".join(str(line) for line in status_lines))
            status.setWordWrap(True)
            status.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(status)

        explainer = _field(self._spec, "explainer", None)
        if explainer is not None:
            layout.addWidget(_section_label("Model"))
            explanation = QLabel(_explainer_text(explainer))
            explanation.setWordWrap(True)
            explanation.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(explanation)

        if not rows and not status_lines and explainer is None:
            layout.addWidget(QLabel("No summary rows available."))
        layout.addStretch(1)
        return _scrollable(page)

    def _technical_details_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        rows = tuple(_field(self._spec, "verdict_rows", ()) or ())
        layout.addWidget(_section_label("Raw AdStat verdict rows"))
        detail = QLabel("These are the engine-facing model and statistic ids used for reproducibility.")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        if rows:
            layout.addWidget(_table_widget(_verdict_columns(rows), rows))
        else:
            layout.addWidget(QLabel("No raw verdict rows available."))
        layout.addStretch(1)
        return _scrollable(page)

    def _panel_tab(self, panel: Any) -> QWidget:
        kind = str(_field(panel, "kind", ""))
        if kind == "table" or _field(panel, "table_rows", None):
            return self._table_panel_tab(panel)
        return self._plot_panel_tab(panel)

    def _table_panel_tab(self, panel: Any) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        layout.addWidget(_caption_label(panel))
        rows = tuple(_field(panel, "table_rows", ()) or ())
        columns = tuple(_field(panel, "table_columns", ()) or ())
        if rows:
            layout.addWidget(_table_widget(columns, rows))
        else:
            message = _field(_field(panel, "metadata", {}) or {}, "empty_message", None)
            layout.addWidget(QLabel(str(message or "No rows available.")))
        layout.addStretch(1)
        return _scrollable(page)

    def _plot_panel_tab(self, panel: Any) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        plot = AdStatPlotWidget(
            panel,
            theme=self._theme,
            data_mode=self._data_mode,
            parent=page,
        )
        layout.addWidget(plot, 1)
        layout.addWidget(_caption_label(panel))
        return page


class AdStatResultsDialog(QDialog):
    """Render a real-data AdStat ``ResultViewSpec`` in ProbeFlow's Qt shell."""

    def __init__(
        self,
        view_spec: Any,
        *,
        source_label: str = "",
        theme: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Particle Statistics results")
        self.resize(980, 680)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self._view = AdStatResultView(
            view_spec,
            source_label=source_label,
            theme=theme,
            data_mode="real",
            parent=self,
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

    @property
    def tab_count(self) -> int:
        """Number of tabs currently rendered, exposed for GUI tests."""

        return self._view.tab_count

    @property
    def tab_titles(self) -> tuple[str, ...]:
        """Rendered tab titles, exposed for contract-style GUI tests."""

        return self._view.tab_titles


class AdStatPlotWidget(QWidget):
    """ProbeFlow-native Qt renderer for one AdStat plot panel."""

    def __init__(
        self,
        panel: Any,
        *,
        theme: dict | None = None,
        data_mode: str = "real",
        curve_mode: str = "comparison",
        show_observed_curve: bool = True,
        show_model_curves: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._panel = panel
        self._theme = theme or {}
        self._data_mode = _normalise_data_mode(data_mode)
        self._curve_mode = _normalise_curve_mode(curve_mode)
        self._show_observed_curve = bool(show_observed_curve)
        self._show_model_curves = bool(show_model_curves)
        self.setObjectName("adstatResultPlot")
        self.setMinimumHeight(320)
        self._cursor_pos: QPointF | None = None
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        try:
            self._cursor_pos = event.position()
        except AttributeError:  # pragma: no cover - older Qt
            self._cursor_pos = QPointF(event.pos())
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._cursor_pos = None
        self.update()
        super().leaveEvent(event)

    @property
    def panel_kind(self) -> str:
        return str(_field(self._panel, "kind", ""))

    @property
    def curve_mode(self) -> str:
        return self._curve_mode

    @property
    def show_observed_curve(self) -> bool:
        return self._show_observed_curve

    @property
    def show_model_curves(self) -> bool:
        return self._show_model_curves

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg = _theme_qcolor(self._theme, ("figure.facecolor", "card_bg", "surface", "bg"), "#ffffff")
        fg = _theme_qcolor(self._theme, ("text.color", "fg", "card_fg"), "#111111")
        border = _theme_qcolor(self._theme, ("border", "sep"), "#d8dbe1")
        painter.fillRect(self.rect(), bg)

        kind = self.panel_kind
        plot_rect = self._plot_rect(with_colorbar=kind == "heatmap")
        if plot_rect.width() < 40 or plot_rect.height() < 40:
            return

        if kind == "realspace":
            ok = self._paint_realspace(painter, plot_rect, fg, border)
        elif kind == "heatmap":
            ok = self._paint_heatmap(painter, plot_rect, fg, border)
        elif kind in {"series_metric", "series_curve"}:
            ok = self._paint_series(painter, plot_rect, fg, border)
        elif kind == "motif_counts":
            ok = self._paint_motif_counts(painter, plot_rect, fg, border)
        else:
            ok = self._paint_curve(painter, plot_rect, fg, border)
        if not ok:
            no_visible_curves = (
                kind not in {"realspace", "heatmap", "series_metric", "series_curve", "motif_counts"}
                and not (self._show_observed_curve or self._show_model_curves)
            )
            self._paint_empty(
                painter,
                plot_rect,
                fg,
                border,
                _empty_panel_message(kind, no_visible_curves=no_visible_curves),
            )

    def _plot_rect(self, *, with_colorbar: bool = False) -> QRectF:
        left = 56.0
        right = 58.0 if with_colorbar else 18.0
        top = 48.0  # room for the title plus a one-line legend above the data box
        bottom = 48.0
        return QRectF(
            left,
            top,
            max(1.0, float(self.width()) - left - right),
            max(1.0, float(self.height()) - top - bottom),
        )

    def _paint_curve(self, painter: QPainter, plot_rect: QRectF, fg: QColor, border: QColor) -> bool:
        observed = _float_array_or_none(_field(self._panel, "observed", None))
        if observed is None:
            return False
        x = _float_array_or_none(_panel_x(self._panel, len(observed)))
        if x is None or len(x) != len(observed):
            return False
        band_low = _float_array_or_none(_field(self._panel, "band_low", None))
        band_high = _float_array_or_none(_field(self._panel, "band_high", None))
        central = _float_array_or_none(_field(self._panel, "central", None))
        ref = _finite_float(_field(self._panel, "reference_line", None))
        ref_x = _finite_float(_field(self._panel, "reference_line_x", None))
        comparison_mode = self._curve_mode == "comparison" and self._show_model_curves
        has_observed = bool(self._show_observed_curve)

        x_range = (
            _range_for_arrays(x, np.asarray([ref_x], dtype=float))
            if ref_x is not None
            else _range_for_arrays(x)
        )
        if str(_field(self._panel, "statistic", "")) == "cluster_size_counts":
            # Random/most real patterns put all counts at small cluster sizes, leaving a
            # long empty tail out to N. Zoom x to the populated support so the curve reads.
            x_range = _populated_x_range(x, (observed, band_high, central), x_range)
        y_arrays = [observed] if has_observed else []
        if comparison_mode and band_low is not None and len(band_low) == len(observed):
            y_arrays.append(band_low)
        if comparison_mode and band_high is not None and len(band_high) == len(observed):
            y_arrays.append(band_high)
        if comparison_mode and central is not None and len(central) == len(observed):
            y_arrays.append(central)
        if not y_arrays:
            return False
        if ref is not None:
            y_arrays.append(np.asarray([ref], dtype=float))
        y_range = _range_for_arrays(*y_arrays)
        transform = _PlotTransform(plot_rect, x_range, y_range)

        painter.save()
        painter.setClipRect(plot_rect.adjusted(1, 1, -1, -1))
        has_band = (
            comparison_mode
            and band_low is not None
            and band_high is not None
            and len(band_low) == len(x)
        )
        has_central = comparison_mode and central is not None and len(central) == len(x)
        if has_band:
            _draw_band(
                painter,
                transform,
                x,
                band_low,
                band_high,
                _CURVE_BAND_COLOR,
                alpha=115,
            )
        if has_central:
            _draw_polyline(painter, transform, x, central, _CURVE_MODEL_COLOR, width=1.7)
        _draw_reference_line(painter, transform, ref, plot_rect, fg)
        _draw_reference_line_x(painter, transform, ref_x, plot_rect, fg)
        if has_observed:
            _draw_polyline(painter, transform, x, observed, _CURVE_OBSERVED_COLOR, width=2.5)
        legend = _curve_legend_entries(
            has_band=has_band,
            has_central=has_central,
            has_observed=has_observed,
        )
        painter.restore()

        self._draw_chrome(painter, plot_rect, x_range, y_range, fg, border, y_anchor=ref)
        _draw_legend(painter, plot_rect, legend, fg, self._theme)
        if has_observed:
            self._draw_cursor_readout(painter, plot_rect, transform, x, observed, fg, border)
        return True

    def _draw_cursor_readout(
        self,
        painter: QPainter,
        plot_rect: QRectF,
        transform: "_PlotTransform",
        x: np.ndarray,
        observed: np.ndarray,
        fg: QColor,
        border: QColor,
    ) -> None:
        cursor = self._cursor_pos
        if cursor is None or not plot_rect.contains(cursor):
            return
        if x is None or len(x) == 0:
            return
        # Map the cursor x back to data space and snap to the nearest sample.
        x_min, x_max = transform.x_min, transform.x_max
        denom = max(x_max - x_min, 1e-12)
        x_data = x_min + ((cursor.x() - plot_rect.left()) / max(plot_rect.width(), 1.0)) * denom
        idx = int(np.argmin(np.abs(np.asarray(x, dtype=float) - x_data)))
        x_val = float(x[idx])
        y_val = float(observed[idx])
        point = transform.point(x_val, y_val)
        if point is None:
            return
        painter.save()
        guide = QColor(fg.red(), fg.green(), fg.blue(), 90)
        painter.setPen(QPen(guide, 1.0, Qt.DashLine))
        painter.drawLine(QPointF(point.x(), plot_rect.top()), QPointF(point.x(), plot_rect.bottom()))
        painter.setPen(QPen(QColor(_CURVE_OBSERVED_COLOR), 1.0))
        painter.setBrush(QColor(_CURVE_OBSERVED_COLOR))
        painter.drawEllipse(point, 3.0, 3.0)

        x_label = str(_field(self._panel, "x_label", "x"))
        y_label = str(_field(self._panel, "y_label", "y"))
        text = f"{_axis_symbol(x_label)}={_format_tick(x_val)}   {_axis_symbol(y_label)}={_format_tick(y_val)}"
        font = QFont(painter.font())
        font.setPointSizeF(8.5)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        pad = 5.0
        box_w = metrics.horizontalAdvance(text) + 2 * pad
        box_h = metrics.height() + 2 * pad
        box_x = min(point.x() + 8.0, plot_rect.right() - box_w)
        box_x = max(box_x, plot_rect.left())
        box_y = plot_rect.top() + 4.0
        box = QRectF(box_x, box_y, box_w, box_h)
        bg = _theme_qcolor(self._theme, ("card_bg", "surface", "bg"), "#ffffff")
        bg.setAlpha(230)
        painter.setPen(QPen(QColor(fg.red(), fg.green(), fg.blue(), 90), 1.0))
        painter.setBrush(bg)
        painter.drawRect(box)
        painter.setPen(QPen(fg))
        painter.drawText(box, Qt.AlignCenter, text)
        painter.restore()

    def _paint_realspace(self, painter: QPainter, plot_rect: QRectF, fg: QColor, border: QColor) -> bool:
        points = _xy_array_or_none(_field(self._panel, "observed", None))
        if points is None or points.size == 0:
            return False
        metadata = _field(self._panel, "metadata", {}) or {}
        simulated = None
        feature_xy = None
        arrays = [points]
        if self._data_mode == "sandbox":
            simulated = _particle_xy_or_none(_field(metadata, "simulated", None))
            feature_xy = _xy_array_or_none(_field(metadata, "feature_xy_nm", None))
            if simulated is not None and simulated.size:
                arrays.append(simulated)
            if feature_xy is not None and feature_xy.size:
                arrays.append(feature_xy)

        x_range, y_range = _xy_ranges(arrays)
        x_range, y_range = _equal_aspect_ranges(plot_rect, x_range, y_range)
        transform = _PlotTransform(plot_rect, x_range, y_range, invert_y=True)

        legend: list[tuple[str, str, str, bool]] = []
        painter.save()
        painter.setClipRect(plot_rect.adjusted(1, 1, -1, -1))
        if self._data_mode == "sandbox" and simulated is not None and simulated.size:
            _draw_marker_series(
                painter,
                transform,
                simulated,
                marker="o",
                color="#6b7280",
                edgecolor="#6b7280",
                radius=4.0,
                hollow=True,
            )
            legend.append(("model sample", "#6b7280", "o", True))
        if self._data_mode == "sandbox" and feature_xy is not None and feature_xy.size:
            _draw_marker_series(
                painter,
                transform,
                feature_xy,
                marker="x",
                color="#7b2cbf",
                edgecolor="#7b2cbf",
                radius=5.0,
            )
            legend.append(("synthetic feature layer", "#7b2cbf", "x", False))

        style = _realspace_marker_style(self._data_mode)
        _draw_marker_series(
            painter,
            transform,
            points,
            marker=str(style["marker"]),
            color=str(style["color"]),
            edgecolor=str(style["edgecolor"]),
            radius=max(3.5, float(style["size"]) ** 0.5 * 0.72),
        )
        legend.append((str(style["label"]), str(style["color"]), str(style["marker"]), False))
        painter.restore()

        self._draw_chrome(painter, plot_rect, x_range, y_range, fg, border, invert_y=True)
        _draw_legend(painter, plot_rect, legend, fg, self._theme)
        return True

    def _paint_heatmap(self, painter: QPainter, plot_rect: QRectF, fg: QColor, border: QColor) -> bool:
        observed = _float_array_or_none(_field(self._panel, "observed", None))
        if observed is None or observed.ndim != 2:
            return False
        heatmap = _heatmap_image(observed)
        if heatmap is None:
            return False
        painter.drawImage(plot_rect, heatmap)
        x_range = (0.0, float(observed.shape[1]))
        y_range = (0.0, float(observed.shape[0]))
        self._draw_chrome(painter, plot_rect, x_range, y_range, fg, border)
        self._draw_colorbar(painter, plot_rect, observed, fg, border)
        return True

    def _paint_series(self, painter: QPainter, plot_rect: QRectF, fg: QColor, border: QColor) -> bool:
        curves = _series_curves(self._panel)
        reference_curves = _series_reference_curves(self._panel)
        if not curves:
            return False
        x_range = _range_for_arrays(
            *(curve["x"] for curve in curves),
            *(curve["x"] for curve in reference_curves),
        )
        y_values: list[np.ndarray] = []
        for curve in curves:
            y_values.append(curve["mean"])
            if curve["low"] is not None:
                y_values.append(curve["low"])
            if curve["high"] is not None:
                y_values.append(curve["high"])
        for curve in reference_curves:
            y_values.append(curve["y"])
        y_range = _range_for_arrays(*y_values)
        transform = _PlotTransform(plot_rect, x_range, y_range)
        palette = ("#2f7ed8", "#7b2cbf", "#178a5a", "#c75d00", "#5e7fb7")
        legend: list[tuple[str, str, str, bool]] = []

        painter.save()
        painter.setClipRect(plot_rect.adjusted(1, 1, -1, -1))
        has_band = False
        for index, curve in enumerate(curves):
            color = palette[index % len(palette)]
            if curve["low"] is not None and curve["high"] is not None:
                _draw_band(painter, transform, curve["x"], curve["low"], curve["high"], color, alpha=45)
                has_band = True
            _draw_polyline(painter, transform, curve["x"], curve["mean"], color, width=1.5)
            _draw_line_markers(painter, transform, curve["x"], curve["mean"], color)
            legend.append((_series_curve_label(curve["label"]), color, "line", False))
        # A single pooled group (one curve) reads better with the band named.
        if has_band and len(curves) == 1:
            legend.append(("image-to-image spread", palette[0], "bar", False))
        for curve in reference_curves:
            color = str(curve["color"])
            _draw_polyline(painter, transform, curve["x"], curve["y"], color, width=1.8)
            legend.append((str(curve["label"]), color, "line", False))
        painter.restore()

        self._draw_chrome(painter, plot_rect, x_range, y_range, fg, border)
        _draw_legend(painter, plot_rect, legend, fg, self._theme)
        return True

    def _paint_motif_counts(self, painter: QPainter, plot_rect: QRectF, fg: QColor, border: QColor) -> bool:
        observed = _float_array_or_none(_field(self._panel, "observed", None))
        if observed is None:
            return False
        x = np.arange(len(observed), dtype=float)
        low = _float_array_or_none(_field(self._panel, "band_low", None))
        high = _float_array_or_none(_field(self._panel, "band_high", None))
        y_values = [observed, np.asarray([0.0], dtype=float)]
        if low is not None and len(low) == len(observed):
            y_values.append(low)
        if high is not None and len(high) == len(observed):
            y_values.append(high)
        x_range = (-0.5, float(len(observed)) - 0.5)
        y_range = _range_for_arrays(*y_values)
        transform = _PlotTransform(plot_rect, x_range, y_range)

        painter.save()
        painter.setClipRect(plot_rect.adjusted(1, 1, -1, -1))
        _draw_bars(painter, transform, x, observed, "#2f7ed8")
        if low is not None and high is not None and len(low) == len(observed):
            _draw_error_bars(painter, transform, x, low, high, "#333333")
        painter.restore()

        labels = _motif_labels(self._panel, len(observed))
        self._draw_chrome(painter, plot_rect, x_range, y_range, fg, border, x_tick_labels=labels)
        _draw_legend(painter, plot_rect, [("observed", "#2f7ed8", "bar", False)], fg, self._theme)
        return True

    def _paint_empty(
        self,
        painter: QPainter,
        plot_rect: QRectF,
        fg: QColor,
        border: QColor,
        message: str,
    ) -> None:
        self._draw_chrome(painter, plot_rect, (0.0, 1.0), (0.0, 1.0), fg, border)
        font = QFont(painter.font())
        font.setPointSizeF(10.0)
        painter.setFont(font)
        painter.setPen(QPen(fg))
        painter.drawText(plot_rect, Qt.AlignCenter, message)

    def _draw_chrome(
        self,
        painter: QPainter,
        plot_rect: QRectF,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        fg: QColor,
        border: QColor,
        *,
        invert_y: bool = False,
        x_tick_labels: tuple[str, ...] | None = None,
        y_anchor: float | None = None,
    ) -> None:
        title = _plot_title(self._panel)
        x_label = str(_field(self._panel, "x_label", ""))
        y_label = str(_field(self._panel, "y_label", ""))
        grid = QColor(border)
        grid.setAlpha(95)

        x_ticks = _ticks(*x_range)
        y_ticks = _ticks(*y_range, anchor=y_anchor)

        painter.setPen(QPen(border, 1.0))
        painter.drawRect(plot_rect)
        painter.setPen(QPen(grid, 1.0))
        for tick in x_ticks:
            x = _axis_x(plot_rect, tick, x_range)
            painter.drawLine(QPointF(x, plot_rect.top()), QPointF(x, plot_rect.bottom()))
        for tick in y_ticks:
            y = _axis_y(plot_rect, tick, y_range, invert_y=invert_y)
            painter.drawLine(QPointF(plot_rect.left(), y), QPointF(plot_rect.right(), y))

        painter.setPen(QPen(fg, 1.0))
        painter.drawRect(plot_rect)
        tick_font = QFont(painter.font())
        tick_font.setPointSizeF(8.5)
        painter.setFont(tick_font)
        for tick in x_ticks:
            x = _axis_x(plot_rect, tick, x_range)
            painter.drawLine(QPointF(x, plot_rect.bottom()), QPointF(x, plot_rect.bottom() + 4.0))
        for tick in y_ticks:
            y = _axis_y(plot_rect, tick, y_range, invert_y=invert_y)
            painter.drawLine(QPointF(plot_rect.left() - 4.0, y), QPointF(plot_rect.left(), y))
            painter.drawText(QRectF(2.0, y - 8.0, plot_rect.left() - 8.0, 16.0), Qt.AlignRight | Qt.AlignVCenter, _format_tick(tick))

        if x_tick_labels:
            for index, label in enumerate(x_tick_labels):
                if index >= 10:
                    break
                x = _axis_x(plot_rect, float(index), x_range)
                painter.drawText(QRectF(x - 30.0, plot_rect.bottom() + 6.0, 60.0, 16.0), Qt.AlignCenter, label)
        else:
            for tick in x_ticks:
                x = _axis_x(plot_rect, tick, x_range)
                painter.drawText(QRectF(x - 24.0, plot_rect.bottom() + 6.0, 48.0, 16.0), Qt.AlignCenter, _format_tick(tick))

        title_font = QFont(painter.font())
        title_font.setPointSizeF(11.0)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(QRectF(0.0, 4.0, float(self.width()), 20.0), Qt.AlignCenter, title)

        label_font = QFont(painter.font())
        label_font.setPointSizeF(10.0)
        label_font.setBold(False)
        painter.setFont(label_font)
        painter.drawText(
            QRectF(plot_rect.left(), float(self.height()) - 25.0, plot_rect.width(), 18.0),
            Qt.AlignCenter,
            x_label,
        )
        if y_label:
            painter.save()
            painter.translate(13.0, plot_rect.center().y())
            painter.rotate(-90.0)
            painter.drawText(
                QRectF(-plot_rect.height() / 2.0, -9.0, plot_rect.height(), 18.0),
                Qt.AlignCenter,
                y_label,
            )
            painter.restore()

    def _draw_colorbar(self, painter: QPainter, plot_rect: QRectF, values: np.ndarray, fg: QColor, border: QColor) -> None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        bar = QRectF(plot_rect.right() + 14.0, plot_rect.top(), 12.0, plot_rect.height())
        steps = max(1, int(bar.height()))
        for row in range(steps):
            t = 1.0 - (row / max(1, steps - 1))
            painter.setPen(QPen(_heatmap_color(t)))
            y = bar.top() + row
            painter.drawLine(QPointF(bar.left(), y), QPointF(bar.right(), y))
        painter.setPen(QPen(border, 1.0))
        painter.drawRect(bar)
        painter.setPen(QPen(fg))
        font = QFont(painter.font())
        font.setPointSizeF(8.5)
        painter.setFont(font)
        painter.drawText(QRectF(bar.right() + 4.0, bar.top() - 2.0, 34.0, 14.0), Qt.AlignLeft | Qt.AlignVCenter, _format_tick(float(np.max(finite))))
        painter.drawText(QRectF(bar.right() + 4.0, bar.bottom() - 12.0, 34.0, 14.0), Qt.AlignLeft | Qt.AlignVCenter, _format_tick(float(np.min(finite))))


class _PlotTransform:
    def __init__(
        self,
        rect: QRectF,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        *,
        invert_y: bool = False,
    ):
        self.rect = rect
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.invert_y = invert_y

    def point(self, x_value: float, y_value: float) -> QPointF | None:
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            return None
        x = _axis_x(self.rect, float(x_value), (self.x_min, self.x_max))
        y = _axis_y(self.rect, float(y_value), (self.y_min, self.y_max), invert_y=self.invert_y)
        return QPointF(x, y)


def _theme_qcolor(theme: dict, keys: tuple[str, ...], fallback: str) -> QColor:
    for key in keys:
        value = theme.get(key)
        if value:
            color = QColor(str(value))
            if color.isValid():
                return color
    return QColor(fallback)


def _float_array_or_none(value: Any) -> np.ndarray | None:
    arr = _array_or_none(value)
    if arr is None:
        return None
    try:
        return np.asarray(arr, dtype=float)
    except (TypeError, ValueError):
        return None


def _range_for_arrays(*arrays: np.ndarray) -> tuple[float, float]:
    finite: list[np.ndarray] = []
    for array in arrays:
        arr = np.asarray(array, dtype=float)
        values = arr[np.isfinite(arr)]
        if values.size:
            finite.append(values)
    if not finite:
        return (0.0, 1.0)
    values = np.concatenate(finite)
    return _padded_range(float(np.min(values)), float(np.max(values)))


def _populated_x_range(
    x: np.ndarray,
    y_arrays: tuple[np.ndarray | None, ...],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    """Trim an x-range to where the data is actually non-zero.

    Count distributions (e.g. cluster sizes) are non-zero only at small x and then
    flat-zero out to N, which wastes the axis. Return ``[min(x), last_nonzero_x]`` with
    a one-sample margin; fall back to the full range if nothing is populated.
    """
    xs = np.asarray(x, dtype=float)
    if xs.size == 0:
        return fallback
    mask = np.zeros(xs.shape, dtype=bool)
    for arr in y_arrays:
        if arr is None:
            continue
        values = np.asarray(arr, dtype=float)
        if values.shape != xs.shape:
            continue
        mask |= np.isfinite(values) & (np.abs(values) > 1e-9)
    if not mask.any():
        return fallback
    last = int(np.flatnonzero(mask)[-1])
    upper_index = min(last + 1, xs.size - 1)
    x_min = float(np.min(xs))
    x_max = float(xs[upper_index])
    if x_max <= x_min:
        return fallback
    return _padded_range(x_min, x_max)


def _padded_range(vmin: float, vmax: float, fraction: float = 0.06) -> tuple[float, float]:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (0.0, 1.0)
    if vmin == vmax:
        pad = max(abs(vmin) * 0.08, 0.5)
    else:
        pad = abs(vmax - vmin) * fraction
    return (vmin - pad, vmax + pad)


def _xy_ranges(arrays: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for array in arrays:
        arr = _xy_array_or_none(array)
        if arr is None:
            continue
        xs.append(arr[:, 0])
        ys.append(arr[:, 1])
    return _range_for_arrays(*xs), _range_for_arrays(*ys)


def _equal_aspect_ranges(
    rect: QRectF,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    rect_ratio = max(rect.width(), 1.0) / max(rect.height(), 1.0)
    data_ratio = x_span / y_span
    if data_ratio > rect_ratio:
        target_y = x_span / rect_ratio
        centre = (y_min + y_max) / 2.0
        y_min = centre - target_y / 2.0
        y_max = centre + target_y / 2.0
    else:
        target_x = y_span * rect_ratio
        centre = (x_min + x_max) / 2.0
        x_min = centre - target_x / 2.0
        x_max = centre + target_x / 2.0
    return (x_min, x_max), (y_min, y_max)


def _axis_x(rect: QRectF, value: float, x_range: tuple[float, float]) -> float:
    x_min, x_max = x_range
    denom = max(x_max - x_min, 1e-12)
    return rect.left() + ((value - x_min) / denom) * rect.width()


def _axis_y(
    rect: QRectF,
    value: float,
    y_range: tuple[float, float],
    *,
    invert_y: bool = False,
) -> float:
    y_min, y_max = y_range
    denom = max(y_max - y_min, 1e-12)
    frac = (value - y_min) / denom
    if invert_y:
        return rect.top() + frac * rect.height()
    return rect.bottom() - frac * rect.height()


def _nice_step(raw: float) -> float:
    """Round a raw step up to the nearest 1 / 2 / 2.5 / 5 x 10**k."""
    if not np.isfinite(raw) or raw <= 0.0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for multiple in (1.0, 2.0, 2.5, 5.0):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10.0 * magnitude


def _ticks(
    vmin: float,
    vmax: float,
    *,
    anchor: float | None = None,
    target: int = 5,
) -> tuple[float, ...]:
    """Evenly spaced "nice" axis ticks.

    Returns ticks at regular 1/2/2.5/5 x 10**k intervals spanning ``[vmin, vmax]``
    so labels read cleanly. When ``anchor`` is given (e.g. the g=1 reference line),
    the tick lattice is shifted to land exactly on it, so the reference is a gridline.
    """
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (0.0, 0.5, 1.0)
    if vmin == vmax:
        return (vmin - 0.5, vmin, vmin + 0.5)
    step = _nice_step((vmax - vmin) / max(1, target))
    if anchor is not None and np.isfinite(anchor):
        first = anchor + math.ceil((vmin - anchor) / step - 1e-9) * step
    else:
        first = math.ceil(vmin / step - 1e-9) * step
    ticks: list[float] = []
    tick = first
    while tick <= vmax + step * 1e-6 and len(ticks) < 40:
        # Snap away tiny floating-point noise around zero.
        ticks.append(0.0 if abs(tick) < step * 1e-9 else tick)
        tick += step
    if not ticks:
        return (vmin, (vmin + vmax) / 2.0, vmax)
    return tuple(ticks)


def _axis_symbol(label: str) -> str:
    """Compact axis symbol for the cursor read-out (e.g. 'distance r (nm)' -> 'r')."""
    s = (label or "").strip()
    if not s:
        return ""
    if s.endswith(")") and "(" in s:
        head = s[: s.rfind("(")].strip()
        unit = s[s.rfind("(") :].lower()
        if head and any(token in unit for token in ("nm", "µm", "um", "deg", "%", "px")):
            s = head
    parts = s.split()
    return parts[-1] if parts else s


def _format_tick(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 1000.0 or (0.0 < abs(value) < 0.01):
        return f"{value:.2g}"
    return f"{value:.3g}"


def _draw_polyline(
    painter: QPainter,
    transform: _PlotTransform,
    x_values: np.ndarray,
    y_values: np.ndarray,
    color: str,
    *,
    width: float = 1.4,
) -> None:
    points = _mapped_points(transform, x_values, y_values)
    if len(points) < 2:
        return
    painter.setPen(QPen(QColor(color), width))
    painter.setBrush(Qt.NoBrush)
    painter.drawPolyline(QPolygonF(points))


def _mapped_points(transform: _PlotTransform, x_values: np.ndarray, y_values: np.ndarray) -> list[QPointF]:
    points: list[QPointF] = []
    for x_value, y_value in zip(x_values, y_values, strict=False):
        point = transform.point(float(x_value), float(y_value))
        if point is not None:
            points.append(point)
    return points


def _draw_band(
    painter: QPainter,
    transform: _PlotTransform,
    x_values: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    color: str,
    *,
    alpha: int = 95,
) -> None:
    valid = np.isfinite(x_values) & np.isfinite(low) & np.isfinite(high)
    if int(np.count_nonzero(valid)) < 2:
        return
    x = x_values[valid]
    lo = low[valid]
    hi = high[valid]
    upper = _mapped_points(transform, x, hi)
    lower = _mapped_points(transform, x[::-1], lo[::-1])
    if len(upper) < 2 or len(lower) < 2:
        return
    path = QPainterPath(upper[0])
    for point in upper[1:]:
        path.lineTo(point)
    for point in lower:
        path.lineTo(point)
    path.closeSubpath()
    fill = QColor(color)
    fill.setAlpha(alpha)
    painter.setPen(Qt.NoPen)
    painter.setBrush(fill)
    painter.drawPath(path)
    painter.setBrush(Qt.NoBrush)


def _draw_reference_line(
    painter: QPainter,
    transform: _PlotTransform,
    value: float | None,
    plot_rect: QRectF,
    color: QColor,
) -> None:
    if value is None:
        return
    y = transform.point(transform.x_min, value)
    if y is None:
        return
    pen = QPen(color, 1.0)
    pen.setStyle(Qt.DashLine)
    pen.setColor(QColor(color.red(), color.green(), color.blue(), 130))
    painter.setPen(pen)
    painter.drawLine(QPointF(plot_rect.left(), y.y()), QPointF(plot_rect.right(), y.y()))


def _draw_reference_line_x(
    painter: QPainter,
    transform: _PlotTransform,
    value: float | None,
    plot_rect: QRectF,
    color: QColor,
) -> None:
    if value is None:
        return
    point = transform.point(value, transform.y_min)
    if point is None:
        return
    pen = QPen(color, 1.0)
    pen.setStyle(Qt.DashLine)
    pen.setColor(QColor(color.red(), color.green(), color.blue(), 130))
    painter.setPen(pen)
    painter.drawLine(QPointF(point.x(), plot_rect.top()), QPointF(point.x(), plot_rect.bottom()))


def _draw_marker_series(
    painter: QPainter,
    transform: _PlotTransform,
    xy: np.ndarray,
    *,
    marker: str,
    color: str,
    edgecolor: str,
    radius: float,
    hollow: bool = False,
) -> None:
    for point in np.asarray(xy, dtype=float):
        mapped = transform.point(float(point[0]), float(point[1]))
        if mapped is not None:
            _draw_marker(
                painter,
                mapped,
                marker=marker,
                color=color,
                edgecolor=edgecolor,
                radius=radius,
                hollow=hollow,
            )


def _draw_marker(
    painter: QPainter,
    point: QPointF,
    *,
    marker: str,
    color: str,
    edgecolor: str,
    radius: float,
    hollow: bool = False,
) -> None:
    fill = QColor(color)
    edge = QColor(edgecolor)
    painter.setPen(QPen(edge, 1.1))
    painter.setBrush(Qt.NoBrush if hollow or marker == "x" else fill)
    if marker == "^":
        poly = QPolygonF(
            [
                QPointF(point.x(), point.y() - radius),
                QPointF(point.x() - radius, point.y() + radius),
                QPointF(point.x() + radius, point.y() + radius),
            ]
        )
        painter.drawPolygon(poly)
    elif marker == "x":
        painter.setPen(QPen(fill, 1.5))
        painter.drawLine(
            QPointF(point.x() - radius, point.y() - radius),
            QPointF(point.x() + radius, point.y() + radius),
        )
        painter.drawLine(
            QPointF(point.x() - radius, point.y() + radius),
            QPointF(point.x() + radius, point.y() - radius),
        )
    else:
        painter.drawEllipse(point, radius, radius)


def _draw_line_markers(
    painter: QPainter,
    transform: _PlotTransform,
    x_values: np.ndarray,
    y_values: np.ndarray,
    color: str,
) -> None:
    for point in _mapped_points(transform, x_values, y_values):
        _draw_marker(
            painter,
            point,
            marker="o",
            color=color,
            edgecolor=color,
            radius=2.4,
        )


def _draw_bars(
    painter: QPainter,
    transform: _PlotTransform,
    x_values: np.ndarray,
    values: np.ndarray,
    color: str,
) -> None:
    painter.setPen(QPen(QColor(color).darker(115), 1.0))
    painter.setBrush(QColor(color))
    for x_value, value in zip(x_values, values, strict=False):
        left = transform.point(float(x_value) - 0.36, 0.0)
        right = transform.point(float(x_value) + 0.36, float(value))
        baseline = transform.point(float(x_value), 0.0)
        if left is None or right is None or baseline is None:
            continue
        rect = QRectF(left.x(), right.y(), right.x() - left.x(), baseline.y() - right.y()).normalized()
        painter.drawRect(rect)
    painter.setBrush(Qt.NoBrush)


def _draw_error_bars(
    painter: QPainter,
    transform: _PlotTransform,
    x_values: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    color: str,
) -> None:
    painter.setPen(QPen(QColor(color), 1.0))
    for x_value, lo, hi in zip(x_values, low, high, strict=False):
        low_pt = transform.point(float(x_value), float(lo))
        high_pt = transform.point(float(x_value), float(hi))
        if low_pt is None or high_pt is None:
            continue
        painter.drawLine(low_pt, high_pt)
        painter.drawLine(QPointF(low_pt.x() - 3.0, low_pt.y()), QPointF(low_pt.x() + 3.0, low_pt.y()))
        painter.drawLine(QPointF(high_pt.x() - 3.0, high_pt.y()), QPointF(high_pt.x() + 3.0, high_pt.y()))


def _draw_legend(
    painter: QPainter,
    plot_rect: QRectF,
    entries: list[tuple[str, str, str, bool]],
    fg: QColor,
    theme: dict,
) -> None:
    if not entries:
        return
    font = QFont(painter.font())
    font.setPointSizeF(8.5)
    painter.setFont(font)
    metrics = painter.fontMetrics()
    # Lay the legend out as a single horizontal strip in the top margin, above the
    # data box, so it never overlaps the plotted curves.
    gap = 16.0
    sample_w = 18.0
    items = [(label, color, marker, hollow, metrics.horizontalAdvance(label)) for label, color, marker, hollow in entries]
    total = sum(sample_w + tw for *_unused, tw in items) + gap * (len(items) - 1)
    y = plot_rect.top() - 12.0
    x = max(plot_rect.left(), plot_rect.right() - total)
    for label, color, marker, hollow, text_w in items:
        sample = QPointF(x + 7.0, y)
        if marker == "line":
            painter.setPen(QPen(QColor(color), 1.5))
            painter.drawLine(QPointF(sample.x() - 6.0, sample.y()), QPointF(sample.x() + 6.0, sample.y()))
        elif marker == "bar":
            painter.setPen(QPen(QColor(color).darker(115), 1.0))
            painter.setBrush(QColor(color))
            painter.drawRect(QRectF(sample.x() - 5.0, sample.y() - 5.0, 10.0, 10.0))
            painter.setBrush(Qt.NoBrush)
        else:
            _draw_marker(
                painter,
                sample,
                marker=marker,
                color=color,
                edgecolor=color,
                radius=4.0,
                hollow=hollow,
            )
        painter.setPen(QPen(fg))
        painter.drawText(QRectF(x + sample_w, y - 8.0, text_w + 4.0, 16.0), Qt.AlignLeft | Qt.AlignVCenter, label)
        x += sample_w + text_w + gap


def _series_curve_label(raw: Any) -> str:
    """Human label for a pooled series curve.

    A single pooled group carries a coverage label like "0 " (value 0, unitless);
    show "pooled mean" instead so the legend is meaningful.
    """
    text = str(raw or "").strip()
    if text in ("", "0") or text.rstrip("0").rstrip(".") in ("", "0"):
        return "pooled mean"
    return text


def _series_curves(panel: Any) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    for curve in tuple(_field(panel, "series_curves", ()) or ()):
        x = _float_array_or_none(_field(curve, "x", None))
        mean = _float_array_or_none(_field(curve, "mean", None))
        if x is None or mean is None or len(x) != len(mean):
            continue
        low = _float_array_or_none(_field(curve, "band_low", None))
        high = _float_array_or_none(_field(curve, "band_high", None))
        if low is not None and len(low) != len(x):
            low = None
        if high is not None and len(high) != len(x):
            high = None
        curves.append(
            {
                "x": x,
                "mean": mean,
                "low": low,
                "high": high,
                "label": str(_field(curve, "label", "series")),
            }
        )
    return curves


def _series_reference_curves(panel: Any) -> list[dict[str, Any]]:
    metadata = _field(panel, "metadata", {}) or {}
    raw_curves = tuple(_field(metadata, "reference_curves", ()) or ())
    curves: list[dict[str, Any]] = []
    for curve in raw_curves:
        x = _float_array_or_none(_field(curve, "x", None))
        y = _float_array_or_none(_field(curve, "y", None))
        if x is None or y is None or len(x) != len(y):
            continue
        curves.append(
            {
                "x": x,
                "y": y,
                "label": str(_field(curve, "label", "single image reference")),
                "color": str(_field(curve, "color", _CURVE_OBSERVED_COLOR)),
            }
        )
    return curves


def _motif_labels(panel: Any, count: int) -> tuple[str, ...] | None:
    motifs = _field(_field(panel, "coordinate_values", {}) or {}, "motif", None)
    if motifs is None:
        return None
    labels = tuple(str(item) for item in np.asarray(motifs, dtype=object).tolist())
    return labels[:count]


def _heatmap_image(values: np.ndarray) -> QImage | None:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    denom = max(vmax - vmin, 1e-12)
    rows, cols = arr.shape
    image = QImage(cols, rows, QImage.Format_RGB32)
    for row in range(rows):
        source_row = rows - row - 1
        for col in range(cols):
            value = arr[source_row, col]
            t = 0.0 if not np.isfinite(value) else (float(value) - vmin) / denom
            image.setPixelColor(col, row, _heatmap_color(float(np.clip(t, 0.0, 1.0))))
    return image


def _heatmap_color(t: float) -> QColor:
    stops = (
        (0.0, QColor("#30295f")),
        (0.35, QColor("#20639b")),
        (0.70, QColor("#29a36a")),
        (1.0, QColor("#f7d13d")),
    )
    t = float(np.clip(t, 0.0, 1.0))
    for (left_t, left), (right_t, right) in zip(stops, stops[1:], strict=False):
        if left_t <= t <= right_t:
            frac = (t - left_t) / max(right_t - left_t, 1e-12)
            return QColor(
                round(left.red() + (right.red() - left.red()) * frac),
                round(left.green() + (right.green() - left.green()) * frac),
                round(left.blue() + (right.blue() - left.blue()) * frac),
            )
    return QColor(stops[-1][1])


def _empty_panel_message(kind: str, *, no_visible_curves: bool = False) -> str:
    if no_visible_curves:
        return "No selected plot layers visible"
    if kind == "realspace":
        return "No points"
    if kind == "heatmap":
        return "No heatmap data"
    if kind in {"series_metric", "series_curve"}:
        return "No series data"
    if kind == "motif_counts":
        return "No motif data"
    return "No curve data"


def _panel_x(panel: Any, length: int) -> np.ndarray:
    x = _array_or_none(_field(panel, "x", None))
    if x is not None and len(x) == length:
        return x
    for value in (_field(panel, "coordinate_values", {}) or {}).values():
        arr = _array_or_none(value)
        if arr is not None and len(arr) == length and np.issubdtype(arr.dtype, np.number):
            return arr
    return np.arange(length, dtype=float)


def _caption_label(panel: Any) -> QLabel:
    lines = tuple(_field(panel, "caption_lines", ()) or ())
    if not lines:
        verdict = _field(panel, "verdict_label", "")
        p = _field(panel, "global_p", None)
        lines = tuple(item for item in (verdict, f"global p: {p}" if p is not None else "") if item)
    label = QLabel("\n".join(str(line) for line in lines) or "No caption.")
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def _model_summary_widget(rows: tuple[tuple[Any, ...], ...]) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        parsed = _parse_verdict_row(row)
        if parsed is None:
            continue
        grouped.setdefault(parsed["model"], []).append(parsed)
    if not grouped:
        layout.addWidget(QLabel("No model verdicts available."))
        return container
    green = "#2fb344"
    red = "#e0564b"
    for model_id, entries in grouped.items():
        # A model is ruled out if any one statistic rejects it; plausible only when
        # every statistic stays consistent. Colour the card so it reads at a glance.
        ruled_out = any(_verdict_is_inconsistent(entry["verdict"]) for entry in entries)
        bar = red if ruled_out else green
        card = QFrame(container)
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet(
            f"QFrame {{ border: 1px solid #384250; border-left: 5px solid {bar}; }}"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 8, 8)
        card_layout.setSpacing(5)
        badge = "ruled out" if ruled_out else "plausible"
        title = QLabel(
            f"<b>{_model_label(model_id)}</b> "
            f"<span style='color:{bar}; font-weight:700;'>— {badge}</span>"
        )
        title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        card_layout.addWidget(title)
        desc = QLabel(_model_description(model_id))
        desc.setWordWrap(True)
        desc.setTextInteractionFlags(Qt.TextSelectableByMouse)
        card_layout.addWidget(desc)
        for entry in entries:
            inconsistent = _verdict_is_inconsistent(entry["verdict"])
            colour = red if inconsistent else green
            line = (
                f"<b>{_statistic_label(entry['statistic'])}</b>: "
                f"<span style='color:{colour};'>{_verdict_label(entry['verdict'])}</span>"
            )
            details = []
            if entry.get("p"):
                details.append(f"p {entry['p']}")
            if entry.get("outside"):
                details.append(f"outside {entry['outside']}")
            if details:
                line += " (" + ", ".join(details) + ")"
            stat = QLabel(line)
            stat.setWordWrap(True)
            stat.setTextInteractionFlags(Qt.TextSelectableByMouse)
            card_layout.addWidget(stat)
        layout.addWidget(card)
    return container


def _verdict_is_inconsistent(verdict_id: str) -> bool:
    return "inconsistent" in str(verdict_id).lower()


def _parse_verdict_row(row: tuple[Any, ...]) -> dict[str, str] | None:
    values = tuple(str(item) for item in row)
    if len(values) >= 7:
        return {
            "model": values[0],
            "statistic": values[1],
            "verdict": values[2],
            "p": values[3],
            "outside": values[5],
        }
    if len(values) >= 5:
        return {
            "model": values[1],
            "statistic": values[2],
            "verdict": values[3],
            "p": values[4],
            "outside": "",
        }
    return None


def _model_label(model_id: str) -> str:
    labels = {
        "homogeneous_poisson": "Random placement",
        "poisson": "Random placement",
        "hard_core_random": "No-overlap random placement",
        "measured_feature_poisson": "Feature-biased placement",
    }
    return labels.get(str(model_id), str(model_id).replace("_", " ").title())


def _model_description(model_id: str) -> str:
    descriptions = {
        "homogeneous_poisson": "Assumes points are placed independently with one average density across the allowed region.",
        "poisson": "Assumes points are placed independently with one average density across the allowed region.",
        "hard_core_random": "Assumes random placement but prevents points from being closer than a minimum distance.",
        "measured_feature_poisson": "Assumes point density can be biased by independently measured feature locations.",
    }
    return descriptions.get(str(model_id), "A spatial null model used as a comparison baseline.")


def _statistic_label(statistic_id: str) -> str:
    labels = {
        "pair_correlation_g_r": "Pair correlation",
        "pair_correlation_g_r_theta": "Pair distance-angle map",
        "bond_order_psi6": "ψ6 triangular order",
        "bond_order_psi4": "ψ4 square order",
        "nearest_neighbor_distribution": "Nearest neighbors",
        "ripley_l_function": "Ripley L",
        "cluster_size_counts": "Cluster sizes",
    }
    return labels.get(str(statistic_id), str(statistic_id).replace("_", " ").title())


def _plot_title(panel: Any) -> str:
    statistic = str(_field(panel, "statistic", "") or "")
    labels = {
        "pair_correlation_g_r": "Pair correlation g(r)",
        "pair_correlation_g_r_theta": "Pair distance-angle map",
        "bond_order_psi6": "ψ6 local order - triangular-like neighborhoods",
        "bond_order_psi4": "ψ4 local order - square-like neighborhoods",
        "nearest_neighbor_distribution": "Nearest-neighbor distances",
        "ripley_l_function": "Ripley L",
        "cluster_size_counts": "Cluster sizes",
    }
    if statistic in labels:
        return labels[statistic]
    return str(_field(panel, "title", "") or statistic)


def _curve_legend_entries(
    *,
    has_band: bool,
    has_central: bool,
    has_observed: bool = True,
) -> list[tuple[str, str, str, bool]]:
    entries: list[tuple[str, str, str, bool]] = []
    if has_band:
        entries.append(("model envelope", _CURVE_BAND_COLOR, "line", False))
    if has_central:
        entries.append(("model median", _CURVE_MODEL_COLOR, "line", False))
    if has_observed:
        entries.append(("observed data", _CURVE_OBSERVED_COLOR, "line", False))
    return entries


def _verdict_label(verdict_id: str) -> str:
    labels = {
        "consistent_with_null": "consistent with this model",
        "inconsistent_with_null": "not consistent with this model",
        "underpowered": "not enough information to decide",
    }
    return labels.get(str(verdict_id), str(verdict_id).replace("_", " "))


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-weight: 600;")
    return label


def _table_widget(columns: tuple[Any, ...], rows: tuple[tuple[Any, ...], ...]) -> QTableWidget:
    max_cols = max([len(tuple(row)) for row in rows] + [len(columns), 1])
    headers = [str(col) for col in columns]
    while len(headers) < max_cols:
        headers.append(f"col {len(headers) + 1}")
    table = QTableWidget(len(rows), max_cols)
    table.setHorizontalHeaderLabels(headers[:max_cols])
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectRows)
    table.setAlternatingRowColors(True)
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(tuple(row)[:max_cols]):
            table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
    table.resizeColumnsToContents()
    table.resizeRowsToContents()
    return table


def _verdict_columns(rows: tuple[tuple[Any, ...], ...]) -> tuple[str, ...]:
    width = max(len(tuple(row)) for row in rows)
    if width == 7:
        return ("model", "statistic", "verdict", "ERL p", "score", "outside", "n_sim")
    if width == 5:
        return ("coverage", "model", "statistic", "verdict", "global p")
    return tuple(f"col {index + 1}" for index in range(width))


def _explainer_text(explainer: Any) -> str:
    pieces = [
        _field(explainer, "friendly_name", ""),
        _field(explainer, "plain_summary", ""),
        _field(explainer, "useful_for", ""),
    ]
    cautions = tuple(_field(explainer, "cautions", ()) or ())
    if cautions:
        pieces.append("Cautions: " + " ".join(str(item) for item in cautions))
    return "\n\n".join(str(piece) for piece in pieces if piece)


def _scrollable(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(widget)
    return scroll


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    return arr if arr.size else None


def _xy_array_or_none(value: Any) -> np.ndarray | None:
    arr = _array_or_none(value)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _particle_xy_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    xy_nm = _field(value, "xy_nm", None)
    if xy_nm is not None:
        return _xy_array_or_none(xy_nm)
    return _xy_array_or_none(value)


def _realspace_marker_style(data_mode: str) -> dict[str, Any]:
    if _normalise_data_mode(data_mode) == "sandbox":
        return {
            "marker": "^",
            "color": "#f28e2b",
            "edgecolor": "#3a2200",
            "size": 38,
            "label": "generated observed",
        }
    return {
        "marker": "o",
        "color": "#2f7ed8",
        "edgecolor": "white",
        "size": 28,
        "label": "observed",
    }


def _normalise_data_mode(data_mode: str) -> str:
    mode = str(data_mode or "real").strip().lower()
    if mode not in {"real", "sandbox"}:
        raise ValueError("data_mode must be 'real' or 'sandbox'")
    return mode


def _normalise_curve_mode(curve_mode: str) -> str:
    mode = str(curve_mode or "comparison").strip().lower()
    if mode not in {"comparison", "observed_only"}:
        raise ValueError("curve_mode must be 'comparison' or 'observed_only'")
    return mode


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _short_tab_title(title: str) -> str:
    text = str(title or "Panel").strip()
    return text if len(text) <= 28 else text[:25].rstrip() + "..."
