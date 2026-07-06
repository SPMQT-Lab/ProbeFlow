"""Qt-native point-field renderer for the Particle Statistics tool.

`ParticleFieldView` paints observed/simulated/feature point layers of a
`ParticleFieldModel` with region mask, legend, and density-scaled markers.
Extracted verbatim from ``particle_statistics.py``; the dialog re-exports
these names for backward compatibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


@dataclass(frozen=True)
class ParticleFieldModel:
    """Display-only point field for the Particle Statistics window."""

    observed_xy_nm: np.ndarray
    width_nm: float
    height_nm: float
    mode: str = "real"
    source_label: str = ""
    region_label: str = "Full field"
    model_label: str = ""
    status: str = ""
    mask: np.ndarray | None = None
    simulated_xy_nm: np.ndarray | None = None
    feature_xy_nm: np.ndarray | None = None
    show_observed: bool = True
    show_simulated: bool = True
    show_features: bool = True
    show_region: bool = True
    direct_labels: tuple[str, ...] = ()


class ParticleFieldView(QWidget):
    """Qt-native field renderer for real and generated particle patterns."""

    def __init__(self, *, theme: dict | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("particleStatisticsField")
        self.setMinimumSize(440, 300)
        self._theme = theme or {}
        self._model = ParticleFieldModel(
            observed_xy_nm=np.empty((0, 2), dtype=float),
            width_nm=100.0,
            height_nm=100.0,
            status="No points to display.",
        )

    @property
    def point_count(self) -> int:
        return int(len(self._model.observed_xy_nm))

    @property
    def data_mode(self) -> str:
        return self._model.mode

    @property
    def marker_style(self) -> dict[str, str]:
        return _marker_style(self._model.mode)

    @property
    def layer_visibility(self) -> dict[str, bool]:
        model = self._model
        return {
            "observed": bool(model.show_observed),
            "simulated": bool(model.show_simulated),
            "features": bool(model.show_features),
            "region": bool(model.show_region),
        }

    @property
    def layer_availability(self) -> dict[str, bool]:
        model = self._model
        return {
            "observed": bool(len(model.observed_xy_nm)),
            "simulated": bool(model.simulated_xy_nm is not None and len(model.simulated_xy_nm)),
            "features": bool(model.feature_xy_nm is not None and len(model.feature_xy_nm)),
            "region": bool(model.mask is not None and model.mask.size),
        }

    @property
    def direct_labels(self) -> tuple[str, ...]:
        return tuple(self._model.direct_labels)

    def set_field_model(self, model: ParticleFieldModel) -> None:
        self._model = model
        self.update()

    def set_layer_visibility(
        self,
        *,
        observed: bool | None = None,
        simulated: bool | None = None,
        features: bool | None = None,
        region: bool | None = None,
    ) -> None:
        model = self._model
        self.set_field_model(
            replace(
                model,
                show_observed=model.show_observed if observed is None else bool(observed),
                show_simulated=model.show_simulated if simulated is None else bool(simulated),
                show_features=model.show_features if features is None else bool(features),
                show_region=model.show_region if region is None else bool(region),
            )
        )

    def set_direct_labels(self, labels: tuple[str, ...] | list[str]) -> None:
        model = self._model
        self.set_field_model(
            replace(model, direct_labels=tuple(str(label) for label in labels if label))
        )

    def set_points(
        self,
        observed_xy_nm: Any,
        *,
        field_size_nm: tuple[float, float],
        mode: str = "real",
        source_label: str = "",
        region_label: str = "Full field",
        model_label: str = "",
        status: str = "",
        mask: Any = None,
        simulated_xy_nm: Any = None,
        feature_xy_nm: Any = None,
        direct_labels: tuple[str, ...] = (),
    ) -> None:
        self.set_field_model(
            ParticleFieldModel(
                observed_xy_nm=_xy_array(observed_xy_nm),
                width_nm=float(field_size_nm[0]),
                height_nm=float(field_size_nm[1]),
                mode=_normalise_field_mode(mode),
                source_label=str(source_label or ""),
                region_label=str(region_label or "Full field"),
                model_label=str(model_label or ""),
                status=str(status or ""),
                mask=_mask_or_none(mask),
                simulated_xy_nm=_xy_array_or_none(simulated_xy_nm),
                feature_xy_nm=_xy_array_or_none(feature_xy_nm),
                direct_labels=tuple(str(label) for label in direct_labels if label),
            )
        )

    def set_region(self, *, region_label: str, mask: Any = None) -> None:
        model = self._model
        self.set_field_model(
            replace(
                model,
                region_label=str(region_label or "Full field"),
                mask=_mask_or_none(mask),
            )
        )

    def set_mode(self, mode: str) -> None:
        model = self._model
        self.set_field_model(
            replace(model, mode=_normalise_field_mode(mode))
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bg = _theme_qcolor(self._theme, ("figure.facecolor", "surface", "bg"), "#161a20")
        fg = _theme_qcolor(self._theme, ("text.color", "fg"), "#e8edf4")
        border = _theme_qcolor(self._theme, ("border", "sep"), "#3b4250")
        painter.fillRect(self.rect(), bg)

        plot_rect = self._plot_rect()
        if plot_rect.width() < 40 or plot_rect.height() < 40:
            return

        model = self._model
        painter.save()
        painter.setPen(QPen(border, 1.0))
        painter.setBrush(QColor("#101419"))
        painter.drawRect(plot_rect)
        if model.show_region:
            self._draw_mask(painter, plot_rect)
        painter.restore()

        transform = _FieldTransform(plot_rect, model.width_nm, model.height_nm)
        plot_w = plot_rect.width()
        painter.save()
        painter.setClipRect(plot_rect.adjusted(1, 1, -1, -1))
        visible_layers = 0
        if model.mode == "generated" and model.show_simulated and model.simulated_xy_nm is not None:
            visible_layers += int(len(model.simulated_xy_nm) > 0)
            _draw_marker_series(
                painter,
                transform,
                model.simulated_xy_nm,
                marker="o",
                color="#b96adf",
                radius=_field_marker_radius(plot_w, len(model.simulated_xy_nm)),
                hollow=True,
            )
        if model.mode == "generated" and model.show_features and model.feature_xy_nm is not None:
            visible_layers += int(len(model.feature_xy_nm) > 0)
            _draw_marker_series(
                painter,
                transform,
                model.feature_xy_nm,
                marker="x",
                color="#9b5de5",
                radius=max(4.0, _field_marker_radius(plot_w, len(model.feature_xy_nm))),
            )
        if model.show_observed:
            visible_layers += int(len(model.observed_xy_nm) > 0)
            style = _marker_style(model.mode)
            _draw_marker_series(
                painter,
                transform,
                model.observed_xy_nm,
                marker=style["marker"],
                color=style["color"],
                radius=_field_marker_radius(plot_w, len(model.observed_xy_nm)),
            )
        painter.restore()

        self._draw_chrome(painter, plot_rect, fg, border)
        self._draw_labels(painter, fg)
        self._draw_legend(painter, plot_rect, fg)
        self._draw_direct_labels(painter, plot_rect, fg)
        if visible_layers == 0 and model.status:
            painter.save()
            painter.setPen(fg)
            font = QFont(painter.font())
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(plot_rect.adjusted(20, 20, -20, -20), Qt.AlignCenter | Qt.TextWordWrap, model.status)
            painter.restore()

    def _plot_rect(self) -> QRectF:
        margin_l, margin_t, margin_r, margin_b = 68.0, 52.0, 28.0, 86.0
        available = QRectF(
            margin_l,
            margin_t,
            max(1.0, self.width() - margin_l - margin_r),
            max(1.0, self.height() - margin_t - margin_b),
        )
        return _aspect_fit_rect(available, self._model.width_nm, self._model.height_nm)

    def _draw_mask(self, painter: QPainter, plot_rect: QRectF) -> None:
        mask = self._model.mask
        if mask is None or mask.size == 0:
            return
        rows, cols = mask.shape
        step_y = max(1, rows // 96)
        step_x = max(1, cols // 96)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 229, 255, 42))
        for y in range(0, rows, step_y):
            y1 = min(rows, y + step_y)
            for x in range(0, cols, step_x):
                x1 = min(cols, x + step_x)
                if not bool(mask[y:y1, x:x1].any()):
                    continue
                rx = plot_rect.left() + (x / cols) * plot_rect.width()
                ry = plot_rect.top() + (y / rows) * plot_rect.height()
                rw = ((x1 - x) / cols) * plot_rect.width()
                rh = ((y1 - y) / rows) * plot_rect.height()
                painter.drawRect(QRectF(rx, ry, max(1.0, rw), max(1.0, rh)))

    def _draw_chrome(
        self,
        painter: QPainter,
        plot_rect: QRectF,
        fg: QColor,
        border: QColor,
    ) -> None:
        model = self._model
        painter.save()
        painter.setPen(QPen(border, 1.0))
        painter.drawRect(plot_rect)
        painter.setPen(QPen(fg, 1.0))
        font = QFont(painter.font())
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(
            QRectF(plot_rect.left(), plot_rect.bottom() + 8, plot_rect.width(), 22),
            Qt.AlignCenter,
            f"x: 0 to {model.width_nm:g} nm",
        )
        painter.save()
        painter.translate(20, plot_rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-plot_rect.height() / 2, 0, plot_rect.height(), 22), Qt.AlignCenter, f"y: 0 to {model.height_nm:g} nm")
        painter.restore()
        painter.restore()

    def _draw_labels(self, painter: QPainter, fg: QColor) -> None:
        model = self._model
        painter.save()
        painter.setPen(fg)
        title_font = QFont(painter.font())
        title_font.setPointSize(13)
        title_font.setBold(True)
        painter.setFont(title_font)
        title = "Generated particle field" if model.mode == "generated" else "Observed particle field"
        painter.drawText(QRectF(14, 8, self.width() - 28, 26), Qt.AlignLeft, title)
        body_font = QFont(painter.font())
        body_font.setPointSize(10)
        body_font.setBold(False)
        painter.setFont(body_font)
        detail = "  ".join(
            part
            for part in (
                model.source_label,
                model.region_label,
                model.model_label,
                f"N={len(model.observed_xy_nm)}",
            )
            if part
        )
        painter.drawText(QRectF(14, 32, self.width() - 28, 22), Qt.AlignLeft, detail)
        painter.restore()

    def _draw_legend(self, painter: QPainter, plot_rect: QRectF, fg: QColor) -> None:
        model = self._model
        legend = []
        if model.show_observed:
            legend.append(("observed", _marker_style(model.mode)["color"], _marker_style(model.mode)["marker"], False))
        if model.mode == "generated" and model.show_simulated and model.simulated_xy_nm is not None:
            legend.append(("model sample", "#b96adf", "o", True))
        if model.mode == "generated" and model.show_features and model.feature_xy_nm is not None:
            legend.append(("feature layer", "#9b5de5", "x", False))
        if not legend:
            return
        # Horizontal strip in the bottom margin (below the x-axis label) so the legend
        # never sits on top of the points.
        painter.save()
        font = QFont(painter.font())
        font.setPointSize(10)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        gap = 18.0
        sample_w = 20.0
        items = [(label, color, marker, hollow, float(metrics.horizontalAdvance(label))) for label, color, marker, hollow in legend]
        total = sum(sample_w + tw for *_unused, tw in items) + gap * (len(items) - 1)
        y = plot_rect.bottom() + 40.0
        x = max(plot_rect.left(), plot_rect.center().x() - total / 2.0)
        for label, color, marker, hollow, text_w in items:
            _draw_marker(painter, QPointF(x + 7, y), marker, QColor(color), 4.4, hollow=hollow)
            painter.setPen(QPen(fg))
            painter.drawText(QRectF(x + sample_w, y - 9.0, text_w + 4.0, 18.0), Qt.AlignLeft | Qt.AlignVCenter, label)
            x += sample_w + text_w + gap
        painter.restore()

    def _draw_direct_labels(self, painter: QPainter, plot_rect: QRectF, fg: QColor) -> None:
        labels = tuple(self._model.direct_labels)
        if not labels:
            return
        painter.save()
        font = QFont(painter.font())
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        x = plot_rect.left() + 10.0
        y = plot_rect.top() + 10.0
        for label in labels[:4]:
            text_w = float(metrics.horizontalAdvance(label))
            box = QRectF(x, y, text_w + 18.0, 24.0)
            painter.setPen(QPen(QColor("#2fb344"), 1.2))
            painter.setBrush(QColor(47, 179, 68, 52))
            painter.drawRoundedRect(box, 4.0, 4.0)
            painter.setPen(QPen(fg))
            painter.drawText(box.adjusted(9, 0, -9, 0), Qt.AlignLeft | Qt.AlignVCenter, label)
            y += 28.0
        painter.restore()


class _FieldTransform:
    def __init__(self, rect: QRectF, width_nm: float, height_nm: float):
        self.rect = rect
        self.width_nm = max(float(width_nm), 1e-9)
        self.height_nm = max(float(height_nm), 1e-9)

    def point(self, xy: Any) -> QPointF:
        x = float(xy[0])
        y = float(xy[1])
        return QPointF(
            self.rect.left() + (x / self.width_nm) * self.rect.width(),
            self.rect.top() + (y / self.height_nm) * self.rect.height(),
        )


def _aspect_fit_rect(available: QRectF, width_nm: float, height_nm: float) -> QRectF:
    """Return the largest centered rect that preserves physical field aspect."""

    target_ratio = max(float(width_nm), 1e-9) / max(float(height_nm), 1e-9)
    available_ratio = max(float(available.width()), 1.0) / max(float(available.height()), 1.0)
    if target_ratio >= available_ratio:
        width = float(available.width())
        height = width / target_ratio
    else:
        height = float(available.height())
        width = height * target_ratio
    x = available.left() + (available.width() - width) / 2.0
    y = available.top() + (available.height() - height) / 2.0
    return QRectF(x, y, max(1.0, width), max(1.0, height))


def _draw_marker_series(
    painter: QPainter,
    transform: _FieldTransform,
    points: np.ndarray,
    *,
    marker: str,
    color: str,
    radius: float,
    hollow: bool = False,
) -> None:
    for xy in np.asarray(points, dtype=float):
        if not np.isfinite(xy).all():
            continue
        _draw_marker(painter, transform.point(xy), marker, QColor(color), radius, hollow=hollow)


def _draw_marker(
    painter: QPainter,
    center: QPointF,
    marker: str,
    color: QColor,
    radius: float,
    *,
    hollow: bool = False,
) -> None:
    painter.save()
    painter.setPen(QPen(color, 1.5))
    painter.setBrush(Qt.NoBrush if hollow else color)
    if marker == "^":
        polygon = QPolygonF(
            [
                QPointF(center.x(), center.y() - radius),
                QPointF(center.x() - radius, center.y() + radius),
                QPointF(center.x() + radius, center.y() + radius),
            ]
        )
        painter.drawPolygon(polygon)
    elif marker == "s":
        painter.drawRect(QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2))
    elif marker == "x":
        painter.drawLine(QPointF(center.x() - radius, center.y() - radius), QPointF(center.x() + radius, center.y() + radius))
        painter.drawLine(QPointF(center.x() - radius, center.y() + radius), QPointF(center.x() + radius, center.y() - radius))
    else:
        painter.drawEllipse(center, radius, radius)
    painter.restore()


def _marker_style(mode: str) -> dict[str, str]:
    if _normalise_field_mode(mode) == "generated":
        return {"marker": "^", "color": "#f28e2b"}
    return {"marker": "o", "color": "#2f7ed8"}


def _field_marker_radius(plot_width_px: float, n_points: int) -> float:
    """Scale field markers to point density so dense fields do not overlap.

    Mean point spacing in pixels is ~ plot_width / sqrt(N); a fraction of that gives
    a radius that shrinks as more points are packed in (≈3.6 px at N=120, ≈2.4 px at
    N=500), clamped to a legible range.
    """
    n = max(int(n_points), 1)
    spacing_px = float(plot_width_px) / math.sqrt(n)
    return max(2.0, min(4.0, 0.13 * spacing_px))


def _normalise_field_mode(mode: str) -> str:
    return "generated" if str(mode).lower() in {"generated", "sandbox", "learn"} else "real"


def _xy_array(value: Any) -> np.ndarray:
    arr = _xy_array_or_none(value)
    return np.empty((0, 2), dtype=float) if arr is None else arr


def _xy_array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr


def _mask_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=bool)
    if arr.ndim != 2 or not arr.any():
        return None
    return arr


def _theme_qcolor(theme: dict, keys: tuple[str, ...], default: str) -> QColor:
    for key in keys:
        value = theme.get(key)
        if value:
            return QColor(str(value))
    return QColor(default)
