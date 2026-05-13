"""Display-trace measurement helpers for spectroscopy views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from probeflow.spectroscopy.models import DisplayedSpectrum


@dataclass(frozen=True)
class SpectrumMeasurementPoint:
    """A snapped point on a displayed spectroscopy trace."""

    source_file: str
    spectrum_id: str
    trace_label: str
    y_channel: str
    index: int
    x: float
    y: float
    x_label: str
    y_label: str
    x_unit: str
    y_unit: str

    @property
    def trace_key(self) -> tuple[str, str, str]:
        return (self.source_file, self.spectrum_id, self.y_channel)

    @property
    def trace_name(self) -> str:
        name = Path(self.source_file).name or self.spectrum_id
        return f"{name}:{self.y_channel}"


@dataclass(frozen=True)
class SpectrumDeltaMeasurement:
    """Delta measurement between two displayed spectroscopy points."""

    point1: SpectrumMeasurementPoint
    point2: SpectrumMeasurementPoint

    @property
    def dx(self) -> float:
        return self.point2.x - self.point1.x

    @property
    def dy(self) -> float:
        return self.point2.y - self.point1.y

    @property
    def slope(self) -> float:
        dx = self.dx
        return self.dy / dx if dx != 0.0 else float("nan")

    @property
    def slope_unit(self) -> str:
        y_unit = self.point1.y_unit
        x_unit = self.point1.x_unit
        if y_unit and x_unit:
            return f"{y_unit}/{x_unit}"
        if y_unit:
            return y_unit
        if x_unit:
            return f"1/{x_unit}"
        return ""


def nearest_finite_point(
    trace: DisplayedSpectrum,
    x_click: float,
    y_click: float,
    *,
    max_normalized_distance: float | None = 0.08,
) -> SpectrumMeasurementPoint | None:
    """Return the nearest finite displayed point to a click, or ``None``."""

    x = np.asarray(trace.x_display, dtype=np.float64)
    y = np.asarray(trace.y_display, dtype=np.float64)
    if x.shape != y.shape or x.size == 0:
        return None
    finite = np.isfinite(x) & np.isfinite(y)
    if trace.mask is not None:
        mask = np.asarray(trace.mask, dtype=bool)
        if mask.shape == finite.shape:
            finite &= mask
    if not np.any(finite):
        return None

    x_span = _finite_span(x[finite])
    y_span = _finite_span(y[finite])
    distances = ((x - float(x_click)) / x_span) ** 2 + ((y - float(y_click)) / y_span) ** 2
    distances[~finite] = np.inf
    idx = int(np.argmin(distances))
    distance = float(np.sqrt(distances[idx]))
    if max_normalized_distance is not None and distance > max_normalized_distance:
        return None
    return _point_from_trace(trace, idx)


def nearest_point_across_traces(
    traces: Iterable[DisplayedSpectrum],
    x_click: float,
    y_click: float,
    *,
    max_normalized_distance: float | None = 0.08,
) -> SpectrumMeasurementPoint | None:
    """Return the nearest finite displayed point across plotted traces."""

    best: tuple[float, SpectrumMeasurementPoint] | None = None
    for trace in traces:
        point = nearest_finite_point(
            trace,
            x_click,
            y_click,
            max_normalized_distance=max_normalized_distance,
        )
        if point is None:
            continue
        distance = _normalized_point_distance(trace, point.index, x_click, y_click)
        if best is None or distance < best[0]:
            best = (distance, point)
    return best[1] if best is not None else None


def measure_delta(
    point1: SpectrumMeasurementPoint,
    point2: SpectrumMeasurementPoint,
) -> SpectrumDeltaMeasurement:
    """Return a same-trace delta measurement."""

    if point1.trace_key != point2.trace_key:
        raise ValueError("crosshair delta measurements require points on the same trace")
    return SpectrumDeltaMeasurement(point1=point1, point2=point2)


def measurement_to_tsv(measurement: SpectrumDeltaMeasurement) -> str:
    """Return one tabular row plus header for lab notebooks and spreadsheets."""

    p1 = measurement.point1
    p2 = measurement.point2
    header = "\t".join([
        "trace",
        "x1",
        "y1",
        "x2",
        "y2",
        "dx",
        "dy",
        "slope",
        "x_unit",
        "y_unit",
        "slope_unit",
    ])
    row = "\t".join([
        p1.trace_name,
        _fmt_number(p1.x),
        _fmt_number(p1.y),
        _fmt_number(p2.x),
        _fmt_number(p2.y),
        _fmt_number(measurement.dx),
        _fmt_number(measurement.dy),
        _fmt_number(measurement.slope),
        p1.x_unit,
        p1.y_unit,
        measurement.slope_unit,
    ])
    return f"{header}\n{row}\n"


def format_measurement_summary(measurement: SpectrumDeltaMeasurement) -> str:
    """Return compact displayed-trace measurement text."""

    p1 = measurement.point1
    p2 = measurement.point2
    return (
        "Displayed trace measurement | "
        f"Trace: {p1.trace_name} | "
        f"P1: {p1.x_label}={_fmt_value(p1.x, p1.x_unit)}, "
        f"{p1.y_label}={_fmt_value(p1.y, p1.y_unit)} | "
        f"P2: {p2.x_label}={_fmt_value(p2.x, p2.x_unit)}, "
        f"{p2.y_label}={_fmt_value(p2.y, p2.y_unit)} | "
        f"Delta x={_fmt_value(measurement.dx, p1.x_unit)}, "
        f"Delta y={_fmt_value(measurement.dy, p1.y_unit)}, "
        f"slope={_fmt_value(measurement.slope, measurement.slope_unit)}"
    )


def _point_from_trace(trace: DisplayedSpectrum, idx: int) -> SpectrumMeasurementPoint:
    return SpectrumMeasurementPoint(
        source_file=trace.source_file,
        spectrum_id=trace.spectrum_id,
        trace_label=trace.label,
        y_channel=trace.y_channel,
        index=idx,
        x=float(trace.x_display[idx]),
        y=float(trace.y_display[idx]),
        x_label=trace.x_label,
        y_label=trace.y_label,
        x_unit=trace.x_unit,
        y_unit=trace.y_unit,
    )


def _normalized_point_distance(
    trace: DisplayedSpectrum,
    idx: int,
    x_click: float,
    y_click: float,
) -> float:
    x = np.asarray(trace.x_display, dtype=np.float64)
    y = np.asarray(trace.y_display, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x_span = _finite_span(x[finite])
    y_span = _finite_span(y[finite])
    return float(np.hypot((x[idx] - x_click) / x_span, (y[idx] - y_click) / y_span))


def _finite_span(values: np.ndarray) -> float:
    span = float(np.nanmax(values) - np.nanmin(values)) if values.size else 0.0
    return span if span > 0.0 else 1.0


def _fmt_value(value: float, unit: str) -> str:
    number = _fmt_number(value)
    return f"{number} {unit}".rstrip()


def _fmt_number(value: float) -> str:
    return f"{float(value):.6g}"
