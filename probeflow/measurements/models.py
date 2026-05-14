"""Simple measurement data models."""

from __future__ import annotations

from dataclasses import dataclass, field

Scalar = float | int | str | None


@dataclass(frozen=True)
class MeasurementResult:
    """A compact, exportable measurement record."""

    measurement_id: str
    kind: str
    source_label: str
    source_path: str | None
    channel: str | None
    x_unit: str | None
    y_unit: str | None
    z_unit: str | None = None
    values: dict[str, Scalar] = field(default_factory=dict)
    context: dict[str, Scalar] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class FeaturePoint:
    """One detected point feature in image and physical coordinates."""

    point_id: str
    x_px: float
    y_px: float
    x_phys: float
    y_phys: float
    z_value: float
    channel: str
    source_label: str
    roi_id: str | None = None


def measurement_main_value(result: MeasurementResult) -> tuple[str, Scalar, str | None]:
    """Return a compact display value for a measurement table row."""
    height_unit = result.z_unit or result.context.get("height_unit")
    preferences = {
        "spectrum_delta": ("dx", result.x_unit),
        "roi_stats": ("mean_height", height_unit),
        "step_height": ("height_difference", height_unit),
        "line_profile": ("length", result.x_unit),
        "feature_maxima": ("n_points", None),
        "point_fft": ("dominant_frequency", result.x_unit),
    }
    key, unit = preferences.get(result.kind, ("", None))
    if key and key in result.values:
        return key, result.values[key], unit
    if result.values:
        first_key = next(iter(result.values))
        return first_key, result.values[first_key], None
    return "", None, None
