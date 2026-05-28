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
    """One detected point feature.

    Pixel coordinates (``x_px``, ``y_px``) and the channel value at that
    pixel (``z_value``) are the intrinsic detection-time fields and have
    no defaults.  All other fields are context that the detector may or
    may not have available — they default to empty / NaN so the
    detection-time call sites (e.g.
    :func:`probeflow.analysis.feature_finder.find_image_features`) can
    construct the canonical type without supplying calibration or
    source metadata they do not have.  The downstream measurement
    helpers (e.g.
    :func:`probeflow.measurements.features.detect_local_maxima`) pass
    the full set of kwargs.

    Review arch-backend #2 (2026-05-28) unified this with the previous
    smaller ``probeflow.analysis.feature_finder.FeaturePoint`` (4
    fields, now removed).  The legacy ``value`` field is renamed to
    ``z_value``; the legacy ``label`` field was never populated in
    production and is dropped.
    """

    x_px: float
    y_px: float
    z_value: float
    point_id: str = ""
    x_phys: float = float("nan")
    y_phys: float = float("nan")
    channel: str = ""
    source_label: str = ""
    roi_id: str | None = None


def measurement_main_value(result: MeasurementResult) -> tuple[str, Scalar, str | None]:
    """Return a compact display value for a measurement table row."""
    height_unit = result.z_unit or result.context.get("height_unit")
    preferences = {
        "spectrum_delta": ("dy", result.y_unit),
        "roi_stats": ("mean_height", height_unit),
        "step_height": ("height_difference", height_unit),
        "line_profile": ("length", result.x_unit),
        "line_profile_delta": ("delta_y", result.y_unit),
        "feature_maxima": ("n_points", None),
        "point_fft": ("dominant_frequency", result.x_unit),
        "line_periodicity": ("period_m", "m"),
        "pair_corr": ("nn_median_nm", "nm"),
        "feat_lattice": ("rms_displacement_m", "m"),
        "distance": ("length_m", "m"),
        "angle":    ("angle_deg", "°"),
    }
    key, unit = preferences.get(result.kind, ("", None))
    if key and key in result.values:
        return key, result.values[key], unit
    if result.values:
        first_key = next(iter(result.values))
        return first_key, result.values[first_key], None
    return "", None, None
