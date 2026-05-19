"""Distance and angle measurements from ROI geometry.

GUI-free. Accepts ROI objects from probeflow.core.roi and returns
MeasurementResult from probeflow.analysis.measurements.
"""

from __future__ import annotations

import math
from typing import Any

from probeflow.analysis.measurements import MeasurementResult


def measure_line_distance(
    roi: Any,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    *,
    measurement_id: str = "",
    source: str = "",
    channel: str = "",
    roi_name: str | None = None,
) -> MeasurementResult:
    """Compute distance, dx, dy and angle from a line ROI.

    Parameters
    ----------
    roi
        A line ROI whose geometry has x1, y1, x2, y2 in pixel coordinates.
    pixel_size_x_m, pixel_size_y_m
        Physical pixel dimensions in metres.
    """
    if getattr(roi, "kind", None) != "line":
        raise ValueError(f"Expected a line ROI, got kind={getattr(roi, 'kind', '?')!r}")
    g = roi.geometry
    x1 = float(g.get("x1", 0.0))
    y1 = float(g.get("y1", 0.0))
    x2 = float(g.get("x2", 0.0))
    y2 = float(g.get("y2", 0.0))

    dx_m = (x2 - x1) * float(pixel_size_x_m)
    dy_m = (y2 - y1) * float(pixel_size_y_m)
    length_m = math.hypot(dx_m, dy_m)
    angle_deg = math.degrees(math.atan2(abs(dy_m), abs(dx_m)))

    l_v, l_u = _fmt_m(length_m)
    dx_v, dx_u = _fmt_m(abs(dx_m))
    dy_v, dy_u = _fmt_m(abs(dy_m))
    summary = (
        f"{l_v:.4g} {l_u}"
        f"  (Δx={dx_v:.4g} {dx_u}, Δy={dy_v:.4g} {dy_u}, θ={angle_deg:.1f}°)"
    )
    label = roi_name or getattr(roi, "name", str(getattr(roi, "id", "")))

    return MeasurementResult(
        id=measurement_id or "M?",
        kind="distance",
        source=source,
        channel=channel,
        roi_id=getattr(roi, "id", None),
        summary=summary,
        values={
            "length_m": length_m,
            "dx_m": dx_m,
            "dy_m": dy_m,
            "angle_deg": angle_deg,
        },
        units={
            "length_m": "m",
            "dx_m": "m",
            "dy_m": "m",
            "angle_deg": "deg",
        },
        notes=label,
    )


def measure_angle_between_lines(
    roi_a: Any,
    roi_b: Any,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    *,
    measurement_id: str = "",
    source: str = "",
    channel: str = "",
) -> MeasurementResult:
    """Compute the acute angle between two line ROIs.

    The angle reported is always in [0°, 90°].
    """
    for roi, label in ((roi_a, "roi_a"), (roi_b, "roi_b")):
        if getattr(roi, "kind", None) != "line":
            raise ValueError(f"{label} must be a line ROI, got {getattr(roi, 'kind', '?')!r}")

    def _vec(roi) -> tuple[float, float]:
        g = roi.geometry
        dx = (float(g.get("x2", 0)) - float(g.get("x1", 0))) * float(pixel_size_x_m)
        dy = (float(g.get("y2", 0)) - float(g.get("y1", 0))) * float(pixel_size_y_m)
        return dx, dy

    vax, vay = _vec(roi_a)
    vbx, vby = _vec(roi_b)
    mag_a = math.hypot(vax, vay)
    mag_b = math.hypot(vbx, vby)
    if mag_a < 1e-30 or mag_b < 1e-30:
        raise ValueError("One or both line ROIs have zero length.")

    cos_theta = (vax * vbx + vay * vby) / (mag_a * mag_b)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_deg = math.degrees(math.acos(cos_theta))
    if angle_deg > 90.0:
        angle_deg = 180.0 - angle_deg

    name_a = getattr(roi_a, "name", "A")
    name_b = getattr(roi_b, "name", "B")
    summary = f"{angle_deg:.2f}°  ({name_a} ∧ {name_b})"

    return MeasurementResult(
        id=measurement_id or "M?",
        kind="angle",
        source=source,
        channel=channel,
        roi_id=None,
        summary=summary,
        values={"angle_deg": angle_deg},
        units={"angle_deg": "deg"},
        notes=f"{name_a} / {name_b}",
    )


def _fmt_m(value_m: float) -> tuple[float, str]:
    """Convert metres to a readable STM distance unit (pm / Å / nm)."""
    v = abs(value_m)
    if v < 5e-11:       # < 0.05 Å → use pm
        return value_m * 1e12, "pm"
    elif v < 1e-9:      # < 1 nm → use Å
        return value_m * 1e10, "Å"
    else:
        return value_m * 1e9, "nm"
