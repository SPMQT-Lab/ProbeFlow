"""ROI statistics backend returning the canonical measurement format."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from probeflow.analysis.simple_measurements import _fmt_m
from probeflow.measurements.models import MeasurementResult


def compute_roi_statistics(
    image: np.ndarray,
    roi_mask: np.ndarray,
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    z_unit: str = "m",
    measurement_id: str = "",
    source: str = "",
    channel: str = "",
    roi_id: str | None = None,
    roi_name: str = "",
) -> MeasurementResult:
    """Compute descriptive statistics for an image region.

    Parameters
    ----------
    image
        2-D float array (physical units, e.g. metres for STM height).
    roi_mask
        Boolean array of the same shape — True inside the ROI.
    pixel_size_x_m, pixel_size_y_m
        Physical pixel dimensions in metres (used for area).
    z_unit
        SI unit string for image values (e.g. ``"m"`` or ``"A"``).
    """
    arr = np.asarray(image, dtype=np.float64)
    mask = np.asarray(roi_mask, dtype=bool)
    if arr.shape != mask.shape:
        raise ValueError(
            f"image shape {arr.shape} != mask shape {mask.shape}"
        )

    finite = arr[mask]
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("ROI contains no finite image pixels.")

    n = int(finite.size)
    mean_val = float(np.mean(finite))
    median_val = float(np.median(finite))
    std_val = float(np.std(finite))
    rms_val = float(math.sqrt(float(np.mean((finite - mean_val) ** 2))))
    range_val = float(np.max(finite) - np.min(finite))
    area_m2 = float(n) * float(pixel_size_x_m) * float(pixel_size_y_m)
    area_nm2 = area_m2 * 1e18

    # Format z values for human-readable summary (use z_unit prefix table).
    mean_v, mean_u = _fmt_z(mean_val, z_unit)
    rms_v, rms_u = _fmt_z(rms_val, z_unit)
    range_v, range_u = _fmt_z(range_val, z_unit)

    summary = (
        f"Area: {area_nm2:.4g} nm²"
        f"  Mean: {mean_v:.4g} {mean_u}"
        f"  RMS: {rms_v:.4g} {rms_u}"
        f"  Range: {range_v:.4g} {range_u}"
    )

    return MeasurementResult(
        measurement_id=measurement_id or "M?",
        kind="roi_stats",
        source_label=source,
        source_path=source or None,
        channel=channel,
        x_unit="m",
        y_unit="m",
        z_unit=z_unit,
        values={
            "area_m2": area_m2,
            "area_nm2": area_nm2,
            "n_finite_pixels": n,
            "mean_height": mean_val,
            "median_height": median_val,
            "std_height": std_val,
            "rms_roughness": rms_val,
            "peak_to_peak": range_val,
        },
        context={
            "summary": summary,
            "roi_id": roi_id,
            "roi_name": roi_name,
            "area_unit": "nm^2",
        },
        notes=roi_name,
    )


def _fmt_z(value: float, z_unit: str) -> tuple[float, str]:
    """Scale a z-axis value to a readable unit.

    For metre-based units (z_unit == "m") delegates to ``_fmt_m``.
    Otherwise returns the raw value with z_unit unchanged.
    """
    if z_unit == "m":
        return _fmt_m(value)
    return float(value), z_unit
