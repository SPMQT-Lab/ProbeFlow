"""ROI statistics backend returning the canonical measurement format.

This module is a thin convenience wrapper around the canonical
:func:`probeflow.measurements.image.roi_statistics` kernel — it adds
the GUI-side extras (``area_nm2`` value, human-readable ``summary``
string) and exposes the legacy parameter spelling
(``roi_mask`` positional, ``pixel_size_x_m`` kw) that the
image-viewer mixin call site relies on.  Review arch-backend #4
(2026-05-28) consolidated the two parallel kernels here — the
arithmetic now happens in exactly one place.
"""

from __future__ import annotations

import numpy as np

from probeflow.measurements.image import roi_statistics
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
    # Shape-mismatch check matches the legacy error message phrasing
    # because tests assert ``ValueError(match="shape")``.  The canonical
    # kernel also raises but with a different message.
    arr = np.asarray(image, dtype=np.float64)
    mask = np.asarray(roi_mask, dtype=bool)
    if arr.shape != mask.shape:
        raise ValueError(
            f"image shape {arr.shape} != mask shape {mask.shape}"
        )

    # Delegate to the canonical kernel.  It already handles the
    # all-NaN error and emits every shared value key
    # (mean_height, median_height, std_height, rms_roughness,
    # peak_to_peak, min_height, max_height, n_finite_pixels,
    # n_nonfinite_pixels, area).  We then patch in the
    # area_nm2/summary GUI extras.
    base = roi_statistics(
        arr,
        measurement_id=measurement_id or "M?",
        source_label=source,
        source_path=source or None,
        channel=channel,
        mask=mask,
        pixel_size_x=pixel_size_x_m,
        pixel_size_y=pixel_size_y_m,
        x_unit="m",
        y_unit="m",
        height_unit=z_unit,
        data_basis="processed_image",
        notes=roi_name,
    )

    # GUI-convenience extras the kernel does not compute.
    area_m2 = float(base.values["area"])
    area_nm2 = area_m2 * 1e18
    mean_val = float(base.values["mean_height"])
    rms_val = float(base.values["rms_roughness"])
    range_val = float(base.values["peak_to_peak"])

    mean_v, mean_u = _fmt_z(mean_val, z_unit)
    rms_v, rms_u = _fmt_z(rms_val, z_unit)
    range_v, range_u = _fmt_z(range_val, z_unit)
    summary = (
        f"Area: {area_nm2:.4g} nm²"
        f"  Mean: {mean_v:.4g} {mean_u}"
        f"  RMS: {rms_v:.4g} {rms_u}"
        f"  Range: {range_v:.4g} {range_u}"
    )

    # Add legacy convenience keys.  ``MeasurementResult`` is frozen
    # but ``values``/``context`` are mutable dicts.
    base.values["area_m2"] = area_m2
    base.values["area_nm2"] = area_nm2
    base.context["summary"] = summary
    base.context["roi_id"] = roi_id
    base.context["roi_name"] = roi_name
    base.context["area_unit"] = "nm^2"
    return base


def _fmt_z(value: float, z_unit: str) -> tuple[float, str]:
    """Scale a z-axis value to a readable unit.

    For metre-based units (z_unit == "m") delegates to the consolidated
    :func:`probeflow.measurements.formatting.scale_length_m`.  Otherwise
    returns the raw value with z_unit unchanged.  Review arch-backend #5.
    """
    if z_unit == "m":
        from probeflow.measurements.formatting import scale_length_m
        return scale_length_m(value)
    return float(value), z_unit
