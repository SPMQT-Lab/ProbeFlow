"""Pure image measurement helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from probeflow.measurements.models import MeasurementResult, Scalar


def roi_statistics(
    image: np.ndarray,
    *,
    measurement_id: str,
    source_label: str,
    source_path: str | None = None,
    channel: str | None = None,
    roi: Any | None = None,
    mask: np.ndarray | None = None,
    pixel_size_x: float | None = None,
    pixel_size_y: float | None = None,
    x_unit: str | None = None,
    y_unit: str | None = None,
    height_unit: str | None = None,
    data_basis: str = "processed_image",
    notes: str = "",
) -> MeasurementResult:
    """Return statistics for an ROI or mask without modifying ``image``."""
    arr = np.asarray(image, dtype=np.float64)
    selected_mask = _mask_from_roi_or_mask(arr.shape, roi=roi, mask=mask)
    values = arr[selected_mask]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("ROI contains no finite image pixels")

    area = _area_from_mask(selected_mask, pixel_size_x=pixel_size_x, pixel_size_y=pixel_size_y)
    roi_id = getattr(roi, "id", None)
    roi_name = getattr(roi, "name", None)
    context: dict[str, Scalar] = {
        "data_basis": data_basis,
        "roi_id": roi_id,
        "roi_name": roi_name,
        "height_unit": height_unit,
        "area_unit": f"{x_unit}^2" if x_unit and x_unit == y_unit else None,
    }
    return MeasurementResult(
        measurement_id=measurement_id,
        kind="roi_stats",
        source_label=source_label,
        source_path=source_path,
        channel=channel,
        x_unit=x_unit,
        y_unit=y_unit,
        z_unit=height_unit,
        values={
            "mean_height": float(np.mean(finite)),
            "median_height": float(np.median(finite)),
            "std_height": float(np.std(finite)),
            "rms_roughness": float(np.sqrt(np.mean((finite - np.mean(finite)) ** 2))),
            "min_height": float(np.min(finite)),
            "max_height": float(np.max(finite)),
            "peak_to_peak": float(np.max(finite) - np.min(finite)),
            "area": area,
            "n_finite_pixels": int(finite.size),
            "n_nonfinite_pixels": int(values.size - finite.size),
        },
        context=context,
        notes=notes,
    )


def step_height_from_rois(
    image: np.ndarray,
    roi_a: Any,
    roi_b: Any,
    *,
    measurement_id: str,
    source_label: str,
    source_path: str | None = None,
    channel: str | None = None,
    x_unit: str | None = None,
    y_unit: str | None = None,
    height_unit: str | None = None,
    data_basis: str = "processed_image",
    notes: str = "",
) -> MeasurementResult:
    """Return a two-ROI step-height measurement using finite pixel means."""
    arr = np.asarray(image, dtype=np.float64)
    mask_a = _mask_from_roi_or_mask(arr.shape, roi=roi_a)
    mask_b = _mask_from_roi_or_mask(arr.shape, roi=roi_b)
    vals_a = arr[mask_a]
    vals_b = arr[mask_b]
    finite_a = vals_a[np.isfinite(vals_a)]
    finite_b = vals_b[np.isfinite(vals_b)]
    if finite_a.size == 0 or finite_b.size == 0:
        raise ValueError("both ROIs must contain at least one finite image pixel")
    mean_a = float(np.mean(finite_a))
    mean_b = float(np.mean(finite_b))
    return MeasurementResult(
        measurement_id=measurement_id,
        kind="step_height",
        source_label=source_label,
        source_path=source_path,
        channel=channel,
        x_unit=x_unit,
        y_unit=y_unit,
        z_unit=height_unit,
        values={
            "roi_a_mean": mean_a,
            "roi_b_mean": mean_b,
            "height_difference": mean_b - mean_a,
            "roi_a_std": float(np.std(finite_a)),
            "roi_b_std": float(np.std(finite_b)),
            "roi_a_median": float(np.median(finite_a)),
            "roi_b_median": float(np.median(finite_b)),
            "roi_a_n": int(finite_a.size),
            "roi_b_n": int(finite_b.size),
        },
        context={
            "data_basis": data_basis,
            "roi_a_id": getattr(roi_a, "id", None),
            "roi_b_id": getattr(roi_b, "id", None),
            "roi_a_name": getattr(roi_a, "name", None),
            "roi_b_name": getattr(roi_b, "name", None),
            "height_unit": height_unit,
        },
        notes=notes,
    )


def line_profile_measurement(
    distance: np.ndarray,
    profile: np.ndarray,
    *,
    measurement_id: str,
    source_label: str,
    source_path: str | None = None,
    channel: str | None = None,
    x_unit: str | None = None,
    y_unit: str | None = None,
    p0: tuple[float, float] | None = None,
    p1: tuple[float, float] | None = None,
    roi_id: str | None = None,
    roi_name: str | None = None,
    swath_width: float | None = None,
    averaging_method: str | None = None,
    data_basis: str = "processed_image",
    notes: str = "",
) -> MeasurementResult:
    """Return an exportable summary for a computed line profile."""
    s = np.asarray(distance, dtype=np.float64)
    z = np.asarray(profile, dtype=np.float64)
    if s.shape != z.shape:
        raise ValueError("distance and profile arrays must have matching shapes")
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise ValueError("line profile contains no finite values")
    values: dict[str, Scalar] = {
        "length": float(s[-1] - s[0]) if s.size else 0.0,
        "height_difference": float(np.max(finite) - np.min(finite)),
        "n_points": int(z.size),
    }
    if p0 is not None:
        values.update({"x1": float(p0[0]), "y1": float(p0[1])})
    if p1 is not None:
        values.update({"x2": float(p1[0]), "y2": float(p1[1])})
    if p0 is not None and p1 is not None:
        values["length_px"] = float(
            np.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
        )
    context: dict[str, Scalar] = {
        "data_basis": data_basis,
        "roi_id": roi_id,
        "roi_name": roi_name,
        "swath_width": swath_width,
        "averaging_method": averaging_method,
    }
    return MeasurementResult(
        measurement_id=measurement_id,
        kind="line_profile",
        source_label=source_label,
        source_path=source_path,
        channel=channel,
        x_unit=x_unit,
        y_unit=y_unit,
        values=values,
        context=context,
        notes=notes,
    )


def line_profile_delta_measurement(
    *,
    delta_x: float,
    delta_y: float,
    p1_distance: float,
    p1_height: float,
    p2_distance: float,
    p2_height: float,
    measurement_id: str,
    source_label: str,
    source_path: str | None = None,
    channel: str | None = None,
    x_unit: str | None = None,
    y_unit: str | None = None,
    roi_id: str | None = None,
    roi_name: str | None = None,
    data_basis: str = "processed_image",
    notes: str = "",
) -> MeasurementResult:
    """Return a measurement record for a two-point line-profile delta."""
    values: dict[str, Scalar] = {
        "delta_x": float(delta_x),
        "delta_y": float(delta_y),
        "p1_distance": float(p1_distance),
        "p1_height": float(p1_height),
        "p2_distance": float(p2_distance),
        "p2_height": float(p2_height),
    }
    context: dict[str, Scalar] = {
        "data_basis": data_basis,
        "roi_id": roi_id,
        "roi_name": roi_name,
    }
    return MeasurementResult(
        measurement_id=measurement_id,
        kind="line_profile_delta",
        source_label=source_label,
        source_path=source_path,
        channel=channel,
        x_unit=x_unit,
        y_unit=y_unit,
        values=values,
        context=context,
        notes=notes,
    )


def line_periodicity_measurement(
    result: "Any",
    *,
    measurement_id: str,
    source_label: str,
    source_path: str | None = None,
    channel: str | None = None,
    roi_id: str | None = None,
    roi_name: str | None = None,
    background: str = "linear",
    smoothing: str = "light_gaussian",
    width_px: float = 1.0,
    data_basis: str = "processed_image",
    notes: str = "",
) -> MeasurementResult:
    """Wrap a PeriodicityResult in a MeasurementResult for the table."""
    import math
    values: dict[str, Scalar] = {
        "period_m": result.period_m if not math.isnan(result.period_m) else None,
        "line_length_m": result.line_length_m,
        "n_periods": result.n_periods,
        "n_samples": result.n_samples,
    }
    if result.uncertainty_m is not None:
        values["uncertainty_m"] = result.uncertainty_m
    context: dict[str, Scalar] = {
        "method": result.method,
        "background": background,
        "smoothing": smoothing,
        "quality": result.quality,
        "message": result.message,
        "roi_id": roi_id,
        "roi_name": roi_name,
        "width_px": width_px,
        "data_basis": data_basis,
    }
    return MeasurementResult(
        measurement_id=measurement_id,
        kind="line_periodicity",
        source_label=source_label,
        source_path=source_path,
        channel=channel,
        x_unit="m",
        y_unit="m",
        z_unit=None,
        values=values,
        context=context,
        notes=notes,
    )


def _mask_from_roi_or_mask(
    shape: tuple[int, ...],
    *,
    roi: Any | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    image_shape = tuple(shape[:2])
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != image_shape:
            raise ValueError("mask shape must match image shape")
        return selected.copy()
    if roi is not None:
        selected = np.asarray(roi.to_mask(image_shape), dtype=bool)
        if selected.shape != image_shape:
            raise ValueError("ROI mask shape must match image shape")
        return selected
    return np.ones(image_shape, dtype=bool)


def _area_from_mask(
    mask: np.ndarray,
    *,
    pixel_size_x: float | None = None,
    pixel_size_y: float | None = None,
) -> float:
    n_pixels = int(np.count_nonzero(mask))
    if pixel_size_x is None or pixel_size_y is None:
        return float(n_pixels)
    return float(n_pixels * float(pixel_size_x) * float(pixel_size_y))
