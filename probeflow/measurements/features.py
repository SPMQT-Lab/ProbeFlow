"""Local feature detection for processed SPM images."""

from __future__ import annotations

from typing import Any

import numpy as np

from probeflow.measurements.models import FeaturePoint, MeasurementResult, Scalar


def detect_local_maxima(
    image: np.ndarray,
    *,
    threshold_mode: str = "absolute",
    threshold_value: float = 0.0,
    min_distance_px: int = 1,
    min_prominence: float | None = None,
    smoothing_sigma: float | None = None,
    max_peaks: int | None = None,
    exclude_border: int = 0,
    roi: Any | None = None,
    roi_mask: np.ndarray | None = None,
    pixel_size_x: float = 1.0,
    pixel_size_y: float = 1.0,
    channel: str = "",
    source_label: str = "",
    roi_id: str | None = None,
) -> list[FeaturePoint]:
    """Detect local maxima in a copy of ``image`` and return point coordinates."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("detect_local_maxima expects a 2-D image")
    work = arr.copy()
    finite = np.isfinite(work)
    if not np.any(finite):
        return []
    fill_value = float(np.nanmedian(work[finite]))
    work[~finite] = fill_value

    if smoothing_sigma is not None and float(smoothing_sigma) > 0.0:
        from scipy.ndimage import gaussian_filter
        work = gaussian_filter(work, sigma=float(smoothing_sigma))

    selected = _roi_mask(arr.shape, roi=roi, roi_mask=roi_mask)
    selected &= finite
    border = int(max(0, exclude_border))
    if border:
        selected[:border, :] = False
        selected[-border:, :] = False
        selected[:, :border] = False
        selected[:, -border:] = False
    if not np.any(selected):
        return []

    threshold = _threshold_value(work[selected], threshold_mode, threshold_value)
    radius = int(max(1, min_distance_px))
    size = 2 * radius + 1
    from scipy.ndimage import maximum_filter, median_filter, minimum_filter
    local_max = work == maximum_filter(work, size=size, mode="nearest")
    local_min = minimum_filter(work, size=size, mode="nearest")
    local_contrast = work > local_min
    candidates = selected & local_max & local_contrast & (work >= threshold)
    if min_prominence is not None:
        local_median = median_filter(work, size=size, mode="nearest")
        candidates &= (work - local_median) >= float(min_prominence)

    rows, cols = np.nonzero(candidates)
    if rows.size == 0:
        return []
    order = np.argsort(work[rows, cols])[::-1]
    accepted: list[tuple[int, int]] = []
    for idx in order:
        row = int(rows[idx])
        col = int(cols[idx])
        if _far_enough(row, col, accepted, radius):
            accepted.append((row, col))
            if max_peaks is not None and len(accepted) >= int(max_peaks):
                break

    point_roi = roi_id or getattr(roi, "id", None)
    return [
        FeaturePoint(
            point_id=f"P{i:04d}",
            x_px=float(col),
            y_px=float(row),
            x_phys=float(col) * float(pixel_size_x),
            y_phys=float(row) * float(pixel_size_y),
            z_value=float(arr[row, col]),
            channel=channel,
            source_label=source_label,
            roi_id=point_roi,
        )
        for i, (row, col) in enumerate(accepted, start=1)
    ]


def feature_maxima_result(
    points: list[FeaturePoint],
    *,
    measurement_id: str,
    source_label: str,
    source_path: str | None = None,
    channel: str | None = None,
    x_unit: str | None = None,
    y_unit: str | None = None,
    threshold_mode: str,
    threshold_value: float,
    min_distance_px: int,
    smoothing_sigma: float | None = None,
    roi_id: str | None = None,
    roi_name: str | None = None,
    selection_scope: str = "roi",
    exclude_border: int = 0,
    data_basis: str = "processed_image",
    notes: str = "",
) -> MeasurementResult:
    """Return a summary MeasurementResult for a detected point list."""
    context: dict[str, Scalar] = {
        "data_basis": data_basis,
        "selection_scope": selection_scope,
        "roi_id": roi_id,
        "roi_name": roi_name,
        "threshold_mode": threshold_mode,
        "detection_smoothing_sigma": smoothing_sigma,
        "exclude_border": exclude_border,
    }
    return MeasurementResult(
        measurement_id=measurement_id,
        kind="feature_maxima",
        source_label=source_label,
        source_path=source_path,
        channel=channel,
        x_unit=x_unit,
        y_unit=y_unit,
        values={
            "n_points": int(len(points)),
            "threshold_value": float(threshold_value),
            "min_distance_px": int(min_distance_px),
        },
        context=context,
        notes=notes,
    )


def _roi_mask(
    shape: tuple[int, int],
    *,
    roi: Any | None,
    roi_mask: np.ndarray | None,
) -> np.ndarray:
    if roi_mask is not None:
        mask = np.asarray(roi_mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError("roi_mask shape must match image shape")
        return mask.copy()
    if roi is not None:
        mask = np.asarray(roi.to_mask(shape), dtype=bool)
        if mask.shape != shape:
            raise ValueError("ROI mask shape must match image shape")
        return mask
    return np.ones(shape, dtype=bool)


def _threshold_value(values: np.ndarray, mode: str, value: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf")
    mode_key = (mode or "absolute").strip().lower().replace("-", "_")
    number = float(value)
    if mode_key == "absolute":
        return number
    if mode_key in {"mean", "mean_offset"}:
        return float(np.mean(finite) + number)
    if mode_key in {"median", "median_offset"}:
        return float(np.median(finite) + number)
    if mode_key in {"mean_std", "std_above_mean"}:
        return float(np.mean(finite) + number * np.std(finite))
    if mode_key in {"median_std", "std_above_median"}:
        return float(np.median(finite) + number * np.std(finite))
    if mode_key in {"percentile", "quantile"}:
        return float(np.percentile(finite, number))
    raise ValueError(f"Unknown threshold mode: {mode!r}")


def _far_enough(row: int, col: int, accepted: list[tuple[int, int]], min_distance: int) -> bool:
    if not accepted:
        return True
    min_d2 = float(min_distance) ** 2
    return all((row - r) ** 2 + (col - c) ** 2 >= min_d2 for r, c in accepted)
