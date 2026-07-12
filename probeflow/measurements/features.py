"""Local feature detection for processed SPM images."""

from __future__ import annotations

import math
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
    for name, value in (
        ("pixel_size_x", pixel_size_x),
        ("pixel_size_y", pixel_size_y),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a positive finite value")
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
    candidates = selected & (work >= threshold)
    accepted = _detect_peaks_nms(
        work, candidates, min_distance_px,
        min_prominence=min_prominence,
        max_peaks=max_peaks,
    )

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
    """Thin wrapper around the canonical
    :func:`probeflow.measurements.roi_resolve.resolve_roi_to_mask`
    (review arch-backend #11).  ``roi_mask`` keyword is preserved
    here for backward compatibility with the local call sites."""
    from probeflow.measurements.roi_resolve import resolve_roi_to_mask
    return resolve_roi_to_mask(shape, roi=roi, mask=roi_mask)


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


def _far_enough(row: int, col: int, accepted: list[tuple[int, int]], min_distance: float) -> bool:
    if not accepted:
        return True
    min_d2 = float(min_distance) ** 2
    return all((row - r) ** 2 + (col - c) ** 2 >= min_d2 for r, c in accepted)


def _detect_peaks_nms(
    detect_arr: np.ndarray,
    candidates_mask: np.ndarray,
    min_distance_px: float,
    *,
    score_arr: np.ndarray | None = None,
    min_prominence: float | None = None,
    max_peaks: int | None = None,
) -> list[tuple[int, int]]:
    """Canonical max-filter + NMS peak detector shared by all detection paths.

    Both :func:`detect_local_maxima` and
    :func:`probeflow.analysis.feature_finder.find_image_features` delegate
    here so NMS behaviour stays in sync (review arch-backend #3).

    Parameters
    ----------
    detect_arr
        Array to run max-filter on.  Negate before calling to find minima.
    candidates_mask
        Boolean mask of pre-filtered candidates (ROI + threshold).
    min_distance_px
        NMS exclusion radius in pixels.
    score_arr
        Array used for ranking; defaults to ``detect_arr``.  Candidates are
        sorted descending so the strongest peak survives NMS first.
    min_prominence
        If set, accept a candidate only when its value exceeds the local
        median (same window as NMS) by at least this amount.
    max_peaks
        Stop after this many accepted peaks.
    """
    from scipy.ndimage import maximum_filter, median_filter, minimum_filter

    radius = int(max(1, min_distance_px))
    size = 2 * radius + 1
    local_max = detect_arr == maximum_filter(detect_arr, size=size, mode="nearest")
    local_min = minimum_filter(detect_arr, size=size, mode="nearest")
    has_contrast = detect_arr > local_min
    cands = candidates_mask & local_max & has_contrast
    if min_prominence is not None:
        local_median = median_filter(detect_arr, size=size, mode="nearest")
        cands &= (detect_arr - local_median) >= float(min_prominence)

    rows, cols = np.nonzero(cands)
    if rows.size == 0:
        return []

    rank_arr = detect_arr if score_arr is None else score_arr
    order = np.argsort(rank_arr[rows, cols])[::-1]
    min_d = float(min_distance_px)
    accepted: list[tuple[int, int]] = []
    for idx in order:
        r, c = int(rows[idx]), int(cols[idx])
        if _far_enough(r, c, accepted, min_d):
            accepted.append((r, c))
            if max_peaks is not None and len(accepted) >= int(max_peaks):
                break
    return accepted
