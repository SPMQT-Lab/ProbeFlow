"""ProbeFlow — local feature finder and selective FFT backend.

Find local maxima or minima with flexible threshold modes, generate feature
images from detected points, and support selective FFT analysis workflows.

No Qt imports. All functions operate on NumPy arrays and return plain dataclasses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Review arch-backend #2 (2026-05-28): the previous local
# ``FeaturePoint`` dataclass (x_px, y_px, value, label) collided with
# ``probeflow.measurements.models.FeaturePoint``.  Both detection
# pipelines now share the canonical model — ``find_image_features``
# below populates the intrinsic fields (x_px, y_px, z_value) and
# leaves the context fields (point_id, x_phys, y_phys, channel,
# source_label, roi_id) at their defaults.  The legacy ``label``
# field was never populated in production and is dropped.
from probeflow.measurements.features import _detect_peaks_nms
from probeflow.measurements.models import FeaturePoint


@dataclass(frozen=True)
class FeatureDetectionResult:
    """Result from find_image_features."""
    points: tuple[FeaturePoint, ...]
    mode: str               # "maxima" | "minima"
    threshold_mode: str     # "above" | "below" | "between"
    threshold_low: float | None
    threshold_high: float | None
    min_distance_px: float
    smoothing_sigma_px: float
    message: str


def find_image_features(
    image: np.ndarray,
    *,
    mode: str = "maxima",
    threshold_mode: str = "above",
    threshold_low: float | None = None,
    threshold_high: float | None = None,
    min_distance_px: float = 3.0,
    smoothing_sigma_px: float = 0.0,
    roi_mask: np.ndarray | None = None,
) -> FeatureDetectionResult:
    """Detect local maxima or minima in an STM image.

    Parameters
    ----------
    image
        2-D float array.
    mode
        ``"maxima"`` finds bright peaks; ``"minima"`` finds dark valleys.
    threshold_mode
        ``"above"`` keeps features with value >= threshold_low.
        ``"below"`` keeps features with value <= threshold_high.
        ``"between"`` keeps features with threshold_low <= value <= threshold_high.
    threshold_low
        Lower threshold. Required for ``"above"`` and ``"between"``.
        Defaults to ``-inf`` if not provided for ``"above"``.
    threshold_high
        Upper threshold. Required for ``"below"`` and ``"between"``.
        Defaults to ``+inf`` if not provided for ``"below"``.
    min_distance_px
        Minimum pixel separation between accepted features.
    smoothing_sigma_px
        Optional Gaussian pre-smoothing sigma (applied to a working copy only;
        reported values are from the original image).
    roi_mask
        Optional boolean array of same shape as image. Only pixels where the
        mask is True are considered as feature candidates.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("image must be 2-D")
    if mode not in {"maxima", "minima"}:
        raise ValueError(f"mode must be 'maxima' or 'minima', got {mode!r}")
    if threshold_mode not in {"above", "below", "between"}:
        raise ValueError(
            f"threshold_mode must be 'above', 'below', or 'between', got {threshold_mode!r}"
        )
    if threshold_mode == "between" and (threshold_low is None or threshold_high is None):
        raise ValueError("threshold_low and threshold_high are both required for threshold_mode='between'")

    _common = dict(
        mode=mode,
        threshold_mode=threshold_mode,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        min_distance_px=float(min_distance_px),
        smoothing_sigma_px=float(smoothing_sigma_px),
    )

    finite = np.isfinite(arr)
    if not np.any(finite):
        return FeatureDetectionResult(points=(), message="Image contains no finite values.", **_common)

    # Working copy for peak detection (smoothed if requested).
    work = arr.copy()
    fill = float(np.nanmedian(arr[finite]))
    work[~finite] = fill

    sigma = float(smoothing_sigma_px)
    if sigma > 0.0:
        from scipy.ndimage import gaussian_filter
        work = gaussian_filter(work, sigma=sigma)

    # Candidate mask: ROI + finite + threshold.
    if roi_mask is not None:
        selected = np.asarray(roi_mask, dtype=bool)
        if selected.shape != arr.shape:
            raise ValueError("roi_mask shape must match image shape")
        selected = selected & finite
    else:
        selected = finite.copy()

    tlo = float(threshold_low) if threshold_low is not None else -np.inf
    thi = float(threshold_high) if threshold_high is not None else np.inf

    if threshold_mode == "above":
        selected &= work >= tlo
    elif threshold_mode == "below":
        selected &= work <= thi
    else:  # "between"
        selected &= (work >= tlo) & (work <= thi)

    if not np.any(selected):
        return FeatureDetectionResult(points=(), message="No candidates after thresholding.", **_common)

    # For minima, negate so local minima become local maxima in the kernel.
    detect_arr = work if mode == "maxima" else -work

    # Delegate NMS to the canonical kernel shared with detect_local_maxima
    # (review arch-backend #3 — prevents NMS divergence between detection paths).
    accepted = _detect_peaks_nms(detect_arr, selected, min_distance_px)
    if not accepted:
        return FeatureDetectionResult(points=(), message="No peaks detected.", **_common)

    points = tuple(
        FeaturePoint(x_px=float(c), y_px=float(r), z_value=float(arr[r, c]))
        for r, c in accepted
    )
    return FeatureDetectionResult(
        points=points, message=f"Detected {len(points)} features.", **_common
    )


def feature_points_to_image(
    points: Sequence[FeaturePoint],
    shape: tuple[int, int],
    *,
    radius_px: float = 2.0,
    smoothing_sigma_px: float = 0.0,
    value: float = 1.0,
) -> np.ndarray:
    """Create a 2-D float image from a sequence of feature points.

    Parameters
    ----------
    points
        Detected feature points.
    shape
        Output array shape (ny, nx).
    radius_px
        Disk radius in pixels. Use 0 for single-pixel points.
    smoothing_sigma_px
        Optional Gaussian smoothing applied after dilation.
    value
        Pixel value written inside each feature disk.

    The inner disk-rasterization loop is shared with
    :func:`probeflow.measurements.fft_points.points_to_mask` via
    :func:`probeflow.measurements.raster.paint_point` (review
    arch-backend #10).
    """
    from probeflow.measurements.raster import paint_point

    ny, nx = int(shape[0]), int(shape[1])
    out = np.zeros((ny, nx), dtype=np.float64)
    v = float(value)
    r = float(radius_px)

    for pt in points:
        paint_point(
            out, pt.x_px, pt.y_px,
            radius_px=r, shape_mode="disk", value=v,
        )

    sigma = float(smoothing_sigma_px)
    if sigma > 0.0:
        from scipy.ndimage import gaussian_filter
        out = gaussian_filter(out, sigma=sigma)

    return out


def feature_points_to_csv(
    points: Sequence[FeaturePoint],
    *,
    pixel_size_x_nm: float = 1.0,
    pixel_size_y_nm: float = 1.0,
) -> str:
    """Return a CSV string for the detected points.

    Columns: index, x_px, y_px, x_nm, y_nm, value

    ``pixel_size_x_nm`` / ``pixel_size_y_nm`` must be finite and positive — a
    zero/NaN calibration would silently emit meaningless nm columns, so it is
    rejected rather than written out.
    """
    import csv
    import io

    for name, value in (
        ("pixel_size_x_nm", pixel_size_x_nm),
        ("pixel_size_y_nm", pixel_size_y_nm),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["index", "x_px", "y_px", "x_nm", "y_nm", "value"])
    for i, pt in enumerate(points):
        writer.writerow([
            i,
            f"{pt.x_px:.4f}",
            f"{pt.y_px:.4f}",
            f"{pt.x_px * pixel_size_x_nm:.6f}",
            f"{pt.y_px * pixel_size_y_nm:.6f}",
            f"{pt.z_value:.10g}",
        ])
    return out.getvalue()
