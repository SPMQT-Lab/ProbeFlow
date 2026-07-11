"""GUI-free descriptive summary of a point pattern.

Computes the "simple" statistics a user expects before (and independently of)
any model comparison: particle count, analysis-region area, density, and
nearest-neighbour distances for the basic particle-statistics summary; needs
only numpy and scipy.

Region semantics: when a mask is given, points outside it are excluded and the
density is the inside count over the mask area. This never raises for too-few
points — a summary of an empty region is descriptive, not an error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PointPatternSummary:
    """Descriptive statistics for one point collection in one region."""

    n_total: int
    n_in_region: int
    region_label: str
    area_nm2: float | None
    density_per_nm2: float | None
    nn_distances_nm: np.ndarray = field(default_factory=lambda: np.empty(0))
    nn_min_nm: float | None = None
    nn_mean_nm: float | None = None
    nn_median_nm: float | None = None
    nn_max_nm: float | None = None
    # Mean nearest-neighbour distance expected for a completely random
    # (Poisson) pattern of the same density: 1 / (2 sqrt(density)).
    expected_csr_nn_mean_nm: float | None = None
    message: str = ""


def expected_csr_nn_nm(density_per_nm2: float | None) -> float | None:
    """Mean NN distance for a random (Poisson) pattern: ``1 / (2 sqrt(density))``.

    Returns ``None`` when the density is unknown or non-positive.
    """

    if density_per_nm2 is None or density_per_nm2 <= 0.0:
        return None
    return 0.5 / math.sqrt(density_per_nm2)


def summarize_point_pattern(
    points_m: np.ndarray,
    *,
    scan_range_m: tuple[float, float] | None,
    image_shape: tuple[int, int] | None,
    mask: np.ndarray | None = None,
    region_label: str = "Full image",
) -> PointPatternSummary:
    """Summarize a point collection: count, area, density, NN distances.

    ``points_m`` is an ``(N, 2)`` array of x,y positions in metres.
    ``scan_range_m`` is the physical ``(width, height)`` of the image and
    ``image_shape`` its ``(ny, nx)`` pixel dims; both ``None`` means no
    calibration, in which case area and density are unavailable but the
    nearest-neighbour distances (pure inter-point geometry) are still
    computed. ``mask`` is an optional row-major boolean allowed-area array
    matching ``image_shape``; points on disallowed or out-of-bounds pixels
    are excluded from every statistic.

    Synchronous and cheap: nearest neighbours use a k-d tree, O(N log N).
    """

    points = np.asarray(points_m, dtype=float).reshape(-1, 2)
    n_total = int(len(points))
    messages: list[str] = []

    px_x_m = px_y_m = None
    if scan_range_m is not None and image_shape is not None:
        ny, nx = int(image_shape[0]), int(image_shape[1])
        width_m, height_m = float(scan_range_m[0]), float(scan_range_m[1])
        if nx > 0 and ny > 0 and width_m > 0.0 and height_m > 0.0:
            px_x_m = width_m / nx
            px_y_m = height_m / ny

    kept = points
    mask_arr = None
    if mask is not None:
        candidate = np.asarray(mask, dtype=bool)
        if px_x_m is None or px_y_m is None:
            messages.append("Region mask ignored: no scan calibration.")
        elif candidate.shape != (int(image_shape[0]), int(image_shape[1])):
            messages.append("Region mask ignored: shape does not match the image.")
        else:
            mask_arr = candidate
    if mask_arr is not None and n_total:
        cols = np.floor(points[:, 0] / px_x_m).astype(int)
        rows = np.floor(points[:, 1] / px_y_m).astype(int)
        in_bounds = (
            (cols >= 0)
            & (cols < mask_arr.shape[1])
            & (rows >= 0)
            & (rows < mask_arr.shape[0])
        )
        inside = np.zeros(n_total, dtype=bool)
        inside[in_bounds] = mask_arr[rows[in_bounds], cols[in_bounds]]
        kept = points[inside]
    n_in_region = int(len(kept))

    area_nm2 = None
    if px_x_m is not None and px_y_m is not None:
        if mask_arr is not None:
            px_area_nm2 = (px_x_m * 1e9) * (px_y_m * 1e9)
            area_nm2 = float(np.count_nonzero(mask_arr)) * px_area_nm2
        else:
            area_nm2 = float(scan_range_m[0]) * 1e9 * float(scan_range_m[1]) * 1e9
    else:
        messages.append("No scan calibration: area and density unavailable.")

    density_per_nm2 = (
        n_in_region / area_nm2 if area_nm2 is not None and area_nm2 > 0.0 else None
    )
    expected_csr_nn_mean_nm = expected_csr_nn_nm(density_per_nm2)

    nn_distances_nm = np.empty(0)
    nn_min_nm = nn_mean_nm = nn_median_nm = nn_max_nm = None
    if n_in_region >= 2:
        from scipy.spatial import cKDTree

        xy_nm = kept * 1e9
        distances, _ = cKDTree(xy_nm).query(xy_nm, k=2)
        nn_distances_nm = np.asarray(distances[:, 1], dtype=float)
        nn_min_nm = float(nn_distances_nm.min())
        nn_mean_nm = float(nn_distances_nm.mean())
        nn_median_nm = float(np.median(nn_distances_nm))
        nn_max_nm = float(nn_distances_nm.max())

    return PointPatternSummary(
        n_total=n_total,
        n_in_region=n_in_region,
        region_label=str(region_label),
        area_nm2=area_nm2,
        density_per_nm2=density_per_nm2,
        nn_distances_nm=nn_distances_nm,
        nn_min_nm=nn_min_nm,
        nn_mean_nm=nn_mean_nm,
        nn_median_nm=nn_median_nm,
        nn_max_nm=nn_max_nm,
        expected_csr_nn_mean_nm=expected_csr_nn_mean_nm,
        message=" ".join(messages),
    )


def nn_histogram_nm(
    nn_distances_nm: np.ndarray,
    *,
    max_bins: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width histogram ``(bin_edges_nm, counts)`` of NN distances.

    Bin count scales as sqrt(N), clamped to [6, max_bins]. A degenerate
    single-valued distribution (perfect lattice) gets a padded range so at
    least one visible bin exists. Empty input returns two empty arrays.
    """

    distances = np.asarray(nn_distances_nm, dtype=float)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return np.empty(0), np.empty(0)
    n_bins = int(min(max_bins, max(6, math.ceil(math.sqrt(distances.size)))))
    lo = float(distances.min())
    hi = float(distances.max())
    if not hi > lo:
        pad = 0.05 * (abs(hi) if hi != 0.0 else 1.0)
        lo, hi = lo - pad, hi + pad
    counts, edges = np.histogram(distances, bins=n_bins, range=(lo, hi))
    return edges, counts
