"""Pair-correlation (radial distribution) from detected feature points.

GUI-free. Accepts physical-space (x, y) coordinates in metres and returns a
PairCorrelationResult. Edge correction is *not* applied; a note is included
in the result message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairCorrelationResult:
    """Radial pair-correlation statistics."""

    n_points: int
    density_m2: float | None
    r_m: np.ndarray          # bin centres (metres)
    g_r: np.ndarray          # pair correlation (dimensionless; 1 = random)
    nearest_neighbour_median_m: float | None
    first_peak_m: float | None
    quality: str             # "good" | "weak" | "failed"
    message: str


def compute_pair_correlation(
    points_xy_m: np.ndarray,
    *,
    roi_area_m2: float | None = None,
    r_max_m: float | None = None,
    bin_width_m: float | None = None,
) -> PairCorrelationResult:
    """Compute g(r) from a set of 2-D physical coordinates.

    Parameters
    ----------
    points_xy_m
        (N, 2) array of (x, y) positions in metres.
    roi_area_m2
        Area of the measurement region in m². Used for density normalisation.
        If ``None`` the histogram is shown unnormalised (raw pair counts).
    r_max_m
        Maximum radius in metres. Defaults to ``0.5 * sqrt(roi_area_m2)``
        when the area is known, otherwise ``5 × NN_median``.
    bin_width_m
        Bin width in metres. Defaults to ``r_max_m / 50``.
    """
    pts = np.asarray(points_xy_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy_m must be an (N, 2) array.")

    n = len(pts)
    if n < 3:
        return PairCorrelationResult(
            n_points=n,
            density_m2=None,
            r_m=np.array([]),
            g_r=np.array([]),
            nearest_neighbour_median_m=None,
            first_peak_m=None,
            quality="failed",
            message=f"Too few points ({n}); need at least 3.",
        )

    # All unique pair distances via broadcasting.
    diff = pts[:, None, :] - pts[None, :, :]          # (N, N, 2)
    dist_mat = np.sqrt((diff ** 2).sum(axis=2))        # (N, N)
    np.fill_diagonal(dist_mat, np.nan)

    # Nearest-neighbour distance for each point.
    nn_dists = np.nanmin(dist_mat, axis=1)
    nn_median = float(np.median(nn_dists))

    # Upper triangular → unique pairs.
    i_upper, j_upper = np.triu_indices(n, k=1)
    all_dists = dist_mat[i_upper, j_upper]

    # Resolve r_max and bin_width.
    if r_max_m is None:
        if roi_area_m2 is not None and roi_area_m2 > 0:
            r_max_m = 0.5 * math.sqrt(float(roi_area_m2))
        else:
            r_max_m = nn_median * 5.0
    r_max_m = float(r_max_m)

    if bin_width_m is None:
        bin_width_m = r_max_m / 50.0
    bin_width_m = max(float(bin_width_m), r_max_m / 1000.0)

    edges = np.arange(0.0, r_max_m + bin_width_m, bin_width_m)
    if len(edges) < 2:
        edges = np.array([0.0, r_max_m])
    counts, _ = np.histogram(all_dists, bins=edges)
    r_centres = 0.5 * (edges[:-1] + edges[1:])

    # Normalise to g(r) when area is known.
    density = float(n) / float(roi_area_m2) if (roi_area_m2 and roi_area_m2 > 0) else None
    n_pairs = n * (n - 1)           # = 2 × unique pairs (pdist convention)

    if roi_area_m2 and roi_area_m2 > 0:
        # g(r) = count * 2*A / (n*(n-1) * 2π*r*dr)
        annulus = 2.0 * math.pi * r_centres * bin_width_m
        annulus = np.where(annulus > 0, annulus, np.nan)
        normalised = (counts * 2.0 * float(roi_area_m2)) / (float(n_pairs) * annulus)
        g_r = np.where(np.isfinite(normalised), normalised, 0.0)
    else:
        # Raw pair counts.
        g_r = counts.astype(float)

    # First peak: highest bin above the NN scale.
    peak_mask = (r_centres > nn_median * 0.3) & (counts > 0)
    if np.any(peak_mask):
        first_peak = float(r_centres[peak_mask][np.argmax(g_r[peak_mask])])
    else:
        first_peak = None

    quality = "good" if n >= 20 else ("weak" if n >= 5 else "failed")
    msg = "Pair correlation is approximate; edge correction is not yet applied."
    if quality == "weak":
        msg = f"Only {n} points — result may be unreliable. " + msg

    return PairCorrelationResult(
        n_points=n,
        density_m2=density,
        r_m=r_centres,
        g_r=g_r,
        nearest_neighbour_median_m=nn_median,
        first_peak_m=first_peak,
        quality=quality,
        message=msg,
    )
