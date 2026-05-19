"""Feature-to-lattice comparison backend.

GUI-free. Accepts feature points in pixel coordinates and a lattice definition
and returns a FeatureLatticeComparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureLatticeAssignment:
    """Assignment of one feature point to its nearest lattice site."""

    feature_xy_px: tuple[float, float]
    site_ij: tuple[int, int]           # integer (i, j) lattice indices
    site_xy_px: tuple[float, float]    # ideal site position in pixels
    displacement_px: float             # Euclidean distance to ideal site
    matched: bool                      # True if within match_radius_px


@dataclass(frozen=True)
class FeatureLatticeComparison:
    """Summary of comparing a feature-point set to a lattice."""

    n_features: int
    n_matched: int
    n_off_lattice: int
    n_duplicate_sites: int
    rms_displacement_px: float | None
    mean_displacement_px: float | None
    occupancy: float | None
    assignments: tuple[FeatureLatticeAssignment, ...]


def compare_features_to_lattice(
    points_xy_px: np.ndarray,
    lattice_origin_px: tuple[float, float],
    a_px: tuple[float, float],
    b_px: tuple[float, float],
    *,
    match_radius_px: float,
    image_shape: tuple[int, int] | None = None,
) -> FeatureLatticeComparison:
    """Assign each feature point to its nearest lattice site.

    Parameters
    ----------
    points_xy_px
        (N, 2) array of (x, y) positions in pixel coordinates.
    lattice_origin_px
        (ox, oy) origin of the lattice in pixel coordinates.
    a_px, b_px
        Lattice basis vectors in pixel coordinates.
    match_radius_px
        A point is "matched" if its displacement from the nearest site is
        within this radius (pixels).
    image_shape
        (height, width) of the image. Used to count total lattice sites for
        occupancy. Pass ``None`` to skip occupancy calculation.
    """
    pts = np.asarray(points_xy_px, dtype=np.float64)
    if pts.ndim == 1 and pts.shape[0] == 2:
        pts = pts[np.newaxis, :]
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points_xy_px must be an (N, 2) array.")

    ox, oy = float(lattice_origin_px[0]), float(lattice_origin_px[1])
    ax, ay = float(a_px[0]), float(a_px[1])
    bx, by = float(b_px[0]), float(b_px[1])

    # Inversion matrix: [a_px | b_px]^{-1}
    det = ax * by - ay * bx
    if abs(det) < 1e-12:
        raise ValueError(
            "Lattice vectors are (near-)singular: cannot invert basis matrix."
        )
    inv_ax = by / det
    inv_ay = -ay / det
    inv_bx = -bx / det
    inv_by = ax / det

    assignments: list[FeatureLatticeAssignment] = []
    for x, y in pts:
        dx, dy = x - ox, y - oy
        u = inv_ax * dx + inv_bx * dy
        v = inv_ay * dx + inv_by * dy
        i = int(round(u))
        j = int(round(v))
        sx = ox + i * ax + j * bx
        sy = oy + i * ay + j * by
        disp = math.hypot(x - sx, y - sy)
        matched = disp <= match_radius_px
        assignments.append(FeatureLatticeAssignment(
            feature_xy_px=(x, y),
            site_ij=(i, j),
            site_xy_px=(sx, sy),
            displacement_px=disp,
            matched=matched,
        ))

    asn_tuple = tuple(assignments)
    n = len(asn_tuple)
    n_matched = sum(1 for a in asn_tuple if a.matched)
    n_off = n - n_matched

    # Duplicate sites: multiple matched features on the same (i, j).
    from collections import Counter
    site_counts = Counter(a.site_ij for a in asn_tuple if a.matched)
    n_dup = sum(c - 1 for c in site_counts.values() if c > 1)

    matched_disps = [a.displacement_px for a in asn_tuple if a.matched]
    rms_disp = float(math.sqrt(sum(d * d for d in matched_disps) / len(matched_disps))) if matched_disps else None
    mean_disp = float(sum(matched_disps) / len(matched_disps)) if matched_disps else None

    # Occupancy: count sites within image bounds.
    occupancy = None
    if image_shape is not None:
        H, W = image_shape[0], image_shape[1]
        n_sites = _count_sites_in_bounds(ox, oy, ax, ay, bx, by, W, H)
        if n_sites > 0:
            n_occupied = len(site_counts)
            occupancy = float(n_occupied) / float(n_sites)

    return FeatureLatticeComparison(
        n_features=n,
        n_matched=n_matched,
        n_off_lattice=n_off,
        n_duplicate_sites=n_dup,
        rms_displacement_px=rms_disp,
        mean_displacement_px=mean_disp,
        occupancy=occupancy,
        assignments=asn_tuple,
    )


def default_match_radius(
    a_px: tuple[float, float],
    b_px: tuple[float, float],
) -> float:
    """Return 0.35 × min(|a|, |b|)."""
    mag_a = math.hypot(float(a_px[0]), float(a_px[1]))
    mag_b = math.hypot(float(b_px[0]), float(b_px[1]))
    return 0.35 * min(mag_a, mag_b)


def _count_sites_in_bounds(
    ox: float, oy: float,
    ax: float, ay: float,
    bx: float, by: float,
    width: int, height: int,
) -> int:
    """Count integer (i, j) sites whose pixel position is inside [0,W)×[0,H)."""
    # Estimate range of i, j from image corners.
    det = ax * by - ay * bx
    if abs(det) < 1e-12:
        return 0
    inv_ax = by / det;  inv_bx = -bx / det
    inv_ay = -ay / det; inv_by = ax / det

    corners = [(0.0, 0.0), (float(width), 0.0),
               (0.0, float(height)), (float(width), float(height))]
    us, vs = [], []
    for cx, cy in corners:
        dx, dy = cx - ox, cy - oy
        us.append(inv_ax * dx + inv_bx * dy)
        vs.append(inv_ay * dx + inv_by * dy)

    i_lo, i_hi = int(math.floor(min(us))) - 1, int(math.ceil(max(us))) + 1
    j_lo, j_hi = int(math.floor(min(vs))) - 1, int(math.ceil(max(vs))) + 1

    count = 0
    for i in range(i_lo, i_hi + 1):
        for j in range(j_lo, j_hi + 1):
            sx = ox + i * ax + j * bx
            sy = oy + i * ay + j * by
            if 0.0 <= sx < width and 0.0 <= sy < height:
                count += 1
    return count
