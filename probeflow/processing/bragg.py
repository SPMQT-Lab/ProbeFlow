"""Reciprocal-lattice / Bragg-peak analysis for FFT overlays and calibration.

Split out of ``processing/filters.py`` (which re-exports these names for
backward compatibility). Covers Bragg shell enumeration, predicted peak radii,
FFT peak detection in real- and q-space annuli, ellipse fitting from picked
peaks, and piezo-constant correction from observed vs predicted radii.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

__all__ = [
    "BraggShell",
    "bragg_shells",
    "first_bragg_q",
    "predicted_bragg_radius",
    "find_bragg_peaks_in_annulus",
    "find_bragg_peaks_in_q_annulus",
    "snap_to_compact_peak_q",
    "fit_axis_aligned_ellipse",
    "piezo_correction",
]


# ═════════════════════════════════════════════════════════════════════════════
# 17.  Bragg shell helpers  — reciprocal lattice shells for FFT overlays
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BraggShell:
    """Low-index reciprocal-lattice shell family."""

    h: int
    k: int
    factor: float
    multiplicity: int
    label: str


def _shell_norm_sq(symmetry: str, h: int, k: int) -> int:
    if symmetry == "square":
        return h * h + k * k
    if symmetry == "hex":
        return h * h + k * k + h * k
    raise ValueError(f"symmetry must be 'square' or 'hex', got {symmetry!r}")


def bragg_shells(
    symmetry: str,
    *,
    max_shells: int = 5,
    max_factor: float | None = None,
    max_index: int = 12,
) -> list[BraggShell]:
    """Return low-index reciprocal-lattice shell families.

    Shell radii are expressed as a factor of the first-shell radius.  For square
    lattices the factor is ``sqrt(h² + k²)``; for hexagonal/triangular lattices
    it is ``sqrt(h² + k² + hk)``.
    """
    if symmetry not in {"square", "hex"}:
        raise ValueError(f"symmetry must be 'square' or 'hex', got {symmetry!r}")
    if max_shells <= 0:
        return []
    if max_index <= 0:
        raise ValueError("max_index must be > 0")

    groups: dict[int, set[tuple[int, int]]] = {}
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            if h == 0 and k == 0:
                continue
            norm_sq = _shell_norm_sq(symmetry, h, k)
            if norm_sq <= 0:
                continue
            factor = math.sqrt(norm_sq)
            if max_factor is not None and factor > max_factor + 1e-12:
                continue
            groups.setdefault(norm_sq, set()).add((h, k))

    shells: list[BraggShell] = []
    for norm_sq in sorted(groups):
        family = _canonical_shell_family(symmetry, norm_sq, max_index)
        if family is None:
            continue
        h, k = family
        shells.append(
            BraggShell(
                h=h,
                k=k,
                factor=math.sqrt(norm_sq),
                multiplicity=len(groups[norm_sq]),
                label=f"({h}{k})",
            )
        )
        if len(shells) >= max_shells:
            break
    return shells


def _canonical_shell_family(
    symmetry: str,
    norm_sq: int,
    max_index: int,
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for h in range(0, max_index + 1):
        for k in range(0, h + 1):
            if h == 0 and k == 0:
                continue
            if _shell_norm_sq(symmetry, h, k) == norm_sq:
                candidates.append((h, k))
    if not candidates:
        return None
    return min(candidates, key=lambda pair: (pair[0] + pair[1], pair[0], pair[1]))


def first_bragg_q(a_real: float, symmetry: str) -> float:
    """Return the first-shell Bragg radius in reciprocal units of ``a_real``."""
    if a_real <= 0:
        raise ValueError(f"a_real must be > 0, got {a_real!r}")
    if symmetry == "square":
        return 1.0 / a_real
    if symmetry == "hex":
        return 2.0 / (a_real * math.sqrt(3.0))
    raise ValueError(f"symmetry must be 'square' or 'hex', got {symmetry!r}")


# ═════════════════════════════════════════════════════════════════════════════
# 18.  predicted_bragg_radius  — predicted Bragg peak radius for lattice overlay
# ═════════════════════════════════════════════════════════════════════════════

def predicted_bragg_radius(
    a_real: float,
    symmetry: str,
    scan_size_m: float,
    n_pixels: int,
    order: int = 1,
) -> float:
    """Return the predicted Bragg peak radius in FFT pixel units, measured
    from the DC bin at the centre.

    The function is unit-agnostic above metres: ``a_real`` and ``scan_size_m``
    must be in the **same** length unit (typically metres). The GUI layer is
    responsible for converting user-facing units (Å, nm) before calling this.

    Parameters
    ----------
    a_real
        Real-space lattice constant. For a hexagonal lattice this is the
        nearest-neighbour distance. Must be positive.
    symmetry
        ``"square"`` or ``"hex"`` (hexagonal/triangular).
    scan_size_m
        Physical scan size in the same unit as ``a_real``. Assumes a square
        scan; for non-square scans the caller should pass the geometric mean.
    n_pixels
        Image side in pixels (assumes a square image). Present in the
        signature for interface completeness; not used in the current formulas.
    order
        Bragg order: 1 for first-order peaks, 2 for second-order.

    Returns
    -------
    float
        Radius in FFT pixel units from the DC bin at the image centre.

    Raises
    ------
    ValueError
        On any invalid input.

    Notes
    -----
    Square lattice, order 1:  r = scan_size_m / a_real
    Square lattice, order 2:  r = scan_size_m / a_real * sqrt(2)   (diagonal)
    Hex lattice, order 1:     r = 2 * scan_size_m / (a_real * sqrt(3))
    Hex lattice, order 2:     r = 2 * scan_size_m / a_real          (= order_1 * sqrt(3))
    """
    if a_real <= 0:
        raise ValueError(f"a_real must be > 0, got {a_real!r}")
    if symmetry not in {"square", "hex"}:
        raise ValueError(f"symmetry must be 'square' or 'hex', got {symmetry!r}")
    if scan_size_m <= 0:
        raise ValueError(f"scan_size_m must be > 0, got {scan_size_m!r}")
    if n_pixels <= 0:
        raise ValueError(f"n_pixels must be > 0, got {n_pixels!r}")
    if order not in {1, 2}:
        raise ValueError(f"order must be 1 or 2, got {order!r}")
    shells = bragg_shells(symmetry, max_shells=order)
    return first_bragg_q(a_real, symmetry) * scan_size_m * shells[order - 1].factor


# ═════════════════════════════════════════════════════════════════════════════
# 19.  find_bragg_peaks_in_annulus  — auto-detect Bragg peaks in FFT
# ═════════════════════════════════════════════════════════════════════════════

def find_bragg_peaks_in_annulus(
    fft_magnitude: np.ndarray,
    r_predicted_px: float,
    width_frac: float = 0.20,
    expected_count: int = 6,
    min_separation_frac: float = 0.30,
) -> np.ndarray:
    """Find local-maximum peaks inside an annulus around a predicted Bragg radius.

    Parameters
    ----------
    fft_magnitude
        2-D FFT magnitude array with DC at the centre (after fftshift). Not
        log-scaled; the raw ``|F|`` values are expected.
    r_predicted_px
        Predicted Bragg peak radius in pixel units, measured from the array
        centre.
    width_frac
        Half-width of the search annulus as a fraction of ``r_predicted_px``.
        The annulus spans ``r_predicted_px * (1 - width_frac)`` to
        ``r_predicted_px * (1 + width_frac)``.
    expected_count
        How many peaks to try to return.  Typically 4 (square) or 6 (hex).
    min_separation_frac
        Minimum angular separation between kept peaks, expressed as a fraction
        of ``2π / expected_count``.  Prevents returning two picks from the
        same broadened peak.

    Returns
    -------
    np.ndarray, shape (M, 2)
        (x, y) pixel offsets from the array centre, ordered brightest-first.
        ``M ≤ expected_count``.  Returns an empty array of shape ``(0, 2)``
        if no local maxima are found.
    """
    mag = np.asarray(fft_magnitude, dtype=np.float64)
    if mag.ndim != 2:
        raise ValueError("fft_magnitude must be a 2-D array")
    if r_predicted_px <= 0:
        return np.empty((0, 2), dtype=np.float64)

    Ny, Nx = mag.shape
    cy, cx = Ny / 2.0, Nx / 2.0

    yy, xx = np.ogrid[:Ny, :Nx]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    r_in  = max(0.0, r_predicted_px * (1.0 - width_frac))
    r_out = r_predicted_px * (1.0 + width_frac)
    annulus_mask = (r >= r_in) & (r <= r_out)

    # Suppress local noise: footprint ~ half the annulus thickness
    footprint_size = max(3, int(r_predicted_px * width_frac))
    local_max_img = maximum_filter(mag, size=footprint_size)
    # Tolerate float jitter from log1p-then-back, downstream casts, etc.
    # Plateau pixels (two adjacent identical maxima) are all kept; the
    # angular-separation filter below then collapses each plateau to a
    # single pick.  Review numerical #5 (fixed 2026-05-28) — previously
    # ``mag == local_max_img`` could silently miss plateau pixels under
    # subtle precision differences and led to asymmetric peak selection
    # on highly symmetric synthetic test data.
    #
    # The tolerance is scaled per-pixel to ``|local_max_img|`` so that
    # near-zero regions retain effective strict equality (otherwise the
    # noise floor — where mag ≈ local_max ≈ tiny — would all match).
    eps_per_pixel = 1e-10 * np.abs(local_max_img)
    candidate_mask = annulus_mask & (mag >= local_max_img - eps_per_pixel)

    cys, cxs = np.where(candidate_mask)
    if cys.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Sort candidates by magnitude (brightest first)
    vals = mag[cys, cxs]
    order = np.argsort(vals)[::-1]
    cys = cys[order]
    cxs = cxs[order]

    # Angular separation filter
    min_angle = min_separation_frac * (2.0 * math.pi / max(1, expected_count))
    kept_angles: list[float] = []
    results: list[tuple[float, float]] = []

    for iy, ix in zip(cys, cxs):
        angle = math.atan2(float(iy) - cy, float(ix) - cx)
        ok = True
        for prev in kept_angles:
            diff = abs(angle - prev)
            if diff > math.pi:
                diff = 2.0 * math.pi - diff
            if diff < min_angle:
                ok = False
                break
        if ok:
            results.append((float(ix) - cx, float(iy) - cy))
            kept_angles.append(angle)
            if len(results) >= expected_count:
                break

    if not results:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(results, dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# 20.  find_bragg_peaks_in_q_annulus  — q-space Bragg peak detector
# ═════════════════════════════════════════════════════════════════════════════

def find_bragg_peaks_in_q_annulus(
    fft_magnitude: np.ndarray,
    qx_axis: np.ndarray,
    qy_axis: np.ndarray,
    q_predicted: float,
    width_frac: float = 0.20,
    expected_count: int = 6,
    min_separation_frac: float = 0.30,
    suppress_origin_streaks: bool = True,
) -> np.ndarray:
    """Find compact Bragg peaks in q-space around a predicted shell radius.

    Returns ``(qx, qy)`` coordinates in the same reciprocal units as the axes.
    The detector scores ``log1p(|FFT|)`` and can suppress bright origin-crossing
    line artifacts by rejecting angular sectors with high radial occupancy.
    """
    mag = np.asarray(fft_magnitude, dtype=np.float64)
    qx = np.asarray(qx_axis, dtype=np.float64)
    qy = np.asarray(qy_axis, dtype=np.float64)
    if mag.ndim != 2:
        raise ValueError("fft_magnitude must be a 2-D array")
    if qx.ndim != 1 or qy.ndim != 1:
        raise ValueError("qx_axis and qy_axis must be 1-D arrays")
    if mag.shape != (qy.size, qx.size):
        raise ValueError(
            f"fft_magnitude shape {mag.shape} does not match axes "
            f"({qy.size}, {qx.size})"
        )
    if q_predicted <= 0:
        return np.empty((0, 2), dtype=np.float64)

    score = _fft_peak_score(mag)
    qxx, qyy = np.meshgrid(qx, qy)
    q_radius = np.hypot(qxx, qyy)
    r_in = max(0.0, q_predicted * (1.0 - width_frac))
    r_out = q_predicted * (1.0 + width_frac)
    annulus_mask = (q_radius >= r_in) & (q_radius <= r_out) & np.isfinite(score)
    if not np.any(annulus_mask):
        return np.empty((0, 2), dtype=np.float64)

    dq = _typical_q_step(qx, qy)
    footprint_size = max(3, int(round(q_predicted * width_frac / max(dq, 1e-12))))
    if footprint_size % 2 == 0:
        footprint_size += 1
    local_max_img = maximum_filter(score, size=footprint_size)
    # Review numerical #5 (already fixed for find_bragg_peaks_in_annulus
    # in commit f758802; same per-pixel relative tolerance applied here
    # so plateau pixels at the local-maximum value aren't silently
    # missed when ``score`` has been transformed through log1p, etc.).
    eps_per_pixel = 1e-10 * np.abs(local_max_img)
    candidate_mask = annulus_mask & (score >= local_max_img - eps_per_pixel)

    streak_angles: list[float] = []
    if suppress_origin_streaks:
        streak_angles = _origin_streak_angles(score, qxx, qyy, annulus_mask)
        if streak_angles:
            angles_mod = np.mod(np.arctan2(qyy, qxx), math.pi)
            streak_mask = np.zeros_like(candidate_mask, dtype=bool)
            for angle in streak_angles:
                streak_mask |= _angle_distance_mod_pi(angles_mod, angle) <= math.radians(2.5)
            candidate_mask &= ~streak_mask

    cys, cxs = np.where(candidate_mask)
    if cys.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    vals = score[cys, cxs]
    order = np.argsort(vals)[::-1]
    cys = cys[order]
    cxs = cxs[order]

    min_angle = min_separation_frac * (2.0 * math.pi / max(1, expected_count))
    kept_angles: list[float] = []
    results: list[tuple[float, float]] = []
    for iy, ix in zip(cys, cxs):
        qx_val = float(qx[ix])
        qy_val = float(qy[iy])
        angle = math.atan2(qy_val, qx_val)
        if any(_angle_distance(angle, prev) < min_angle for prev in kept_angles):
            continue
        if not _is_compact_peak(score, int(iy), int(ix)):
            continue
        results.append((qx_val, qy_val))
        kept_angles.append(angle)
        if len(results) >= expected_count:
            break

    if not results:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(results, dtype=np.float64)


def snap_to_compact_peak_q(
    fft_magnitude: np.ndarray,
    qx_axis: np.ndarray,
    qy_axis: np.ndarray,
    qx_click: float,
    qy_click: float,
    *,
    search_radius_px: int = 8,
) -> tuple[float, float] | None:
    """Return the nearby compact local maximum closest to a clicked q point."""
    mag = np.asarray(fft_magnitude, dtype=np.float64)
    qx = np.asarray(qx_axis, dtype=np.float64)
    qy = np.asarray(qy_axis, dtype=np.float64)
    if mag.ndim != 2 or mag.shape != (qy.size, qx.size):
        return None
    ix = int(np.argmin(np.abs(qx - qx_click)))
    iy = int(np.argmin(np.abs(qy - qy_click)))
    r = max(1, int(search_radius_px))
    y0, y1 = max(0, iy - r), min(mag.shape[0], iy + r + 1)
    x0, x1 = max(0, ix - r), min(mag.shape[1], ix + r + 1)
    score = _fft_peak_score(mag)
    patch = score[y0:y1, x0:x1]
    if patch.size == 0 or not np.isfinite(patch).any():
        return None
    local_max = maximum_filter(score, size=3)
    mask = np.isfinite(score[y0:y1, x0:x1]) & (patch == local_max[y0:y1, x0:x1])
    cys, cxs = np.where(mask)
    if cys.size == 0:
        return None
    candidates: list[tuple[float, float, int, int]] = []
    for py, px in zip(cys, cxs):
        cy = int(y0 + py)
        cx = int(x0 + px)
        if not _is_compact_peak(score, cy, cx):
            continue
        dist = math.hypot(cx - ix, cy - iy)
        candidates.append((float(score[cy, cx]), -dist, cy, cx))
    if not candidates:
        return None
    _, _, best_y, best_x = max(candidates)
    return (float(qx[best_x]), float(qy[best_y]))


def _fft_peak_score(mag: np.ndarray) -> np.ndarray:
    finite = np.isfinite(mag)
    clipped = np.where(finite, np.maximum(mag, 0.0), 0.0)
    score = np.log1p(clipped)
    score[~finite] = np.nan
    return score


def _typical_q_step(qx: np.ndarray, qy: np.ndarray) -> float:
    steps: list[float] = []
    for axis in (qx, qy):
        if axis.size > 1:
            diffs = np.diff(axis)
            finite = np.abs(diffs[np.isfinite(diffs)])
            if finite.size:
                steps.append(float(np.median(finite)))
    return min(steps) if steps else 1.0


def _origin_streak_angles(
    score: np.ndarray,
    qxx: np.ndarray,
    qyy: np.ndarray,
    annulus_mask: np.ndarray,
) -> list[float]:
    vals = score[annulus_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size < 20:
        return []
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    threshold = max(med + 4.0 * max(mad, 1e-12), float(np.percentile(vals, 90)))
    bright = annulus_mask & (score >= threshold)
    angles = np.mod(np.arctan2(qyy, qxx), math.pi)
    bins = 180
    total, edges = np.histogram(angles[annulus_mask], bins=bins, range=(0.0, math.pi))
    high, _ = np.histogram(angles[bright], bins=edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        occupancy = np.where(total > 0, high / total, 0.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    q_radius = np.hypot(qxx, qyy)
    annulus_r = q_radius[annulus_mask]
    annulus_width = max(1e-12, float(annulus_r.max() - annulus_r.min()))
    streak_angles: list[float] = []
    for i, centre in enumerate(centres):
        if total[i] < 6 or high[i] < 4 or occupancy[i] < 0.35:
            continue
        in_bin = bright & (angles >= edges[i]) & (angles < edges[i + 1])
        radii = q_radius[in_bin]
        if radii.size < 4:
            continue
        radial_span = float(radii.max() - radii.min()) / annulus_width
        # Compact Bragg blobs are bright in one radial patch; line artifacts
        # remain bright over much of the annulus along one origin-crossing line.
        if radial_span >= 0.85:
            streak_angles.append(float(centre))
    return streak_angles


def _angle_distance(a: float, b: float) -> float:
    diff = abs(a - b)
    if diff > math.pi:
        diff = 2.0 * math.pi - diff
    return diff


def _angle_distance_mod_pi(angles: np.ndarray, target: float) -> np.ndarray:
    diff = np.abs(angles - target)
    return np.minimum(diff, math.pi - diff)


def _is_compact_peak(score: np.ndarray, iy: int, ix: int, radius: int = 3) -> bool:
    y0, y1 = max(0, iy - radius), min(score.shape[0], iy + radius + 1)
    x0, x1 = max(0, ix - radius), min(score.shape[1], ix + radius + 1)
    patch = score[y0:y1, x0:x1]
    if patch.size < 4 or not np.isfinite(patch).any():
        return False
    patch = np.where(np.isfinite(patch), patch, np.nanmin(patch[np.isfinite(patch)]))
    weights = patch - float(np.nanmin(patch))
    if float(weights.sum()) <= 1e-12:
        return False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xbar = float((weights * xx).sum() / weights.sum())
    ybar = float((weights * yy).sum() / weights.sum())
    dx = xx - xbar
    dy = yy - ybar
    cov_xx = float((weights * dx * dx).sum() / weights.sum())
    cov_yy = float((weights * dy * dy).sum() / weights.sum())
    cov_xy = float((weights * dx * dy).sum() / weights.sum())
    trace = cov_xx + cov_yy
    det = max(0.0, cov_xx * cov_yy - cov_xy * cov_xy)
    disc = max(0.0, trace * trace - 4.0 * det)
    lam_hi = 0.5 * (trace + math.sqrt(disc))
    lam_lo = 0.5 * (trace - math.sqrt(disc))
    if lam_lo <= 1e-12:
        return False
    return (lam_hi / lam_lo) <= 8.0


# ═════════════════════════════════════════════════════════════════════════════
# 21.  fit_axis_aligned_ellipse  — least-squares ellipse from Bragg picks
# ═════════════════════════════════════════════════════════════════════════════

def fit_axis_aligned_ellipse(
    points: np.ndarray,
) -> tuple[float, float, float]:
    """Fit an axis-aligned ellipse (x/r_x)² + (y/r_y)² = 1 to the given points.

    Uses linear least squares on the substituted variables ``u = 1/r_x²`` and
    ``v = 1/r_y²``, so the design matrix is ``[x², y²]`` and the right-hand
    side is the constant vector ``1``.

    Parameters
    ----------
    points
        Array of shape ``(M, 2)`` containing ``(x, y)`` coordinates measured
        from the ellipse centre.  Must contain at least three points.

    Returns
    -------
    r_x : float
        Fitted semi-axis along x, in the same units as the input coordinates.
    r_y : float
        Fitted semi-axis along y.
    rms_residual : float
        RMS distance (in input units) between each point and its closest point
        on the fitted ellipse, computed radially along the direction of each
        point from the origin.

    Raises
    ------
    ValueError
        If fewer than three points are supplied, or if the fit is degenerate
        (one of the estimated ``1/r²`` values is ≤ 0).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"points must have shape (M, 2), got {pts.shape}"
        )
    M = pts.shape[0]
    if M < 3:
        raise ValueError(f"Need at least 3 points to fit an ellipse, got {M}")

    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([x ** 2, y ** 2])
    b = np.ones(M, dtype=np.float64)
    (u, v), *_ = np.linalg.lstsq(A, b, rcond=None)

    if u <= 0:
        raise ValueError(
            f"Degenerate fit: 1/r_x² = {u:.6g} ≤ 0; "
            "check that picks are not all on the y-axis"
        )
    if v <= 0:
        raise ValueError(
            f"Degenerate fit: 1/r_y² = {v:.6g} ≤ 0; "
            "check that picks are not all on the x-axis"
        )

    r_x = 1.0 / math.sqrt(u)
    r_y = 1.0 / math.sqrt(v)

    # RMS residual: radial distance between each observed point and the
    # ellipse evaluated at the same angle.
    angles = np.arctan2(y, x)
    r_fit = 1.0 / np.sqrt(u * np.cos(angles) ** 2 + v * np.sin(angles) ** 2)
    r_obs = np.hypot(x, y)
    rms = float(np.sqrt(np.mean((r_obs - r_fit) ** 2)))

    return r_x, r_y, rms


# ═════════════════════════════════════════════════════════════════════════════
# 20.  piezo_correction  — convert observed/predicted radii to new piezo values
# ═════════════════════════════════════════════════════════════════════════════

def piezo_correction(
    r_x_obs: float,
    r_y_obs: float,
    r_predicted: float,
    c_x_current: float,
    c_y_current: float,
) -> tuple[float, float]:
    """Compute corrected piezo constants from observed Bragg semi-axes.

    The function is unit-agnostic: ``r_x_obs``, ``r_y_obs``, and
    ``r_predicted`` must be in the same unit (e.g., all in nm⁻¹ or all in
    FFT pixel offsets); ``c_x_current`` and ``c_y_current`` must be in the
    same unit as each other (e.g., both in Å/V).

    Parameters
    ----------
    r_x_obs
        Observed Bragg semi-axis along x from the ellipse fit.
    r_y_obs
        Observed Bragg semi-axis along y.
    r_predicted
        Predicted isotropic Bragg radius (same unit as the observed axes).
    c_x_current
        Current piezo constant for the x axis.
    c_y_current
        Current piezo constant for the y axis.

    Returns
    -------
    c_x_new, c_y_new : float, float
        Recommended corrected piezo constants.

    Raises
    ------
    ValueError
        If any input is not strictly positive.
    """
    for name, val in [
        ("r_x_obs",     r_x_obs),
        ("r_y_obs",     r_y_obs),
        ("r_predicted", r_predicted),
        ("c_x_current", c_x_current),
        ("c_y_current", c_y_current),
    ]:
        if not (val > 0):
            raise ValueError(f"{name} must be > 0, got {val!r}")

    return c_x_current * (r_x_obs / r_predicted), c_y_current * (r_y_obs / r_predicted)
