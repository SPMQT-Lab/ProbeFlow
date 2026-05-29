"""STM scan-line and polynomial background subtraction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

from ._image_utils import (
    _finite_mean,
    _finite_median,
    _nonnegative_finite,
    _positive_finite,
)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  STM scan-line background subtraction
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class STMBackgroundParams:
    """Parameters for ImageJ-style STM scan-line background subtraction."""

    fit_region: str = "whole_image"
    line_statistic: str = "median"
    model: str = "linear"
    linear_x_first: bool = False
    blur_length: float | None = None
    jump_threshold: float | None = None
    preserve_level: str = "median"


@dataclass(frozen=True)
class STMBackgroundResult:
    """Preview/apply result for STM scan-line background subtraction."""

    corrected: np.ndarray
    background_image: np.ndarray
    line_profile: np.ndarray
    fitted_profile: np.ndarray
    params: STMBackgroundParams
    fit_status: str


def _normalise_stm_background_model(model: str) -> str:
    key = str(model or "linear").lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "linear": "linear",
        "poly1": "linear",
        "polynomial_1": "linear",
        "2nd_order_polynomial": "poly2",
        "second_order_polynomial": "poly2",
        "poly2": "poly2",
        "quadratic": "poly2",
        "3rd_order_polynomial": "poly3",
        "third_order_polynomial": "poly3",
        "poly3": "poly3",
        "cubic": "poly3",
        "low_pass": "low_pass",
        "lowpass": "low_pass",
        "line_by_line": "line_by_line",
        "line": "line_by_line",
        "piezo_creep": "piezo_creep",
        "piezo_creep_x2": "piezo_creep_x2",
        "piezo_creep_x3": "piezo_creep_x3",
        "sqrt_creep": "sqrt_creep",
    }
    if key not in aliases:
        raise ValueError(
            "STM background model must be linear, poly2, poly3, low_pass, "
            "line_by_line, piezo_creep, piezo_creep_x2, piezo_creep_x3, "
            f"or sqrt_creep, got {model!r}"
        )
    return aliases[key]


def _normalise_line_statistic(statistic: str) -> str:
    key = str(statistic or "median").lower()
    if key not in {"median", "mean"}:
        raise ValueError(f"line_statistic must be 'median' or 'mean', got {statistic!r}")
    return key


def _coerce_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.asarray(mask, dtype=bool)
    if m.shape != shape:
        raise ValueError(f"mask shape {m.shape!r} does not match image shape {shape!r}")
    return m


def compute_scanline_profile(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    statistic: str = "median",
) -> np.ndarray:
    """Return one representative background value per fast-scan row.

    ``mask`` controls where the background is estimated.  Rows with no selected
    finite pixels return ``NaN`` so the fitting stage can interpolate or
    extrapolate from rows that do contain fit data.
    """
    stat = _normalise_line_statistic(statistic)
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("image must be a 2-D array")
    m = _coerce_mask(mask, a.shape)
    profile = np.full(a.shape[0], np.nan, dtype=np.float64)
    for row in range(a.shape[0]):
        values = a[row] if m is None else a[row, m[row]]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        profile[row] = float(np.mean(values) if stat == "mean" else np.median(values))
    return profile


def _interp_nan_profile(profile: np.ndarray) -> np.ndarray:
    p = np.asarray(profile, dtype=np.float64)
    x = np.arange(p.size, dtype=np.float64)
    finite = np.isfinite(p)
    if not finite.any():
        raise ValueError("scan-line profile has no finite values to fit")
    if finite.all():
        return p.copy()
    return np.interp(x, x[finite], p[finite])


def _smooth_profile_ignore_nan(profile: np.ndarray, sigma: float) -> np.ndarray:
    sigma = _positive_finite(sigma, "blur_length")
    p = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(p)
    if not finite.any():
        raise ValueError("scan-line profile has no finite values to smooth")
    values = np.where(finite, p, 0.0)
    weights = finite.astype(np.float64)
    smooth_values = gaussian_filter(values, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = smooth_values / smooth_weights
    if not np.isfinite(smoothed).all():
        smoothed = _interp_nan_profile(smoothed)
    return smoothed


def _eliminate_profile_jumps_directional(
    profile: np.ndarray, threshold: float, *, reverse: bool
) -> np.ndarray:
    """One pass of jump elimination, scanning forward or backward."""
    p = np.asarray(profile, dtype=np.float64).copy()
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size < 2:
        return p
    if reverse:
        finite_idx = finite_idx[::-1]
    offset = 0.0
    prev_value = float(p[int(finite_idx[0])])
    for idx in finite_idx[1:]:
        idx = int(idx)
        value = float(p[idx]) + offset
        jump = value - prev_value
        if abs(jump) > threshold:
            offset -= jump
            value = float(p[idx]) + offset
        p[idx] = value
        prev_value = value
    return p


def _eliminate_profile_jumps(profile: np.ndarray, threshold: float | None) -> np.ndarray:
    """Smooth large discontinuities in a 1-D profile.

    Review numerical #8 (fixed 2026-05-28): the previous implementation
    scanned forward once.  A single jump near the start of the profile
    permanently shifted every subsequent value; symmetric noise spikes
    were therefore not handled symmetrically and a transient at index
    5 of a 100-element profile silently re-baselined indices 5..99.

    Now run a forward pass and a backward pass and average them.  The
    result is order-invariant: a noise spike near either end of the
    profile contributes only half its bias to the rest of the profile,
    while a genuine multi-terrace step (which both passes "see") is
    still corrected near-fully.
    """
    if threshold is None:
        return np.asarray(profile, dtype=np.float64).copy()
    threshold = _positive_finite(threshold, "jump_threshold")
    forward = _eliminate_profile_jumps_directional(profile, threshold, reverse=False)
    backward = _eliminate_profile_jumps_directional(profile, threshold, reverse=True)
    # Average the two passes; preserve NaN where the input was NaN.
    f_fin = np.isfinite(forward)
    b_fin = np.isfinite(backward)
    out = np.full_like(forward, np.nan)
    both = f_fin & b_fin
    out[both] = 0.5 * (forward[both] + backward[both])
    only_f = f_fin & ~b_fin
    out[only_f] = forward[only_f]
    only_b = b_fin & ~f_fin
    out[only_b] = backward[only_b]
    return out


def _fit_scanline_background(
    profile: np.ndarray,
    model: str = "linear",
    *,
    blur_length: float | None = None,
    jump_threshold: float | None = None,
) -> np.ndarray:
    """Fit or smooth a 1-D scan-line background profile.

    Models
    ------
    linear
        B(y) = a + b·y  — least-squares line through the row profile.
    poly2
        B(y) = a + b·y + c·y²  — least-squares quadratic.
    poly3
        B(y) = a + b·y + c·y² + d·y³  — least-squares cubic.
    low_pass
        B(y) = Gaussian-smoothed profile (sigma = blur_length px).
    line_by_line
        B(y) = raw per-row median/mean; each row zeroed independently.
    piezo_creep
        B(y) = a + b·y + c·log(|y − d|)  — models logarithmic creep.
        d is the fitted singularity anchor (scan-start offset).
    piezo_creep_x2
        B(y) = a + b·y + c·log(|y − d|) + e·y²
    piezo_creep_x3
        B(y) = a + b·y + c·log(|y − d|) + e·y³
    sqrt_creep
        B(y) = a + b·y + c·√|y − d|  — square-root creep variant.

    y is normalised to [−1, 1] across the scan height for all parametric
    models.  Nonlinear models (piezo_creep, sqrt_creep) are fitted with
    ``scipy.optimize.curve_fit``; d is constrained to (−3, −0.01) to keep
    the singularity off-scan.
    """
    model = _normalise_stm_background_model(model)
    working = _eliminate_profile_jumps(profile, jump_threshold)
    if model == "line_by_line":
        return _interp_nan_profile(working)
    if model == "low_pass":
        sigma = 5.0 if blur_length is None else float(blur_length)
        return _smooth_profile_ignore_nan(working, sigma=sigma)

    if model in {"piezo_creep", "piezo_creep_x2", "piezo_creep_x3", "sqrt_creep"}:
        from scipy.optimize import curve_fit, OptimizeWarning
        p = np.asarray(working, dtype=np.float64)
        y = np.linspace(-1.0, 1.0, p.size, dtype=np.float64)
        finite = np.isfinite(p)
        if int(finite.sum()) < 4:
            raise ValueError(
                f"not enough finite scan-line values for {model} background fit"
            )
        # Center the profile before fitting so the offset 'a' starts near zero,
        # and compute an explicit x_scale so TRF steps are sized appropriately.
        #
        # Real STM height data: mean ~1e-7 m, creep variation ~1e-9 m.
        # Without centering, the large 'a' offset dominates the residual before
        # b and c depart from zero (trivial flat solution).  Without x_scale,
        # the 10^9 ratio between d (O(1) normalised coordinate) and b,c (O(1e-9)
        # SI units) causes TRF's initial trust radius to overshoot b and c by
        # orders of magnitude, preventing convergence.  Both fixes are required.
        p_mean = float(np.nanmean(p[finite]))
        p_c = p - p_mean
        # Characteristic scale of the variation in the centered profile.
        p_scale = float(np.nanstd(p_c[finite]))
        if p_scale == 0.0:
            p_scale = 1.0  # degenerate flat profile — fallback
        eps = 1e-6
        if model == "piezo_creep":
            def _model(y, a, b, c, d):
                return a + b * y + c * np.log(np.abs(y - d) + eps)
            p0 = [0.0, 0.0, 0.0, -1.5]
            bounds = ([-np.inf, -np.inf, -np.inf, -3.0], [np.inf, np.inf, np.inf, -0.01])
            x_scale = [p_scale, p_scale, p_scale, 1.0]
        elif model == "piezo_creep_x2":
            def _model(y, a, b, c, d, e):
                return a + b * y + c * np.log(np.abs(y - d) + eps) + e * y ** 2
            p0 = [0.0, 0.0, 0.0, -1.5, 0.0]
            bounds = ([-np.inf, -np.inf, -np.inf, -3.0, -np.inf], [np.inf, np.inf, np.inf, -0.01, np.inf])
            x_scale = [p_scale, p_scale, p_scale, 1.0, p_scale]
        elif model == "piezo_creep_x3":
            def _model(y, a, b, c, d, e):
                return a + b * y + c * np.log(np.abs(y - d) + eps) + e * y ** 3
            p0 = [0.0, 0.0, 0.0, -1.5, 0.0]
            bounds = ([-np.inf, -np.inf, -np.inf, -3.0, -np.inf], [np.inf, np.inf, np.inf, -0.01, np.inf])
            x_scale = [p_scale, p_scale, p_scale, 1.0, p_scale]
        else:  # sqrt_creep
            def _model(y, a, b, c, d):
                return a + b * y + c * np.sqrt(np.abs(y - d))
            p0 = [0.0, 0.0, 0.0, -1.5]
            bounds = ([-np.inf, -np.inf, -np.inf, -3.0], [np.inf, np.inf, np.inf, -0.01])
            x_scale = [p_scale, p_scale, p_scale, 1.0]
        try:
            popt, _ = curve_fit(
                _model, y[finite], p_c[finite],
                p0=p0, bounds=bounds, x_scale=x_scale, maxfev=5000,
            )
        except (RuntimeError, OptimizeWarning) as exc:
            raise ValueError(f"{model} background fit did not converge: {exc}") from exc
        return (_model(y, *popt) + p_mean).astype(np.float64, copy=False)

    p = np.asarray(working, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, p.size, dtype=np.float64)
    finite = np.isfinite(p)
    degree = {"linear": 1, "poly2": 2, "poly3": 3}[model]
    if int(finite.sum()) < degree + 1:
        raise ValueError(
            f"not enough finite scan-line values for {model} background fit"
        )
    coeff = np.polyfit(x[finite], p[finite], degree)
    return np.polyval(coeff, x).astype(np.float64, copy=False)


def _linear_x_background(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    a = np.asarray(image, dtype=np.float64)
    m = _coerce_mask(mask, a.shape)
    Ny, Nx = a.shape
    x = np.linspace(-1.0, 1.0, Nx, dtype=np.float64)
    bg = np.zeros_like(a, dtype=np.float64)
    for row in range(Ny):
        row_values = a[row]
        row_mask = np.isfinite(row_values) if m is None else (m[row] & np.isfinite(row_values))
        if int(row_mask.sum()) < 2:
            continue
        coeff = np.polyfit(x[row_mask], row_values[row_mask], 1)
        bg[row] = np.polyval(coeff, x)
    return bg


def _subtract_scanline_background(
    image: np.ndarray,
    fitted_profile: np.ndarray,
    *,
    x_background: np.ndarray | None = None,
    preserve_level: str = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract a fitted scan-line profile from the full image."""
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("image must be a 2-D array")
    fitted = np.asarray(fitted_profile, dtype=np.float64)
    if fitted.shape != (a.shape[0],):
        raise ValueError("fitted_profile length must match image rows")
    background = fitted[:, None] + np.zeros_like(a, dtype=np.float64)
    if x_background is not None:
        xb = np.asarray(x_background, dtype=np.float64)
        if xb.shape != a.shape:
            raise ValueError("x_background shape must match image shape")
        background = background + xb
    if preserve_level == "median":
        reference = _finite_median(background, default=0.0)
    elif preserve_level == "mean":
        reference = _finite_mean(background, default=0.0)
    elif preserve_level in (None, "none"):
        reference = 0.0
    else:
        raise ValueError("preserve_level must be 'median', 'mean', or 'none'")
    return a - background + reference, background


_VALID_FIT_REGIONS = {"whole_image", "active_roi"}


def preview_stm_background(
    image: np.ndarray,
    params: STMBackgroundParams | None = None,
    mask: np.ndarray | None = None,
) -> STMBackgroundResult:
    """Return non-destructive STM background preview data.

    Parameters
    ----------
    image
        2-D scan plane.
    params
        Background parameters.  ``params.fit_region`` must be
        ``"whole_image"`` or ``"active_roi"``.  When ``"active_roi"`` is
        requested the caller **must** supply ``mask`` — the function cannot
        resolve a ROI id to a pixel mask on its own.
    mask
        Boolean array (same shape as ``image``).  Pixels where ``mask`` is
        True are used to estimate the background; subtraction is applied to
        the full image regardless.  Must be provided when
        ``params.fit_region == "active_roi"``.
    """
    params = params or STMBackgroundParams()
    fit_region = str(params.fit_region or "whole_image")
    if fit_region not in _VALID_FIT_REGIONS:
        raise ValueError(
            f"fit_region must be one of {sorted(_VALID_FIT_REGIONS)}, got {fit_region!r}"
        )
    if fit_region == "active_roi" and mask is None:
        raise ValueError(
            "fit_region='active_roi' requires a mask; pass the ROI pixel mask "
            "as the mask argument"
        )
    model = _normalise_stm_background_model(params.model)
    statistic = _normalise_line_statistic(params.line_statistic)
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("image must be a 2-D array")
    m = _coerce_mask(mask, a.shape)
    x_bg = _linear_x_background(a, m) if params.linear_x_first else None
    working = a - x_bg if x_bg is not None else a
    profile = compute_scanline_profile(working, m, statistic)
    fitted = _fit_scanline_background(
        profile,
        model,
        blur_length=params.blur_length,
        jump_threshold=params.jump_threshold,
    )
    corrected, background = _subtract_scanline_background(
        a,
        fitted,
        x_background=x_bg,
        preserve_level=params.preserve_level,
    )
    resolved = STMBackgroundParams(
        fit_region=params.fit_region,
        line_statistic=statistic,
        model=model,
        linear_x_first=bool(params.linear_x_first),
        blur_length=params.blur_length,
        jump_threshold=params.jump_threshold,
        preserve_level=params.preserve_level,
    )
    return STMBackgroundResult(
        corrected=corrected,
        background_image=background,
        line_profile=profile,
        fitted_profile=fitted,
        params=resolved,
        fit_status="success",
    )


def apply_stm_background(
    image: np.ndarray,
    params: STMBackgroundParams | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply STM scan-line background subtraction and return corrected data."""
    return preview_stm_background(image, params=params, mask=mask).corrected


# ═════════════════════════════════════════════════════════════════════════════
# 3.  subtract_background
# ═════════════════════════════════════════════════════════════════════════════

def _poly_terms(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Return an (N, M) design matrix for a 2D polynomial up to total degree `order`.

    All terms x^i * y^j where i + j <= order, ordered by total degree then by
    increasing power of x.  The exact column order is deterministic and used
    consistently in both fitting and reconstruction.
    """
    cols = []
    for total in range(order + 1):
        for i in range(total + 1):
            j = total - i
            cols.append((x ** i) * (y ** j))
    return np.column_stack(cols)


def subtract_background(
    arr: np.ndarray,
    order: int = 1,
    *,
    fit_roi: "Any | None" = None,
    apply_roi: "Any | None" = None,
    exclude_roi: "Any | None" = None,
    step_tolerance: bool = False,
    step_threshold_deg: float = 3.0,
    fit_rect: Optional[tuple[int, int, int, int]] = None,
    fit_mask: Optional[np.ndarray] = None,
    pixel_size_x_m: float = 1.0,
    pixel_size_y_m: float = 1.0,
) -> np.ndarray:
    """Fit and subtract a 2-D polynomial background from an image.

    Parameters
    ----------
    arr:
        2-D image array.
    order:
        Total polynomial order. Supported values are 1, 2, 3, 4.
    fit_roi:
        ROI whose pixels are used to *estimate* the background.  Pixels
        outside this region do not influence the polynomial fit, but the
        fitted background is extrapolated and (optionally) subtracted over
        the full image or ``apply_roi``.  ``None`` means all pixels.
    apply_roi:
        ROI whose pixels are *modified* by subtracting the fitted background.
        Pixels outside this region are returned unchanged.  ``None`` means
        subtract the background everywhere.

        .. note::
            Where ``apply_roi`` ends, there will be a discontinuity in the
            output equal to the local background value.  This is correct
            behaviour for the "patch-only" correction case.  For per-terrace
            fits where a smooth boundary matters, ensure ``fit_roi`` includes
            the full terrace and ``apply_roi = None``.

    exclude_roi:
        ROI whose pixels are *removed from the fit*.  Applied after
        ``fit_roi``; the effective fit region is
        ``fit_roi AND NOT exclude_roi``.  If ``exclude_roi`` is partially
        outside ``fit_roi``, only the overlapping part is excluded.
        ``None`` means no exclusion.

    Canonical use patterns
    ----------------------
    Global plane fit (standard background removal)::

        subtract_background(img)

    Fit on a clean terrace, subtract globally::

        subtract_background(img, fit_roi=terrace_roi)

    Fit on a terrace, apply only to an adjacent region::

        subtract_background(img, fit_roi=terrace_roi, apply_roi=target_roi)

    Fit globally but exclude contaminated molecules::

        subtract_background(img, exclude_roi=molecules_roi)

    Correct only a small patch (fit and apply to the same region)::

        subtract_background(img, fit_roi=patch_roi, apply_roi=patch_roi)

    Legacy parameters (still accepted, combined with ROI parameters)
    -----------------------------------------------------------------
    step_tolerance:
        When True, use a step-tolerant surface mask: pixels whose finite-
        difference gradient exceeds ``tan(step_threshold_deg)`` are excluded
        from the fit. Falls back to a full-pixel fit when fewer than
        ``n_terms`` pixels remain after masking.
    step_threshold_deg:
        Slope angle (degrees) above which a pixel is treated as a step edge.
    pixel_size_x_m:
        Physical pixel width in metres. Used only when ``step_tolerance=True``
        to convert the finite-difference gradient from data-units/pixel to the
        dimensionless slope ratio ``dz/dx`` before comparing against
        ``tan(step_threshold_deg)``.  Default 1.0 (pixel-unit gradient).
    pixel_size_y_m:
        Physical pixel height in metres, same purpose as ``pixel_size_x_m``.
    fit_rect:
        Optional inclusive pixel rectangle ``(x0, y0, x1, y1)`` restricting
        the fit region.  Combined with ``fit_roi`` (intersection).
    fit_mask:
        Optional boolean mask restricting the fit region.  Combined with
        ``fit_roi`` and ``fit_rect`` (intersection).

    Coordinates are normalised to [-1, 1] for numerical stability. Only
    finite pixels participate in the least-squares fit. NaNs in the input
    are preserved in the output.
    """
    if order < 1 or order > 4:
        raise ValueError(f"order must be 1..4, got {order}")
    if step_tolerance and not np.isfinite(step_threshold_deg):
        raise ValueError("step_threshold_deg must be finite")

    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape

    ys = np.linspace(-1.0, 1.0, Ny)
    xs = np.linspace(-1.0, 1.0, Nx)
    Xg, Yg = np.meshgrid(xs, ys)

    flat_x = Xg.ravel()
    flat_y = Yg.ravel()
    flat_z = arr.ravel()

    finite = np.isfinite(flat_z)
    n_terms = (order + 1) * (order + 2) // 2
    # Graceful fallback when the image itself lacks enough finite pixels
    if finite.sum() < n_terms:
        return arr

    # ── Build fit mask from new ROI parameters ────────────────────────────────
    user_fit_mask = fit_mask  # legacy param
    has_explicit_region = (
        fit_roi is not None or exclude_roi is not None
        or fit_rect is not None or user_fit_mask is not None
    )

    # Merge fit_roi and exclude_roi into a single ROI-level mask
    roi_fit_mask: Optional[np.ndarray] = None
    if fit_roi is not None:
        roi_fit_mask = np.asarray(fit_roi.to_mask(arr.shape), dtype=bool)
    if exclude_roi is not None:
        excl = np.asarray(exclude_roi.to_mask(arr.shape), dtype=bool)
        roi_fit_mask = (~excl) if roi_fit_mask is None else (roi_fit_mask & ~excl)

    # ── Accumulate fit_mask (intersection of all constraints) ─────────────────
    fit_mask_acc = finite.copy()

    if fit_rect is not None:
        try:
            x0r, y0r, x1r, y1r = [int(v) for v in fit_rect]
        except (TypeError, ValueError):
            return arr
        x0r = max(0, min(Nx - 1, x0r))
        x1r = max(0, min(Nx - 1, x1r))
        y0r = max(0, min(Ny - 1, y0r))
        y1r = max(0, min(Ny - 1, y1r))
        if x1r <= x0r or y1r <= y0r:
            return arr
        rect_mask = np.zeros(arr.shape, dtype=bool)
        rect_mask[y0r:y1r + 1, x0r:x1r + 1] = True
        fit_mask_acc &= rect_mask.ravel()

    if user_fit_mask is not None:
        try:
            um = np.asarray(user_fit_mask, dtype=bool)
        except (TypeError, ValueError):
            return arr
        if um.shape != arr.shape:
            return arr
        fit_mask_acc &= um.ravel()

    if roi_fit_mask is not None:
        fit_mask_acc &= roi_fit_mask.ravel()

    if fit_mask_acc.sum() < n_terms:
        if has_explicit_region:
            raise ValueError(
                f"Fit region has only {int(fit_mask_acc.sum())} pixel(s), but order "
                f"{order} requires at least {n_terms} pixels."
            )
        return arr  # graceful fallback for whole-image degenerate cases

    # ── Step-tolerance masking ────────────────────────────────────────────────
    if step_tolerance and Ny >= 3 and Nx >= 3:
        gy, gx = np.gradient(np.where(np.isfinite(arr), arr, _finite_median(arr)))
        # Clamp pixel sizes to 1e-30 m/px as a guard against division by zero or NaN.
        # If pixel_size is 0 or NaN, the clamp ensures the code doesn't crash; the
        # resulting unphysical slope is then rejected by the threshold comparison.
        psx = max(float(pixel_size_x_m), 1e-30)
        psy = max(float(pixel_size_y_m), 1e-30)
        slope_mag = np.sqrt((gx / psx) ** 2 + (gy / psy) ** 2).ravel()
        tan_thresh = math.tan(math.radians(step_threshold_deg))
        candidate = finite & (slope_mag < tan_thresh) & fit_mask_acc
        if candidate.sum() >= n_terms:
            fit_mask_acc = candidate

    # ── Fit polynomial ────────────────────────────────────────────────────────
    A = _poly_terms(flat_x[fit_mask_acc], flat_y[fit_mask_acc], order)
    b = flat_z[fit_mask_acc]
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    bg = (_poly_terms(flat_x, flat_y, order) @ coeffs).reshape(Ny, Nx)

    # ── Apply background subtraction ──────────────────────────────────────────
    if apply_roi is not None:
        apply_mask = np.asarray(apply_roi.to_mask(arr.shape), dtype=bool)
        arr[apply_mask] -= bg[apply_mask]
        return arr
    return arr - bg


# ═════════════════════════════════════════════════════════════════════════════
# 3.  stm_line_background
# ═════════════════════════════════════════════════════════════════════════════

def _modal_shift(values: np.ndarray, *, bins: int = 128) -> Optional[float]:
    """Estimate the dominant value from the peak of a 1-D histogram."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if values.size == 1:
        return float(values[0])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmin == vmax:
        return vmin
    try:
        # numpy>=2 raises ValueError when (vmax-vmin)/n_bins underflows; fall
        # back to the median for residuals whose range is below float64 precision.
        hist, edges = np.histogram(values, bins=min(bins, max(8, values.size)))
    except ValueError:
        return float(np.nanmedian(values))
    if hist.size == 0 or int(hist.max()) == 0:
        return float(np.nanmedian(values))
    peak = int(np.argmax(hist))
    # Match np.histogram's right-open bin convention exactly:
    #   bin k contains v iff edges[k] <= v < edges[k+1]
    # except the LAST bin which is closed-right (edges[k] <= v <= edges[k+1]).
    # The previous implementation used >= and <= on both sides, so on a
    # value sitting exactly at an inner boundary the "in_peak" mask could
    # disagree with which bin np.histogram counted it in (review
    # numerical #6 / image-proc #14, fixed 2026-05-28).
    bin_idx = np.minimum(
        np.searchsorted(edges, values, side="right") - 1,
        len(edges) - 2,
    )
    in_peak = bin_idx == peak
    if np.any(in_peak):
        return float(np.nanmedian(values[in_peak]))
    return float(0.5 * (edges[peak] + edges[peak + 1]))


def stm_line_background(arr: np.ndarray, mode: str = "step_tolerant") -> np.ndarray:
    """Subtract an STM-style step-tolerant line background.

    The step-tolerant mode estimates each adjacent scan-line offset from the
    modal peak of a histogram of pixelwise row differences.  This follows the
    dominant terrace-to-terrace shift rather than the mean height, so partial
    step edges do not dominate the correction.
    """
    if mode != "step_tolerant":
        raise ValueError(f"mode must be 'step_tolerant', got {mode!r}")

    arr = arr.astype(np.float64, copy=True)
    Ny, Nx = arr.shape
    if Ny < 2 or Nx < 1:
        return arr

    shifts = np.zeros(Ny, dtype=np.float64)
    prev_shift = 0.0
    have_shift = False

    for r in range(1, Ny):
        diff = arr[r] - arr[r - 1]
        diff = diff[np.isfinite(diff)]
        if diff.size == 0:
            shifts[r] = shifts[r - 1]
            continue

        if have_shift:
            residual = diff - prev_shift
            delta = _modal_shift(residual)
            if delta is None:
                warnings.warn(
                    f"stm_line_background: modal shift estimation failed for row {r};"
                    " propagating previous shift",
                    RuntimeWarning,
                    stacklevel=2,
                )
                shift = prev_shift
            else:
                shift = prev_shift + delta
        else:
            modal = _modal_shift(diff)
            if modal is None:
                shifts[r] = shifts[r - 1]
                continue
            shift = modal
            have_shift = True

        prev_shift = shift
        shifts[r] = shifts[r - 1] + shift

    if not have_shift:
        return arr

    profile = shifts - float(np.nanmedian(shifts))
    return arr - profile[:, None]
