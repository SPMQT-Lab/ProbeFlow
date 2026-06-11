"""Advanced edge detection: Canny and Sobel/Scharr gradient filters.

These detectors are *non-destructive analysis* operations: they return an
:class:`EdgeDetectionResult` carrying any of a display image, a boolean edge
mask, a gradient magnitude, and a gradient orientation, plus the parameters
used (for provenance and auto-naming).  Unlike :func:`probeflow.processing.
filters.edge_detect` (a history-replayable display filter), these functions are
the backend for the Advanced Edge Detection dialog and the active-mask layer.

NaN handling follows the same convention as ``edge_detect``: non-finite pixels
are filled with the finite mean before filtering, and NaN is restored in the
returned ``display_image`` / excluded from any mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._image_utils import _finite_mean, _nonnegative_finite

__all__ = [
    "EdgeDetectionResult",
    "CANNY_PRESETS",
    "canny_edges",
    "gradient_filter",
]


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EdgeDetectionResult:
    """Outputs of one advanced edge-detection operation.

    An edge detector can yield more than one useful product, so each field is
    optional and populated only when meaningful for the method:

    * ``display_image``       — array suitable for direct display (NaN-preserving).
    * ``edge_mask``           — boolean array suitable for the active-mask layer.
    * ``gradient_magnitude``  — continuous gradient response (Sobel/Scharr).
    * ``gradient_orientation``— gradient direction in radians, ``arctan2(gy, gx)``,
      in **image coordinates** (the row axis y increases *downward*), so a
      positive angle turns from +x toward the bottom of the image — not the
      math/physical y-up convention.  NaN marks pixels with no valid gradient
      (NaN-contaminated or outside the ROI).
    """

    method: str
    source_channel: str | None = None
    display_image: np.ndarray | None = None
    edge_mask: np.ndarray | None = None
    gradient_magnitude: np.ndarray | None = None
    gradient_orientation: np.ndarray | None = None
    parameters: dict[str, object] = field(default_factory=dict)
    pixel_size_nm: float | None = None


# ── Canny presets (sigma in px; low/high as percentiles) ────────────────────────

CANNY_PRESETS: dict[str, dict[str, float]] = {
    "Fine atomic/defect edges": {"sigma": 0.8, "low": 70.0, "high": 90.0},
    "Step edges / islands":     {"sigma": 2.0, "low": 60.0, "high": 85.0},
    "Noisy scan":               {"sigma": 3.0, "low": 50.0, "high": 80.0},
    "Strict edges only":        {"sigma": 1.5, "low": 80.0, "high": 95.0},
}


# ── Shared NaN-handling helper ──────────────────────────────────────────────────

def _prepare(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(filled, nan_mask)``: a finite float64 copy and the NaN mask.

    Non-finite pixels are replaced with the finite mean so spatial filters do
    not propagate NaN; the mask lets callers restore NaN afterwards.
    """
    a = np.asarray(arr, dtype=np.float64)
    nan_mask = ~np.isfinite(a)
    if nan_mask.any():
        a = a.copy()
        a[nan_mask] = _finite_mean(a)
    return a, nan_mask


def _coerce_roi_mask(roi_mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if roi_mask is None:
        return None
    m = np.asarray(roi_mask, dtype=bool)
    if m.shape != shape:
        raise ValueError(f"roi_mask shape {m.shape} != image shape {shape}")
    return m


# ── Canny ───────────────────────────────────────────────────────────────────────

def canny_edges(
    arr: np.ndarray,
    *,
    sigma: float = 1.0,
    threshold_mode: str = "percentile",
    low: float = 70.0,
    high: float = 90.0,
    roi_mask: np.ndarray | None = None,
    preset: str | None = None,
    pixel_size_nm: float | None = None,
    pixel_size_x_nm: float | None = None,
    pixel_size_y_nm: float | None = None,
    source_channel: str | None = None,
) -> EdgeDetectionResult:
    """Detect edges with the Canny algorithm (``skimage.feature.canny``).

    Parameters
    ----------
    sigma:
        Gaussian smoothing width **in pixels**.  ``skimage.feature.canny`` only
        accepts a scalar sigma, so the smoothing is isotropic *in pixel space*.
        On anisotropic scans (``pixel_size_x_nm != pixel_size_y_nm``) it is
        therefore not isotropic in physical space; the recorded ``sigma_x_nm`` /
        ``sigma_y_nm`` express the per-axis physical extent for provenance.
    threshold_mode:
        ``"percentile"`` (default) interprets *low*/*high* as percentiles
        (0–100) of the gradient magnitude — robust across STM channels whose
        absolute scale varies.  ``"absolute"`` uses *low*/*high* directly.
    low, high:
        Hysteresis thresholds.  Percentiles in percentile mode, raw gradient
        values in absolute mode.
    roi_mask:
        Optional boolean mask; edges are only detected where it is True.
    preset:
        Name in :data:`CANNY_PRESETS`; when given it overrides *sigma*/*low*/
        *high*.
    pixel_size_x_nm, pixel_size_y_nm:
        Optional physical pixel spacings, recorded for provenance only (the
        smoothing remains pixel-space).  Fall back to *pixel_size_nm*.
    """
    from skimage.feature import canny as _canny

    if preset is not None:
        if preset not in CANNY_PRESETS:
            raise ValueError(f"Unknown Canny preset {preset!r}. Choices: {sorted(CANNY_PRESETS)}")
        p = CANNY_PRESETS[preset]
        sigma, low, high = p["sigma"], p["low"], p["high"]

    sigma = _nonnegative_finite(sigma, "sigma")
    if threshold_mode not in ("percentile", "absolute"):
        raise ValueError(f"Unknown threshold_mode {threshold_mode!r}")
    if low > high:
        low, high = high, low

    a, nan_mask = _prepare(arr)
    roi = _coerce_roi_mask(roi_mask, a.shape)

    # Restrict Canny to finite pixels (and the ROI, if given).
    valid = ~nan_mask
    canny_mask = valid if roi is None else (valid & roi)

    use_quantiles = threshold_mode == "percentile"
    if use_quantiles:
        low_t, high_t = low / 100.0, high / 100.0
    else:
        low_t, high_t = float(low), float(high)

    if nan_mask.all() or not canny_mask.any():
        edge = np.zeros(a.shape, dtype=bool)
    else:
        if use_quantiles and not canny_mask.all():
            # skimage computes quantile thresholds over the WHOLE gradient-
            # magnitude array even when a mask is given, so the near-zero
            # gradient outside the ROI/valid region dilutes the percentile
            # cut and systematically over-detects edges inside the region
            # (review: outside-region pixels must not enter threshold
            # statistics). Compute the percentiles over the restricted
            # region ourselves — mirroring skimage's smoothed ndi.sobel
            # magnitude — and hand skimage absolute thresholds instead.
            from scipy import ndimage as _ndi

            smoothed = _ndi.gaussian_filter(a, max(sigma, 1e-3))
            mag = np.hypot(
                _ndi.sobel(smoothed, axis=0), _ndi.sobel(smoothed, axis=1)
            )
            vals = mag[canny_mask]
            low_t, high_t = (
                float(v) for v in np.percentile(vals, [float(low), float(high)])
            )
            use_quantiles = False
        edge = _canny(
            a,
            sigma=max(sigma, 1e-3),
            low_threshold=low_t,
            high_threshold=high_t,
            mask=canny_mask,
            use_quantiles=use_quantiles,
        )

    display = edge.astype(np.float64)
    if nan_mask.any():
        display[nan_mask] = np.nan

    params: dict[str, object] = {
        "method": "canny",
        "sigma": float(sigma),
        "threshold_mode": threshold_mode,
        "low": float(low),
        "high": float(high),
        "roi_restricted": roi is not None,
        "preset": preset,
        "source_channel": source_channel,
    }
    # Per-axis physical extent of the (pixel-space) Gaussian, for provenance.
    dx = pixel_size_x_nm or pixel_size_nm
    dy = pixel_size_y_nm or pixel_size_nm
    if dx:
        params["sigma_x_nm"] = float(sigma) * float(dx)
    if dy:
        params["sigma_y_nm"] = float(sigma) * float(dy)
    if pixel_size_nm is not None:
        params["sigma_nm"] = float(sigma) * float(pixel_size_nm)

    return EdgeDetectionResult(
        method="canny",
        source_channel=source_channel,
        display_image=display,
        edge_mask=edge,
        parameters=params,
        pixel_size_nm=pixel_size_nm,
    )


# ── Sobel / Scharr gradient ──────────────────────────────────────────────────────

_GRADIENT_OUTPUTS = ("magnitude", "x", "y", "orientation")


def gradient_filter(
    arr: np.ndarray,
    *,
    operator: str = "sobel",
    output: str = "magnitude",
    normalize: bool = True,
    threshold_to_mask: bool = False,
    threshold: float = 90.0,
    roi_mask: np.ndarray | None = None,
    pixel_size_nm: float | None = None,
    pixel_size_x_nm: float | None = None,
    pixel_size_y_nm: float | None = None,
    source_channel: str | None = None,
) -> EdgeDetectionResult:
    """Compute a Sobel or Scharr gradient response.

    Parameters
    ----------
    operator:
        ``"sobel"`` or ``"scharr"``.
    output:
        ``"magnitude"`` (default), ``"x"``, ``"y"``, or ``"orientation"``.
    normalize:
        Scale the chosen output to [0, 1] (magnitude) or [-1, 1] (signed) by
        its peak absolute value.  Orientation is never rescaled.
    threshold_to_mask:
        When True, threshold the gradient *magnitude* (always, regardless of
        *output*) at the *threshold* percentile to produce ``edge_mask``.
    roi_mask:
        Optional boolean mask; gradients outside it are excluded (magnitude → 0,
        orientation → NaN) and dropped from any threshold mask.
    pixel_size_x_nm, pixel_size_y_nm:
        Physical pixel spacings.  When given, the column/row derivatives are
        scaled to physical units (∂z/∂x, ∂z/∂y) before forming the magnitude
        and orientation, so anisotropic pixels yield physically correct values.
        Falls back to *pixel_size_nm* (isotropic) and then to 1 px.

    Notes
    -----
    Pixels whose 3×3 operator footprint touches a non-finite value are treated
    as invalid (their gradient is a mean-fill artefact, not a real edge):
    magnitude → 0, orientation → NaN, display → NaN, and they are excluded from
    the threshold mask.  ``gradient_orientation`` is in image coordinates (row
    axis points down); see :class:`EdgeDetectionResult`.
    """
    from skimage import filters as _skf

    if operator not in ("sobel", "scharr"):
        raise ValueError(f"Unknown operator {operator!r}. Choices: 'sobel', 'scharr'")
    if output not in _GRADIENT_OUTPUTS:
        raise ValueError(f"Unknown output {output!r}. Choices: {_GRADIENT_OUTPUTS}")

    a, nan_mask = _prepare(arr)
    roi = _coerce_roi_mask(roi_mask, a.shape)

    if operator == "sobel":
        gy = _skf.sobel_h(a)   # derivative along axis 0 (rows / y)
        gx = _skf.sobel_v(a)   # derivative along axis 1 (cols / x)
    else:
        gy = _skf.scharr_h(a)
        gx = _skf.scharr_v(a)
    # skimage's normalised Sobel/Scharr kernels respond with 2.0 on a
    # unit-slope ramp (their [-1, 0, 1] difference spans two pixels with no
    # /2), so halve the responses to get the true central-difference
    # derivative per pixel; only then does the physical scaling below yield
    # real ∂z/∂x, ∂z/∂y. Relative outputs (normalised display, percentile
    # thresholds, orientation) are unaffected.
    gx = gx * 0.5
    gy = gy * 0.5

    # Scale per-pixel derivatives to physical units so magnitude/orientation
    # are correct on anisotropic pixels (∂z/∂x = Gₓ / dx, ∂z/∂y = G_y / dy).
    dx = pixel_size_x_nm or pixel_size_nm
    dy = pixel_size_y_nm or pixel_size_nm
    if dx and dx > 0:
        gx = gx / float(dx)
    if dy and dy > 0:
        gy = gy / float(dy)

    magnitude = np.hypot(gx, gy)
    orientation = np.arctan2(gy, gx)

    # Mean-filling non-finite pixels (see ``_prepare``) creates a step at the
    # hole rim, so every pixel whose 3×3 operator footprint touches a NaN has a
    # spurious gradient.  Treat that 1-px band as invalid, not just the NaN
    # pixels themselves.
    if nan_mask.any():
        from scipy.ndimage import binary_dilation
        invalid = binary_dilation(nan_mask, structure=np.ones((3, 3), dtype=bool))
    else:
        invalid = nan_mask  # all-False, correct shape

    # Pixels to keep: finite-and-clean, restricted to the ROI when given.
    keep = ~invalid if roi is None else (~invalid & roi)

    # Null the gradient outside ``keep`` *before* normalisation/thresholding so
    # the spurious rim never sets the normalisation peak or the percentile cut.
    # Magnitude/components use 0 (= no gradient); orientation uses NaN because
    # 0.0 is a valid direction (+x) and would be mistaken for genuine data.
    gx = np.where(keep, gx, 0.0)
    gy = np.where(keep, gy, 0.0)
    magnitude = np.where(keep, magnitude, 0.0)
    orientation = np.where(keep, orientation, np.nan)

    if output == "magnitude":
        chosen = magnitude
    elif output == "x":
        chosen = gx
    elif output == "y":
        chosen = gy
    else:  # orientation
        chosen = orientation

    if normalize and output != "orientation":
        peak = float(np.nanmax(np.abs(chosen))) if chosen.size else 0.0
        if peak > 0:
            chosen = chosen / peak

    display = chosen.astype(np.float64, copy=True)
    display[invalid] = np.nan

    edge_mask = None
    if threshold_to_mask:
        ref = magnitude[keep]
        finite_ref = ref[np.isfinite(ref)] if ref.size else ref
        if finite_ref.size:
            cut = float(np.percentile(finite_ref, float(threshold)))
            # Require a meaningfully positive gradient: when the percentile
            # cut is ~0 (flat or sparse-step images) ``>= cut`` would mark the
            # entire zero-gradient background as an edge. Exact ``> 0.0`` is
            # not enough — the Sobel/Scharr kernels leave a float-cancellation
            # residue of O(eps·|data|) on constant non-zero images (~2e-16 for
            # a flat 3.7), which made the whole image an edge mask. The floor
            # scales with the data magnitude and the pixel-size division
            # applied to the gradients; 1e3·eps margin is still ~10 orders of
            # magnitude below any representable real gradient.
            data_scale = float(np.max(np.abs(a))) if a.size else 0.0
            grad_scale = max(
                1.0 / float(dx) if dx and dx > 0 else 1.0,
                1.0 / float(dy) if dy and dy > 0 else 1.0,
            )
            tiny = data_scale * grad_scale * np.finfo(np.float64).eps * 1e3
            edge_mask = (magnitude >= cut) & (magnitude > tiny) & keep
        else:
            edge_mask = np.zeros(a.shape, dtype=bool)

    params: dict[str, object] = {
        "method": operator,
        "output": output,
        "normalize": bool(normalize),
        "threshold_to_mask": bool(threshold_to_mask),
        "threshold": float(threshold),
        "roi_restricted": roi is not None,
        "source_channel": source_channel,
        "pixel_size_x_nm": float(dx) if dx else None,
        "pixel_size_y_nm": float(dy) if dy else None,
    }

    return EdgeDetectionResult(
        method=operator,
        source_channel=source_channel,
        display_image=display,
        edge_mask=edge_mask,
        gradient_magnitude=magnitude,
        gradient_orientation=orientation,
        parameters=params,
        pixel_size_nm=pixel_size_nm or (float(dx) if dx else None),
    )
