"""Inverse FFT / Fourier reconstruction backend.

Makes FFT filtering auditable: build a keep/remove mask in Fourier space, apply
it, inverse-transform, and return the reconstructed image **plus the residual**
(``original − result``) and the masked spectrum, so the user can see exactly
which Fourier components were removed or isolated in real space.

Design (mirrors the invertible path of :func:`probeflow.processing.filters.
periodic_notch_filter`):

* FFT the **raw, unwindowed** image (windowing belongs only to the display
  magnitude — it is not invertible).  Operate on the full spectrum so the DC
  bin (the image mean) is preserved unless a selection covers it.
* Reconstruction is exactly real **only** when the mask is conjugate-symmetric,
  ``mask(−q) = mask(q)``.  Conjugate handling therefore lives in the mask
  builder (:func:`fourier_ellipse_mask`), which places an exact partner at the
  point-reflected centre — this is correct for even-N ``fftshift`` grids where a
  naive array flip is off by one.
* ``inverse_fft_from_mask`` reports ``imag_residual_norm`` so a non-symmetric
  mask (expert mode) surfaces its (small) imaginary part rather than hiding it.

GUI-free (numpy only); import lazily from the FFT viewer / CLI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "FourierEllipse",
    "FourierRect",
    "FourierStrokes",
    "InverseFFTResult",
    "fourier_ellipse_mask",
    "fourier_region_mask",
    "fourier_region_from_dict",
    "inverse_fft_from_mask",
    "inverse_fft_filter",
]


@dataclass(frozen=True)
class FourierEllipse:
    """An elliptical Fourier selection, in FFT-pixel offsets from the centred DC.

    ``dx``/``dy`` are the centre offsets from DC (column/qx and row/qy); ``rx``/
    ``ry`` are the semi-axes in pixels; ``angle_deg`` rotates the ellipse.  A
    circle is ``rx == ry`` with ``angle_deg == 0``.
    """
    dx: float
    dy: float
    rx: float
    ry: float
    angle_deg: float = 0.0


@dataclass(frozen=True)
class FourierRect:
    """A rectangular Fourier selection, in FFT-pixel offsets from the centred DC.

    ``dx``/``dy`` are the centre offsets from DC; ``half_w``/``half_h`` are the
    half-extents in pixels; ``angle_deg`` rotates the rectangle.  A square is
    ``half_w == half_h`` with ``angle_deg == 0``.
    """
    dx: float
    dy: float
    half_w: float
    half_h: float
    angle_deg: float = 0.0


@dataclass(frozen=True)
class FourierStrokes:
    """A freehand (painted) Fourier selection.

    ``stamps`` is a list of ``(dx, dy)`` circular-brush centres in FFT-pixel
    offsets from DC; ``radius`` is the shared brush radius in pixels.  The union
    of the stamped discs forms the region.
    """
    stamps: tuple[tuple[float, float], ...]
    radius: float


@dataclass
class InverseFFTResult:
    result: np.ndarray          # reconstructed real-space image
    residual: np.ndarray        # original − result
    masked_fft: np.ndarray      # |F · effective_mask|, fftshift-centred
    mask_used: np.ndarray       # the effective mask actually applied to F
    imag_residual_norm: float   # ‖Im(ifft)‖ / ‖Re(ifft)‖ — ~0 when symmetric


def _soft_ramp(signed_px: np.ndarray, soft_px: float) -> np.ndarray:
    """Linear ramp from 1 (inside, ``signed_px <= 0``) to 0 over ``soft_px``.

    ``signed_px`` is the signed distance to the region boundary in pixels
    (negative inside, positive outside).  With ``soft_px == 0`` this is a hard
    edge; otherwise the boundary feathers over ``soft_px`` pixels (reduces
    ringing in the reconstruction).
    """
    if soft_px > 0:
        return np.clip(1.0 - signed_px / float(soft_px), 0.0, 1.0)
    return (signed_px <= 0.0).astype(np.float64)


def _ellipse_contribution(xx, yy, cx, cy, e, soft_px):
    rx = max(float(e.rx), 1e-6)
    ry = max(float(e.ry), 1e-6)
    theta = math.radians(float(e.angle_deg))
    ct, st = math.cos(theta), math.sin(theta)
    px, py = cx + e.dx, cy + e.dy
    xr = (xx - px) * ct + (yy - py) * st
    yr = -(xx - px) * st + (yy - py) * ct
    d = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)   # 1.0 on the boundary
    # Convert the normalised radius to an approximate signed distance in pixels.
    signed_px = (d - 1.0) * max(rx, ry)
    return _soft_ramp(signed_px, soft_px)


def _rect_contribution(xx, yy, cx, cy, r, soft_px):
    hw = max(float(r.half_w), 1e-6)
    hh = max(float(r.half_h), 1e-6)
    theta = math.radians(float(r.angle_deg))
    ct, st = math.cos(theta), math.sin(theta)
    px, py = cx + r.dx, cy + r.dy
    xr = (xx - px) * ct + (yy - py) * st
    yr = -(xx - px) * st + (yy - py) * ct
    # Signed distance to the rectangle boundary (approximate, exterior-exact).
    ox = np.abs(xr) - hw
    oy = np.abs(yr) - hh
    outside = np.sqrt(np.maximum(ox, 0.0) ** 2 + np.maximum(oy, 0.0) ** 2)
    inside = np.minimum(np.maximum(ox, oy), 0.0)
    signed_px = outside + inside
    return _soft_ramp(signed_px, soft_px)


def _strokes_contribution(xx, yy, cx, cy, s, sign, soft_px):
    radius = max(float(s.radius), 1e-6)
    stamps = s.stamps if sign > 0 else [(-dx, -dy) for (dx, dy) in s.stamps]
    if not stamps:
        return np.zeros(xx.shape, dtype=np.float64)
    # Nearest-stamp distance → signed distance to the painted region boundary.
    nearest = np.full(xx.shape, np.inf, dtype=np.float64)
    for dx, dy in stamps:
        px, py = cx + float(dx), cy + float(dy)
        nearest = np.minimum(nearest, np.sqrt((xx - px) ** 2 + (yy - py) ** 2))
    signed_px = nearest - radius
    return _soft_ramp(signed_px, soft_px)


def fourier_region_mask(
    shape: tuple[int, int],
    regions,
    *,
    conjugate: bool = True,
    soft_px: float = 0.0,
) -> np.ndarray:
    """Build a float selection mask (0–1) in fftshift-centred FFT space.

    ``regions`` is a heterogeneous list of :class:`FourierEllipse`,
    :class:`FourierRect`, and :class:`FourierStrokes`.  Each marks the pixels it
    covers as 1, accumulated with ``np.maximum``.  When ``conjugate`` is True
    (the default, required for a real reconstruction) an exact point-reflected
    partner is added for every region.  ``soft_px > 0`` feathers boundaries over
    that pixel width (reduces ringing); ``soft_px == 0`` gives a hard edge.
    """
    Ny, Nx = int(shape[0]), int(shape[1])
    cy, cx = Ny // 2, Nx // 2
    yy, xx = np.mgrid[:Ny, :Nx].astype(np.float64)
    mask = np.zeros((Ny, Nx), dtype=np.float64)

    for region in regions:
        if isinstance(region, FourierStrokes):
            signs = (1, -1) if conjugate else (1,)
            for sign in signs:
                mask = np.maximum(
                    mask, _strokes_contribution(xx, yy, cx, cy, region, sign, soft_px))
            continue
        # Ellipse / rect: build the primary, then mirror it through DC.
        variants = [region]
        if conjugate:
            variants.append(replace(region, dx=-region.dx, dy=-region.dy))
        for v in variants:
            if isinstance(v, FourierRect):
                contrib = _rect_contribution(xx, yy, cx, cy, v, soft_px)
            else:
                contrib = _ellipse_contribution(xx, yy, cx, cy, v, soft_px)
            mask = np.maximum(mask, contrib)

    return mask


def fourier_ellipse_mask(
    shape: tuple[int, int],
    ellipses,
    *,
    conjugate: bool = True,
    soft_px: float = 0.0,
) -> np.ndarray:
    """Back-compat wrapper over :func:`fourier_region_mask` for ellipse lists."""
    return fourier_region_mask(
        shape, ellipses, conjugate=conjugate, soft_px=soft_px)


def fourier_region_from_dict(d: dict):
    """Rebuild a region object from a serialised dict (op params / overlay).

    The ``kind`` key selects the type; a missing ``kind`` defaults to
    ``"ellipse"`` for backward compatibility with states saved before rect/paint
    existed.
    """
    kind = str(d.get("kind", "ellipse"))
    if kind == "paint":
        stamps = tuple((float(p[0]), float(p[1])) for p in d.get("stamps", []))
        return FourierStrokes(stamps=stamps, radius=float(d.get("radius", 1.0)))
    if kind == "rect":
        return FourierRect(
            dx=float(d.get("dx", 0.0)), dy=float(d.get("dy", 0.0)),
            half_w=float(d.get("half_w", 1.0)), half_h=float(d.get("half_h", 1.0)),
            angle_deg=float(d.get("angle_deg", 0.0)))
    return FourierEllipse(
        dx=float(d.get("dx", 0.0)), dy=float(d.get("dy", 0.0)),
        rx=float(d.get("rx", 1.0)), ry=float(d.get("ry", 1.0)),
        angle_deg=float(d.get("angle_deg", 0.0)))


def inverse_fft_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str = "remove_selected",
) -> InverseFFTResult:
    """Apply a Fourier mask and inverse-transform.

    ``mode="remove_selected"`` suppresses the selected components (effective mask
    ``1 − mask``); ``mode="keep_selected"`` keeps only them (effective mask
    ``mask``).  Returns the reconstructed image, the residual (``original −
    result``), the masked spectrum magnitude, the effective mask, and the
    imaginary-residual norm.
    """
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("inverse_fft_from_mask expects a 2-D image")
    m = np.asarray(mask, dtype=np.float64)
    if m.shape != a.shape:
        raise ValueError("mask must have the same shape as image")
    if mode not in ("remove_selected", "keep_selected"):
        raise ValueError("mode must be 'remove_selected' or 'keep_selected'")

    nan_mask = ~np.isfinite(a)
    mean_val = float(np.nanmean(a)) if (~nan_mask).any() else 0.0
    filled = np.where(nan_mask, mean_val, a)

    F = np.fft.fftshift(np.fft.fft2(filled))
    effective = (1.0 - m) if mode == "remove_selected" else m
    masked = F * effective

    comp = np.fft.ifft2(np.fft.ifftshift(masked))
    re_norm = float(np.linalg.norm(comp.real))
    imag_residual_norm = float(np.linalg.norm(comp.imag) / (re_norm + 1e-12))

    result = comp.real
    result[nan_mask] = np.nan
    residual = a - result

    return InverseFFTResult(
        result=result,
        residual=residual,
        masked_fft=np.abs(masked),
        mask_used=effective,
        imag_residual_norm=imag_residual_norm,
    )


def inverse_fft_filter(
    image: np.ndarray,
    regions,
    *,
    mode: str = "remove_selected",
    conjugate: bool = True,
    soft_px: float = 0.0,
) -> np.ndarray:
    """Convenience: build the region mask and return only the reconstructed image.

    ``regions`` is a list of :class:`FourierEllipse` / :class:`FourierRect` /
    :class:`FourierStrokes`.  This is the entry point the ``inverse_fft_filter``
    ProcessingState op calls.  Returns the input unchanged when there are no
    selections.
    """
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("inverse_fft_filter expects a 2-D image")
    if not regions:
        return a.copy()
    mask = fourier_region_mask(a.shape, regions, conjugate=conjugate, soft_px=soft_px)
    return inverse_fft_from_mask(a, mask, mode=mode).result
