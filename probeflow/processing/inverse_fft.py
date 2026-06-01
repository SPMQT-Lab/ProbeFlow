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
from dataclasses import dataclass

import numpy as np

__all__ = [
    "FourierEllipse",
    "InverseFFTResult",
    "fourier_ellipse_mask",
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


@dataclass
class InverseFFTResult:
    result: np.ndarray          # reconstructed real-space image
    residual: np.ndarray        # original − result
    masked_fft: np.ndarray      # |F · effective_mask|, fftshift-centred
    mask_used: np.ndarray       # the effective mask actually applied to F
    imag_residual_norm: float   # ‖Im(ifft)‖ / ‖Re(ifft)‖ — ~0 when symmetric


def fourier_ellipse_mask(
    shape: tuple[int, int],
    ellipses,
    *,
    conjugate: bool = True,
    soft_px: float = 0.0,
) -> np.ndarray:
    """Build a float selection mask (0–1) in fftshift-centred FFT space.

    Each :class:`FourierEllipse` marks pixels inside it as 1.  When
    ``conjugate`` is True (the default, required for a real reconstruction) an
    exact point-reflected partner is added for every ellipse.  ``soft_px > 0``
    gives the boundary a linear cosine-like ramp of that pixel width (reduces
    ringing); ``soft_px == 0`` gives a hard edge.
    """
    Ny, Nx = int(shape[0]), int(shape[1])
    cy, cx = Ny // 2, Nx // 2
    yy, xx = np.mgrid[:Ny, :Nx].astype(np.float64)
    mask = np.zeros((Ny, Nx), dtype=np.float64)

    for e in ellipses:
        centres = [(e.dx, e.dy)]
        if conjugate:
            centres.append((-e.dx, -e.dy))   # exact point reflection through DC
        rx = max(float(e.rx), 1e-6)
        ry = max(float(e.ry), 1e-6)
        theta = math.radians(float(e.angle_deg))
        ct, st = math.cos(theta), math.sin(theta)
        for sx, sy in centres:
            px, py = cx + sx, cy + sy
            xr = (xx - px) * ct + (yy - py) * st
            yr = -(xx - px) * st + (yy - py) * ct
            d = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)   # 1.0 on the boundary
            if soft_px > 0:
                # Linear ramp from 1 (inside) to 0 over ``soft_px`` past the edge.
                ramp = 1.0 - (d - 1.0) / (float(soft_px) / max(rx, ry))
                contribution = np.clip(ramp, 0.0, 1.0)
            else:
                contribution = (d <= 1.0).astype(np.float64)
            mask = np.maximum(mask, contribution)

    return mask


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
    ellipses,
    *,
    mode: str = "remove_selected",
    conjugate: bool = True,
    soft_px: float = 0.0,
) -> np.ndarray:
    """Convenience: build the ellipse mask and return only the reconstructed image.

    This is the entry point the ``inverse_fft_filter`` ProcessingState op calls.
    Returns the input unchanged when there are no selections.
    """
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("inverse_fft_filter expects a 2-D image")
    if not ellipses:
        return a.copy()
    mask = fourier_ellipse_mask(a.shape, ellipses, conjugate=conjugate, soft_px=soft_px)
    return inverse_fft_from_mask(a, mask, mode=mode).result
