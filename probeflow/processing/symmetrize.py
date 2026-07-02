"""Rotational / dihedral symmetrization backend.

Enforce an n-fold rotational (optionally dihedral, i.e. + mirror) symmetry on
an image by averaging its symmetry-equivalent copies, and return the result
**plus the residual** (``original − result``) so the user can see exactly what
the symmetrization removed.  Symmetrization fabricates data — it paints the
assumed symmetry over defects, domain boundaries, and drift — so the residual
is a first-class output, not a diagnostic afterthought (same philosophy as
:mod:`probeflow.processing.inverse_fft`).

Design notes
------------
* Rotating the *real-space* image by ``k·360°/n`` and averaging is
  mathematically identical to rotating the complex FFT in ``k·360°/n``
  segments about DC and averaging (rotation about the origin commutes with the
  Fourier transform; translations only add phase).  We do it in real space to
  avoid interpolating complex spectra, whose phase wraps make interpolation
  artifacts much worse.
* **Registration is load-bearing.**  Averaging a lattice with its rotated copy
  is constructive only when the rotation axis passes through a symmetry centre
  of the pattern; an arbitrary axis translates the rotated lattice by a
  non-lattice vector and the average loses contrast (each Fourier component is
  scaled by ``(1 + e^{iq·t})/2``).  Each rotated copy is therefore registered
  back onto the original by FFT cross-correlation (with parabolic sub-pixel
  refinement) before averaging.  ``register=False`` disables this for the
  pre-centred case.
* Rotation leaves the frame at the corners.  A validity weight is rotated and
  shifted alongside each copy, and the average renormalizes per pixel by the
  accumulated weight (``coverage``).  ``strict_coverage=True`` instead returns
  NaN wherever any copy is missing.  Pixels that are NaN in the input carry
  zero weight, so symmetry-equivalent data can fill them in.
* Angles follow the ``atan2(dy, dx)`` convention used by
  :func:`probeflow.processing.bragg.find_bragg_peaks_in_annulus` peak offsets:
  measured from the +x (column) axis toward +y (row) axis.  Pixels are assumed
  square; on anisotropic pixel scales a pixel-space rotation is not a physical
  rotation (the GUI should warn before offering the tool there).

GUI-free (numpy + scipy.ndimage); import lazily from the FFT viewer / CLI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import rotate as _nd_rotate
from scipy.ndimage import shift as _nd_shift

__all__ = [
    "SymmetrizeResult",
    "fold_axis_from_peaks",
    "symmetrize_image",
    "symmetrize_filter",
]

# Below this accumulated weight a pixel has no meaningful contribution and the
# renormalized average would amplify interpolation dust; emit NaN instead.
_MIN_WEIGHT = 0.05


@dataclass
class SymmetrizeResult:
    result: np.ndarray          # symmetrized real-space image (NaN where uncovered)
    residual: np.ndarray        # original − result (NaN where either is NaN)
    symmetrized_fft: np.ndarray  # |FFT(result)|, fftshift-centred, for display
    coverage: np.ndarray        # accumulated validity weight per pixel (max = op count)
    shifts: np.ndarray          # (n_ops, 2) registration shifts (dy, dx) in px
    symmetry_residual_norm: float  # rms(residual) / rms(original − mean) — ~0 when symmetric
    n_ops: int                  # symmetry operations averaged (n_fold, ×2 if mirror)


def _rotated_copy(filled: np.ndarray, weight: np.ndarray,
                  angle_deg: float, mirrored: bool, order: int):
    """Rotate (and optionally pre-mirror) an image and its validity weight.

    Positive ``angle_deg`` moves a feature at ``atan2(dy, dx)`` angle φ to
    φ + angle.  ``scipy.ndimage.rotate`` with default ``axes=(1, 0)`` rotates
    the opposite way in (row, col) space, hence the sign flip.  The mirror is
    a row flip (y → −y about the array centre), applied before the rotation so
    the pair composes to a reflection across the line at ``angle_deg / 2``.
    """
    img, w = filled, weight
    if mirrored:
        img, w = np.flipud(img), np.flipud(w)
    if angle_deg % 360.0 == 0.0:
        return img.copy(), w.copy()
    kw = dict(reshape=False, order=order, mode="constant", cval=0.0,
              prefilter=(order > 1))
    return (_nd_rotate(img, -angle_deg, **kw),
            np.clip(_nd_rotate(w, -angle_deg, **kw), 0.0, 1.0))


def _register_shift(ref_fft: np.ndarray, copy_img: np.ndarray,
                    copy_w: np.ndarray, mean_val: float,
                    max_shift_px: int) -> tuple[float, float]:
    """Best (dy, dx) translation aligning ``copy`` onto the reference.

    FFT cross-correlation of the demeaned, validity-weighted images, peak
    search restricted to ``±max_shift_px`` around zero shift (the registration
    shift is only defined modulo a lattice vector — we want the branch nearest
    zero), 3-point parabolic sub-pixel refinement per axis.
    """
    b_c = (copy_img - mean_val) * copy_w
    # ref_fft is fft2 of the demeaned, weighted reference; the correlation
    # c[s] = Σ ref(x)·copy(x − s) peaks at the shift to apply to the copy.
    cc = np.fft.ifft2(ref_fft * np.conj(np.fft.fft2(b_c))).real
    cc = np.fft.fftshift(cc)
    Ny, Nx = cc.shape
    cy, cx = Ny // 2, Nx // 2

    r = int(max_shift_px)
    win = cc[max(0, cy - r): cy + r + 1, max(0, cx - r): cx + r + 1]
    iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
    py, px = iy + max(0, cy - r), ix + max(0, cx - r)

    def _parabolic(v_m, v_0, v_p) -> float:
        denom = v_m - 2.0 * v_0 + v_p
        if abs(denom) < 1e-30:
            return 0.0
        return float(np.clip(0.5 * (v_m - v_p) / denom, -0.5, 0.5))

    dy = float(py - cy)
    dx = float(px - cx)
    if 0 < py < Ny - 1:
        dy += _parabolic(cc[py - 1, px], cc[py, px], cc[py + 1, px])
    if 0 < px < Nx - 1:
        dx += _parabolic(cc[py, px - 1], cc[py, px], cc[py, px + 1])
    return dy, dx


def symmetrize_image(
    image: np.ndarray,
    n_fold: int,
    *,
    mirror: bool = False,
    mirror_axis_deg: float = 0.0,
    register: bool = True,
    max_shift_px: int | None = None,
    interpolation: str = "linear",
    strict_coverage: bool = False,
) -> SymmetrizeResult:
    """Average an image over the C_n (or D_n, with ``mirror``) symmetry group.

    Parameters
    ----------
    image
        2-D real-space image.  NaNs are treated as missing data (zero weight)
        and may be filled in by symmetry-equivalent pixels.
    n_fold
        Rotation order ``n ≥ 1``; copies are rotated by ``k·360°/n``.
        ``n_fold=1`` with ``mirror=False`` is the identity.
    mirror
        Also average the ``n`` mirrored copies (dihedral group D_n).
    mirror_axis_deg
        Angle of the mirror line through the image centre, ``atan2(dy, dx)``
        convention.  Irrelevant for pure rotations; for mirrors, align it to
        the lattice (see :func:`fold_axis_from_peaks`).
    register
        Register every copy back onto the original by cross-correlation before
        averaging.  Without this, a symmetry axis that does not pass through
        the image centre destroys contrast instead of denoising.
    max_shift_px
        Half-width of the registration search window (default
        ``min(shape) // 4``).  The optimum is only defined modulo a lattice
        vector, so the window keeps the branch nearest zero shift.
    interpolation
        ``"linear"`` (default, order 1 — matches the geometry ops) or
        ``"cubic"`` (order 3, sharper lattices at slightly more ringing).
    strict_coverage
        When True, output NaN wherever any symmetry copy left the frame
        (coverage below the full operation count) instead of renormalizing.
    """
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("symmetrize_image expects a 2-D image")
    n = int(n_fold)
    if n < 1:
        raise ValueError("n_fold must be >= 1")
    if interpolation not in ("linear", "cubic"):
        raise ValueError("interpolation must be 'linear' or 'cubic'")
    order = 1 if interpolation == "linear" else 3

    valid = np.isfinite(a)
    if not valid.any():
        raise ValueError("image contains no finite pixels")
    mean_val = float(a[valid].mean())
    filled = np.where(valid, a, mean_val)
    w0 = valid.astype(np.float64)

    if max_shift_px is None:
        max_shift_px = max(1, min(a.shape) // 4)

    # (angle, mirrored) members of C_n / D_n.  A mirror across the line at
    # angle φ is R(2φ) ∘ (y → −y); composing with the k-th rotation gives
    # R(k·α + 2φ) ∘ flip.
    alpha = 360.0 / n
    ops: list[tuple[float, bool]] = [(k * alpha, False) for k in range(n)]
    if mirror:
        ops += [(k * alpha + 2.0 * float(mirror_axis_deg), True) for k in range(n)]
    n_ops = len(ops)

    if n_ops == 1:
        result = np.where(valid, a, np.nan)
        return SymmetrizeResult(
            result=result, residual=np.where(valid, 0.0, np.nan), coverage=w0,
            symmetrized_fft=np.abs(np.fft.fftshift(np.fft.fft2(filled))),
            shifts=np.zeros((1, 2)), symmetry_residual_norm=0.0, n_ops=1)

    # Skip registration on (near-)constant images: the correlation surface is
    # flat and argmax would return an arbitrary corner of the window.
    contrast = float(a[valid].std())
    do_register = register and contrast > 1e-12 * (abs(mean_val) + 1.0)
    ref_fft = np.fft.fft2((filled - mean_val) * w0) if do_register else None

    num = np.zeros_like(a)
    den = np.zeros_like(a)
    shifts = np.zeros((n_ops, 2), dtype=np.float64)
    for i, (angle, mirrored) in enumerate(ops):
        if not mirrored and angle % 360.0 == 0.0:
            num += filled * w0
            den += w0
            continue
        img, w = _rotated_copy(filled, w0, angle, mirrored, order)
        if do_register:
            dy, dx = _register_shift(ref_fft, img, w, mean_val, max_shift_px)
            if dy != 0.0 or dx != 0.0:
                skw = dict(order=1, mode="constant", cval=0.0)
                img = _nd_shift(img, (dy, dx), **skw)
                w = np.clip(_nd_shift(w, (dy, dx), **skw), 0.0, 1.0)
            shifts[i] = (dy, dx)
        num += img * w
        den += w

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(den > _MIN_WEIGHT, num / np.maximum(den, 1e-12), np.nan)
    if strict_coverage:
        result = np.where(den >= n_ops - 1e-3, result, np.nan)

    residual = np.where(valid & np.isfinite(result), a - result, np.nan)
    res_valid = np.isfinite(residual)
    if res_valid.any():
        rms_res = float(np.sqrt(np.nanmean(residual[res_valid] ** 2)))
        rms_sig = float(np.sqrt(np.mean((a[valid] - mean_val) ** 2)))
        symmetry_residual_norm = rms_res / (rms_sig + 1e-12)
    else:
        symmetry_residual_norm = float("nan")

    fft_src = np.where(np.isfinite(result), result, mean_val)
    return SymmetrizeResult(
        result=result,
        residual=residual,
        symmetrized_fft=np.abs(np.fft.fftshift(np.fft.fft2(fft_src))),
        coverage=den,
        shifts=shifts,
        symmetry_residual_norm=symmetry_residual_norm,
        n_ops=n_ops,
    )


def symmetrize_filter(
    image: np.ndarray,
    n_fold: int,
    *,
    mirror: bool = False,
    mirror_axis_deg: float = 0.0,
    register: bool = True,
    interpolation: str = "linear",
    strict_coverage: bool = False,
) -> np.ndarray:
    """Convenience: return only the symmetrized image.

    This is the entry point for the (future) ``symmetrize_fft``
    ProcessingState op, mirroring :func:`probeflow.processing.inverse_fft.
    inverse_fft_filter`.  ``n_fold=1`` without ``mirror`` returns a copy.
    """
    return symmetrize_image(
        np.asarray(image, dtype=np.float64), n_fold,
        mirror=mirror, mirror_axis_deg=mirror_axis_deg, register=register,
        interpolation=interpolation, strict_coverage=strict_coverage,
    ).result


def fold_axis_from_peaks(peaks_xy: np.ndarray, n_fold: int) -> float:
    """Lattice axis angle (degrees, mod ``360/n``) from Bragg peak offsets.

    ``peaks_xy`` is the ``(M, 2)`` array of (x, y) pixel offsets from DC
    returned by :func:`probeflow.processing.bragg.find_bragg_peaks_in_annulus`.
    Under an n-fold symmetry the peak angles are ``θ0 + k·360/n``; multiplying
    by ``n`` collapses them onto one direction, whose circular mean is robust
    to a missing or spurious peak.  Use the result to align
    ``mirror_axis_deg`` (or a GUI overlay) with the lattice.
    """
    pts = np.asarray(peaks_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        raise ValueError("peaks_xy must be a non-empty (M, 2) array")
    n = int(n_fold)
    if n < 1:
        raise ValueError("n_fold must be >= 1")
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    # Weight the circular mean by peak radius so a weak spurious pick near DC
    # cannot swing the axis.
    weights = np.hypot(pts[:, 0], pts[:, 1])
    z = np.sum(weights * np.exp(1j * n * theta))
    if abs(z) < 1e-12:
        raise ValueError("peak angles have no consistent n-fold axis")
    return math.degrees(np.angle(z) / n) % (360.0 / n)
