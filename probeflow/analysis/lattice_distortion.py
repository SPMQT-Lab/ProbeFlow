"""
Lattice distortion analysis for ProbeFlow.

GUI-independent.  All physical quantities are in nanometres.

The correction transform T satisfies:
    T @ m_a = i_a
    T @ m_b = i_b

where m_a, m_b are the measured real-space vectors and i_a, i_b are the
ideal real-space vectors.

Polar decomposition
-------------------
T is further decomposed into T = R_polar @ S_polar (polar decomposition)
where R_polar is a pure rotation and S_polar is symmetric positive definite
(stretch + shear, no rotation).

When "preserve image orientation" is selected, only S_polar is applied to the
image.  This corrects anisotropic scale and shear while keeping the image
approximately aligned with the original scan axes.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass(frozen=True)
class IdealLattice:
    """Known ideal real-space lattice parameters."""

    a_nm: float
    b_nm: float
    angle_deg: float


@dataclass(frozen=True)
class MeasuredLattice:
    """Measured real-space lattice vectors (physical, in nm)."""

    a_nm: tuple[float, float]   # (ax, ay) in nm
    b_nm: tuple[float, float]   # (bx, by) in nm


@dataclass(frozen=True)
class LatticeCorrection:
    """
    Computed affine correction from measured to ideal lattice.

    The full 2×2 matrix (``matrix``) maps measured → ideal including any
    global rotation.  The scalar quantities are derived for display only.

    Polar decomposition ``matrix = rotation_matrix @ stretch_matrix``:
    - ``rotation_matrix`` is the pure-rotation component of the full transform.
    - ``stretch_matrix`` is the orientation-preserving (stretch + shear) part.
    - ``polar_rotation_deg`` is the angle of the stripped rigid rotation.

    When applying the correction with "preserve image orientation" enabled,
    use ``stretch_matrix`` rather than ``matrix``.
    """

    measured: MeasuredLattice
    ideal: IdealLattice
    matrix: np.ndarray           # 2×2 full T = I @ inv(M), dtype float64
    x_scale: float               # QR-derived display quantities
    y_over_x: float
    shear: float
    rotation_deg: float
    rotation_matrix: np.ndarray  # R from polar decomp T = R @ S
    stretch_matrix: np.ndarray   # S from polar decomp (symmetric, no rotation)
    polar_rotation_deg: float    # rigid rotation angle stripped by preserve mode


def ideal_vectors_nm(
    ideal: IdealLattice,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Construct ideal basis vectors in nm.

    Convention: i_a lies along +x; i_b is at ideal.angle_deg CCW from i_a.
    """
    angle_rad = math.radians(ideal.angle_deg)
    i_a: tuple[float, float] = (ideal.a_nm, 0.0)
    i_b: tuple[float, float] = (
        ideal.b_nm * math.cos(angle_rad),
        ideal.b_nm * math.sin(angle_rad),
    )
    return i_a, i_b


def compute_correction(
    measured: MeasuredLattice,
    ideal: IdealLattice,
) -> Union[LatticeCorrection, str]:
    """
    Compute the 2×2 affine matrix T = I @ inv(M) mapping measured to ideal.

    Also computes the polar decomposition T = R @ S, where R is a pure
    rotation and S is the orientation-preserving stretch/shear part.

    Returns a LatticeCorrection on success, or an error string if the
    measured lattice is too close to collinear.
    """
    m_a, m_b = measured.a_nm, measured.b_nm
    i_a, i_b = ideal_vectors_nm(ideal)

    M = np.array([[m_a[0], m_b[0]],
                  [m_a[1], m_b[1]]], dtype=float)
    I_mat = np.array([[i_a[0], i_b[0]],
                      [i_a[1], i_b[1]]], dtype=float)

    la = math.hypot(*m_a)
    lb = math.hypot(*m_b)
    det = float(np.linalg.det(M))
    tol = 1e-6 * la * lb

    if abs(det) < tol or la < 1e-12 or lb < 1e-12:
        return (
            "Measured lattice vectors are too close to collinear.\n"
            "Cannot compute linear correction."
        )

    T = I_mat @ np.linalg.inv(M)
    x_scale, y_over_x, shear, rotation_deg = _decompose_affine(T)
    R_polar, S_polar, polar_rotation_deg = _polar_decompose(T)

    return LatticeCorrection(
        measured=measured,
        ideal=ideal,
        matrix=T,
        x_scale=x_scale,
        y_over_x=y_over_x,
        shear=shear,
        rotation_deg=rotation_deg,
        rotation_matrix=R_polar,
        stretch_matrix=S_polar,
        polar_rotation_deg=polar_rotation_deg,
    )


def _decompose_affine(
    T: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Decompose a 2×2 affine matrix via QR to extract interpretable quantities.

    Returns (x_scale, y_over_x, shear, rotation_deg).

    T = Q @ R where Q is orthogonal (rotation) and R is upper-triangular
    (scale + shear).  Diagonal of R is sign-corrected to be positive.

    Note: these are display quantities derived from T.  The full matrix T
    is the authoritative correction object.
    """
    Q, R = np.linalg.qr(T)

    # Ensure positive diagonal elements in R
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs          # broadcast over columns
    R = (R.T * signs).T    # broadcast over rows

    x_scale = float(R[0, 0])
    y_scale = float(R[1, 1])
    shear_raw = float(R[0, 1])

    y_over_x = y_scale / x_scale if abs(x_scale) > 1e-12 else float("nan")
    shear = shear_raw / x_scale if abs(x_scale) > 1e-12 else float("nan")

    # Rotation angle from Q (first column of Q is [cos θ, sin θ])
    rotation_deg = math.degrees(math.atan2(Q[1, 0], Q[0, 0]))

    return x_scale, y_over_x, shear, rotation_deg


def _polar_decompose(
    T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Polar decomposition of T into (R, S, rotation_deg) where T = R @ S.

    Uses SVD: T = U @ diag(sigma) @ Vt, then R = U @ Vt and
    S = Vt.T @ diag(sigma) @ Vt.  Because sigma >= 0, S is always
    symmetric positive semi-definite.

    For non-reflective T (``det(T) >= 0``):
        - R is a proper rotation (``det(R) = +1``)
        - ``rotation_deg`` is the rotation angle of R
        - In "preserve image orientation" mode, applying S corrects
          stretch + shear while keeping the image axis-aligned.

    For reflective T (``det(T) < 0``):
        - R is improper orthogonal (``det(R) = -1``: rotation + flip)
        - S remains positive semi-definite (no reflection)
        - ``rotation_deg`` is returned as ``NaN`` because ``atan2`` on
          an improper orthogonal matrix does not yield a meaningful
          rotation angle.  In "preserve image orientation" mode the
          stretch is still well defined; the reflection in T is
          dropped along with the rigid rotation.
        - A ``UserWarning`` is emitted so callers can surface the
          reflection to the user (it usually signals swapped lattice
          vectors or a flipped scan direction).
    """
    U, sigma, Vt = np.linalg.svd(T)
    # S is built from the (non-negative) singular values: always PSD.
    S = Vt.T @ np.diag(sigma) @ Vt
    # R inherits the sign of det(T).  Do NOT flip a singular value to
    # force det(R) = +1: that would make S indefinite and silently
    # apply a reflection in "preserve image orientation" mode.
    R = U @ Vt
    if float(np.linalg.det(R)) < 0:
        warnings.warn(
            "Lattice correction T has a reflection component "
            "(det(T) < 0). The polar rotation angle is undefined; "
            "stretch_matrix remains positive semi-definite. "
            "Check measured lattice-vector ordering and scan direction.",
            UserWarning,
            stacklevel=2,
        )
        rotation_deg = float("nan")
    else:
        rotation_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return R, S, rotation_deg
