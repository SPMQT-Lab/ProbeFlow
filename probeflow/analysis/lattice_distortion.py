"""
Lattice distortion analysis for ProbeFlow.

GUI-independent.  All physical quantities are in nanometres.

The correction transform T satisfies:
    T @ m_a = i_a
    T @ m_b = i_b

where m_a, m_b are the measured real-space vectors and i_a, i_b are the
ideal real-space vectors.
"""

from __future__ import annotations

import math
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

    The full 2×2 matrix is the authoritative correction object.
    The scalar quantities (x_scale, y_over_x, shear, rotation_deg) are
    derived via QR decomposition and reported for display only.
    """

    measured: MeasuredLattice
    ideal: IdealLattice
    matrix: np.ndarray      # 2×2 affine T, dtype float64
    x_scale: float
    y_over_x: float
    shear: float
    rotation_deg: float


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

    return LatticeCorrection(
        measured=measured,
        ideal=ideal,
        matrix=T,
        x_scale=x_scale,
        y_over_x=y_over_x,
        shear=shear,
        rotation_deg=rotation_deg,
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
