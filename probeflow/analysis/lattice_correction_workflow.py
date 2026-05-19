"""GUI-independent helpers for applying lattice correction results."""

from __future__ import annotations

from typing import Any

import numpy as np

from probeflow.analysis.lattice_distortion import LatticeCorrection


def lattice_correction_matrix_px(
    correction: LatticeCorrection,
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    preserve_orientation: bool,
) -> np.ndarray | None:
    """Return the image pixel-space matrix for a lattice correction."""
    matrix_nm = correction.stretch_matrix if preserve_orientation else correction.matrix
    return _nm_matrix_to_px_matrix(
        matrix_nm,
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
    )


def lattice_correction_operation_params(
    correction: LatticeCorrection,
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    expand_canvas: bool,
    interpolation: str,
    fill_mode: str,
    preserve_orientation: bool,
) -> dict[str, Any] | None:
    """Build provenance parameters for an applied lattice correction."""
    applied_matrix = lattice_correction_matrix_px(
        correction,
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
        preserve_orientation=preserve_orientation,
    )
    full_matrix = _nm_matrix_to_px_matrix(
        correction.matrix,
        pixel_size_x_m=pixel_size_x_m,
        pixel_size_y_m=pixel_size_y_m,
    )
    if applied_matrix is None or full_matrix is None:
        return None
    return {
        "matrix": applied_matrix.tolist(),
        "full_matrix": full_matrix.tolist(),
        "expand_canvas": bool(expand_canvas),
        "interpolation": interpolation,
        "fill_mode": fill_mode,
        "preserve_orientation": bool(preserve_orientation),
        "polar_rotation_deg": float(correction.polar_rotation_deg),
        "measured_a_nm": [float(v) for v in correction.measured.a_nm],
        "measured_b_nm": [float(v) for v in correction.measured.b_nm],
        "ideal_a_nm": float(correction.ideal.a_nm),
        "ideal_b_nm": float(correction.ideal.b_nm),
        "ideal_angle_deg": float(correction.ideal.angle_deg),
    }


def _nm_matrix_to_px_matrix(
    matrix_nm: np.ndarray,
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> np.ndarray | None:
    px_nm_x = float(pixel_size_x_m) * 1e9
    px_nm_y = float(pixel_size_y_m) * 1e9
    if px_nm_x <= 0 or px_nm_y <= 0:
        return None
    scale = np.diag([1.0 / px_nm_x, 1.0 / px_nm_y])
    scale_inv = np.diag([px_nm_x, px_nm_y])
    return scale @ matrix_nm @ scale_inv
