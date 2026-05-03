"""Tests for probeflow.analysis.lattice — SIFT-based lattice extraction."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("sklearn")

from probeflow.analysis.lattice import LatticeParams, extract_lattice


def _synthetic_lattice(a_px, b_px, shape=(256, 256), sigma_px=2.0, amp=5.0):
    """Place gaussian 'atoms' on a 2-D lattice defined by (a, b)."""
    Ny, Nx = shape
    Y, X = np.mgrid[:Ny, :Nx]
    img = np.zeros(shape, dtype=np.float64)
    # Generate lattice points inside the image
    ax, ay = a_px
    bx, by = b_px
    cx, cy = Nx / 2, Ny / 2
    # How many repeats fit
    n = 20
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = cx + i * ax + j * bx
            y = cy + i * ay + j * by
            if 8 < x < Nx - 8 and 8 < y < Ny - 8:
                img += amp * np.exp(
                    -((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma_px ** 2)
                )
    return img


class TestExtractLattice:
    def test_square_lattice_recovers_a_b(self):
        a = (20.0, 0.0)
        b = (0.0, 20.0)
        img = _synthetic_lattice(a, b, shape=(256, 256))
        res = extract_lattice(img, pixel_size_m=1e-10,
                              params=LatticeParams(cluster_kp_low=2,
                                                   cluster_kp_high=4,
                                                   cluster_kNN_low=4,
                                                   cluster_kNN_high=12))
        # |a|, |b| ≈ 20 px = 2 nm
        assert 1.6e-9 < res.a_length_m < 2.4e-9
        assert 1.6e-9 < res.b_length_m < 2.4e-9
        # γ ≈ 90°
        assert 85.0 < res.gamma_deg < 95.0

    def test_hex_lattice_gamma_60(self):
        # Hexagonal lattice: a=(20, 0), b=(10, 10*sqrt(3))
        a = (20.0, 0.0)
        b = (10.0, 10.0 * np.sqrt(3))
        img = _synthetic_lattice(a, b, shape=(256, 256))
        res = extract_lattice(img, pixel_size_m=1e-10,
                              params=LatticeParams(cluster_kp_low=2,
                                                   cluster_kp_high=4,
                                                   cluster_kNN_low=6,
                                                   cluster_kNN_high=18))
        # γ should be 60° (or 120° depending on the pair chosen — both are valid).
        gamma = res.gamma_deg
        assert abs(gamma - 60.0) < 10.0 or abs(gamma - 120.0) < 10.0

    def test_rejects_blank_image(self):
        arr = np.zeros((64, 64), dtype=np.float64)
        with pytest.raises(RuntimeError):
            extract_lattice(arr, pixel_size_m=1e-10)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            extract_lattice(np.zeros((4, 4, 4)), pixel_size_m=1e-10)

    def test_rejects_bad_pixel_size(self):
        with pytest.raises(ValueError):
            extract_lattice(np.zeros((32, 32)), pixel_size_m=0.0)

    def test_to_dict(self):
        a = (20.0, 0.0)
        b = (0.0, 20.0)
        img = _synthetic_lattice(a, b, shape=(256, 256))
        res = extract_lattice(img, pixel_size_m=1e-10,
                              params=LatticeParams(cluster_kp_low=2,
                                                   cluster_kp_high=4,
                                                   cluster_kNN_low=4,
                                                   cluster_kNN_high=12))
        d = res.to_dict()
        assert "a_length_m" in d
        assert "gamma_deg" in d
