"""Tests for probeflow.analysis.lattice.average_unit_cell."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("sklearn")

from probeflow.analysis.lattice import (
    LatticeParams, average_unit_cell, extract_lattice,
)


def _synthetic_lattice(a_px, b_px, shape=(256, 256), sigma_px=2.0, amp=5.0):
    Ny, Nx = shape
    Y, X = np.mgrid[:Ny, :Nx]
    img = np.zeros(shape, dtype=np.float64)
    ax, ay = a_px
    bx, by = b_px
    cx, cy = Nx / 2, Ny / 2
    n = 20
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = cx + i * ax + j * bx
            y = cy + i * ay + j * by
            if 8 < x < Nx - 8 and 8 < y < Ny - 8:
                img += amp * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma_px ** 2))
    return img


class TestAverageUnitCell:
    def test_square_lattice_yields_motif(self):
        a = (24.0, 0.0)
        b = (0.0, 24.0)
        img = _synthetic_lattice(a, b, shape=(256, 256))
        lat = extract_lattice(img, pixel_size_m=1e-10, params=LatticeParams())
        cell = average_unit_cell(img, lat, oversample=1.5, border_margin_px=4)

        # Should aggregate many cells (well over 10 in a 256x256 grid).
        assert cell.n_cells >= 10
        # Output is a 2-D square array with side derived from the recovered
        # primitive vectors (which may not match the synthetic ones exactly,
        # since SIFT can pick a smaller equivalent primitive cell).
        assert cell.avg_cell.ndim == 2
        h, w = cell.avg_cell.shape
        assert h == w and h >= 8
        # The averaged cell should preserve the bright Gaussian peak —
        # max should clearly exceed the mean for a non-trivial motif.
        avg = cell.avg_cell
        assert avg.max() > avg.mean() + 0.1 * (avg.max() - avg.min())

    def test_no_interior_cells_raises(self):
        # Hand-build a LatticeResult with vectors larger than the image so no
        # interior cell fits — bypasses SIFT to test only the averaging logic.
        from probeflow.analysis.lattice import LatticeResult
        img = np.zeros((64, 64), dtype=np.float64)
        lat = LatticeResult(
            a_vector_m=(2e-8, 0.0),
            b_vector_m=(0.0, 2e-8),
            a_length_m=2e-8,
            b_length_m=2e-8,
            gamma_deg=90.0,
            a_vector_px=(200.0, 0.0),
            b_vector_px=(0.0, 200.0),
            n_keypoints=0,
            n_keypoints_used=0,
            keypoints_xy_px=[],
            cluster_labels=[],
            primary_cluster=-1,
            pixel_size_m=1e-10,
        )
        with pytest.raises(RuntimeError):
            average_unit_cell(img, lat, border_margin_px=2)
