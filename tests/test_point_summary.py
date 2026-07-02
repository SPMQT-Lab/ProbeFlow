"""Tests for the GUI-free point-pattern summary (Particle Statistics quick stats)."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.analysis.point_summary import (
    PointPatternSummary,
    nn_histogram_nm,
    summarize_point_pattern,
)


def _lattice_points_m(spacing_nm: float = 3.0, n_side: int = 3) -> np.ndarray:
    xs = (np.arange(n_side) + 1.0) * spacing_nm
    grid = np.array([[x, y] for y in xs for x in xs], dtype=float)
    return grid * 1e-9


def test_lattice_nn_stats_are_exact():
    points_m = _lattice_points_m(spacing_nm=3.0, n_side=3)
    summary = summarize_point_pattern(
        points_m,
        scan_range_m=(12e-9, 12e-9),
        image_shape=(12, 12),
    )
    assert summary.n_total == summary.n_in_region == 9
    for value in (summary.nn_min_nm, summary.nn_mean_nm, summary.nn_median_nm, summary.nn_max_nm):
        assert value == pytest.approx(3.0)
    assert len(summary.nn_distances_nm) == 9


def test_full_image_area_density_and_csr_expectation():
    points_m = _lattice_points_m(spacing_nm=2.0, n_side=2)  # 4 points
    summary = summarize_point_pattern(
        points_m,
        scan_range_m=(8e-9, 6e-9),
        image_shape=(6, 8),  # (ny, nx): anisotropy-safe (1 nm pixels here)
    )
    assert summary.area_nm2 == pytest.approx(48.0)
    assert summary.density_per_nm2 == pytest.approx(4.0 / 48.0)
    assert summary.expected_csr_nn_mean_nm == pytest.approx(0.5 / np.sqrt(4.0 / 48.0))


def test_anisotropic_pixels_mask_area():
    # 8 nm x 6 nm field on 4x12 px grid -> px 2.0 nm wide, 0.5 nm tall.
    mask = np.zeros((12, 4), dtype=bool)
    mask[:, :2] = True  # left half: 24 px * (2.0 * 0.5) nm^2 = 24 nm^2
    points_m = np.array([[1.0e-9, 3.0e-9], [3.0e-9, 3.0e-9], [6.0e-9, 3.0e-9]])
    summary = summarize_point_pattern(
        points_m,
        scan_range_m=(8e-9, 6e-9),
        image_shape=(12, 4),
        mask=mask,
        region_label="Active mask",
    )
    assert summary.n_total == 3
    assert summary.n_in_region == 2  # x=6 nm is in the right half
    assert summary.area_nm2 == pytest.approx(24.0)
    assert summary.density_per_nm2 == pytest.approx(2.0 / 24.0)
    assert summary.region_label == "Active mask"
    # NN computed on in-region points only.
    assert summary.nn_mean_nm == pytest.approx(2.0)


def test_point_at_exact_field_edge_is_outside_mask():
    mask = np.ones((4, 4), dtype=bool)
    points_m = np.array([[4.0e-9, 2.0e-9], [1.0e-9, 1.0e-9]])  # x = width exactly
    summary = summarize_point_pattern(
        points_m,
        scan_range_m=(4e-9, 4e-9),
        image_shape=(4, 4),
        mask=mask,
    )
    assert summary.n_in_region == 1


def test_zero_one_and_all_outside_points_never_raise():
    empty = summarize_point_pattern(
        np.empty((0, 2)), scan_range_m=(4e-9, 4e-9), image_shape=(4, 4)
    )
    assert empty.n_total == 0
    assert empty.nn_mean_nm is None
    assert len(empty.nn_distances_nm) == 0

    single = summarize_point_pattern(
        np.array([[1e-9, 1e-9]]), scan_range_m=(4e-9, 4e-9), image_shape=(4, 4)
    )
    assert single.n_in_region == 1
    assert single.nn_mean_nm is None

    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    outside = summarize_point_pattern(
        np.array([[3e-9, 3e-9], [2e-9, 2e-9]]),
        scan_range_m=(4e-9, 4e-9),
        image_shape=(4, 4),
        mask=mask,
    )
    assert outside.n_total == 2
    assert outside.n_in_region == 0
    assert outside.density_per_nm2 == 0.0 or outside.density_per_nm2 is None
    assert outside.nn_mean_nm is None


def test_no_calibration_gives_none_area_but_real_nn():
    points_m = _lattice_points_m(spacing_nm=5.0, n_side=2)
    summary = summarize_point_pattern(points_m, scan_range_m=None, image_shape=None)
    assert summary.area_nm2 is None
    assert summary.density_per_nm2 is None
    assert summary.expected_csr_nn_mean_nm is None
    assert summary.nn_mean_nm == pytest.approx(5.0)
    assert "calibration" in summary.message


def test_duplicate_points_report_zero_nn_min():
    points_m = np.array([[1e-9, 1e-9], [1e-9, 1e-9], [3e-9, 1e-9]])
    summary = summarize_point_pattern(
        points_m, scan_range_m=(4e-9, 4e-9), image_shape=(4, 4)
    )
    assert summary.nn_min_nm == pytest.approx(0.0)


def test_nn_histogram_counts_conserved_and_degenerate_input():
    rng = np.random.default_rng(0)
    distances = rng.uniform(1.0, 9.0, size=200)
    edges, counts = nn_histogram_nm(distances)
    assert counts.sum() == 200
    assert len(edges) == len(counts) + 1
    assert 6 <= len(counts) <= 24

    edges_d, counts_d = nn_histogram_nm(np.full(9, 3.0))
    assert counts_d.sum() == 9
    assert len(counts_d) >= 1
    assert edges_d[0] < 3.0 < edges_d[-1]

    edges_e, counts_e = nn_histogram_nm(np.empty(0))
    assert len(edges_e) == 0 and len(counts_e) == 0


def test_summary_is_frozen_dataclass():
    summary = summarize_point_pattern(
        np.array([[1e-9, 1e-9], [2e-9, 2e-9]]),
        scan_range_m=(4e-9, 4e-9),
        image_shape=(4, 4),
    )
    assert isinstance(summary, PointPatternSummary)
    with pytest.raises(Exception):
        summary.n_total = 5  # type: ignore[misc]
