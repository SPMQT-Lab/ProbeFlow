"""Tests for probeflow.analysis.feature_lattice."""

from __future__ import annotations

import math

import numpy as np
import pytest

from probeflow.analysis.feature_lattice import (
    FeatureLatticeComparison,
    compare_features_to_lattice,
    default_match_radius,
)


def _square_lattice_pts(n: int, spacing: float, origin=(0.0, 0.0)):
    ox, oy = origin
    pts = []
    for i in range(n):
        for j in range(n):
            pts.append([ox + i * spacing, oy + j * spacing])
    return np.array(pts)


# ── default_match_radius ──────────────────────────────────────────────────────

def test_default_match_radius():
    r = default_match_radius((10.0, 0.0), (0.0, 8.0))
    assert abs(r - 0.35 * 8.0) < 1e-10


# ── perfect match ─────────────────────────────────────────────────────────────

def test_perfect_points_all_matched():
    pts = _square_lattice_pts(4, 10.0)
    result = compare_features_to_lattice(
        pts, (0.0, 0.0), (10.0, 0.0), (0.0, 10.0), match_radius_px=2.0,
    )
    assert result.n_matched == 16
    assert result.n_off_lattice == 0
    assert result.n_duplicate_sites == 0


def test_perfect_points_zero_rms():
    pts = _square_lattice_pts(3, 10.0)
    result = compare_features_to_lattice(
        pts, (0.0, 0.0), (10.0, 0.0), (0.0, 10.0), match_radius_px=2.0,
    )
    assert result.rms_displacement_px is not None
    assert result.rms_displacement_px < 1e-10


# ── noisy points ──────────────────────────────────────────────────────────────

def test_noisy_points_rms_matches_noise():
    rng = np.random.default_rng(0)
    sigma = 0.5
    base = _square_lattice_pts(5, 10.0)
    noisy = base + rng.normal(0, sigma, base.shape)
    result = compare_features_to_lattice(
        noisy, (0.0, 0.0), (10.0, 0.0), (0.0, 10.0),
        match_radius_px=3.0,
    )
    assert result.n_matched == 25
    assert result.rms_displacement_px is not None
    assert 0.2 < result.rms_displacement_px < 1.0


# ── off-lattice points ────────────────────────────────────────────────────────

def test_off_lattice_points_counted():
    on = _square_lattice_pts(3, 10.0)
    off = np.array([[55.0, 55.0], [57.0, 55.0]])   # far from any site
    pts = np.vstack([on, off])
    result = compare_features_to_lattice(
        pts, (0.0, 0.0), (10.0, 0.0), (0.0, 10.0),
        match_radius_px=2.0,
    )
    assert result.n_off_lattice == 2
    assert result.n_matched == 9


# ── duplicate sites ───────────────────────────────────────────────────────────

def test_duplicate_sites_counted():
    # Two points both mapping to site (0, 0).
    pts = np.array([[0.5, 0.0], [-0.3, 0.1]])  # both near origin site
    result = compare_features_to_lattice(
        pts, (0.0, 0.0), (10.0, 0.0), (0.0, 10.0),
        match_radius_px=2.0,
    )
    assert result.n_duplicate_sites >= 1


# ── occupancy ─────────────────────────────────────────────────────────────────

def test_occupancy_full_coverage():
    # 5×5 lattice exactly covering a 50×50 pixel image.
    pts = _square_lattice_pts(5, 10.0)
    result = compare_features_to_lattice(
        pts, (0.0, 0.0), (10.0, 0.0), (0.0, 10.0),
        match_radius_px=2.0,
        image_shape=(50, 50),
    )
    assert result.occupancy is not None
    assert 0.8 <= result.occupancy <= 1.0


def test_occupancy_none_without_shape():
    pts = _square_lattice_pts(3, 10.0)
    result = compare_features_to_lattice(
        pts, (0.0, 0.0), (10.0, 0.0), (0.0, 10.0),
        match_radius_px=2.0,
    )
    assert result.occupancy is None


def test_partial_occupancy():
    # Only half the sites occupied.
    pts = _square_lattice_pts(2, 20.0)   # 4 points on 4×4-site grid
    result = compare_features_to_lattice(
        pts, (0.0, 0.0), (10.0, 0.0), (0.0, 10.0),
        match_radius_px=2.0,
        image_shape=(40, 40),
    )
    assert result.occupancy is not None
    assert result.occupancy < 1.0


# ── singular lattice vectors ──────────────────────────────────────────────────

def test_singular_vectors_raise():
    pts = np.array([[1.0, 1.0]])
    with pytest.raises(ValueError, match="singular"):
        compare_features_to_lattice(
            pts, (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
            match_radius_px=1.0,
        )


# ── assignments integrity ─────────────────────────────────────────────────────

def test_assignments_count_matches_n_features():
    pts = _square_lattice_pts(3, 10.0)
    result = compare_features_to_lattice(
        pts, (0.0, 0.0), (10.0, 0.0), (0.0, 10.0),
        match_radius_px=2.0,
    )
    assert len(result.assignments) == result.n_features == 9


def test_single_point_handled():
    pts = np.array([[5.0, 5.0]])
    result = compare_features_to_lattice(
        pts, (0.0, 0.0), (10.0, 0.0), (0.0, 10.0),
        match_radius_px=2.0,
    )
    assert result.n_features == 1
