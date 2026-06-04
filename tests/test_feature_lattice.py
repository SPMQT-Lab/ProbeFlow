"""Tests for probeflow.analysis.feature_lattice."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from probeflow.analysis.feature_lattice import (
    compare_features_to_lattice,
    default_match_radius,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")
    app = QApplication.instance()
    return app if app is not None else QApplication([])


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


def test_half_site_ties_do_not_use_bankers_rounding():
    pts = np.array([[5.0, 0.0], [15.0, 0.0], [-5.0, 0.0]])

    result = compare_features_to_lattice(
        pts, (0.0, 0.0), (10.0, 0.0), (0.0, 10.0), match_radius_px=6.0,
    )

    assert [a.site_ij for a in result.assignments] == [(1, 0), (2, 0), (-1, 0)]


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


def test_physical_displacement_uses_per_axis_pixel_calibration():
    pts = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    result = compare_features_to_lattice(
        pts,
        (0.0, 0.0),
        (10.0, 0.0),
        (0.0, 10.0),
        match_radius_px=2.0,
        pixel_size_x_m=1e-9,
        pixel_size_y_m=2e-9,
    )
    assert result.rms_displacement_m == pytest.approx(math.sqrt((1e-18 + 4e-18) / 2))
    assert result.mean_displacement_m == pytest.approx(1.5e-9)
    assert result.assignments[0].displacement_m == pytest.approx(1e-9)
    assert result.assignments[1].displacement_m == pytest.approx(2e-9)


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


def test_dialog_measurement_context_preserves_lattice_and_match_settings(qapp):
    from probeflow.gui.dialogs.feature_lattice_dialog import FeatureLatticeDialog

    captured = []
    dlg = FeatureLatticeDialog(
        {"Detected feature maxima": np.array([[0.5, 0.0], [10.0, 0.5]])},
        lattice_origin_px=(0.0, 0.0),
        a_px=(10.0, 0.0),
        b_px=(0.0, 12.0),
        pixel_size_x_m=1e-9,
        pixel_size_y_m=2e-9,
        image_shape=(48, 40),
        source_label="scan:Height",
        source_path="/tmp/scan.sxm",
        channel="Height",
        source_metadata={
            "Detected feature maxima": {
                "point_source_type": "feature_maxima",
                "selection_scope": "roi",
                "threshold_mode": "percentile",
            }
        },
        on_add_result=captured.append,
    )
    dlg._radius_sb.setValue(2.5)

    dlg._run()
    dlg._add_to_table()

    result = captured[0]
    assert result.context["point_source"] == "Detected feature maxima"
    assert result.context["source_path"] == "/tmp/scan.sxm"
    assert result.context["match_radius_px"] == pytest.approx(2.5)
    assert result.context["match_radius_mode"] == "manual"
    assert result.context["lattice_a_x_px"] == pytest.approx(10.0)
    assert result.context["lattice_b_y_px"] == pytest.approx(12.0)
    assert result.context["pixel_size_y_m"] == pytest.approx(2e-9)
    assert result.context["image_shape_y"] == 48
    assert result.context["occupancy_region"] == "image_bounds"
    assert result.context["point_source_type"] == "feature_maxima"
    assert result.context["point_source_selection_scope"] == "roi"
    assert result.context["point_source_threshold_mode"] == "percentile"
    dlg.close()
    dlg.deleteLater()
