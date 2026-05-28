"""Tests for probeflow.analysis.pair_correlation."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from probeflow.analysis.pair_correlation import compute_pair_correlation

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")
    app = QApplication.instance()
    return app if app is not None else QApplication([])


def _square_lattice(n: int, spacing_m: float) -> np.ndarray:
    """Return (n×n, 2) points on a square lattice with given spacing."""
    xs = np.arange(n) * spacing_m
    ys = np.arange(n) * spacing_m
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def test_square_lattice_first_peak_near_spacing():
    spacing = 1e-9  # 1 nm
    pts = _square_lattice(8, spacing)
    area = (8 * spacing) ** 2
    result = compute_pair_correlation(pts, roi_area_m2=area)
    assert result.quality == "good"
    assert result.first_peak_m is not None
    assert abs(result.first_peak_m - spacing) < 0.15 * spacing


def test_nn_median_square_lattice():
    spacing = 2e-9
    pts = _square_lattice(6, spacing)
    area = (6 * spacing) ** 2
    result = compute_pair_correlation(pts, roi_area_m2=area)
    assert result.nearest_neighbour_median_m is not None
    assert abs(result.nearest_neighbour_median_m - spacing) < 0.05 * spacing


def test_random_points_quality_good():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1e-8, size=(50, 2))
    area = (1e-8) ** 2
    result = compute_pair_correlation(pts, roi_area_m2=area)
    assert result.quality == "good"
    assert result.n_points == 50


def test_few_points_quality_weak():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 1e-9, size=(8, 2))
    result = compute_pair_correlation(pts)
    assert result.quality == "weak"


def test_too_few_points_quality_failed():
    pts = np.array([[0.0, 0.0], [1e-9, 0.0]])
    result = compute_pair_correlation(pts)
    assert result.quality == "failed"
    assert len(result.r_m) == 0


def test_density_computed_when_area_given():
    pts = _square_lattice(5, 1e-9)
    area = (5e-9) ** 2
    result = compute_pair_correlation(pts, roi_area_m2=area)
    expected_density = 25.0 / area
    assert result.density_m2 is not None
    assert abs(result.density_m2 - expected_density) < 1e-3 * expected_density


def test_no_area_gives_none_density():
    pts = _square_lattice(5, 1e-9)
    result = compute_pair_correlation(pts)
    assert result.density_m2 is None


def test_g_r_normalised_to_one_for_random():
    """g(r) should average near 1 for a large random set when area is known."""
    rng = np.random.default_rng(123)
    pts = rng.uniform(0, 5e-8, size=(200, 2))
    area = (5e-8) ** 2
    result = compute_pair_correlation(pts, roi_area_m2=area)
    # Ignore first few bins (small-r noise) and last few (edge effects).
    mid = result.g_r[5:-5]
    if len(mid) > 0:
        assert 0.5 < float(np.mean(mid)) < 2.0


def test_r_m_shape_matches_g_r():
    pts = _square_lattice(5, 1e-9)
    result = compute_pair_correlation(pts)
    assert len(result.r_m) == len(result.g_r)


def test_custom_r_max_and_bin_width():
    pts = _square_lattice(5, 1e-9)
    result = compute_pair_correlation(pts, r_max_m=3e-9, bin_width_m=0.5e-9)
    assert result.r_m[-1] <= 3e-9 + 0.5e-9
    assert abs(result.r_m[1] - result.r_m[0] - 0.5e-9) < 1e-12


def test_message_contains_edge_correction_note():
    pts = _square_lattice(6, 1e-9)
    result = compute_pair_correlation(pts)
    assert "edge correction" in result.message.lower()


def test_dialog_measurement_context_preserves_bins_area_and_warning(qapp):
    from probeflow.gui.dialogs.pair_correlation import PairCorrelationDialog

    captured = []
    dlg = PairCorrelationDialog(
        {"Detected feature maxima": _square_lattice(5, 1e-9)},
        roi_area_m2=25e-18,
        pixel_size_x_m=1e-9,
        pixel_size_y_m=2e-9,
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
    dlg._rmax_sb.setValue(4.0)
    dlg._bw_sb.setValue(0.5)

    dlg._run()
    dlg._add_to_table()

    result = captured[0]
    assert result.context["point_source"] == "Detected feature maxima"
    assert result.context["source_path"] == "/tmp/scan.sxm"
    assert result.context["roi_area_m2"] == pytest.approx(25e-18)
    assert result.context["r_max_m"] == pytest.approx(4e-9)
    assert result.context["bin_width_m"] == pytest.approx(0.5e-9)
    assert result.context["edge_correction"] == "not_applied"
    assert "edge correction" in result.context["message"].lower()
    assert result.context["pixel_size_y_m"] == pytest.approx(2e-9)
    assert result.context["point_source_type"] == "feature_maxima"
    assert result.context["point_source_selection_scope"] == "roi"
    assert result.context["point_source_threshold_mode"] == "percentile"
    dlg.close()
    dlg.deleteLater()


def test_invalid_array_raises():
    with pytest.raises(ValueError):
        compute_pair_correlation(np.array([1.0, 2.0, 3.0]))


def test_coincident_points_return_failed_empty_result():
    pts = np.zeros((5, 2), dtype=float)

    result = compute_pair_correlation(pts)

    assert result.quality == "failed"
    assert result.r_m.size == 0
    assert result.g_r.size == 0
    assert result.first_peak_m is None
    assert "zero" in result.message.lower()
