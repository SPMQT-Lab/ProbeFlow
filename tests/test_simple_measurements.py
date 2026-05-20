"""Tests for probeflow.analysis.simple_measurements and roi_statistics."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from probeflow.analysis.simple_measurements import (
    _fmt_m,
    measure_angle_between_lines,
    measure_line_distance,
)
from probeflow.analysis.roi_statistics import compute_roi_statistics


# ── Helpers ───────────────────────────────────────────────────────────────────

def _line_roi(x1, y1, x2, y2, name="L1", roi_id="r1"):
    g = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return types.SimpleNamespace(kind="line", geometry=g, name=name, id=roi_id)


def _bad_roi():
    return types.SimpleNamespace(kind="rectangle", geometry={}, name="R", id="r0")


# ── _fmt_m ────────────────────────────────────────────────────────────────────

def test_fmt_m_pm():
    v, u = _fmt_m(1e-12)
    assert u == "pm"
    assert abs(v - 1.0) < 1e-9


def test_fmt_m_angstrom():
    v, u = _fmt_m(1e-10)
    assert u == "Å"
    assert abs(v - 1.0) < 1e-9


def test_fmt_m_nm():
    v, u = _fmt_m(1e-9)
    assert u == "nm"
    assert abs(v - 1.0) < 1e-9


# ── measure_line_distance ─────────────────────────────────────────────────────

def test_distance_horizontal():
    roi = _line_roi(0, 0, 10, 0)
    px = 1e-10  # 1 Å per pixel
    result = measure_line_distance(roi, px, px)
    assert result.kind == "distance"
    assert math.isclose(result.values["length_m"], 10 * px, rel_tol=1e-9)
    assert math.isclose(result.values["angle_deg"], 0.0, abs_tol=1e-9)
    assert result.values["dy_m"] == 0.0


def test_distance_vertical():
    roi = _line_roi(0, 0, 0, 10)
    px = 1e-10
    result = measure_line_distance(roi, px, px)
    assert math.isclose(result.values["length_m"], 10 * px, rel_tol=1e-9)
    assert math.isclose(result.values["angle_deg"], 90.0, abs_tol=1e-9)


def test_distance_diagonal():
    roi = _line_roi(0, 0, 3, 4)
    px = 1e-10
    result = measure_line_distance(roi, px, px)
    assert math.isclose(result.values["length_m"], 5 * px, rel_tol=1e-9)
    assert math.isclose(result.values["angle_deg"],
                        math.degrees(math.atan2(4, 3)), rel_tol=1e-6)


def test_distance_asymmetric_pixels():
    roi = _line_roi(0, 0, 1, 0)
    result = measure_line_distance(roi, 2e-10, 1e-10)
    assert math.isclose(result.values["dx_m"], 2e-10, rel_tol=1e-9)


def test_distance_bad_roi():
    with pytest.raises(ValueError, match="line ROI"):
        measure_line_distance(_bad_roi(), 1e-10, 1e-10)


def test_distance_summary_contains_units():
    roi = _line_roi(0, 0, 10, 0)
    result = measure_line_distance(roi, 1e-10, 1e-10)
    summary = str(result.context["summary"])
    assert "Å" in summary or "nm" in summary or "pm" in summary


def test_distance_uses_roi_name_as_notes():
    roi = _line_roi(0, 0, 5, 0, name="my_line")
    result = measure_line_distance(roi, 1e-10, 1e-10)
    assert "my_line" in result.notes


def test_distance_measurement_id():
    roi = _line_roi(0, 0, 1, 0)
    result = measure_line_distance(roi, 1e-10, 1e-10, measurement_id="M42")
    assert result.measurement_id == "M42"


# ── measure_angle_between_lines ───────────────────────────────────────────────

def test_angle_perpendicular():
    h = _line_roi(0, 0, 10, 0, name="H")
    v = _line_roi(0, 0, 0, 10, name="V")
    result = measure_angle_between_lines(h, v, 1e-10, 1e-10)
    assert math.isclose(result.values["angle_deg"], 90.0, abs_tol=1e-6)


def test_angle_parallel_is_zero():
    a = _line_roi(0, 0, 10, 0, name="A")
    b = _line_roi(0, 5, 10, 5, name="B")
    result = measure_angle_between_lines(a, b, 1e-10, 1e-10)
    assert math.isclose(result.values["angle_deg"], 0.0, abs_tol=1e-6)


def test_angle_45_degrees():
    a = _line_roi(0, 0, 1, 0, name="A")
    b = _line_roi(0, 0, 1, 1, name="B")
    result = measure_angle_between_lines(a, b, 1e-10, 1e-10)
    assert math.isclose(result.values["angle_deg"], 45.0, abs_tol=1e-6)


def test_angle_always_acute():
    a = _line_roi(0, 0, 10, 0)
    b = _line_roi(0, 0, -1, -10)  # obtuse angle to a
    result = measure_angle_between_lines(a, b, 1e-10, 1e-10)
    assert result.values["angle_deg"] <= 90.0


def test_angle_bad_roi():
    h = _line_roi(0, 0, 10, 0)
    with pytest.raises(ValueError, match="roi_b must be a line"):
        measure_angle_between_lines(h, _bad_roi(), 1e-10, 1e-10)


def test_angle_zero_length_raises():
    a = _line_roi(0, 0, 0, 0)
    b = _line_roi(0, 0, 1, 0)
    with pytest.raises(ValueError, match="zero length"):
        measure_angle_between_lines(a, b, 1e-10, 1e-10)


def test_angle_summary_contains_names():
    a = _line_roi(0, 0, 1, 0, name="LineA")
    b = _line_roi(0, 0, 0, 1, name="LineB")
    result = measure_angle_between_lines(a, b, 1e-10, 1e-10)
    assert "LineA" in result.context["summary"]
    assert "LineB" in result.context["summary"]


# ── compute_roi_statistics ────────────────────────────────────────────────────

def _uniform_image(value=1e-10, shape=(10, 10)):
    return np.full(shape, value, dtype=float)


def test_roi_stats_uniform():
    img = _uniform_image(1e-10)
    mask = np.ones((10, 10), dtype=bool)
    result = compute_roi_statistics(
        img, mask,
        pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
        z_unit="m",
    )
    assert result.kind == "roi_stats"
    assert math.isclose(result.values["mean_height"], 1e-10, rel_tol=1e-9)
    assert math.isclose(result.values["rms_roughness"], 0.0, abs_tol=1e-15)
    assert math.isclose(result.values["peak_to_peak"], 0.0, abs_tol=1e-15)


def test_roi_stats_area():
    img = _uniform_image()
    mask = np.ones((4, 5), dtype=bool)
    result = compute_roi_statistics(
        img[:4, :5], mask,
        pixel_size_x_m=2e-10, pixel_size_y_m=3e-10,
        z_unit="m",
    )
    expected_nm2 = 20 * 2e-10 * 3e-10 * 1e18
    assert math.isclose(result.values["area_nm2"], expected_nm2, rel_tol=1e-9)


def test_roi_stats_n_pixels():
    img = np.ones((10, 10))
    mask = np.zeros((10, 10), dtype=bool)
    mask[:3, :3] = True
    result = compute_roi_statistics(
        img, mask,
        pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
        z_unit="m",
    )
    assert result.values["n_finite_pixels"] == 9


def test_roi_stats_rms_roughness():
    rng = np.random.default_rng(0)
    img = rng.standard_normal((50, 50)) * 1e-11
    mask = np.ones((50, 50), dtype=bool)
    result = compute_roi_statistics(
        img, mask,
        pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
        z_unit="m",
    )
    finite = img[mask]
    expected_rms = float(np.sqrt(np.mean((finite - np.mean(finite)) ** 2)))
    assert math.isclose(result.values["rms_roughness"], expected_rms, rel_tol=1e-6)


def test_roi_stats_nan_pixels_excluded():
    img = np.ones((5, 5))
    img[2, 2] = float("nan")
    mask = np.ones((5, 5), dtype=bool)
    result = compute_roi_statistics(
        img, mask,
        pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
        z_unit="m",
    )
    assert result.values["n_finite_pixels"] == 24


def test_roi_stats_all_nan_raises():
    img = np.full((5, 5), float("nan"))
    mask = np.ones((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="no finite"):
        compute_roi_statistics(
            img, mask,
            pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
            z_unit="m",
        )


def test_roi_stats_shape_mismatch_raises():
    img = np.ones((5, 5))
    mask = np.ones((4, 5), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        compute_roi_statistics(
            img, mask,
            pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
            z_unit="m",
        )


def test_roi_stats_summary_contains_area_and_mean():
    img = _uniform_image()
    mask = np.ones((10, 10), dtype=bool)
    result = compute_roi_statistics(
        img, mask,
        pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
        z_unit="m",
    )
    summary = str(result.context["summary"])
    assert "Area" in summary
    assert "Mean" in summary
    assert "RMS" in summary


def test_roi_stats_non_metre_z_unit():
    img = np.ones((5, 5)) * 42.0
    mask = np.ones((5, 5), dtype=bool)
    result = compute_roi_statistics(
        img, mask,
        pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
        z_unit="nA",
    )
    assert result.z_unit == "nA"
    assert math.isclose(result.values["mean_height"], 42.0, rel_tol=1e-9)
