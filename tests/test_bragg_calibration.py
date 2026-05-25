"""Tests for the three FFT piezo-calibration pure functions:
find_bragg_peaks_in_annulus, fit_axis_aligned_ellipse, piezo_correction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from probeflow.processing.filters import (
    find_bragg_peaks_in_annulus,
    find_bragg_peaks_in_q_annulus,
    fit_axis_aligned_ellipse,
    piezo_correction,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_synthetic_fft(N: int, peaks_xy: list[tuple[float, float]], sigma: float = 2.0) -> np.ndarray:
    """Create a blank N×N FFT magnitude array with Gaussian blobs at given
    (x, y) pixel offsets from the centre."""
    cy, cx = N / 2.0, N / 2.0
    mag = np.zeros((N, N), dtype=np.float64)
    yy, xx = np.mgrid[:N, :N]
    for px, py in peaks_xy:
        mag += np.exp(-((xx - (cx + px)) ** 2 + (yy - (cy + py)) ** 2) / (2 * sigma ** 2))
    return mag


def _make_synthetic_fft_q(
    qx: np.ndarray,
    qy: np.ndarray,
    peaks_q: list[tuple[float, float]],
    sigma_q: float = 0.015,
) -> np.ndarray:
    qxx, qyy = np.meshgrid(qx, qy)
    mag = np.zeros((qy.size, qx.size), dtype=np.float64)
    for px, py in peaks_q:
        mag += np.exp(-((qxx - px) ** 2 + (qyy - py) ** 2) / (2 * sigma_q ** 2))
    return mag


def _hex_peaks(r: float) -> list[tuple[float, float]]:
    """Six ideal hex peaks at radius r."""
    return [(r * math.cos(k * math.pi / 3), r * math.sin(k * math.pi / 3)) for k in range(6)]


def _square_peaks(r: float) -> list[tuple[float, float]]:
    """Four ideal square peaks at radius r."""
    return [(r, 0.0), (0.0, r), (-r, 0.0), (0.0, -r)]


# ── find_bragg_peaks_in_annulus ────────────────────────────────────────────────

class TestFindBraggPeaksInAnnulus:
    def test_hex_six_peaks_at_known_radius(self):
        N = 256
        r = 40.0
        fft_mag = _make_synthetic_fft(N, _hex_peaks(r))
        peaks = find_bragg_peaks_in_annulus(fft_mag, r_predicted_px=r, expected_count=6)
        assert peaks.shape[1] == 2
        assert len(peaks) == 6
        # Every detected peak should be within 1 pixel of some expected peak
        for px, py in peaks:
            dists = [math.hypot(px - ex, py - ey) for ex, ey in _hex_peaks(r)]
            assert min(dists) < 1.0, f"No expected hex peak near ({px:.2f}, {py:.2f})"

    def test_square_four_peaks_at_known_radius(self):
        N = 256
        r = 35.0
        fft_mag = _make_synthetic_fft(N, _square_peaks(r))
        peaks = find_bragg_peaks_in_annulus(fft_mag, r_predicted_px=r, expected_count=4)
        assert len(peaks) == 4
        for px, py in peaks:
            dists = [math.hypot(px - ex, py - ey) for ex, ey in _square_peaks(r)]
            assert min(dists) < 1.0

    def test_annulus_too_narrow_returns_fewer_than_expected(self):
        N = 256
        r = 40.0
        fft_mag = _make_synthetic_fft(N, _hex_peaks(r))
        # width_frac=0.001 means the annulus is only 0.08 px wide — highly unlikely to
        # contain any pixel centre exactly at r
        peaks = find_bragg_peaks_in_annulus(fft_mag, r_predicted_px=r, width_frac=0.001, expected_count=6)
        assert len(peaks) < 6  # fewer than expected — no error raised

    def test_wrong_radius_returns_empty(self):
        N = 256
        r = 40.0
        fft_mag = _make_synthetic_fft(N, _hex_peaks(r))
        # Search at r=80, far from the actual peaks at r=40
        peaks = find_bragg_peaks_in_annulus(fft_mag, r_predicted_px=80.0, width_frac=0.10, expected_count=6)
        assert len(peaks) == 0

    def test_returns_ndarray_shape(self):
        N = 128
        fft_mag = _make_synthetic_fft(N, _hex_peaks(20.0))
        peaks = find_bragg_peaks_in_annulus(fft_mag, r_predicted_px=20.0, expected_count=6)
        assert isinstance(peaks, np.ndarray)
        assert peaks.ndim == 2
        assert peaks.shape[1] == 2

    def test_nonpositive_r_returns_empty(self):
        N = 64
        mag = np.ones((N, N))
        peaks = find_bragg_peaks_in_annulus(mag, r_predicted_px=0.0)
        assert peaks.shape == (0, 2)

    def test_non_2d_input_raises(self):
        with pytest.raises(ValueError):
            find_bragg_peaks_in_annulus(np.ones(64), r_predicted_px=10.0)


class TestFindBraggPeaksInQAnnulus:
    def test_hex_six_peaks_at_known_q_radius(self):
        qx = np.linspace(-1.0, 1.0, 257)
        qy = np.linspace(-1.0, 1.0, 257)
        q = 0.35
        expected = _hex_peaks(q)
        fft_mag = _make_synthetic_fft_q(qx, qy, expected)
        peaks = find_bragg_peaks_in_q_annulus(fft_mag, qx, qy, q, expected_count=6)
        assert len(peaks) == 6
        for px, py in peaks:
            assert min(math.hypot(px - ex, py - ey) for ex, ey in expected) < 0.02

    def test_square_four_peaks_at_known_q_radius(self):
        qx = np.linspace(-1.0, 1.0, 257)
        qy = np.linspace(-1.0, 1.0, 257)
        q = 0.30
        expected = _square_peaks(q)
        fft_mag = _make_synthetic_fft_q(qx, qy, expected)
        peaks = find_bragg_peaks_in_q_annulus(fft_mag, qx, qy, q, expected_count=4)
        assert len(peaks) == 4
        for px, py in peaks:
            assert min(math.hypot(px - ex, py - ey) for ex, ey in expected) < 0.02

    def test_vertical_origin_line_is_not_selected_as_peaks(self):
        qx = np.linspace(-1.0, 1.0, 257)
        qy = np.linspace(-1.0, 1.0, 257)
        q = 0.35
        expected = _hex_peaks(q)
        qxx, _qyy = np.meshgrid(qx, qy)
        fft_mag = _make_synthetic_fft_q(qx, qy, expected)
        fft_mag += 3.0 * np.exp(-(qxx ** 2) / (2 * 0.004 ** 2))

        peaks = find_bragg_peaks_in_q_annulus(fft_mag, qx, qy, q, expected_count=6)

        assert len(peaks) == 6
        assert not any(abs(px) < 0.02 and abs(abs(py) - q) < 0.04 for px, py in peaks)

    def test_contaminated_true_sector_returns_fewer_not_false_line_peaks(self):
        qx = np.linspace(-1.0, 1.0, 257)
        qy = np.linspace(-1.0, 1.0, 257)
        q = 0.30
        qxx, _qyy = np.meshgrid(qx, qy)
        fft_mag = _make_synthetic_fft_q(qx, qy, _square_peaks(q))
        fft_mag += 3.0 * np.exp(-(qxx ** 2) / (2 * 0.004 ** 2))

        peaks = find_bragg_peaks_in_q_annulus(fft_mag, qx, qy, q, expected_count=4)

        assert 0 < len(peaks) < 4
        assert not any(abs(px) < 0.02 for px, _py in peaks)

    def test_non_square_q_axes_detect_physical_circle(self):
        qx = np.linspace(-1.0, 1.0, 257)
        qy = np.linspace(-2.0, 2.0, 257)
        q = 0.45
        expected = _hex_peaks(q)
        fft_mag = _make_synthetic_fft_q(qx, qy, expected, sigma_q=0.018)
        peaks = find_bragg_peaks_in_q_annulus(fft_mag, qx, qy, q, expected_count=6)
        assert len(peaks) == 6
        for px, py in peaks:
            assert math.hypot(px, py) == pytest.approx(q, abs=0.03)


# ── fit_axis_aligned_ellipse ───────────────────────────────────────────────────

def _ellipse_points(r_x: float, r_y: float, n: int = 12, noise: float = 0.0,
                     seed: int = 0) -> np.ndarray:
    """Sample n points uniformly distributed on the ellipse."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = r_x * np.cos(angles)
    y = r_y * np.sin(angles)
    if noise > 0:
        rng = np.random.RandomState(seed)
        x += rng.normal(0, noise, n)
        y += rng.normal(0, noise, n)
    return np.column_stack([x, y])


class TestFitAxisAlignedEllipse:
    def test_perfect_ellipse_r_x_50_r_y_60(self):
        pts = _ellipse_points(50.0, 60.0, n=12)
        r_x, r_y, rms = fit_axis_aligned_ellipse(pts)
        assert r_x == pytest.approx(50.0, abs=0.1)
        assert r_y == pytest.approx(60.0, abs=0.1)
        assert rms < 0.01  # near-zero residual

    def test_circle_special_case(self):
        # r_x == r_y → unit circle scaled
        pts = _ellipse_points(45.0, 45.0, n=8)
        r_x, r_y, rms = fit_axis_aligned_ellipse(pts)
        assert r_x == pytest.approx(45.0, abs=0.1)
        assert r_y == pytest.approx(45.0, abs=0.1)
        assert rms < 0.01

    def test_noisy_ellipse_recovers_axes_within_few_percent(self):
        pts = _ellipse_points(50.0, 60.0, n=12, noise=1.0, seed=42)
        r_x, r_y, rms = fit_axis_aligned_ellipse(pts)
        assert r_x == pytest.approx(50.0, rel=0.05)
        assert r_y == pytest.approx(60.0, rel=0.05)
        assert rms > 0.0  # nonzero residual due to noise

    def test_noisy_rms_scales_with_noise(self):
        # Higher noise → higher RMS
        pts_lo = _ellipse_points(50.0, 60.0, n=24, noise=0.5, seed=7)
        pts_hi = _ellipse_points(50.0, 60.0, n=24, noise=5.0, seed=7)
        _, _, rms_lo = fit_axis_aligned_ellipse(pts_lo)
        _, _, rms_hi = fit_axis_aligned_ellipse(pts_hi)
        assert rms_hi > rms_lo

    def test_fewer_than_three_points_raises(self):
        pts = _ellipse_points(50.0, 60.0, n=2)
        with pytest.raises(ValueError, match="at least 3"):
            fit_axis_aligned_ellipse(pts)

    def test_two_points_raises(self):
        with pytest.raises(ValueError):
            fit_axis_aligned_ellipse(np.array([[1.0, 0.0], [0.0, 1.0]]))

    def test_one_point_raises(self):
        with pytest.raises(ValueError):
            fit_axis_aligned_ellipse(np.array([[1.0, 0.0]]))

    def test_zero_points_raises(self):
        with pytest.raises(ValueError):
            fit_axis_aligned_ellipse(np.empty((0, 2)))

    def test_degenerate_all_on_x_axis_raises(self):
        # All y == 0: v = 1/r_y² is unconstrained → fit returns v ≤ 0
        pts = np.column_stack([np.linspace(10, 50, 5), np.zeros(5)])
        with pytest.raises(ValueError):
            fit_axis_aligned_ellipse(pts)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            fit_axis_aligned_ellipse(np.ones((5, 3)))  # 3 columns

    def test_returns_positive_axes(self):
        pts = _ellipse_points(30.0, 40.0, n=8)
        r_x, r_y, rms = fit_axis_aligned_ellipse(pts)
        assert r_x > 0
        assert r_y > 0
        assert rms >= 0


# ── piezo_correction ───────────────────────────────────────────────────────────

class TestPiezoCorrection:
    def test_basic_x_correction(self):
        # r_x_obs < r_predicted → c_x_new < c_x_current
        cx_new, cy_new = piezo_correction(
            r_x_obs=45.0, r_y_obs=50.0, r_predicted=50.0,
            c_x_current=100.0, c_y_current=100.0,
        )
        assert cx_new < 100.0
        assert cy_new == pytest.approx(100.0)

    def test_basic_y_correction(self):
        # r_y_obs > r_predicted → c_y_new > c_y_current
        cx_new, cy_new = piezo_correction(
            r_x_obs=50.0, r_y_obs=55.0, r_predicted=50.0,
            c_x_current=96.52, c_y_current=96.52,
        )
        assert cy_new > 96.52
        assert cx_new == pytest.approx(96.52)

    def test_sign_when_r_obs_less_than_predicted(self):
        """If r_x_obs < r_predicted, the corrected piezo is smaller."""
        cx_new, _ = piezo_correction(0.994, 1.0, 1.0, 96.52, 96.52)
        assert cx_new < 96.52

    def test_worked_example_from_brief(self):
        # c_x_old=96.52, r_x_obs/r_pred=0.9946 → c_x_new ≈ 96.00
        cx_new, _ = piezo_correction(
            r_x_obs=0.9946, r_y_obs=1.0, r_predicted=1.0,
            c_x_current=96.52, c_y_current=96.52,
        )
        assert cx_new == pytest.approx(96.52 * 0.9946, rel=1e-6)

    def test_no_correction_when_r_obs_equals_r_predicted(self):
        cx_new, cy_new = piezo_correction(1.0, 1.0, 1.0, 96.52, 88.30)
        assert cx_new == pytest.approx(96.52)
        assert cy_new == pytest.approx(88.30)

    def test_round_trip_stability(self):
        """Applying the correction to already-corrected values is a no-op."""
        cx_old, cy_old = 100.0, 100.0
        cx_new, cy_new = piezo_correction(45.0, 55.0, 50.0, cx_old, cy_old)
        # Treat cx_new as the "new" current piezo, r_x_obs=r_predicted → no change
        cx_final, cy_final = piezo_correction(50.0, 50.0, 50.0, cx_new, cy_new)
        assert cx_final == pytest.approx(cx_new)
        assert cy_final == pytest.approx(cy_new)

    def test_unit_agnostic_same_result_different_units(self):
        # Scaling all radii by the same factor should not change the correction.
        cx1, cy1 = piezo_correction(45.0, 50.0, 50.0, 96.52, 96.52)
        cx2, cy2 = piezo_correction(4.5, 5.0, 5.0, 96.52, 96.52)
        assert cx1 == pytest.approx(cx2)
        assert cy1 == pytest.approx(cy2)

    # ── invalid inputs ────────────────────────────────────────────────────────

    def test_r_x_obs_zero_raises(self):
        with pytest.raises(ValueError, match="r_x_obs"):
            piezo_correction(0.0, 1.0, 1.0, 1.0, 1.0)

    def test_r_y_obs_negative_raises(self):
        with pytest.raises(ValueError, match="r_y_obs"):
            piezo_correction(1.0, -1.0, 1.0, 1.0, 1.0)

    def test_r_predicted_zero_raises(self):
        with pytest.raises(ValueError, match="r_predicted"):
            piezo_correction(1.0, 1.0, 0.0, 1.0, 1.0)

    def test_c_x_zero_raises(self):
        with pytest.raises(ValueError, match="c_x_current"):
            piezo_correction(1.0, 1.0, 1.0, 0.0, 1.0)

    def test_c_y_negative_raises(self):
        with pytest.raises(ValueError, match="c_y_current"):
            piezo_correction(1.0, 1.0, 1.0, 1.0, -1.0)
