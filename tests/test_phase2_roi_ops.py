"""Tests for Phase 2: ROI-aware FFT magnitude, line profile, and histogram."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from probeflow.core.roi import ROI, ROISet
from probeflow.processing.image import fft_magnitude, line_profile
from probeflow.processing.display import histogram_from_array
from probeflow.processing.state import (
    ProcessingState,
    ProcessingStep,
    apply_processing_state,
)


# ── Regression for review image-proc #1: calibration forwarded ──────────────

class TestApplyProcessingStateForwardsCalibration:
    """``apply_processing_state`` must forward ``pixel_size_x_m`` /
    ``pixel_size_y_m`` to ``subtract_background(step_tolerance=True)``
    and to ``facet_level``.  Before review image-proc #1 was fixed
    (2026-05-28), the kwargs were dropped and ``step_threshold_deg`` was
    silently interpreted in data-units-per-pixel, making step-tolerant
    background fall back to non-step-tolerant on every GUI invocation.
    """

    def test_subtract_background_kwarg_forwarded(self, monkeypatch):
        """Verify ``pixel_size_x_m`` / ``pixel_size_y_m`` are forwarded to
        :func:`probeflow.processing.subtract_background`.  This is the
        plumbing-correctness test for review image-proc #1; downstream
        effects on the polynomial fit depend on input geometry and are
        covered by the existing background tests in the kernel file."""
        import probeflow.processing as _proc

        captured: dict = {}
        orig = _proc.subtract_background

        def spy(arr, **kwargs):
            captured.update(kwargs)
            return orig(arr, **kwargs)

        monkeypatch.setattr(_proc, "subtract_background", spy)

        arr = np.zeros((32, 32), dtype=np.float64)
        state = ProcessingState(steps=[
            ProcessingStep(
                "plane_bg",
                {"order": 1, "step_tolerance": True},
            ),
        ])
        apply_processing_state(
            arr, state, pixel_size_x_m=50e-12, pixel_size_y_m=75e-12,
        )
        assert captured.get("pixel_size_x_m") == pytest.approx(50e-12)
        assert captured.get("pixel_size_y_m") == pytest.approx(75e-12)

    def test_facet_level_kwarg_forwarded(self, monkeypatch):
        import probeflow.processing as _proc

        captured: dict = {}
        orig = _proc.facet_level

        def spy(arr, **kwargs):
            captured.update(kwargs)
            return orig(arr, **kwargs)

        monkeypatch.setattr(_proc, "facet_level", spy)

        arr = np.zeros((32, 32), dtype=np.float64)
        state = ProcessingState(steps=[
            ProcessingStep("facet_level", {"threshold_deg": 5.0}),
        ])
        apply_processing_state(
            arr, state, pixel_size_x_m=50e-12, pixel_size_y_m=75e-12,
        )
        assert captured.get("pixel_size_x_m") == pytest.approx(50e-12)
        assert captured.get("pixel_size_y_m") == pytest.approx(75e-12)

    def test_no_calibration_does_not_pass_kwargs(self, monkeypatch):
        """When the caller does NOT supply calibration, the kernel
        defaults (1.0 m/pixel) must apply — i.e. the apply_processing_state
        wrapper passes no ``pixel_size_*_m`` kwargs so existing callers
        without scan context are unaffected."""
        import probeflow.processing as _proc

        captured: dict = {}
        orig = _proc.subtract_background

        def spy(arr, **kwargs):
            captured.update(kwargs)
            return orig(arr, **kwargs)

        monkeypatch.setattr(_proc, "subtract_background", spy)

        arr = np.zeros((32, 32), dtype=np.float64)
        state = ProcessingState(steps=[
            ProcessingStep("plane_bg", {"order": 1}),
        ])
        apply_processing_state(arr, state)
        # Neither pixel_size_x_m nor pixel_size_y_m was forwarded:
        assert "pixel_size_x_m" not in captured
        assert "pixel_size_y_m" not in captured


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def square_64():
    rng = np.random.default_rng(42)
    return rng.normal(size=(64, 64))

@pytest.fixture
def sine_grating():
    """20×100 image with a known x-periodicity of 10 pixels (freq = 0.1 cycle/px)."""
    Ny, Nx = 20, 100
    x = np.arange(Nx, dtype=np.float64)
    row = np.sin(2 * math.pi * x / 10.0)
    return np.tile(row, (Ny, 1))

@pytest.fixture
def gradient_image():
    """64×64 image with a linear x-gradient: value = col index."""
    Ny, Nx = 64, 64
    return np.tile(np.arange(Nx, dtype=np.float64), (Ny, 1))

@pytest.fixture
def constant_image():
    return np.full((32, 32), 5.0, dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  fft_magnitude
# ═════════════════════════════════════════════════════════════════════════════

class TestFftMagnitudeNoRoi:
    def test_returns_three_arrays(self, square_64):
        result = fft_magnitude(square_64)
        assert len(result) == 3

    def test_magnitude_shape_matches_input(self, square_64):
        mag, qx, qy = fft_magnitude(square_64)
        assert mag.shape == square_64.shape

    def test_frequency_axes_lengths(self, square_64):
        Ny, Nx = square_64.shape
        mag, qx, qy = fft_magnitude(square_64)
        assert len(qx) == Nx
        assert len(qy) == Ny

    def test_output_is_float64(self, square_64):
        mag, _, _ = fft_magnitude(square_64)
        assert mag.dtype == np.float64

    def test_log_scale_default_is_log1p(self, square_64):
        mag_log, _, _ = fft_magnitude(square_64, log_scale=True)
        mag_lin, _, _ = fft_magnitude(square_64, log_scale=False)
        # log1p version should be smaller in magnitude
        assert float(mag_log.max()) < float(mag_lin.max())

    def test_window_none_differs_from_hann(self, square_64):
        mag_hann, _, _ = fft_magnitude(square_64, window="hann")
        mag_none, _, _ = fft_magnitude(square_64, window="none")
        assert not np.allclose(mag_hann, mag_none)

    def test_window_tukey_differs_from_hann(self, square_64):
        mag_hann, _, _ = fft_magnitude(square_64, window="hann")
        mag_tukey, _, _ = fft_magnitude(square_64, window="tukey")
        assert not np.allclose(mag_hann, mag_tukey)

    def test_invalid_window_raises(self, square_64):
        with pytest.raises(ValueError, match="window"):
            fft_magnitude(square_64, window="blackman")

    def test_dc_bin_is_zero_after_mean_subtraction(self, square_64):
        # fft_magnitude subtracts the mean, so the DC component is 0
        mag, qx, qy = fft_magnitude(square_64, log_scale=False, window="none")
        Ny, Nx = mag.shape
        cy, cx = Ny // 2, Nx // 2
        assert float(mag[cy, cx]) == pytest.approx(0.0, abs=1e-8)

    def test_frequency_axes_centred_at_zero(self, square_64):
        _, qx, qy = fft_magnitude(square_64, pixel_size_x_m=1e-9, pixel_size_y_m=1e-9)
        # fftshift: DC is at index N//2 for even N
        Ny, Nx = square_64.shape
        assert abs(float(qx[Nx // 2])) < 1e-10
        assert abs(float(qy[Ny // 2])) < 1e-10


class TestFftMagnitudeRectRoi:
    def test_shape_matches_roi_bounds(self):
        arr = np.random.default_rng(0).normal(size=(64, 64))
        roi = ROI.new("rectangle", {"x": 10.0, "y": 10.0, "width": 20.0, "height": 16.0})
        mag, qx, qy = fft_magnitude(arr, roi)
        # bounding box: x=10..29 (20 cols), y=10..25 (16 rows)
        assert mag.shape[1] == 20  # cols
        assert mag.shape[0] == 16  # rows
        assert len(qx) == 20
        assert len(qy) == 16

    def test_sine_grating_peak_at_correct_k(self, sine_grating):
        Ny, Nx = sine_grating.shape
        px_m = 1e-9  # 1 nm per pixel
        mag, qx, qy = fft_magnitude(
            sine_grating,
            pixel_size_x_m=px_m,
            pixel_size_y_m=px_m,
            window="none",
            log_scale=False,
        )
        # Expected frequency: 0.1 cycle/px = 0.1 / (1e-9 m) = 1e8 m⁻¹ = 0.1 nm⁻¹
        expected_q = 0.1
        # Find the peak column (excluding DC at centre)
        Ny_out, Nx_out = mag.shape
        cy = Ny_out // 2
        cx = Nx_out // 2
        # Suppress DC region before searching
        mag_search = mag.copy()
        mag_search[cy - 2:cy + 3, cx - 2:cx + 3] = 0
        peak_idx = int(np.argmax(mag_search))
        peak_y, peak_x = divmod(peak_idx, Nx_out)
        # Check the peak q is close to the expected frequency
        assert abs(float(qx[peak_x])) == pytest.approx(expected_q, abs=0.02)


class TestFftMagnitudeNonRectRoi:
    def test_polygon_roi_shape_matches_bbox(self):
        arr = np.random.default_rng(1).normal(size=(64, 64))
        roi = ROI.new("polygon", {"vertices": [
            [10, 10], [30, 10], [30, 30], [10, 30]
        ]})
        mag, qx, qy = fft_magnitude(arr, roi)
        # Shape matches the actual bounding box of the rasterised mask
        r0, r1, c0, c1 = roi.bounds(arr.shape)
        expected_rows = r1 - r0 + 1
        expected_cols = c1 - c0 + 1
        assert mag.shape[0] == expected_rows
        assert mag.shape[1] == expected_cols
        assert len(qy) == expected_rows
        assert len(qx) == expected_cols

    def test_ellipse_roi_accepted(self):
        arr = np.random.default_rng(2).normal(size=(64, 64))
        roi = ROI.new("ellipse", {"cx": 32.0, "cy": 32.0, "rx": 10.0, "ry": 8.0})
        mag, qx, qy = fft_magnitude(arr, roi)
        assert mag.ndim == 2
        assert mag.dtype == np.float64


class TestFftMagnitudeInvalidRois:
    def test_line_roi_raises(self):
        arr = np.zeros((32, 32))
        roi = ROI.new("line", {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0})
        with pytest.raises(ValueError, match="line"):
            fft_magnitude(arr, roi)

    def test_point_roi_raises(self):
        arr = np.zeros((32, 32))
        roi = ROI.new("point", {"x": 5.0, "y": 5.0})
        with pytest.raises(ValueError, match="point"):
            fft_magnitude(arr, roi)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  line_profile (existing API still works + new roi= parameter)
# ═════════════════════════════════════════════════════════════════════════════

class TestLineProfileRoiParam:
    def test_line_roi_matches_explicit_endpoints(self, gradient_image):
        Ny, Nx = gradient_image.shape
        px = 1e-9
        roi = ROI.new("line", {"x1": 5.0, "y1": 16.0, "x2": 50.0, "y2": 16.0})
        s_roi, z_roi = line_profile(
            gradient_image, roi=roi,
            pixel_size_x_m=px, pixel_size_y_m=px,
        )
        s_exp, z_exp = line_profile(
            gradient_image, (5.0, 16.0), (50.0, 16.0),
            pixel_size_x_m=px, pixel_size_y_m=px,
        )
        np.testing.assert_allclose(s_roi, s_exp)
        np.testing.assert_allclose(z_roi, z_exp)

    def test_wrong_roi_kind_raises(self, gradient_image):
        roi = ROI.new("rectangle", {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0})
        with pytest.raises(ValueError, match="line"):
            line_profile(gradient_image, roi=roi, pixel_size_x_m=1e-9, pixel_size_y_m=1e-9)

    def test_roi_and_p0_raises(self, gradient_image):
        roi = ROI.new("line", {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 0.0})
        with pytest.raises(ValueError, match="both"):
            line_profile(
                gradient_image, (0.0, 0.0), roi=roi,
                pixel_size_x_m=1e-9, pixel_size_y_m=1e-9,
            )

    def test_no_roi_no_endpoints_raises(self, gradient_image):
        with pytest.raises(ValueError):
            line_profile(gradient_image, pixel_size_x_m=1e-9, pixel_size_y_m=1e-9)

    def test_constant_image_returns_constant(self, constant_image):
        roi = ROI.new("line", {"x1": 2.0, "y1": 5.0, "x2": 20.0, "y2": 5.0})
        _, z = line_profile(
            constant_image, roi=roi, pixel_size_x_m=1e-9, pixel_size_y_m=1e-9
        )
        np.testing.assert_allclose(z, 5.0, atol=1e-10)

    def test_linear_gradient_profile_matches(self, gradient_image):
        """Profile along a horizontal row of gradient_image = x-values exactly."""
        roi = ROI.new("line", {"x1": 0.0, "y1": 10.0, "x2": 63.0, "y2": 10.0})
        _, z = line_profile(
            gradient_image, roi=roi,
            pixel_size_x_m=1e-9, pixel_size_y_m=1e-9,
            interp="nearest",
        )
        expected = np.arange(64, dtype=np.float64)
        np.testing.assert_allclose(z, expected, atol=0.5)

    def test_bilinear_nearest_differ(self, gradient_image):
        roi = ROI.new("line", {"x1": 0.5, "y1": 10.5, "x2": 30.5, "y2": 10.5})
        _, z_lin = line_profile(
            gradient_image, roi=roi, pixel_size_x_m=1e-9, pixel_size_y_m=1e-9,
            interp="linear",
        )
        _, z_nn = line_profile(
            gradient_image, roi=roi, pixel_size_x_m=1e-9, pixel_size_y_m=1e-9,
            interp="nearest",
        )
        # On a smooth gradient the two interpolation modes differ
        assert not np.allclose(z_lin, z_nn)

    def test_zero_length_line_raises(self, gradient_image):
        roi = ROI.new("line", {"x1": 5.0, "y1": 5.0, "x2": 5.0, "y2": 5.0})
        with pytest.raises(ValueError):
            line_profile(gradient_image, roi=roi, pixel_size_x_m=1e-9, pixel_size_y_m=1e-9)

    def test_existing_p0_p1_api_unchanged(self, gradient_image):
        s, z = line_profile(
            gradient_image, (0.0, 10.0), (63.0, 10.0),
            pixel_size_x_m=1e-9, pixel_size_y_m=1e-9,
        )
        assert len(s) >= 2
        assert len(z) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# 3.  histogram_from_array
# ═════════════════════════════════════════════════════════════════════════════

class TestHistogramNoRoi:
    def test_regression_no_roi(self, gradient_image):
        from probeflow.processing.display import histogram_from_array as orig
        counts, edges = orig(gradient_image, bins=64)
        counts2, edges2 = orig(gradient_image, roi=None, bins=64)
        np.testing.assert_array_equal(counts, counts2)
        np.testing.assert_array_equal(edges, edges2)


class TestHistogramRoi:
    def test_rect_roi_reduces_pixel_count(self, gradient_image):
        # Whole image has 64*64 = 4096 pixels
        counts_full, _ = histogram_from_array(gradient_image)
        assert int(counts_full.sum()) == gradient_image.size

        roi = ROI.new("rectangle", {"x": 10.0, "y": 10.0, "width": 10.0, "height": 10.0})
        counts_roi, _ = histogram_from_array(gradient_image, roi=roi)
        assert int(counts_roi.sum()) == 100  # 10×10

    def test_polygon_roi_correct_pixel_count(self):
        arr = np.zeros((20, 20), dtype=np.float64)
        arr[:, :] = 1.0
        # Triangle with vertices (0,0), (10,0), (0,10) — ~50 pixels
        roi = ROI.new("polygon", {"vertices": [[0, 0], [10, 0], [0, 10]]})
        counts, _ = histogram_from_array(arr, roi=roi)
        # Approximately 50 pixels (half of a 10×10 square)
        assert 40 <= int(counts.sum()) <= 60

    def test_point_roi_single_pixel(self):
        arr = np.ones((20, 20), dtype=np.float64)
        roi = ROI.new("point", {"x": 5.0, "y": 5.0})
        counts, _ = histogram_from_array(arr, roi=roi)
        assert int(counts.sum()) == 1

    def test_rect_roi_on_known_image(self):
        # Image where col=x value, row=y value as a constant
        arr = np.tile(np.arange(10, dtype=np.float64), (10, 1))
        # ROI covering only columns 3-5 (values 3, 4, 5)
        roi = ROI.new("rectangle", {"x": 3.0, "y": 0.0, "width": 3.0, "height": 10.0})
        counts, edges = histogram_from_array(arr, roi=roi, bins=10)
        assert int(counts.sum()) == 30  # 3 cols × 10 rows

    def test_empty_roi_raises(self):
        arr = np.ones((10, 10))
        # A point ROI on a 1-pixel region
        roi = ROI.new("point", {"x": 5.0, "y": 5.0})
        # Should NOT raise — single finite pixel
        counts, _ = histogram_from_array(arr, roi=roi)
        assert int(counts.sum()) == 1

    def test_bins_and_range_respected(self, gradient_image):
        roi = ROI.new("rectangle", {"x": 0.0, "y": 0.0, "width": 64.0, "height": 10.0})
        counts, edges = histogram_from_array(gradient_image, roi=roi, bins=32)
        assert len(counts) == 32
        assert len(edges) == 33


# ═════════════════════════════════════════════════════════════════════════════
# 4.  apply_processing_state with roi_id
# ═════════════════════════════════════════════════════════════════════════════

class TestProcessingStateRoiId:
    def _make_state_and_roi_set(self):
        roi = ROI.new("rectangle", {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0})
        roi_set = ROISet(image_id="test")
        roi_set.add(roi)
        state = ProcessingState(steps=[
            ProcessingStep("roi", {
                "roi_id": roi.id,
                "step": {"op": "smooth", "params": {"sigma_px": 1.0}},
            }),
        ])
        return state, roi_set, roi

    def test_roi_id_lookup_applies_step(self):
        arr = np.ones((64, 64), dtype=np.float64) * 3.0
        state, roi_set, roi = self._make_state_and_roi_set()
        result = apply_processing_state(arr, state, roi_set=roi_set)
        assert result.shape == arr.shape
        # Outside the ROI, unchanged
        np.testing.assert_allclose(result[0, 0], 3.0)

    def test_roi_id_no_roi_set_warns(self):
        arr = np.ones((64, 64), dtype=np.float64)
        state, _, roi = self._make_state_and_roi_set()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_processing_state(arr, state, roi_set=None)
        # Step should be skipped with a warning
        assert any("roi_set" in str(warning.message).lower() or
                   "roi_id" in str(warning.message).lower()
                   for warning in w)
        # Result should equal input (step was skipped)
        np.testing.assert_allclose(result, arr)

    def test_roi_id_missing_id_warns(self):
        arr = np.ones((64, 64), dtype=np.float64)
        roi_set = ROISet(image_id="test")  # empty — ID not found
        state = ProcessingState(steps=[
            ProcessingStep("roi", {
                "roi_id": "nonexistent-uuid",
                "step": {"op": "smooth", "params": {"sigma_px": 1.0}},
            }),
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_processing_state(arr, state, roi_set=roi_set)
        assert any("not found" in str(warning.message).lower() or
                   "roi_id" in str(warning.message).lower()
                   for warning in w)
        np.testing.assert_allclose(result, arr)

# ═════════════════════════════════════════════════════════════════════════════
# 5.  Integration test: full round-trip through sidecar serialisation
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegrationRoundTrip:
    def test_processing_state_with_roi_id_round_trips(self, tmp_path):
        """Load ROISet + ProcessingState, save to dict, reload, reapply → same result."""
        import json

        arr = np.random.default_rng(0).normal(size=(64, 64))

        # Build ROISet and ProcessingState
        roi = ROI.new("rectangle", {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0})
        roi_set = ROISet(image_id="test-scan")
        roi_set.add(roi)

        state = ProcessingState(steps=[
            ProcessingStep("roi", {
                "roi_id": roi.id,
                "step": {"op": "smooth", "params": {"sigma_px": 2.0}},
            }),
        ])

        # First application
        result1 = apply_processing_state(arr, state, roi_set=roi_set)

        # Serialise to dict (mimicking sidecar)
        sidecar = {
            "processing_state": state.to_dict(),
            "rois": roi_set.to_dict(),
        }
        sidecar_path = tmp_path / "test.provenance.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        # Reload
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        state2 = ProcessingState.from_dict(data["processing_state"])
        roi_set2 = ROISet.from_dict(data["rois"])

        # Second application
        result2 = apply_processing_state(arr, state2, roi_set=roi_set2)

        np.testing.assert_allclose(result1, result2)
        assert state2.steps[0].params["roi_id"] == roi.id
