"""Tests for the advanced edge-detection backend.

Synthetic arrays only.  No GUI.  No external files.  Deterministic.
"""
from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing import (
    canny_edges,
    edge_detect,
    gradient_filter,
)
from probeflow.processing.edge_detection import CANNY_PRESETS, EdgeDetectionResult


# ── synthetic images ────────────────────────────────────────────────────────

def _step_image(n: int = 64, edge_col: int = 32) -> np.ndarray:
    """A vertical step edge: left half low, right half high."""
    img = np.zeros((n, n), dtype=np.float64)
    img[:, edge_col:] = 1.0
    return img


def _disk_image(n: int = 64, radius: int = 18) -> np.ndarray:
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = n // 2
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2).astype(np.float64)


# ── Canny ────────────────────────────────────────────────────────────────────

class TestCanny:
    def test_finds_step_edge_near_boundary(self):
        edge_col = 32
        res = canny_edges(_step_image(edge_col=edge_col), sigma=1.0)
        assert isinstance(res, EdgeDetectionResult)
        assert res.edge_mask is not None and res.edge_mask.any()
        # Edge pixels cluster within a couple of px of the true boundary.
        cols = np.flatnonzero(res.edge_mask.any(axis=0))
        assert np.all(np.abs(cols - edge_col) <= 2)

    def test_disk_edges_form_a_ring(self):
        res = canny_edges(_disk_image(), sigma=1.0, threshold_mode="percentile")
        # The boundary of a disk has many more edge pixels than its area=0 interior row.
        assert res.edge_mask.sum() > 30

    def test_percentile_and_absolute_modes_both_run(self):
        img = _step_image()
        a = canny_edges(img, sigma=1.0, threshold_mode="percentile", low=70, high=90)
        b = canny_edges(img, sigma=1.0, threshold_mode="absolute", low=0.0, high=0.0)
        assert a.edge_mask.any()
        assert b.edge_mask.any()

    def test_nan_input_preserved_in_display_mask_finite(self):
        img = _step_image()
        img[0, 0] = np.nan
        res = canny_edges(img, sigma=1.0)
        assert np.isnan(res.display_image[0, 0])
        assert res.edge_mask.dtype == bool
        assert np.isfinite(res.edge_mask).all()  # bool is always finite
        assert not res.edge_mask[0, 0]

    def test_roi_mask_confines_edges(self):
        img = _step_image(edge_col=32)
        roi = np.zeros_like(img, dtype=bool)
        roi[:, :16] = True  # ROI excludes the real edge at col 32
        res = canny_edges(img, sigma=1.0, roi_mask=roi)
        assert res.parameters["roi_restricted"] is True
        assert not res.edge_mask[:, 16:].any()

    @pytest.mark.parametrize("name", sorted(CANNY_PRESETS))
    def test_preset_runs_and_overrides_params(self, name):
        res = canny_edges(_step_image(), preset=name)
        assert res.parameters["preset"] == name
        assert res.parameters["sigma"] == CANNY_PRESETS[name]["sigma"]
        assert res.edge_mask.any()

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            canny_edges(_step_image(), preset="nope")

    def test_pixel_size_records_sigma_nm(self):
        res = canny_edges(_step_image(), sigma=2.0, pixel_size_nm=0.05)
        assert res.parameters["sigma_nm"] == pytest.approx(0.1)


# ── Sobel / Scharr ─────────────────────────────────────────────────────────────

class TestGradient:
    @pytest.mark.parametrize("operator", ["sobel", "scharr"])
    def test_magnitude_peaks_at_edge(self, operator):
        edge_col = 32
        res = gradient_filter(_step_image(edge_col=edge_col), operator=operator,
                              output="magnitude", normalize=False)
        assert res.gradient_magnitude is not None
        # Peak response is at the step boundary.
        col_response = res.gradient_magnitude.sum(axis=0)
        assert abs(int(np.argmax(col_response)) - edge_col) <= 1

    def test_orientation_of_vertical_step_is_horizontal_gradient(self):
        # A vertical step edge has a gradient pointing in +x (orientation ~ 0).
        res = gradient_filter(_step_image(), output="orientation")
        edge = gradient_filter(_step_image(), output="magnitude").gradient_magnitude
        strong = edge > 0.5 * edge.max()
        ori = res.gradient_orientation[strong]
        # arctan2(gy, gx) ~ 0 for a pure +x gradient.
        assert np.allclose(np.cos(ori), 1.0, atol=1e-6)

    def test_normalize_scales_to_unit_peak(self):
        res = gradient_filter(_disk_image(), output="magnitude", normalize=True)
        assert np.nanmax(res.display_image) == pytest.approx(1.0)

    def test_threshold_to_mask_produces_binary(self):
        res = gradient_filter(_step_image(), output="magnitude",
                              threshold_to_mask=True, threshold=90)
        assert res.edge_mask is not None
        assert res.edge_mask.dtype == bool
        assert res.edge_mask.any()

    def test_flat_image_threshold_marks_no_edges(self):
        # Regression: a zero percentile cutoff must not mark the flat
        # zero-gradient background as an edge.
        flat = np.zeros((32, 32), dtype=np.float64)
        res = gradient_filter(flat, threshold_to_mask=True, threshold=90)
        assert res.edge_mask is not None
        assert not res.edge_mask.any()

    def test_sparse_step_threshold_not_whole_image(self):
        # A single step edge in a mostly-flat image: the cutoff percentile can
        # be 0, but only the genuine edge pixels (a small fraction) qualify.
        img = _step_image(n=64, edge_col=32)
        res = gradient_filter(img, threshold_to_mask=True, threshold=90)
        frac = res.edge_mask.mean()
        assert 0.0 < frac < 0.5

    def test_anisotropic_pixels_change_orientation(self):
        # On a diagonal ramp, physical pixel anisotropy must rotate the
        # gradient orientation relative to the isotropic (pixel-space) case.
        yy, xx = np.mgrid[0:64, 0:64]
        ramp = (xx + yy).astype(np.float64)
        iso = gradient_filter(ramp, output="orientation")
        aniso = gradient_filter(ramp, output="orientation",
                                pixel_size_x_nm=1.0, pixel_size_y_nm=4.0)
        mid = (slice(8, 56), slice(8, 56))
        assert not np.allclose(iso.gradient_orientation[mid],
                               aniso.gradient_orientation[mid])

    def test_isotropic_scaling_matches_unscaled_orientation(self):
        yy, xx = np.mgrid[0:64, 0:64]
        ramp = (xx + 2 * yy).astype(np.float64)
        a = gradient_filter(ramp, output="orientation")
        b = gradient_filter(ramp, output="orientation",
                            pixel_size_x_nm=2.0, pixel_size_y_nm=2.0)
        mid = (slice(8, 56), slice(8, 56))
        assert np.allclose(a.gradient_orientation[mid], b.gradient_orientation[mid])

    def test_nan_preserved_in_display(self):
        img = _step_image()
        img[5, 5] = np.nan
        res = gradient_filter(img, output="magnitude")
        assert np.isnan(res.display_image[5, 5])

    def test_x_and_y_outputs_differ_for_diagonal(self):
        yy, xx = np.mgrid[0:64, 0:64]
        ramp = (xx + 2 * yy).astype(np.float64)
        gx = gradient_filter(ramp, output="x", normalize=False).display_image
        gy = gradient_filter(ramp, output="y", normalize=False).display_image
        # y-slope is twice the x-slope, so |gy| median should exceed |gx|.
        assert np.median(np.abs(gy)) > np.median(np.abs(gx))

    def test_invalid_operator_and_output_raise(self):
        with pytest.raises(ValueError):
            gradient_filter(_step_image(), operator="prewitt")
        with pytest.raises(ValueError):
            gradient_filter(_step_image(), output="curl")


# ── edge_detect (history-replayable) sobel/scharr extension ────────────────────

class TestEdgeDetectGradient:
    @pytest.mark.parametrize("method", ["sobel", "scharr"])
    def test_returns_nonnegative_magnitude(self, method):
        out = edge_detect(_step_image(), method=method)
        assert out.shape == (64, 64)
        assert np.nanmin(out) >= 0.0
        assert np.nanmax(out) > 0.0

    def test_nan_roundtrips(self):
        img = _step_image()
        img[10, 10] = np.nan
        out = edge_detect(img, method="sobel")
        assert np.isnan(out[10, 10])

    def test_unknown_method_still_raises(self):
        with pytest.raises(ValueError):
            edge_detect(_step_image(), method="bogus")

    @pytest.mark.parametrize("method", ["sobel", "scharr"])
    def test_replayable_through_gui_state(self, method):
        # Connectivity: a Sobel/Scharr "Edge:" selection must survive the GUI →
        # ProcessingState → replay chain (the "history-replayable" claim).
        from probeflow.processing.gui_adapter import processing_state_from_gui
        from probeflow.processing.state import apply_processing_state

        state = processing_state_from_gui({"edge_method": method, "edge_sigma": 1})
        ops = [step.op for step in state.steps]
        assert "edge_detect" in ops
        step = next(s for s in state.steps if s.op == "edge_detect")
        assert step.params["method"] == method
        out = apply_processing_state(_step_image(), state)
        assert out.shape == (64, 64)
        assert np.nanmax(out) > 0.0
