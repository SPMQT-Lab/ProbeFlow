"""Tests for Phase 0 geometric transform operations."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from probeflow.processing.image import (
    flip_horizontal,
    flip_vertical,
    rotate_90_cw,
    rotate_180,
    rotate_270_cw,
    rotate_arbitrary,
)
from probeflow.processing.state import (
    ProcessingState,
    ProcessingStep,
    apply_processing_state,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def asymmetric_arr():
    """A 4×6 array with distinct values so orientation is verifiable."""
    return np.arange(24, dtype=np.float64).reshape(4, 6)


@pytest.fixture
def square_arr():
    """8×8 array for rotation tests where shape is preserved."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(8, 8))


# ── Flip operations ──────────────────────────────────────────────────────────

class TestFlipHorizontal:
    def test_shape_preserved(self, asymmetric_arr):
        out = flip_horizontal(asymmetric_arr)
        assert out.shape == asymmetric_arr.shape

    def test_first_col_becomes_last(self, asymmetric_arr):
        out = flip_horizontal(asymmetric_arr)
        np.testing.assert_array_equal(out[:, 0], asymmetric_arr[:, -1])

    def test_double_flip_is_identity(self, asymmetric_arr):
        out = flip_horizontal(flip_horizontal(asymmetric_arr))
        np.testing.assert_array_almost_equal(out, asymmetric_arr)

    def test_output_is_float64(self, asymmetric_arr):
        assert flip_horizontal(asymmetric_arr).dtype == np.float64

    def test_does_not_mutate_input(self, asymmetric_arr):
        original = asymmetric_arr.copy()
        flip_horizontal(asymmetric_arr)
        np.testing.assert_array_equal(asymmetric_arr, original)

    def test_multi_plane_consistent_shape(self, asymmetric_arr):
        planes = [asymmetric_arr.copy(), asymmetric_arr.copy() + 1.0]
        results = [flip_horizontal(p) for p in planes]
        assert results[0].shape == results[1].shape


class TestFlipVertical:
    def test_shape_preserved(self, asymmetric_arr):
        out = flip_vertical(asymmetric_arr)
        assert out.shape == asymmetric_arr.shape

    def test_first_row_becomes_last(self, asymmetric_arr):
        out = flip_vertical(asymmetric_arr)
        np.testing.assert_array_equal(out[0, :], asymmetric_arr[-1, :])

    def test_double_flip_is_identity(self, asymmetric_arr):
        out = flip_vertical(flip_vertical(asymmetric_arr))
        np.testing.assert_array_almost_equal(out, asymmetric_arr)

    def test_output_is_float64(self, asymmetric_arr):
        assert flip_vertical(asymmetric_arr).dtype == np.float64


# ── Lossless 90/180/270 rotations ─────────────────────────────────────────────

class TestRotate90CW:
    def test_swaps_shape(self, asymmetric_arr):
        Ny, Nx = asymmetric_arr.shape
        out = rotate_90_cw(asymmetric_arr)
        assert out.shape == (Nx, Ny)

    def test_four_rotations_is_identity(self, asymmetric_arr):
        a = asymmetric_arr
        for _ in range(4):
            a = rotate_90_cw(a)
        np.testing.assert_array_almost_equal(a, asymmetric_arr)

    def test_pixel_count_preserved(self, asymmetric_arr):
        out = rotate_90_cw(asymmetric_arr)
        assert out.size == asymmetric_arr.size

    def test_no_new_pixel_values(self, asymmetric_arr):
        out = rotate_90_cw(asymmetric_arr)
        assert set(out.ravel().tolist()) == set(asymmetric_arr.ravel().tolist())

    def test_multi_plane_consistent_shape(self, asymmetric_arr):
        planes = [asymmetric_arr.copy(), asymmetric_arr.copy() + 1.0]
        results = [rotate_90_cw(p) for p in planes]
        assert results[0].shape == results[1].shape


class TestRotate180:
    def test_preserves_shape(self, asymmetric_arr):
        out = rotate_180(asymmetric_arr)
        assert out.shape == asymmetric_arr.shape

    def test_double_is_identity(self, asymmetric_arr):
        out = rotate_180(rotate_180(asymmetric_arr))
        np.testing.assert_array_almost_equal(out, asymmetric_arr)

    def test_pixel_count_preserved(self, asymmetric_arr):
        assert rotate_180(asymmetric_arr).size == asymmetric_arr.size


class TestRotate270CW:
    def test_swaps_shape(self, asymmetric_arr):
        Ny, Nx = asymmetric_arr.shape
        out = rotate_270_cw(asymmetric_arr)
        assert out.shape == (Nx, Ny)

    def test_90_plus_270_is_identity(self, asymmetric_arr):
        out = rotate_90_cw(rotate_270_cw(asymmetric_arr))
        np.testing.assert_array_almost_equal(out, asymmetric_arr)


# ── Arbitrary rotation ────────────────────────────────────────────────────────

class TestRotateArbitrary:
    def test_zero_angle_returns_input_within_tolerance(self, square_arr):
        out = rotate_arbitrary(square_arr, 0.0)
        assert out.shape == square_arr.shape
        np.testing.assert_allclose(out, square_arr, atol=1e-10)

    def test_90_via_arbitrary_matches_lossless_shape(self, square_arr):
        lossless = rotate_90_cw(square_arr)
        via_arb = rotate_arbitrary(square_arr, -90.0)  # -90 CCW = 90 CW
        assert via_arb.shape == lossless.shape

    def test_180_via_arbitrary_matches_lossless_shape(self, square_arr):
        lossless = rotate_180(square_arr)
        via_arb = rotate_arbitrary(square_arr, 180.0)
        assert via_arb.shape == lossless.shape

    def test_45deg_canvas_larger_than_input(self):
        arr = np.ones((20, 20))
        out = rotate_arbitrary(arr, 45.0)
        assert out.shape[0] > 20
        assert out.shape[1] > 20

    def test_no_input_pixel_lost(self):
        # Use a region rather than a single pixel: bilinear spreads a 1-pixel
        # spike but conserves total energy (within interpolation tolerance).
        arr = np.zeros((20, 20))
        arr[8:12, 8:12] = 1.0  # 4×4 non-zero region
        out = rotate_arbitrary(arr, 45.0)
        assert float(out.sum()) > float(arr.sum()) * 0.8

    def test_corner_regions_are_zero(self):
        arr = np.ones((30, 30))
        out = rotate_arbitrary(arr, 45.0)
        assert float(out[0, 0]) == pytest.approx(0.0)
        assert float(out[0, -1]) == pytest.approx(0.0)
        assert float(out[-1, 0]) == pytest.approx(0.0)
        assert float(out[-1, -1]) == pytest.approx(0.0)

    def test_multi_plane_consistent_shape(self):
        planes = [np.ones((12, 12)), np.ones((12, 12)) * 2.0]
        results = [rotate_arbitrary(p, 30.0) for p in planes]
        assert results[0].shape == results[1].shape

    def test_output_is_float64(self, square_arr):
        assert rotate_arbitrary(square_arr, 15.0).dtype == np.float64

    def test_invalid_order_raises(self, square_arr):
        with pytest.raises(ValueError):
            rotate_arbitrary(square_arr, 10.0, order=5)


# ── ProcessingState integration ───────────────────────────────────────────────

class TestProcessingStateIntegration:
    def test_flip_horizontal_via_state(self, asymmetric_arr):
        state = ProcessingState(steps=[ProcessingStep("flip_horizontal")])
        result = apply_processing_state(asymmetric_arr, state)
        np.testing.assert_array_almost_equal(result, flip_horizontal(asymmetric_arr))

    def test_flip_vertical_via_state(self, asymmetric_arr):
        state = ProcessingState(steps=[ProcessingStep("flip_vertical")])
        result = apply_processing_state(asymmetric_arr, state)
        np.testing.assert_array_almost_equal(result, flip_vertical(asymmetric_arr))

    def test_rotate_90_cw_via_state(self, asymmetric_arr):
        state = ProcessingState(steps=[ProcessingStep("rotate_90_cw")])
        result = apply_processing_state(asymmetric_arr, state)
        np.testing.assert_array_almost_equal(result, rotate_90_cw(asymmetric_arr))

    def test_rotate_180_via_state(self, asymmetric_arr):
        state = ProcessingState(steps=[ProcessingStep("rotate_180")])
        result = apply_processing_state(asymmetric_arr, state)
        np.testing.assert_array_almost_equal(result, rotate_180(asymmetric_arr))

    def test_rotate_270_cw_via_state(self, asymmetric_arr):
        state = ProcessingState(steps=[ProcessingStep("rotate_270_cw")])
        result = apply_processing_state(asymmetric_arr, state)
        np.testing.assert_array_almost_equal(result, rotate_270_cw(asymmetric_arr))

    def test_rotate_arbitrary_via_state(self, square_arr):
        state = ProcessingState(steps=[
            ProcessingStep("rotate_arbitrary", {"angle_degrees": 30.0}),
        ])
        result = apply_processing_state(square_arr, state)
        assert result.dtype == np.float64
        assert result.shape[0] > square_arr.shape[0]

    def test_rotate_arbitrary_warns_when_roi_present(self, square_arr):
        state = ProcessingState(steps=[
            ProcessingStep("roi", {
                "roi_id": "roi-1",
                "step": {"op": "smooth", "params": {"sigma_px": 1.0}},
            }),
            ProcessingStep("rotate_arbitrary", {"angle_degrees": 30.0}),
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            apply_processing_state(square_arr, state)
        assert any("rotate_arbitrary" in str(warning.message) for warning in w)

    def test_serialisation_round_trip(self):
        state = ProcessingState(steps=[
            ProcessingStep("flip_horizontal"),
            ProcessingStep("rotate_arbitrary", {"angle_degrees": 15.0, "order": 1}),
        ])
        restored = ProcessingState.from_dict(state.to_dict())
        assert [s.op for s in restored.steps] == ["flip_horizontal", "rotate_arbitrary"]
        assert abs(restored.steps[1].params["angle_degrees"] - 15.0) < 1e-12


# ── GUI adapter integration ───────────────────────────────────────────────────

class TestGuiAdapterIntegration:
    def test_flip_horizontal_from_gui(self):
        from probeflow.processing.gui_adapter import processing_state_from_gui
        state = processing_state_from_gui({
            "geometric_ops": [{"op": "flip_horizontal", "params": {}}],
        })
        assert len(state.steps) == 1
        assert state.steps[0].op == "flip_horizontal"

    def test_flip_vertical_from_gui(self):
        from probeflow.processing.gui_adapter import processing_state_from_gui
        state = processing_state_from_gui({
            "geometric_ops": [{"op": "flip_vertical", "params": {}}],
        })
        assert state.steps[0].op == "flip_vertical"

    def test_rotate_90_cw_from_gui(self):
        from probeflow.processing.gui_adapter import processing_state_from_gui
        state = processing_state_from_gui({
            "geometric_ops": [{"op": "rotate_90_cw", "params": {}}],
        })
        assert state.steps[0].op == "rotate_90_cw"

    def test_rotate_arbitrary_from_gui_with_angle(self):
        from probeflow.processing.gui_adapter import processing_state_from_gui
        state = processing_state_from_gui({
            "geometric_ops": [{
                "op": "rotate_arbitrary",
                "params": {"angle_degrees": 45.0},
            }],
        })
        assert len(state.steps) == 1
        assert state.steps[0].op == "rotate_arbitrary"
        assert abs(state.steps[0].params["angle_degrees"] - 45.0) < 1e-12

    def test_multiple_geometric_ops_from_gui(self):
        from probeflow.processing.gui_adapter import processing_state_from_gui
        state = processing_state_from_gui({
            "geometric_ops": [
                {"op": "flip_horizontal", "params": {}},
                {"op": "rotate_90_cw", "params": {}},
            ],
        })
        assert [s.op for s in state.steps] == ["flip_horizontal", "rotate_90_cw"]

    def test_geometric_ops_combined_with_other_processing(self):
        from probeflow.processing.gui_adapter import processing_state_from_gui
        state = processing_state_from_gui({
            "align_rows": "median",
            "geometric_ops": [{"op": "flip_horizontal", "params": {}}],
        })
        ops = [s.op for s in state.steps]
        assert "align_rows" in ops
        assert "flip_horizontal" in ops
