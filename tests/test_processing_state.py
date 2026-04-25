"""Tests for probeflow.processing_state and the GUI processing bridge."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing_state import (
    ProcessingState,
    ProcessingStep,
    apply_processing_state,
)
from probeflow.gui_processing import processing_state_from_gui


# ── Test A: empty state is identity ──────────────────────────────────────────

class TestEmptyState:
    def test_returns_array_equal_to_input(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        state = ProcessingState()
        result = apply_processing_state(arr, state)
        np.testing.assert_array_almost_equal(result, arr)

    def test_does_not_mutate_input(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        original = arr.copy()
        state = ProcessingState()
        apply_processing_state(arr, state)
        np.testing.assert_array_equal(arr, original)

    def test_returns_new_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        state = ProcessingState()
        result = apply_processing_state(arr, state)
        # Must be a new object even for empty state (no aliasing)
        assert result is not arr

    def test_output_is_float64(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        state = ProcessingState()
        result = apply_processing_state(arr, state)
        assert result.dtype == np.float64

    def test_shape_preserved(self):
        arr = np.ones((7, 13))
        state = ProcessingState()
        result = apply_processing_state(arr, state)
        assert result.shape == (7, 13)


# ── Test B: serialisation round-trip ─────────────────────────────────────────

class TestSerialisation:
    def test_round_trip_preserves_op_names(self):
        state = ProcessingState(steps=[
            ProcessingStep("remove_bad_lines", {"threshold_mad": 5.0}),
            ProcessingStep("align_rows", {"method": "median"}),
        ])
        restored = ProcessingState.from_dict(state.to_dict())
        assert [s.op for s in restored.steps] == ["remove_bad_lines", "align_rows"]

    def test_round_trip_preserves_params(self):
        state = ProcessingState(steps=[
            ProcessingStep("plane_bg", {"order": 2}),
            ProcessingStep("smooth",   {"sigma_px": 1.5}),
        ])
        restored = ProcessingState.from_dict(state.to_dict())
        assert restored.steps[0].params["order"] == 2
        assert abs(restored.steps[1].params["sigma_px"] - 1.5) < 1e-12

    def test_to_dict_has_steps_key(self):
        state = ProcessingState()
        d = state.to_dict()
        assert "steps" in d
        assert isinstance(d["steps"], list)

    def test_empty_state_round_trip(self):
        state = ProcessingState()
        restored = ProcessingState.from_dict(state.to_dict())
        assert len(restored.steps) == 0

    def test_from_dict_unknown_keys_ignored(self):
        data = {"steps": [{"op": "align_rows", "params": {"method": "mean"}}],
                "version": "1.0"}
        state = ProcessingState.from_dict(data)
        assert len(state.steps) == 1

    def test_serialised_form_is_json_compatible(self):
        import json
        state = ProcessingState(steps=[
            ProcessingStep("plane_bg", {"order": 1}),
        ])
        # Should not raise
        json.dumps(state.to_dict())


# ── Test C: unknown operation raises ─────────────────────────────────────────

class TestUnknownOperation:
    def test_raises_value_error(self):
        state = ProcessingState(steps=[
            ProcessingStep("magic_filter", {"strength": 9000}),
        ])
        arr = np.ones((10, 10))
        with pytest.raises(ValueError, match="magic_filter"):
            apply_processing_state(arr, state)

    def test_error_message_names_operation(self):
        state = ProcessingState(steps=[ProcessingStep("nonexistent_op")])
        arr = np.ones((10, 10))
        with pytest.raises(ValueError, match="nonexistent_op"):
            apply_processing_state(arr, state)

    def test_valid_op_before_invalid_still_raises(self):
        state = ProcessingState(steps=[
            ProcessingStep("align_rows", {"method": "median"}),
            ProcessingStep("bad_op"),
        ])
        arr = np.ones((10, 10))
        with pytest.raises(ValueError):
            apply_processing_state(arr, state)


# ── Test D: GUI conversion excludes display-only keys ────────────────────────

class TestGuiConversion:
    FULL_GUI_STATE = {
        # numeric processing
        "remove_bad_lines": True,
        "align_rows":       "median",
        "bg_order":         1,
        "facet_level":      False,
        "smooth_sigma":     None,
        "edge_method":      None,
        "fft_mode":         None,
        # display-only — must NOT appear in ProcessingState
        "colormap":         "inferno",
        "clip_low":         1.0,
        "clip_high":        99.0,
        "grain_threshold":  50.0,
        "grain_above":      True,
    }

    def test_display_keys_absent_from_steps(self):
        state = processing_state_from_gui(self.FULL_GUI_STATE)
        op_names = {s.op for s in state.steps}
        display_only = {"colormap", "clip_low", "clip_high",
                        "grain_threshold", "grain_above"}
        assert op_names.isdisjoint(display_only)

    def test_numeric_ops_present(self):
        gui = {
            "remove_bad_lines": True,
            "align_rows": "mean",
            "bg_order": 2,
        }
        state = processing_state_from_gui(gui)
        op_names = [s.op for s in state.steps]
        assert "remove_bad_lines" in op_names
        assert "align_rows" in op_names
        assert "plane_bg" in op_names

    def test_false_bool_ops_excluded(self):
        gui = {"remove_bad_lines": False, "facet_level": False}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 0

    def test_none_value_ops_excluded(self):
        gui = {"align_rows": None, "smooth_sigma": None,
               "edge_method": None, "fft_mode": None}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 0

    def test_edge_params_captured(self):
        gui = {"edge_method": "laplacian", "edge_sigma": 2.0, "edge_sigma2": 3.0}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 1
        step = state.steps[0]
        assert step.op == "edge_detect"
        assert abs(step.params["sigma"]  - 2.0) < 1e-12
        assert abs(step.params["sigma2"] - 3.0) < 1e-12

    def test_fft_params_captured(self):
        gui = {"fft_mode": "low_pass", "fft_cutoff": 0.15, "fft_window": "hanning"}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 1
        step = state.steps[0]
        assert step.op == "fourier_filter"
        assert abs(step.params["cutoff"] - 0.15) < 1e-12
        assert step.params["window"] == "hanning"

    def test_bg_step_tolerance_captured(self):
        gui = {"bg_order": 2, "bg_step_tolerance": True}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 1
        step = state.steps[0]
        assert step.op == "plane_bg"
        assert step.params == {"order": 2, "step_tolerance": True}

    def test_bg_step_tolerance_defaults_false(self):
        state = processing_state_from_gui({"bg_order": 1})
        assert state.steps[0].params["step_tolerance"] is False

    def test_fft_soft_border_params_captured(self):
        gui = {
            "fft_soft_border":      True,
            "fft_soft_mode":        "high_pass",
            "fft_soft_cutoff":      0.20,
            "fft_soft_border_frac": 0.05,
        }
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 1
        step = state.steps[0]
        assert step.op == "fft_soft_border"
        assert step.params["mode"] == "high_pass"
        assert abs(step.params["cutoff"]      - 0.20) < 1e-12
        assert abs(step.params["border_frac"] - 0.05) < 1e-12

    def test_linear_undistort_emitted_only_if_nondefault(self):
        # Both defaults → no step
        state = processing_state_from_gui({"linear_undistort": True})
        assert len(state.steps) == 0
        # Non-default shear → step emitted
        state = processing_state_from_gui({
            "linear_undistort": True, "undistort_shear_x": 1.5,
        })
        assert len(state.steps) == 1
        assert state.steps[0].op == "linear_undistort"
        assert abs(state.steps[0].params["shear_x"] - 1.5) < 1e-12

    def test_set_zero_point_params_captured(self):
        gui = {"set_zero_xy": (10, 20), "set_zero_patch": 3}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 1
        step = state.steps[0]
        assert step.op == "set_zero_point"
        assert step.params == {"x_px": 10, "y_px": 20, "patch": 3}

    def test_set_zero_point_bad_input_skipped(self):
        # Malformed coordinate must not crash; just no step emitted.
        state = processing_state_from_gui({"set_zero_xy": "not-a-tuple"})
        assert len(state.steps) == 0

    def test_empty_gui_state(self):
        state = processing_state_from_gui({})
        assert len(state.steps) == 0

    def test_order_preserved(self):
        # The canonical GUI application order is fixed; test it is preserved.
        gui = {
            "remove_bad_lines": True,
            "align_rows": "median",
            "bg_order": 1,
        }
        state = processing_state_from_gui(gui)
        ops = [s.op for s in state.steps]
        assert ops == ["remove_bad_lines", "align_rows", "plane_bg"]


# ── Test E: GUI / export processing equivalence ───────────────────────────────

class TestGuiExportEquivalence:
    """Preview and export must produce the same processed array."""

    def _make_arr(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.normal(loc=1e-9, scale=1e-10, size=(32, 32))

    def test_same_result_from_state_and_gui_dict(self):
        """apply_processing_state and the legacy _apply_processing give same output."""
        from probeflow.gui import _apply_processing

        arr = self._make_arr()
        gui = {"align_rows": "median", "bg_order": 1}

        result_gui    = _apply_processing(arr, gui)
        state         = processing_state_from_gui(gui)
        result_state  = apply_processing_state(arr, state)

        np.testing.assert_array_almost_equal(result_gui, result_state)

    def test_raw_array_not_mutated_by_either_path(self):
        from probeflow.gui import _apply_processing

        arr = self._make_arr()
        original = arr.copy()
        gui = {"remove_bad_lines": True, "align_rows": "median"}

        _apply_processing(arr, gui)
        np.testing.assert_array_equal(arr, original)

        apply_processing_state(arr, processing_state_from_gui(gui))
        np.testing.assert_array_equal(arr, original)

    def test_apply_processing_state_to_scan_uses_same_path(self):
        """apply_processing_state_to_scan and apply_processing_state agree."""
        from unittest.mock import MagicMock
        from probeflow.gui_processing import apply_processing_state_to_scan

        arr = self._make_arr()
        gui = {"align_rows": "mean", "bg_order": 2}

        # Build a minimal mock Scan
        scan = MagicMock()
        scan.planes = [arr.copy()]
        scan.processing_history = []

        apply_processing_state_to_scan(scan, gui, plane_idx=0)
        result_export = scan.planes[0]

        result_direct = apply_processing_state(arr, processing_state_from_gui(gui))
        np.testing.assert_array_almost_equal(result_export, result_direct)


# ── Test F: apply_processing_state with known steps ──────────────────────────

class TestApplyKnownSteps:
    def test_align_rows_median(self):
        rng = np.random.default_rng(0)
        arr = rng.normal(size=(20, 20))
        state = ProcessingState(steps=[ProcessingStep("align_rows", {"method": "median"})])
        result = apply_processing_state(arr, state)
        assert result.shape == arr.shape
        assert result.dtype == np.float64

    def test_plane_bg_order1_removes_tilt(self):
        x = np.linspace(0, 1, 30)
        arr = np.outer(np.ones(30), x)  # pure linear tilt
        state = ProcessingState(steps=[ProcessingStep("plane_bg", {"order": 1})])
        result = apply_processing_state(arr, state)
        # Residual after plane subtraction should be near-zero
        assert float(np.std(result)) < 1e-10

    def test_smooth_reduces_variance(self):
        rng = np.random.default_rng(1)
        arr = rng.normal(size=(40, 40))
        state = ProcessingState(steps=[ProcessingStep("smooth", {"sigma_px": 2.0})])
        result = apply_processing_state(arr, state)
        assert float(np.std(result)) < float(np.std(arr))

    def test_multi_step_does_not_mutate_intermediate(self):
        arr = np.ones((20, 20)) * 5.0
        state = ProcessingState(steps=[
            ProcessingStep("align_rows", {"method": "median"}),
            ProcessingStep("plane_bg", {"order": 1}),
        ])
        original = arr.copy()
        apply_processing_state(arr, state)
        np.testing.assert_array_equal(arr, original)

    def test_plane_bg_step_tolerance_runs(self):
        # Tilted plane with a sharp step edge superimposed.
        x = np.linspace(0, 1, 40)
        arr = np.outer(np.ones(40), x).copy()
        arr[:, 25:] += 0.5  # step
        state = ProcessingState(steps=[
            ProcessingStep("plane_bg", {"order": 1, "step_tolerance": True}),
        ])
        result = apply_processing_state(arr, state)
        assert result.shape == arr.shape
        assert result.dtype == np.float64

    def test_fft_soft_border_runs_and_preserves_shape(self):
        rng = np.random.default_rng(2)
        arr = rng.normal(size=(32, 32))
        state = ProcessingState(steps=[ProcessingStep("fft_soft_border", {
            "mode": "low_pass", "cutoff": 0.20, "border_frac": 0.10,
        })])
        result = apply_processing_state(arr, state)
        assert result.shape == arr.shape
        # Low-pass should reduce variance vs raw noise.
        assert float(np.std(result)) < float(np.std(arr))

    def test_linear_undistort_preserves_shape(self):
        arr = np.ones((20, 20)) * 3.0
        state = ProcessingState(steps=[ProcessingStep("linear_undistort", {
            "shear_x": 1.0, "scale_y": 0.95,
        })])
        result = apply_processing_state(arr, state)
        assert result.shape == arr.shape

    def test_set_zero_point_anchors_pixel_to_zero(self):
        arr = np.full((10, 10), 42.0)
        state = ProcessingState(steps=[ProcessingStep("set_zero_point", {
            "x_px": 4, "y_px": 5, "patch": 1,
        })])
        result = apply_processing_state(arr, state)
        # Anchored pixel must now read zero (within fp tolerance).
        assert abs(float(result[5, 4])) < 1e-12
        # And every other pixel shifts by the same offset (-42.0).
        np.testing.assert_array_almost_equal(result, arr - 42.0)
