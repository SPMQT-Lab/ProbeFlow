"""Tests for probeflow.processing.state and the GUI processing bridge."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.state import (
    ProcessingState,
    ProcessingStep,
    apply_processing_state,
)
from probeflow.processing.gui_adapter import processing_state_from_gui


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

    def test_background_fit_rect_captured_for_polynomial_fit(self):
        state = processing_state_from_gui({
            "bg_order": 1,
            "background_fit_rect": (0, 1, 8, 9),
        })
        assert len(state.steps) == 1
        assert state.steps[0].op == "plane_bg"
        assert state.steps[0].params == {
            "order": 1,
            "step_tolerance": False,
            "fit_rect": (0, 1, 8, 9),
        }

    def test_bad_background_fit_rect_is_ignored(self):
        state = processing_state_from_gui({
            "bg_order": 1,
            "background_fit_rect": "bad",
        })
        assert len(state.steps) == 1
        assert "fit_rect" not in state.steps[0].params

    def test_stm_line_background_step_tolerant_captured(self):
        state = processing_state_from_gui({"stm_line_bg": "step_tolerant"})
        assert len(state.steps) == 1
        assert state.steps[0].op == "stm_line_bg"
        assert state.steps[0].params == {"mode": "step_tolerant"}

    def test_roi_scope_wraps_local_filter_step(self):
        gui = {
            "processing_scope": "roi",
            "roi_rect": (2, 3, 8, 9),
            "smooth_sigma": 1.5,
        }
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 1
        step = state.steps[0]
        assert step.op == "roi"
        assert step.params["rect"] == (2, 3, 8, 9)
        assert step.params["step"] == {
            "op": "smooth",
            "params": {"sigma_px": 1.5},
        }

    def test_roi_scope_wraps_local_filter_with_shape_geometry(self):
        geometry = {"kind": "ellipse", "rect_px": (2, 3, 8, 9)}
        state = processing_state_from_gui({
            "processing_scope": "roi",
            "roi_geometry": geometry,
            "smooth_sigma": 1.5,
        })
        assert len(state.steps) == 1
        step = state.steps[0]
        assert step.op == "roi"
        assert step.params["geometry"] == geometry
        assert "rect" not in step.params
        assert step.params["step"]["op"] == "smooth"

    def test_roi_scope_keeps_global_background_steps_global(self):
        gui = {
            "processing_scope": "roi",
            "roi_rect": (2, 3, 8, 9),
            "align_rows": "median",
            "bg_order": 1,
            "stm_line_bg": "step_tolerant",
            "smooth_sigma": 1.0,
        }
        state = processing_state_from_gui(gui)
        assert [s.op for s in state.steps] == [
            "align_rows",
            "plane_bg",
            "stm_line_bg",
            "roi",
        ]
        assert state.steps[-1].params["step"]["op"] == "smooth"

    def test_bad_roi_rect_falls_back_to_global_local_filter(self):
        gui = {
            "processing_scope": "roi",
            "roi_rect": "not-a-rect",
            "smooth_sigma": 1.0,
        }
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 1
        assert state.steps[0].op == "smooth"

    def test_highpass_gui_state_captured(self):
        state = processing_state_from_gui({"highpass_sigma": 12})
        assert len(state.steps) == 1
        assert state.steps[0].op == "gaussian_high_pass"
        assert state.steps[0].params == {"sigma_px": 12.0}

    def test_periodic_notches_gui_state_captured(self):
        state = processing_state_from_gui({
            "periodic_notches": [(8, 0), ("bad", 2), (0, -6)],
            "periodic_notch_radius": 4,
        })
        assert len(state.steps) == 1
        assert state.steps[0].op == "periodic_notch_filter"
        assert state.steps[0].params == {
            "peaks": [(8, 0), (0, -6)],
            "radius_px": 4.0,
        }

    def test_patch_interpolate_gui_state_captured(self):
        state = processing_state_from_gui({
            "patch_interpolate_rect": (3, 4, 8, 9),
            "patch_interpolate_iterations": 50,
        })
        assert len(state.steps) == 1
        assert state.steps[0].op == "patch_interpolate"
        assert state.steps[0].params == {
            "rect": (3, 4, 8, 9),
            "method": "line_fit",
            "iterations": 50,
        }

    def test_patch_interpolate_gui_shape_geometry_captured(self):
        geometry = {
            "kind": "polygon",
            "points_px": [(3, 4), (8, 4), (5, 9)],
        }
        state = processing_state_from_gui({
            "patch_interpolate_geometry": geometry,
            "patch_interpolate_iterations": 50,
        })
        assert len(state.steps) == 1
        assert state.steps[0].op == "patch_interpolate"
        assert state.steps[0].params == {
            "geometry": geometry,
            "method": "line_fit",
            "iterations": 50,
        }

    def test_roi_scope_wraps_soft_border_fft(self):
        gui = {
            "processing_scope": "roi",
            "roi_rect": (1, 2, 12, 14),
            "fft_soft_border": True,
            "fft_soft_mode": "high_pass",
            "fft_soft_cutoff": 0.25,
        }
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 1
        step = state.steps[0]
        assert step.op == "roi"
        assert step.params["step"]["op"] == "fft_soft_border"
        assert step.params["step"]["params"]["mode"] == "high_pass"

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

    def test_set_zero_plane_params_captured(self):
        gui = {"set_zero_plane_points": [(0, 0), (9, 0), (0, 9)], "set_zero_patch": 0}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 1
        step = state.steps[0]
        assert step.op == "set_zero_plane"
        assert step.params == {
            "points_px": [(0, 0), (9, 0), (0, 9)],
            "patch": 0,
        }

    def test_set_zero_plane_requires_three_good_points(self):
        state = processing_state_from_gui({
            "set_zero_plane_points": [(0, 0), "bad", (0, 9)],
        })
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
        from probeflow.processing.gui_adapter import apply_processing_state_to_scan

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

    def test_plane_bg_fit_rect_runs(self):
        y = np.linspace(-1.0, 1.0, 20)
        x = np.linspace(-1.0, 1.0, 20)
        X, Y = np.meshgrid(x, y)
        arr = 2.0 * X - 0.5 * Y + 7.0
        arr[:, 12:] += 25.0
        state = ProcessingState(steps=[
            ProcessingStep("plane_bg", {"order": 1, "fit_rect": (0, 0, 8, 19)}),
        ])
        result = apply_processing_state(arr, state)
        assert float(np.nanstd(result[:, :9])) < 1e-10
        assert abs(float(np.nanmedian(result[:, 12:])) - 25.0) < 1e-10

    def test_stm_line_bg_runs(self):
        arr = np.ones((20, 20), dtype=float)
        arr += np.linspace(0.0, 1.0, 20)[:, None]
        state = ProcessingState(steps=[
            ProcessingStep("stm_line_bg", {"mode": "step_tolerant"}),
        ])
        result = apply_processing_state(arr, state)
        assert result.shape == arr.shape
        assert float(np.std(np.nanmedian(result, axis=1))) < 1e-10

    def test_remove_bad_lines_threshold_is_forwarded(self, monkeypatch):
        captured = {}

        def fake_remove_bad_lines(arr, threshold_mad=5.0, *, method="mad"):
            captured["threshold_mad"] = threshold_mad
            return arr

        monkeypatch.setattr(
            "probeflow.processing.remove_bad_lines",
            fake_remove_bad_lines,
        )
        state = ProcessingState(steps=[
            ProcessingStep("remove_bad_lines", {"threshold_mad": 3.25}),
        ])
        apply_processing_state(np.ones((8, 8)), state)
        assert captured["threshold_mad"] == 3.25

    def test_facet_level_threshold_is_forwarded(self, monkeypatch):
        captured = {}

        def fake_facet_level(arr, threshold_deg=3.0):
            captured["threshold_deg"] = threshold_deg
            return arr

        monkeypatch.setattr("probeflow.processing.facet_level", fake_facet_level)
        state = ProcessingState(steps=[
            ProcessingStep("facet_level", {"threshold_deg": 5.5}),
        ])
        apply_processing_state(np.ones((8, 8)), state)
        assert captured["threshold_deg"] == 5.5

    def test_roi_smooth_changes_only_selected_rectangle(self):
        rng = np.random.default_rng(3)
        arr = np.zeros((16, 16), dtype=float)
        arr[5:11, 5:11] = rng.normal(size=(6, 6))
        state = ProcessingState(steps=[
            ProcessingStep("roi", {
                "rect": (5, 5, 10, 10),
                "step": {"op": "smooth", "params": {"sigma_px": 1.0}},
            }),
        ])
        result = apply_processing_state(arr, state)

        outside = np.ones(arr.shape, dtype=bool)
        outside[5:11, 5:11] = False
        np.testing.assert_array_equal(result[outside], arr[outside])
        assert not np.allclose(result[5:11, 5:11], arr[5:11, 5:11])

    def test_roi_smooth_changes_only_selected_ellipse_mask(self):
        rng = np.random.default_rng(4)
        arr = np.zeros((18, 18), dtype=float)
        arr[4:14, 4:14] = rng.normal(size=(10, 10))
        geometry = {
            "kind": "ellipse",
            "rect_px": (4, 4, 13, 13),
        }
        state = ProcessingState(steps=[
            ProcessingStep("roi", {
                "geometry": geometry,
                "step": {"op": "smooth", "params": {"sigma_px": 1.0}},
            }),
        ])
        result = apply_processing_state(arr, state)

        from probeflow.processing.state import roi_geometry_mask
        mask = roi_geometry_mask(arr.shape, geometry)
        assert mask is not None
        np.testing.assert_array_equal(result[~mask], arr[~mask])
        assert not np.allclose(result[mask], arr[mask])

    def test_roi_geometry_mask_accepts_fractional_geometry(self):
        from probeflow.processing.state import roi_geometry_mask

        rect_mask = roi_geometry_mask((10, 10), {
            "kind": "rectangle",
            "bounds_frac": (0.2, 0.3, 0.6, 0.7),
        })
        polygon_mask = roi_geometry_mask((10, 10), {
            "kind": "polygon",
            "points_frac": [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)],
        })

        assert rect_mask is not None
        assert rect_mask.any()
        assert polygon_mask is not None
        assert polygon_mask.any()

    def test_patch_interpolate_accepts_polygon_mask(self):
        arr = np.ones((16, 16), dtype=float)
        geometry = {
            "kind": "polygon",
            "points_px": [(5, 5), (11, 5), (8, 11)],
        }
        from probeflow.processing.state import roi_geometry_mask
        mask = roi_geometry_mask(arr.shape, geometry)
        assert mask is not None
        arr[mask] = 20.0
        state = ProcessingState(steps=[
            ProcessingStep("patch_interpolate", {
                "geometry": geometry,
                "iterations": 80,
            }),
        ])
        result = apply_processing_state(arr, state)

        np.testing.assert_array_equal(result[~mask], arr[~mask])
        assert float(np.mean(result[mask])) < 5.0

    def test_roi_wrapper_ignores_nonlocal_nested_operation(self):
        x = np.linspace(0, 1, 12)
        arr = np.outer(np.ones(12), x)
        state = ProcessingState(steps=[
            ProcessingStep("roi", {
                "rect": (2, 2, 8, 8),
                "step": {"op": "plane_bg", "params": {"order": 1}},
            }),
        ])
        result = apply_processing_state(arr, state)
        np.testing.assert_array_equal(result, arr)

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

    def test_gaussian_high_pass_runs(self):
        Y, X = np.mgrid[:32, :32]
        arr = 10.0 + 0.1 * X + np.sin(2 * np.pi * X / 4.0)
        state = ProcessingState(steps=[
            ProcessingStep("gaussian_high_pass", {"sigma_px": 8.0}),
        ])
        result = apply_processing_state(arr, state)
        assert result.shape == arr.shape
        assert abs(float(np.mean(result))) < 0.5

    def test_periodic_notch_filter_runs(self):
        Y, X = np.mgrid[:64, :64]
        arr = np.sin(2 * np.pi * X / 8.0)
        state = ProcessingState(steps=[
            ProcessingStep("periodic_notch_filter", {
                "peaks": [(8, 0)],
                "radius_px": 2.0,
            }),
        ])
        result = apply_processing_state(arr, state)
        assert float(np.std(result)) < float(np.std(arr)) * 0.35

    def test_patch_interpolate_rect_runs(self):
        arr = np.ones((16, 16), dtype=float)
        arr[6:10, 6:10] = 20.0
        state = ProcessingState(steps=[
            ProcessingStep("patch_interpolate", {
                "rect": (6, 6, 9, 9),
                "iterations": 80,
            }),
        ])
        result = apply_processing_state(arr, state)
        assert float(np.mean(result[6:10, 6:10])) < 5.0

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

    def test_set_zero_plane_removes_clicked_plane(self):
        yy, xx = np.mgrid[:10, :10]
        arr = 1.25 * xx - 0.5 * yy + 7.0
        state = ProcessingState(steps=[ProcessingStep("set_zero_plane", {
            "points_px": [(0, 0), (9, 0), (0, 9)],
            "patch": 0,
        })])
        result = apply_processing_state(arr, state)

        np.testing.assert_allclose(result, np.zeros_like(arr), atol=1e-12)
