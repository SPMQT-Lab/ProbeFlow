"""Code-audit tests: _SUPPORTED_OPS completeness, gui_adapter coverage,
and import boundaries.

These tests are machine-checkable invariants.  They will fail loudly if any
of the following changes without a corresponding update here:
- a new op is added to _SUPPORTED_OPS without a dispatch branch
- a GUI-adapter path is added for stm_line_bg or facet_level
- the probeflow.analysis public API accidentally exports LatticeCorrection
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from probeflow.processing.state import (
    _SUPPORTED_OPS,
    ProcessingState,
    ProcessingStep,
    apply_processing_state,
)
from probeflow.processing.gui_adapter import processing_state_from_gui


# ── minimal dispatch params for each supported op ─────────────────────────────

_MINIMAL_PARAMS: dict[str, dict] = {
    "remove_bad_lines": {
        "threshold_mad": 5.0, "method": "mad", "polarity": "bright",
        "min_segment_length_px": 2, "max_adjacent_bad_lines": 1,
    },
    "align_rows": {"method": "median"},
    "plane_bg": {"order": 1},
    "stm_line_bg": {"mode": "step_tolerant"},
    "stm_background": {
        "fit_region": "whole_image", "line_statistic": "median",
        "model": "linear", "preserve_level": "median",
    },
    "facet_level": {"threshold_deg": 3.0},
    "smooth": {"sigma_px": 1.0},
    "gaussian_high_pass": {"sigma_px": 8.0},
    "edge_detect": {"method": "laplacian"},
    "fourier_filter": {"mode": "low_pass", "cutoff": 0.1},
    "fft_soft_border": {"mode": "low_pass", "cutoff": 0.1, "border_frac": 0.12},
    "periodic_notch_filter": {"peaks": [(2, 0)], "radius_px": 1.0},
    "linear_undistort": {"shear_x": 0.0, "scale_y": 1.0},
    "affine_lattice_correction": {
        "matrix": [[1.0, 0.0], [0.0, 1.0]], "expand_canvas": False,
    },
    "set_zero_point": {"x_px": 2, "y_px": 2, "patch": 1},
    "set_zero_plane": {"points_px": [[0, 0], [7, 0], [0, 7]], "patch": 1},
    "roi": {
        "step": {"op": "smooth", "params": {"sigma_px": 1.0}},
        "roi_id": "test-roi-id",
    },
    "flip_horizontal": {},
    "flip_vertical": {},
    "rotate_90_cw": {},
    "rotate_180": {},
    "rotate_270_cw": {},
    "rotate_arbitrary": {"angle_degrees": 0.0, "order": 1},
}


# ── _SUPPORTED_OPS completeness ───────────────────────────────────────────────

class TestSupportedOpsCompleteness:
    """Every op in _SUPPORTED_OPS must have a working dispatch branch.

    If an op is added to the frozenset but the dispatch branch is missing,
    apply_processing_state will raise ValueError("Unknown ...").  These tests
    catch that at CI time, not in production.
    """

    @pytest.mark.parametrize("op", sorted(_SUPPORTED_OPS))
    def test_op_dispatches_without_unknown_error(self, op):
        arr = np.linspace(0, 1, 64, dtype=np.float64).reshape(8, 8)
        params = _MINIMAL_PARAMS.get(op, {})
        state = ProcessingState(steps=[ProcessingStep(op, params)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = apply_processing_state(arr, state, roi_set=None)
        assert result.ndim == 2

    def test_all_supported_ops_have_minimal_params_entry(self):
        missing = sorted(_SUPPORTED_OPS - set(_MINIMAL_PARAMS))
        assert missing == [], (
            f"_MINIMAL_PARAMS is missing entries for: {missing}. "
            "Add entries to _MINIMAL_PARAMS to keep this audit complete."
        )

    def test_unknown_op_raises_at_construction(self):
        with pytest.raises(ValueError, match="Unknown"):
            ProcessingStep("definitely_not_a_real_op_xyz", {})

    def test_roi_nesting_guard(self):
        # Manually calling apply with _depth >= 2 should raise
        inner = ProcessingStep("roi", {
            "step": {"op": "smooth", "params": {}},
            "roi_id": "x",
        })
        state = ProcessingState(steps=[inner])
        arr = np.ones((8, 8))
        with pytest.raises(ValueError, match="nesting"):
            apply_processing_state(arr, state, _depth=2)


# ── gui_adapter coverage ──────────────────────────────────────────────────────

class TestGUIAdapterCoverage:
    """Verify which geometric_ops entries do and don't produce processing steps."""

    @pytest.mark.parametrize("op_name", [
        "flip_horizontal",
        "flip_vertical",
        "rotate_90_cw",
        "rotate_180",
        "rotate_270_cw",
    ])
    def test_simple_geometric_op_produces_step(self, op_name):
        gui = {"geometric_ops": [{"op": op_name, "params": {}}]}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 1
        assert state.steps[0].op == op_name

    def test_rotate_arbitrary_produces_step(self):
        gui = {"geometric_ops": [{"op": "rotate_arbitrary", "params": {"angle_degrees": 45.0}}]}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 1
        assert state.steps[0].op == "rotate_arbitrary"

    def test_affine_lattice_correction_produces_step(self):
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        gui = {"geometric_ops": [{
            "op": "affine_lattice_correction",
            "params": {
                "matrix": matrix,
                "expand_canvas": True,
                "interpolation": "bilinear",
                "fill_mode": "nan",
            },
        }]}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 1
        assert state.steps[0].op == "affine_lattice_correction"

    def test_stm_line_bg_in_geometric_ops_produces_no_step(self):
        # stm_line_bg has dispatch in state.py but NO path in gui_adapter.py.
        # Passing it as a geometric_ops entry should be silently skipped.
        gui = {"geometric_ops": [{"op": "stm_line_bg", "params": {"mode": "step_tolerant"}}]}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 0, (
            "stm_line_bg unexpectedly has a gui_adapter path. "
            "Update this test and document the new GUI path."
        )

    def test_facet_level_in_geometric_ops_produces_no_step(self):
        # Same intentional gap as stm_line_bg.
        gui = {"geometric_ops": [{"op": "facet_level", "params": {"threshold_deg": 3.0}}]}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 0, (
            "facet_level unexpectedly has a gui_adapter path. "
            "Update this test and document the new GUI path."
        )

    def test_none_in_geometric_ops_is_silently_skipped(self):
        gui = {"geometric_ops": [None, {"op": "flip_horizontal", "params": {}}]}
        state = processing_state_from_gui(gui)
        # None entry should be skipped; flip_horizontal should remain
        assert len(state.steps) == 1
        assert state.steps[0].op == "flip_horizontal"

    def test_missing_op_key_is_silently_skipped(self):
        gui = {"geometric_ops": [{"params": {}}, {"op": "flip_vertical", "params": {}}]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = processing_state_from_gui(gui)
        # Entry with missing "op" key should be skipped
        assert len(state.steps) == 1
        assert state.steps[0].op == "flip_vertical"

    def test_unrecognised_op_in_geometric_ops_skipped(self):
        gui = {"geometric_ops": [{"op": "warp_space", "params": {}}]}
        state = processing_state_from_gui(gui)
        assert len(state.steps) == 0

    def test_affine_correction_round_trip_expand_canvas(self):
        # affine_lattice_correction with expand_canvas=True goes through
        # gui_adapter → apply_processing_state; output shape may differ
        matrix = np.diag([1.5, 1.0]).tolist()
        gui = {"geometric_ops": [{
            "op": "affine_lattice_correction",
            "params": {"matrix": matrix, "expand_canvas": True, "fill_mode": "nan"},
        }]}
        state = processing_state_from_gui(gui)
        arr = np.ones((16, 16), dtype=np.float64)
        out = apply_processing_state(arr, state)
        # With x-scale 1.5 and expand_canvas, output width should be larger
        assert out.shape[1] > arr.shape[1]


# ── import boundaries ─────────────────────────────────────────────────────────

class TestImportPaths:
    def test_affine_lattice_correction_importable_from_processing(self):
        from probeflow.processing import affine_lattice_correction
        assert callable(affine_lattice_correction)

    def test_lattice_correction_importable_from_lattice_distortion(self):
        from probeflow.analysis.lattice_distortion import LatticeCorrection
        assert LatticeCorrection is not None

    def test_compute_correction_importable(self):
        from probeflow.analysis.lattice_distortion import compute_correction
        assert callable(compute_correction)

    def test_lattice_correction_not_in_probeflow_analysis_public_api(self):
        # LatticeCorrection is intentionally excluded from probeflow.analysis.__all__
        import probeflow.analysis as pa
        assert "LatticeCorrection" not in dir(pa) or not hasattr(pa, "LatticeCorrection"), (
            "LatticeCorrection has been added to probeflow.analysis public API. "
            "If intentional, update this test."
        )

    def test_processing_state_importable(self):
        from probeflow.processing.state import ProcessingState
        assert ProcessingState is not None
