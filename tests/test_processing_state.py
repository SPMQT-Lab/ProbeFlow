"""Contract tests for processing-state serialization, GUI conversion, and apply."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from probeflow.processing.gui_adapter import (
    apply_processing_state_to_scan,
    processing_state_from_gui,
)
from probeflow.processing.arithmetic import generate_arithmetic_pattern
from probeflow.processing.state import (
    ProcessingState,
    ProcessingStep,
    apply_operation_with_optional_roi,
    apply_processing_state,
    missing_roi_references,
    roi_references_from_state,
)


def _ops(state: ProcessingState) -> list[str]:
    return [step.op for step in state.steps]


def _rect_roi_set(*, name: str | None = None):
    from probeflow.core.roi import ROI, ROISet

    roi = ROI.new(
        "rectangle",
        {"x": 5.0, "y": 5.0, "width": 6.0, "height": 6.0},
        name=name,
    )
    roi_set = ROISet(image_id="img")
    roi_set.add(roi)
    return roi_set, roi


def _sample_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(loc=1e-9, scale=1e-10, size=(32, 32))


def _scan_with_plane(plane: np.ndarray, *, path: str = "/tmp/source.sxm"):
    from probeflow.core.scan_model import Scan

    return Scan(
        planes=[np.asarray(plane, dtype=np.float64)],
        plane_names=["Z forward"],
        plane_units=["m"],
        plane_synthetic=[False],
        header={},
        scan_range_m=(1e-9, 1e-9),
        source_path=Path(path),
        source_format="sxm",
    )


def test_empty_state_is_new_float64_identity_without_mutating_input():
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    original = arr.copy()

    result = apply_processing_state(arr, ProcessingState())

    assert result is not arr
    assert result.dtype == np.float64
    assert result.shape == arr.shape
    np.testing.assert_array_equal(arr, original)
    np.testing.assert_allclose(result, original.astype(float))


@pytest.mark.parametrize(
    ("operation", "params", "expected"),
    [
        ("add", {"value_si": 1.5}, [[3.5, 5.5]]),
        ("subtract", {"value_si": 1.5}, [[0.5, 2.5]]),
        ("multiply", {"factor": 2.0}, [[4.0, 8.0]]),
        ("divide", {"factor": 2.0}, [[1.0, 2.0]]),
    ],
)
def test_arithmetic_constant_operations(operation, params, expected):
    arr = np.array([[2.0, 4.0]])
    state = ProcessingState(steps=[
        ProcessingStep("arithmetic", {
            "operation": operation,
            "operand_type": "constant",
            **params,
        }),
    ])

    result = apply_processing_state(arr, state)

    np.testing.assert_allclose(result, np.asarray(expected, dtype=float))


def test_arithmetic_divide_by_zero_raises():
    state = ProcessingState(steps=[
        ProcessingStep("arithmetic", {
            "operation": "divide",
            "operand_type": "constant",
            "factor": 0.0,
        }),
    ])

    with pytest.raises(ValueError, match="divide by zero"):
        apply_processing_state(np.ones((2, 2)), state)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("add", [[11.0, 22.0], [33.0, 44.0]]),
        ("subtract", [[-9.0, -18.0], [-27.0, -36.0]]),
    ],
)
def test_arithmetic_image_add_subtract(monkeypatch, operation, expected):
    operand = np.array([[10.0, 20.0], [30.0, 40.0]])
    monkeypatch.setattr(
        "probeflow.core.scan_loader.load_scan",
        lambda _path: _scan_with_plane(operand),
    )
    state = ProcessingState(steps=[
        ProcessingStep("arithmetic", {
            "operation": operation,
            "operand_type": "image",
            "source_path": "/tmp/source.sxm",
            "plane_idx": 0,
        }),
    ])

    result = apply_processing_state(np.array([[1.0, 2.0], [3.0, 4.0]]), state)

    np.testing.assert_allclose(result, np.asarray(expected, dtype=float))


def test_arithmetic_image_shape_mismatch_raises(monkeypatch):
    monkeypatch.setattr(
        "probeflow.core.scan_loader.load_scan",
        lambda _path: _scan_with_plane(np.ones((3, 3))),
    )
    state = ProcessingState(steps=[
        ProcessingStep("arithmetic", {
            "operation": "add",
            "operand_type": "image",
            "source_path": "/tmp/source.sxm",
            "plane_idx": 0,
        }),
    ])

    with pytest.raises(ValueError, match="operand shape"):
        apply_processing_state(np.ones((2, 2)), state)


def test_generated_arithmetic_checkerboard_pattern_values():
    pattern = generate_arithmetic_pattern(
        (4, 4),
        "checkerboard",
        2.0,
        period_px=2,
    )

    np.testing.assert_allclose(pattern, np.array([
        [2.0, 2.0, -2.0, -2.0],
        [2.0, 2.0, -2.0, -2.0],
        [-2.0, -2.0, 2.0, 2.0],
        [-2.0, -2.0, 2.0, 2.0],
    ]))


def test_generated_arithmetic_ramp_endpoints():
    ramp_x = generate_arithmetic_pattern((2, 3), "ramp_x", 1.0)
    ramp_y = generate_arithmetic_pattern((3, 2), "ramp_y", 1.0)

    np.testing.assert_allclose(ramp_x, np.array([
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ]))
    np.testing.assert_allclose(ramp_y, np.array([
        [-1.0, -1.0],
        [0.0, 0.0],
        [1.0, 1.0],
    ]))


def test_generated_arithmetic_speckle_is_seeded():
    first = generate_arithmetic_pattern((6, 6), "speckle", 0.25, seed=7)
    second = generate_arithmetic_pattern((6, 6), "speckle", 0.25, seed=7)
    third = generate_arithmetic_pattern((6, 6), "speckle", 0.25, seed=8)

    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, third)


def test_generated_arithmetic_add_and_subtract_through_processing_state():
    arr = np.ones((3, 3), dtype=float)
    add_state = ProcessingState(steps=[
        ProcessingStep("arithmetic", {
            "operation": "add",
            "operand_type": "generated",
            "pattern": "impulse_grid",
            "amplitude_si": 4.0,
            "period_px": 2,
        }),
    ])
    subtract_state = ProcessingState(steps=[
        ProcessingStep("arithmetic", {
            "operation": "subtract",
            "operand_type": "generated",
            "pattern": "impulse_grid",
            "amplitude_si": 4.0,
            "period_px": 2,
        }),
    ])

    np.testing.assert_allclose(apply_processing_state(arr, add_state), np.array([
        [5.0, 1.0, 5.0],
        [1.0, 1.0, 1.0],
        [5.0, 1.0, 5.0],
    ]))
    np.testing.assert_allclose(apply_processing_state(arr, subtract_state), np.array([
        [-3.0, 1.0, -3.0],
        [1.0, 1.0, 1.0],
        [-3.0, 1.0, -3.0],
    ]))


def test_roi_scoped_arithmetic_leaves_outside_pixels_unchanged():
    roi_set, roi = _rect_roi_set()
    arr = np.zeros((16, 16), dtype=float)
    state = ProcessingState(steps=[
        ProcessingStep("roi", {
            "roi_id": roi.id,
            "step": {
                "op": "arithmetic",
                "params": {
                    "operation": "add",
                    "operand_type": "constant",
                    "value_si": 5.0,
                },
            },
        }),
    ])

    result = apply_processing_state(arr, state, roi_set=roi_set)
    mask = roi.to_mask(arr.shape)

    assert np.all(result[mask] == 5.0)
    assert np.all(result[~mask] == 0.0)


def test_roi_scoped_generated_arithmetic_leaves_outside_pixels_unchanged():
    roi_set, roi = _rect_roi_set()
    arr = np.zeros((16, 16), dtype=float)
    state = ProcessingState(steps=[
        ProcessingStep("roi", {
            "roi_id": roi.id,
            "step": {
                "op": "arithmetic",
                "params": {
                    "operation": "add",
                    "operand_type": "generated",
                    "pattern": "ramp_x",
                    "amplitude_si": 3.0,
                },
            },
        }),
    ])

    result = apply_processing_state(arr, state, roi_set=roi_set)
    mask = roi.to_mask(arr.shape)
    expected = generate_arithmetic_pattern(arr.shape, "ramp_x", 3.0)

    np.testing.assert_allclose(result[mask], expected[mask])
    assert np.all(result[~mask] == 0.0)


def test_arithmetic_state_serialization_and_gui_conversion():
    state = ProcessingState(steps=[
        ProcessingStep("arithmetic", {
            "operation": "multiply",
            "operand_type": "constant",
            "factor": 2.0,
        }),
    ])
    restored = ProcessingState.from_dict(state.to_dict())

    assert restored.steps[0].op == "arithmetic"
    assert restored.steps[0].params["factor"] == 2.0

    gui_state = {
        "arithmetic_ops": [
            {
                "op": "arithmetic",
                "params": {
                    "operation": "add",
                    "operand_type": "constant",
                    "value_si": 1.0,
                },
            },
            {
                "op": "arithmetic",
                "roi_id": "roi-1",
                "params": {
                    "operation": "subtract",
                    "operand_type": "constant",
                    "value_si": 2.0,
                },
            },
        ],
    }
    converted = processing_state_from_gui(gui_state)

    assert [step.op for step in converted.steps] == ["arithmetic", "roi"]
    assert converted.steps[1].params["roi_id"] == "roi-1"
    assert converted.steps[1].params["step"]["op"] == "arithmetic"


def test_serialisation_contract_preserves_steps_and_copies_nested_params():
    state = ProcessingState(
        steps=[
            ProcessingStep("remove_bad_lines", {"threshold_mad": 5.0}),
            ProcessingStep("plane_bg", {"order": 2, "fit_roi": {"ref": "terrace"}}),
            ProcessingStep("smooth", {"sigma_px": 1.5}),
        ],
    )

    data = state.to_dict()
    assert "steps" in data
    assert "probeflow_version" in data
    json.dumps(data)
    data["steps"][1]["params"]["fit_roi"]["ref"] = "changed"
    assert state.steps[1].params["fit_roi"]["ref"] == "terrace"

    restored = ProcessingState.from_dict(
        {"steps": data["steps"], "ignored_future_key": "ok"},
    )
    assert _ops(restored) == ["remove_bad_lines", "plane_bg", "smooth"]
    assert restored.steps[1].params["fit_roi"]["ref"] == "changed"
    data["steps"][1]["params"]["fit_roi"]["ref"] = "mutated-again"
    assert restored.steps[1].params["fit_roi"]["ref"] == "changed"
    assert ProcessingState.from_dict(ProcessingState().to_dict()).steps == []


def test_roi_reference_collection_and_missing_reference_lookup():
    roi_set, roi = _rect_roi_set(name="terrace")
    state = ProcessingState(
        steps=[
            ProcessingStep(
                "plane_bg",
                {
                    "fit_roi_id": roi.id,
                    "exclude_roi_expr": {"combine": ["terrace", "missing-mask"]},
                },
            ),
            ProcessingStep(
                "roi",
                {
                    "roi_id": "missing-patch",
                    "step": {"op": "smooth", "params": {"sigma_px": 1.0}},
                },
            ),
        ],
    )

    refs = roi_references_from_state(state)
    assert [ref["value"] for ref in refs] == [
        roi.id,
        "terrace",
        "missing-mask",
        "missing-patch",
    ]
    assert [ref["value"] for ref in missing_roi_references(state, roi_set)] == [
        "missing-mask",
        "missing-patch",
    ]
    assert [ref["value"] for ref in missing_roi_references(state, None)] == [
        roi.id,
        "terrace",
        "missing-mask",
        "missing-patch",
    ]


def test_processing_step_rejects_unknown_operations_with_the_operation_name():
    for op in ("magic_filter", "nonexistent_op", "bad_op"):
        with pytest.raises(ValueError, match=op):
            ProcessingStep(op)


def test_gui_conversion_excludes_display_empty_and_false_values():
    gui = {
        "remove_bad_lines": False,
        "align_rows": None,
        "smooth_sigma": None,
        "edge_method": None,
        "fft_mode": None,
        "colormap": "inferno",
        "clip_low": 1.0,
        "clip_high": 99.0,
        "grain_threshold": 50.0,
        "grain_above": True,
    }

    state = processing_state_from_gui(gui)

    assert state.steps == []


def test_gui_conversion_emits_ordered_processing_steps_and_parameters():
    gui = {
        "remove_bad_lines": "step",
        "remove_bad_lines_threshold": 7.5,
        "remove_bad_lines_polarity": "dark",
        "remove_bad_lines_min_segment_length_px": 8,
        "remove_bad_lines_max_adjacent_bad_lines": 2,
        "align_rows": "mean",
        "smooth_sigma": 1.5,
        "stm_background": {
            "fit_region": "active_roi",
            "fit_roi_id": "terrace-1",
            "line_statistic": "mean",
            "model": "poly2",
            "linear_x_first": True,
            "blur_length": 6.0,
            "jump_threshold": 2.5,
        },
    }

    state = processing_state_from_gui(gui)

    assert _ops(state) == ["remove_bad_lines", "stm_background", "align_rows", "smooth"]
    assert state.steps[0].params == {
        "threshold_mad": 7.5,
        "method": "step",
        "polarity": "dark",
        "min_segment_length_px": 8,
        "max_adjacent_bad_lines": 2,
    }
    assert state.steps[1].params == {
        "fit_region": "active_roi",
        "line_statistic": "mean",
        "model": "poly2",
        "linear_x_first": True,
        "preserve_level": "median",
        "blur_length": 6.0,
        "jump_threshold": 2.5,
        "fit_roi_id": "terrace-1",
        "applied_to": "whole_image",
    }
    assert state.steps[2].params == {"method": "mean"}
    assert state.steps[3].params == {"sigma_px": 1.5}


def test_gui_conversion_emits_fft_edge_frequency_and_geometry_steps():
    cases = [
        (
            {"edge_method": "laplacian", "edge_sigma": 2.0, "edge_sigma2": 3.0},
            "edge_detect",
            {"method": "laplacian", "sigma": 2.0, "sigma2": 3.0},
        ),
        (
            {"fft_mode": "low_pass", "fft_cutoff": 0.15, "fft_window": "hanning"},
            "fourier_filter",
            {"mode": "low_pass", "cutoff": 0.15, "window": "hanning"},
        ),
        ({"highpass_sigma": 12}, "gaussian_high_pass", {"sigma_px": 12.0}),
        (
            {"periodic_notches": [(8, 0), ("bad", 2), (0, -6)], "periodic_notch_radius": 4},
            "periodic_notch_filter",
            {"peaks": [(8, 0), (0, -6)], "radius_px": 4.0},
        ),
        (
            {
                "fft_soft_border": True,
                "fft_soft_mode": "high_pass",
                "fft_soft_cutoff": 0.20,
                "fft_soft_border_frac": 0.05,
            },
            "fft_soft_border",
            {"mode": "high_pass", "cutoff": 0.20, "border_frac": 0.05},
        ),
        (
            {"linear_undistort": True, "undistort_shear_x": 1.5},
            "linear_undistort",
            {"shear_x": 1.5, "scale_y": 1.0},
        ),
    ]

    for gui, expected_op, expected_params in cases:
        state = processing_state_from_gui(gui)
        assert _ops(state) == [expected_op]
        assert state.steps[0].params == expected_params

    assert processing_state_from_gui({"linear_undistort": True}).steps == []


def test_gui_conversion_emits_zero_reference_steps_and_skips_bad_inputs():
    point = processing_state_from_gui({"set_zero_xy": (10, 20), "set_zero_patch": 3})
    plane = processing_state_from_gui(
        {"set_zero_plane_points": [(0, 0), (9, 0), (0, 9)], "set_zero_patch": 0},
    )

    assert _ops(point) == ["set_zero_point"]
    assert point.steps[0].params == {"x_px": 10, "y_px": 20, "patch": 3}
    assert _ops(plane) == ["set_zero_plane"]
    assert plane.steps[0].params == {
        "points_px": [(0, 0), (9, 0), (0, 9)],
        "patch": 0,
    }
    with pytest.warns(UserWarning, match="set_zero_point"):
        assert processing_state_from_gui({"set_zero_xy": "bad"}).steps == []
    with pytest.warns(UserWarning, match="set_zero_plane"):
        assert processing_state_from_gui(
            {"set_zero_plane_points": [(0, 0), "bad", (0, 9)]},
        ).steps == []


def test_gui_conversion_warns_when_step_specs_are_skipped():
    with pytest.warns(UserWarning, match="affine_lattice_correction"):
        state = processing_state_from_gui({
            "geometric_ops": [{"op": "affine_lattice_correction", "params": {}}],
        })
    assert state.steps == []

    with pytest.warns(UserWarning, match="arithmetic"):
        state = processing_state_from_gui({"arithmetic_ops": ["bad"]})
    assert state.steps == []


def test_gui_conversion_roi_scope_wraps_local_steps_and_keeps_global_steps_global():
    state = processing_state_from_gui(
        {
            "processing_scope": "roi",
            "processing_roi_id": "roi-123",
            "align_rows": "median",
            "stm_background": {
                "fit_region": "whole_image",
                "line_statistic": "median",
                "model": "linear",
            },
            "smooth_sigma": 1.0,
            "fft_soft_border": True,
            "fft_soft_mode": "high_pass",
            "fft_soft_cutoff": 0.25,
        },
    )

    assert _ops(state) == ["stm_background", "align_rows", "roi", "roi"]
    local_steps = [step.params["step"] for step in state.steps if step.op == "roi"]
    assert [step["op"] for step in local_steps] == ["smooth", "fft_soft_border"]
    assert [step.params["roi_id"] for step in state.steps if step.op == "roi"] == [
        "roi-123",
        "roi-123",
    ]

    with pytest.warns(UserWarning, match="ROI-scoped processing"):
        assert processing_state_from_gui({"processing_scope": "roi", "smooth_sigma": 1.0}).steps == []


def test_gui_conversion_runs_background_before_row_alignment():
    from probeflow.processing import align_rows, subtract_background

    yy, xx = np.mgrid[:64, :64]
    arr = 5.0 + 0.03 * xx + 0.2 * yy
    arr = arr.astype(np.float64)
    arr[24:40, 24:40] += 4.0

    state = processing_state_from_gui({
        "align_rows": "median",
        "plane_bg": {"order": 1},
    })

    result = apply_processing_state(arr, state)
    background_first = align_rows(
        subtract_background(arr, order=1),
        method="median",
    )
    legacy_align_first = subtract_background(
        align_rows(arr, method="median"),
        order=1,
    )

    assert _ops(state) == ["plane_bg", "align_rows"]
    np.testing.assert_allclose(result, background_first, atol=1e-12)
    assert np.nanmax(np.abs(result - legacy_align_first)) > 0.1


def test_gui_preview_export_and_scan_paths_share_processing_state_results():
    from probeflow.gui import _apply_processing

    arr = _sample_image()
    gui = {"align_rows": "median", "smooth_sigma": 0.75}
    state = processing_state_from_gui(gui)

    np.testing.assert_array_almost_equal(
        _apply_processing(arr, gui),
        apply_processing_state(arr, state),
    )
    original = arr.copy()
    _apply_processing(arr, {"remove_bad_lines": True, "align_rows": "median"})
    apply_processing_state(arr, state)
    np.testing.assert_array_equal(arr, original)

    scan = MagicMock()
    scan.planes = [arr.copy()]
    scan.processing_history = []
    apply_processing_state_to_scan(scan, gui, plane_idx=0)
    np.testing.assert_array_almost_equal(scan.planes[0], apply_processing_state(arr, state))


def test_core_processing_steps_preserve_shape_dtype_and_expected_effects():
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(40, 40))
    tilt = np.outer(np.ones(30), np.linspace(0, 1, 30))
    constant = np.ones((20, 20)) * 5.0

    aligned = apply_processing_state(
        noise,
        ProcessingState([ProcessingStep("align_rows", {"method": "median"})]),
    )
    smoothed = apply_processing_state(
        noise,
        ProcessingState([ProcessingStep("smooth", {"sigma_px": 2.0})]),
    )
    flattened = apply_processing_state(
        tilt,
        ProcessingState([ProcessingStep("plane_bg", {"order": 1})]),
    )
    apply_processing_state(
        constant,
        ProcessingState(
            [
                ProcessingStep("align_rows", {"method": "median"}),
                ProcessingStep("plane_bg", {"order": 1}),
            ],
        ),
    )

    assert aligned.shape == noise.shape
    assert aligned.dtype == np.float64
    assert float(np.std(smoothed)) < float(np.std(noise))
    assert float(np.std(flattened)) < 1e-10
    np.testing.assert_array_equal(constant, np.ones((20, 20)) * 5.0)


def test_processing_forwards_parameters_to_backend_operations(monkeypatch):
    captured: dict[str, object] = {}

    def fake_subtract_background(input_arr, **kwargs):
        captured["plane_bg"] = kwargs
        return input_arr + 7.0

    def fake_apply_stm_background(arr, params=None, mask=None):
        captured["stm_background"] = (params, mask)
        return arr + 1.0

    def fake_remove_bad_lines(
        arr,
        threshold_mad=5.0,
        *,
        method="mad",
        polarity="bright",
        min_segment_length_px=2,
        max_adjacent_bad_lines=1,
    ):
        captured["remove_bad_lines"] = {
            "threshold_mad": threshold_mad,
            "method": method,
            "polarity": polarity,
            "min_segment_length_px": min_segment_length_px,
            "max_adjacent_bad_lines": max_adjacent_bad_lines,
        }
        return arr

    def fake_facet_level(arr, threshold_deg=3.0):
        captured["facet_level"] = threshold_deg
        return arr

    monkeypatch.setattr("probeflow.processing.subtract_background", fake_subtract_background)
    monkeypatch.setattr("probeflow.processing.apply_stm_background", fake_apply_stm_background)
    monkeypatch.setattr("probeflow.processing.remove_bad_lines", fake_remove_bad_lines)
    monkeypatch.setattr("probeflow.processing.facet_level", fake_facet_level)

    arr = np.ones((8, 8), dtype=float)
    result = apply_processing_state(
        arr,
        ProcessingState(
            [
                ProcessingStep("plane_bg", {"order": 1, "step_tolerance": True}),
                ProcessingStep(
                    "stm_background",
                    {
                        "fit_region": "whole_image",
                        "line_statistic": "mean",
                        "model": "poly3",
                        "linear_x_first": True,
                        "blur_length": 7.0,
                        "jump_threshold": 2.0,
                        "preserve_level": "mean",
                    },
                ),
                ProcessingStep(
                    "remove_bad_lines",
                    {
                        "threshold_mad": 3.25,
                        "polarity": "dark",
                        "min_segment_length_px": 8,
                        "max_adjacent_bad_lines": 2,
                    },
                ),
                ProcessingStep("facet_level", {"threshold_deg": 5.5}),
            ],
        ),
    )

    assert result.shape == arr.shape
    assert result.dtype == np.float64
    assert captured["plane_bg"] == {
        "order": 1,
        "step_tolerance": True,
        "fit_rect": None,
        "fit_roi": None,
        "exclude_roi": None,
        "apply_roi": None,
    }
    params, mask = captured["stm_background"]
    assert params.fit_region == "whole_image"
    assert params.line_statistic == "mean"
    assert params.model == "poly3"
    assert params.linear_x_first is True
    assert params.blur_length == 7.0
    assert params.jump_threshold == 2.0
    assert params.preserve_level == "mean"
    assert mask is None
    assert captured["remove_bad_lines"] == {
        "threshold_mad": 3.25,
        "method": "mad",
        "polarity": "dark",
        "min_segment_length_px": 8,
        "max_adjacent_bad_lines": 2,
    }
    assert captured["facet_level"] == 5.5


def test_stm_and_fit_rect_backgrounds_run_on_expected_regions():
    y = np.linspace(-1.0, 1.0, 20)
    x = np.linspace(-1.0, 1.0, 20)
    grid_x, grid_y = np.meshgrid(x, y)
    arr = 2.0 * grid_x - 0.5 * grid_y + 7.0
    arr[:, 12:] += 25.0

    fit_rect_result = apply_processing_state(
        arr,
        ProcessingState([ProcessingStep("plane_bg", {"order": 1, "fit_rect": (0, 0, 8, 19)})]),
    )
    assert float(np.nanstd(fit_rect_result[:, :9])) < 1e-10
    assert abs(float(np.nanmedian(fit_rect_result[:, 12:])) - 25.0) < 1e-10

    line_bg_source = np.ones((20, 20), dtype=float) + np.linspace(0.0, 1.0, 20)[:, None]
    line_bg_result = apply_processing_state(
        line_bg_source,
        ProcessingState([ProcessingStep("stm_line_bg", {"mode": "step_tolerant"})]),
    )
    assert float(np.std(np.nanmedian(line_bg_result, axis=1))) < 1e-10

    yy, _xx = np.mgrid[:30, :20]
    roi_source = 0.2 * yy + np.zeros((30, 20))
    roi_source[:, 12:] += 10.0
    roi_set, roi = _rect_roi_set()
    roi.geometry = {"x": 0, "y": 0, "width": 8, "height": 30}
    roi_bg_result = apply_processing_state(
        roi_source,
        ProcessingState(
            [
                ProcessingStep(
                    "stm_background",
                    {
                        "fit_region": "active_roi",
                        "fit_roi_id": roi.id,
                        "line_statistic": "median",
                        "model": "linear",
                    },
                ),
            ],
        ),
        roi_set=roi_set,
    )
    assert float(np.nanstd(np.nanmedian(roi_bg_result[:, :8], axis=1))) < 1e-10
    assert float(np.nanstd(np.nanmedian(roi_bg_result[:, 12:], axis=1))) < 1e-10
    assert abs(
        float(np.nanmedian(roi_bg_result[:, 12:]))
        - float(np.nanmedian(roi_bg_result[:, :8]))
        - 10.0
    ) < 1e-10


def test_optional_roi_application_and_roi_wrapped_processing_contract():
    arr = np.arange(36, dtype=float).reshape(6, 6)
    mask = np.zeros(arr.shape, dtype=bool)
    mask[2:5, 1:4] = True

    masked = apply_operation_with_optional_roi(arr, lambda image: image + 100.0, mask)
    global_result = apply_operation_with_optional_roi(arr, lambda image: image + 10.0, None)

    np.testing.assert_array_equal(masked[~mask], arr[~mask])
    np.testing.assert_array_equal(masked[mask], arr[mask] + 100.0)
    np.testing.assert_array_equal(global_result, arr + 10.0)

    rng = np.random.default_rng(5)
    roi_source = np.zeros((16, 16), dtype=float)
    roi_source[5:11, 5:11] = rng.normal(size=(6, 6))
    roi_set, roi = _rect_roi_set(name="terrace")
    for roi_ref in (roi.id, "terrace"):
        state = ProcessingState(
            [
                ProcessingStep(
                    "roi",
                    {
                        "roi_id": roi_ref,
                        "step": {"op": "smooth", "params": {"sigma_px": 1.0}},
                    },
                ),
            ],
        )
        assert missing_roi_references(state, roi_set) == []
        result = apply_processing_state(roi_source, state, roi_set=roi_set)
        roi_mask = roi.to_mask(roi_source.shape)
        np.testing.assert_array_equal(result[~roi_mask], roi_source[~roi_mask])
        assert not np.allclose(result[roi_mask], roi_source[roi_mask])

    nested_global = ProcessingState(
        [
            ProcessingStep(
                "roi",
                {
                    "roi_id": roi.id,
                    "step": {"op": "plane_bg", "params": {"order": 1}},
                },
            ),
        ],
    )
    np.testing.assert_array_equal(
        apply_processing_state(roi_source, nested_global, roi_set=roi_set),
        roi_source,
    )


def test_frequency_and_linear_geometry_processing_contract():
    rng = np.random.default_rng(2)
    noise = rng.normal(size=(32, 32))
    soft = apply_processing_state(
        noise,
        ProcessingState(
            [ProcessingStep("fft_soft_border", {"mode": "low_pass", "cutoff": 0.20, "border_frac": 0.10})],
        ),
    )
    assert soft.shape == noise.shape
    assert float(np.std(soft)) < float(np.std(noise))

    yy, xx = np.mgrid[:32, :32]
    highpass_source = 10.0 + 0.1 * xx + np.sin(2 * np.pi * xx / 4.0)
    highpass = apply_processing_state(
        highpass_source,
        ProcessingState([ProcessingStep("gaussian_high_pass", {"sigma_px": 8.0})]),
    )
    assert highpass.shape == highpass_source.shape
    assert abs(float(np.mean(highpass))) < 0.5

    notch_source = np.sin(2 * np.pi * np.mgrid[:64, :64][1] / 8.0)
    notch = apply_processing_state(
        notch_source,
        ProcessingState([ProcessingStep("periodic_notch_filter", {"peaks": [(8, 0)], "radius_px": 2.0})]),
    )
    assert float(np.std(notch)) < float(np.std(notch_source)) * 0.35

    yy, _xx = np.indices((20, 20), dtype=float)
    undistorted = apply_processing_state(
        yy,
        ProcessingState([ProcessingStep("linear_undistort", {"shear_x": 0.0, "scale_y": 2.0})]),
    )
    np.testing.assert_allclose(undistorted, yy / 2.0, atol=1e-12)


def test_zero_reference_processing_contract():
    arr = np.full((10, 10), 42.0)
    point = apply_processing_state(
        arr,
        ProcessingState([ProcessingStep("set_zero_point", {"x_px": 4, "y_px": 5, "patch": 1})]),
    )
    assert abs(float(point[5, 4])) < 1e-12
    np.testing.assert_array_almost_equal(point, arr - 42.0)

    yy, xx = np.mgrid[:10, :10]
    plane_source = 1.25 * xx - 0.5 * yy + 7.0
    plane = apply_processing_state(
        plane_source,
        ProcessingState(
            [ProcessingStep("set_zero_plane", {"points_px": [(0, 0), (9, 0), (0, 9)], "patch": 0})],
        ),
    )
    np.testing.assert_allclose(plane, np.zeros_like(plane_source), atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# scan_range_m bookkeeping through shape-changing steps (review image-proc #4)
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyProcessingStateWithCalibration:
    """``apply_processing_state_with_calibration`` must keep
    ``scan_range_m / shape`` consistent across shape-changing steps."""

    def test_no_steps_passes_scan_range_through(self):
        from probeflow.processing.state import apply_processing_state_with_calibration

        arr = np.ones((8, 8), dtype=np.float64)
        out, new_range = apply_processing_state_with_calibration(
            arr, ProcessingState(), scan_range_m=(2e-9, 4e-9),
        )
        assert out.shape == (8, 8)
        assert new_range == (2e-9, 4e-9)

    def test_scale_image_preserves_extent(self):
        """scale_image keeps the physical extent fixed; pixel size scales."""
        from probeflow.processing.state import apply_processing_state_with_calibration

        arr = np.ones((4, 4), dtype=np.float64)
        state = ProcessingState(steps=[
            ProcessingStep("scale_image", {"new_height": 8, "new_width": 12}),
        ])
        out, new_range = apply_processing_state_with_calibration(
            arr, state, scan_range_m=(3e-9, 5e-9),
        )
        # New shape must be (8, 12); scan_range_m must be unchanged so the
        # downstream pixel_size = scan_range_m / shape halves correctly.
        assert out.shape == (8, 12)
        np.testing.assert_allclose(new_range, (3e-9, 5e-9))

    def test_rotate_arbitrary_grows_canvas_and_extent(self):
        """rotate_arbitrary preserves pixel size; scan_range_m must grow."""
        from probeflow.processing.state import apply_processing_state_with_calibration

        rng = np.random.default_rng(0)
        arr = rng.normal(size=(32, 32))
        state = ProcessingState(steps=[
            ProcessingStep("rotate_arbitrary", {"angle_degrees": 30.0, "order": 1}),
        ])
        psx_in = 1e-9 / 32  # pixel size before
        psy_in = 1e-9 / 32
        out, new_range = apply_processing_state_with_calibration(
            arr, state, scan_range_m=(1e-9, 1e-9),
        )
        h_out, w_out = out.shape
        assert (h_out, w_out) != (32, 32)
        # The post-processing pixel size derived from the returned scan_range
        # must equal the pre-processing pixel size (canvas expanded by exactly
        # the new shape factor).
        psx_out = new_range[0] / w_out
        psy_out = new_range[1] / h_out
        np.testing.assert_allclose(psx_out, psx_in, rtol=1e-12)
        np.testing.assert_allclose(psy_out, psy_in, rtol=1e-12)

    def test_shear_with_canvas_expansion_grows_extent(self):
        from probeflow.processing.state import apply_processing_state_with_calibration

        arr = np.zeros((20, 20), dtype=np.float64)
        state = ProcessingState(steps=[
            ProcessingStep("shear", {"shear_x": 0.5, "shear_y": 0.0}),
        ])
        psx_in = 2e-9 / 20
        out, new_range = apply_processing_state_with_calibration(
            arr, state, scan_range_m=(2e-9, 2e-9),
        )
        assert out.shape[1] > 20  # canvas expanded along x
        np.testing.assert_allclose(new_range[0] / out.shape[1], psx_in, rtol=1e-12)

    def test_non_shape_changing_steps_leave_extent_alone(self):
        from probeflow.processing.state import apply_processing_state_with_calibration

        arr = np.linspace(0.0, 1.0, 100, dtype=np.float64).reshape(10, 10)
        state = ProcessingState(steps=[
            ProcessingStep("align_rows", {"method": "median"}),
            ProcessingStep("plane_bg", {"order": 1}),
        ])
        out, new_range = apply_processing_state_with_calibration(
            arr, state, scan_range_m=(7e-9, 9e-9),
        )
        assert out.shape == (10, 10)
        np.testing.assert_allclose(new_range, (7e-9, 9e-9))

    def test_none_scan_range_propagates(self):
        from probeflow.processing.state import apply_processing_state_with_calibration

        arr = np.ones((6, 6), dtype=np.float64)
        state = ProcessingState(steps=[
            ProcessingStep("rotate_arbitrary", {"angle_degrees": 45.0, "order": 1}),
        ])
        out, new_range = apply_processing_state_with_calibration(
            arr, state, scan_range_m=None,
        )
        assert out.shape != (6, 6)
        assert new_range is None

    def test_apply_processing_state_to_scan_updates_scan_range_after_rotate(self):
        """The GUI-export entry point must update scan.scan_range_m too."""

        arr = np.ones((16, 16), dtype=np.float64)
        scan = _scan_with_plane(arr)
        scan.scan_range_m = (4e-9, 4e-9)
        psx_in = 4e-9 / 16
        apply_processing_state_to_scan(
            scan,
            {"geometric_ops": [
                {"op": "rotate_arbitrary",
                 "params": {"angle_degrees": 30.0, "order": 1}},
            ]},
        )
        h_out, w_out = scan.planes[0].shape
        assert (h_out, w_out) != (16, 16)
        np.testing.assert_allclose(scan.scan_range_m[0] / w_out, psx_in, rtol=1e-12)
        np.testing.assert_allclose(scan.scan_range_m[1] / h_out, psx_in, rtol=1e-12)

    def test_apply_processing_state_to_scan_preserves_extent_after_scale(self):
        """scale_image must leave scan_range_m unchanged."""

        arr = np.ones((8, 8), dtype=np.float64)
        scan = _scan_with_plane(arr)
        scan.scan_range_m = (3e-9, 5e-9)
        apply_processing_state_to_scan(
            scan,
            {"geometric_ops": [
                {"op": "scale_image", "params": {"new_height": 32, "new_width": 24}},
            ]},
        )
        assert scan.planes[0].shape == (32, 24)
        np.testing.assert_allclose(scan.scan_range_m, (3e-9, 5e-9))


def test_stm_background_active_roi_without_roi_set_degrades_to_whole_image():
    """A constant-Δf/STM background step fit over an ROI must not crash a
    thumbnail/preview render that has no roi_set: the unresolved ROI mask
    degrades to a whole-image fit instead of raising in the kernel."""

    rng = np.random.default_rng(0)
    arr = rng.standard_normal((24, 24)).astype(np.float64) + np.linspace(0, 5, 24)

    state = ProcessingState(
        [
            ProcessingStep(
                "stm_background",
                {
                    "fit_region": "active_roi",
                    "fit_roi_id": "be8e9cc6-84b2-4e4d-9990-24bdc4476258",
                    "model": "linear",
                    "line_statistic": "median",
                },
            )
        ]
    )

    with pytest.warns(UserWarning, match="ROI fit mask ignored"):
        out = apply_processing_state(arr, state, roi_set=None)

    assert out.shape == arr.shape
    assert np.isfinite(out).all()


# ── Durable + frozen ROI-scoped filters (P1) ──────────────────────────────────


def _rect_frozen(x, y, w, h):
    return {
        "kind": "rectangle",
        "geometry": {"x": x, "y": y, "width": w, "height": h},
        "coord_system": "pixel",
    }


def test_two_roi_scoped_filters_coexist_via_durable_list():
    """Two ROI-scoped blurs on different regions both apply (no overwrite)."""
    rng = np.random.RandomState(0)
    arr = rng.rand(20, 20)
    gui = {"roi_filter_ops": [
        {"op": "smooth", "params": {"sigma_px": 2.0}, "roi_id": "A",
         "frozen_geometry": _rect_frozen(0, 0, 8, 8)},
        {"op": "smooth", "params": {"sigma_px": 2.0}, "roi_id": "B",
         "frozen_geometry": _rect_frozen(12, 12, 8, 8)},
    ]}
    state = processing_state_from_gui(gui)
    assert _ops(state) == ["roi", "roi"]
    out = apply_processing_state(arr, state)  # no roi_set: frozen geometry
    assert not np.allclose(out[0:8, 0:8], arr[0:8, 0:8])      # A blurred
    assert not np.allclose(out[12:20, 12:20], arr[12:20, 12:20])  # B blurred
    assert np.allclose(out[9:11, 9:11], arr[9:11, 9:11])      # centre untouched


def test_frozen_roi_filter_does_not_follow_live_roi_and_is_not_missing():
    """Frozen-geometry steps ignore roi_set, so a moved/deleted ROI is fine."""
    from probeflow.core.roi import ROI, ROISet
    rng = np.random.RandomState(1)
    arr = rng.rand(20, 20)
    gui = {"roi_filter_ops": [
        {"op": "smooth", "params": {"sigma_px": 2.0}, "roi_id": "A",
         "frozen_geometry": _rect_frozen(0, 0, 8, 8)},
    ]}
    state = processing_state_from_gui(gui)
    # Live ROI of the same id sits elsewhere; frozen geometry must win.
    rs = ROISet(image_id="img")
    rs.add(ROI(id="A", name="A", kind="rectangle",
               geometry={"x": 12, "y": 12, "width": 8, "height": 8}))
    out = apply_processing_state(arr, state, roi_set=rs)
    assert not np.allclose(out[0:8, 0:8], arr[0:8, 0:8])          # frozen region
    assert np.allclose(out[12:20, 12:20], arr[12:20, 12:20])      # live region untouched
    # Frozen steps are self-resolving — never reported missing.
    assert missing_roi_references(state, None) == []
    assert missing_roi_references(state, ROISet(image_id="img")) == []


def test_roi_scoped_step_carries_masked_paste_semantics_marker():
    gui = {"roi_filter_ops": [
        {"op": "smooth", "params": {"sigma_px": 1.0}, "roi_id": "A",
         "frozen_geometry": _rect_frozen(0, 0, 4, 4)},
    ]}
    state = processing_state_from_gui(gui)
    assert state.steps[0].params["scope_semantics"] == "full_image_compute_masked_paste"


# ── Mask as a first-class processing scope (P2) ───────────────────────────────


def test_mask_step_applies_filter_inside_mask_only():
    from probeflow.core.mask import ImageMask, MaskSet
    rng = np.random.RandomState(2)
    arr = rng.rand(20, 20)
    m = np.zeros((20, 20), bool)
    m[2:6, 2:6] = True
    img = ImageMask.new(m, name="mm")
    ms = MaskSet(image_id="img")
    ms.add(img)
    gui = {"mask_filter_ops": [
        {"op": "smooth", "params": {"sigma_px": 2.0}, "mask_id": img.id},
    ]}
    state = processing_state_from_gui(gui)
    assert _ops(state) == ["mask"]
    out = apply_processing_state(arr, state, mask_set=ms)
    assert not np.allclose(out[2:6, 2:6], arr[2:6, 2:6])      # inside mask blurred
    assert np.allclose(out[10:12, 10:12], arr[10:12, 10:12])  # outside untouched


def test_mask_step_missing_when_no_mask_set():
    from probeflow.core.mask import ImageMask, MaskSet
    arr = np.zeros((10, 10))
    img = ImageMask.new(np.ones((10, 10), bool), name="mm")
    ms = MaskSet(image_id="img")
    ms.add(img)
    gui = {"mask_filter_ops": [
        {"op": "smooth", "params": {"sigma_px": 1.0}, "mask_id": img.id},
    ]}
    state = processing_state_from_gui(gui)
    miss = missing_roi_references(state, None, None)
    assert miss and miss[0]["param"] == "mask_id"
    assert missing_roi_references(state, None, ms) == []
    # Skipped (no-op) with a warning when mask_set is absent at apply time.
    with pytest.warns(UserWarning, match="no mask_set"):
        out = apply_processing_state(arr, state, mask_set=None)
    assert out.shape == arr.shape
