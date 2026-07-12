"""Tests for the Tier-1 standard image ops: median filter, crop, remove spots."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.core.mask import _pack_bool
from probeflow.processing.filters import median_smooth
from probeflow.processing.geometry import crop
from probeflow.processing.repair import interpolate_masked
from probeflow.processing.gui_adapter import processing_state_from_gui
from probeflow.processing.state import (
    ProcessingState,
    ProcessingStep,
    apply_processing_state,
    apply_processing_state_with_calibration,
)


# ── median_smooth ─────────────────────────────────────────────────────────────

def test_median_removes_single_pixel_spike_and_keeps_step_edge():
    arr = np.zeros((10, 10))
    arr[:, 5:] = 1.0          # a step edge
    arr[2, 2] = 100.0         # a salt spike
    out = median_smooth(arr, size_px=3)
    assert out[2, 2] == 0.0                       # spike gone
    assert np.array_equal(out[:, :4], np.zeros((10, 4)))
    assert np.array_equal(out[:, 6:], np.ones((10, 4)))  # edge not blurred away


def test_median_preserves_nan_and_excludes_it_from_windows():
    arr = np.ones((6, 6))
    arr[3, 3] = np.nan
    out = median_smooth(arr, size_px=3)
    assert np.isnan(out[3, 3])                    # NaN stays NaN
    assert np.all(out[np.isfinite(out)] == 1.0)   # neighbours unaffected


def test_median_size_is_clamped_odd():
    arr = np.random.default_rng(0).normal(size=(8, 8))
    # Even size bumps to the next odd; tiny sizes clamp to 3.
    assert np.array_equal(median_smooth(arr, 4), median_smooth(arr, 5))
    assert np.array_equal(median_smooth(arr, 1), median_smooth(arr, 3))


def test_median_via_processing_state():
    arr = np.zeros((8, 8))
    arr[4, 4] = 50.0
    state = ProcessingState(steps=[ProcessingStep("median_smooth", {"size_px": 3})])
    out = apply_processing_state(arr, state)
    assert out[4, 4] == 0.0


# ── crop ──────────────────────────────────────────────────────────────────────

def test_crop_values_and_shape():
    arr = np.arange(100, dtype=float).reshape(10, 10)
    out = crop(arr, 2, 3, 5, 7)
    assert out.shape == (5, 4)
    assert out[0, 0] == arr[3, 2]
    assert out[-1, -1] == arr[7, 5]


def test_crop_swaps_and_clamps_bounds():
    arr = np.arange(64, dtype=float).reshape(8, 8)
    assert np.array_equal(crop(arr, 5, 6, 2, 1), crop(arr, 2, 1, 5, 6))
    out = crop(arr, -3, -3, 100, 100)  # clamps to the full image
    assert np.array_equal(out, arr)


def test_crop_outside_image_raises():
    arr = np.zeros((8, 8))
    with pytest.raises(ValueError):
        crop(arr, 20, 20, 30, 30)


def test_crop_preserves_pixel_size_through_calibration():
    arr = np.zeros((100, 100))
    state = ProcessingState(steps=[
        ProcessingStep("crop", {"x0": 10, "y0": 20, "x1": 59, "y1": 59}),
    ])
    out, new_range = apply_processing_state_with_calibration(
        arr, state, scan_range_m=(100e-9, 100e-9),
    )
    assert out.shape == (40, 50)
    # 1 nm/px before; extent shrinks with the pixel count.
    assert new_range[0] == pytest.approx(50e-9)
    assert new_range[1] == pytest.approx(40e-9)


def test_crop_step_with_stale_bounds_is_skipped_not_fatal():
    arr = np.zeros((8, 8))
    state = ProcessingState(steps=[
        ProcessingStep("crop", {"x0": 50, "y0": 50, "x1": 60, "y1": 60}),
        ProcessingStep("smooth", {"sigma_px": 1.0}),
    ])
    with pytest.warns(UserWarning, match="crop step skipped"):
        out = apply_processing_state(arr, state)
    assert out.shape == (8, 8)  # crop skipped, pipeline continued


def test_crop_translates_rois_and_masks():
    from probeflow.core.roi import ROI, ROISet

    roi_set = ROISet(image_id="scan")
    inside = ROI.new("rectangle", {"x": 4, "y": 4, "width": 2, "height": 2})
    outside = ROI.new("rectangle", {"x": 0, "y": 0, "width": 1, "height": 1})
    roi_set.add(inside)
    roi_set.add(outside)
    params = {"x0": 3, "y0": 3, "x1": 9, "y1": 9}
    invalidated = roi_set.transform_all("crop", params, (12, 12))
    assert outside.id in invalidated
    moved = roi_set.get(inside.id)
    assert moved.geometry["x"] == pytest.approx(1)  # 4 - 3
    assert moved.geometry["y"] == pytest.approx(1)


# ── interpolate_masked (remove spots) ─────────────────────────────────────────

def test_interpolate_masked_reconstructs_linear_ramp_exactly():
    # A plane is harmonic: Laplace interpolation must reproduce it exactly.
    yy, xx = np.mgrid[0:12, 0:12]
    arr = 2.0 * xx + 3.0 * yy
    original = arr.copy()
    mask = np.zeros_like(arr, dtype=bool)
    mask[4:8, 5:9] = True
    arr[mask] = 999.0  # a big "defect"
    out = interpolate_masked(arr, mask)
    assert np.allclose(out, original, atol=1e-8)


def test_interpolate_masked_fills_nan_under_mask_only():
    arr = np.ones((8, 8))
    arr[3, 3] = np.nan   # under the mask → repaired
    arr[6, 6] = np.nan   # outside the mask → untouched
    mask = np.zeros_like(arr, dtype=bool)
    mask[2:5, 2:5] = True
    out = interpolate_masked(arr, mask)
    assert out[3, 3] == pytest.approx(1.0)
    assert np.isnan(out[6, 6])


def test_interpolate_masked_degenerate_inputs():
    arr = np.random.default_rng(1).normal(size=(6, 6))
    # Empty mask: unchanged.
    assert np.array_equal(interpolate_masked(arr, np.zeros((6, 6), bool)), arr)
    # Fully-masked image: no boundary information — unchanged.
    out = interpolate_masked(arr, np.ones((6, 6), bool))
    assert np.allclose(out, arr)
    # Shape mismatch is a programming error.
    with pytest.raises(ValueError):
        interpolate_masked(arr, np.zeros((3, 3), bool))


def test_interpolate_masked_never_extrapolates_beyond_boundary():
    arr = np.zeros((10, 10))
    arr[:, 5:] = 4.0
    mask = np.zeros_like(arr, dtype=bool)
    mask[4:6, 3:7] = True
    out = interpolate_masked(arr, mask)
    region = out[mask]
    assert region.min() >= 0.0 - 1e-9
    assert region.max() <= 4.0 + 1e-9


def test_interpolate_masked_via_frozen_mask_replay():
    yy, xx = np.mgrid[0:10, 0:10]
    arr = 1.0 * xx
    original = arr.copy()
    mask = np.zeros_like(arr, dtype=bool)
    mask[4:6, 4:6] = True
    arr[mask] = -50.0
    state = ProcessingState(steps=[
        ProcessingStep("interpolate_masked", {
            "frozen_mask": {"data": _pack_bool(mask), "shape": [10, 10]},
        }),
    ])
    out = apply_processing_state(arr, state)
    assert np.allclose(out, original, atol=1e-8)


def test_interpolate_masked_shape_mismatch_step_is_skipped():
    arr = np.zeros((8, 8))
    mask = np.zeros((10, 10), dtype=bool)
    mask[3, 3] = True
    state = ProcessingState(steps=[
        ProcessingStep("interpolate_masked", {
            "frozen_mask": {"data": _pack_bool(mask), "shape": [10, 10]},
        }),
    ])
    with pytest.warns(UserWarning, match="does not match the image"):
        out = apply_processing_state(arr, state)
    assert out.shape == (8, 8)


def test_remove_spots_auto_repairs_spikes_only():
    from probeflow.processing.repair import remove_spots_auto

    rng = np.random.default_rng(3)
    arr = rng.normal(0.0, 1.0, size=(24, 24))
    clean = arr.copy()
    spikes = [(4, 5), (10, 17), (20, 8)]
    for y, x in spikes:
        arr[y, x] = 60.0  # far beyond 6 robust sigmas
    out = remove_spots_auto(arr, threshold_mad=6.0, window_px=3)
    for y, x in spikes:
        assert abs(out[y, x]) < 10.0, (y, x, out[y, x])
    # Non-spike pixels are untouched.
    untouched = np.ones(arr.shape, dtype=bool)
    for y, x in spikes:
        untouched[y, x] = False
    assert np.allclose(out[untouched], clean[untouched])


def test_remove_spots_auto_leaves_clean_and_degenerate_images_alone():
    from probeflow.processing.repair import remove_spots_auto

    flat = np.full((12, 12), 2.5)
    assert np.array_equal(remove_spots_auto(flat), flat)  # zero MAD
    all_nan = np.full((6, 6), np.nan)
    assert np.all(np.isnan(remove_spots_auto(all_nan)))
    rng = np.random.default_rng(4)
    noise = rng.normal(size=(20, 20))
    out = remove_spots_auto(noise, threshold_mad=8.0)
    assert np.allclose(out, noise)  # 8-sigma on clean noise: nothing flagged


def test_remove_spots_auto_via_state_and_adapter():
    arr = np.zeros((16, 16))
    arr[8, 8] = 100.0
    state = processing_state_from_gui({
        "geometric_ops": [
            {"op": "remove_spots_auto",
             "params": {"threshold_mad": 6.0, "window_px": 3}},
        ],
    })
    assert [s.op for s in state.steps] == ["remove_spots_auto"]
    out = apply_processing_state(arr, state)
    assert abs(out[8, 8]) < 1.0


# ── GUI adapter mapping ───────────────────────────────────────────────────────

def test_adapter_maps_median_size():
    state = processing_state_from_gui({"median_size": 5})
    assert [(s.op, s.params) for s in state.steps] == [
        ("median_smooth", {"size_px": 5}),
    ]


def test_adapter_maps_crop_spec():
    state = processing_state_from_gui({
        "geometric_ops": [
            {"op": "crop", "params": {"x0": 1, "y0": 2, "x1": 10, "y1": 12}},
        ],
    })
    assert [(s.op, s.params) for s in state.steps] == [
        ("crop", {"x0": 1, "y0": 2, "x1": 10, "y1": 12}),
    ]


def test_adapter_maps_repair_ops_with_frame_stamp():
    frozen = {"data": _pack_bool(np.ones((4, 4), bool)), "shape": [4, 4]}
    state = processing_state_from_gui({
        "geometric_ops": [{"op": "flip_horizontal"}],
        "repair_ops": [
            # Committed before the flip: replays before it.
            {"frozen_mask": frozen, "after_geometric_ops": 0},
        ],
    })
    ops = [s.op for s in state.steps]
    assert ops == ["interpolate_masked", "flip_horizontal"]

    state2 = processing_state_from_gui({
        "geometric_ops": [{"op": "flip_horizontal"}],
        "repair_ops": [
            {"frozen_mask": frozen, "after_geometric_ops": 1},
        ],
    })
    assert [s.op for s in state2.steps] == ["flip_horizontal", "interpolate_masked"]


def test_adapter_maps_repair_op_from_frozen_roi_geometry():
    state = processing_state_from_gui({
        "repair_ops": [{
            "frozen_geometry": {
                "kind": "rect",
                "geometry": {"x": 1, "y": 1, "width": 3, "height": 3},
                "coord_system": "pixel",
            },
        }],
    })
    assert len(state.steps) == 1
    step = state.steps[0]
    assert step.op == "interpolate_masked"
    assert step.params["frozen_geometry"]["kind"] == "rect"


def test_median_is_roi_scopable():
    state = processing_state_from_gui({
        "median_size": 3,
        "processing_scope": "roi",
        "processing_roi_id": "r1",
    })
    assert len(state.steps) == 1
    assert state.steps[0].op == "roi"
    assert state.steps[0].params["step"]["op"] == "median_smooth"
