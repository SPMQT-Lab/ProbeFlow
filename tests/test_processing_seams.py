"""Seam tests for the processing module.

These tests exercise multi-step workflows across module boundaries:
processing functions / ROI / measurement functions / export.

Seam numbers correspond to the review brief's §5 classification:
  5.2  preview → apply consistency
  5.4  ROI → processing (scope restriction)
  5.6  processing → measurement → export (provenance)
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from probeflow.core.roi import ROI
from probeflow.measurements.export import measurements_to_json_text
from probeflow.measurements.image import roi_statistics
from probeflow.processing.image import (
    STMBackgroundParams,
    align_rows,
    apply_stm_background,
    fourier_filter,
    preview_stm_background,
    stm_line_background,
    subtract_background,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _rect_roi(x, y, w, h, name="roi"):
    return ROI.new("rectangle", {"x": x, "y": y, "width": w, "height": h}, name=name)


def _ramp_image(shape=(32, 32)):
    """Monotonically increasing gradient — easy to flatten analytically."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return xx.astype(float) + yy.astype(float)


def _row_offset_image(shape=(16, 16)):
    """Uniform image where each row has a large unique offset."""
    offsets = np.arange(shape[0], dtype=float) * 10.0
    arr = np.ones(shape, dtype=float)
    arr += offsets[:, None]  # offset increases row-by-row
    return arr


# ── seam 5.2: preview → apply produce identical corrected data ────────────────

def test_preview_apply_consistency_corrected_arrays_identical():
    """preview_stm_background.corrected must equal apply_stm_background output."""
    image = _ramp_image()
    params = STMBackgroundParams(line_statistic="median", model="linear")

    preview = preview_stm_background(image, params=params)
    applied = apply_stm_background(image, params=params)

    np.testing.assert_array_equal(
        preview.corrected, applied,
        err_msg="preview.corrected and apply_stm_background must be identical arrays",
    )


def test_preview_apply_consistency_with_roi_mask():
    """preview and apply must agree when a ROI mask restricts the fit region."""
    image = _ramp_image()
    mask = np.zeros(image.shape, dtype=bool)
    mask[8:24, 8:24] = True  # fit only the central 16×16 region

    params = STMBackgroundParams(
        fit_region="active_roi", line_statistic="median", model="linear"
    )

    preview = preview_stm_background(image, params=params, mask=mask)
    applied = apply_stm_background(image, params=params, mask=mask)

    np.testing.assert_array_equal(preview.corrected, applied)


def test_preview_result_contains_background_image_and_profile():
    """STMBackgroundResult must expose background_image, line_profile, fitted_profile."""
    image = _ramp_image()
    result = preview_stm_background(image)

    assert result.background_image.shape == image.shape
    assert result.line_profile.shape[0] == image.shape[0]
    assert result.fitted_profile.shape[0] == image.shape[0]
    assert result.fit_status == "success"


# ── seam 5.4: ROI-scoped background subtraction ───────────────────────────────

def test_subtract_background_apply_roi_leaves_outside_pixels_unchanged():
    """subtract_background(apply_roi=...) must not modify pixels outside the ROI."""
    arr = _ramp_image()
    apply_roi = _rect_roi(8, 8, 16, 16, name="patch")

    original = arr.copy()
    corrected = subtract_background(arr, order=1, apply_roi=apply_roi)

    # Build the ROI mask to check which pixels were modified
    roi_mask = apply_roi.to_mask(arr.shape)

    # Outside the ROI: values must be unchanged
    outside_original = original[~roi_mask]
    outside_corrected = corrected[~roi_mask]
    np.testing.assert_array_equal(
        outside_original, outside_corrected,
        err_msg="Pixels outside apply_roi must not be altered by subtract_background",
    )


def test_subtract_background_fit_mask_confines_fit_to_clean_region():
    """subtract_background with fit_mask must fit polynomial to masked pixels only.

    A 10 nm spike in the unmasked region must not distort the background estimate
    when the fit is restricted to the masked region.
    """
    arr = np.zeros((32, 32), dtype=float)
    arr += np.linspace(0, 2, 32)[None, :]  # mild horizontal ramp

    # Add spike outside the fit mask — should not affect the fit
    arr[28:32, 28:32] += 1000.0

    fit_mask = np.ones(arr.shape, dtype=bool)
    fit_mask[26:, 26:] = False  # exclude spike region from fit

    corrected = subtract_background(arr, order=1, fit_mask=fit_mask)

    # Within the clean region the ramp should have been removed
    clean = corrected[:24, :24]
    assert float(np.ptp(clean)) < 0.5, (
        f"Residual range in clean region too large: {float(np.ptp(clean)):.3f}"
    )


# ── seam 5.6: processing → measurement → export ───────────────────────────────

def test_processing_to_measurement_to_export_data_basis_in_json():
    """data_basis='processed_image' must survive the full processing→export pipeline."""
    raw = _ramp_image()
    processed = align_rows(raw, method="median")

    roi = _rect_roi(4, 4, 20, 20, name="scan_region")
    result = roi_statistics(
        processed,
        measurement_id="M0001",
        source_label="scan:Z",
        channel="Z",
        roi=roi,
        height_unit="nm",
        x_unit="nm",
        y_unit="nm",
        data_basis="processed_image",
    )

    payload = json.loads(measurements_to_json_text([result]))
    m = payload["measurements"][0]

    assert m["context"]["data_basis"] == "processed_image"


def test_processing_raw_vs_processed_measurement_values_differ():
    """Measuring raw vs processed image must produce different mean_height values."""
    raw = _row_offset_image()
    processed = align_rows(raw, method="median")

    roi = _rect_roi(0, 0, 16, 16)

    raw_result = roi_statistics(
        raw, measurement_id="M0001", source_label="scan:Z",
        channel="Z", roi=roi, height_unit="nm", data_basis="raw_image",
    )
    proc_result = roi_statistics(
        processed, measurement_id="M0002", source_label="scan:Z",
        channel="Z", roi=roi, height_unit="nm", data_basis="processed_image",
    )

    assert raw_result.values["mean_height"] != pytest.approx(
        proc_result.values["mean_height"], abs=0.1
    )
    assert raw_result.context["data_basis"] == "raw_image"
    assert proc_result.context["data_basis"] == "processed_image"


# ── align_rows — high-risk unit tests ─────────────────────────────────────────

def test_align_rows_median_zeroes_row_medians():
    """align_rows(method='median') must leave all row medians near zero."""
    arr = _row_offset_image()
    result = align_rows(arr, method="median")

    row_medians = np.median(result, axis=1)
    np.testing.assert_allclose(
        row_medians, 0.0, atol=1e-10,
        err_msg="Row medians must be zero after align_rows(method='median')",
    )


def test_align_rows_mean_zeroes_row_means():
    """align_rows(method='mean') must leave all row means near zero."""
    arr = _row_offset_image()
    result = align_rows(arr, method="mean")

    row_means = result.mean(axis=1)
    np.testing.assert_allclose(row_means, 0.0, atol=1e-10)


def test_align_rows_nan_rows_are_preserved():
    """align_rows must not crash on rows that are entirely NaN."""
    arr = _row_offset_image()
    arr[7, :] = np.nan  # one all-NaN row

    result = align_rows(arr, method="median")

    # NaN row must stay NaN
    assert np.all(np.isnan(result[7, :]))
    # Other rows must still have zero median
    finite_rows = [r for r in range(arr.shape[0]) if r != 7]
    for r in finite_rows:
        assert abs(float(np.median(result[r]))) < 1e-10


# ── stm_line_background — high-risk unit tests ────────────────────────────────

def test_stm_line_background_does_not_crash_on_nan_rows():
    """stm_line_background must handle all-NaN rows without raising."""
    arr = np.random.default_rng(0).standard_normal((16, 16))
    arr[3, :] = np.nan  # all-NaN row
    arr[10, :] = np.nan

    result = stm_line_background(arr)

    assert result.shape == arr.shape
    assert np.all(np.isnan(result[3, :]))
    assert np.all(np.isnan(result[10, :]))


def test_stm_line_background_reduces_row_to_row_offsets():
    """stm_line_background must reduce systematic row-to-row step offsets."""
    rng = np.random.default_rng(42)
    flat = rng.standard_normal((32, 32)) * 0.1
    # Add a known step: rows 16+ are offset by +5
    flat[16:, :] += 5.0

    result = stm_line_background(flat)

    row_medians = np.median(result, axis=1)
    # After correction, the jump in row medians should be much smaller
    jump_before = abs(float(np.median(flat[16:])) - float(np.median(flat[:16])))
    jump_after = abs(float(np.median(result[16:])) - float(np.median(result[:16])))
    assert jump_after < jump_before * 0.1, (
        f"Step not adequately corrected: before={jump_before:.3f}, after={jump_after:.3f}"
    )


def test_stm_line_background_output_shape_unchanged():
    """stm_line_background must return the same shape as the input."""
    arr = np.ones((24, 32), dtype=float)
    result = stm_line_background(arr)
    assert result.shape == arr.shape


# ── fourier_filter — high-risk unit tests ────────────────────────────────────

def test_fourier_filter_output_shape_matches_input():
    """fourier_filter must return an array with the same shape as the input."""
    arr = np.random.default_rng(0).standard_normal((32, 40))
    for mode in ("low_pass", "high_pass"):
        out = fourier_filter(arr, mode=mode, cutoff=0.3)
        assert out.shape == arr.shape, f"Shape mismatch for mode={mode}"


def test_fourier_filter_low_pass_suppresses_high_frequencies():
    """Low-pass filter must suppress a high-frequency sinusoid more than a low one."""
    Ny, Nx = 64, 64
    yy, xx = np.mgrid[:Ny, :Nx]
    # Low-frequency signal (period ≈ 32 px) + high-frequency signal (period ≈ 4 px)
    low_freq = np.sin(2 * np.pi * xx / 32.0)
    high_freq = np.sin(2 * np.pi * xx / 4.0)
    image = low_freq + high_freq

    filtered = fourier_filter(image, mode="low_pass", cutoff=0.2, window="none")

    # After low-pass, correlation with high_freq should be much weaker than with low_freq
    corr_low = float(np.corrcoef(filtered.ravel(), low_freq.ravel())[0, 1])
    corr_high = float(np.corrcoef(filtered.ravel(), high_freq.ravel())[0, 1])
    assert abs(corr_low) > abs(corr_high) * 5, (
        f"Low-pass filter did not suppress high-frequency content: "
        f"corr_low={corr_low:.3f}, corr_high={corr_high:.3f}"
    )


def test_fourier_filter_preserves_nan_mask():
    """fourier_filter must preserve NaN pixels at the same locations as the input."""
    arr = np.random.default_rng(0).standard_normal((32, 32))
    arr[5, :] = np.nan
    arr[:, 28] = np.nan

    out = fourier_filter(arr, mode="low_pass", cutoff=0.5)

    np.testing.assert_array_equal(
        np.isnan(arr), np.isnan(out),
        err_msg="fourier_filter must not move or add NaN pixels",
    )
