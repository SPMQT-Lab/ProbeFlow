"""Seam tests for the processing module.

These tests exercise multi-step workflows across module boundaries:
processing functions / ROI / measurement functions / export.

Seam numbers correspond to the review brief's §5 classification:
  5.2  preview → apply consistency
  5.4  ROI → processing (scope restriction)
  5.6  processing → measurement → export (provenance)

Unit-level tests for individual processing functions (align_rows, fourier_filter,
stm_line_background, etc.) live in test_processing.py, not here.
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
    preview_stm_background,
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
    arr += offsets[:, None]
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
    mask[8:24, 8:24] = True

    params = STMBackgroundParams(
        fit_region="active_roi", line_statistic="median", model="linear"
    )

    preview = preview_stm_background(image, params=params, mask=mask)
    applied = apply_stm_background(image, params=params, mask=mask)

    np.testing.assert_array_equal(preview.corrected, applied)


def test_preview_result_exposes_background_image_and_profiles():
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

    roi_mask = apply_roi.to_mask(arr.shape)
    np.testing.assert_array_equal(
        original[~roi_mask], corrected[~roi_mask],
        err_msg="Pixels outside apply_roi must not be altered by subtract_background",
    )


def test_subtract_background_fit_mask_confines_fit_to_clean_region():
    """A spike outside the fit mask must not distort the background estimate."""
    arr = np.zeros((32, 32), dtype=float)
    arr += np.linspace(0, 2, 32)[None, :]  # mild horizontal ramp

    arr[28:32, 28:32] += 1000.0  # spike that must be excluded from fit

    fit_mask = np.ones(arr.shape, dtype=bool)
    fit_mask[26:, 26:] = False

    corrected = subtract_background(arr, order=1, fit_mask=fit_mask)

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
    """Measuring raw vs align_rows-processed image must produce different mean_height."""
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
