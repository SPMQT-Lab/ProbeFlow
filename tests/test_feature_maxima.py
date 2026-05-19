"""Tests for local maxima, point-list export, masks, and point FFTs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from probeflow.measurements.export import (
    feature_points_to_csv_text,
    feature_points_to_json_text,
)
from probeflow.measurements.features import detect_local_maxima, feature_maxima_result
from probeflow.measurements.fft_points import (
    fft_from_point_mask,
    point_fft_summary_result,
    point_fft_to_csv_text,
    point_mask_to_csv_text,
    points_to_mask,
)


def _gaussian_image(peaks, shape=(64, 64), sigma=1.2):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    image = np.zeros(shape, dtype=float)
    for x0, y0, amp in peaks:
        image += amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return image


def test_detect_local_maxima_finds_separated_gaussian_peaks():
    image = _gaussian_image([(16, 20, 5.0), (45, 42, 3.0)])

    points = detect_local_maxima(
        image,
        threshold_mode="absolute",
        threshold_value=1.0,
        min_distance_px=5,
        pixel_size_x=0.2,
        pixel_size_y=0.2,
        channel="Z",
        source_label="synthetic",
    )

    assert len(points) == 2
    coords = sorted((round(p.x_px), round(p.y_px)) for p in points)
    assert coords == [(16, 20), (45, 42)]
    assert points[0].x_phys == pytest.approx(points[0].x_px * 0.2)


def test_detect_local_maxima_respects_min_distance_and_roi_mask():
    image = np.zeros((24, 24), dtype=float)
    image[10, 10] = 5.0
    image[10, 12] = 4.0
    image[18, 18] = 6.0
    roi_mask = np.zeros_like(image, dtype=bool)
    roi_mask[:15, :15] = True

    points = detect_local_maxima(
        image,
        threshold_mode="absolute",
        threshold_value=1.0,
        min_distance_px=4,
        roi_mask=roi_mask,
        roi_id="ROI-1",
    )

    assert len(points) == 1
    assert (points[0].x_px, points[0].y_px) == (10.0, 10.0)
    assert points[0].roi_id == "ROI-1"


def test_detect_local_maxima_rejects_flat_background_plateaus():
    image = np.zeros((20, 20), dtype=float)
    image[10, 10] = 10.0

    points = detect_local_maxima(
        image,
        threshold_mode="percentile",
        threshold_value=95.0,
        min_distance_px=2,
    )

    assert len(points) == 1
    assert (points[0].x_px, points[0].y_px) == (10.0, 10.0)


def test_feature_point_exports_and_summary():
    image = _gaussian_image([(8, 8, 2.0)])
    points = detect_local_maxima(
        image,
        threshold_mode="percentile",
        threshold_value=99,
        min_distance_px=3,
        channel="Z",
        source_label="scan",
    )

    csv_text = feature_points_to_csv_text(
        points,
        metadata={"x_unit": "nm", "y_unit": "nm", "z_unit": "pm"},
    )
    payload = json.loads(feature_points_to_json_text(points, metadata={"unit": "nm"}))
    result = feature_maxima_result(
        points,
        measurement_id="M0001",
        source_label="scan:Z",
        channel="Z",
        threshold_mode="percentile",
        threshold_value=99,
        min_distance_px=3,
    )

    assert "point_id,x_px,y_px" in csv_text
    assert "x_unit,y_unit,z_unit" in csv_text
    assert ",nm,nm,pm," in csv_text
    assert payload["points"][0]["source_label"] == "scan"
    assert result.values["n_points"] == len(points)
    assert result.context["threshold_mode"] == "percentile"


def test_points_to_mask_dilates_without_crashing_at_edges():
    points = [(0.0, 0.0), (5.0, 5.0)]

    centers = points_to_mask(points, (8, 8), radius_px=0)
    disk = points_to_mask(points, (8, 8), radius_px=2, shape_mode="disk")
    square = points_to_mask(points, (8, 8), radius_px=1, shape_mode="square")

    assert centers[0, 0]
    assert centers[5, 5]
    assert int(disk.sum()) > int(centers.sum())
    assert int(square.sum()) > int(centers.sum())


def test_fft_from_square_lattice_point_mask_has_expected_frequency():
    points = [(float(x), float(y)) for y in range(0, 64, 8) for x in range(0, 64, 8)]
    mask = points_to_mask(points, (64, 64))

    result = fft_from_point_mask(
        mask,
        pixel_size_x=1.0,
        pixel_size_y=1.0,
        spatial_unit="nm",
        n_points=len(points),
    )
    mag = result.fft_magnitude.copy()
    mag[32, 32] = 0.0
    cy = int(np.argmin(np.abs(result.qy)))
    cx = int(np.argmin(np.abs(result.qx)))
    qx_idx = int(np.argmin(np.abs(result.qx - 1 / 8)))
    qy_idx = int(np.argmin(np.abs(result.qy - 1 / 8)))

    assert result.units == "cycles/nm"
    assert result.n_points == len(points)
    assert result.qx[qx_idx] == pytest.approx(1 / 8)
    assert result.qy[qy_idx] == pytest.approx(1 / 8)
    assert mag[cy, qx_idx] > 0
    assert mag[qy_idx, cx] > 0


def test_point_mask_and_fft_exports_are_self_describing():
    points = [(0.0, 0.0), (4.0, 4.0)]
    mask = points_to_mask(points, (8, 8), radius_px=1, shape_mode="square")
    result = fft_from_point_mask(
        mask,
        pixel_size_x=0.5,
        pixel_size_y=0.5,
        spatial_unit="nm",
        n_points=len(points),
        radius_px=1,
    )

    metadata = {
        "export_type": "probeflow_feature_point_mask",
        "source_path": "/tmp/scan.sxm",
        "radius_px": 1,
        "shape_mode": "square",
        "point_count": len(points),
        "pixel_size_x_nm": 0.5,
        "pixel_size_y_nm": 0.5,
    }
    mask_csv = point_mask_to_csv_text(mask, metadata=metadata)
    fft_csv = point_fft_to_csv_text(
        result,
        metadata={**metadata, "export_type": "probeflow_point_mask_fft"},
    )
    summary = point_fft_summary_result(
        result,
        measurement_id="M0002",
        source_label="scan:Z",
        channel="Z",
        mask_pixels=int(mask.sum()),
        shape_mode="square",
    )

    assert "# export_type,probeflow_feature_point_mask" in mask_csv
    assert "# source_path,/tmp/scan.sxm" in mask_csv
    assert "# radius_px,1" in mask_csv
    assert "qx,qy,magnitude,unit" in fft_csv
    assert "# export_type,probeflow_point_mask_fft" in fft_csv
    assert "# pixel_size_x_nm,0.5" in fft_csv
    assert "cycles/nm" in fft_csv
    assert summary.kind == "point_fft"
    assert summary.values["n_points"] == 2
    assert summary.values["n_mask_pixels"] == int(mask.sum())
    assert summary.context["shape_mode"] == "square"
