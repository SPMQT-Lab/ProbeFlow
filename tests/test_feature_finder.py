"""Tests for probeflow.analysis.feature_finder."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.analysis.feature_finder import (
    FeaturePoint,
    feature_points_to_image,
    find_image_features,
)


def _gaussian(shape, cx, cy, amp=1.0, sigma=1.5):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))


def _image(*peaks, shape=(64, 64)):
    """Build image from a list of (cx, cy, amp) tuples."""
    img = np.zeros(shape, dtype=float)
    for cx, cy, amp in peaks:
        img += _gaussian(shape, cx, cy, amp)
    return img


# --- Test 1: single bright Gaussian gives one maximum ---

def test_single_bright_gaussian_gives_one_maximum():
    img = _image((32, 32, 5.0))
    result = find_image_features(
        img,
        mode="maxima",
        threshold_mode="above",
        threshold_low=1.0,
        min_distance_px=3.0,
    )
    assert len(result.points) == 1
    pt = result.points[0]
    assert round(pt.x_px) == 32
    assert round(pt.y_px) == 32


# --- Test 2: single dark Gaussian gives one minimum ---

def test_single_dark_gaussian_gives_one_minimum():
    img = -_image((24, 24, 3.0))  # dark depression
    result = find_image_features(
        img,
        mode="minima",
        threshold_mode="below",
        threshold_high=-0.5,
        min_distance_px=3.0,
    )
    assert len(result.points) == 1
    pt = result.points[0]
    assert round(pt.x_px) == 24
    assert round(pt.y_px) == 24


# --- Test 3: nearby peaks merge when min_distance_px is large ---

def test_nearby_peaks_merge_when_min_distance_large():
    img = _image((30, 30, 5.0), (34, 30, 4.0))
    result = find_image_features(
        img,
        mode="maxima",
        threshold_mode="above",
        threshold_low=0.5,
        min_distance_px=10.0,
    )
    assert len(result.points) == 1


# --- Test 4: nearby peaks are separate when min_distance_px is small ---

def test_nearby_peaks_separate_when_min_distance_small():
    img = _image((10, 32, 5.0), (52, 32, 4.5))
    result = find_image_features(
        img,
        mode="maxima",
        threshold_mode="above",
        threshold_low=0.5,
        min_distance_px=2.0,
    )
    assert len(result.points) == 2


# --- Test 5: above-threshold detection excludes low peaks ---

def test_above_threshold_excludes_low_peaks():
    img = _image((16, 16, 1.0), (48, 48, 5.0))
    result = find_image_features(
        img,
        mode="maxima",
        threshold_mode="above",
        threshold_low=2.0,
        min_distance_px=3.0,
    )
    assert len(result.points) == 1
    assert round(result.points[0].x_px) == 48


# --- Test 6: between-threshold selects only intermediate features ---

def test_between_threshold_selects_intermediate_features():
    shape = (64, 64)
    img = np.zeros(shape, dtype=float)
    # Low: amp 1 at (10,10), Medium: amp 3 at (30,30), High: amp 8 at (50,50)
    img += _gaussian(shape, 10, 10, amp=1.0)
    img += _gaussian(shape, 30, 30, amp=3.0)
    img += _gaussian(shape, 50, 50, amp=8.0)

    result = find_image_features(
        img,
        mode="maxima",
        threshold_mode="between",
        threshold_low=1.5,
        threshold_high=5.0,
        min_distance_px=3.0,
    )
    # Only the medium peak should be within [1.5, 5.0]
    assert len(result.points) == 1
    assert round(result.points[0].x_px) == 30
    assert round(result.points[0].y_px) == 30


# --- Test 7: ROI mask excludes features outside ROI ---

def test_roi_mask_excludes_features_outside():
    img = _image((10, 10, 5.0), (50, 50, 5.0))
    roi = np.zeros((64, 64), dtype=bool)
    roi[:30, :30] = True  # only top-left quadrant

    result = find_image_features(
        img,
        mode="maxima",
        threshold_mode="above",
        threshold_low=0.5,
        min_distance_px=3.0,
        roi_mask=roi,
    )
    assert len(result.points) == 1
    pt = result.points[0]
    assert pt.x_px < 30.0
    assert pt.y_px < 30.0


# --- Test 8: feature image generation places pixels at expected coordinates ---

def test_feature_image_places_pixels_at_expected_coords():
    points = [FeaturePoint(x_px=10.0, y_px=20.0, z_value=5.0)]
    img = feature_points_to_image(points, (32, 32), radius_px=0.0)
    assert img[20, 10] == pytest.approx(1.0)
    assert img[20, 11] == pytest.approx(0.0)
    assert img[19, 10] == pytest.approx(0.0)


# --- Test 9: dilation increases feature area ---

def test_dilation_increases_feature_area():
    points = [FeaturePoint(x_px=16.0, y_px=16.0, z_value=1.0)]
    shape = (32, 32)
    single_pixel = feature_points_to_image(points, shape, radius_px=0.0)
    dilated = feature_points_to_image(points, shape, radius_px=3.0)
    assert int(np.count_nonzero(dilated)) > int(np.count_nonzero(single_pixel))


# --- Test 10: smoothing preserves total feature image approximately ---

def test_smoothing_preserves_total_feature_image():
    points = [FeaturePoint(x_px=16.0, y_px=16.0, z_value=1.0)]
    shape = (64, 64)
    unsmoothed = feature_points_to_image(points, shape, radius_px=3.0, smoothing_sigma_px=0.0)
    smoothed = feature_points_to_image(points, shape, radius_px=3.0, smoothing_sigma_px=1.5)
    # Gaussian smoothing (with sufficient margin from edges) preserves total mass.
    assert float(smoothed.sum()) == pytest.approx(float(unsmoothed.sum()), rel=0.01)


# --- Regression for review arch-backend #2 ---

def test_feature_point_is_the_canonical_class():
    """``probeflow.analysis.feature_finder.FeaturePoint`` is now an alias
    of ``probeflow.measurements.models.FeaturePoint`` — not a separate
    dataclass.  Both names must resolve to the same object so legacy
    imports and the canonical-detector path produce interoperable
    records."""
    from probeflow.measurements.models import FeaturePoint as CanonicalFP
    assert FeaturePoint is CanonicalFP


def test_find_image_features_returns_canonical_records():
    """``find_image_features`` populates intrinsic fields (x_px, y_px,
    z_value) and leaves context fields at canonical defaults so the
    resulting points are valid canonical
    ``measurements.models.FeaturePoint`` instances."""
    import math
    img = _image((10, 20, 5.0))
    result = find_image_features(img, threshold_low=1.0, min_distance_px=3.0)
    assert len(result.points) == 1
    pt = result.points[0]
    assert pt.x_px == pytest.approx(10.0)
    assert pt.y_px == pytest.approx(20.0)
    assert pt.z_value == pytest.approx(5.0, rel=0.01)
    # Context defaults
    assert pt.point_id == ""
    assert pt.channel == ""
    assert pt.source_label == ""
    assert pt.roi_id is None
    assert math.isnan(pt.x_phys)
    assert math.isnan(pt.y_phys)


def test_find_image_features_and_detect_local_maxima_agree_on_peak_locations():
    """Both detection paths share ``_detect_peaks_nms`` (review arch-backend #3).

    A two-peak image with the same min_distance and an absolute threshold
    must yield identical pixel locations from both APIs.  This guards
    against future NMS divergence between the FeatureFinder dialog
    (``find_image_features``) and the measurement dock
    (``detect_local_maxima``).
    """
    from probeflow.measurements.features import detect_local_maxima

    img = _image((20, 10, 3.0), (44, 40, 3.0), shape=(64, 64))
    threshold = 1.0
    min_dist = 5.0

    ff_result = find_image_features(
        img,
        threshold_mode="above",
        threshold_low=threshold,
        min_distance_px=min_dist,
    )
    dlm_points = detect_local_maxima(
        img,
        threshold_mode="absolute",
        threshold_value=threshold,
        min_distance_px=int(min_dist),
    )

    ff_locs = sorted((round(pt.x_px), round(pt.y_px)) for pt in ff_result.points)
    dlm_locs = sorted((round(pt.x_px), round(pt.y_px)) for pt in dlm_points)
    assert ff_locs == dlm_locs


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
