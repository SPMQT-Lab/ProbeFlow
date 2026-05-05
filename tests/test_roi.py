"""Tests for probeflow.core.roi — ROI data model, masks, and transforms."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from probeflow.core.roi import (
    ROI,
    ROISet,
    combine_masks,
    invert_mask,
    roi_from_legacy_geometry_dict,
    translate,
)
from probeflow.processing.state import apply_processing_state, ProcessingState, ProcessingStep


# ── Fixtures ──────────────────────────────────────────────────────────────────

SHAPE = (100, 100)   # (Ny, Nx)


def rect_roi(x=10, y=10, w=20, h=20):
    return ROI.new("rectangle", {"x": float(x), "y": float(y),
                                  "width": float(w), "height": float(h)})


def ellipse_roi(cx=50, cy=50, rx=10, ry=10):
    return ROI.new("ellipse", {"cx": float(cx), "cy": float(cy),
                                "rx": float(rx), "ry": float(ry)})


def polygon_roi():
    return ROI.new("polygon", {"vertices": [[10.0, 10.0], [30.0, 10.0], [20.0, 30.0]]})


def freehand_roi():
    return ROI.new("freehand", {"vertices": [[1.0, 2.0], [3.0, 5.0], [8.0, 13.0]]})


def line_roi():
    return ROI.new("line", {"x1": 5.0, "y1": 5.0, "x2": 15.0, "y2": 15.0})


def point_roi(x=50, y=50):
    return ROI.new("point", {"x": float(x), "y": float(y)})


# ── ROI serialisation ─────────────────────────────────────────────────────────

class TestROISerialisation:
    def test_rectangle_round_trip(self):
        roi = rect_roi()
        d = roi.to_dict()
        restored = ROI.from_dict(d)
        assert restored.id == roi.id
        assert restored.name == roi.name
        assert restored.kind == roi.kind
        assert abs(restored.geometry["x"] - 10.0) < 1e-9
        assert abs(restored.geometry["width"] - 20.0) < 1e-9

    def test_ellipse_round_trip(self):
        roi = ellipse_roi()
        restored = ROI.from_dict(roi.to_dict())
        assert restored.kind == "ellipse"
        assert abs(restored.geometry["cx"] - 50.0) < 1e-9

    def test_polygon_round_trip(self):
        roi = polygon_roi()
        restored = ROI.from_dict(roi.to_dict())
        assert restored.kind == "polygon"
        verts = restored.geometry["vertices"]
        assert len(verts) == 3
        assert abs(verts[0][0] - 10.0) < 1e-9

    def test_line_round_trip(self):
        roi = line_roi()
        restored = ROI.from_dict(roi.to_dict())
        assert restored.kind == "line"
        assert abs(restored.geometry["x2"] - 15.0) < 1e-9

    def test_point_round_trip(self):
        roi = point_roi()
        restored = ROI.from_dict(roi.to_dict())
        assert restored.kind == "point"
        assert abs(restored.geometry["x"] - 50.0) < 1e-9

    def test_linked_file_preserved(self):
        roi = ROI.new("point", {"x": 5.0, "y": 5.0}, linked_file="spec_001.dat")
        restored = ROI.from_dict(roi.to_dict())
        assert restored.linked_file == "spec_001.dat"

    def test_linked_file_none_preserved(self):
        roi = rect_roi()
        restored = ROI.from_dict(roi.to_dict())
        assert restored.linked_file is None

    def test_multipolygon_round_trip(self):
        roi = ROI.new("multipolygon", {
            "components": [{
                "exterior": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                "holes": [[[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]]],
            }],
        }, name="donut")
        restored = ROI.from_dict(roi.to_dict())
        assert restored.kind == "multipolygon"
        assert restored.geometry["components"][0]["exterior"][2] == [10.0, 10.0]
        assert restored.geometry["components"][0]["holes"][0][0] == [2.0, 2.0]


# ── to_mask ───────────────────────────────────────────────────────────────────

class TestToMask:
    def test_rectangle_pixel_count(self):
        roi = rect_roi(x=10, y=10, w=10, h=10)
        mask = roi.to_mask(SHAPE)
        assert mask.shape == SHAPE
        assert int(mask.sum()) == 100

    def test_rectangle_clipped_to_boundary(self):
        roi = rect_roi(x=90, y=90, w=20, h=20)  # extends outside 100×100
        mask = roi.to_mask(SHAPE)
        assert mask.sum() > 0
        assert mask.sum() < 400  # clipped

    def test_ellipse_approx_pixel_count(self):
        roi = ellipse_roi(cx=50.5, cy=50.5, rx=5, ry=5)
        mask = roi.to_mask(SHAPE)
        # Area = pi*r^2 ≈ 78.5; accept 65-92
        assert 65 <= int(mask.sum()) <= 92

    def test_ellipse_full_circle(self):
        roi = ellipse_roi(cx=50.5, cy=50.5, rx=5, ry=5)
        mask = roi.to_mask(SHAPE)
        assert mask.shape == SHAPE

    def test_polygon_shape(self):
        roi = polygon_roi()
        mask = roi.to_mask(SHAPE)
        assert mask.shape == SHAPE
        assert mask.sum() > 0

    def test_line_connects_endpoints(self):
        roi = line_roi()
        mask = roi.to_mask(SHAPE)
        assert mask[5, 5]   # (y=5, x=5)
        assert mask[15, 15]  # (y=15, x=15)

    def test_point_single_pixel(self):
        roi = point_roi(x=30, y=40)
        mask = roi.to_mask(SHAPE)
        assert int(mask.sum()) == 1
        assert mask[40, 30]

    def test_out_of_bounds_point(self):
        roi = point_roi(x=200, y=200)
        mask = roi.to_mask(SHAPE)
        assert int(mask.sum()) == 0


# ── bounds / crop ─────────────────────────────────────────────────────────────

class TestBoundsAndCrop:
    def test_bounds_rectangle(self):
        roi = rect_roi(x=10, y=20, w=5, h=8)
        r0, r1, c0, c1 = roi.bounds(SHAPE)
        assert r0 == 20
        assert r1 == 27   # y=20 to y+h-1=27
        assert c0 == 10
        assert c1 == 14   # x=10 to x+w-1=14

    def test_crop_returns_correct_subarray(self):
        arr = np.zeros(SHAPE)
        arr[20:28, 10:15] = 1.0
        roi = rect_roi(x=10, y=20, w=5, h=8)
        crop = roi.crop(arr)
        assert crop.shape == (8, 5)
        assert np.all(crop == 1.0)


# ── Lossless transforms ───────────────────────────────────────────────────────

class TestTransformLossless:
    """Verify exact coordinate transforms for all lossless operations."""

    def _rect_coverage(self, roi, shape):
        """Return the set of pixel columns covered by a rectangle ROI."""
        g = roi.geometry
        x = int(round(g["x"]))
        w = int(round(g["width"]))
        return set(range(x, x + w))

    def test_flip_horizontal_rectangle(self):
        # 100×100 image, rect at x=10, w=20 → should land at x=70, w=20
        roi = rect_roi(x=10, y=10, w=20, h=20)
        t = roi.transform("flip_horizontal", {}, (100, 100))
        assert t is not None
        assert abs(t.geometry["x"] - 70.0) < 1e-9
        assert abs(t.geometry["y"] - 10.0) < 1e-9
        assert abs(t.geometry["width"] - 20.0) < 1e-9

    def test_flip_vertical_rectangle(self):
        roi = rect_roi(x=10, y=10, w=20, h=20)
        t = roi.transform("flip_vertical", {}, (100, 100))
        assert t is not None
        assert abs(t.geometry["y"] - 70.0) < 1e-9
        assert abs(t.geometry["height"] - 20.0) < 1e-9

    def test_rot90_cw_rectangle_swaps_dims(self):
        # 80×100 image (Ny=80, Nx=100), rect at (x=10, y=5, w=30, h=20)
        # After rot90_cw (image becomes 100×80):
        # new_x = Ny-y-h = 80-5-20 = 55, new_y = x = 10, new_w=h=20, new_h=w=30
        roi = rect_roi(x=10, y=5, w=30, h=20)
        t = roi.transform("rot90_cw", {}, (80, 100))
        assert t is not None
        assert abs(t.geometry["x"] - 55.0) < 1e-9
        assert abs(t.geometry["y"] - 10.0) < 1e-9
        assert abs(t.geometry["width"] - 20.0) < 1e-9
        assert abs(t.geometry["height"] - 30.0) < 1e-9

    def test_rot180_rectangle(self):
        # 100×100, rect at (x=10, y=10, w=20, h=20)
        # After rot180: new_x=100-10-20=70, new_y=100-10-20=70, same dims
        roi = rect_roi(x=10, y=10, w=20, h=20)
        t = roi.transform("rot180", {}, (100, 100))
        assert t is not None
        assert abs(t.geometry["x"] - 70.0) < 1e-9
        assert abs(t.geometry["y"] - 70.0) < 1e-9

    def test_rot270_cw_rectangle(self):
        # 80×100 image, rect at (x=10, y=5, w=30, h=20)
        # After rot270_cw: new_x=y=5, new_y=Nx-x-w=100-10-30=60, new_w=h=20, new_h=w=30
        roi = rect_roi(x=10, y=5, w=30, h=20)
        t = roi.transform("rot270_cw", {}, (80, 100))
        assert t is not None
        assert abs(t.geometry["x"] - 5.0) < 1e-9
        assert abs(t.geometry["y"] - 60.0) < 1e-9
        assert abs(t.geometry["width"] - 20.0) < 1e-9
        assert abs(t.geometry["height"] - 30.0) < 1e-9

    def test_ellipse_rot90_swaps_radii(self):
        roi = ellipse_roi(cx=30, cy=20, rx=10, ry=5)
        t = roi.transform("rot90_cw", {}, (100, 100))
        assert t is not None
        # Radii swap for 90-degree rotations
        assert abs(t.geometry["rx"] - 5.0) < 1e-9
        assert abs(t.geometry["ry"] - 10.0) < 1e-9

    def test_flip_preserves_roi_id(self):
        roi = rect_roi()
        t = roi.transform("flip_horizontal", {}, (100, 100))
        assert t.id == roi.id

    def test_polygon_vertices_transformed(self):
        roi = polygon_roi()
        t = roi.transform("rot90_cw", {}, (100, 100))
        assert t is not None
        assert len(t.geometry["vertices"]) == 3

    def test_point_transformed(self):
        roi = point_roi(x=10, y=0)
        t = roi.transform("flip_horizontal", {}, (100, 100))
        assert t is not None
        # x=10 → new_x = 99-10 = 89
        assert abs(t.geometry["x"] - 89.0) < 1e-9

    def test_rotate_arbitrary_returns_none(self):
        roi = rect_roi()
        result = roi.transform("rotate_arbitrary", {}, (100, 100))
        assert result is None

    def test_unknown_op_raises(self):
        roi = rect_roi()
        with pytest.raises(ValueError):
            roi.transform("shear", {}, (100, 100))


class TestTransformCrop:
    def test_crop_shifts_coordinates(self):
        roi = rect_roi(x=20, y=30, w=10, h=10)
        t = roi.transform("crop", {"x0": 15, "y0": 25, "x1": 40, "y1": 50}, (100, 100))
        assert t is not None
        assert abs(t.geometry["x"] - 5.0) < 1e-9   # 20-15=5
        assert abs(t.geometry["y"] - 5.0) < 1e-9   # 30-25=5

    def test_crop_drops_roi_outside(self):
        roi = rect_roi(x=60, y=60, w=10, h=10)
        t = roi.transform("crop", {"x0": 0, "y0": 0, "x1": 30, "y1": 30}, (100, 100))
        assert t is None

    def test_crop_clips_partial_overlap(self):
        roi = rect_roi(x=20, y=20, w=20, h=20)
        t = roi.transform("crop", {"x0": 25, "y0": 25, "x1": 50, "y1": 50}, (100, 100))
        assert t is not None
        assert t.geometry["width"] < 20.0   # clipped

    def test_point_outside_crop_dropped(self):
        roi = point_roi(x=80, y=80)
        t = roi.transform("crop", {"x0": 0, "y0": 0, "x1": 50, "y1": 50}, (100, 100))
        assert t is None


# ── ROISet ────────────────────────────────────────────────────────────────────

class TestROISet:
    def test_add_and_get(self):
        rs = ROISet(image_id="img1")
        roi = rect_roi()
        rs.add(roi)
        assert rs.get(roi.id) is roi

    def test_remove(self):
        rs = ROISet(image_id="img1")
        roi = rect_roi()
        rs.add(roi)
        rs.remove(roi.id)
        assert rs.get(roi.id) is None

    def test_remove_nonexistent_is_noop(self):
        rs = ROISet(image_id="img1")
        rs.remove("nonexistent")  # must not raise

    def test_get_by_name(self):
        rs = ROISet(image_id="img1")
        roi = ROI.new("point", {"x": 1.0, "y": 1.0}, name="my_point")
        rs.add(roi)
        assert rs.get_by_name("my_point") is roi

    def test_set_active(self):
        rs = ROISet(image_id="img1")
        roi = rect_roi()
        rs.add(roi)
        rs.set_active(roi.id)
        assert rs.active_roi_id == roi.id

    def test_set_active_none(self):
        rs = ROISet(image_id="img1")
        rs.set_active(None)
        assert rs.active_roi_id is None

    def test_set_active_unknown_raises(self):
        rs = ROISet(image_id="img1")
        with pytest.raises(ValueError):
            rs.set_active("nonexistent")

    def test_round_trip_serialisation(self):
        rs = ROISet(image_id="img1")
        rs.add(rect_roi())
        rs.add(ellipse_roi())
        rs.add(point_roi())
        rs.set_active(rs.rois[0].id)
        d = rs.to_dict()
        restored = ROISet.from_dict(d)
        assert len(restored.rois) == 3
        assert restored.image_id == "img1"
        assert restored.active_roi_id == rs.rois[0].id
        assert restored.rois[0].kind == "rectangle"
        assert restored.rois[1].kind == "ellipse"

    def test_from_dict_bad_roi_raises(self):
        d = {
            "image_id": "img1",
            "rois": [{"bad": "data"}],
            "active_roi_id": None,
        }
        with pytest.raises(ValueError, match="Failed to reconstruct ROI"):
            ROISet.from_dict(d)


class TestROISetTransformAll:
    def test_lossless_transforms_all_rois(self):
        rs = ROISet(image_id="img1")
        for _ in range(3):
            rs.add(rect_roi())
        invalidated = rs.transform_all("flip_horizontal", {}, (100, 100))
        assert invalidated == []
        assert len(rs.rois) == 3

    def test_rotate_arbitrary_invalidates_all(self):
        rs = ROISet(image_id="img1")
        rs.add(rect_roi())
        rs.add(point_roi())
        invalidated = rs.transform_all("rotate_arbitrary", {}, (100, 100))
        assert len(invalidated) == 2
        # ROIs still present (caller decides to remove)
        assert len(rs.rois) == 2

    def test_transforms_are_applied_to_rois(self):
        rs = ROISet(image_id="img1")
        roi = rect_roi(x=10, y=10, w=20, h=20)
        rs.add(roi)
        rs.transform_all("flip_horizontal", {}, (100, 100))
        assert abs(rs.rois[0].geometry["x"] - 70.0) < 1e-9


class TestROITranslate:
    def test_line_translation_updates_both_endpoints(self):
        roi = line_roi()
        moved = translate(roi, 2.5, -1.5)
        assert moved.geometry == {
            "x1": 7.5, "y1": 3.5,
            "x2": 17.5, "y2": 13.5,
        }

    def test_rectangle_translation_shifts_origin(self):
        moved = translate(rect_roi(x=10, y=20, w=30, h=40), -3.0, 4.0)
        assert moved.geometry["x"] == pytest.approx(7.0)
        assert moved.geometry["y"] == pytest.approx(24.0)
        assert moved.geometry["width"] == pytest.approx(30.0)
        assert moved.geometry["height"] == pytest.approx(40.0)

    def test_ellipse_translation_shifts_center(self):
        moved = translate(ellipse_roi(cx=12, cy=24, rx=5, ry=6), 10.0, -2.0)
        assert moved.geometry["cx"] == pytest.approx(22.0)
        assert moved.geometry["cy"] == pytest.approx(22.0)
        assert moved.geometry["rx"] == pytest.approx(5.0)
        assert moved.geometry["ry"] == pytest.approx(6.0)

    def test_polygon_translation_shifts_all_vertices(self):
        moved = translate(polygon_roi(), 1.0, 2.0)
        assert moved.geometry["vertices"] == [
            [11.0, 12.0],
            [31.0, 12.0],
            [21.0, 32.0],
        ]

    def test_freehand_translation_shifts_all_vertices(self):
        moved = translate(freehand_roi(), -1.0, 3.0)
        assert moved.geometry["vertices"] == [
            [0.0, 5.0],
            [2.0, 8.0],
            [7.0, 16.0],
        ]

    def test_point_translation_shifts_point(self):
        moved = translate(point_roi(x=4, y=9), 0.5, -2.5)
        assert moved.geometry["x"] == pytest.approx(4.5)
        assert moved.geometry["y"] == pytest.approx(6.5)

    def test_translation_preserves_roi_identity_fields(self):
        roi = ROI.new(
            "point",
            {"x": 4.0, "y": 9.0},
            name="probe_site",
            linked_file="scan_001.sxm",
        )
        roi.coord_system = "physical"
        moved = translate(roi, 1.0, 1.0)
        assert moved.id == roi.id
        assert moved.name == "probe_site"
        assert moved.kind == "point"
        assert moved.coord_system == "physical"
        assert moved.linked_file == "scan_001.sxm"

    def test_roi_set_active_roi_survives_translate_update(self):
        rs = ROISet(image_id="img1")
        roi = rect_roi()
        rs.add(roi)
        rs.set_active(roi.id)
        moved = translate(roi, 5.0, 6.0)
        rs.remove(roi.id)
        rs.add(moved)
        rs.set_active(moved.id)
        assert rs.active_roi_id == roi.id
        assert rs.get(roi.id).geometry["x"] == pytest.approx(15.0)

    def test_removing_active_roi_clears_active_id(self):
        rs = ROISet(image_id="img1")
        roi = rect_roi()
        other = point_roi()
        rs.add(roi)
        rs.add(other)
        rs.set_active(roi.id)
        rs.remove(roi.id)
        # Current ROISet policy is conservative: deletion clears active selection
        # instead of auto-selecting a neighbor.
        assert rs.active_roi_id is None
        assert rs.get(other.id) is other


# ── Mask helpers ──────────────────────────────────────────────────────────────

class TestCombineMasks:
    def _masks(self):
        a = np.zeros((5, 5), dtype=bool)
        a[0:3, 0:3] = True   # top-left 3×3
        b = np.zeros((5, 5), dtype=bool)
        b[2:5, 2:5] = True   # bottom-right 3×3
        return a, b

    def test_union(self):
        a, b = self._masks()
        r = combine_masks([a, b], "union")
        assert r.sum() == 9 + 9 - 1  # overlap is one pixel (2,2)

    def test_intersection(self):
        a, b = self._masks()
        r = combine_masks([a, b], "intersection")
        assert r.sum() == 1
        assert r[2, 2]

    def test_difference(self):
        a, b = self._masks()
        r = combine_masks([a, b], "difference")
        assert r.sum() == 8  # a minus the overlap pixel

    def test_xor(self):
        a, b = self._masks()
        r = combine_masks([a, b], "xor")
        assert r.sum() == 16  # 8 + 8

    def test_single_mask(self):
        a, _ = self._masks()
        r = combine_masks([a], "union")
        np.testing.assert_array_equal(r, a)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            combine_masks([], "union")

    def test_unknown_mode_raises(self):
        a, b = self._masks()
        with pytest.raises(ValueError):
            combine_masks([a, b], "subtract")  # type: ignore[arg-type]


class TestInvertMask:
    def test_invert(self):
        m = np.array([[True, False], [False, True]])
        inv = invert_mask(m)
        assert inv[0, 0] is np.bool_(False)
        assert inv[0, 1] is np.bool_(True)


# ── Legacy compatibility ──────────────────────────────────────────────────────

class TestLegacyCompat:
    def test_rect_px_converted(self):
        geometry = {"kind": "rectangle", "rect_px": (5, 10, 24, 29)}
        roi = roi_from_legacy_geometry_dict((100, 100), geometry)
        assert roi is not None
        assert roi.kind == "rectangle"
        assert abs(roi.geometry["x"] - 5.0) < 1e-9
        assert abs(roi.geometry["width"] - 20.0) < 1e-9   # x1-x0+1 = 19+1=20

    def test_bounds_frac_converted(self):
        geometry = {"kind": "rectangle", "bounds_frac": (0.1, 0.1, 0.3, 0.3)}
        roi = roi_from_legacy_geometry_dict((100, 100), geometry)
        assert roi is not None
        assert roi.kind == "rectangle"

    def test_ellipse_from_rect_px(self):
        geometry = {"kind": "ellipse", "rect_px": (40, 40, 60, 60)}
        roi = roi_from_legacy_geometry_dict((100, 100), geometry)
        assert roi is not None
        assert roi.kind == "ellipse"
        assert abs(roi.geometry["cx"] - 50.0) < 1e-9
        assert abs(roi.geometry["rx"] - 10.5) < 1e-9   # (60-40+1)/2 = 10.5 → max(0.5, 10.5)

    def test_polygon_points_px(self):
        geometry = {
            "kind": "polygon",
            "points_px": [(10.0, 10.0), (30.0, 10.0), (20.0, 30.0)],
        }
        roi = roi_from_legacy_geometry_dict((100, 100), geometry)
        assert roi is not None
        assert roi.kind == "polygon"
        assert len(roi.geometry["vertices"]) == 3

    def test_unknown_kind_returns_none(self):
        roi = roi_from_legacy_geometry_dict((100, 100), {"kind": "star"})
        assert roi is None

    def test_none_geometry_returns_none(self):
        roi = roi_from_legacy_geometry_dict((100, 100), None)  # type: ignore[arg-type]
        assert roi is None

    def test_state_py_roi_geometry_mask_unchanged(self):
        """apply_processing_state ROI step continues to work after migration."""
        from probeflow.processing.state import roi_geometry_mask
        shape = (20, 20)
        geometry = {"kind": "rectangle", "rect_px": (5, 5, 14, 14)}
        mask = roi_geometry_mask(shape, geometry)
        assert mask is not None
        assert mask.shape == shape
        assert int(mask.sum()) == 100   # 10×10

    def test_state_py_ellipse_mask_unchanged(self):
        from probeflow.processing.state import roi_geometry_mask
        shape = (20, 20)
        geometry = {"kind": "ellipse", "rect_px": (1, 1, 18, 18)}
        mask = roi_geometry_mask(shape, geometry)
        assert mask is not None
        assert mask.shape == shape

    def test_state_py_polygon_mask_unchanged(self):
        from probeflow.processing.state import roi_geometry_mask
        shape = (50, 50)
        geometry = {
            "kind": "polygon",
            "points_px": [(10, 10), (30, 10), (20, 30)],
        }
        mask = roi_geometry_mask(shape, geometry)
        assert mask is not None
        assert mask.sum() > 0


# ── apply_geometric_op_to_scan ────────────────────────────────────────────────

class TestApplyGeometricOpToScan:
    def _make_scan(self):
        from unittest.mock import MagicMock
        scan = MagicMock()
        scan.planes = [np.arange(12.0).reshape(3, 4), np.ones((3, 4))]
        return scan

    def test_flip_horizontal_updates_planes(self):
        from probeflow.processing.state import apply_geometric_op_to_scan
        scan = self._make_scan()
        original = scan.planes[0].copy()
        apply_geometric_op_to_scan(scan, "flip_horizontal")
        np.testing.assert_array_equal(scan.planes[0], np.fliplr(original))

    def test_all_planes_processed(self):
        from probeflow.processing.state import apply_geometric_op_to_scan
        scan = self._make_scan()
        apply_geometric_op_to_scan(scan, "rot90_cw")
        for plane in scan.planes:
            assert plane.shape == (4, 3)  # 3×4 → 4×3 after 90°CW

    def test_roi_set_transformed(self):
        from probeflow.processing.state import apply_geometric_op_to_scan
        scan = self._make_scan()
        rs = ROISet(image_id="img1")
        roi = rect_roi(x=0, y=0, w=2, h=2)
        rs.add(roi)
        _, rs2 = apply_geometric_op_to_scan(scan, "flip_horizontal", roi_set=rs)
        assert rs2 is rs
        assert len(rs.rois) == 1

    def test_rotate_arbitrary_warns_and_removes_rois(self):
        from probeflow.processing.state import apply_geometric_op_to_scan
        scan = self._make_scan()
        rs = ROISet(image_id="img1")
        rs.add(rect_roi())
        rs.add(point_roi())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            apply_geometric_op_to_scan(scan, "rotate_arbitrary",
                                       params={"angle_degrees": 30.0},
                                       roi_set=rs)
        assert len(rs.rois) == 0
        assert any("rotate_arbitrary" in str(warning.message) for warning in w)

    def test_no_roi_set_is_fine(self):
        from probeflow.processing.state import apply_geometric_op_to_scan
        scan = self._make_scan()
        scan2, rs = apply_geometric_op_to_scan(scan, "flip_vertical")
        assert rs is None


# ── Spectrum position ROIs ────────────────────────────────────────────────────

class TestSpectrumPositionROIs:
    def test_point_roi_with_linked_file(self):
        roi = ROI.new("point", {"x": 25.0, "y": 30.0},
                       name="spectrum_scan001_001",
                       linked_file="scan001_001.dat")
        assert roi.kind == "point"
        assert roi.name == "spectrum_scan001_001"
        assert roi.linked_file == "scan001_001.dat"
        mask = roi.to_mask((100, 100))
        assert mask[30, 25]

    def test_spec_roi_round_trips_with_linked_file(self):
        roi = ROI.new("point", {"x": 5.0, "y": 7.0},
                       name="spectrum_003",
                       linked_file="my_spec_003.dat")
        restored = ROI.from_dict(roi.to_dict())
        assert restored.linked_file == "my_spec_003.dat"
        assert restored.name == "spectrum_003"


# ── Provenance integration ────────────────────────────────────────────────────

class TestProvenanceIntegration:
    def test_roi_set_in_provenance_dict(self):
        from probeflow.provenance.export import build_scan_export_provenance
        from unittest.mock import MagicMock
        import numpy as np

        scan = MagicMock()
        scan.source_path = "/tmp/scan.sxm"
        scan.source_format = "sxm"
        scan.planes = [np.zeros((10, 10))]
        scan.plane_names = ["Z-fwd"]
        scan.plane_units = ["m"]
        scan.scan_range_m = (1e-8, 1e-8)
        scan.processing_history = []

        rs = ROISet(image_id="test_scan")
        rs.add(rect_roi())

        prov = build_scan_export_provenance(scan, roi_set=rs)
        d = prov.to_dict()
        assert "rois" in d
        assert d["rois"] is not None
        assert "rois" in d["rois"]

    def test_provenance_without_roi_set(self):
        from probeflow.provenance.export import build_scan_export_provenance
        from unittest.mock import MagicMock
        import numpy as np

        scan = MagicMock()
        scan.source_path = "/tmp/scan.sxm"
        scan.source_format = "sxm"
        scan.planes = [np.zeros((10, 10))]
        scan.plane_names = ["Z-fwd"]
        scan.plane_units = ["m"]
        scan.scan_range_m = (1e-8, 1e-8)
        scan.processing_history = []

        prov = build_scan_export_provenance(scan)
        d = prov.to_dict()
        assert "rois" in d
        assert d["rois"] is None
