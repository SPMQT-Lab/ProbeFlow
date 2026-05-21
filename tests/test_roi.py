"""Contract tests for ROI data models, masks, transforms, and integration."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from probeflow.core.roi import ROI, ROISet, combine_masks, invert_mask, translate


SHAPE = (100, 100)  # (Ny, Nx)


def rect_roi(x=10, y=10, w=20, h=20):
    return ROI.new(
        "rectangle",
        {"x": float(x), "y": float(y), "width": float(w), "height": float(h)},
    )


def ellipse_roi(cx=50, cy=50, rx=10, ry=10):
    return ROI.new(
        "ellipse",
        {"cx": float(cx), "cy": float(cy), "rx": float(rx), "ry": float(ry)},
    )


def polygon_roi():
    return ROI.new(
        "polygon",
        {"vertices": [[10.0, 10.0], [30.0, 10.0], [20.0, 30.0]]},
    )


def freehand_roi():
    return ROI.new(
        "freehand",
        {"vertices": [[1.0, 2.0], [3.0, 5.0], [8.0, 13.0]]},
    )


def line_roi():
    return ROI.new("line", {"x1": 5.0, "y1": 5.0, "x2": 15.0, "y2": 15.0})


def point_roi(x=50, y=50, *, name=None, linked_file=None):
    return ROI.new(
        "point",
        {"x": float(x), "y": float(y)},
        name=name,
        linked_file=linked_file,
    )


def _masks():
    a = np.zeros((5, 5), dtype=bool)
    a[0:3, 0:3] = True
    b = np.zeros((5, 5), dtype=bool)
    b[2:5, 2:5] = True
    return a, b


def _scan():
    scan = MagicMock()
    scan.planes = [np.arange(12.0).reshape(3, 4), np.ones((3, 4))]
    scan.scan_range_m = (4e-9, 3e-9)
    return scan


def test_roi_serialisation_round_trips_all_supported_shapes_and_links():
    rois = [
        rect_roi(),
        ellipse_roi(),
        polygon_roi(),
        freehand_roi(),
        line_roi(),
        point_roi(5, 7, name="spectrum_003", linked_file="my_spec_003.dat"),
        ROI.new(
            "multipolygon",
            {
                "components": [
                    {
                        "exterior": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                        "holes": [[[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]]],
                    },
                ],
            },
            name="donut",
        ),
    ]

    for roi in rois:
        restored = ROI.from_dict(roi.to_dict())
        assert restored.id == roi.id
        assert restored.name == roi.name
        assert restored.kind == roi.kind
        assert restored.geometry == roi.geometry
        assert restored.linked_file == roi.linked_file


def test_roi_masks_cover_expected_pixels_for_supported_shapes():
    rect = rect_roi(x=10, y=10, w=10, h=10).to_mask(SHAPE)
    clipped_rect = rect_roi(x=90, y=90, w=20, h=20).to_mask(SHAPE)
    ellipse = ellipse_roi(cx=50.5, cy=50.5, rx=5, ry=5).to_mask(SHAPE)
    polygon = polygon_roi().to_mask(SHAPE)
    line = line_roi().to_mask(SHAPE)
    point = point_roi(x=30, y=40).to_mask(SHAPE)
    out_of_bounds_point = point_roi(x=200, y=200).to_mask(SHAPE)

    for mask in (rect, clipped_rect, ellipse, polygon, line, point, out_of_bounds_point):
        assert mask.shape == SHAPE
    assert int(rect.sum()) == 100
    assert 0 < clipped_rect.sum() < 400
    assert 65 <= int(ellipse.sum()) <= 92
    assert polygon.sum() > 0
    assert line[5, 5]
    assert line[15, 15]
    assert int(point.sum()) == 1
    assert point[40, 30]
    assert int(out_of_bounds_point.sum()) == 0


def test_roi_bounds_and_crop_contract():
    roi = rect_roi(x=10, y=20, w=5, h=8)
    arr = np.zeros(SHAPE)
    arr[20:28, 10:15] = 1.0

    assert roi.bounds(SHAPE) == (20, 27, 10, 14)
    crop = roi.crop(arr)
    assert crop.shape == (8, 5)
    assert np.all(crop == 1.0)


def test_lossless_transforms_preserve_expected_coordinates_and_identity():
    cases = [
        ("flip_horizontal", rect_roi(x=10, y=10, w=20, h=20), (100, 100), {"x": 70.0, "y": 10.0, "width": 20.0}),
        ("flip_vertical", rect_roi(x=10, y=10, w=20, h=20), (100, 100), {"y": 70.0, "height": 20.0}),
        ("rot90_cw", rect_roi(x=10, y=5, w=30, h=20), (80, 100), {"x": 55.0, "y": 10.0, "width": 20.0, "height": 30.0}),
        ("rot180", rect_roi(x=10, y=10, w=20, h=20), (100, 100), {"x": 70.0, "y": 70.0}),
        ("rot270_cw", rect_roi(x=10, y=5, w=30, h=20), (80, 100), {"x": 5.0, "y": 60.0, "width": 20.0, "height": 30.0}),
    ]

    for op, roi, shape, expected in cases:
        transformed = roi.transform(op, {}, shape)
        assert transformed is not None
        assert transformed.id == roi.id
        for key, value in expected.items():
            assert transformed.geometry[key] == pytest.approx(value)

    ellipse = ellipse_roi(cx=30, cy=20, rx=10, ry=5).transform("rot90_cw", {}, SHAPE)
    assert ellipse is not None
    assert ellipse.geometry["rx"] == pytest.approx(5.0)
    assert ellipse.geometry["ry"] == pytest.approx(10.0)

    polygon = polygon_roi().transform("rot90_cw", {}, SHAPE)
    assert polygon is not None
    assert len(polygon.geometry["vertices"]) == 3

    point = point_roi(x=10, y=0).transform("flip_horizontal", {}, SHAPE)
    assert point is not None
    assert point.geometry["x"] == pytest.approx(89.0)

    assert rect_roi().transform("rotate_arbitrary", {}, SHAPE) is None
    with pytest.raises(ValueError):
        rect_roi().transform("shear", {}, SHAPE)


def test_crop_transform_shifts_clips_and_drops_rois_as_expected():
    shifted = rect_roi(x=20, y=30, w=10, h=10).transform(
        "crop",
        {"x0": 15, "y0": 25, "x1": 40, "y1": 50},
        SHAPE,
    )
    clipped = rect_roi(x=20, y=20, w=20, h=20).transform(
        "crop",
        {"x0": 25, "y0": 25, "x1": 50, "y1": 50},
        SHAPE,
    )

    assert shifted is not None
    assert shifted.geometry["x"] == pytest.approx(5.0)
    assert shifted.geometry["y"] == pytest.approx(5.0)
    assert rect_roi(x=60, y=60, w=10, h=10).transform(
        "crop",
        {"x0": 0, "y0": 0, "x1": 30, "y1": 30},
        SHAPE,
    ) is None
    assert clipped is not None
    assert clipped.geometry["width"] < 20.0
    assert point_roi(x=80, y=80).transform(
        "crop",
        {"x0": 0, "y0": 0, "x1": 50, "y1": 50},
        SHAPE,
    ) is None


def test_roi_set_add_remove_lookup_active_and_serialisation_contract():
    roi_set = ROISet(image_id="img1")
    rect = rect_roi()
    named = point_roi(x=1, y=1, name="my_point")

    roi_set.add(rect)
    roi_set.add(named)
    assert roi_set.get(rect.id) is rect
    assert roi_set.get_by_name("my_point") is named
    roi_set.set_active(rect.id)
    assert roi_set.active_roi_id == rect.id
    roi_set.remove(rect.id)
    assert roi_set.active_roi_id is None
    assert roi_set.get(named.id) is named
    roi_set.remove("nonexistent")
    roi_set.set_active(None)
    assert roi_set.active_roi_id is None

    with pytest.raises(ValueError):
        roi_set.set_active("nonexistent")

    roi_set.add(ellipse_roi())
    roi_set.set_active(roi_set.rois[0].id)
    restored = ROISet.from_dict(roi_set.to_dict())
    assert restored.image_id == "img1"
    assert restored.active_roi_id == roi_set.rois[0].id
    assert [roi.kind for roi in restored.rois] == ["point", "ellipse"]

    with pytest.raises(ValueError, match="Failed to reconstruct ROI"):
        ROISet.from_dict({"image_id": "img1", "rois": [{"bad": "data"}], "active_roi_id": None})


def test_roi_set_transform_all_updates_supported_rois_and_reports_invalidated_rois():
    roi_set = ROISet(image_id="img1")
    roi_set.add(rect_roi())
    roi_set.add(rect_roi())

    assert roi_set.transform_all("flip_horizontal", {}, SHAPE) == []
    assert len(roi_set.rois) == 2
    assert roi_set.rois[0].geometry["x"] == pytest.approx(70.0)

    roi_set.add(point_roi())
    invalidated = roi_set.transform_all("rotate_arbitrary", {}, SHAPE)
    assert len(invalidated) == 3
    assert len(roi_set.rois) == 3


def test_translate_contract_for_supported_shapes_and_identity_fields():
    assert translate(line_roi(), 2.5, -1.5).geometry == {
        "x1": 7.5,
        "y1": 3.5,
        "x2": 17.5,
        "y2": 13.5,
    }
    rect = translate(rect_roi(x=10, y=20, w=30, h=40), -3.0, 4.0)
    ellipse = translate(ellipse_roi(cx=12, cy=24, rx=5, ry=6), 10.0, -2.0)
    polygon = translate(polygon_roi(), 1.0, 2.0)
    freehand = translate(freehand_roi(), -1.0, 3.0)
    point = translate(point_roi(x=4, y=9), 0.5, -2.5)

    assert rect.geometry == {"x": 7.0, "y": 24.0, "width": 30.0, "height": 40.0}
    assert ellipse.geometry == {"cx": 22.0, "cy": 22.0, "rx": 5.0, "ry": 6.0}
    assert polygon.geometry["vertices"] == [[11.0, 12.0], [31.0, 12.0], [21.0, 32.0]]
    assert freehand.geometry["vertices"] == [[0.0, 5.0], [2.0, 8.0], [7.0, 16.0]]
    assert point.geometry == {"x": 4.5, "y": 6.5}

    linked = point_roi(x=4, y=9, name="probe_site", linked_file="scan_001.sxm")
    linked.coord_system = "physical"
    moved = translate(linked, 1.0, 1.0)
    assert moved.id == linked.id
    assert moved.name == "probe_site"
    assert moved.kind == "point"
    assert moved.coord_system == "physical"
    assert moved.linked_file == "scan_001.sxm"


def test_mask_helper_contracts():
    a, b = _masks()

    assert combine_masks([a, b], "union").sum() == 17
    intersection = combine_masks([a, b], "intersection")
    assert intersection.sum() == 1
    assert intersection[2, 2]
    assert combine_masks([a, b], "difference").sum() == 8
    assert combine_masks([a, b], "xor").sum() == 16
    np.testing.assert_array_equal(combine_masks([a], "union"), a)

    with pytest.raises(ValueError):
        combine_masks([], "union")
    with pytest.raises(ValueError):
        combine_masks([a, b], "subtract")  # type: ignore[arg-type]

    inverted = invert_mask(np.array([[True, False], [False, True]]))
    np.testing.assert_array_equal(inverted, np.array([[False, True], [True, False]]))


def test_apply_geometric_op_to_scan_updates_planes_and_roi_set():
    from probeflow.processing.state import apply_geometric_op_to_scan

    scan = _scan()
    original = scan.planes[0].copy()
    apply_geometric_op_to_scan(scan, "flip_horizontal")
    np.testing.assert_array_equal(scan.planes[0], np.fliplr(original))

    scan = _scan()
    apply_geometric_op_to_scan(scan, "rot90_cw")
    for plane in scan.planes:
        assert plane.shape == (4, 3)

    scan = _scan()
    roi_set = ROISet(image_id="img1")
    roi_set.add(rect_roi(x=0, y=0, w=2, h=2))
    _, returned_roi_set = apply_geometric_op_to_scan(scan, "flip_horizontal", roi_set=roi_set)
    assert returned_roi_set is roi_set
    assert len(roi_set.rois) == 1

    scan = _scan()
    _, returned_none = apply_geometric_op_to_scan(scan, "flip_vertical")
    assert returned_none is None


def test_rotate_arbitrary_geometric_op_warns_and_removes_rois():
    from probeflow.processing.state import apply_geometric_op_to_scan

    scan = _scan()
    roi_set = ROISet(image_id="img1")
    active = rect_roi()
    roi_set.add(active)
    roi_set.add(point_roi())
    roi_set.set_active(active.id)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        apply_geometric_op_to_scan(
            scan,
            "rotate_arbitrary",
            params={"angle_degrees": 30.0},
            roi_set=roi_set,
        )

    assert len(roi_set.rois) == 0
    assert roi_set.active_roi_id is None
    assert any("rotate_arbitrary" in str(warning.message) for warning in caught)


def test_spectrum_position_roi_contract():
    roi = point_roi(
        x=25,
        y=30,
        name="spectrum_scan001_001",
        linked_file="scan001_001.dat",
    )

    assert roi.kind == "point"
    assert roi.name == "spectrum_scan001_001"
    assert roi.linked_file == "scan001_001.dat"
    assert roi.to_mask(SHAPE)[30, 25]
    assert ROI.from_dict(roi.to_dict()).linked_file == "scan001_001.dat"


def test_roi_set_provenance_contract():
    from probeflow.provenance.export import build_scan_export_provenance

    scan = MagicMock()
    scan.source_path = "/tmp/scan.sxm"
    scan.source_format = "sxm"
    scan.planes = [np.zeros((10, 10))]
    scan.plane_names = ["Z-fwd"]
    scan.plane_units = ["m"]
    scan.scan_range_m = (1e-8, 1e-8)
    scan.processing_history = []

    roi_set = ROISet(image_id="test_scan")
    roi_set.add(rect_roi())

    assert build_scan_export_provenance(scan, roi_set=roi_set).to_dict()["rois"]["rois"]
    assert build_scan_export_provenance(scan).to_dict()["rois"] is None
