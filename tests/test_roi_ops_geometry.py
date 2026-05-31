"""Tests for the generic ROI geometry-commit helpers in roi_ops.

These are GUI-free: they operate on a real ROISet and a plain callback, so no
QApplication is needed.
"""

from __future__ import annotations

from probeflow.core.roi import ROI, ROISet
from probeflow.gui.viewer.roi_ops import roi_geometry_changed


def _set_with(roi: ROI) -> ROISet:
    rs = ROISet(image_id="img")
    rs.add(roi)
    rs.set_active(roi.id)
    return rs


class TestRoiGeometryChanged:
    def test_commits_new_geometry_and_keeps_identity(self):
        roi = ROI.new("rectangle", {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0})
        roi.coord_system = "physical"
        roi.linked_file = "scan_001.sxm"
        rs = _set_with(roi)
        calls = []

        roi_geometry_changed(
            rs, roi.id,
            {"x": 5.0, "y": 6.0, "width": 7.0, "height": 8.0},
            lambda: calls.append(1),
        )

        out = rs.get(roi.id)
        assert out.geometry == {"x": 5.0, "y": 6.0, "width": 7.0, "height": 8.0}
        assert out.id == roi.id
        assert out.name == roi.name
        assert out.kind == "rectangle"
        assert out.coord_system == "physical"
        assert out.linked_file == "scan_001.sxm"
        assert rs.active_roi_id == roi.id
        assert calls == [1]

    def test_copies_geometry_dict(self):
        roi = ROI.new("rectangle", {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0})
        rs = _set_with(roi)
        geom = {"x": 2.0, "y": 2.0, "width": 4.0, "height": 4.0}

        roi_geometry_changed(rs, roi.id, geom, lambda: None)
        geom["x"] = 999.0  # mutating the caller's dict must not affect the ROI

        assert rs.get(roi.id).geometry["x"] == 2.0

    def test_missing_roi_is_noop(self):
        rs = ROISet(image_id="img")
        calls = []
        roi_geometry_changed(rs, "nope", {"x": 1.0}, lambda: calls.append(1))
        assert calls == []

    def test_none_roi_set_is_noop(self):
        # Must not raise.
        roi_geometry_changed(None, "any", {"x": 1.0}, lambda: None)

    def test_commits_line_geometry_preserving_width(self):
        # Line endpoints + width now flow through the generic helper (the canvas
        # builds the geometry dict via resize_roi, which preserves extra keys).
        line = ROI.new("line", {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0, "width": 3})
        rs = _set_with(line)
        roi_geometry_changed(
            rs, line.id,
            {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "width": 3},
            lambda: None,
        )
        assert rs.get(line.id).geometry == {
            "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "width": 3,
        }
