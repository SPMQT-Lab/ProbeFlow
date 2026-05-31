"""Headless GUI tests for the generic ROI resize-handle drag on ImageCanvas.

Exercises the migrated handle system end to end: handle rendering (count +
visibility), and the press→move→release path that emits roi_geometry_changed
with the resized geometry. Rectangle (newly resizable) and line (migrated onto
the same path) are both covered.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PySide6 unavailable: {exc}")
    app = QApplication.instance()
    if app is not None:
        return app
    try:
        return QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"QApplication unavailable: {exc}")


def _canvas_with_roi(roi):
    """Build an ImageCanvas at 1:1 zoom with a single active ROI."""
    from PySide6.QtGui import QPixmap
    from probeflow.core.roi import ROISet
    from probeflow.gui.image_canvas import ImageCanvas

    canvas = ImageCanvas()
    pm = QPixmap(64, 64)
    pm.fill()
    canvas._view_scale_mode = "one_to_one"
    canvas.set_source(pm, reset_zoom=True)
    rs = ROISet(image_id="img")
    rs.add(roi)
    rs.set_active(roi.id)
    canvas.set_roi_set(rs)
    return canvas


def _press_move_release(canvas, qapp, press_xy, release_xy):
    """Drive a left-button press→move→release at the given view positions."""
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    def _evt(kind, xy):
        return QMouseEvent(
            kind, QPointF(*xy), Qt.LeftButton,
            Qt.LeftButton if kind != QEvent.MouseButtonRelease else Qt.NoButton,
            Qt.NoModifier,
        )

    canvas.mousePressEvent(_evt(QEvent.MouseButtonPress, press_xy))
    canvas.mouseMoveEvent(_evt(QEvent.MouseMove, release_xy))
    canvas.mouseReleaseEvent(_evt(QEvent.MouseButtonRelease, release_xy))
    qapp.processEvents()
    return QPoint, QPointF


class TestRectangleHandleRendering:
    def test_active_rectangle_shows_eight_handles(self, qapp):
        from probeflow.core.roi import ROI
        roi = ROI.new("rectangle", {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0})
        canvas = _canvas_with_roi(roi)
        try:
            item = canvas._roi_items[roi.id]
            handles = item._resize_handles
            assert set(handles) == {"nw", "ne", "se", "sw", "n", "e", "s", "w"}
            assert all(h.isVisible() for h in handles.values())
        finally:
            canvas.deleteLater()
            qapp.processEvents()

    def test_inactive_rectangle_hides_handles(self, qapp):
        from probeflow.core.roi import ROI
        roi = ROI.new("rectangle", {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0})
        canvas = _canvas_with_roi(roi)
        try:
            canvas._image_roi_set.set_active(None)
            canvas._rebuild_roi_items()
            item = canvas._roi_items[roi.id]
            assert all(not h.isVisible() for h in item._resize_handles.values())
        finally:
            canvas.deleteLater()
            qapp.processEvents()


class TestResizeDragCommits:
    def test_dragging_se_corner_emits_resized_geometry(self, qapp):
        from probeflow.core.roi import ROI
        roi = ROI.new("rectangle", {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0})
        canvas = _canvas_with_roi(roi)
        committed = []
        canvas.roi_geometry_changed.connect(lambda rid, g: committed.append((rid, g)))
        try:
            # SE corner is at scene (30, 30); at 1:1 zoom view≈scene. Drag to (50, 60).
            _press_move_release(canvas, qapp, (30, 30), (50, 60))
            assert committed, "expected a roi_geometry_changed emission"
            rid, geom = committed[-1]
            assert rid == roi.id
            # top-left anchored at (10,10); new width/height = 40/50.
            assert geom["x"] == pytest.approx(10.0)
            assert geom["y"] == pytest.approx(10.0)
            assert geom["width"] == pytest.approx(40.0)
            assert geom["height"] == pytest.approx(50.0)
        finally:
            canvas.deleteLater()
            qapp.processEvents()

    def test_press_away_from_handle_does_not_resize(self, qapp):
        from probeflow.core.roi import ROI
        # Use a large rectangle so the centre is well clear (>12px) of every
        # handle — handles sit at corners and edge midpoints.
        roi = ROI.new("rectangle", {"x": 4.0, "y": 4.0, "width": 56.0, "height": 56.0})
        canvas = _canvas_with_roi(roi)
        committed = []
        canvas.roi_geometry_changed.connect(lambda rid, g: committed.append((rid, g)))
        try:
            # Centre (32,32): nearest handle is an edge midpoint ~28px away.
            _press_move_release(canvas, qapp, (32, 32), (37, 37))
            assert committed == []
            assert canvas._handle_roi_id is None
        finally:
            canvas.deleteLater()
            qapp.processEvents()


class TestEllipseResizable:
    def test_ellipse_shows_four_cardinal_handles(self, qapp):
        from probeflow.core.roi import ROI
        roi = ROI.new("ellipse", {"cx": 30.0, "cy": 30.0, "rx": 10.0, "ry": 10.0})
        canvas = _canvas_with_roi(roi)
        try:
            item = canvas._roi_items[roi.id]
            assert set(item._resize_handles) == {"n", "e", "s", "w"}
            assert all(h.isVisible() for h in item._resize_handles.values())
        finally:
            canvas.deleteLater()
            qapp.processEvents()

    def test_dragging_east_handle_grows_rx_about_centre(self, qapp):
        from probeflow.core.roi import ROI
        roi = ROI.new("ellipse", {"cx": 30.0, "cy": 30.0, "rx": 10.0, "ry": 10.0})
        canvas = _canvas_with_roi(roi)
        committed = []
        canvas.roi_geometry_changed.connect(lambda rid, g: committed.append((rid, g)))
        try:
            # East handle at scene (40,30); drag to (55,30) → rx=25, centre fixed.
            _press_move_release(canvas, qapp, (40, 30), (55, 30))
            assert committed
            _, geom = committed[-1]
            assert geom["cx"] == pytest.approx(30.0)
            assert geom["cy"] == pytest.approx(30.0)
            assert geom["rx"] == pytest.approx(25.0)
            assert geom["ry"] == pytest.approx(10.0)
        finally:
            canvas.deleteLater()
            qapp.processEvents()


class TestLineStillEditable:
    def test_line_shows_two_handles_and_resizes(self, qapp):
        from probeflow.core.roi import ROI
        roi = ROI.new("line", {"x1": 5.0, "y1": 5.0, "x2": 25.0, "y2": 25.0, "width": 2})
        canvas = _canvas_with_roi(roi)
        committed = []
        canvas.roi_geometry_changed.connect(lambda rid, g: committed.append((rid, g)))
        try:
            item = canvas._roi_items[roi.id]
            assert set(item._resize_handles) == {"p1", "p2"}
            # Drag p2 from scene (25,25) to (40,30).
            _press_move_release(canvas, qapp, (25, 25), (40, 30))
            assert committed
            rid, geom = committed[-1]
            assert geom["x1"] == pytest.approx(5.0)
            assert geom["y1"] == pytest.approx(5.0)
            assert geom["x2"] == pytest.approx(40.0)
            assert geom["y2"] == pytest.approx(30.0)
            assert geom["width"] == 2  # preserved
        finally:
            canvas.deleteLater()
            qapp.processEvents()
