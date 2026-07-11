"""ImageCanvas selection-interaction seams (GUI).

2026-07-06 review of the carried-forward image_canvas.py selection internals:

* Resize handles are pure data-model geometry, not scene items, so their hit
  test must honour ``set_rois_visible(False)`` itself — a pan click near an
  *invisible* handle used to silently start a resize and commit a geometry
  change the user could not see.
* Spec-marker clicks must only be intercepted in pan mode; with a drawing
  tool armed, a press near a marker used to open the spectrum dialog instead
  of starting the shape.
* Fit-to-view used the manual-zoom floor (0.25), so images larger than 4x
  the viewport could never actually fit.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

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


def _canvas_with_roi(roi, size: int = 64):
    from PySide6.QtGui import QPixmap
    from probeflow.core.roi import ROISet
    from probeflow.gui.image_canvas import ImageCanvas

    canvas = ImageCanvas()
    pm = QPixmap(size, size)
    pm.fill()
    canvas._view_scale_mode = "one_to_one"
    canvas.set_source(pm, reset_zoom=True)
    rs = ROISet(image_id="img")
    rs.add(roi)
    rs.set_active(roi.id)
    canvas.set_roi_set(rs)
    return canvas


def _press_move_release(canvas, qapp, press_xy, release_xy):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    def _evt(kind, xy):
        return QMouseEvent(
            kind, QPointF(*xy), QPointF(*xy), Qt.LeftButton,
            Qt.LeftButton if kind != QEvent.MouseButtonRelease else Qt.NoButton,
            Qt.NoModifier,
        )

    canvas.mousePressEvent(_evt(QEvent.MouseButtonPress, press_xy))
    canvas.mouseMoveEvent(_evt(QEvent.MouseMove, release_xy))
    canvas.mouseReleaseEvent(_evt(QEvent.MouseButtonRelease, release_xy))
    qapp.processEvents()


class TestHiddenOverlayHandleGuard:
    def _rect_roi(self):
        from probeflow.core.roi import ROI
        return ROI.new(
            "rectangle", {"x": 10.0, "y": 10.0, "width": 20.0, "height": 20.0}
        )

    def test_hidden_rois_do_not_start_handle_resize(self, qapp):
        roi = self._rect_roi()
        canvas = _canvas_with_roi(roi)
        try:
            canvas.set_rois_visible(False)
            changed = []
            canvas.roi_geometry_changed.connect(
                lambda rid, g: changed.append((rid, g)))
            # Press exactly on the SE corner handle (30, 30) and drag.
            _press_move_release(canvas, qapp, (30, 30), (40, 40))
            assert changed == []
            assert canvas._handle_roi_id is None
        finally:
            canvas.deleteLater()

    def test_visible_rois_still_resize(self, qapp):
        roi = self._rect_roi()
        canvas = _canvas_with_roi(roi)
        try:
            changed = []
            canvas.roi_geometry_changed.connect(
                lambda rid, g: changed.append((rid, g)))
            _press_move_release(canvas, qapp, (30, 30), (40, 40))
            assert len(changed) == 1
            assert changed[0][0] == roi.id
        finally:
            canvas.deleteLater()

    def test_hidden_rois_do_not_show_handle_cursor(self, qapp):
        from PySide6.QtCore import QPoint

        roi = self._rect_roi()
        canvas = _canvas_with_roi(roi)
        try:
            assert canvas._active_handle_hovered(QPoint(30, 30))
            canvas.set_rois_visible(False)
            assert not canvas._active_handle_hovered(QPoint(30, 30))
        finally:
            canvas.deleteLater()


class TestMarkerClickToolPriority:
    def _canvas_with_marker(self):
        from PySide6.QtGui import QPixmap
        from probeflow.gui.image_canvas import ImageCanvas

        canvas = ImageCanvas()
        pm = QPixmap(64, 64)
        pm.fill()
        canvas._view_scale_mode = "one_to_one"
        canvas.set_source(pm, reset_zoom=True)
        entry = SimpleNamespace(stem="specA")
        canvas.set_markers(
            [{"frac_x": 0.5, "frac_y": 0.5, "entry": entry}]
        )
        return canvas, entry

    def test_pan_mode_click_opens_marker(self, qapp):
        canvas, entry = self._canvas_with_marker()
        try:
            clicked = []
            canvas.marker_clicked.connect(clicked.append)
            _press_move_release(canvas, qapp, (32, 32), (32, 32))
            assert clicked == [entry]
        finally:
            canvas.deleteLater()

    def test_drawing_tool_click_draws_instead_of_opening_marker(self, qapp):
        canvas, _entry = self._canvas_with_marker()
        try:
            clicked = []
            drawn = []
            canvas.marker_clicked.connect(clicked.append)
            canvas.selection_drawn.connect(
                lambda kind, geom: drawn.append((kind, geom)))
            canvas.set_tool("rectangle")
            _press_move_release(canvas, qapp, (32, 32), (48, 48))
            assert clicked == []
            assert len(drawn) == 1
            assert drawn[0][0] == "rectangle"
        finally:
            canvas.deleteLater()


def _canvas_with_selection(kind="rectangle", geometry=None, size=100):
    from PySide6.QtGui import QPixmap
    from probeflow.gui.image_canvas import ImageCanvas

    canvas = ImageCanvas()
    pm = QPixmap(size, size)
    pm.fill()
    canvas._view_scale_mode = "one_to_one"
    canvas.set_source(pm, reset_zoom=True)
    canvas.set_selection(
        kind,
        geometry or {"x": 10.0, "y": 10.0, "width": 60.0, "height": 40.0},
    )
    return canvas


class TestQuickSelectionEditing:
    """The quick selection must be resizable and movable like a user expects.

    Regression guard: when the area tools switched from creating ROIs to
    ephemeral selections (2f4d0d3), the marquee lost the resize handles the
    ROIs had — selections could only be redrawn from scratch.
    """

    def test_selection_renders_resize_handles(self, qapp):
        canvas = _canvas_with_selection()
        try:
            assert len(canvas._selection_handle_items) == 8  # rect: 4 corners + 4 edges
            canvas.set_selection("ellipse", {"cx": 50.0, "cy": 50.0, "rx": 20.0, "ry": 15.0})
            assert len(canvas._selection_handle_items) == 4  # ellipse: n/e/s/w
            canvas.clear_selection(emit=False)
            assert canvas._selection_handle_items == []
        finally:
            canvas.deleteLater()

    def test_delete_key_clears_selection_when_no_active_roi(self, qapp):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent

        canvas = _canvas_with_selection()
        try:
            cleared = []
            canvas.selection_cleared.connect(lambda: cleared.append(True))
            event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)
            canvas.keyPressEvent(event)
            assert canvas._selection is None
            assert cleared == [True]
        finally:
            canvas.deleteLater()

    def test_delete_key_prefers_active_roi_over_selection(self, qapp):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from probeflow.core.roi import ROI, ROISet

        canvas = _canvas_with_selection()
        try:
            roi_set = ROISet(image_id="scan")
            roi = ROI.new("rect", {"x": 1, "y": 1, "width": 5, "height": 5})
            roi_set.add(roi)
            roi_set.active_roi_id = roi.id
            canvas._image_roi_set = roi_set
            deletes = []
            canvas.roi_delete_requested.connect(lambda roi_id: deletes.append(roi_id))
            event = QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)
            canvas.keyPressEvent(event)
            assert deletes == [roi.id]
            assert canvas._selection is not None  # ROI deletion wins; selection stays
        finally:
            canvas.deleteLater()

    def test_corner_handle_drag_resizes_selection(self, qapp):
        canvas = _canvas_with_selection()
        try:
            drawn = []
            canvas.selection_drawn.connect(lambda k, g: drawn.append((k, g)))
            # SE corner handle sits at (70, 50); drag it to (80, 60).
            _press_move_release(canvas, qapp, (70, 50), (80, 60))
            sel = canvas.selection()
            assert sel["kind"] == "rectangle"
            g = sel["geometry"]
            assert (g["x"], g["y"]) == (10.0, 10.0)
            assert (g["width"], g["height"]) == (70.0, 50.0)
            # Commit re-emits through the same pipeline as a fresh draw.
            assert len(drawn) == 1 and drawn[0][0] == "rectangle"
        finally:
            canvas.deleteLater()

    def test_outline_drag_moves_selection(self, qapp):
        canvas = _canvas_with_selection()
        try:
            # (25, 10) lies on the top edge, > 12 px from both the nw (10,10)
            # and n (40,10) handles.
            _press_move_release(canvas, qapp, (25, 10), (30, 20))
            g = canvas.selection()["geometry"]
            assert (g["x"], g["y"]) == (15.0, 20.0)
            assert (g["width"], g["height"]) == (60.0, 40.0)
        finally:
            canvas.deleteLater()

    def test_interior_press_still_pans_not_moves(self, qapp):
        canvas = _canvas_with_selection()
        try:
            before = canvas.selection()["geometry"]
            _press_move_release(canvas, qapp, (40, 30), (50, 40))  # centre
            assert canvas.selection()["geometry"] == before
        finally:
            canvas.deleteLater()

    def test_polygon_selection_moves_via_outline(self, qapp):
        canvas = _canvas_with_selection(
            "polygon",
            {"vertices": [[20.0, 20.0], [80.0, 20.0], [50.0, 80.0]]},
        )
        try:
            assert canvas._selection_handle_items == []  # move-only, like ROIs
            _press_move_release(canvas, qapp, (50, 20), (55, 30))  # top edge
            vs = canvas.selection()["geometry"]["vertices"]
            assert vs[0] == [25.0, 30.0]
        finally:
            canvas.deleteLater()


class TestOutlinePensStayThin:
    def test_selection_and_roi_pens_are_cosmetic(self, qapp):
        from probeflow.gui import roi_items
        from probeflow.gui.image_canvas import _PEN_PREVIEW, _PEN_SELECTION

        canvas = _canvas_with_selection()
        try:
            pen = canvas._selection_item.pen()
            assert pen.isCosmetic()
            assert pen.widthF() == 1.0
        finally:
            canvas.deleteLater()
        assert _PEN_SELECTION.isCosmetic() and _PEN_SELECTION.widthF() == 1.0
        assert _PEN_PREVIEW.isCosmetic()
        for pen in (roi_items._PEN_INACTIVE, roi_items._PEN_ACTIVE, roi_items._PEN_HOVER):
            assert pen.isCosmetic()


class TestAdaptiveZoomCeiling:
    """Small scans need far more than the classic 8x to inspect pixels."""

    def _canvas_with_image(self, size: int):
        from PySide6.QtGui import QPixmap
        from probeflow.gui.image_canvas import ImageCanvas

        canvas = ImageCanvas()
        pm = QPixmap(size, size)
        pm.fill()
        canvas.set_source(pm, reset_zoom=True)
        return canvas

    def test_small_image_can_zoom_well_beyond_8x(self, qapp):
        canvas = self._canvas_with_image(64)
        try:
            for _ in range(40):
                canvas.zoom_by(2.0)
            assert canvas.zoom() == pytest.approx(8192.0 / 64)  # 128x
        finally:
            canvas.deleteLater()

    def test_large_image_keeps_classic_8x_cap(self, qapp):
        canvas = self._canvas_with_image(2048)
        try:
            for _ in range(10):
                canvas.zoom_by(2.0)
            assert canvas.zoom() == pytest.approx(8.0)
        finally:
            canvas.deleteLater()


class TestFitZoomFloor:
    def test_fit_zoom_can_shrink_below_manual_floor(self, qapp):
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QScrollArea
        from probeflow.gui.image_canvas import ImageCanvas

        sa = QScrollArea()
        canvas = ImageCanvas()
        sa.setWidget(canvas)
        sa.resize(320, 320)
        try:
            pm = QPixmap(4000, 4000)
            pm.fill()
            canvas.set_source(pm, reset_zoom=False)
            canvas.fit_to_view()
            qapp.processEvents()
            vp_w = sa.viewport().width()
            vp_h = sa.viewport().height()
            assert vp_w > 0 and vp_h > 0
            # A 4000 px image in a small viewport must fit below the manual
            # zoom floor; before the fix the clamp pinned this at 0.25.
            assert canvas.zoom() < 0.25
            assert canvas.width() <= vp_w + 1
            assert canvas.height() <= vp_h + 1
        finally:
            canvas.deleteLater()
            sa.deleteLater()

    def test_canvas_can_shrink_after_growing(self, qapp):
        # Regression (dataset_builder merge): _apply_zoom floored the new
        # size at self.minimumWidth() — but setFixedSize itself installs
        # minimum == maximum, so after one large layout the canvas could
        # never shrink and zoom-out/fit silently stopped working.
        from PySide6.QtGui import QPixmap
        from probeflow.gui.image_canvas import ImageCanvas

        canvas = ImageCanvas()
        try:
            pm = QPixmap(400, 400)
            pm.fill()
            canvas._view_scale_mode = "one_to_one"
            canvas.set_source(pm, reset_zoom=True)
            assert canvas.width() == 400
            canvas.zoom_by(0.5)  # 400 -> 200
            assert canvas.width() == 200
        finally:
            canvas.deleteLater()

    def test_external_minimum_size_still_honoured(self, qapp):
        # The Dataset Builder pane sets a minimum canvas size before any
        # pixmap is loaded; zooming small must not go below it.
        from PySide6.QtGui import QPixmap
        from probeflow.gui.image_canvas import ImageCanvas

        canvas = ImageCanvas()
        try:
            canvas.setMinimumSize(420, 360)
            pm = QPixmap(400, 400)
            pm.fill()
            canvas._view_scale_mode = "one_to_one"
            canvas.set_source(pm, reset_zoom=True)
            canvas.zoom_by(0.5)
            assert canvas.width() >= 420
            assert canvas.height() >= 360
        finally:
            canvas.deleteLater()
