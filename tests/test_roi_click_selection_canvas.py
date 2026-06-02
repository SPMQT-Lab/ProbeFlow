"""Headless GUI tests for line-ROI click selection on ImageCanvas.

Regression cover for the "hover highlight lies about what a click does" bug:
when a non-active line ROI is highlighted under the cursor, a click must select
*that* line, not grab the nearby endpoint handle of the active line. The
convenience path (grabbing the active ROI's endpoint when nothing else is
hovered) must keep working.
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


def _canvas_with_lines(*geoms):
    """Build an ImageCanvas at 1:1 zoom with the given line ROIs; first active."""
    from PySide6.QtGui import QPixmap
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui.image_canvas import ImageCanvas

    canvas = ImageCanvas()
    pm = QPixmap(64, 64)
    pm.fill()
    canvas._view_scale_mode = "one_to_one"
    canvas.set_source(pm, reset_zoom=True)
    rs = ROISet(image_id="img")
    rois = [ROI.new("line", dict(g)) for g in geoms]
    for r in rois:
        rs.add(r)
    rs.set_active(rois[0].id)
    canvas.set_roi_set(rs)
    return canvas, rois


def _move(canvas, xy):
    """Drive a no-button move in pan mode (updates the hover highlight)."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    canvas.mouseMoveEvent(
        QMouseEvent(QEvent.MouseMove, QPointF(*xy), Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    )


def _press(canvas, xy):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    canvas.mousePressEvent(
        QMouseEvent(QEvent.MouseButtonPress, QPointF(*xy), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    )


class TestHighlightedLineSelection:
    def test_click_highlighted_nonactive_line_activates_it_not_active_handle(self, qapp):
        # Active line (10,10)->(50,50); other line (10,30)->(52,52) converges near
        # the active endpoint but is clearly separate at its start.
        canvas, (active, other) = _canvas_with_lines(
            {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
            {"x1": 10.0, "y1": 30.0, "x2": 52.0, "y2": 52.0},
        )
        activated = []
        canvas.roi_activate_requested.connect(activated.append)
        try:
            # Hover the start of the other line, far from the active line: this is
            # what the user sees highlighted.
            _move(canvas, (10, 30))
            assert canvas._hover_roi_id == other.id
            # Press near the active line's p2 endpoint (within the 12px box).
            _press(canvas, (50, 50))
            # The highlighted (other) line is activated; the active line's handle
            # is NOT grabbed.
            assert activated == [other.id]
            assert canvas._handle_roi_id is None
        finally:
            canvas.deleteLater()
            qapp.processEvents()

    def test_click_near_active_endpoint_with_no_hover_still_grabs_handle(self, qapp):
        # A single active line; press just past its p2 endpoint (off the segment
        # but within 12px) with nothing hovered -> endpoint grab convenience.
        canvas, (active,) = _canvas_with_lines(
            {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
        )
        try:
            assert canvas._hover_roi_id is None
            _press(canvas, (54, 54))
            assert canvas._handle_roi_id == active.id
            assert canvas._handle_name == "p2"
        finally:
            canvas.deleteLater()
            qapp.processEvents()

    def test_click_on_active_line_starts_move(self, qapp):
        canvas, (active,) = _canvas_with_lines(
            {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0},
        )
        try:
            # Hover + press the middle of the active line, clear of its endpoints.
            _move(canvas, (30, 30))
            assert canvas._hover_roi_id == active.id
            _press(canvas, (30, 30))
            assert canvas._move_roi_id == active.id
            assert canvas._handle_roi_id is None
        finally:
            canvas.deleteLater()
            qapp.processEvents()
