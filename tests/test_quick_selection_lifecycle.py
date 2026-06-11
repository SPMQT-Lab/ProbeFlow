"""Headless GUI tests for the quick-selection (ImageJ-style) lifecycle.

Review focus #2: the quick selection is ephemeral canvas state that the
viewer mixin promotes / clears / transforms. These tests pin the lifecycle
contracts: clearing always resynchronises selection-gated UI (the 2026-06-11
review found `_clear_quick_selection` skipped the resync its suppressed
canvas signal would have performed), geometric ops carry or drop the
selection correctly, and promotion works for every area kind.
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


def _canvas(qapp):
    from PySide6.QtGui import QPixmap

    from probeflow.gui.image_canvas import ImageCanvas

    canvas = ImageCanvas()
    pm = QPixmap(64, 64)
    pm.fill()
    canvas._view_scale_mode = "one_to_one"
    canvas.set_source(pm, reset_zoom=True)
    return canvas


def _host(canvas):
    """Minimal viewer stand-in mixing in the selection lifecycle handlers."""
    from probeflow.core.roi import ROISet
    from probeflow.gui.viewer.image_viewer_selection_mixin import (
        ImageViewerSelectionMixin,
    )

    class Host(ImageViewerSelectionMixin):
        def __init__(self):
            self._zoom_lbl = canvas
            self._image_roi_set = ROISet(image_id="img")
            self._status_lbl = SimpleNamespace(setText=self._record_status)
            self.statuses: list[str] = []
            self.synced = 0

        def _record_status(self, text):
            self.statuses.append(text)

        def _sync_viewer_menu_actions(self):
            self.synced += 1

        def _on_image_roi_set_changed(self):
            pass

        def _set_drawing_tool(self, kind):
            pass

        def _current_array_shape(self):
            return (64, 64)

    return Host()


RECT = ("rectangle", {"x": 4, "y": 6, "width": 10, "height": 8})


class TestClearResync:
    def test_clear_quick_selection_resyncs_menu_state(self, qapp):
        """The clear helper suppresses the canvas signal, so it must perform
        the menu/toolbar resync itself — otherwise selection-gated actions
        stay stale after drop / navigation / reset."""
        canvas = _canvas(qapp)
        host = _host(canvas)
        canvas.set_selection(*RECT)

        host._clear_quick_selection()

        assert canvas.selection() is None
        assert host.synced >= 1, "menu state never resynchronised after clear"

    def test_escape_clears_committed_selection_and_emits(self, qapp):
        """Escape on the canvas clears via the signalling path, so the viewer
        handler (status + resync) runs."""
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent

        canvas = _canvas(qapp)
        cleared = []
        canvas.selection_cleared.connect(lambda: cleared.append(True))
        canvas.set_selection(*RECT)

        canvas.keyPressEvent(
            QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))

        assert canvas.selection() is None
        assert cleared == [True]

    def test_set_selection_replaces_previous(self, qapp):
        canvas = _canvas(qapp)
        canvas.set_selection(*RECT)
        canvas.set_selection("ellipse", {"cx": 30, "cy": 30, "rx": 6, "ry": 4})
        sel = canvas.selection()
        assert sel["kind"] == "ellipse"


class TestDisplayOpTransforms:
    def test_flip_carries_selection_to_mirrored_position(self, qapp):
        canvas = _canvas(qapp)
        host = _host(canvas)
        canvas.set_selection(*RECT)

        host._transform_quick_selection_for_display_op("flip_horizontal", {})

        sel = canvas.selection()
        assert sel is not None, "flip must carry the selection, not drop it"
        g = sel["geometry"]
        assert g["x"] == pytest.approx(64 - RECT[1]["x"] - RECT[1]["width"])
        assert g["y"] == pytest.approx(RECT[1]["y"])

    def test_invalidating_op_drops_selection_and_resyncs(self, qapp):
        """rotate_arbitrary invalidates the geometry: the selection must be
        dropped, the user told, and the menu state resynchronised."""
        canvas = _canvas(qapp)
        host = _host(canvas)
        canvas.set_selection(*RECT)
        host.synced = 0

        host._transform_quick_selection_for_display_op(
            "rotate_arbitrary", {"angle_degrees": 17.0})

        assert canvas.selection() is None
        assert host.synced >= 1, (
            "selection dropped by a display op but menu state never resynced"
        )
        assert any("rotate_arbitrary" in s for s in host.statuses)

    def test_transform_without_selection_is_a_no_op(self, qapp):
        canvas = _canvas(qapp)
        host = _host(canvas)
        host._transform_quick_selection_for_display_op("flip_horizontal", {})
        assert canvas.selection() is None


class TestPromotion:
    @pytest.mark.parametrize("kind,geometry", [
        ("rectangle", {"x": 4, "y": 6, "width": 10, "height": 8}),
        ("ellipse", {"cx": 20, "cy": 22, "rx": 8, "ry": 5}),
        ("polygon", {"vertices": [[5, 5], [25, 8], [18, 24]]}),
        ("freehand", {"vertices": [[5, 5], [25, 8], [18, 24], [6, 20]]}),
    ])
    def test_promote_each_area_kind(self, qapp, kind, geometry):
        """Promotion must create a managed ROI with the selection's exact
        kind and geometry, clear the selection, and resync the menus."""
        canvas = _canvas(qapp)
        host = _host(canvas)
        canvas.set_selection(kind, geometry)
        host.synced = 0

        host._promote_selection_to_roi()

        assert canvas.selection() is None, "selection survived promotion"
        assert len(host._image_roi_set.rois) == 1
        roi = host._image_roi_set.rois[0]
        assert roi.kind == kind
        assert roi.geometry == geometry
        assert host._image_roi_set.active_roi_id == roi.id
        assert host.synced >= 1

    def test_promote_without_selection_is_safe(self, qapp):
        canvas = _canvas(qapp)
        host = _host(canvas)
        host._promote_selection_to_roi()
        assert host._image_roi_set.rois == []
        assert any("No selection to promote" in s for s in host.statuses)

    def test_promoted_roi_is_not_retargeted_by_selection_changes(self, qapp):
        """After promotion, drawing a new quick selection must not touch the
        managed ROI — they are independent objects."""
        canvas = _canvas(qapp)
        host = _host(canvas)
        canvas.set_selection(*RECT)
        host._promote_selection_to_roi()
        roi_geom_before = dict(host._image_roi_set.rois[0].geometry)

        canvas.set_selection("rectangle",
                             {"x": 40, "y": 40, "width": 6, "height": 6})
        host._clear_quick_selection()

        assert host._image_roi_set.rois[0].geometry == roi_geom_before
