"""Lifetime tests for QGraphicsItem wrappers owned by ImageCanvas.

QGraphicsItem is not a QObject: when the canvas's scene dies C++-side,
Shiboken never invalidates the Python wrappers for its items. A wrapper that
outlives its C++ item keeps a dangling pointer binding in Shiboken's cache,
and a later C++ allocation reusing the address resurrects the stale wrapper
as the wrong type — the intermittent CI failure
"'QGraphicsItemGroup' object has no attribute 'connect'/'triggered'" inside
QMenu.addAction (see tests/conftest.py drain notes).

These tests pin the fix: the canvas's ``destroyed`` hook drops every item
reference while the C++ items are still alive, so the wrappers deallocate
and unregister their bindings deterministically — even when the canvas
*wrapper* itself is still referenced from Python (the hazardous case).
"""

from __future__ import annotations

import gc
import os
import weakref

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


def _populated_canvas(qapp):
    """Canvas with a ROI group, a quick selection, and zero markers."""
    from PySide6.QtGui import QPixmap

    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui.image_canvas import ImageCanvas

    canvas = ImageCanvas()
    pm = QPixmap(64, 64)
    pm.fill()
    canvas._view_scale_mode = "one_to_one"
    canvas.set_source(pm, reset_zoom=True)

    rs = ROISet(image_id="img")
    rs.add(ROI.new("rectangle", {"x": 4, "y": 4, "width": 12, "height": 10}))
    canvas.set_roi_set(rs)
    canvas.set_selection("rectangle", {"x": 20, "y": 20, "width": 8, "height": 8})
    canvas.set_zero_markers([{"frac_x": 0.1, "frac_y": 0.1, "label": "z"}])
    return canvas


def test_cpp_destruction_releases_item_wrappers_while_canvas_referenced(qapp):
    """Destroying the C++ view (parent teardown / processed deleteLater)
    while the Python canvas wrapper is still referenced must release every
    QGraphicsItem wrapper — they must not survive as dangling bindings."""
    import shiboken6

    canvas = _populated_canvas(qapp)
    assert canvas._roi_items, "fixture expected ROI items"

    # Comprehensions only: a plain for-loop would leave its loop variable
    # bound in this frame, keeping the last item alive and failing the test
    # by its own construction.
    item_refs = [weakref.ref(group) for group in canvas._roi_items.values()]
    item_refs += [weakref.ref(m) for m in canvas._zero_marker_items]
    item_refs.append(weakref.ref(canvas._selection_item))
    item_refs.append(weakref.ref(canvas._pixmap_item))

    # The hazardous scenario: `canvas` (the wrapper) stays referenced while
    # the C++ object dies, exactly like a deleteLater()'d dialog that is
    # still held somewhere in Python.
    shiboken6.delete(canvas)
    gc.collect()

    survivors = [r for r in item_refs if r() is not None]
    assert not survivors, (
        f"{len(survivors)} QGraphicsItem wrapper(s) outlived their C++ "
        "items — dangling Shiboken bindings, the wrapper-recycling flake"
    )


def test_release_hook_does_not_keep_canvas_alive(qapp):
    """The destroyed hook captures the attribute dict, not self: dropping
    the last Python reference must still collect the canvas."""
    canvas = _populated_canvas(qapp)
    canvas_ref = weakref.ref(canvas)

    del canvas
    gc.collect()

    assert canvas_ref() is None, "destroyed hook leaked the canvas wrapper"


def test_normal_python_teardown_still_works(qapp):
    """Plain Python-side destruction (test-style: drop the reference, let
    gc run) must not be disturbed by the release hook."""
    canvas = _populated_canvas(qapp)
    item_refs = [weakref.ref(g) for g in canvas._roi_items.values()]

    del canvas
    gc.collect()

    assert all(r() is None for r in item_refs)
