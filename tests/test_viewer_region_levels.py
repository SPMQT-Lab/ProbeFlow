"""Tests for the viewer-side per-region display-range assembly.

Exercises ImageViewerDisplayMixin._region_levels_for_render and _target_drs
without constructing the full ImageViewerDialog: a tiny QObject stub supplies
the handful of attributes the methods read.
"""

from __future__ import annotations

import os

import numpy as np
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


def _make_stub():
    from PySide6.QtCore import QObject
    from probeflow.gui.viewer.image_viewer_display_mixin import ImageViewerDisplayMixin

    class _Stub(QObject, ImageViewerDisplayMixin):
        def __init__(self):
            super().__init__()
            self._clip_low = 1.0
            self._clip_high = 99.0
            self._display_scope = "global"
            self._region_drs = {}
            self._display_arr = None
            self._image_roi_set = None

        def _refresh_display_range(self):  # signal sink; no-op for tests
            pass

    return _Stub()


def _roi_set_with_rect():
    from probeflow.core.roi import ROI, ROISet
    rs = ROISet(image_id="img")
    rect = ROI.new("rectangle", {"x": 0.0, "y": 0.0, "width": 2.0, "height": 4.0})
    rs.add(rect)
    rs.set_active(rect.id)
    return rs, rect


def test_region_levels_includes_only_manual_regions(qapp):
    stub = _make_stub()
    stub._display_arr = np.linspace(0.0, 1.0, 16).reshape(4, 4).astype(float)
    rs, rect = _roi_set_with_rect()
    stub._image_roi_set = rs

    # One manual region (rect) + one untouched percentile region (other id).
    manual = stub._region_drs_for(rect.id)
    manual.set_manual(0.3, 0.7)
    # A percentile-mode region for a non-existent ROI must be ignored.
    stub._region_drs_for("ghost-id")

    levels = stub._region_levels_for_render()
    assert levels is not None
    assert len(levels) == 1
    mask, vmin, vmax = levels[0]
    assert (vmin, vmax) == (0.3, 0.7)
    expected_mask = rect.to_mask((4, 4))
    assert np.array_equal(mask, expected_mask)


def test_region_levels_none_when_no_manual_regions(qapp):
    stub = _make_stub()
    stub._display_arr = np.zeros((4, 4))
    rs, rect = _roi_set_with_rect()
    stub._image_roi_set = rs
    stub._region_drs_for(rect.id)  # left in percentile mode
    assert stub._region_levels_for_render() is None


def test_target_drs_follows_scope(qapp):
    stub = _make_stub()
    from probeflow.gui.viewer.display_range import DisplayRangeController
    stub._drs = DisplayRangeController(parent=stub)
    rs, rect = _roi_set_with_rect()
    stub._image_roi_set = rs

    # Global scope -> global controller.
    stub._display_scope = "global"
    assert stub._target_drs() is stub._drs

    # ROI scope with an active area ROI -> that ROI's own controller.
    stub._display_scope = "roi"
    target = stub._target_drs()
    assert target is stub._region_drs[rect.id]
    assert target is not stub._drs
