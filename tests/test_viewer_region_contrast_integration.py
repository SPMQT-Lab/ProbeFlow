"""End-to-end smoke test for per-region contrast + hide-overlay in the viewer.

Builds a real ImageViewerDialog (no on-disk data) and drives the new controls:
switching contrast scope to a per-ROI range, tuning it, and hiding overlays.
Verifies the wiring holds together and produces region_levels for rendering.
"""

from __future__ import annotations

import os
from pathlib import Path

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


def _dialog(monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self, **kw: None)
    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    return ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])


def test_scope_to_roi_tunes_region_and_builds_region_levels(qapp, monkeypatch):
    from probeflow.core.roi import ROI, ROISet

    dlg = _dialog(monkeypatch)
    try:
        dlg._display_arr = np.linspace(0.0, 1.0, 64).reshape(8, 8).astype(float)
        rs = ROISet(image_id="example")
        rect = ROI.new("rectangle", {"x": 0.0, "y": 0.0, "width": 4.0, "height": 8.0})
        rs.add(rect)
        rs.set_active(rect.id)
        dlg._image_roi_set = rs

        # Global scope -> sliders edit the global controller, no region levels.
        assert dlg._target_drs() is dlg._drs
        assert dlg._region_levels_for_render() is None

        # Switch to per-ROI scope via the combo (index 1 == "Active ROI").
        dlg._display_scope_cb.setCurrentIndex(1)
        assert dlg._display_scope == "roi"
        target = dlg._target_drs()
        assert target is dlg._region_drs[rect.id]

        # Tune the region's range; it should now contribute region_levels.
        target.set_manual(0.2, 0.6)
        levels = dlg._region_levels_for_render()
        assert levels is not None and len(levels) == 1
        mask, vmin, vmax = levels[0]
        assert (vmin, vmax) == (0.2, 0.6)
        assert np.array_equal(mask, rect.to_mask((8, 8)))
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_hide_overlay_checkbox_toggles_canvas_visibility(qapp, monkeypatch):
    from probeflow.core.roi import ROI, ROISet

    dlg = _dialog(monkeypatch)
    try:
        rs = ROISet(image_id="example")
        rect = ROI.new("rectangle", {"x": 1.0, "y": 1.0, "width": 4.0, "height": 4.0})
        rs.add(rect)
        rs.set_active(rect.id)
        dlg._zoom_lbl.set_roi_set(rs)
        dlg._image_roi_set = rs

        item = dlg._zoom_lbl._roi_items[rect.id]
        assert item.isVisible()

        dlg._hide_rois_cb.setChecked(True)
        assert dlg._rois_hidden is True
        assert not item.isVisible()

        # Hidden state survives an ROI-set rebuild.
        dlg._zoom_lbl.set_roi_set(rs)
        assert not dlg._zoom_lbl._roi_items[rect.id].isVisible()

        dlg._hide_rois_cb.setChecked(False)
        assert dlg._zoom_lbl._roi_items[rect.id].isVisible()
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()
