"""GUI wiring tests for the Tier-1 ops: median panel, crop, remove spots."""

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
    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=16, Ny=16)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._raw_arr = np.zeros((16, 16), dtype=float)
    return dlg


# ── median filter in the processing panel ─────────────────────────────────────

def test_processing_panel_median_state_roundtrip(qapp):
    from probeflow.gui.processing import ProcessingControlPanel

    panel = ProcessingControlPanel("viewer_full")
    try:
        panel._smooth_combo.setCurrentIndex(2)  # Median
        panel._median_size_sl.setValue(5)
        state = panel.state()
        assert state["median_size"] == 5
        assert state["smooth_sigma"] is None

        fresh = ProcessingControlPanel("viewer_full")
        try:
            fresh.set_state({"median_size": 7})
            assert fresh._smooth_combo.currentIndex() == 2
            assert fresh.state()["median_size"] == 7
            # Gaussian still restores as before.
            fresh.set_state({"smooth_sigma": 1.5})
            assert fresh._smooth_combo.currentIndex() == 1
            assert fresh.state()["median_size"] is None
        finally:
            fresh.deleteLater()
    finally:
        panel.deleteLater()
        qapp.processEvents()


# ── crop ──────────────────────────────────────────────────────────────────────

def test_crop_to_selection_records_op_and_consumes_selection(qapp, monkeypatch):
    dlg = _dialog(monkeypatch)
    try:
        dlg._zoom_lbl.set_selection(
            "rectangle", {"x": 2.0, "y": 3.0, "width": 8.0, "height": 6.0})
        dlg._on_crop_to_selection()
        ops = dlg._processing.get("geometric_ops") or []
        assert len(ops) == 1
        assert ops[0]["op"] == "crop"
        p = ops[0]["params"]
        assert (p["x0"], p["y0"]) == (2, 3)
        assert (p["x1"], p["y1"]) == (9, 8)  # inclusive bounding box
        # The selection was consumed.
        assert dlg._active_quick_selection() is None
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_crop_to_roi_translates_surviving_rois(qapp, monkeypatch):
    from probeflow.core.roi import ROI, ROISet

    dlg = _dialog(monkeypatch)
    try:
        roi_set = ROISet(image_id="scan")
        target = ROI.new("rectangle", {"x": 4, "y": 4, "width": 6, "height": 6})
        other = ROI.new("rectangle", {"x": 0, "y": 0, "width": 2, "height": 2})
        roi_set.add(target)
        roi_set.add(other)
        dlg._image_roi_set = roi_set
        dlg._zoom_lbl.set_roi_set(roi_set)

        dlg._on_crop_to_roi(target.id)

        ops = dlg._processing.get("geometric_ops") or []
        assert [op["op"] for op in ops] == ["crop"]
        # The target ROI now sits at the origin of the cropped frame.
        moved = dlg._image_roi_set.get(target.id)
        assert moved.geometry["x"] == pytest.approx(0.0)
        assert moved.geometry["y"] == pytest.approx(0.0)
        # The out-of-crop ROI was dropped.
        assert dlg._image_roi_set.get(other.id) is None
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_crop_without_selection_is_a_noop_with_hint(qapp, monkeypatch):
    dlg = _dialog(monkeypatch)
    try:
        dlg._on_crop_to_selection()
        assert not (dlg._processing.get("geometric_ops") or [])
        assert "selection" in dlg._status_lbl.text().lower()
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


# ── remove spots ──────────────────────────────────────────────────────────────

def test_repair_under_roi_commits_frozen_geometry_with_stamp(qapp, monkeypatch):
    from probeflow.core.roi import ROI, ROISet

    dlg = _dialog(monkeypatch)
    try:
        roi_set = ROISet(image_id="scan")
        roi = ROI.new("rectangle", {"x": 5, "y": 5, "width": 4, "height": 4})
        roi_set.add(roi)
        dlg._image_roi_set = roi_set
        # A prior geometric op: the repair must be stamped after it.
        dlg._processing["geometric_ops"] = [{"op": "flip_horizontal", "params": {}}]

        dlg._commit_repair_under_roi(roi.id)

        repairs = dlg._processing.get("repair_ops") or []
        assert len(repairs) == 1
        assert repairs[0]["frozen_geometry"]["kind"] == "rectangle"
        assert repairs[0]["after_geometric_ops"] == 1

        # The committed entry converts to a replayable step.
        from probeflow.processing.gui_adapter import processing_state_from_gui
        state = processing_state_from_gui(dlg._processing)
        assert [s.op for s in state.steps] == [
            "flip_horizontal", "interpolate_masked",
        ]
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_repair_under_mask_commits_frozen_raster(qapp, monkeypatch):
    from probeflow.core.mask import ImageMask, MaskSet

    dlg = _dialog(monkeypatch)
    try:
        data = np.zeros((16, 16), dtype=bool)
        data[6:9, 6:9] = True
        mask_set = MaskSet(image_id="scan")
        mask = ImageMask.new(data, name="defect")
        mask_set.add(mask)
        dlg._image_mask_set = mask_set

        dlg._commit_repair_under_mask(mask.id)

        repairs = dlg._processing.get("repair_ops") or []
        assert len(repairs) == 1
        frozen = repairs[0]["frozen_mask"]
        assert frozen["shape"] == [16, 16]

        # Round-trip: the frozen raster replays to a working repair.
        from probeflow.processing.gui_adapter import processing_state_from_gui
        from probeflow.processing.state import apply_processing_state

        state = processing_state_from_gui(dlg._processing)
        yy, xx = np.mgrid[0:16, 0:16]
        arr = 1.0 * xx
        expected = arr.copy()
        arr[data] = 500.0
        out = apply_processing_state(arr, state)
        assert np.allclose(out, expected, atol=1e-8)
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_viewer_commands_include_crop():
    from probeflow.gui.viewer.shortcuts import viewer_command

    cmd = viewer_command("image.crop_selection")
    assert cmd.label == "Crop to selection"
