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


def _dialog(monkeypatch, *, entry_path: Path | None = None, patch_saves: bool = True):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES
    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self, **kw: None)
    if patch_saves:
        monkeypatch.setattr(ImageViewerDialog, "_save_image_roi_set", lambda self: None)
        monkeypatch.setattr(ImageViewerDialog, "_save_image_mask_set", lambda self: None)
    path = entry_path or Path("/tmp/example.sxm")
    entry = SxmFile(path=path, stem=path.stem, Nx=16, Ny=16)
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

        dlg._on_undo_processing()
        assert dlg._processing == {}
        restored = dlg._active_quick_selection()
        assert restored == {
            "kind": "rectangle",
            "geometry": {"x": 2.0, "y": 3.0, "width": 8.0, "height": 6.0},
        }
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


def test_crop_undo_redo_restores_processing_rois_and_masks(qapp, monkeypatch):
    from probeflow.core.mask import ImageMask, MaskSet
    from probeflow.core.roi import ROI, ROISet

    dlg = _dialog(monkeypatch)
    try:
        roi_set = ROISet(image_id="scan")
        target = ROI.new("rectangle", {"x": 4, "y": 4, "width": 6, "height": 6})
        outside = ROI.new("point", {"x": 1, "y": 1})
        roi_set.add(target)
        roi_set.add(outside)
        dlg._image_roi_set = roi_set
        dlg._zoom_lbl.set_roi_set(roi_set)

        mask_data = np.zeros((16, 16), dtype=bool)
        mask_data[5:8, 5:8] = True
        mask_set = MaskSet(image_id="scan")
        mask = ImageMask.new(mask_data, name="inside")
        mask_set.add(mask)
        mask_set.set_active(mask.id)
        dlg._image_mask_set = mask_set

        dlg._on_crop_to_roi(target.id)
        assert dlg._image_roi_set.get(outside.id) is None
        assert dlg._image_mask_set.get(mask.id).shape == (6, 6)
        assert dlg._proc_undo_ctrl.can_undo

        dlg._on_undo_processing()
        assert dlg._processing == {}
        restored_target = dlg._image_roi_set.get(target.id)
        assert restored_target.geometry == target.geometry
        assert dlg._image_roi_set.get(outside.id) is not None
        restored_mask = dlg._image_mask_set.get(mask.id)
        assert restored_mask.shape == (16, 16)
        assert np.array_equal(restored_mask.data, mask_data)

        dlg._on_redo_processing()
        assert [op["op"] for op in dlg._processing["geometric_ops"]] == ["crop"]
        assert dlg._image_roi_set.get(outside.id) is None
        assert dlg._image_mask_set.get(mask.id).shape == (6, 6)
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_geometric_processing_never_rewrites_raw_sidecars(qapp, monkeypatch, tmp_path):
    from probeflow.core.mask import ImageMask, MaskSet
    from probeflow.core.roi import ROI, ROISet
    from probeflow.io.mask_sidecar import load_mask_set_sidecar, save_mask_set_sidecar
    from probeflow.io.roi_sidecar import load_roi_set_sidecar, save_roi_set_sidecar

    scan_path = tmp_path / "scan.sxm"
    scan_path.touch()
    dlg = _dialog(
        monkeypatch, entry_path=scan_path, patch_saves=False,
    )
    try:
        roi_set = ROISet(image_id=str(scan_path))
        target = ROI.new("rectangle", {"x": 4, "y": 4, "width": 6, "height": 6})
        outside = ROI.new("point", {"x": 1, "y": 1})
        roi_set.add(target)
        roi_set.add(outside)
        mask_data = np.zeros((16, 16), dtype=bool)
        mask_data[5:8, 5:8] = True
        mask_set = MaskSet(image_id=str(scan_path))
        mask = ImageMask.new(mask_data, name="raw-mask")
        mask_set.add(mask)
        save_roi_set_sidecar(roi_set, scan_path)
        save_mask_set_sidecar(mask_set, scan_path)

        dlg._image_roi_set = ROISet.from_dict(roi_set.to_dict())
        dlg._image_mask_set = MaskSet.from_dict(mask_set.to_dict())
        dlg._raw_image_roi_payload = roi_set.to_dict()
        dlg._raw_image_mask_payload = mask_set.to_dict()
        dlg._zoom_lbl.set_roi_set(dlg._image_roi_set)

        dlg._on_crop_to_roi(target.id)
        assert dlg._image_roi_set.get(outside.id) is None
        assert dlg._image_mask_set.get(mask.id).shape == (6, 6)

        persisted_rois, _ = load_roi_set_sidecar(scan_path)
        persisted_masks, _ = load_mask_set_sidecar(scan_path)
        assert persisted_rois.get(outside.id) is not None
        assert persisted_rois.get(target.id).geometry == target.geometry
        assert persisted_masks.get(mask.id).shape == (16, 16)
        assert np.array_equal(persisted_masks.get(mask.id).data, mask_data)

        # A deliberate edit in the processed frame remains available for this
        # session but is not allowed to poison the raw-coordinate sidecar.
        session_roi = ROI.new("point", {"x": 2, "y": 2}, name="session-only")
        dlg._image_roi_set.add(session_roi)
        dlg._on_image_roi_set_changed()
        persisted_rois, _ = load_roi_set_sidecar(scan_path)
        assert persisted_rois.get(session_roi.id) is None
        assert dlg._processed_roi_edits_pending is True
        assert "session-only" in dlg._roi_status_lbl.text()

        dlg._on_reset_processing()
        assert dlg._processing == {}
        assert dlg._image_roi_set.get(outside.id) is not None
        assert dlg._image_roi_set.get(session_roi.id) is None
        assert dlg._image_mask_set.get(mask.id).shape == (16, 16)
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_raw_sidecars_project_into_restored_processing_frame(qapp, monkeypatch):
    from probeflow.core.mask import ImageMask, MaskSet
    from probeflow.core.roi import ROI, ROISet

    dlg = _dialog(monkeypatch)
    try:
        roi_set = ROISet(image_id="scan")
        target = ROI.new("rectangle", {"x": 4, "y": 4, "width": 6, "height": 6})
        outside = ROI.new("point", {"x": 1, "y": 1})
        roi_set.add(target)
        roi_set.add(outside)
        mask_data = np.zeros((16, 16), dtype=bool)
        mask_data[5:8, 5:8] = True
        mask_set = MaskSet(image_id="scan")
        mask = ImageMask.new(mask_data, name="raw-mask")
        mask_set.add(mask)
        dlg._raw_image_roi_payload = roi_set.to_dict()
        dlg._raw_image_mask_payload = mask_set.to_dict()
        dlg._processing = {"geometric_ops": [{
            "op": "crop",
            "params": {"x0": 4, "y0": 4, "x1": 9, "y1": 9},
        }]}

        dlg._project_raw_overlays_to_processing()
        projected = dlg._image_roi_set.get(target.id)
        assert projected.geometry["x"] == pytest.approx(0.0)
        assert projected.geometry["y"] == pytest.approx(0.0)
        assert dlg._image_roi_set.get(outside.id) is None
        assert dlg._image_mask_set.get(mask.id).shape == (6, 6)
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_apply_undistort_reprojects_overlays_and_undo_restores_them(
    qapp, monkeypatch,
):
    from probeflow.core.mask import ImageMask, MaskSet
    from probeflow.core.roi import ROI, ROISet

    dlg = _dialog(monkeypatch)
    try:
        roi_set = ROISet(image_id="scan")
        roi = ROI.new("point", {"x": 4, "y": 5})
        roi_set.add(roi)
        mask_set = MaskSet(image_id="scan")
        mask = ImageMask.new(np.eye(16, dtype=bool), name="raw-mask")
        mask_set.add(mask)
        dlg._raw_image_roi_payload = roi_set.to_dict()
        dlg._raw_image_mask_payload = mask_set.to_dict()
        dlg._image_roi_set = ROISet.from_dict(roi_set.to_dict())
        dlg._image_mask_set = MaskSet.from_dict(mask_set.to_dict())

        dlg._undistort_shear_spin.setValue(2.0)
        dlg._on_apply_processing()

        assert dlg._overlays_are_in_processed_frame()
        assert dlg._image_roi_set.rois == []
        assert dlg._image_mask_set.masks == []

        dlg._on_undo_processing()
        assert dlg._processing == {}
        assert dlg._image_roi_set.get(roi.id) is not None
        assert dlg._image_mask_set.get(mask.id) is not None

        dlg._on_redo_processing()
        assert dlg._overlays_are_in_processed_frame()
        assert dlg._image_roi_set.rois == []
        assert dlg._image_mask_set.masks == []

        # Removing the undistortion through Apply projects fresh copies from
        # the canonical raw payloads instead of leaving the invalidated view.
        dlg._undistort_shear_spin.setValue(0.0)
        dlg._on_apply_processing()
        assert not dlg._overlays_are_in_processed_frame()
        assert dlg._image_roi_set.get(roi.id) is not None
        assert dlg._image_mask_set.get(mask.id) is not None
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_raw_frame_overlay_edits_still_persist(qapp, monkeypatch, tmp_path):
    from probeflow.core.mask import ImageMask, MaskSet
    from probeflow.core.roi import ROI, ROISet
    from probeflow.io.mask_sidecar import load_mask_set_sidecar
    from probeflow.io.roi_sidecar import load_roi_set_sidecar

    scan_path = tmp_path / "scan.sxm"
    scan_path.touch()
    dlg = _dialog(monkeypatch, entry_path=scan_path, patch_saves=False)
    try:
        roi_set = ROISet(image_id=str(scan_path))
        roi = ROI.new("point", {"x": 3, "y": 7}, name="persistent-point")
        roi_set.add(roi)
        dlg._image_roi_set = roi_set
        dlg._on_image_roi_set_changed()

        mask_set = MaskSet(image_id=str(scan_path))
        mask = ImageMask.new(np.eye(16, dtype=bool), name="persistent-mask")
        mask_set.add(mask)
        dlg._image_mask_set = mask_set
        dlg._on_image_mask_set_changed()

        persisted_rois, roi_path = load_roi_set_sidecar(scan_path)
        persisted_masks, mask_path = load_mask_set_sidecar(scan_path)
        assert roi_path.name == "scan.rois.json"
        assert mask_path.name == "scan.masks.json"
        assert persisted_rois.get(roi.id) is not None
        assert persisted_masks.get(mask.id) is not None
        assert dlg._raw_image_roi_payload == roi_set.to_dict()
        assert dlg._raw_image_mask_payload == mask_set.to_dict()
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_undo_snapshot_never_restores_overlays_into_another_scan(qapp, monkeypatch):
    from probeflow.core.roi import ROI, ROISet
    from probeflow.gui import SxmFile

    dlg = _dialog(monkeypatch)
    try:
        first_set = ROISet(image_id="first")
        first_set.add(ROI.new("point", {"x": 2, "y": 3}, name="first-point"))
        dlg._image_roi_set = first_set
        snapshot = dlg._capture_proc_undo_snapshot()

        second = SxmFile(
            path=Path("/tmp/second.sxm"), stem="second", Nx=16, Ny=16
        )
        dlg._entries.append(second)
        dlg._idx = 1
        second_set = ROISet(image_id="second")
        second_set.add(ROI.new("point", {"x": 9, "y": 10}, name="second-point"))
        dlg._image_roi_set = second_set
        dlg._zoom_lbl.set_roi_set(second_set)

        dlg._restore_processing_state(snapshot)
        assert dlg._image_roi_set.image_id == "second"
        assert dlg._image_roi_set.rois[0].name == "second-point"
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

        dlg._on_undo_processing()
        assert "repair_ops" not in dlg._processing
        assert [op["op"] for op in dlg._processing["geometric_ops"]] == [
            "flip_horizontal",
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
