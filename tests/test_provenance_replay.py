"""Replay of exported provenance without canonical sidecars (Qt-free).

2026-07-06 review of the carried-forward ``processed_export.py`` /
replay-without-sidecars item.  The ``.probeflow.json`` ExportRecord is the
declared "reliable current record" (see ``write_provenance_sidecars``), but
its ``processing_history.steps`` use ``operation_id``/``parameters`` keys —
``processing_state_from_history`` used to silently return an EMPTY state for
them, so a replay reproduced the raw image while claiming to be processed.

These tests pin:
  * record-format steps reconstruct the same canonical state as legacy
    entries, with bookkeeping steps (file_load / export_* / dat_to_sxm)
    skipped;
  * a full round trip — export via the viewer's Qt-free helpers, then
    rebuild the processing state AND the ROI set from the exported
    ``.probeflow.json`` alone (no ``.rois.json``, ``.provenance.json``
    deleted) and replay to the identical array.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from probeflow.processing.history import processing_state_from_history

TEST_DATA = Path(__file__).parent.parent / "test_data"
SCAN_FIXTURE = TEST_DATA / "createc_scan_terrace_109nm.dat"


class TestRecordFormatReconstruction:
    def test_record_steps_reconstruct_ops_and_params(self):
        steps = [
            {"operation_id": "file_load", "parameters": {"source_path": "x.dat"},
             "step_id": "step-0001", "operation_name": "Loaded"},
            {"operation_id": "plane_bg", "parameters": {"order": 1},
             "step_id": "step-0002", "operation_name": "Background subtraction"},
            {"operation_id": "roi",
             "parameters": {"step": {"op": "smooth", "params": {"sigma_px": 2.0}},
                            "roi_id": "abc"},
             "step_id": "step-0003", "operation_name": "ROI-scoped filter"},
            {"operation_id": "export_csv", "parameters": {},
             "step_id": "step-0004", "operation_name": "Export: CSV"},
        ]
        state = processing_state_from_history(steps)
        assert [s.op for s in state.steps] == ["plane_bg", "roi"]
        assert state.steps[0].params == {"order": 1}
        assert state.steps[1].params["step"] == {"op": "smooth", "params": {"sigma_px": 2.0}}

    def test_dat_to_sxm_conversion_step_is_skipped(self):
        state = processing_state_from_history([
            {"operation_id": "dat_to_sxm", "parameters": {"export_format": "sxm"}},
            {"operation_id": "align_rows", "parameters": {"method": "median"}},
        ])
        assert [s.op for s in state.steps] == ["align_rows"]

    def test_legacy_entries_unchanged(self):
        state = processing_state_from_history([
            {"op": "plane_bg", "params": {"order": 2}, "timestamp": "t"},
            {"op": "smooth", "sigma_px": 1.5},  # legacy params-scraped form
        ])
        assert [s.op for s in state.steps] == ["plane_bg", "smooth"]
        assert state.steps[0].params == {"order": 2}
        assert state.steps[1].params == {"sigma_px": 1.5}

    def test_junk_entries_tolerated(self):
        state = processing_state_from_history([
            None, "text", {}, {"operation_id": ""}, {"parameters": {"x": 1}},
        ])
        assert state.steps == []


class TestReplayFromProbeflowSidecarOnly:
    """End-to-end: state + ROIs recovered from the .probeflow.json alone."""

    @pytest.fixture
    def exported(self, tmp_path):
        from probeflow.core.roi import ROI, ROISet
        from probeflow.core.scan_loader import load_scan
        from probeflow.gui.viewer.processed_export import (
            build_processed_scan_for_export,
            save_processed_image,
        )
        from probeflow.processing.gui_adapter import processing_state_from_gui
        from probeflow.processing.state import (
            apply_processing_state_with_calibration,
        )

        scan_path = tmp_path / SCAN_FIXTURE.name
        shutil.copy(SCAN_FIXTURE, scan_path)

        roi_set = ROISet(image_id=str(scan_path))
        roi = ROI.new(
            "rectangle",
            {"x": 10.0, "y": 10.0, "width": 60.0, "height": 40.0},
            name="patch",
        )
        roi_set.add(roi)

        gui_state = {
            "plane_bg": {"order": 1},
            "processing_scope": "roi",
            "processing_roi_id": roi.id,
            "smooth_sigma": 2.0,
        }
        state = processing_state_from_gui(gui_state)
        assert [s.op for s in state.steps] == ["plane_bg", "roi"]

        scan = load_scan(scan_path)
        raw = np.asarray(scan.planes[0], dtype=np.float64).copy()
        arr, rng = apply_processing_state_with_calibration(
            scan.planes[0], state, roi_set,
            mask_set=None, scan_range_m=scan.scan_range_m,
        )
        exp_scan, idx = build_processed_scan_for_export(
            scan_path, 0, arr, gui_state, scan_range_m=rng,
        )
        out = tmp_path / "processed.csv"
        msg = save_processed_image(
            exp_scan, idx, out,
            display_settings={"colormap": "gray"},
            roi_set=roi_set, mask_set=None,
        )
        assert msg.startswith("Saved"), msg

        # Only the scan and the .probeflow.json remain: no canonical
        # .rois.json was ever written, and the legacy sidecar is removed.
        legacy = tmp_path / "processed.provenance.json"
        if legacy.exists():
            legacy.unlink()
        assert not (tmp_path / f"{scan_path.stem}.rois.json").exists()
        sidecar = tmp_path / "processed.probeflow.json"
        assert sidecar.exists()
        return scan_path, sidecar, raw, arr

    def test_replay_reproduces_exported_array(self, exported):
        from probeflow.core.processing_state import ProcessingState
        from probeflow.io.roi_sidecar import load_roi_set_sidecar
        from probeflow.processing.state import (
            apply_processing_state_with_calibration,
        )

        scan_path, sidecar, raw, expected = exported

        record = json.loads(sidecar.read_text(encoding="utf-8"))
        state_dict = processing_state_from_history(
            record["processing_history"]["steps"]
        ).to_dict()
        state = ProcessingState.from_dict(state_dict)
        assert [s["op"] for s in state_dict["steps"]] == ["plane_bg", "roi"]

        roi_set, used = load_roi_set_sidecar(scan_path, sidecar=sidecar)
        assert used == sidecar
        assert [r.name for r in roi_set.rois] == ["patch"]

        replayed, _rng = apply_processing_state_with_calibration(
            raw, state, roi_set, mask_set=None, scan_range_m=None,
        )
        np.testing.assert_array_equal(
            replayed, expected,
            err_msg=".probeflow.json-only replay does not reproduce the export",
        )

    def test_pre_fix_behaviour_would_have_replayed_raw(self, exported):
        # The failure mode this guards: an empty state replays the raw
        # image — assert the processed array actually differs from raw so
        # the equality above is discriminating.
        _scan_path, _sidecar, raw, expected = exported
        assert not np.array_equal(raw, expected)
