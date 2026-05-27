"""Tests for processing-history wiring in the CLI layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from probeflow.cli import (
    _Op,
    _apply_to_plane,
    _load_named_roi,
    _op_align_rows,
    _op_fft,
    _op_plane_bg,
    _op_remove_bad_lines,
    _op_smooth,
    _op_edge,
    _op_facet_level,
    _record_op,
    main,
)
from probeflow.core.roi import ROI, ROISet
from probeflow.core.scan_model import Scan
from probeflow.io.roi_sidecar import (
    default_roi_sidecar_path,
    load_roi_set_sidecar,
    save_roi_set_sidecar,
)


# ─── _Op class ───────────────────────────────────────────────────────────────

class TestOp:
    def test_callable(self):
        op = _Op("test", {}, lambda a: a * 2)
        arr = np.ones((4, 4))
        result = op(arr)
        assert np.all(result == 2.0)


class TestCliRoiSidecarLookup:
    def test_shared_sidecar_helper_round_trips_gui_rois_sidecar(self, tmp_path):
        scan_path = tmp_path / "scan.sxm"
        roi = ROI.new("point", {"x": 1.0, "y": 2.0}, name="site")
        roi_set = ROISet(image_id=str(scan_path))
        roi_set.add(roi)

        written = save_roi_set_sidecar(roi_set, scan_path)
        loaded, used = load_roi_set_sidecar(scan_path)

        assert written == default_roi_sidecar_path(scan_path)
        assert used == written
        assert loaded.get_by_name("site").id == roi.id

    def test_dotted_filename_does_not_collapse_to_date_prefix(self, tmp_path):
        """Regression for review IO #2 — Createc timestamp-style names.

        Files saved by Createc use ``Aymmdd.HHmmss.dat``.  The old
        ``with_suffix("").with_suffix(".rois.json")`` chain collapsed
        ``A250320.191933.dat`` to ``A250320.rois.json`` — colliding across
        every scan saved on the same date.
        """
        scan_path = tmp_path / "A250320.191933.dat"
        scan_path.write_bytes(b"")

        sidecar = default_roi_sidecar_path(scan_path)
        # Per-scan sidecar, not date-prefix-collapsed:
        assert sidecar.name == "A250320.191933.rois.json"
        # And the second-on-the-same-day file gets a distinct path:
        other = tmp_path / "A250320.191948.dat"
        assert default_roi_sidecar_path(other).name == "A250320.191948.rois.json"
        assert default_roi_sidecar_path(other) != sidecar

    def test_dotted_filename_round_trip(self, tmp_path):
        scan_path = tmp_path / "A250320.191933.dat"
        scan_path.write_bytes(b"")
        roi = ROI.new("point", {"x": 1.0, "y": 2.0}, name="defect")
        roi_set = ROISet(image_id=str(scan_path))
        roi_set.add(roi)

        written = save_roi_set_sidecar(roi_set, scan_path)
        assert written.name == "A250320.191933.rois.json"

        loaded, used = load_roi_set_sidecar(scan_path)
        assert used == written
        assert loaded.get_by_name("defect").id == roi.id

    def test_legacy_buggy_sidecar_still_loadable(self, tmp_path):
        """Sidecars written before the IO #2 fix (at the wrong, collapsed
        path) are still found via the candidate fallback list."""
        from probeflow.io.roi_sidecar import roi_sidecar_candidates
        scan_path = tmp_path / "A250320.191933.dat"
        scan_path.write_bytes(b"")
        # Manually place a sidecar at the LEGACY (buggy) path:
        legacy = tmp_path / "A250320.rois.json"
        roi = ROI.new("point", {"x": 1.0, "y": 2.0}, name="legacy")
        roi_set = ROISet(image_id=str(scan_path))
        roi_set.add(roi)
        legacy.write_text(__import__("json").dumps(roi_set.to_dict()))

        assert legacy in roi_sidecar_candidates(scan_path)
        loaded, used = load_roi_set_sidecar(scan_path)
        assert used == legacy
        assert loaded.get_by_name("legacy").id == roi.id

    def test_load_named_roi_defaults_to_gui_rois_sidecar(self, tmp_path):
        scan_path = tmp_path / "scan.sxm"
        scan_path.write_bytes(b"")
        roi = ROI.new("rectangle", {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0},
                      name="terrace")
        roi_set = ROISet(image_id=str(scan_path))
        roi_set.add(roi)

        import json
        (tmp_path / "scan.rois.json").write_text(
            json.dumps(roi_set.to_dict()),
            encoding="utf-8",
        )

        loaded = _load_named_roi(scan_path, "terrace")

        assert loaded is not None
        assert loaded.id == roi.id

    def test_load_named_roi_falls_back_to_provenance_sidecar(self, tmp_path):
        scan_path = tmp_path / "scan.sxm"
        scan_path.write_bytes(b"")
        roi = ROI.new("rectangle", {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0},
                      name="terrace")
        roi_set = ROISet(image_id=str(scan_path))
        roi_set.add(roi)

        import json
        (tmp_path / "scan.provenance.json").write_text(
            json.dumps({"rois": roi_set.to_dict()}),
            encoding="utf-8",
        )

        loaded = _load_named_roi(scan_path, "terrace")

        assert loaded is not None
        assert loaded.id == roi.id

    def test_load_named_roi_falls_back_to_probeflow_sidecar(self, tmp_path):
        scan_path = tmp_path / "scan.sxm"
        scan_path.write_bytes(b"")
        roi = ROI.new("rectangle", {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0},
                      name="terrace")
        roi_set = ROISet(image_id=str(scan_path))
        roi_set.add(roi)

        import json
        (tmp_path / "scan.probeflow.json").write_text(
            json.dumps({"record_type": "probeflow_export", "rois": roi_set.to_dict()}),
            encoding="utf-8",
        )

        loaded = _load_named_roi(scan_path, "terrace")

        assert loaded is not None
        assert loaded.id == roi.id

    def test_carries_name_and_params(self):
        op = _Op("plane_bg", {"order": 1}, lambda a: a)
        assert op.name == "plane_bg"
        assert op.params == {"order": 1}


# ─── _record_op ──────────────────────────────────────────────────────────────

class TestRecordOp:
    def _minimal_scan(self):
        return Scan(
            planes=[np.zeros((4, 4))],
            plane_names=["Z forward"],
            plane_units=["m"],
            plane_synthetic=[False],
            header={},
            scan_range_m=(1e-8, 1e-8),
            source_path=Path("/fake/file.sxm"),
            source_format="sxm",
        )

    def test_appends_entry(self):
        scan = self._minimal_scan()
        _record_op(scan, "plane_bg", {"order": 1})
        assert len(scan.processing_history) == 1
        entry = scan.processing_history[0]
        assert entry["op"] == "plane_bg"
        assert entry["params"] == {"order": 1}
        assert "timestamp" in entry
        assert [step.op for step in scan.processing_state.steps] == ["plane_bg"]
        assert scan.processing_state.steps[0].params == {"order": 1}

    def test_timestamp_is_iso_string(self):
        from datetime import datetime
        scan = self._minimal_scan()
        _record_op(scan, "smooth", {"sigma_px": 1.5})
        ts = scan.processing_history[0]["timestamp"]
        datetime.fromisoformat(ts)  # raises if not valid ISO

    def test_multiple_entries_accumulate(self):
        scan = self._minimal_scan()
        _record_op(scan, "align_rows", {"method": "median"})
        _record_op(scan, "plane_bg", {"order": 2})
        assert len(scan.processing_history) == 2
        assert scan.processing_history[0]["op"] == "align_rows"
        assert scan.processing_history[1]["op"] == "plane_bg"
        assert [step.op for step in scan.processing_state.steps] == [
            "align_rows",
            "plane_bg",
        ]


# ─── _apply_to_plane with _Op ─────────────────────────────────────────────────

class TestApplyToPlane:
    def test_op_records_history(self, first_sample_dat):
        op = _op_plane_bg(order=1)
        scan = _apply_to_plane(first_sample_dat, 0, op)
        assert len(scan.processing_history) == 1
        assert scan.processing_history[0]["op"] == "plane_bg"
        assert scan.processing_history[0]["params"] == {"order": 1}

    def test_plain_callable_does_not_record(self, first_sample_dat):
        plain = lambda a: a
        scan = _apply_to_plane(first_sample_dat, 0, plain)
        assert scan.processing_history == []


# ─── factory ops ─────────────────────────────────────────────────────────────

class TestOpFactories:
    @pytest.mark.parametrize("op,expected_name,expected_params", [
        (_op_plane_bg(2),                      "plane_bg",       {"order": 2}),
        (_op_align_rows("mean"),               "align_rows",     {"method": "mean"}),
        (_op_remove_bad_lines(3.0),            "remove_bad_lines", {"threshold_mad": 3.0}),
        (_op_facet_level(5.0),                 "facet_level",    {"threshold_deg": 5.0}),
        (_op_smooth(2.0),                      "smooth",         {"sigma_px": 2.0}),
        (_op_edge("log", 1.5, 2.5),            "edge_detect",    {"method": "log", "sigma": 1.5, "sigma2": 2.5}),
        (_op_fft("high_pass", 0.2, "hamming"), "fourier_filter", {"mode": "high_pass", "cutoff": 0.2, "window": "hamming"}),
    ])
    def test_factory_returns_op_with_correct_metadata(self, op, expected_name, expected_params):
        assert isinstance(op, _Op)
        assert op.name == expected_name
        assert op.params == expected_params

    def test_factory_ops_execute_through_canonical_processing_state(self, monkeypatch):
        calls = []

        def fake_apply_processing_state(arr, state):
            calls.append(state.to_dict())
            return arr + 1

        monkeypatch.setattr(
            "probeflow.processing.state.apply_processing_state",
            fake_apply_processing_state,
        )

        arr = np.zeros((4, 4))
        result = _op_remove_bad_lines(3.0)(arr)

        np.testing.assert_array_equal(result, np.ones((4, 4)))
        assert len(calls) == 1
        assert calls[0]["steps"] == [{
            "op": "remove_bad_lines",
            "params": {"threshold_mad": 3.0},
        }]


# ─── pipeline records each step in order ─────────────────────────────────────

class TestPipelineHistory:
    def _run_pipeline(self, first_sample_dat, tmp_path, steps):
        """Run the pipeline CLI and return the in-flight Scan (before save)."""
        captured = []

        def _capture(args, scan, default_suffix):
            captured.append(scan)
            return tmp_path / "out.sxm"

        with patch("probeflow.cli.commands.processing._write_output", side_effect=_capture):
            rc = main([
                "pipeline", str(first_sample_dat),
                "-o", str(tmp_path / "out.sxm"),
                "--steps", *steps,
            ])
        assert rc == 0
        return captured[0]

    def test_two_steps_recorded_in_order(self, first_sample_dat, tmp_path):
        scan = self._run_pipeline(first_sample_dat, tmp_path,
                                  ["align-rows:median", "plane-bg:1"])
        ops = [e["op"] for e in scan.processing_history]
        assert ops == ["align_rows", "plane_bg"]

    def test_pipeline_history_params_correct(self, first_sample_dat, tmp_path):
        scan = self._run_pipeline(first_sample_dat, tmp_path, ["smooth:2.0"])
        assert scan.processing_history[0]["params"] == {"sigma_px": 2.0}


# ─── CLI output safety ───────────────────────────────────────────────────────

class TestCliOutputSafety:
    def test_single_op_default_blocks_selected_plane_sxm(self, first_sample_dat, tmp_path):
        src = tmp_path / first_sample_dat.name
        src.write_bytes(first_sample_dat.read_bytes())
        existing = tmp_path / f"{src.stem}.sxm"
        existing.write_bytes(b"sentinel")

        rc = main(["smooth", str(src)])

        assert rc == 1
        assert existing.read_bytes() == b"sentinel"
        assert not (tmp_path / f"{src.stem}_smooth.sxm").exists()

    def test_existing_explicit_output_requires_force(self, first_sample_dat, tmp_path):
        src = tmp_path / first_sample_dat.name
        src.write_bytes(first_sample_dat.read_bytes())
        out = tmp_path / "processed.png"
        out.write_bytes(b"sentinel")

        rc = main(["smooth", str(src), "--png", "-o", str(out)])

        assert rc == 1
        assert out.read_bytes() == b"sentinel"

    def test_force_allows_explicit_output_replacement(self, first_sample_dat, tmp_path):
        src = tmp_path / first_sample_dat.name
        src.write_bytes(first_sample_dat.read_bytes())
        out = tmp_path / "processed.png"
        out.write_bytes(b"sentinel")

        rc = main(["smooth", str(src), "--png", "-o", str(out), "--force"])

        assert rc == 0
        assert out.read_bytes() != b"sentinel"
