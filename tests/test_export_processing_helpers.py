"""Tests for export safety and GUI processing helper behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probeflow.processing.gui_adapter import (
    NUMERIC_PROC_KEYS,
    apply_processing_state_to_scan,
    gui_state_has_numeric_processing,
    processing_history_entries_from_state,
)
from probeflow.processing.state import ProcessingState, ProcessingStep
from probeflow.core.scan_model import Scan
from probeflow.core.scan_loader import load_scan
from probeflow.io.sxm_io import parse_sxm_header


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_scan(plane: np.ndarray, source_path=None) -> Scan:
    return Scan(
        planes=[plane],
        plane_names=["Z forward"],
        plane_units=["m"],
        plane_synthetic=[False],
        header={},
        scan_range_m=(1e-8, 1e-8),
        source_path=source_path or Path("/fake/file.sxm"),
        source_format="sxm",
    )


# ─── Task 1: Overwrite protection through a real save path ───────────────────

class TestOverwriteProtection:
    def test_save_sxm_to_same_path_raises(self, tmp_path, first_sample_dat):
        scan = load_scan(first_sample_dat)
        out = tmp_path / "out.sxm"
        scan.save_sxm(out)

        # Re-load from the written file and try to overwrite it
        scan2 = load_scan(out)
        with pytest.raises(ValueError, match="overwrite"):
            scan2.save_sxm(out)

    def test_save_sxm_to_different_path_succeeds(self, tmp_path, first_sample_dat):
        scan = load_scan(first_sample_dat)
        out_a = tmp_path / "a.sxm"
        out_b = tmp_path / "b.sxm"
        scan.save_sxm(out_a)
        scan2 = load_scan(out_a)
        scan2.save_sxm(out_b)  # must not raise
        assert out_b.exists()

    def test_save_png_to_same_path_raises(self, tmp_path):
        plane = np.ones((8, 8))
        out = tmp_path / "test.png"
        # Source path set to the PNG output path to simulate attempted overwrite
        scan = _make_scan(plane, source_path=out)
        with pytest.raises(ValueError, match="overwrite"):
            scan.save_png(out)

    def test_save_csv_to_same_path_raises(self, tmp_path):
        plane = np.ones((8, 8))
        out = tmp_path / "test.csv"
        scan = _make_scan(plane, source_path=out)
        with pytest.raises(ValueError, match="overwrite"):
            scan.save_csv(out)

    def test_save_pdf_to_same_path_raises(self, tmp_path):
        plane = np.ones((8, 8))
        out = tmp_path / "test.pdf"
        scan = _make_scan(plane, source_path=out)
        with pytest.raises(ValueError, match="overwrite"):
            scan.save_pdf(out)

    def test_save_dispatch_to_same_path_raises(self, tmp_path, first_sample_dat):
        scan = load_scan(first_sample_dat)
        out = tmp_path / "out.sxm"
        scan.save_sxm(out)
        scan2 = load_scan(out)
        with pytest.raises(ValueError, match="overwrite"):
            scan2.save(out)

    def test_source_path_none_allows_any_output(self, tmp_path):
        plane = np.ones((8, 8))
        scan = _make_scan(plane, source_path=None)
        out = tmp_path / "test.csv"
        scan.save_csv(out)  # must not raise
        assert out.exists()


# ─── Task 2: GUI processing helper — no processing ───────────────────────────

class TestApplyProcessingStateNoOp:
    def test_empty_state_leaves_plane_unchanged(self):
        plane = np.arange(16, dtype=float).reshape(4, 4)
        scan = _make_scan(plane.copy())
        apply_processing_state_to_scan(scan, {})
        np.testing.assert_array_equal(scan.planes[0], plane)

    def test_empty_state_leaves_history_empty(self):
        scan = _make_scan(np.zeros((4, 4)))
        apply_processing_state_to_scan(scan, {})
        assert scan.processing_history == []

    def test_display_only_grain_state_is_ignored(self):
        plane = np.arange(16, dtype=float).reshape(4, 4)
        scan = _make_scan(plane.copy())
        proc_state = {"grain_threshold": 50, "grain_above": True}
        apply_processing_state_to_scan(scan, proc_state)
        np.testing.assert_array_equal(scan.planes[0], plane)
        assert scan.processing_history == []

    def test_all_falsy_numeric_keys_are_noop(self):
        plane = np.ones((4, 4))
        scan = _make_scan(plane.copy())
        falsy_state = {k: None for k in NUMERIC_PROC_KEYS}
        falsy_state["remove_bad_lines"] = False
        falsy_state["facet_level"] = False
        apply_processing_state_to_scan(scan, falsy_state)
        assert scan.processing_history == []

    def test_returns_same_scan_object(self):
        scan = _make_scan(np.zeros((4, 4)))
        result = apply_processing_state_to_scan(scan, {})
        assert result is scan

    def test_gui_state_has_numeric_processing_uses_canonical_conversion(self):
        assert gui_state_has_numeric_processing({"grain_threshold": 50}) is False
        assert gui_state_has_numeric_processing({"align_rows": "median"}) is True
        assert gui_state_has_numeric_processing({
            "set_zero_plane_points": [(0, 0), (3, 0), (0, 3)],
        }) is True

    def test_processing_history_entries_share_canonical_shape(self):
        state = ProcessingState(steps=[
            ProcessingStep("smooth", {"sigma_px": 2.0}),
            ProcessingStep("plane_bg", {"order": 1, "step_tolerance": False}),
        ])
        entries = processing_history_entries_from_state(
            state,
            timestamp="2026-04-28T00:00:00",
        )
        assert entries == [
            {
                "op": "smooth",
                "params": {"sigma_px": 2.0},
                "timestamp": "2026-04-28T00:00:00",
            },
            {
                "op": "plane_bg",
                "params": {"order": 1, "step_tolerance": False},
                "timestamp": "2026-04-28T00:00:00",
            },
        ]


# ─── Task 3: GUI processing helper — one operation ───────────────────────────

class TestApplyProcessingStateSingleOp:
    def test_plane_bg_changes_plane(self):
        # Tilted plane: subtract_background(order=1) should remove the tilt
        y, x = np.mgrid[0:8, 0:8]
        plane = (x + y).astype(float)
        scan = _make_scan(plane.copy())
        apply_processing_state_to_scan(scan, {"bg_order": 1})
        # After plane subtraction the result should be nearly flat
        result = scan.planes[0]
        assert not np.allclose(result, plane), "Plane unchanged — expected modification"
        assert np.std(result) < np.std(plane)

    def test_plane_bg_appends_one_history_entry(self):
        scan = _make_scan(np.ones((8, 8)))
        apply_processing_state_to_scan(scan, {"bg_order": 1})
        assert len(scan.processing_history) == 1

    def test_plane_bg_history_op_and_params_correct(self):
        scan = _make_scan(np.ones((8, 8)))
        apply_processing_state_to_scan(scan, {"bg_order": 2})
        entry = scan.processing_history[0]
        assert entry["op"] == "plane_bg"
        assert entry["params"] == {"order": 2, "step_tolerance": False}
        assert "timestamp" in entry

    def test_align_rows_appends_entry(self):
        scan = _make_scan(np.ones((8, 8)))
        apply_processing_state_to_scan(scan, {"align_rows": "median"})
        assert len(scan.processing_history) == 1
        assert scan.processing_history[0]["op"] == "align_rows"
        assert scan.processing_history[0]["params"] == {"method": "median"}

    def test_smooth_appends_entry(self):
        scan = _make_scan(np.ones((8, 8)))
        apply_processing_state_to_scan(scan, {"smooth_sigma": 1.5})
        assert len(scan.processing_history) == 1
        assert scan.processing_history[0]["op"] == "smooth"
        assert scan.processing_history[0]["params"] == {"sigma_px": 1.5}

    def test_fft_appends_entry(self):
        scan = _make_scan(np.random.default_rng(0).random((16, 16)))
        apply_processing_state_to_scan(scan, {
            "fft_mode": "low_pass", "fft_cutoff": 0.1, "fft_window": "hanning"
        })
        assert len(scan.processing_history) == 1
        assert scan.processing_history[0]["op"] == "fourier_filter"

    def test_edge_detect_appends_entry_with_sigma(self):
        scan = _make_scan(np.ones((16, 16)))
        apply_processing_state_to_scan(scan, {
            "edge_method": "laplacian", "edge_sigma": 1, "edge_sigma2": 2
        })
        assert len(scan.processing_history) == 1
        h = scan.processing_history[0]
        assert h["op"] == "edge_detect"
        assert h["params"]["method"] == "laplacian"
        assert h["params"]["sigma"] == 1.0

    def test_remove_bad_lines_appends_entry(self):
        scan = _make_scan(np.ones((16, 16)))
        apply_processing_state_to_scan(scan, {"remove_bad_lines": True})
        assert len(scan.processing_history) == 1
        h = scan.processing_history[0]
        assert h["op"] == "remove_bad_lines"
        assert h["params"]["threshold_mad"] == 5.0


# ─── Task 4: GUI processing helper — multiple operations ─────────────────────

class TestApplyProcessingStateMultipleOps:
    def test_two_ops_appended_in_order(self):
        plane = np.arange(64, dtype=float).reshape(8, 8)
        scan = _make_scan(plane.copy())
        apply_processing_state_to_scan(scan, {
            "align_rows": "median",
            "bg_order": 1,
        })
        assert len(scan.processing_history) == 2
        assert scan.processing_history[0]["op"] == "align_rows"
        assert scan.processing_history[1]["op"] == "plane_bg"

    def test_three_ops_in_correct_pipeline_order(self):
        plane = np.arange(64, dtype=float).reshape(8, 8)
        scan = _make_scan(plane.copy())
        apply_processing_state_to_scan(scan, {
            "remove_bad_lines": True,
            "align_rows": "mean",
            "smooth_sigma": 1,
        })
        ops = [h["op"] for h in scan.processing_history]
        assert ops == ["remove_bad_lines", "align_rows", "smooth"]

    def test_multiple_ops_mutate_plane(self):
        plane = np.arange(64, dtype=float).reshape(8, 8)
        scan = _make_scan(plane.copy())
        apply_processing_state_to_scan(scan, {
            "align_rows": "median",
            "bg_order": 1,
        })
        assert not np.allclose(scan.planes[0], plane)


# ─── Task 5: SXM COMMENT round-trip after apply_processing_state_to_scan ─────

class TestSxmCommentAfterProcessing:
    @pytest.fixture
    def sample_sxm(self):
        p = Path(__file__).parent.parent / "data" / "sample_input" / "sxm"
        files = sorted(p.glob("*.sxm"))
        if not files:
            pytest.skip("No sample .sxm files found")
        return files[0]

    def test_comment_contains_source_and_ops_after_helper(self, sample_sxm, tmp_path):
        scan = load_scan(sample_sxm)
        apply_processing_state_to_scan(scan, {"bg_order": 1, "align_rows": "median"})
        assert len(scan.processing_history) == 2

        out = tmp_path / "processed.sxm"
        scan.save_sxm(out)

        hdr = parse_sxm_header(out)
        comment = hdr.get("COMMENT", "")
        assert f"Source: {sample_sxm.name}" in comment
        assert "Operations:" in comment
        assert "plane_bg" in comment
        assert "align_rows" in comment

    def test_no_processing_comment_has_no_operations_section(self, sample_sxm, tmp_path):
        scan = load_scan(sample_sxm)
        apply_processing_state_to_scan(scan, {})

        out = tmp_path / "pristine.sxm"
        scan.save_sxm(out)

        hdr = parse_sxm_header(out)
        comment = hdr.get("COMMENT", "")
        assert f"Source: {sample_sxm.name}" in comment
        assert "Operations" not in comment
