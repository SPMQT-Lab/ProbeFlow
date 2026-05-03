"""Tests for processing provenance written to the .sxm COMMENT field."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probeflow.core.scan_loader import load_scan
from probeflow.io.sxm_io import _patch_comment_in_header, parse_sxm_header
from probeflow.io.writers.sxm import _build_comment


# ─── _build_comment ──────────────────────────────────────────────────────────

class TestBuildComment:
    def _scan_stub(self, source_name: str, history=None):
        from probeflow.core.scan_model import Scan
        return Scan(
            planes=[np.zeros((4, 4))],
            plane_names=["Z forward"],
            plane_units=["m"],
            plane_synthetic=[False],
            header={},
            scan_range_m=(1e-8, 1e-8),
            source_path=Path(f"/fake/{source_name}"),
            source_format="sxm",
            processing_history=history or [],
        )

    def test_no_history_gives_source_only(self):
        scan = self._scan_stub("scan.sxm")
        comment = _build_comment(scan)
        assert comment.startswith("Source: scan.sxm")
        assert "ProcessingStateHash:" in comment
        assert "SourceId:" in comment
        assert "Operations" not in comment

    def test_history_includes_source_and_ops(self):
        scan = self._scan_stub("scan.sxm", history=[
            {"op": "plane_bg",   "params": {"order": 1},         "timestamp": "T"},
            {"op": "align_rows", "params": {"method": "median"},  "timestamp": "T"},
        ])
        comment = _build_comment(scan)
        assert comment.startswith("Source: scan.sxm")
        assert "Operations:" in comment
        assert "1. plane_bg order=1" in comment
        assert "2. align_rows method=median" in comment

    def test_op_without_params_has_no_trailing_space(self):
        scan = self._scan_stub("x.sxm", history=[
            {"op": "no_param_op", "params": {}, "timestamp": "T"},
        ])
        comment = _build_comment(scan)
        assert "1. no_param_op\n" in comment + "\n"  # not "1. no_param_op \n"


# ─── _patch_comment_in_header ─────────────────────────────────────────────────

class TestPatchCommentInHeader:
    _STUB = (
        b":SCANIT_TYPE:\nFLOAT MSBFIRST\n"
        b":COMMENT:\nDefault\n"
        b":DATA_INFO:\nsome info\n"
        b":SCANIT_END:\n"
    )

    def test_replaces_existing_value(self):
        patched = _patch_comment_in_header(self._STUB, "Hello World")
        hdr_str = patched.decode("latin-1")
        assert "Hello World" in hdr_str
        assert "Default" not in hdr_str

    def test_preserves_surrounding_sections(self):
        patched = _patch_comment_in_header(self._STUB, "New")
        hdr_str = patched.decode("latin-1")
        assert ":SCANIT_TYPE:" in hdr_str
        assert ":DATA_INFO:" in hdr_str
        assert ":SCANIT_END:" in hdr_str

    def test_no_comment_section_is_noop(self):
        no_comment = b":SCANIT_TYPE:\nFLOAT\n:SCANIT_END:\n"
        result = _patch_comment_in_header(no_comment, "anything")
        assert result == no_comment

    def test_multiline_new_comment(self):
        patched = _patch_comment_in_header(self._STUB, "Line1\nLine2")
        hdr_str = patched.decode("latin-1")
        assert "Line1" in hdr_str
        assert "Line2" in hdr_str
        assert ":DATA_INFO:" in hdr_str

    def test_replaces_multiline_old_value(self):
        src = (
            b":COMMENT:\nOld line 1\nOld line 2\n"
            b":NEXT:\nval\n"
        )
        patched = _patch_comment_in_header(src, "New")
        hdr_str = patched.decode("latin-1")
        assert "Old line 1" not in hdr_str
        assert "Old line 2" not in hdr_str
        assert "New" in hdr_str
        assert ":NEXT:" in hdr_str


# ─── Round-trip: SXM-sourced path ────────────────────────────────────────────

class TestSxmSourcedComment:
    @pytest.fixture
    def sample_sxm(self):
        p = Path(__file__).parent.parent / "data" / "sample_input" / "sxm"
        files = sorted(p.glob("*.sxm"))
        if not files:
            pytest.skip("No sample .sxm files found")
        return files[0]

    def test_comment_written_with_history(self, sample_sxm, tmp_path):
        scan = load_scan(sample_sxm)
        scan.processing_history = [
            {"op": "align_rows", "params": {"method": "median"}, "timestamp": "T"},
            {"op": "plane_bg",   "params": {"order": 1},         "timestamp": "T"},
        ]
        out = tmp_path / "out.sxm"
        scan.save_sxm(out)

        hdr = parse_sxm_header(out)
        comment = hdr.get("COMMENT", "")
        assert f"Source: {sample_sxm.name}" in comment
        assert "align_rows" in comment
        assert "plane_bg" in comment
        assert "order=1" in comment

    def test_comment_source_only_when_empty_history(self, sample_sxm, tmp_path):
        scan = load_scan(sample_sxm)
        assert scan.processing_history == []
        out = tmp_path / "out.sxm"
        scan.save_sxm(out)

        hdr = parse_sxm_header(out)
        comment = hdr.get("COMMENT", "")
        assert f"Source: {sample_sxm.name}" in comment
        assert "ArtifactId:" in comment
        assert "Operations" not in comment

    def test_non_comment_fields_survive_roundtrip(self, sample_sxm, tmp_path):
        scan = load_scan(sample_sxm)
        out = tmp_path / "out.sxm"
        scan.save_sxm(out)

        orig_hdr = parse_sxm_header(sample_sxm)
        new_hdr  = parse_sxm_header(out)
        for key in ("SCAN_PIXELS", "SCAN_RANGE", "SCAN_DIR", "BIAS"):
            assert new_hdr.get(key) == orig_hdr.get(key), \
                f"Header field {key!r} changed after round-trip"

    def test_data_planes_survive_roundtrip(self, sample_sxm, tmp_path):
        scan = load_scan(sample_sxm)
        scan.processing_history = [
            {"op": "smooth", "params": {"sigma_px": 0.0}, "timestamp": "T"},
        ]
        out = tmp_path / "out.sxm"
        scan.save_sxm(out)

        reloaded = load_scan(out)
        assert reloaded.n_planes == scan.n_planes
        for i, (a, b) in enumerate(zip(scan.planes, reloaded.planes)):
            finite = np.isfinite(a) & np.isfinite(b)
            assert np.allclose(a[finite], b[finite], atol=0, rtol=0), \
                f"Plane {i} changed after COMMENT round-trip"

    def test_file_readable_by_probeflow_reader(self, sample_sxm, tmp_path):
        scan = load_scan(sample_sxm)
        scan.processing_history = [
            {"op": "plane_bg", "params": {"order": 2}, "timestamp": "T"},
        ]
        out = tmp_path / "out.sxm"
        scan.save_sxm(out)

        reloaded = load_scan(out)
        assert reloaded.source_format == "sxm"
        assert reloaded.n_planes >= 1


# ─── Round-trip: DAT-sourced path ────────────────────────────────────────────

class TestDatSourcedComment:
    def test_comment_written_with_history(self, first_sample_dat, tmp_path):
        scan = load_scan(first_sample_dat)
        scan.processing_history = [
            {"op": "smooth", "params": {"sigma_px": 1.5}, "timestamp": "T"},
        ]
        out = tmp_path / "out.sxm"
        scan.save_sxm(out)

        hdr = parse_sxm_header(out)
        comment = hdr.get("COMMENT", "")
        assert f"Source: {first_sample_dat.name}" in comment
        assert "smooth" in comment
        assert "sigma_px=1.5" in comment

    def test_comment_source_only_when_empty_history(self, first_sample_dat, tmp_path):
        scan = load_scan(first_sample_dat)
        out = tmp_path / "out.sxm"
        scan.save_sxm(out)

        hdr = parse_sxm_header(out)
        comment = hdr.get("COMMENT", "")
        assert f"Source: {first_sample_dat.name}" in comment
        assert "Operations" not in comment

    def test_dat_output_readable_by_probeflow_reader(self, first_sample_dat, tmp_path):
        scan = load_scan(first_sample_dat)
        scan.processing_history = [
            {"op": "plane_bg", "params": {"order": 1}, "timestamp": "T"},
        ]
        out = tmp_path / "out.sxm"
        scan.save_sxm(out)

        reloaded = load_scan(out)
        assert reloaded.source_format == "sxm"
        assert reloaded.dims == scan.dims
