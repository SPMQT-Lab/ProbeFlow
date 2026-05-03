"""Tests for the writers (PDF / CSV) + save_scan dispatch.

We use a real Scan loaded from a bundled ``.dat`` sample so the writers are
exercised end-to-end on realistic data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from probeflow import load_scan
from probeflow.io.writers import (
    SUPPORTED_OUTPUT_SUFFIXES,
    save_scan,
    write_csv,
    write_gwy,
    write_pdf,
)


@pytest.fixture
def dat_scan(first_sample_dat):
    return load_scan(first_sample_dat)


class TestSupportedOutputSuffixes:
    def test_has_all_expected(self):
        for s in (".sxm", ".gwy", ".png", ".pdf", ".csv"):
            assert s in SUPPORTED_OUTPUT_SUFFIXES

    def test_removed_suffixes_absent(self):
        for s in (".tif", ".tiff"):
            assert s not in SUPPORTED_OUTPUT_SUFFIXES


class TestPdf:
    def test_writes_file(self, dat_scan, tmp_path):
        out = tmp_path / "out.pdf"
        write_pdf(dat_scan, out, plane_idx=0, colormap="gray")
        assert out.exists() and out.stat().st_size > 0
        # PDF files start with the magic %PDF
        assert out.read_bytes()[:4] == b"%PDF"

    def test_via_save_method(self, dat_scan, tmp_path):
        out = tmp_path / "via_save.pdf"
        dat_scan.save(out, plane_idx=0)
        assert out.exists()


class TestCsv:
    def test_writes_grid(self, dat_scan, tmp_path):
        out = tmp_path / "out.csv"
        write_csv(dat_scan, out, plane_idx=0)
        # Use numpy.loadtxt (skips # comments) to read back.
        arr = np.loadtxt(out, delimiter=",")
        # Shape should match the plane (after stripping comment header)
        assert arr.shape == dat_scan.planes[0].shape

    def test_header_line_present(self, dat_scan, tmp_path):
        out = tmp_path / "out.csv"
        write_csv(dat_scan, out, plane_idx=0)
        first_line = out.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#")
        assert "plane=" in first_line
        assert "width_m=" in first_line

    def test_via_save_method(self, dat_scan, tmp_path):
        out = tmp_path / "via_save.csv"
        dat_scan.save(out, plane_idx=0)
        assert out.exists()


class TestGwy:
    def test_writes_file(self, dat_scan, tmp_path):
        pytest.importorskip("gwyfile")
        from gwyfile.objects import GwyContainer
        out = tmp_path / "out.gwy"
        plane_idx = 2
        write_gwy(dat_scan, out, plane_idx=plane_idx)
        assert out.exists() and out.stat().st_size > 4
        assert out.read_bytes()[:4] == b"GWYP"
        obj = GwyContainer.fromfile(str(out))
        assert obj["/0/data/title"] == dat_scan.plane_names[plane_idx]
        assert obj["/0/data"].data.shape == dat_scan.planes[plane_idx].shape
        meta = obj["/0/meta"]
        prov = json.loads(meta["ProbeFlow export provenance"])
        assert prov["export_kind"] == "gwy"
        assert prov["channel_index"] == plane_idx
        assert meta["ProbeFlow processing state hash"] == prov["processing_state_hash"]
        assert "/1/data" not in obj

    def test_via_save_method(self, dat_scan, tmp_path):
        pytest.importorskip("gwyfile")
        out = tmp_path / "via_save.gwy"
        dat_scan.save_gwy(out, plane_idx=1)
        assert out.exists()


class TestSaveScanDispatch:
    def test_unknown_suffix_raises(self, dat_scan, tmp_path):
        with pytest.raises(ValueError, match="Unsupported output"):
            save_scan(dat_scan, tmp_path / "out.zzz")

    def test_png_routes_correctly(self, dat_scan, tmp_path):
        out = tmp_path / "ok.png"
        save_scan(dat_scan, out, plane_idx=0, colormap="gray")
        assert out.exists()

    def test_pdf_routes_correctly(self, dat_scan, tmp_path):
        out = tmp_path / "ok.pdf"
        save_scan(dat_scan, out, plane_idx=0, colormap="gray")
        assert out.exists()

    def test_gwy_routes_correctly(self, dat_scan, tmp_path):
        pytest.importorskip("gwyfile")
        out = tmp_path / "ok.gwy"
        save_scan(dat_scan, out, plane_idx=3)
        assert out.exists()
