"""Tests for the writers (PDF / CSV) + save_scan dispatch.

We use a real Scan loaded from a bundled ``.dat`` sample so the writers are
exercised end-to-end on realistic data.
"""

from __future__ import annotations

import json

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
    def test_supported_suffix_policy_matches_current_export_surface(self):
        assert {".sxm", ".gwy", ".png", ".pdf", ".csv"}.issubset(SUPPORTED_OUTPUT_SUFFIXES)
        assert {".tif", ".tiff"}.isdisjoint(SUPPORTED_OUTPUT_SUFFIXES)


class TestPdf:
    def test_pdf_writer_and_scan_save_method_write_pdf_with_provenance(self, dat_scan, tmp_path):
        out = tmp_path / "out.pdf"
        write_pdf(dat_scan, out, plane_idx=0, colormap="gray")
        assert out.exists() and out.stat().st_size > 0
        assert out.read_bytes()[:4] == b"%PDF"
        sidecar = out.with_suffix(".probeflow.json")
        assert sidecar.exists()
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["export_format"] == "pdf"
        assert data["source_info"]["channel"] == dat_scan.plane_names[0]

        via_save = tmp_path / "via_save.pdf"
        dat_scan.save(via_save, plane_idx=0)
        assert via_save.read_bytes()[:4] == b"%PDF"


class TestCsv:
    def test_csv_writer_and_scan_save_method_preserve_grid_shape_and_header(self, dat_scan, tmp_path):
        out = tmp_path / "out.csv"
        write_csv(dat_scan, out, plane_idx=0)
        arr = np.loadtxt(out, delimiter=",")
        assert arr.shape == dat_scan.planes[0].shape
        first_line = out.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#")
        assert "plane=" in first_line
        assert "width_m=" in first_line
        sidecar = out.with_suffix(".probeflow.json")
        assert sidecar.exists()
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data["export_format"] == "csv"
        assert data["source_info"]["channel"] == dat_scan.plane_names[0]

        via_save = tmp_path / "via_save.csv"
        dat_scan.save(via_save, plane_idx=0)
        assert np.loadtxt(via_save, delimiter=",").shape == dat_scan.planes[0].shape


class TestGwy:
    def test_gwy_writer_and_scan_save_method_preserve_channel_and_provenance(self, dat_scan, tmp_path):
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

        via_save = tmp_path / "via_save.gwy"
        dat_scan.save_gwy(via_save, plane_idx=1)
        assert via_save.read_bytes()[:4] == b"GWYP"


class TestSaveScanDispatch:
    def test_unknown_suffix_raises(self, dat_scan, tmp_path):
        with pytest.raises(ValueError, match="Unsupported output"):
            save_scan(dat_scan, tmp_path / "out.zzz")

    def test_dispatch_routes_png_and_pdf_to_format_writers(self, dat_scan, tmp_path):
        png = tmp_path / "ok_png.png"
        pdf = tmp_path / "ok_pdf.pdf"
        save_scan(dat_scan, png, plane_idx=0, colormap="gray")
        save_scan(dat_scan, pdf, plane_idx=0, colormap="gray")
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert pdf.read_bytes()[:4] == b"%PDF"

    def test_gwy_routes_correctly(self, dat_scan, tmp_path):
        pytest.importorskip("gwyfile")
        out = tmp_path / "ok.gwy"
        save_scan(dat_scan, out, plane_idx=3)
        assert out.read_bytes()[:4] == b"GWYP"
