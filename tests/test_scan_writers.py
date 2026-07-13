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
from probeflow.gui.viewer.processed_export import (
    build_processed_scan_for_export,
    save_processed_image,
)
from probeflow.io.sxm_io import parse_sxm_header
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

    def test_pdf_writer_can_omit_provenance_sidecars(self, dat_scan, tmp_path):
        out = tmp_path / "without_provenance.pdf"

        dat_scan.save_pdf(out, plane_idx=0, include_provenance=False)

        assert out.read_bytes()[:4] == b"%PDF"
        assert not out.with_suffix(".probeflow.json").exists()
        assert not out.with_suffix(".provenance.json").exists()


class TestSxmViewerExport:
    def test_processed_multichannel_createc_scan_records_selected_plane(
        self,
        tmp_path,
    ):
        source = Path(__file__).resolve().parents[1] / "test_data" / "createc_afm.dat"
        raw = load_scan(source)
        processed = np.asarray(raw.planes[0]) + 1e-12
        scan, plane_idx = build_processed_scan_for_export(
            source,
            channel_idx=0,
            display_arr=processed,
            processing_gui_state={"align_rows": "median"},
        )
        out = tmp_path / "processed_multichannel.sxm"

        msg = save_processed_image(
            scan,
            plane_idx,
            out,
            display_settings={"colormap": "magma"},
            include_provenance=True,
        )

        assert msg.startswith("Saved processed image")
        reloaded = load_scan(out)
        assert reloaded.plane_names == raw.plane_names
        assert reloaded.plane_units == raw.plane_units
        np.testing.assert_allclose(
            reloaded.planes[0], processed, rtol=1e-6, atol=1e-15,
        )
        np.testing.assert_allclose(
            reloaded.planes[1], raw.planes[1], rtol=1e-6, atol=1e-15,
        )
        comment = parse_sxm_header(out)["COMMENT"]
        assert "ProcessedPlane: 0 (Z forward)" in comment
        assert "other planes preserved without this processing" in comment
        sidecar = json.loads(
            out.with_suffix(".probeflow.json").read_text(encoding="utf-8")
        )
        assert sidecar["source_info"]["channel"] == "Z forward"
        assert sidecar["source_info"]["metadata"]["channel_index"] == 0

    def test_sxm_writer_can_omit_provenance_sidecar(self, first_sample_dat, tmp_path):
        out = tmp_path / "without_provenance.sxm"

        load_scan(first_sample_dat).save_sxm(out, include_provenance=False)

        assert out.is_file()
        assert not out.with_suffix(".probeflow.json").exists()


class TestCsv:
    def test_csv_writer_and_scan_save_method_preserve_grid_shape_and_header(self, dat_scan, tmp_path):
        out = tmp_path / "out.csv"
        write_csv(dat_scan, out, plane_idx=0)
        arr = np.loadtxt(out, delimiter=",")
        assert arr.shape == dat_scan.planes[0].shape
        first_line = out.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#")
        assert "plane=" in first_line
        lines = out.read_text(encoding="utf-8").splitlines()
        assert any(line.startswith("# width_m=") for line in lines)
        assert any(line.startswith("# channel_index=0") for line in lines)
        assert any(line.startswith("# processing_state_hash=") for line in lines)
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
        sidecar = out.with_suffix(".probeflow.json")
        assert sidecar.exists()
        sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert sidecar_data["export_format"] == "gwy"
        assert sidecar_data["source_info"]["channel"] == dat_scan.plane_names[plane_idx]

        via_save = tmp_path / "via_save.gwy"
        dat_scan.save_gwy(via_save, plane_idx=1)
        assert via_save.read_bytes()[:4] == b"GWYP"

    def test_basic_metadata_does_not_require_provenance(self, dat_scan, tmp_path, monkeypatch):
        import probeflow.io.writers.gwy as gwy_writer

        class FakeContainer(dict):
            last = None

            def tofile(self, _path):
                FakeContainer.last = self

        class FakeDataField:
            def __init__(self, data, **kwargs):
                self.data = data
                self.kwargs = kwargs

        monkeypatch.setattr(
            gwy_writer, "_import_gwyfile", lambda: (FakeContainer, FakeDataField)
        )

        gwy_writer.write_gwy(
            dat_scan,
            tmp_path / "metadata_only.gwy",
            include_meta=True,
            include_provenance=False,
        )

        meta = FakeContainer.last["/0/meta"]
        assert meta["ProbeFlow plane name"] == dat_scan.plane_names[0]
        assert "ProbeFlow export provenance" not in meta


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
