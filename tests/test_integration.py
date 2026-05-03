"""Integration tests: run full conversion pipelines on sample .dat files."""

import json
from pathlib import Path

import numpy as np
import pytest

from probeflow.io.converters.createc_dat_to_png import dat_to_hdr_imgs
from probeflow.io.converters.createc_dat_to_sxm import (
    convert_dat_to_sxm,
    load_layout_and_format,
    process_dat,
    reconstruct_from_hdr_imgs,
)
from probeflow.core.metadata import read_scan_metadata
from probeflow.core.scan_loader import load_scan


# ─── PNG conversion ───────────────────────────────────────────────────────────

class TestDatToPng:
    def test_produces_output_files(self, first_sample_dat, tmp_path):
        out_dir = tmp_path / "out"
        rep = dat_to_hdr_imgs(first_sample_dat, out_dir)

        assert (out_dir / "hdr.txt").exists(), "hdr.txt should be created"
        assert (out_dir / "pngs").is_dir(), "pngs/ subdirectory should exist"

        pngs = list((out_dir / "pngs").glob("*.png"))
        assert len(pngs) > 0, "At least one PNG should be produced"

    def test_return_dict_keys(self, first_sample_dat, tmp_path):
        rep = dat_to_hdr_imgs(first_sample_dat, tmp_path / "out")
        for key in ("Nx", "Ny", "num_channels", "z_scale_m_per_dac", "i_scale_a_per_dac"):
            assert key in rep

    def test_pixel_dimensions_positive(self, first_sample_dat, tmp_path):
        rep = dat_to_hdr_imgs(first_sample_dat, tmp_path / "out")
        assert rep["Nx"] > 0
        assert rep["Ny"] > 0

    def test_z_scale_is_positive(self, first_sample_dat, tmp_path):
        rep = dat_to_hdr_imgs(first_sample_dat, tmp_path / "out")
        assert rep["z_scale_m_per_dac"] > 0

    def test_pngs_are_valid_images(self, first_sample_dat, tmp_path):
        from PIL import Image
        out_dir = tmp_path / "out"
        dat_to_hdr_imgs(first_sample_dat, out_dir)
        for png in (out_dir / "pngs").glob("*.png"):
            img = Image.open(png)
            assert img.mode == "L"
            assert img.size[0] > 0 and img.size[1] > 0

    def test_all_sample_files(self, sample_dat_files, tmp_path):
        errors = {}
        for dat in sample_dat_files:
            try:
                dat_to_hdr_imgs(dat, tmp_path / dat.stem)
            except Exception as exc:
                errors[dat.name] = str(exc)
        assert not errors, f"Conversion failed for: {errors}"

    def test_custom_clip_levels(self, first_sample_dat, tmp_path):
        rep = dat_to_hdr_imgs(first_sample_dat, tmp_path / "out", clip_low=5.0, clip_high=95.0)
        pngs = list((tmp_path / "out" / "pngs").glob("*.png"))
        assert len(pngs) > 0

    def test_invalid_file_raises(self, tmp_path):
        bad = tmp_path / "bad.dat"
        bad.write_bytes(b"not a nanonis file at all")
        with pytest.raises(ValueError, match="missing DATA marker"):
            dat_to_hdr_imgs(bad, tmp_path / "out")


# ─── SXM conversion (process_dat) ────────────────────────────────────────────

class TestProcessDat:
    def test_returns_correct_types(self, first_sample_dat):
        hdr, imgs, num_chan = process_dat(first_sample_dat)
        assert isinstance(hdr, dict)
        assert isinstance(imgs, list)
        assert num_chan in (2, 4)

    def test_always_returns_4_image_planes(self, first_sample_dat):
        _hdr, imgs, _n = process_dat(first_sample_dat)
        assert len(imgs) == 4

    def test_image_planes_are_float32(self, first_sample_dat):
        _hdr, imgs, _n = process_dat(first_sample_dat)
        for _nm, _un, _dr, arr in imgs:
            assert arr.dtype == np.float32

    def test_hdr_has_required_keys(self, first_sample_dat):
        hdr, _imgs, _n = process_dat(first_sample_dat)
        for key in ("NANONIS_VERSION", "SCAN_PIXELS", "SCAN_RANGE", "DATA_INFO"):
            assert key in hdr, f"Missing expected header key: {key}"

    def test_forward_backward_shapes_match(self, first_sample_dat):
        _hdr, imgs, _n = process_dat(first_sample_dat)
        fwd_z  = imgs[0][3]
        bwd_z  = imgs[1][3]
        fwd_i  = imgs[2][3]
        bwd_i  = imgs[3][3]
        assert fwd_z.shape == bwd_z.shape
        assert fwd_i.shape == bwd_i.shape

    def test_invalid_file_raises(self, tmp_path):
        bad = tmp_path / "bad.dat"
        bad.write_bytes(b"garbage data here")
        with pytest.raises(ValueError, match="missing DATA marker"):
            process_dat(bad)


# ─── SXM full conversion ──────────────────────────────────────────────────────

class TestConvertDatToSxm:
    @staticmethod
    def _write_legacy_sxm(dat: Path, out_path: Path, cushion_dir: Path) -> Path:
        hdr, imgs, _num_chan = process_dat(dat)
        layout, header_format = load_layout_and_format(cushion_dir)
        reconstruct_from_hdr_imgs(
            hdr=hdr,
            imgs=imgs,
            header_format=header_format,
            post_end_bytes=layout["post_end_bytes"],
            pre_payload_bytes=layout["pre_payload_bytes"],
            out_path=out_path,
            tail_bytes=layout["tail_bytes"],
            force_data_offset=layout["data_offset"],
            filler_char=b" ",
        )
        return out_path

    def test_uses_validated_scan_writer_path(self, tmp_path, monkeypatch):
        import probeflow.io.converters.createc_dat_to_sxm as dat_sxm_mod

        dat = tmp_path / "scan.dat"
        dat.write_bytes(b"placeholder")
        cushion = tmp_path / "cushion"
        seen = {}

        def fake_load_scan(path):
            seen["load_path"] = path
            return "scan-object"

        def fake_write_sxm(scan, out_path, **kwargs):
            seen["scan"] = scan
            seen["out_path"] = out_path
            seen["kwargs"] = kwargs

        monkeypatch.setattr(dat_sxm_mod, "load_scan", fake_load_scan)
        monkeypatch.setattr(dat_sxm_mod, "write_sxm", fake_write_sxm)

        dat_sxm_mod.convert_dat_to_sxm(
            dat,
            tmp_path / "out",
            cushion,
            clip_low=2.0,
            clip_high=98.0,
        )

        assert seen["load_path"] == dat
        assert seen["scan"] == "scan-object"
        assert seen["out_path"] == tmp_path / "out" / "scan.sxm"
        assert seen["kwargs"] == {
            "cushion_dir": cushion,
            "clip_low": 2.0,
            "clip_high": 98.0,
        }

    def test_validated_conversion_roundtrips_like_legacy_reconstruction(
        self, first_sample_dat, tmp_path, cushion_dir
    ):
        legacy_path = self._write_legacy_sxm(
            first_sample_dat,
            tmp_path / "legacy" / f"{first_sample_dat.stem}.sxm",
            cushion_dir,
        )
        modern_out = tmp_path / "modern"
        convert_dat_to_sxm(first_sample_dat, modern_out, cushion_dir)
        modern_path = modern_out / f"{first_sample_dat.stem}.sxm"

        legacy = load_scan(legacy_path)
        modern = load_scan(modern_path)

        assert legacy.dims == modern.dims
        assert legacy.plane_names == modern.plane_names
        assert legacy.plane_units == modern.plane_units
        assert np.allclose(legacy.scan_range_m, modern.scan_range_m)
        assert legacy.n_planes == modern.n_planes
        for old_plane, new_plane in zip(legacy.planes, modern.planes):
            np.testing.assert_array_equal(old_plane, new_plane)

    def test_validated_conversion_metadata_matches_legacy_reconstruction(
        self, first_sample_dat, tmp_path, cushion_dir
    ):
        legacy_path = self._write_legacy_sxm(
            first_sample_dat,
            tmp_path / "legacy" / f"{first_sample_dat.stem}.sxm",
            cushion_dir,
        )
        modern_out = tmp_path / "modern"
        convert_dat_to_sxm(first_sample_dat, modern_out, cushion_dir)
        modern_path = modern_out / f"{first_sample_dat.stem}.sxm"

        legacy = read_scan_metadata(legacy_path)
        modern = read_scan_metadata(modern_path)

        assert legacy.source_format == modern.source_format == "nanonis_sxm"
        assert legacy.item_type == modern.item_type == "scan"
        assert legacy.display_name == modern.display_name
        assert legacy.shape == modern.shape
        assert legacy.plane_names == modern.plane_names
        assert legacy.units == modern.units
        assert np.allclose(legacy.scan_range, modern.scan_range)
        assert legacy.bias == modern.bias
        assert legacy.setpoint == modern.setpoint
        assert legacy.acquisition_datetime == modern.acquisition_datetime

    def test_produces_sxm_file(self, first_sample_dat, tmp_path, cushion_dir):
        out_dir = tmp_path / "sxm_out"
        convert_dat_to_sxm(first_sample_dat, out_dir, cushion_dir)
        sxm_files = list(out_dir.glob("*.sxm"))
        assert len(sxm_files) == 1

    def test_sxm_filename_matches_stem(self, first_sample_dat, tmp_path, cushion_dir):
        out_dir = tmp_path / "sxm_out"
        convert_dat_to_sxm(first_sample_dat, out_dir, cushion_dir)
        expected = out_dir / (first_sample_dat.stem + ".sxm")
        assert expected.exists()

    def test_sxm_starts_with_nanonis_header(self, first_sample_dat, tmp_path, cushion_dir):
        out_dir = tmp_path / "sxm_out"
        convert_dat_to_sxm(first_sample_dat, out_dir, cushion_dir)
        sxm = list(out_dir.glob("*.sxm"))[0]
        content = sxm.read_bytes()
        assert b":NANONIS_VERSION:" in content

    def test_sxm_contains_scanit_end_marker(self, first_sample_dat, tmp_path, cushion_dir):
        out_dir = tmp_path / "sxm_out"
        convert_dat_to_sxm(first_sample_dat, out_dir, cushion_dir)
        sxm = list(out_dir.glob("*.sxm"))[0]
        content = sxm.read_bytes()
        assert b":SCANIT_END:" in content

    def test_all_sample_files(self, sample_dat_files, tmp_path, cushion_dir):
        errors = {}
        out_dir = tmp_path / "sxm_out"
        for dat in sample_dat_files:
            try:
                convert_dat_to_sxm(dat, out_dir, cushion_dir)
            except Exception as exc:
                errors[dat.name] = str(exc)
        assert not errors, f"SXM conversion failed for: {errors}"
