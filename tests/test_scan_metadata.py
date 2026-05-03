"""Tests for probeflow.core.metadata — ScanMetadata and read_scan_metadata()."""

from __future__ import annotations

from pathlib import Path

import pytest

from probeflow import read_scan_metadata
from probeflow.core.metadata import _extract_createc_fields, metadata_from_scan, ScanMetadata
from probeflow.core.scan_loader import load_scan


TESTDATA = Path(__file__).resolve().parents[1] / "anonymised_testdata"
_CREATEC_STEP    = TESTDATA / "createc_scan_step_20nm.dat"
_CREATEC_TERRACE = TESTDATA / "createc_scan_terrace_109nm.dat"
_CREATEC_OVERVIEW = TESTDATA / "createc_scan_overview_240nm_pos.dat"
_NANONIS_SXM     = TESTDATA / "sxm_moire_10nm.sxm"
_CREATEC_VERT    = TESTDATA / "createc_ivt_telegraph_300mv_a.VERT"


# ── Test A: Createc metadata matches loaded scan ──────────────────────────────

class TestCreatecMetadataMatchesScan:
    @pytest.mark.parametrize("path,expected_shape", [
        (_CREATEC_STEP,    (330, 511)),
        (_CREATEC_TERRACE, (512, 511)),
    ])
    def test_source_format(self, path, expected_shape):
        meta = read_scan_metadata(path)
        assert meta.source_format == "createc_dat"

    @pytest.mark.parametrize("path,expected_shape", [
        (_CREATEC_STEP,    (330, 511)),
        (_CREATEC_TERRACE, (512, 511)),
    ])
    def test_item_type(self, path, expected_shape):
        meta = read_scan_metadata(path)
        assert meta.item_type == "scan"

    @pytest.mark.parametrize("path,expected_shape", [
        (_CREATEC_STEP,    (330, 511)),
        (_CREATEC_TERRACE, (512, 511)),
    ])
    def test_shape_matches_scan(self, path, expected_shape):
        scan = load_scan(path)
        meta = read_scan_metadata(path)
        assert meta.shape == scan.planes[0].shape

    @pytest.mark.parametrize("path,expected_shape", [
        (_CREATEC_STEP,    (330, 511)),
        (_CREATEC_TERRACE, (512, 511)),
    ])
    def test_post_fix_dimensions(self, path, expected_shape):
        meta = read_scan_metadata(path)
        assert meta.shape == expected_shape

    @pytest.mark.parametrize("path", [_CREATEC_STEP, _CREATEC_TERRACE])
    def test_plane_names_match_scan(self, path):
        scan = load_scan(path)
        meta = read_scan_metadata(path)
        assert meta.plane_names == tuple(scan.plane_names)

    @pytest.mark.parametrize("path", [_CREATEC_STEP, _CREATEC_TERRACE])
    def test_units_match_scan(self, path):
        scan = load_scan(path)
        meta = read_scan_metadata(path)
        assert meta.units == tuple(scan.plane_units)

    @pytest.mark.parametrize("path", [_CREATEC_STEP, _CREATEC_TERRACE])
    def test_experiment_metadata_matches_scan(self, path):
        scan = load_scan(path)
        meta = read_scan_metadata(path)
        assert meta.experiment_metadata == scan.experiment_metadata

    def test_modern_stm_fixture_infers_current_feedback(self):
        meta = read_scan_metadata(_CREATEC_OVERVIEW)
        assert meta.experiment_metadata["acquisition_mode"] == "stm"
        assert meta.experiment_metadata["feedback_mode"] == "current"


# ── Test B: Createc metadata width agrees with Num.X ─────────────────────────

class TestCreatecHeaderConsistency:
    @pytest.mark.parametrize("path", [_CREATEC_STEP, _CREATEC_TERRACE])
    def test_width_matches_num_x(self, path):
        meta = read_scan_metadata(path)
        nx_hdr = int(float(meta.raw_header["Num.X"]))
        assert meta.shape is not None
        _, Nx = meta.shape
        assert Nx == nx_hdr

    @pytest.mark.parametrize("path", [_CREATEC_STEP, _CREATEC_TERRACE])
    def test_height_matches_num_y(self, path):
        meta = read_scan_metadata(path)
        ny_hdr = int(float(meta.raw_header["Num.Y"]))
        assert meta.shape is not None
        Ny, _ = meta.shape
        assert Ny == ny_hdr


# ── Test C: Nanonis metadata matches loaded scan ──────────────────────────────

class TestNanonisMetadataMatchesScan:
    def test_source_format(self):
        meta = read_scan_metadata(_NANONIS_SXM)
        assert meta.source_format == "nanonis_sxm"

    def test_item_type(self):
        meta = read_scan_metadata(_NANONIS_SXM)
        assert meta.item_type == "scan"

    def test_shape_matches_scan(self):
        scan = load_scan(_NANONIS_SXM)
        meta = read_scan_metadata(_NANONIS_SXM)
        assert meta.shape == scan.planes[0].shape

    def test_plane_names_match_scan(self):
        scan = load_scan(_NANONIS_SXM)
        meta = read_scan_metadata(_NANONIS_SXM)
        assert meta.plane_names == tuple(scan.plane_names)

    def test_units_match_scan(self):
        scan = load_scan(_NANONIS_SXM)
        meta = read_scan_metadata(_NANONIS_SXM)
        assert meta.units == tuple(scan.plane_units)


# ── Test D: spectroscopy files are rejected ───────────────────────────────────

class TestSpectroscopyRejected:
    def test_createc_vert_raises(self):
        with pytest.raises(ValueError, match="spectroscopy"):
            read_scan_metadata(_CREATEC_VERT)


# ── Test E: unknown files are rejected ───────────────────────────────────────

class TestUnknownFileRejected:
    def test_text_file_raises(self, tmp_path):
        bad = tmp_path / "not_a_scan.txt"
        bad.write_text("this is not an SPM file")
        with pytest.raises(ValueError):
            read_scan_metadata(bad)


# ── Additional: ScanMetadata contract ────────────────────────────────────────

class TestScanMetadataContract:
    def test_is_frozen(self):
        meta = read_scan_metadata(_CREATEC_STEP)
        with pytest.raises((AttributeError, TypeError)):
            meta.shape = (1, 1)  # type: ignore[misc]

    def test_scan_range_is_positive(self):
        meta = read_scan_metadata(_CREATEC_STEP)
        assert meta.scan_range is not None
        w, h = meta.scan_range
        assert w > 0 and h > 0

    def test_bias_is_float(self):
        meta = read_scan_metadata(_CREATEC_STEP)
        assert isinstance(meta.bias, float)

    def test_setpoint_is_float(self):
        meta = read_scan_metadata(_CREATEC_STEP)
        assert isinstance(meta.setpoint, float)

    def test_display_name_is_stem(self):
        meta = read_scan_metadata(_CREATEC_STEP)
        assert meta.display_name == _CREATEC_STEP.stem

    def test_raw_header_is_dict(self):
        meta = read_scan_metadata(_CREATEC_STEP)
        assert isinstance(meta.raw_header, dict)
        assert len(meta.raw_header) > 0

    def test_reader_specific_functions(self):
        from probeflow.io.readers.createc_scan import read_dat_metadata
        from probeflow.io.readers.nanonis_sxm import read_sxm_metadata
        meta_dat = read_dat_metadata(_CREATEC_STEP)
        meta_sxm = read_sxm_metadata(_NANONIS_SXM)
        assert isinstance(meta_dat, ScanMetadata)
        assert isinstance(meta_sxm, ScanMetadata)
        assert meta_dat.source_format == "createc_dat"
        assert meta_sxm.source_format == "nanonis_sxm"

    def test_sxm_metadata_does_not_load_full_scan(self, monkeypatch):
        def fail_load_scan(_path):
            raise AssertionError("SXM metadata should not construct a full Scan")

        def fail_read_planes(*_args, **_kwargs):
            raise AssertionError("SXM metadata should not decode image planes")

        monkeypatch.setattr("probeflow.core.scan_loader.load_scan", fail_load_scan)
        monkeypatch.setattr("probeflow.io.readers.nanonis_sxm.read_all_sxm_planes", fail_read_planes)

        meta = read_scan_metadata(_NANONIS_SXM)

        assert meta.source_format == "nanonis_sxm"
        assert meta.shape is not None
        assert meta.plane_names


class TestCreatecSetpointExtraction:
    def test_current_a_preferred(self):
        _bias, setpoint, _comment, _dt = _extract_createc_fields({
            "Current[A]": "4.4E-10",
            "SetPoint": "1.0E-9",
            "FBLogIset": "999",
        })
        assert setpoint == pytest.approx(4.4e-10)

    def test_setpoint_fallback_is_amps(self):
        _bias, setpoint, _comment, _dt = _extract_createc_fields({
            "SetPoint": "1.01E-10",
            "FBLogIset": "999",
        })
        assert setpoint == pytest.approx(1.01e-10)

    def test_fblogiset_last_resort_is_pa(self):
        _bias, setpoint, _comment, _dt = _extract_createc_fields({
            "FBLogIset": "88.4",
        })
        assert setpoint == pytest.approx(88.4e-12)

    def test_zero_setpoints_remain_unknown(self):
        _bias, setpoint, _comment, _dt = _extract_createc_fields({
            "Current[A]": "0",
            "SetPoint": "0.00E+00",
            "FBLogIset": "0.000",
        })
        assert setpoint is None

    def test_newer_createc_fixture_uses_setpoint_header(self):
        meta = read_scan_metadata(TESTDATA / "createc_scan_overview_240nm_pos.dat")
        assert meta.setpoint == pytest.approx(1.01e-10)

    def test_createc_preview_fixture_uses_setpoint_header(self):
        meta = read_scan_metadata(TESTDATA / "createc_scan_preview_120nm.dat")
        assert meta.setpoint == pytest.approx(1.01e-10)

    def test_createc_zero_setpoint_fixture_remains_unknown(self):
        meta = read_scan_metadata(TESTDATA / "createc_scan_island_60nm.dat")
        assert meta.setpoint is None
