"""Tests for probeflow.core.metadata — ScanMetadata and read_scan_metadata()."""

from __future__ import annotations

from pathlib import Path

import pytest

from probeflow import read_scan_metadata
from probeflow.core.metadata import (
    _extract_createc_fields,
    _extract_nanonis_fields,
    ScanMetadata,
)
from probeflow.core.scan_loader import load_scan
from probeflow.io.converters.createc_dat_to_sxm import construct_hdr


TESTDATA = Path(__file__).resolve().parents[1] / "test_data"
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
    def test_createc_metadata_matches_loaded_scan_and_known_post_fix_dimensions(self, path, expected_shape):
        scan = load_scan(path)
        meta = read_scan_metadata(path)
        assert meta.source_format == "createc_dat"
        assert meta.item_type == "scan"
        assert meta.shape == scan.planes[0].shape
        assert meta.shape == expected_shape
        assert meta.plane_names == tuple(scan.plane_names)
        assert meta.units == tuple(scan.plane_units)
        assert meta.experiment_metadata == scan.experiment_metadata

    def test_modern_stm_fixture_infers_current_feedback(self):
        meta = read_scan_metadata(_CREATEC_OVERVIEW)
        assert meta.experiment_metadata["acquisition_mode"] == "stm"
        assert meta.experiment_metadata["feedback_mode"] == "current"


# ── Test B: Createc metadata width agrees with Num.X ─────────────────────────

class TestCreatecHeaderConsistency:
    @pytest.mark.parametrize("path", [_CREATEC_STEP, _CREATEC_TERRACE])
    def test_shape_matches_createc_num_x_and_num_y_headers(self, path):
        meta = read_scan_metadata(path)
        nx_hdr = int(float(meta.raw_header["Num.X"]))
        ny_hdr = int(float(meta.raw_header["Num.Y"]))
        assert meta.shape is not None
        Ny, _ = meta.shape
        _, Nx = meta.shape
        assert Nx == nx_hdr
        assert Ny == ny_hdr


# ── Test C: Nanonis metadata matches loaded scan ──────────────────────────────

class TestNanonisMetadataMatchesScan:
    def test_nanonis_metadata_matches_loaded_scan_contract(self):
        scan = load_scan(_NANONIS_SXM)
        meta = read_scan_metadata(_NANONIS_SXM)
        assert meta.source_format == "nanonis_sxm"
        assert meta.item_type == "scan"
        assert meta.shape == scan.planes[0].shape
        assert meta.plane_names == tuple(scan.plane_names)
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

    def test_scan_metadata_scalar_and_header_contract(self):
        meta = read_scan_metadata(_CREATEC_STEP)
        assert meta.scan_range is not None
        w, h = meta.scan_range
        assert w > 0 and h > 0
        assert isinstance(meta.bias, float)
        assert isinstance(meta.setpoint, float)
        assert meta.display_name == _CREATEC_STEP.stem
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

    def test_constant_df_afm_does_not_report_setpoint_as_current(self):
        # FBChannel=4, PLLOn=1: SetPoint (7.000) is a Δf in Hz, not a current.
        # FBLogIset (700) is a stale current-loop value. Neither is amps.
        _bias, setpoint, _comment, _dt = _extract_createc_fields({
            "FBChannel": "4",
            "PLLOn": "1",
            "SetPoint": "7.000",
            "FBLogIset": "700.0",
        })
        assert setpoint is None

    def test_pll_off_current_channel_still_reads_setpoint_as_amps(self):
        _bias, setpoint, _comment, _dt = _extract_createc_fields({
            "FBChannel": "0",
            "PLLOn": "0",
            "SetPoint": "1.01E-10",
        })
        assert setpoint == pytest.approx(1.01e-10)

    def test_qplus_afm_fixture_reports_df_setpoint_not_current(self):
        meta = read_scan_metadata(TESTDATA / "createc_scan_qplus_10ch_afm.dat")
        assert meta.setpoint is None
        assert meta.feedback_setpoint == pytest.approx(7.0)
        assert meta.feedback_setpoint_unit == "Hz"
        assert meta.feedback_setpoint_label == "Δf setpoint"

    def test_stm_fixture_has_no_feedback_setpoint(self):
        meta = read_scan_metadata(TESTDATA / "createc_scan_overview_240nm_pos.dat")
        assert meta.feedback_setpoint is None


class TestNanonisSetpointExtraction:
    def test_converted_dat_header_preserves_z_controller_setpoint(self):
        dat_meta = read_scan_metadata(_CREATEC_OVERVIEW)
        sxm_hdr = construct_hdr(
            dict(dat_meta.raw_header),
            _CREATEC_OVERVIEW,
            num_chan=4,
        )

        _bias, setpoint, _comment, _dt = _extract_nanonis_fields(sxm_hdr)

        assert setpoint == pytest.approx(dat_meta.setpoint)

    def test_qplus_afm_conversion_does_not_emit_current_setpoint(self):
        # The constant-Δf qPlus AFM fixture has SetPoint=7.000 Hz with FBChannel=4
        # / PLLOn=1.  Writing that into Z-CONTROLLER as "7.000E+0 A" would
        # propagate the bogus 7e+12 pA tunnel current, so the converter must label
        # the controller with the frequency-shift channel instead.
        dat_meta = read_scan_metadata(TESTDATA / "createc_scan_qplus_10ch_afm.dat")
        sxm_hdr = construct_hdr(
            dict(dat_meta.raw_header),
            TESTDATA / "createc_scan_qplus_10ch_afm.dat",
            num_chan=4,
        )

        z_controller = sxm_hdr["Z-CONTROLLER"]
        assert "A" not in z_controller.split("\n")[-1].split("\t")[3]
        assert "Hz" in z_controller
        assert "Frequency Shift" in z_controller

        # The Nanonis-side extractor must not read any current out of the header.
        _bias, setpoint, _comment, _dt = _extract_nanonis_fields(sxm_hdr)
        assert setpoint is None
