"""Tests for probeflow.io.spectroscopy — Createc .VERT file reader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from probeflow.io.readers.createc_vert import read_createc_vert_report
from probeflow.io.spectroscopy import SpecData, SpecMetadata, parse_spec_header, read_spec_file, read_spec_metadata

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VERT_TIME_TRACE  = DATA_DIR / "A180201.152542.M0001.VERT"
VERT_BIAS_SWEEP  = DATA_DIR / "A180201.151737.M0001.VERT"
VERT_TT_50MV     = DATA_DIR / "A180201.124928.VERT"       # time trace, -50 mV
VERT_TT_450MV    = DATA_DIR / "A180208.194656.M0003.VERT"  # time trace, -450 mV
VERT_DIDZ_FIXTURE = Path(__file__).resolve().parent.parent / "anonymised_testdata" / "createc_vert_didz_image_state.VERT"


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def time_trace_spec():
    return read_spec_file(VERT_TIME_TRACE)


@pytest.fixture(scope="module")
def bias_sweep_spec():
    return read_spec_file(VERT_BIAS_SWEEP)


@pytest.fixture(scope="module")
def tt_50mv_spec():
    return read_spec_file(VERT_TT_50MV)


@pytest.fixture(scope="module")
def tt_450mv_spec():
    return read_spec_file(VERT_TT_450MV)


# ─── parse_spec_header ───────────────────────────────────────────────────────

class TestParseSpecHeader:
    def test_returns_dict(self):
        hdr = parse_spec_header(VERT_TIME_TRACE)
        assert isinstance(hdr, dict)
        assert len(hdr) > 10

    def test_dac_type_present(self):
        hdr = parse_spec_header(VERT_TIME_TRACE)
        assert "DAC-Type" in hdr
        assert "20bit" in hdr["DAC-Type"]

    def test_dac_to_a_xy_present(self):
        hdr = parse_spec_header(VERT_TIME_TRACE)
        assert "Dacto[A]xy" in hdr
        val = float(hdr["Dacto[A]xy"])
        assert 0 < val < 1.0  # Createc Dacto field, physically reasonable

    def test_offset_xy_present(self):
        hdr = parse_spec_header(VERT_TIME_TRACE)
        assert "OffsetX" in hdr
        assert "OffsetY" in hdr

    def test_spec_freq_present(self):
        hdr = parse_spec_header(VERT_TIME_TRACE)
        assert "SpecFreq" in hdr
        assert float(hdr["SpecFreq"]) > 0

    def test_does_not_require_full_file_read(self):
        # parse_spec_header should succeed even on a file that is mostly data.
        hdr = parse_spec_header(VERT_BIAS_SWEEP)
        assert "DAC-Type" in hdr


class TestReadSpecMetadata:
    def test_returns_metadata_without_arrays(self):
        meta = read_spec_metadata(VERT_BIAS_SWEEP)
        assert isinstance(meta, SpecMetadata)
        assert not hasattr(meta, "x_array")
        assert not hasattr(meta, "channels_data")

    def test_createc_metadata_matches_full_reader_summary(self, bias_sweep_spec):
        meta = read_spec_metadata(VERT_BIAS_SWEEP)
        assert meta.source_format == "createc_vert"
        assert meta.channels == ("I", "Z", "V")
        assert meta.units == ("A", "m", "V")
        assert meta.metadata["sweep_type"] == bias_sweep_spec.metadata["sweep_type"]
        assert meta.metadata["n_points"] == bias_sweep_spec.metadata["n_points"]
        assert meta.bias == pytest.approx(bias_sweep_spec.metadata["bias_mv"] / 1000.0)

    def test_time_trace_metadata_classification(self, time_trace_spec):
        meta = read_spec_metadata(VERT_TIME_TRACE)
        assert meta.metadata["sweep_type"] == time_trace_spec.metadata["sweep_type"]
        assert meta.metadata["n_points"] == time_trace_spec.metadata["n_points"]

    def test_source_fingerprint_matches_full_reader(self, bias_sweep_spec):
        meta = read_spec_metadata(VERT_BIAS_SWEEP)
        full_source = bias_sweep_spec.metadata["source"]
        metadata_source = meta.metadata["source"]
        assert full_source["source_format"] == "createc_vert"
        assert metadata_source["item_type"] == "spectrum"
        assert metadata_source["sha256"] == full_source["sha256"]
        assert len(metadata_source["sha256"]) == 64
        assert metadata_source["file_size_bytes"] > 0
        assert metadata_source["data_offset"] is not None


# ─── Createc VERT report layer ───────────────────────────────────────────────

class TestCreatecVertReport:
    def test_real_fixture_report_has_source_and_points(self):
        report = read_createc_vert_report(VERT_BIAS_SWEEP)
        assert report.file_version in {"ParVERT30", "ParVERT32"}
        assert report.raw_table_shape == (5000, 4)
        assert report.spec_total_points == 5000
        assert report.source["source_format"] == "createc_vert"
        assert report.source["item_type"] == "spectrum"
        assert len(report.source["sha256"]) == 64
        assert report.source["data_offset"] == report.data_offset
        assert report.raw_columns is not None
        assert set(("idx", "V", "Z", "I")).issubset(report.raw_columns)

    def test_metadata_fast_path_omits_arrays_but_keeps_summary(self):
        report = read_createc_vert_report(VERT_BIAS_SWEEP, include_arrays=False)
        assert report.raw_columns is None
        assert report.raw_table_shape == (5000, 4)
        assert [info.canonical_name for info in report.channel_info] == ["V", "Z", "I"]
        assert [info.unit for info in report.channel_info] == ["V", "m", "A"]
        assert report.bias_min_mv == pytest.approx(-300.0, abs=10.0)
        assert report.bias_max_mv == pytest.approx(-50.0, abs=10.0)
        assert len(report.source["sha256"]) == 64

    def test_parvert30_base_columns(self, tmp_path):
        f = _write_createc_vert(
            tmp_path,
            "par30.VERT",
            _createc_vert_text(
                version="ParVERT30",
                n_rows=2,
                channel_code=1,
                rows=[
                    [0, -50.0, 10.0, -1000.0],
                    [1, -60.0, 20.0, -2000.0],
                ],
            ),
        )
        report = read_createc_vert_report(f)
        assert report.column_names == ("idx", "V", "Z", "I")
        assert [info.origin for info in report.channel_info] == [
            "base",
            "base",
            "bitmask",
        ]

    def test_parvert32_v2_base_columns(self, tmp_path):
        f = _write_createc_vert(
            tmp_path,
            "par32_v2.VERT",
            _createc_vert_text(
                version="ParVERT32",
                n_rows=2,
                channel_code=1,
                rows=[
                    [0, -50.0, 10.0, -1000.0],
                    [1, -60.0, 20.0, -2000.0],
                ],
            ),
        )
        report = read_createc_vert_report(f)
        assert report.output_channel_count_marker == "v2"
        assert report.column_names == ("idx", "V", "Z", "I")

    def test_parvert32_v3_includes_x_column(self, tmp_path):
        f = _write_createc_vert(
            tmp_path,
            "par32_v3.VERT",
            _createc_vert_text(
                version="ParVERT32",
                n_rows=2,
                channel_code=1,
                marker=3,
                rows=[
                    [0, -50.0, 10.0, 101.0, -1000.0],
                    [1, -60.0, 20.0, 102.0, -2000.0],
                ],
            ),
        )
        report = read_createc_vert_report(f)
        assert report.output_channel_count_marker == "v3"
        assert report.column_names == ("idx", "V", "Z", "X", "I")
        assert "X" in [info.canonical_name for info in report.channel_info]

    def test_bitmask_channels_are_exposed_in_order(self, tmp_path):
        f = _write_createc_vert(
            tmp_path,
            "bitmask.VERT",
            _createc_vert_text(
                n_rows=2,
                channel_code=0b1011,
                rows=[
                    [0, -50.0, 0.0, -1000.0, 1.25, 2.5],
                    [1, -60.0, 0.0, -2000.0, 1.50, 3.5],
                ],
            ),
        )
        report = read_createc_vert_report(f)
        assert report.column_names == ("idx", "V", "Z", "I", "dI/dV", "ADC0")
        assert [info.canonical_name for info in report.channel_info] == [
            "V",
            "Z",
            "I",
            "dI/dV",
            "ADC0",
        ]
        assert any("dI/dV decoded without physical calibration" in w for w in report.warnings)
        assert any("ADC0 decoded without physical calibration" in w for w in report.warnings)

    def test_excess_columns_are_preserved_with_warning(self, tmp_path):
        f = _write_createc_vert(
            tmp_path,
            "extra.VERT",
            _createc_vert_text(
                n_rows=1,
                channel_code=1,
                rows=[[0, -50.0, 0.0, -1000.0, 42.0]],
            ),
        )
        report = read_createc_vert_report(f)
        assert report.column_names == ("idx", "V", "Z", "I", "Raw column 4")
        assert "Raw column 4" in [info.canonical_name for info in report.channel_info]
        assert any("preserved unrecognised spectroscopy column 4" in w for w in report.warnings)

    def test_malformed_params_line_raises(self, tmp_path):
        f = _write_createc_vert(
            tmp_path,
            "bad_params.VERT",
            "[ParVERT30]\r\nDATA\r\nnot params\r\n0\t-50\t0\t1\r\n",
        )
        with pytest.raises(ValueError, match="malformed.*params"):
            read_createc_vert_report(f)

    def test_missing_data_marker_raises(self, tmp_path):
        f = _write_createc_vert(tmp_path, "missing.VERT", "[ParVERT30]\r\nkey=val\r\n")
        with pytest.raises(ValueError, match="missing DATA marker"):
            read_createc_vert_report(f)

    def test_mismatched_column_count_raises(self, tmp_path):
        f = _write_createc_vert(
            tmp_path,
            "short_columns.VERT",
            _createc_vert_text(
                n_rows=1,
                channel_code=0b11,
                rows=[[0, -50.0, 0.0, -1000.0]],
            ),
        )
        with pytest.raises(ValueError, match="expected at least 5 data columns"):
            read_createc_vert_report(f)


# ─── read_spec_file — time trace ─────────────────────────────────────────────

class TestReadSpecFileTimeTrace:
    def test_returns_specdata(self, time_trace_spec):
        assert isinstance(time_trace_spec, SpecData)

    def test_sweep_type(self, time_trace_spec):
        assert time_trace_spec.metadata["sweep_type"] == "time_trace"

    def test_n_points(self, time_trace_spec):
        assert time_trace_spec.metadata["n_points"] == 5000

    def test_x_axis_is_time(self, time_trace_spec):
        assert time_trace_spec.x_unit == "s"
        assert "Time" in time_trace_spec.x_label

    def test_x_array_shape(self, time_trace_spec):
        assert time_trace_spec.x_array.shape == (5000,)

    def test_x_array_monotonic(self, time_trace_spec):
        assert np.all(np.diff(time_trace_spec.x_array) >= 0)

    def test_x_range_seconds(self, time_trace_spec):
        # SpecFreq=1000 Hz, 5000 pts → 0 to 4.999 s
        assert time_trace_spec.x_array[0] == pytest.approx(0.0)
        assert time_trace_spec.x_array[-1] == pytest.approx(4.999, rel=1e-3)

    def test_channels_present(self, time_trace_spec):
        for ch in ("I", "Z", "V"):
            assert ch in time_trace_spec.channels

    def test_channel_shapes(self, time_trace_spec):
        for arr in time_trace_spec.channels.values():
            assert arr.shape == (5000,)

    def test_z_channel_units_metres(self, time_trace_spec):
        z = time_trace_spec.channels["Z"]
        assert z.min() > -20e-10  # >-20 Å
        assert z.max() < 20e-10   # <+20 Å

    def test_position_is_tuple_of_floats(self, time_trace_spec):
        px, py = time_trace_spec.position
        assert isinstance(px, float)
        assert isinstance(py, float)

    def test_position_in_metres(self, time_trace_spec):
        px, py = time_trace_spec.position
        assert abs(px) < 1e-6
        assert abs(py) < 1e-6

    def test_y_units_dict(self, time_trace_spec):
        assert time_trace_spec.y_units["I"] == "A"
        assert time_trace_spec.y_units["Z"] == "m"

    def test_bias_constant(self, time_trace_spec):
        v = time_trace_spec.channels["V"]
        assert v.max() - v.min() < 1e-3  # < 1 mV variation


# ─── read_spec_file — bias sweep ─────────────────────────────────────────────

class TestReadSpecFileBiasSweep:
    def test_returns_specdata(self, bias_sweep_spec):
        assert isinstance(bias_sweep_spec, SpecData)

    def test_sweep_type(self, bias_sweep_spec):
        assert bias_sweep_spec.metadata["sweep_type"] == "bias_sweep"

    def test_x_axis_is_bias(self, bias_sweep_spec):
        assert bias_sweep_spec.x_unit == "V"
        assert "Bias" in bias_sweep_spec.x_label

    def test_x_array_shape(self, bias_sweep_spec):
        assert bias_sweep_spec.x_array.shape == (5000,)

    def test_x_range_volts(self, bias_sweep_spec):
        x = bias_sweep_spec.x_array
        assert x.min() == pytest.approx(-0.300, abs=0.01)
        assert x.max() == pytest.approx(-0.050, abs=0.01)

    def test_channels_present(self, bias_sweep_spec):
        for ch in ("I", "Z", "V"):
            assert ch in bias_sweep_spec.channels

    def test_i_varies(self, bias_sweep_spec):
        # For a real I(V) sweep the current must change across the voltage range.
        i = bias_sweep_spec.channels["I"]
        assert float(i.max() - i.min()) > 0


# ─── error handling ──────────────────────────────────────────────────────────

class TestReadSpecFileErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_spec_file(tmp_path / "nonexistent.VERT")

    def test_missing_data_marker(self, tmp_path):
        bad = tmp_path / "bad.VERT"
        bad.write_bytes(b"key=val\r\nother=stuff\r\n")
        with pytest.raises(ValueError, match="DATA"):
            read_spec_file(bad)

    def test_too_few_columns(self, tmp_path):
        # 3 columns instead of required 4
        body = "DATA\r\n    3    0    0    1\r\n"
        body += "0\t-50.0\t0.0\r\n" * 3
        bad = tmp_path / "short.VERT"
        bad.write_bytes(body.encode())
        with pytest.raises(ValueError, match="4"):
            read_spec_file(bad)

    def test_threshold_kwarg_classifies_short_sweep(self, tmp_path):
        # A 0.5 mV sweep should be classified as a sweep if threshold is 0.1 mV.
        # Build a minimal header with Vpoint entries spanning 0.5 mV.
        hdr = (
            "[ParVERT30]\r\n"
            "DAC-Type=20bit\r\n"
            "GainPre 10^=9\r\n"
            "Dacto[A]xy=0.00083\r\n"
            "Dacto[A]z=0.00018\r\n"
            "OffsetX=0\r\nOffsetY=0\r\n"
            "SpecFreq=1000\r\n"
            "Vpoint0.t=0\r\nVpoint0.V=-50.0\r\n"
            "Vpoint1.t=100\r\nVpoint1.V=-50.5\r\n"  # 0.5 mV span
            "Vpoint2.t=0\r\nVpoint2.V=0\r\n"
        )
        body = hdr + "DATA\r\n    10    0    0    1\r\n"
        for i in range(10):
            v = -50.0 - i * 0.05
            body += f"{i}\t{v}\t0.0\t-5000.0\r\n"
        f = tmp_path / "short_sweep.VERT"
        f.write_bytes(body.encode())
        # Default threshold (1 mV) would classify this as time trace
        spec_default = read_spec_file(f)
        assert spec_default.metadata["sweep_type"] == "time_trace"
        # With tighter threshold (0.1 mV) it's a sweep
        spec_tight = read_spec_file(f, time_trace_threshold_mv=0.1)
        assert spec_tight.metadata["sweep_type"] == "bias_sweep"


# ─── unit-conversion validation against real instrument files ─────────────────

class TestUnitConversionTT50mV:
    """A180201.124928.VERT — time trace at -50 mV, 1 kHz, feedback off."""

    def test_sweep_type(self, tt_50mv_spec):
        assert tt_50mv_spec.metadata["sweep_type"] == "time_trace"

    def test_n_points(self, tt_50mv_spec):
        assert tt_50mv_spec.metadata["n_points"] == 5000

    def test_x_array_seconds(self, tt_50mv_spec):
        assert tt_50mv_spec.x_array[0] == pytest.approx(0.0)
        assert tt_50mv_spec.x_array[-1] == pytest.approx(4.999, rel=1e-3)

    def test_bias_constant_50mv(self, tt_50mv_spec):
        v = tt_50mv_spec.channels["V"]
        assert v.mean() == pytest.approx(-0.050, abs=1e-4)
        assert v.max() - v.min() < 1e-6

    def test_z_channel_zero(self, tt_50mv_spec):
        # Feedback off during time trace — Z column is all DAC zeros, so z_m == 0.
        z = tt_50mv_spec.channels["Z"]
        assert np.all(z == 0.0)

    def test_current_magnitude(self, tt_50mv_spec):
        # Expected ~-80 pA; allow 5% tolerance.
        i = tt_50mv_spec.channels["I"]
        assert i.mean() == pytest.approx(-8.0e-11, rel=0.05)

    def test_current_negative(self, tt_50mv_spec):
        assert np.all(tt_50mv_spec.channels["I"] < 0)


class TestUnitConversionTT450mV:
    """A180208.194656.M0003.VERT — time trace at -450 mV, 1 kHz, feedback off."""

    def test_sweep_type(self, tt_450mv_spec):
        assert tt_450mv_spec.metadata["sweep_type"] == "time_trace"

    def test_bias_constant_450mv(self, tt_450mv_spec):
        v = tt_450mv_spec.channels["V"]
        assert v.mean() == pytest.approx(-0.450, abs=1e-4)
        assert v.max() - v.min() < 1e-6

    def test_z_channel_zero(self, tt_450mv_spec):
        z = tt_450mv_spec.channels["Z"]
        assert np.all(z == 0.0)

    def test_current_mean(self, tt_450mv_spec):
        # Expected ~-340 pA mean; allow 5% tolerance.
        i = tt_450mv_spec.channels["I"]
        assert i.mean() == pytest.approx(-3.4e-10, rel=0.05)

    def test_current_range(self, tt_450mv_spec):
        # Data spans roughly -356 pA to -240 pA (telegraph noise).
        i = tt_450mv_spec.channels["I"]
        assert i.min() < -3.0e-10
        assert i.max() > -2.8e-10

    def test_current_negative(self, tt_450mv_spec):
        assert np.all(tt_450mv_spec.channels["I"] < 0)


# ─── Synthetic DAC-scale and sign validation ─────────────────────────────────

def _make_synthetic_vert(tmp_path, z_dac: float, i_dac: float, bias_mv: float = -50.0):
    """Build a minimal 5-row .VERT file with known raw DAC column values."""
    n_rows = 5
    hdr = (
        "[ParVERT30]\r\n"
        "DAC-Type=20bit\r\n"
        "GainPre 10^=9\r\n"
        "Dacto[A]xy=0.00083\r\n"
        "Dacto[A]z=0.00018\r\n"
        "OffsetX=0\r\nOffsetY=0\r\n"
        "SpecFreq=1000\r\n"
        f"Vpoint0.t=0\r\nVpoint0.V={bias_mv}\r\n"
        f"Vpoint1.t={n_rows}\r\nVpoint1.V={bias_mv}\r\n"
        "Vpoint2.t=0\r\nVpoint2.V=0\r\n"
    )
    body = hdr + f"DATA\r\n    {n_rows}    0    0    1\r\n"
    for i in range(n_rows):
        body += f"{i}\t{bias_mv}\t{z_dac}\t{i_dac}\r\n"
    f = tmp_path / "synthetic.VERT"
    f.write_bytes(body.encode())
    return f


def _createc_vert_text(
    *,
    version: str = "ParVERT30",
    n_rows: int,
    channel_code: int,
    marker: int | None = None,
    extra_header: str = "",
    rows: list[list[float]],
) -> str:
    """Build a minimal synthetic Createc VERT fixture."""
    hdr = (
        f"[{version}]\r\n"
        "DAC-Type=20bit\r\n"
        "GainPre 10^=9\r\n"
        "Dacto[A]xy=0.00083\r\n"
        "Dacto[A]z=0.00018\r\n"
        "OffsetX=0\r\nOffsetY=0\r\n"
        "SpecFreq=1000\r\n"
        "Vpoint0.t=0\r\nVpoint0.V=-50.0\r\n"
        "Vpoint1.t=10\r\nVpoint1.V=-60.0\r\n"
        f"{extra_header}"
    )
    params = f"    {n_rows}    0    0    {channel_code}"
    if marker is not None:
        params += f"    0    0    {marker}"
    body = hdr + f"DATA\r\n{params}\r\n"
    for row in rows:
        body += "\t".join(str(v) for v in row) + "\r\n"
    return body


def _write_createc_vert(tmp_path, name: str, text: str) -> Path:
    f = tmp_path / name
    f.write_bytes(text.encode("latin-1"))
    return f


class TestSyntheticDACConversion:
    """Verify 2× DAC scale and sign convention with known raw values (#2/#22)."""

    def test_z_dac_to_metres(self, tmp_path):
        # Createc labels this Dacto field "[A]", but it behaves as nm/DAC.
        # Dacto[A]z=0.00018 nm/DAC → 1.8e-13 m/DAC; z_dac=100 → 1.8e-11 m.
        f = _make_synthetic_vert(tmp_path, z_dac=100.0, i_dac=0.0)
        spec = read_spec_file(f)
        expected_z = 100.0 * 0.00018 * 1e-9  # 1.8e-11 m
        assert spec.channels["Z"].mean() == pytest.approx(expected_z, rel=1e-6)

    def test_i_dac_sign_and_scale(self, tmp_path):
        # 20-bit DAC: vpd = (10/2^20)*2 = 20/2^20 V/DAC; gain=10^9 V/A
        # is_ = vpd/gain = 20/(2^20 * 1e9) ≈ 1.9073e-14 A/DAC
        # i_dac=-5000 → current ≈ -9.537e-11 A (negative)
        import math
        f = _make_synthetic_vert(tmp_path, z_dac=0.0, i_dac=-5000.0)
        spec = read_spec_file(f)
        vpd = (10.0 / 2**20) * 2
        expected_i = -5000.0 * vpd / 1e9
        assert spec.channels["I"].mean() == pytest.approx(expected_i, rel=1e-6)
        assert spec.channels["I"].mean() < 0

    def test_positive_i_dac_gives_positive_current(self, tmp_path):
        f = _make_synthetic_vert(tmp_path, z_dac=0.0, i_dac=3000.0)
        spec = read_spec_file(f)
        assert spec.channels["I"].mean() > 0


# ─── parse_spec_header edge cases ────────────────────────────────────────────

class TestParseSpecHeaderEdgeCases:
    def test_single_row_parse(self, tmp_path):
        f = tmp_path / "tiny.VERT"
        f.write_bytes(b"key=val\r\nDATA\r\n")
        hdr = parse_spec_header(f)
        assert hdr["key"] == "val"

    def test_truncated_file_raises(self, tmp_path):
        f = tmp_path / "trunc.VERT"
        f.write_bytes(b"key=val\r\nno_data_marker_here")
        with pytest.raises(ValueError, match="DATA"):
            parse_spec_header(f)

    def test_latin1_non_ascii_round_trip(self, tmp_path):
        # Å (0xC5) appears in Createc headers as the unit for angstrom.
        f = tmp_path / "latin1.VERT"
        f.write_bytes("Dacto[\xc5]xy=0.00083\r\nDATA\r\n".encode("latin-1"))
        hdr = parse_spec_header(f)
        assert "Dacto[\xc5]xy" in hdr or "Dacto[Å]xy" in hdr


# ─── z_scale_m_per_dac fallback path ─────────────────────────────────────────

class TestZScaleFallback:
    """z_scale_m_per_dac falls back to GainZ * ZPiezoconst when Dacto[A]z is absent."""

    def test_z_scale_fallback_no_dacto_z(self, tmp_path):
        hdr_text = (
            "[ParVERT30]\r\n"
            "DAC-Type=20bit\r\n"
            "GainPre 10^=9\r\n"
            "Dacto[A]xy=0.00083\r\n"
            # No Dacto[A]z — triggers fallback to GainZ * ZPiezoconst
            "GainZ=10\r\n"
            "ZPiezoconst=19.2\r\n"
            "OffsetX=0\r\nOffsetY=0\r\n"
            "SpecFreq=1000\r\n"
            "Vpoint0.t=0\r\nVpoint0.V=-50\r\n"
            "Vpoint1.t=5\r\nVpoint1.V=-50\r\n"
            "Vpoint2.t=0\r\nVpoint2.V=0\r\n"
            "DATA\r\n    5    0    0    1\r\n"
        )
        rows = "".join(f"{i}\t-50.0\t0.0\t0.0\r\n" for i in range(5))
        f = tmp_path / "noz.VERT"
        f.write_bytes((hdr_text + rows).encode())
        spec = read_spec_file(f)
        # Z column is all zero so z_m must be zero regardless of scale.
        assert np.all(spec.channels["Z"] == 0.0)
        # Metadata should parse without error.
        assert spec.metadata["n_points"] == 5


# ─── bias_sweep absolute current magnitude ───────────────────────────────────

class TestBiasSweepCurrentMagnitude:
    def test_i_magnitude_reasonable(self, bias_sweep_spec):
        # I(V) on a metal surface at -300 mV: order-of-magnitude ~10 pA–10 nA.
        i = bias_sweep_spec.channels["I"]
        assert float(np.abs(i).max()) > 1e-12
        assert float(np.abs(i).max()) < 1e-6


# ─── channel_order / default_channels ────────────────────────────────────────

class TestChannelMetadata:
    def test_channel_order_nonempty(self, bias_sweep_spec, time_trace_spec):
        assert bias_sweep_spec.channel_order == ["I", "Z", "V"]
        assert time_trace_spec.channel_order == ["I", "Z", "V"]

    def test_default_channels_nonempty(self, bias_sweep_spec, time_trace_spec):
        assert len(bias_sweep_spec.default_channels) >= 1
        assert len(time_trace_spec.default_channels) >= 1

    def test_defaults_reference_valid_channels(self, bias_sweep_spec, time_trace_spec):
        for ch in bias_sweep_spec.default_channels:
            assert ch in bias_sweep_spec.channels
        for ch in time_trace_spec.default_channels:
            assert ch in time_trace_spec.channels

    def test_additional_decoded_channels_reach_specdata(self, tmp_path):
        f = _write_createc_vert(
            tmp_path,
            "extra_channels.VERT",
            _createc_vert_text(
                n_rows=2,
                channel_code=0b1011,
                rows=[
                    [0, -50.0, 0.0, -1000.0, 1.25, 2.5],
                    [1, -60.0, 0.0, -2000.0, 1.50, 3.5],
                ],
            ),
        )
        spec = read_spec_file(f)
        assert spec.channel_order == ["I", "Z", "V", "dI/dV", "ADC0"]
        assert spec.default_channels == ["I"]
        assert set(spec.channels) == {"I", "Z", "V", "dI/dV", "ADC0"}
        assert spec.y_units["dI/dV"] == "unknown"
        assert spec.y_units["ADC0"] == "unknown"
        assert set(spec.channel_info) == set(spec.channel_order)
        assert spec.channel_info["dI/dV"].source_name == "dI/dV"
        assert "lockin_derivative" in spec.channel_info["dI/dV"].roles
        assert spec.metadata["source_channels"] == ["V", "Z", "I", "dI/dV", "ADC0"]
        assert spec.metadata["channel_roles"]["ADC0"] == ["auxiliary"]
        np.testing.assert_allclose(spec.channels["dI/dV"], [1.25, 1.50])
        np.testing.assert_allclose(spec.channels["ADC0"], [2.5, 3.5])
        assert "source" in spec.metadata

    def test_metadata_tracks_additional_decoded_channels(self, tmp_path):
        f = _write_createc_vert(
            tmp_path,
            "extra_metadata.VERT",
            _createc_vert_text(
                n_rows=2,
                channel_code=0b1011,
                rows=[
                    [0, -50.0, 0.0, -1000.0, 1.25, 2.5],
                    [1, -60.0, 0.0, -2000.0, 1.50, 3.5],
                ],
            ),
        )
        spec = read_spec_file(f)
        meta = read_spec_metadata(f)
        assert meta.channels == tuple(spec.channel_order)
        assert meta.units == tuple(spec.y_units[ch] for ch in spec.channel_order)
        assert tuple(ch.key for ch in meta.channel_info) == tuple(spec.channel_order)
        assert meta.metadata["source_channels"] == spec.metadata["source_channels"]
        assert meta.metadata["channel_roles"] == spec.metadata["channel_roles"]
        assert meta.metadata["n_points"] == spec.metadata["n_points"]
        assert meta.metadata["source"]["sha256"] == spec.metadata["source"]["sha256"]


class TestMeasurementInterpretation:
    def test_bias_sweep_defaults_to_sts(self, bias_sweep_spec):
        assert bias_sweep_spec.metadata["sweep_type"] == "bias_sweep"
        assert bias_sweep_spec.metadata["measurement_family"] == "sts"
        assert bias_sweep_spec.metadata["derivative_label"] is None

    def test_time_trace_defaults_to_sts(self, time_trace_spec):
        assert time_trace_spec.metadata["sweep_type"] == "time_trace"
        assert time_trace_spec.metadata["measurement_family"] == "sts"

    def test_feedback_lockin_vert_infers_iz_without_changing_x_axis(self, tmp_path):
        f = _write_createc_vert(
            tmp_path,
            "iz_like.VERT",
            _createc_vert_text(
                version="ParVERT32",
                n_rows=2,
                channel_code=0b11,
                marker=3,
                extra_header=(
                    "FBOff=0\r\n"
                    "VertFBMode=1\r\n"
                    "Zpoint0.t=0\r\nZpoint0.z=0\r\n"
                    "Zpoint1.t=0\r\nZpoint1.z=0\r\n"
                ),
                rows=[
                    [0, 2500.0, 0.0, 0.0, 5000.0, 1.25],
                    [1, 9000.0, 0.0, 0.0, 6000.0, 1.50],
                ],
            ),
        )

        spec = read_spec_file(f)
        meta = read_spec_metadata(f)

        assert spec.metadata["sweep_type"] == "bias_sweep"
        assert spec.x_unit == "V"
        assert spec.metadata["measurement_family"] == "iz"
        assert spec.metadata["derivative_label"] == "dI/dz"
        assert meta.metadata["measurement_family"] == "iz"

    def test_anonymized_didz_fixture_preserves_all_channels(self):
        spec = read_spec_file(VERT_DIDZ_FIXTURE)
        meta = read_spec_metadata(VERT_DIDZ_FIXTURE)

        assert spec.metadata["sweep_type"] == "bias_sweep"
        assert spec.metadata["measurement_family"] == "iz"
        assert spec.metadata["derivative_label"] == "dI/dz"
        assert spec.metadata["height_channel"] == "Z feedback"
        assert spec.metadata["height_source_channel"] == "Raw column 9"
        assert spec.metadata["z_command_channel"] == "Z command"
        assert spec.channel_order == [
            "I",
            "Z feedback",
            "dI/dV",
            "di_q",
            "V",
            "Z command",
            "X",
            "ADC0",
            "NA02",
        ]
        assert "Raw column 9" not in spec.channels
        assert "Z" not in spec.channels
        assert spec.y_units["Z feedback"] == "DAC"
        assert spec.y_units["Z command"] == "m"
        assert spec.channel_info["Z feedback"].source_name == "Raw column 9"
        assert spec.channel_info["Z feedback"].display_label == "Raw column 9 - Z feedback"
        assert "z_feedback" in spec.channel_info["Z feedback"].roles
        assert spec.channel_info["Z command"].source_name == "Z"
        assert spec.channel_info["Z command"].display_label == "Z - command"
        assert "z_command" in spec.channel_info["Z command"].roles
        assert spec.metadata["channel_roles"]["Z feedback"] == [
            "z_feedback",
            "height_counts",
        ]
        assert spec.metadata["source_channels"] == [
            "V",
            "Z",
            "X",
            "I",
            "dI/dV",
            "ADC0",
            "NA02",
            "di_q",
            "Raw column 9",
        ]
        assert np.ptp(spec.channels["Z feedback"]) > 0
        assert np.allclose(spec.channels["Z command"], 0.0)
        assert meta.channels == tuple(spec.channel_order)
        assert meta.units == tuple(spec.y_units[ch] for ch in spec.channel_order)
        assert {ch.key: ch.source_name for ch in meta.channel_info}["Z feedback"] == "Raw column 9"

    def test_measurement_mode_override_takes_precedence(self, tmp_path):
        f = _write_createc_vert(
            tmp_path,
            "override.VERT",
            _createc_vert_text(
                n_rows=2,
                channel_code=1,
                rows=[
                    [0, -50.0, 0.0, -1000.0],
                    [1, -60.0, 0.0, -2000.0],
                ],
            ),
        )

        sts = read_spec_file(f, measurement_mode="sts")
        iz = read_spec_metadata(f, measurement_mode="iz")

        assert sts.metadata["measurement_family"] == "sts"
        assert sts.metadata["measurement_confidence"] == "override"
        assert iz.metadata["measurement_family"] == "iz"
        assert iz.metadata["derivative_label"] == "dI/dz"

    def test_invalid_measurement_mode_raises(self, tmp_path):
        f = _write_createc_vert(
            tmp_path,
            "invalid_mode.VERT",
            _createc_vert_text(
                n_rows=1,
                channel_code=1,
                rows=[[0, -50.0, 0.0, -1000.0]],
            ),
        )

        with pytest.raises(ValueError, match="measurement_mode"):
            read_spec_file(f, measurement_mode="not-a-mode")

    def test_spec_info_json_includes_measurement_metadata(self, capsys):
        from probeflow.cli import _cmd_spec_info

        rc = _cmd_spec_info(
            SimpleNamespace(input=VERT_BIAS_SWEEP, json=True, verbose=False)
        )
        out = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert out["measurement_family"] == "sts"
        assert "feedback_mode" in out
        assert "measurement_evidence" in out
        assert "channel_info" in out
        assert out["channel_info"][0]["key"] == out["channels"][0]
        assert "source_name" in out["channel_info"][0]
