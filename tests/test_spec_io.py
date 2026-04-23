"""Tests for probeflow.spec_io — Createc .VERT file reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probeflow.spec_io import SpecData, parse_spec_header, read_spec_file

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

VERT_TIME_TRACE = DATA_DIR / "A180201.152542.M0001.VERT"
VERT_BIAS_SWEEP = DATA_DIR / "A180201.151737.M0001.VERT"


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def time_trace_spec():
    return read_spec_file(VERT_TIME_TRACE)


@pytest.fixture(scope="module")
def bias_sweep_spec():
    return read_spec_file(VERT_BIAS_SWEEP)


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
        assert 0 < val < 1.0  # Å/DAC, physically reasonable

    def test_offset_xy_present(self):
        hdr = parse_spec_header(VERT_TIME_TRACE)
        assert "OffsetX" in hdr
        assert "OffsetY" in hdr

    def test_spec_freq_present(self):
        hdr = parse_spec_header(VERT_TIME_TRACE)
        assert "SpecFreq" in hdr
        assert float(hdr["SpecFreq"]) > 0


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
        # Z should be in metres; 77 K STM is typically < 1 nm range
        assert z.min() > -20e-10  # >-20 Å
        assert z.max() < 20e-10   # <+20 Å

    def test_position_is_tuple_of_floats(self, time_trace_spec):
        px, py = time_trace_spec.position
        assert isinstance(px, float)
        assert isinstance(py, float)

    def test_position_in_metres(self, time_trace_spec):
        # OffsetX=3823 DAC, Dacto[A]xy=0.00083 → ~3.17e-10 m
        # OffsetY=-91743 DAC → ~-7.61e-9 m
        px, py = time_trace_spec.position
        assert abs(px) < 1e-6  # within 1 µm of centre
        assert abs(py) < 1e-6

    def test_y_units_dict(self, time_trace_spec):
        assert time_trace_spec.y_units["I"] == "A"
        assert time_trace_spec.y_units["Z"] == "m"

    def test_bias_constant(self, time_trace_spec):
        # For time trace, voltage should be constant
        v = time_trace_spec.channels["V"]
        assert v.max() - v.min() < 1e-3  # less than 1 mV variation


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
        # sweep from -50 mV to -300 mV
        x = bias_sweep_spec.x_array
        assert x.min() == pytest.approx(-0.300, abs=0.01)
        assert x.max() == pytest.approx(-0.050, abs=0.01)

    def test_channels_present(self, bias_sweep_spec):
        for ch in ("I", "Z", "V"):
            assert ch in bias_sweep_spec.channels

    def test_z_varies(self, bias_sweep_spec):
        z = bias_sweep_spec.channels["Z"]
        z_range = float(z.max() - z.min())
        assert z_range > 0


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
