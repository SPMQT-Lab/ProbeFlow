"""Tests for the Nanonis .dat point-spectroscopy reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probeflow.readers.nanonis_spec import read_nanonis_spec
from probeflow.spec_io import SpecMetadata, read_spec_file, read_spec_metadata


TESTDATA = Path(__file__).resolve().parents[1] / "anonymised_testdata"

KELVIN = TESTDATA / "nanonis_kelvin_parabola_500mv.dat"
STS    = TESTDATA / "nanonis_sts_15mv.dat"


@pytest.fixture(scope="module")
def kelvin_spec():
    return read_nanonis_spec(KELVIN)


@pytest.fixture(scope="module")
def sts_spec():
    return read_nanonis_spec(STS)


class TestParseBasics:
    def test_kelvin_parses(self, kelvin_spec):
        assert kelvin_spec.metadata["n_points"] > 0

    def test_sts_parses(self, sts_spec):
        assert sts_spec.metadata["n_points"] > 0

    def test_kelvin_sweep_type(self, kelvin_spec):
        assert kelvin_spec.metadata["sweep_type"] == "bias_sweep"

    def test_sts_sweep_type(self, sts_spec):
        assert sts_spec.metadata["sweep_type"] == "bias_sweep"


class TestChannels:
    def test_kelvin_has_bias_and_current(self, kelvin_spec):
        assert "Bias calc" in kelvin_spec.channels
        assert "Current" in kelvin_spec.channels
        # Any secondary channel — Kelvin file has OC M1 Freq. Shift and Input 6.
        secondary = {"OC M1 Freq. Shift", "Input 6"}
        assert secondary & set(kelvin_spec.channels)

    def test_sts_has_bias_and_current_avg(self, sts_spec):
        assert "Bias calc" in sts_spec.channels
        assert "Current [AVG]" in sts_spec.channels
        assert "LockIn [AVG]" in sts_spec.channels


class TestXAxis:
    def test_kelvin_x_is_bias_v(self, kelvin_spec):
        assert kelvin_spec.x_unit == "V"
        assert "Bias" in kelvin_spec.x_label

    def test_sts_x_is_bias_v(self, sts_spec):
        assert sts_spec.x_unit == "V"
        assert "Bias" in sts_spec.x_label

    def test_x_array_matches_column(self, kelvin_spec):
        assert np.allclose(kelvin_spec.x_array, kelvin_spec.channels["Bias calc"])


class TestDefaults:
    def test_kelvin_default_channels_include_current_and_secondary(self, kelvin_spec):
        defaults = kelvin_spec.default_channels
        assert any(name.startswith("Current") for name in defaults)
        # At least one non-current secondary
        assert any(not name.startswith("Current") for name in defaults)

    def test_sts_default_channels_include_current_avg(self, sts_spec):
        defaults = sts_spec.default_channels
        assert any(name.startswith("Current") for name in defaults)

    def test_defaults_reference_real_channels(self, kelvin_spec, sts_spec):
        for spec in (kelvin_spec, sts_spec):
            for name in spec.default_channels:
                assert name in spec.channels


class TestPosition:
    def test_kelvin_position_floats(self, kelvin_spec):
        px, py = kelvin_spec.position
        assert isinstance(px, float)
        assert isinstance(py, float)
        assert px != 0.0
        assert py != 0.0

    def test_sts_position_floats(self, sts_spec):
        px, py = sts_spec.position
        assert isinstance(px, float)
        assert isinstance(py, float)
        assert px != 0.0
        assert py != 0.0


class TestDispatcher:
    def test_read_spec_file_routes_nanonis(self):
        spec = read_spec_file(STS)
        assert spec.metadata["sweep_type"] == "bias_sweep"
        assert "Current [AVG]" in spec.channels

    def test_read_spec_file_routes_createc_vert(self):
        # Existing Createc files should still go through the VERT reader.
        vert = TESTDATA / "createc_ivt_telegraph_300mv_a.VERT"
        if not vert.exists():
            pytest.skip("missing .VERT fixture")
        spec = read_spec_file(vert)
        assert "I" in spec.channels
        assert "Z" in spec.channels

    def test_read_spec_metadata_routes_nanonis(self, sts_spec):
        meta = read_spec_metadata(STS)
        assert isinstance(meta, SpecMetadata)
        assert meta.source_format == "nanonis_dat_spectrum"
        assert meta.metadata["sweep_type"] == sts_spec.metadata["sweep_type"]
        assert meta.metadata["n_points"] == sts_spec.metadata["n_points"]
        assert meta.channels == tuple(sts_spec.channel_order)

    def test_read_spec_metadata_has_position_and_bias(self):
        meta = read_spec_metadata(STS)
        assert meta.position[0] != 0.0
        assert meta.position[1] != 0.0
        assert meta.bias == pytest.approx(-15e-3)

    def test_full_and_metadata_reads_include_matching_source_fingerprint(self, sts_spec):
        meta = read_spec_metadata(STS)
        full_source = sts_spec.metadata["source"]
        metadata_source = meta.metadata["source"]

        assert full_source["source_format"] == "nanonis_dat_spectrum"
        assert full_source["item_type"] == "spectrum"
        assert metadata_source["sha256"] == full_source["sha256"]
        assert len(metadata_source["sha256"]) == 64
        assert metadata_source["file_size_bytes"] > 0
        assert metadata_source["data_offset"] is not None
