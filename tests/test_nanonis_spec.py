"""Tests for the Nanonis .dat point-spectroscopy reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probeflow.io.readers.nanonis_spec import read_nanonis_spec
from probeflow.io.spectroscopy import SpecMetadata, read_spec_file, read_spec_metadata


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

    def test_channel_info_preserves_source_labels_and_roles(self, sts_spec):
        assert set(sts_spec.channel_info) == set(sts_spec.channel_order)
        bias = sts_spec.channel_info["Bias calc"]
        current = sts_spec.channel_info["Current [AVG]"]
        lockin = sts_spec.channel_info["LockIn [AVG]"]

        assert bias.source_name == "Bias calc"
        assert bias.source_label.startswith("Bias calc")
        assert bias.unit == sts_spec.y_units["Bias calc"]
        assert "bias_axis" in bias.roles
        assert current.source_label.startswith("Current [AVG]")
        assert "current" in current.roles
        assert "lockin_derivative" in lockin.roles
        assert sts_spec.metadata["channel_roles"]["LockIn [AVG]"] == [
            "lockin_derivative"
        ]
        assert "Bias calc" in sts_spec.metadata["source_channels"]

    def test_duplicate_column_names_are_uniquified(self, tmp_path):
        f = tmp_path / "duplicate_columns.dat"
        f.write_text(
            "Experiment\tBias Spectroscopy\n"
            "[DATA]\n"
            "Bias (V)\tSignal (A)\tSignal (A)\n"
            "0\t1\t2\n"
            "1\t3\t4\n",
            encoding="latin-1",
        )

        spec = read_spec_file(f)
        meta = read_spec_metadata(f)

        assert spec.channel_order == ["Bias", "Signal", "Signal 2"]
        np.testing.assert_allclose(spec.channels["Signal"], [1, 3])
        np.testing.assert_allclose(spec.channels["Signal 2"], [2, 4])
        assert spec.channel_info["Signal"].source_name == "Signal"
        assert spec.channel_info["Signal 2"].source_name == "Signal"
        assert spec.channel_info["Signal 2"].source_label == "Signal (A)"
        assert meta.channels == tuple(spec.channel_order)
        assert tuple(ch.source_name for ch in meta.channel_info) == (
            "Bias",
            "Signal",
            "Signal",
        )


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
        assert tuple(ch.key for ch in meta.channel_info) == tuple(sts_spec.channel_order)
        assert {
            ch.key: ch.source_label
            for ch in meta.channel_info
        }["Current [AVG]"].startswith("Current [AVG]")

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

    def test_metadata_rejects_non_numeric_rows(self, tmp_path):
        f = tmp_path / "bad_numeric.dat"
        f.write_text(
            "Experiment\tBias Spectroscopy\n"
            "[DATA]\n"
            "Bias (V)\tCurrent (A)\n"
            "0\t1\n"
            "1\tnope\n",
            encoding="latin-1",
        )

        with pytest.raises(ValueError, match="failed to parse data row"):
            read_spec_metadata(f)

    def test_metadata_rejects_wrong_column_count(self, tmp_path):
        f = tmp_path / "bad_columns.dat"
        f.write_text(
            "Experiment\tBias Spectroscopy\n"
            "[DATA]\n"
            "Bias (V)\tCurrent (A)\n"
            "0\t1\n"
            "1\t2\t3\n",
            encoding="latin-1",
        )

        with pytest.raises(ValueError, match="expected 2"):
            read_spec_metadata(f)
