"""Tests for the low-level Createc DAT decode/report layer."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pytest

from probeflow.io.converters.createc_dat_to_sxm import convert_dat_to_sxm
from probeflow.io.createc_interpretation import createc_dat_experiment_metadata
from probeflow.core.metadata import read_scan_metadata
from probeflow.io.readers.createc_dat import (
    read_createc_dat_report,
    scale_channels_for_scan,
)
from probeflow.core.scan_loader import load_scan

TESTDATA = Path(__file__).resolve().parents[1] / "anonymised_testdata"
QPLUS_10CH_DAT = TESTDATA / "createc_scan_qplus_10ch_afm.dat"


def test_report_records_trim_first_column_and_tail(first_sample_dat):
    report = read_createc_dat_report(first_sample_dat)

    assert report.original_Nx > report.decoded_Nx
    assert report.decoded_Nx == report.original_Nx - 1
    assert report.trimmed_Ny == report.decoded_Ny
    assert report.first_column_removed is True
    assert report.ignored_tail_float_count >= 0
    assert report.raw_channels_dac is not None
    assert report.raw_channels_dac.shape == (
        report.detected_channel_count,
        report.decoded_Ny,
        report.decoded_Nx,
    )


def test_report_header_only_omits_arrays_but_keeps_corrected_shape(first_sample_dat):
    full = load_scan(first_sample_dat)
    report = read_createc_dat_report(first_sample_dat, include_raw=False)

    assert report.raw_channels_dac is None
    assert (report.decoded_Ny, report.decoded_Nx) == full.planes[0].shape
    assert int(report.header["Num.X"]) == report.decoded_Nx
    assert int(report.header["Num.Y"]) == report.decoded_Ny


def test_report_preserves_legacy_channel_detection_order(sample_dat_files):
    """Trailing floats must not make legacy 2-channel files look like 4+."""

    four_channel = [
        path
        for path in sample_dat_files
        if (
            read_createc_dat_report(path, include_raw=False).detected_channel_count
            == 4
        )
    ]
    two_channel = [
        path
        for path in sample_dat_files
        if (
            read_createc_dat_report(path, include_raw=False).detected_channel_count
            == 2
        )
    ]
    assert four_channel, "sample data should include a legacy 4-channel DAT"
    assert two_channel, "sample data should include a legacy 2-channel DAT"


def test_scale_channels_for_scan_applies_channel_scale_factors(first_sample_dat):
    report = read_createc_dat_report(first_sample_dat)
    scaled = scale_channels_for_scan(report)

    assert report.raw_channels_dac is not None
    assert np.allclose(
        scaled[0],
        report.raw_channels_dac[0] * report.scale_factors["z_m_per_dac"],
    )
    assert np.allclose(
        scaled[1],
        report.raw_channels_dac[1] * report.scale_factors["current_a_per_dac"],
    )


def test_2_channel_dat_roundtrip_preserves_synthetic_backward(
    sample_dat_files, tmp_path, cushion_dir
):
    two_channel = next(
        path
        for path in sample_dat_files
        if (
            read_createc_dat_report(path, include_raw=False).detected_channel_count
            == 2
        )
    )

    direct = load_scan(two_channel)
    convert_dat_to_sxm(two_channel, tmp_path, cushion_dir)
    converted = load_scan(next(tmp_path.glob("*.sxm")))

    assert np.array_equal(direct.planes[1], converted.planes[1])
    assert np.array_equal(direct.planes[3], converted.planes[3])


def test_exact_non_stm_channel_count_and_raw_fallback(tmp_path):
    dat = tmp_path / "nine_channels.dat"
    header = b"[Paramco32]\nNum.X=2\nNum.Y=2\n"
    payload = np.arange(1, 37, dtype="<f4").tobytes()
    dat.write_bytes(header + b"DATA" + zlib.compress(payload))

    report = read_createc_dat_report(dat, include_raw=False)

    assert report.detected_channel_count == 9
    assert report.channel_info[-1].name == "Raw channel 8"
    assert report.channel_info[-1].unit == "DAC"


def test_header_declared_channel_count_wins_when_payload_fits(tmp_path):
    dat = tmp_path / "ten_channels.dat"
    header = (
        b"[Paramco32]\n"
        b"Num.X=2\n"
        b"Num.Y=2\n"
        b"Channels=10\n"
        b"Channelselectval=899\n"
    )
    payload = np.arange(1, 44, dtype="<f4").tobytes()
    dat.write_bytes(header + b"DATA" + zlib.compress(payload))

    report = read_createc_dat_report(dat)

    assert report.detected_channel_count == 10
    assert report.ignored_tail_float_count == 3
    assert report.raw_channels_dac is not None
    assert report.raw_channels_dac.shape == (10, 2, 1)
    assert [info.name for info in report.channel_info] == [
        "Z forward",
        "Current forward",
        "Aux6 forward",
        "Aux7 forward",
        "Aux8 forward",
        "Z backward",
        "Current backward",
        "Aux6 backward",
        "Aux7 backward",
        "Aux8 backward",
    ]


def test_selected_two_plane_dat_stays_native_without_synthetic_current(tmp_path):
    dat = tmp_path / "two_z_planes.dat"
    header = (
        b"[Paramco32]\n"
        b"Num.X=3\n"
        b"Num.Y=2\n"
        b"Channels=2\n"
        b"Channelselectval=1\n"
        b"Length x[A]=10\n"
        b"Length y[A]=10\n"
    )
    payload = np.arange(12, dtype="<f4").tobytes()
    dat.write_bytes(header + b"DATA" + zlib.compress(payload))

    report = read_createc_dat_report(dat)
    scan = load_scan(dat)
    meta = read_scan_metadata(dat)

    assert [info.name for info in report.channel_info] == [
        "Z forward",
        "Z backward",
    ]
    assert scan.plane_names == ["Z forward", "Z backward"]
    assert scan.plane_units == ["m", "m"]
    assert scan.plane_synthetic == [False, False]
    assert meta.plane_names == tuple(scan.plane_names)
    assert meta.units == tuple(scan.plane_units)


def test_selected_four_plane_non_stm_dat_stays_native(tmp_path):
    dat = tmp_path / "z_aux_planes.dat"
    header = (
        b"[Paramco32]\n"
        b"Num.X=3\n"
        b"Num.Y=2\n"
        b"Channels=4\n"
        b"Channelselectval=129\n"
        b"Length x[A]=10\n"
        b"Length y[A]=10\n"
    )
    payload = np.arange(24, dtype="<f4").tobytes()
    dat.write_bytes(header + b"DATA" + zlib.compress(payload))

    scan = load_scan(dat)

    assert scan.plane_names == [
        "Z forward",
        "Aux6 forward",
        "Z backward",
        "Aux6 backward",
    ]
    assert scan.plane_units == ["m", "DAC", "m", "DAC"]
    assert scan.plane_synthetic == [False, False, False, False]


def test_implausible_header_channel_count_falls_back_to_payload(tmp_path):
    dat = tmp_path / "bad_count.dat"
    header = b"[Paramco32]\nNum.X=2\nNum.Y=2\nChannels=10\n"
    payload = np.arange(1, 17, dtype="<f4").tobytes()
    dat.write_bytes(header + b"DATA" + zlib.compress(payload))

    report = read_createc_dat_report(dat, include_raw=False)

    assert report.detected_channel_count == 4


def test_anonymized_qplus_fixture_decodes_all_10_channels():
    report = read_createc_dat_report(QPLUS_10CH_DAT)
    scan = load_scan(QPLUS_10CH_DAT)

    assert report.detected_channel_count == 10
    assert scan.n_planes == 10
    assert report.ignored_tail_float_count == 1056
    assert scan.plane_names == [
        "Z forward",
        "Current forward",
        "Aux6 forward",
        "Aux7 forward",
        "Aux8 forward",
        "Z backward",
        "Current backward",
        "Aux6 backward",
        "Aux7 backward",
        "Aux8 backward",
    ]


def test_dat_to_sxm_rejects_noncanonical_multichannel_dat(tmp_path, cushion_dir):
    with pytest.raises(ValueError, match="canonical STM"):
        convert_dat_to_sxm(QPLUS_10CH_DAT, tmp_path, cushion_dir)


def test_metadata_uses_createc_report_without_constructing_scan(
    first_sample_dat, monkeypatch
):
    def fail_load_scan(_path):
        raise AssertionError("DAT metadata should not construct a full Scan")

    monkeypatch.setattr("probeflow.core.scan_loader.load_scan", fail_load_scan)
    meta = read_scan_metadata(first_sample_dat)

    assert meta.source_format == "createc_dat"
    assert meta.shape is not None
    assert meta.shape[1] == int(meta.raw_header["Num.X"])


def test_missing_data_marker_raises_createc_error(tmp_path):
    bad = tmp_path / "bad.dat"
    bad.write_bytes(b"[Paramco32]\nNum.X=2\nNum.Y=2\n")

    with pytest.raises(ValueError, match="missing DATA marker"):
        read_createc_dat_report(bad)


def test_createc_qplus_header_infers_afm_topography():
    meta = createc_dat_experiment_metadata(
        {
            "PLLOn": "1",
            "MEMO_STMAFM": "bias cable removed and grounded with qPlus sensor",
            "FBChannel": "4",
        }
    )

    assert meta["acquisition_mode"] == "afm"
    assert meta["feedback_mode"] == "constant_frequency_shift"
    assert meta["topography_role"] == "afm_topography"
    assert meta["feedback_channel"] == 4
    assert meta["confidence"] == "high"


def test_createc_stm_header_infers_current_feedback():
    meta = createc_dat_experiment_metadata(
        {
            "PLLOn": "0",
            "MEMO_STMAFM": "",
            "FBChannel": "0",
        }
    )

    assert meta["acquisition_mode"] == "stm"
    assert meta["feedback_mode"] == "current"
    assert meta["topography_role"] == "stm_topography"


def test_createc_ambiguous_header_stays_unknown():
    meta = createc_dat_experiment_metadata({"FBChannel": "0"})

    assert meta["acquisition_mode"] == "unknown"
    assert meta["feedback_mode"] == "unknown"
    assert meta["confidence"] == "low"
