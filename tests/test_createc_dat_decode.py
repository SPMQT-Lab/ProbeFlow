"""Tests for the low-level Createc DAT decode/report layer."""

from __future__ import annotations

import pytest

from probeflow.metadata import read_scan_metadata
from probeflow.readers.createc_dat import read_createc_dat_report
from probeflow.scan import load_scan


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

    two_channel = [
        path
        for path in sample_dat_files
        if (
            read_createc_dat_report(path, include_raw=False).detected_channel_count
            == 2
        )
    ]
    assert two_channel, "sample data should include a legacy 2-channel DAT"


def test_metadata_uses_createc_report_without_constructing_scan(
    first_sample_dat, monkeypatch
):
    def fail_load_scan(_path):
        raise AssertionError("DAT metadata should not construct a full Scan")

    monkeypatch.setattr("probeflow.scan.load_scan", fail_load_scan)
    meta = read_scan_metadata(first_sample_dat)

    assert meta.source_format == "createc_dat"
    assert meta.shape is not None
    assert meta.shape[1] == int(meta.raw_header["Num.X"])


def test_missing_data_marker_raises_createc_error(tmp_path):
    bad = tmp_path / "bad.dat"
    bad.write_bytes(b"[Paramco32]\nNum.X=2\nNum.Y=2\n")

    with pytest.raises(ValueError, match="missing DATA marker"):
        read_createc_dat_report(bad)
