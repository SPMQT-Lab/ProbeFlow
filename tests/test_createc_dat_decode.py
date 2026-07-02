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

TESTDATA = Path(__file__).resolve().parents[1] / "test_data"
QPLUS_10CH_DAT = TESTDATA / "createc_scan_qplus_10ch_afm.dat"
CREATEC_SCAN_FIXTURES = sorted(TESTDATA.glob("createc_scan_*.dat"))


def test_report_records_trim_first_column_and_tail(first_sample_dat):
    report = read_createc_dat_report(first_sample_dat)

    assert report.original_Nx > report.decoded_Nx
    assert report.decoded_Nx == report.original_Nx - 1
    assert report.trimmed_Ny == report.decoded_Ny
    assert report.first_column_removed is True
    assert report.ignored_tail_float_count >= 0
    assert report.raw_channels_dac is not None
    assert report.raw_channels_dac is report.decoded_channels_dac
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


def test_4_channel_dat_roundtrip_backward_orientation_is_preserved(
    sample_dat_files, tmp_path, cushion_dir
):
    """Verify the backward-channel round-trip invariant for real 4-channel data.

    Createc stores backward rows in display order; the converter flips them to
    SXM acquisition order; the SXM reader flips them back.  Net result: both
    load paths must produce byte-identical backward planes.
    """
    four_channel = next(
        (
            path
            for path in sample_dat_files
            if (
                read_createc_dat_report(path, include_raw=False).detected_channel_count
                == 4
            )
        ),
        None,
    )
    if four_channel is None:
        pytest.skip("no 4-channel sample .dat available")

    direct = load_scan(four_channel)
    convert_dat_to_sxm(four_channel, tmp_path, cushion_dir)
    converted = load_scan(next(tmp_path.glob("*.sxm")))

    # planes[1] = Z backward, planes[3] = Current backward
    np.testing.assert_array_equal(direct.planes[1], converted.planes[1])
    np.testing.assert_array_equal(direct.planes[3], converted.planes[3])


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


def test_header_two_channels_with_extra_telemetry_tail_stays_two(tmp_path):
    """Regression for review IO #3 — Createc routinely tail-pads .dat
    files with telemetry past the declared image-plane payload.  When
    ``Channels=2`` is set in the header, a tail that makes
    ``payload_float_count >= 4 * Nx * Ny`` must not be re-interpreted
    as two synthetic backward planes; the header is authoritative.
    """
    dat = tmp_path / "two_with_tail.dat"
    header = (
        b"[Paramco32]\n"
        b"Num.X=2\n"
        b"Num.Y=2\n"
        b"Channels=2\n"
        b"Channelselectval=1\n"
    )
    # Two real channels of (2,2) pixels = 8 floats, then 12 telemetry
    # floats so total payload (20 floats) >= 4*Nx*Ny (16 floats).
    n_pixels = 2 * 2
    real_floats = np.arange(1, 2 * n_pixels + 1, dtype="<f4")
    telemetry = np.arange(100, 100 + 12, dtype="<f4")
    payload = np.concatenate([real_floats, telemetry]).tobytes()
    dat.write_bytes(header + b"DATA" + zlib.compress(payload))

    report = read_createc_dat_report(dat, include_raw=False)

    # Must trust the header: 2 channels, not 4 manufactured from telemetry.
    assert report.detected_channel_count == 2
    assert report.ignored_tail_float_count >= 12


@pytest.mark.parametrize("path", CREATEC_SCAN_FIXTURES, ids=lambda p: p.name)
def test_normal_appendix_tail_is_recorded_but_not_warned(path):
    """Every healthy Createc image carries a zero appendix after the planes
    (four spare scan-line buffers plus an optional 32-float block).  The
    viewer surfaces ``scan.warnings`` in the status bar, so this normal
    format layout must not produce a user-facing warning; the size stays
    available on the report for diagnostics.
    """
    report = read_createc_dat_report(path, include_raw=False)

    assert 0 < report.ignored_tail_float_count <= 4 * report.original_Nx + 32
    assert not any("trailing float32" in w for w in report.warnings)

    scan = load_scan(path)
    assert not any("trailing float32" in w for w in scan.warnings)


def test_oversized_tail_still_warns(tmp_path):
    """A tail beyond the known appendix budget means payload the reader does
    not understand and must stay loud."""
    dat = tmp_path / "oversized_tail.dat"
    header = b"[Paramco32]\nNum.X=2\nNum.Y=2\nChannels=2\n"
    # 2 channels of (2,2) pixels = 8 floats, then 41 tail floats: one more
    # than the 4*Nx + 32 = 40 appendix budget.
    payload = np.arange(1, 50, dtype="<f4").tobytes()
    dat.write_bytes(header + b"DATA" + zlib.compress(payload))

    report = read_createc_dat_report(dat, include_raw=False)

    assert report.detected_channel_count == 2
    assert report.ignored_tail_float_count == 41
    assert any("trailing float32" in w for w in report.warnings)


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


def test_data_marker_inside_header_is_not_used_as_payload_start(tmp_path):
    dat = tmp_path / "comment_mentions_data.dat"
    header = (
        b"[Paramco32]\n"
        b"Comment=operator checked DATA integrity before scan\n"
        b"Num.X=3\n"
        b"Num.Y=2\n"
        b"Channels=1\n"
    )
    payload = np.arange(1, 7, dtype="<f4").tobytes()
    dat.write_bytes(header + b"DATA" + zlib.compress(payload))

    report = read_createc_dat_report(dat)

    assert report.detected_channel_count == 1
    assert report.decoded_channels_dac is not None
    np.testing.assert_array_equal(
        report.decoded_channels_dac[0],
        np.array([[2, 3], [5, 6]], dtype=np.float32),
    )


def test_createc_header_aliases_preserve_channels_and_latin1_dacto(tmp_path):
    dat = tmp_path / "alias_header.dat"
    header = (
        "[Paramco32]\n"
        "Num.X=3\n"
        "Num.Y=2\n"
        "ScanChannels=99\n"
        "InternalChannels / Channels=1\n"
        "InternalZ / Dacto[Å]z=2.0\n"
        "GainZ=10\n"
    ).encode("latin-1")
    payload = np.arange(1, 7, dtype="<f4").tobytes()
    dat.write_bytes(header + b"DATA" + zlib.compress(payload))

    report = read_createc_dat_report(dat)

    assert report.detected_channel_count == 1
    assert report.original_header["Channels"] == "1"
    assert report.original_header["InternalChannels"] == "1"
    assert report.original_header["ScanChannels"] == "99"
    assert report.scale_factors["z_m_per_dac"] == pytest.approx(2.0e-9)


def test_image_y_pos_max_records_partial_scan_without_guessing_full_height(tmp_path):
    dat = tmp_path / "partial.dat"
    header = (
        b"[Paramco32]\n"
        b"Num.X=3\n"
        b"Num.Y=5\n"
        b"Channels=1\n"
        b"ImageYPosMax=4\n"
    )
    payload = np.arange(1, 16, dtype="<f4").tobytes()
    dat.write_bytes(header + b"DATA" + zlib.compress(payload))

    report = read_createc_dat_report(dat)

    assert report.image_y_pos_max == 4
    assert report.is_partial_scan is True
    assert report.original_Ny == 5
    assert report.trimmed_Ny == 3
    assert report.decoded_Ny == 3
    assert report.decoded_channels_dac is not None
    assert report.decoded_channels_dac.shape == (1, 3, 2)


def test_image_y_pos_max_complete_scan_keeps_zero_valued_trailing_rows(tmp_path):
    # ImageYPosMax = Num.Y + 1 means every declared row completed; the legacy
    # channel-0 nonzero heuristic must not trim genuine all-zero trailing rows.
    dat = tmp_path / "complete_flat_tail.dat"
    header = (
        b"[Paramco32]\n"
        b"Num.X=3\n"
        b"Num.Y=4\n"
        b"Channels=1\n"
        b"ImageYPosMax=5\n"
    )
    data = np.arange(1, 13, dtype="<f4").reshape(4, 3)
    data[-1, :] = 0.0  # genuine data that happens to be zero DAC
    dat.write_bytes(header + b"DATA" + zlib.compress(data.tobytes()))

    report = read_createc_dat_report(dat)

    assert report.image_y_pos_max == 5
    assert report.is_partial_scan is False
    assert report.trimmed_Ny == 4
    assert report.decoded_channels_dac is not None
    assert report.decoded_channels_dac.shape == (1, 4, 2)


def test_read_dat_preserves_decode_warnings_on_scan(tmp_path):
    dat = tmp_path / "partial.dat"
    header = (
        b"[Paramco32]\n"
        b"Num.X=3\n"
        b"Num.Y=5\n"
        b"Channels=1\n"
        b"ImageYPosMax=4\n"
    )
    payload = np.arange(1, 16, dtype="<f4").tobytes()
    dat.write_bytes(header + b"DATA" + zlib.compress(payload))

    scan = load_scan(dat)

    assert scan.warnings
    assert any("trimmed image height from 5 to 3" in msg for msg in scan.warnings)


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


def test_truncated_zlib_payload_reports_corruption_not_unsupported(tmp_path):
    """A DATA marker followed by a truncated zlib stream (e.g. a half-copied
    network file) must report corruption/truncation, not an unsupported variant."""
    dat = tmp_path / "truncated.dat"
    header = b"[Paramco32]\nNum.X=2\nNum.Y=2\n"
    full = zlib.compress(np.arange(1, 17, dtype="<f4").tobytes())
    dat.write_bytes(header + b"DATA" + full[: len(full) // 2])  # cut the stream

    with pytest.raises(ValueError, match="corrupt or truncated"):
        read_createc_dat_report(dat, include_raw=False)


def test_non_zlib_payload_reports_unsupported_variant(tmp_path):
    """A DATA marker with no zlib header (0x78) following it is an unsupported
    layout, and the message names the leading format token."""
    dat = tmp_path / "uncompressed.dat"
    header = b"[Paramco99]\nNum.X=2\nNum.Y=2\n"
    dat.write_bytes(header + b"DATA" + b"\x00\x01\x02\x03 raw uncompressed bytes")

    with pytest.raises(ValueError, match="unsupported Createc .dat variant"):
        read_createc_dat_report(dat, include_raw=False)


def test_trailing_bytes_after_zlib_stream_still_decode(tmp_path):
    """Padding/trailing bytes after a complete zlib stream must not break decode
    (zlib.decompress tolerates them); only genuine truncation should fail."""
    dat = tmp_path / "trailing.dat"
    header = b"[Paramco32]\nNum.X=2\nNum.Y=2\n"
    comp = zlib.compress(np.arange(1, 17, dtype="<f4").tobytes())
    dat.write_bytes(header + b"DATA" + comp + b"\x00\x00trailing")

    report = read_createc_dat_report(dat, include_raw=False)
    assert report.detected_channel_count == 4


# --------------------------------------------------------------------------- #
# Decoded scan range: pixel step from the acquisition grid, extent from the
# decoded shape (physics review 2026-07-02)
# --------------------------------------------------------------------------- #
def test_decoded_scan_range_keeps_acquisition_pixel_step(sample_dat_files):
    """Pairing full-frame Length x/y with the decoded (first-column-removed)
    shape overestimated pixel_size_x by Nx/(Nx-1) and made square scans read
    as anisotropic. The decoded range must preserve the acquisition step."""
    from probeflow.io.readers.createc_dat import (
        decoded_scan_range_m,
        scan_range_m_from_header,
    )
    from probeflow.io.readers.createc_scan import read_dat

    path = sample_dat_files[0]
    report = read_createc_dat_report(path, include_raw=False)
    full = scan_range_m_from_header(report.original_header)
    step_x = full[0] / report.original_Nx
    step_y = full[1] / report.original_Ny

    got = decoded_scan_range_m(report)
    assert got[0] == pytest.approx(step_x * report.decoded_Nx, rel=1e-12)
    assert got[1] == pytest.approx(step_y * report.decoded_Ny, rel=1e-12)

    scan = read_dat(path)
    ny, nx = scan.planes[0].shape
    px_x = scan.scan_range_m[0] / nx
    px_y = scan.scan_range_m[1] / ny
    assert px_x == pytest.approx(step_x, rel=1e-12)
    assert px_y == pytest.approx(step_y, rel=1e-12)
    # This fixture is a physically square scan: pixel sizes must match, not
    # differ by the phantom Nx/(Nx-1) anisotropy the old pairing produced.
    if abs(full[0] - full[1]) < 1e-15 and report.original_Nx == report.original_Ny:
        assert px_x == pytest.approx(px_y, rel=1e-12)


def test_decoded_scan_range_shrinks_with_partial_scan_trim(tmp_path):
    """A scan stopped early keeps the programmed Length y in its header; the
    decoded range must scale by trimmed_Ny/original_Ny or every y-distance is
    wrong by the inverse factor (2x for a half-finished scan)."""
    from probeflow.io.readers.createc_dat import decoded_scan_range_m

    Nx = Ny = 4
    header = (
        b"[Paramco32]\n"
        b"Num.X=4\nNum.Y=4\n"
        b"Length x[A]=40.0\nLength y[A]=40.0\n"
        b"ImageYPosMax=3\n"  # one-based next-Y: 2 completed rows
    )
    payload = zlib.compress(
        np.arange(1, 2 * Ny * Nx + 1, dtype="<f4").tobytes()
    )
    dat = tmp_path / "partial.dat"
    dat.write_bytes(header + b"DATA" + payload)

    report = read_createc_dat_report(dat, include_raw=False)
    assert report.trimmed_Ny == 2
    assert report.is_partial_scan
    assert report.first_column_removed

    width_m, height_m = decoded_scan_range_m(report)
    # step = 40 A / 4 px = 1 nm per pixel on both axes
    assert width_m == pytest.approx(1e-9 * 3, rel=1e-12)   # 3 columns remain
    assert height_m == pytest.approx(1e-9 * 2, rel=1e-12)  # 2 rows completed
