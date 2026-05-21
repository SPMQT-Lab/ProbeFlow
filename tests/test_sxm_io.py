"""Tests for probeflow.io.sxm_io — pure-python .sxm reader / writer."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from probeflow.io.converters.createc_dat_to_sxm import convert_dat_to_sxm
from probeflow.io.sxm_io import (
    _cushion_tail_lens,
    parse_sxm_header,
    read_all_sxm_planes,
    read_sxm_plane,
    sxm_data_info,
    sxm_dims,
    sxm_payload_plane_count,
    sxm_plane_metadata,
    sxm_scan_range,
    write_sxm_with_planes,
)


@pytest.fixture
def sample_sxm(tmp_path, first_sample_dat, cushion_dir) -> Path:
    """Convert one .dat → .sxm and return the output path."""
    out_dir = tmp_path / "sxm"
    convert_dat_to_sxm(first_sample_dat, out_dir, cushion_dir)
    files = list(out_dir.glob("*.sxm"))
    assert files, "conversion produced no .sxm"
    return files[0]


class TestHeader:
    def test_header_contract_has_dimensions_range_and_nanonis_fields(self, sample_sxm):
        hdr = parse_sxm_header(sample_sxm)
        assert isinstance(hdr, dict)
        assert "NANONIS_VERSION" in hdr
        assert "SCAN_PIXELS" in hdr
        Nx, Ny = sxm_dims(hdr)
        assert Nx > 0 and Ny > 0
        w_m, h_m = sxm_scan_range(hdr)
        assert w_m > 0 and h_m > 0

    def test_sxm_data_info_parses_saved_channels(self):
        hdr = {
            "DATA_INFO": (
                "Channel Name Unit Direction Calibration Offset "
                "14 Z m both 2.100E-8 0.000E+0 "
                "0 Current A both -3.300E-10 1.662E-12 "
                "18 OC_M1_Freq._Shift Hz both 1.526E+1 0.000E+0"
            )
        }
        rows = sxm_data_info(hdr)
        assert [row["name"] for row in rows] == [
            "Z",
            "Current",
            "OC M1 Freq. Shift",
        ]
        assert [row["unit"] for row in rows] == ["m", "A", "Hz"]
        names, units = sxm_plane_metadata(hdr, 6)
        assert names == [
            "Z forward",
            "Z backward",
            "Current forward",
            "Current backward",
            "OC M1 Freq. Shift forward",
            "OC M1 Freq. Shift backward",
        ]
        assert units == ["m", "m", "A", "A", "Hz", "Hz"]

    def test_parse_missing_scanit_end_raises(self, tmp_path):
        bad = tmp_path / "truncated.sxm"
        bad.write_bytes(b":NANONIS_VERSION:\n2\n:SCAN_PIXELS:\n4 4\n")
        with pytest.raises(ValueError, match="SCANIT_END"):
            parse_sxm_header(bad)

    def test_sxm_dims_missing_scan_pixels_raises(self):
        with pytest.raises(ValueError, match="SCAN_PIXELS"):
            sxm_dims({"NANONIS_VERSION": "2"})

    def test_sxm_dims_nonpositive_scan_pixels_raises(self):
        with pytest.raises(ValueError, match="invalid SCAN_PIXELS"):
            sxm_dims({"SCAN_PIXELS": "0 4"})


class TestReadPlanes:
    def test_single_and_all_plane_reads_match_payload_contract(self, sample_sxm):
        hdr, planes = read_all_sxm_planes(sample_sxm)
        assert len(planes) >= 1
        Nx, Ny = sxm_dims(hdr)
        for p in planes:
            assert p.shape == (Ny, Nx)
            assert p.ndim == 2
        first = read_sxm_plane(sample_sxm, plane_idx=0)
        assert first is not None
        np.testing.assert_array_equal(first, planes[0])
        assert np.isfinite(first).any()
        assert sxm_payload_plane_count(sample_sxm, hdr) == len(planes)
        assert read_sxm_plane(sample_sxm, plane_idx=99) is None

    def test_corrupt_header_raises(self, tmp_path):
        bad = tmp_path / "corrupt.sxm"
        bad.write_bytes(b":NANONIS_VERSION:\n2\n:SCAN_PIXELS:\n4 4\n")
        with pytest.raises(ValueError, match="SCANIT_END"):
            read_sxm_plane(bad, plane_idx=0)

    def test_missing_scan_pixels_raises(self, tmp_path):
        bad = tmp_path / "missing_pixels.sxm"
        bad.write_bytes(b":NANONIS_VERSION:\n2\n:SCANIT_END:\n")
        with pytest.raises(ValueError, match="SCAN_PIXELS"):
            read_sxm_plane(bad, plane_idx=0)


def test_read_all_sxm_planes_warns_on_truncated_payload(tmp_path):
    """A payload shorter than DATA_INFO promises must emit a UserWarning.

    Constructs a minimal SXM whose DATA_INFO header lists two channels but
    whose binary payload only contains one plane's worth of bytes. The reader
    must warn the caller rather than silently returning a truncated planes
    list.
    """
    post_len, pre_len = _cushion_tail_lens()
    header = (
        b":NANONIS_VERSION:\n2\n"
        b":SCAN_PIXELS:\n4 4\n"
        b":SCAN_RANGE:\n1.0E-9 1.0E-9\n"
        b":SCAN_DIR:\ndown\n"
        b":DATA_INFO:\n"
        b"\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
        b"\t14\tZ\tm\tforward\t1.0E-9\t0.0\n"
        b"\t0\tCurrent\tA\tforward\t1.0E-9\t0.0\n"
        b":SCANIT_END:\n"
    )
    cushion = b"\x00" * (post_len + pre_len)
    # DATA_INFO promises 2 planes; ship only 1 (4*4*4 = 64 bytes).
    payload_one_plane = np.ones((4, 4), dtype=">f4").tobytes()

    truncated = tmp_path / "truncated.sxm"
    truncated.write_bytes(header + cushion + payload_one_plane)

    # Sanity-check the construction before asserting the warning behaviour.
    hdr = parse_sxm_header(truncated)
    assert len(sxm_data_info(hdr)) == 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hdr_out, planes = read_all_sxm_planes(truncated)

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings, "expected a UserWarning for truncated SXM payload"
    msg = str(user_warnings[0].message)
    assert "incompletely written" in msg
    assert "truncated.sxm" in msg
    assert len(planes) == 1
    assert len(planes) < len(sxm_data_info(hdr_out))


class TestRoundTrip:
    def test_rewrite_preserves_plane_count_shape_and_finite_values(self, sample_sxm, tmp_path):
        out = tmp_path / "rewritten.sxm"
        hdr, planes = read_all_sxm_planes(sample_sxm)
        write_sxm_with_planes(sample_sxm, out, planes)
        assert out.exists()

        hdr2, planes2 = read_all_sxm_planes(out)
        assert sxm_dims(hdr2) == sxm_dims(hdr)
        assert len(planes2) == len(planes)
        for a, b in zip(planes, planes2):
            assert a.shape == b.shape
            finite = np.isfinite(a) & np.isfinite(b)
            assert np.allclose(a[finite], b[finite], atol=0, rtol=0)
