"""Tests for probeflow.sxm_io — pure-python .sxm reader / writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from probeflow.dat_sxm import convert_dat_to_sxm
from probeflow.sxm_io import (
    parse_sxm_header,
    read_all_sxm_planes,
    read_sxm_plane,
    sxm_dims,
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
    def test_parse_returns_dict(self, sample_sxm):
        hdr = parse_sxm_header(sample_sxm)
        assert isinstance(hdr, dict)
        assert "NANONIS_VERSION" in hdr
        assert "SCAN_PIXELS" in hdr

    def test_sxm_dims_positive(self, sample_sxm):
        hdr = parse_sxm_header(sample_sxm)
        Nx, Ny = sxm_dims(hdr)
        assert Nx > 0 and Ny > 0

    def test_sxm_scan_range_positive(self, sample_sxm):
        hdr = parse_sxm_header(sample_sxm)
        w_m, h_m = sxm_scan_range(hdr)
        assert w_m > 0 and h_m > 0

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
    def test_read_single_plane(self, sample_sxm):
        arr = read_sxm_plane(sample_sxm, plane_idx=0)
        assert arr is not None
        assert arr.ndim == 2
        assert np.isfinite(arr).any()

    def test_read_all_planes(self, sample_sxm):
        hdr, planes = read_all_sxm_planes(sample_sxm)
        assert len(planes) >= 1
        Nx, Ny = sxm_dims(hdr)
        for p in planes:
            assert p.shape == (Ny, Nx)

    def test_missing_plane_returns_none(self, sample_sxm):
        arr = read_sxm_plane(sample_sxm, plane_idx=99)
        assert arr is None

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


class TestRoundTrip:
    def test_rewrite_preserves_shape(self, sample_sxm, tmp_path):
        out = tmp_path / "rewritten.sxm"
        hdr, planes = read_all_sxm_planes(sample_sxm)
        write_sxm_with_planes(sample_sxm, out, planes)
        assert out.exists()

        hdr2, planes2 = read_all_sxm_planes(out)
        assert len(planes2) == len(planes)
        for a, b in zip(planes, planes2):
            assert a.shape == b.shape

    def test_rewrite_preserves_values(self, sample_sxm, tmp_path):
        out = tmp_path / "rewritten.sxm"
        _, planes = read_all_sxm_planes(sample_sxm)
        write_sxm_with_planes(sample_sxm, out, planes)
        _, planes2 = read_all_sxm_planes(out)
        for a, b in zip(planes, planes2):
            finite = np.isfinite(a) & np.isfinite(b)
            # float32 round-trip — values identical when both finite
            assert np.allclose(a[finite], b[finite], atol=0, rtol=0)
