"""Tests for probeflow.io.sxm_io — pure-python .sxm reader / writer."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from probeflow.io.converters.createc_dat_to_sxm import convert_dat_to_sxm
from probeflow.io.readers.nanonis_sxm import read_sxm
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

    def test_parse_header_preserves_multiline_values(self, tmp_path):
        sxm = tmp_path / "commented.sxm"
        sxm.write_bytes(
            b":COMMENT:\nfirst paragraph\n\nsecond paragraph\n:SCANIT_END:\n"
        )

        hdr = parse_sxm_header(sxm)

        assert hdr["COMMENT"] == "first paragraph\n\nsecond paragraph"

    def test_sxm_data_info_warns_on_partial_row(self):
        hdr = {
            "DATA_INFO": (
                "Channel Name Unit Direction Calibration Offset\n"
                "14 Z m forward 1.0E-9 0.0\n"
                "0 Current A forward"
            )
        }

        with pytest.warns(UserWarning, match="Malformed DATA_INFO row"):
            rows = sxm_data_info(hdr)

        assert [row["name"] for row in rows] == ["Z"]

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


def test_read_sxm_records_truncated_payload_warning_on_scan(tmp_path):
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
    payload_one_plane = np.ones((4, 4), dtype=">f4").tobytes()
    truncated = tmp_path / "truncated.sxm"
    truncated.write_bytes(header + cushion + payload_one_plane)

    scan = read_sxm(truncated)

    assert scan.warnings
    assert "incompletely written" in scan.warnings[0]
    assert scan.n_planes == 1


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

    def test_rewrite_redacts_identity_and_directory_header_fields(self, tmp_path):
        post_len, pre_len = _cushion_tail_lens()
        header = (
            b":NANONIS_VERSION:\n2\n"
            b":SCAN_PIXELS:\n2 2\n"
            b":SCAN_RANGE:\n1.0E-9 1.0E-9\n"
            b":SCAN_DIR:\ndown\n"
            b":SCAN_FILE:\n/Users/alice/private/scan.sxm\n"
            b":Username:\nalice\n"
            b":NanonisMain>Session Path:\nC:\\Users\\alice\\session\n"
            b":DATA_INFO:\n"
            b"\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
            b"\t14\tZ\tm\tforward\t1.0\t0.0\n"
            b":SCANIT_END:\n"
        )
        payload = np.arange(4, dtype=">f4").reshape(2, 2).tobytes()
        source = tmp_path / "source.sxm"
        source.write_bytes(header + b"\x00" * (post_len + pre_len) + payload)
        out = tmp_path / "out.sxm"

        _, planes = read_all_sxm_planes(source)
        write_sxm_with_planes(source, out, planes)
        safe = parse_sxm_header(out)

        assert safe["SCAN_FILE"] == "scan.sxm"
        assert safe["Username"] == ""
        assert safe["NanonisMain>Session Path"] == "session"
        assert "/Users/" not in out.read_bytes().split(b":SCANIT_END:", 1)[0].decode(
            "latin-1"
        )


class TestWriteSxmSafetyGuards:
    """Regression tests for review IO #1: write_sxm_with_planes must
    refuse to overwrite its source and must respect overwrite=False."""

    def test_refuses_to_overwrite_source_file(self, sample_sxm):
        _, planes = read_all_sxm_planes(sample_sxm)
        with pytest.raises(ValueError, match="overwrite the source"):
            write_sxm_with_planes(sample_sxm, sample_sxm, planes)

    def test_refuses_existing_destination_without_overwrite(self, sample_sxm, tmp_path):
        out = tmp_path / "exists.sxm"
        out.write_bytes(b"placeholder")
        _, planes = read_all_sxm_planes(sample_sxm)
        with pytest.raises(FileExistsError, match="already exists"):
            write_sxm_with_planes(sample_sxm, out, planes)
        # Original placeholder content is preserved
        assert out.read_bytes() == b"placeholder"

    def test_overwrite_true_replaces_existing_destination(self, sample_sxm, tmp_path):
        out = tmp_path / "exists.sxm"
        out.write_bytes(b"placeholder")
        _, planes = read_all_sxm_planes(sample_sxm)
        write_sxm_with_planes(sample_sxm, out, planes, overwrite=True)
        # Now the file is a real SXM
        hdr_out, planes_out = read_all_sxm_planes(out)
        assert len(planes_out) == len(planes)

    def test_source_equality_uses_resolved_paths(self, sample_sxm, tmp_path):
        """Disguised same-file path (relative vs absolute) must still raise."""
        # Build a path that resolves to the same file but spells differently:
        # use a symlink or a relative path through tmp_path that points back.
        link = tmp_path / "link.sxm"
        link.symlink_to(sample_sxm)
        _, planes = read_all_sxm_planes(sample_sxm)
        with pytest.raises(ValueError, match="overwrite the source"):
            write_sxm_with_planes(sample_sxm, link, planes)


class TestScanDirUpRoundTrip:
    """Regression for review IO #5 — DAT files acquired with reverse Y
    scan direction (Createc ``ScanYDirec=0``) produce SXM headers with
    ``SCAN_DIR='up'``.  The writer must un-orient the planes (np.flipud)
    so the reader's orient_plane flip restores the original display
    orientation.  Before the fix the DAT-sourced write path dropped
    this flip and the round-trip image was upside-down.
    """

    def _build_minimal_sxm_args(self, cushion_dir, scan_dir: str):
        """Build a header + asymmetric plane so flipud is observable."""
        from probeflow.io.converters.createc_dat_to_sxm import (
            load_layout_and_format,
        )
        # Asymmetric pattern: top half = 1.0, bottom half = 0.0 → flipud
        # produces the inverse pattern, immediately visible in equality test.
        Ny, Nx = 8, 8
        plane = np.zeros((Ny, Nx), dtype=np.float32)
        plane[: Ny // 2, :] = 1.0
        hdr = {
            "NANONIS_VERSION": "2",
            "SCAN_PIXELS": f"{Nx} {Ny}",
            "SCAN_RANGE": "1.0E-9 1.0E-9",
            "SCAN_DIR": scan_dir,
            "DATA_INFO": (
                "\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n"
                "\t14\tZ\tm\tforward\t1.0E-9\t0.0"
            ),
            "SCANIT_TYPE": "FLOAT MSBFIRST",
        }
        imgs = [("Z", "m", "forward", plane)]
        layout, header_format = load_layout_and_format(Path(cushion_dir))
        return hdr, imgs, layout, header_format, plane

    def test_scan_dir_up_round_trips(self, tmp_path, cushion_dir):
        from probeflow.io.converters.createc_dat_to_sxm import (
            reconstruct_from_hdr_imgs,
        )
        hdr, imgs, layout, header_format, original = self._build_minimal_sxm_args(
            cushion_dir, scan_dir="up"
        )
        out = tmp_path / "up.sxm"
        reconstruct_from_hdr_imgs(
            hdr=hdr, imgs=imgs, header_format=header_format,
            post_end_bytes=layout["post_end_bytes"],
            pre_payload_bytes=layout["pre_payload_bytes"],
            out_path=out,
            tail_bytes=layout["tail_bytes"],
            force_data_offset=layout["data_offset"],
        )

        _, planes = read_all_sxm_planes(out)
        assert len(planes) == 1
        np.testing.assert_array_equal(planes[0], original)

    def test_scan_dir_down_unchanged(self, tmp_path, cushion_dir):
        """Sanity check: SCAN_DIR='down' (the common case) is unchanged
        by the new flipud branch."""
        from probeflow.io.converters.createc_dat_to_sxm import (
            reconstruct_from_hdr_imgs,
        )
        hdr, imgs, layout, header_format, original = self._build_minimal_sxm_args(
            cushion_dir, scan_dir="down"
        )
        out = tmp_path / "down.sxm"
        reconstruct_from_hdr_imgs(
            hdr=hdr, imgs=imgs, header_format=header_format,
            post_end_bytes=layout["post_end_bytes"],
            pre_payload_bytes=layout["pre_payload_bytes"],
            out_path=out,
            tail_bytes=layout["tail_bytes"],
            force_data_offset=layout["data_offset"],
        )
        _, planes = read_all_sxm_planes(out)
        np.testing.assert_array_equal(planes[0], original)
