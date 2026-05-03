"""Tests for the content-sniffing file-type dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from probeflow.io.file_type import FileType, sniff_file_type


TESTDATA = Path(__file__).resolve().parents[1] / "anonymised_testdata"


# Expected mapping from each test data file to its FileType.
# This drives the full sniffing roundtrip check.
_EXPECTED = {
    "createc_scan_step_20nm.dat":     FileType.CREATEC_IMAGE,
    "createc_scan_terrace_109nm.dat": FileType.CREATEC_IMAGE,
    "createc_scan_qplus_10ch_afm.dat": FileType.CREATEC_IMAGE,
    "createc_ivt_telegraph_300mv_a.VERT": FileType.CREATEC_SPEC,
    "createc_ivt_telegraph_300mv_b.VERT": FileType.CREATEC_SPEC,
    "createc_ivt_telegraph_450mv.VERT":   FileType.CREATEC_SPEC,
    "createc_vert_didz_image_state.VERT": FileType.CREATEC_SPEC,
    "nanonis_kelvin_parabola_500mv.dat":  FileType.NANONIS_SPEC,
    "nanonis_sts_15mv.dat":         FileType.NANONIS_SPEC,
    "sxm_moire_10nm.sxm":                 FileType.NANONIS_IMAGE,
}


@pytest.mark.parametrize("fname,expected", sorted(_EXPECTED.items()))
def test_sniff_known_files(fname: str, expected: FileType) -> None:
    """Every anonymised test file is classified correctly."""
    path = TESTDATA / fname
    if not path.exists():
        pytest.skip(f"missing test fixture: {fname}")
    assert sniff_file_type(path) == expected


def test_sniff_createc_dat_is_image() -> None:
    """The Createc .dat regression cases both classify as CREATEC_IMAGE."""
    assert sniff_file_type(TESTDATA / "createc_scan_step_20nm.dat") == FileType.CREATEC_IMAGE


def test_sniff_nanonis_dat_is_spec() -> None:
    """Nanonis .dat spectroscopy is NOT confused with a Createc .dat image."""
    assert sniff_file_type(TESTDATA / "nanonis_sts_15mv.dat") == FileType.NANONIS_SPEC


def test_sniff_missing_file_returns_unknown(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_file.dat"
    assert sniff_file_type(missing) == FileType.UNKNOWN


def test_sniff_corrupt_file_returns_unknown(tmp_path: Path) -> None:
    """Random bytes that match no signature return UNKNOWN without raising."""
    corrupt = tmp_path / "corrupt.bin"
    corrupt.write_bytes(b"\x00\x01\x02\x03random garbage 12345 not a known header")
    assert sniff_file_type(corrupt) == FileType.UNKNOWN


def test_sniff_empty_file_returns_unknown(tmp_path: Path) -> None:
    empty = tmp_path / "empty.dat"
    empty.write_bytes(b"")
    assert sniff_file_type(empty) == FileType.UNKNOWN
