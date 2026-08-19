"""Read-only source-file invariants for the architecture refactor."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from probeflow.core.file_type import FileType, sniff_file_type
from probeflow.core.scan_loader import load_scan
from probeflow.io.spectroscopy import read_spec_file


TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def _fingerprint(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    digest = sha256(path.read_bytes()).hexdigest()
    return stat.st_size, stat.st_mtime_ns, digest


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("createc_scan_11nm.dat", FileType.CREATEC_IMAGE),
        ("nanonis.sxm", FileType.NANONIS_IMAGE),
        ("rhk.sm4", FileType.RHK_SM4_IMAGE),
    ],
)
def test_loading_real_scan_fixture_is_read_only(name: str, expected_type: FileType) -> None:
    path = TEST_DATA / name
    before = _fingerprint(path)

    assert sniff_file_type(path) is expected_type
    load_scan(path)

    assert _fingerprint(path) == before


@pytest.mark.parametrize(
    ("fixture_name", "expected_type"),
    [
        ("createc_time_spec", FileType.CREATEC_SPEC),
        ("nanonis_spec", FileType.NANONIS_SPEC),
    ],
)
def test_loading_spectrum_fixture_is_read_only(
    request: pytest.FixtureRequest,
    fixture_name: str,
    expected_type: FileType,
) -> None:
    path: Path = request.getfixturevalue(fixture_name)
    before = _fingerprint(path)

    assert sniff_file_type(path) is expected_type
    read_spec_file(path)

    assert _fingerprint(path) == before
