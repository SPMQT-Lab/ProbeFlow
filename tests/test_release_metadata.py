"""Release metadata and generated macOS asset contracts."""

from __future__ import annotations

from pathlib import Path
import struct
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = REPO_ROOT / "packaging" / "macos" / "app_metadata.toml"
ICON_PATH = REPO_ROOT / "packaging" / "macos" / "ProbeFlow.icns"
NOTICES_PATH = REPO_ROOT / "packaging" / "THIRD_PARTY_NOTICES.md"


def _application_metadata() -> dict:
    return tomllib.loads(METADATA_PATH.read_text(encoding="utf-8"))["application"]


def test_macos_release_identity_is_complete():
    metadata = _application_metadata()

    assert metadata["name"] == metadata["executable_name"] == "ProbeFlow"
    assert metadata["bundle_identifier"] == "au.edu.uq.spmqt.probeflow"
    assert metadata["version_source"] == "probeflow.__version__"
    assert metadata["copyright"] == "Copyright © 2026 SPMQT-Lab and contributors"
    assert metadata["license"] == "MIT"


def test_first_macos_release_target_is_explicit():
    metadata = _application_metadata()

    assert metadata["minimum_macos_version"] == "15.0"
    assert metadata["primary_architecture"] == "arm64"
    assert set(metadata["future_architectures"]) == {"x86_64", "universal2"}


def test_generated_icon_is_a_well_formed_icns_container():
    data = ICON_PATH.read_bytes()

    assert data[:4] == b"icns"
    assert struct.unpack(">I", data[4:8])[0] == len(data)


def test_third_party_notice_covers_adapted_work_and_direct_dependencies():
    notice = NOTICES_PATH.read_text(encoding="utf-8")

    for expected in (
        "Rohan Platts",
        "AiSurf",
        "Total Variation",
        "NumPy",
        "SciPy",
        "Pillow",
        "PySide6",
        "Matplotlib",
        "Shapely",
        "scikit-image",
        "OpenCV",
        "scikit-learn",
        "gwyfile",
    ):
        assert expected in notice
