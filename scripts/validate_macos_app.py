#!/usr/bin/env python3
"""Validate a built ProbeFlow.app before DMG creation or signing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import plistlib
import subprocess
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = REPO_ROOT / "packaging" / "macos" / "app_metadata.toml"
FORBIDDEN_DEPENDENCY_PREFIXES = ("/Users/", "/opt/homebrew/", "/opt/anaconda3/")
FORBIDDEN_QT_COMPONENTS = (
    "Frameworks/PySide6/Qt/lib/QtVirtualKeyboard.framework",
    "Frameworks/PySide6/Qt/plugins/platforminputcontexts/libqtvirtualkeyboardplugin.dylib",
    "Frameworks/PySide6/Qt/plugins/imageformats/libqpdf.dylib",
)


def _run(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout


def _codesign_details(path: Path) -> str:
    return subprocess.run(
        ["codesign", "-dv", "--verbose=4", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stderr


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _mach_o_files(contents: Path) -> list[Path]:
    files = [path for path in contents.rglob("*") if path.is_file() and not path.is_symlink()]
    mach_o = []
    for offset in range(0, len(files), 100):
        output = _run("file", *(str(path) for path in files[offset : offset + 100]))
        for line in output.splitlines():
            name, _, description = line.partition(":")
            if "Mach-O" in description:
                mach_o.append(Path(name))
    return mach_o


def validate_app(app: Path) -> None:
    app = app.resolve()
    contents = app / "Contents"
    metadata = tomllib.loads(METADATA_PATH.read_text(encoding="utf-8"))["application"]
    plist = plistlib.loads((contents / "Info.plist").read_bytes())

    expected_plist = {
        "CFBundleIdentifier": metadata["bundle_identifier"],
        "CFBundleShortVersionString": metadata["bundle_short_version"],
        "CFBundleVersion": str(metadata["bundle_build_number"]),
        "LSMinimumSystemVersion": metadata["minimum_macos_version"],
    }
    for key, expected in expected_plist.items():
        if plist.get(key) != expected:
            raise RuntimeError(f"{key}: expected {expected!r}, found {plist.get(key)!r}")

    resources = contents / "Resources"
    required_resources = (
        resources / "ProbeFlow.icns",
        resources / "LICENSE",
        resources / "THIRD_PARTY_NOTICES.md",
        resources / "QT_LGPL_COMPLIANCE.md",
        resources / "THIRD_PARTY_LICENSES" / "python" / "PYTHON_DISTRIBUTIONS.txt",
        resources / "THIRD_PARTY_LICENSES" / "qt" / "QT_CORRESPONDING_SOURCE.txt",
        resources
        / "THIRD_PARTY_LICENSES"
        / "runtime"
        / "CPython-3.13.14"
        / "LICENSE.txt",
        resources / "probeflow" / "assets" / "logo.png",
        resources / "probeflow" / "data" / "file_cushions" / "header_format.json",
    )
    missing = [str(path) for path in required_resources if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing application resources: {', '.join(missing)}")
    if not list(resources.glob("gwyfile-*.dist-info/METADATA")):
        raise RuntimeError("gwyfile package metadata is missing")

    forbidden_components = [
        str(contents / relative)
        for relative in FORBIDDEN_QT_COMPONENTS
        if (contents / relative).exists()
    ]
    if forbidden_components:
        raise RuntimeError(
            "Unused or GPL-only Qt component in application: "
            + ", ".join(forbidden_components)
        )

    mach_o_files = _mach_o_files(contents)
    if not mach_o_files:
        raise RuntimeError("No Mach-O binaries found in application bundle")

    maximum_minos = (0,)
    maximum_minos_text = "0"
    for binary in mach_o_files:
        description = _run("file", "-b", str(binary))
        if "arm64" not in description or "x86_64" in description:
            raise RuntimeError(f"Non-arm64 binary in application: {binary}: {description.strip()}")

        build_info = _run("vtool", "-show-build", str(binary))
        for line in build_info.splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] == "minos":
                version = _version_tuple(fields[1])
                if version > maximum_minos:
                    maximum_minos = version
                    maximum_minos_text = fields[1]

        dependencies = _run("otool", "-L", str(binary))
        for line in dependencies.splitlines()[1:]:
            dependency = line.strip().split(" (", 1)[0]
            for forbidden in FORBIDDEN_DEPENDENCY_PREFIXES:
                if dependency.startswith(forbidden):
                    raise RuntimeError(
                        f"Build-machine dependency in {binary}: {dependency}"
                    )

    declared_minimum = _version_tuple(metadata["minimum_macos_version"])
    if maximum_minos > declared_minimum:
        raise RuntimeError(
            f"Bundled binary requires macOS {maximum_minos_text}, above declared "
            f"minimum {metadata['minimum_macos_version']}"
        )

    subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", str(app)],
        check=True,
    )
    signature_details = _codesign_details(app)
    expected_identity = os.environ.get("PROBEFLOW_CODESIGN_IDENTITY")
    if expected_identity:
        if expected_identity not in signature_details:
            raise RuntimeError(
                f"Application is not signed by expected identity: {expected_identity}"
            )
        if "runtime" not in signature_details:
            raise RuntimeError("Developer ID application is missing hardened runtime")
        team_lines = [
            line for line in signature_details.splitlines() if line.startswith("TeamIdentifier=")
        ]
        if len(team_lines) != 1 or team_lines[0] == "TeamIdentifier=not set":
            raise RuntimeError("Developer ID application has no signing TeamIdentifier")
        expected_team = team_lines[0]
        for binary in mach_o_files:
            binary_details = _codesign_details(binary)
            if expected_team not in binary_details:
                raise RuntimeError(
                    f"Nested binary has a different signing team: {binary}"
                )
            if "runtime" not in binary_details:
                raise RuntimeError(f"Nested binary lacks hardened runtime: {binary}")
        signature_kind = "Developer ID and hardened-runtime signatures"
    else:
        if "Signature=adhoc" not in signature_details:
            raise RuntimeError("Expected an ad-hoc signature for the unsigned test build")
        signature_kind = "ad-hoc signatures"
    print(
        f"Validated {app.name}: {len(mach_o_files)} arm64 Mach-O files, "
        f"maximum deployment target macOS {maximum_minos_text}, metadata and "
        f"{signature_kind} valid"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "app",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "build" / "macos" / "dist" / "ProbeFlow.app",
    )
    args = parser.parse_args()
    validate_app(args.app)


if __name__ == "__main__":
    main()
