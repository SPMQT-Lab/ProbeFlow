#!/usr/bin/env python3
"""Validate the frozen Windows x64 ProbeFlow application directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


IMAGE_FILE_MACHINE_AMD64 = 0x8664
NATIVE_SUFFIXES = {".dll", ".exe", ".pyd"}


def _text(value: bytes | str) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _version_strings(pe) -> dict[str, str]:
    values: dict[str, str] = {}
    for group in getattr(pe, "FileInfo", ()) or ():
        entries = group if isinstance(group, list) else [group]
        for entry in entries:
            if _text(getattr(entry, "Key", "")) != "StringFileInfo":
                continue
            for table in getattr(entry, "StringTable", ()):
                values.update({_text(key): _text(value) for key, value in table.entries.items()})
    return values


def validate(app_dir: Path) -> None:
    try:
        import pefile
    except ImportError as exc:
        raise RuntimeError("pefile is required to validate a Windows bundle") from exc

    app_dir = app_dir.resolve()
    executable = app_dir / "ProbeFlow.exe"
    internal = app_dir / "_internal"
    required = (
        executable,
        internal / "probeflow" / "assets" / "logo.png",
        internal / "probeflow" / "assets" / "toolbar" / "open_fft.png",
        internal / "probeflow" / "data" / "file_cushions" / "header_format.json",
        internal / "probeflow" / "data" / "file_cushions" / "pre_payload_bytes.bin",
        internal / "LICENSE",
        internal / "THIRD_PARTY_NOTICES.md",
        internal / "QT_LGPL_COMPLIANCE.md",
        internal / "THIRD_PARTY_LICENSES" / "python" / "PYTHON_DISTRIBUTIONS.txt",
        internal
        / "THIRD_PARTY_LICENSES"
        / "runtime"
        / "CPython-3.13.14"
        / "LICENSE.txt",
        internal / "THIRD_PARTY_LICENSES" / "qt" / "QT_CORRESPONDING_SOURCE.txt",
        internal / "PySide6" / "Qt" / "bin" / "Qt6Core.dll",
        internal / "PySide6" / "Qt" / "bin" / "Qt6Widgets.dll",
        internal / "python313.dll",
    )
    missing = [str(path.relative_to(app_dir)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Windows bundle is missing required files: {', '.join(missing)}")

    forbidden_tokens = (
        "qpdf.dll",
        "qt6pdf.dll",
        "qt6qml.dll",
        "qt6quick.dll",
        "qtvirtualkeyboard",
    )
    forbidden = [
        str(path.relative_to(app_dir))
        for path in app_dir.rglob("*")
        if path.is_file() and any(token in path.name.lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise RuntimeError(f"Windows bundle contains unused Qt components: {', '.join(forbidden)}")

    foreign = [
        str(path.relative_to(app_dir))
        for path in app_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".dylib", ".so"}
    ]
    if foreign:
        raise RuntimeError(f"Windows bundle contains foreign native files: {', '.join(foreign)}")

    binaries = [
        path
        for path in app_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in NATIVE_SUFFIXES
    ]
    if not binaries:
        raise RuntimeError("Windows bundle contains no PE binaries")
    for path in binaries:
        with pefile.PE(str(path), fast_load=True) as pe:
            if pe.FILE_HEADER.Machine != IMAGE_FILE_MACHINE_AMD64:
                raise RuntimeError(
                    f"Non-x64 PE binary: {path.relative_to(app_dir)} "
                    f"(machine 0x{pe.FILE_HEADER.Machine:04x})"
                )

    with pefile.PE(str(executable), fast_load=False) as pe:
        version = _version_strings(pe)
    expected_version = {
        "CompanyName": "SPMQT-Lab",
        "FileDescription": "ProbeFlow desktop application",
        "OriginalFilename": "ProbeFlow.exe",
        "ProductName": "ProbeFlow",
        "ProductVersion": "1.0.0 RC 1",
    }
    mismatched = {
        key: (version.get(key), expected)
        for key, expected in expected_version.items()
        if version.get(key) != expected
    }
    if mismatched:
        raise RuntimeError(f"ProbeFlow.exe version metadata mismatch: {mismatched}")

    print(
        f"Validated {app_dir.name}: {len(binaries)} x64 PE files, "
        "resources, licenses, and version metadata valid"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app_dir", type=Path)
    args = parser.parse_args()
    try:
        validate(args.app_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Windows bundle validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
