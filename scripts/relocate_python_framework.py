#!/usr/bin/env python3
"""Make an extracted Python.org framework self-contained for release builds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


INSTALL_PREFIX = "/Library/Frameworks/Python.framework/Versions/3.13/"


def _run(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout


def _mach_o_files(root: Path):
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if "Mach-O" in _run("file", "-b", str(path)):
            yield path


def _dependencies(path: Path) -> list[str]:
    dependencies = []
    for line in _run("otool", "-L", str(path)).splitlines()[1:]:
        candidate = line.strip().split(" (", 1)[0]
        if candidate.startswith(INSTALL_PREFIX) and candidate not in dependencies:
            dependencies.append(candidate)
    return dependencies


def _install_id(path: Path) -> str | None:
    lines = _run("otool", "-D", str(path)).splitlines()[1:]
    for line in lines:
        candidate = line.strip()
        if candidate.startswith(INSTALL_PREFIX):
            return candidate
    return None


def _loader_reference(root: Path, binary: Path, dependency: str) -> str:
    target = root / dependency.removeprefix(INSTALL_PREFIX)
    if not target.exists():
        raise FileNotFoundError(
            f"{binary} references {dependency}, but the extracted target is missing"
        )
    relative = os.path.relpath(target, start=binary.parent)
    return f"@loader_path/{relative}"


def relocate_framework(root: Path) -> int:
    root = root.resolve()
    changed = 0

    for binary in _mach_o_files(root):
        command = ["install_name_tool"]
        for dependency in _dependencies(binary):
            command.extend(
                ["-change", dependency, _loader_reference(root, binary, dependency)]
            )
        install_id = _install_id(binary)
        if install_id is not None:
            command.extend(["-id", f"@loader_path/{binary.name}"])
        if len(command) == 1:
            continue

        command.append(str(binary))
        subprocess.run(command, check=True)
        subprocess.run(["codesign", "--force", "--sign", "-", str(binary)], check=True)
        changed += 1

    python_app = root / "Resources" / "Python.app"
    subprocess.run(
        ["codesign", "--force", "--deep", "--sign", "-", str(python_app)],
        check=True,
    )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("framework_version_root", type=Path)
    args = parser.parse_args()

    changed = relocate_framework(args.framework_version_root)
    print(f"Relocated {changed} Python framework binaries")


if __name__ == "__main__":
    main()
