#!/usr/bin/env python3
"""Collect license files from the pinned runtime distributions."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path, PurePosixPath
import shutil
import tomllib


NOTICE_PREFIXES = ("authors", "copying", "copyright", "license", "notice")
NON_TEXT_SUFFIXES = {".dll", ".dylib", ".py", ".pyc", ".pyd", ".so"}


def _is_notice_file(path: PurePosixPath) -> bool:
    name = path.name.lower()
    return name.startswith(NOTICE_PREFIXES) and path.suffix.lower() not in NON_TEXT_SUFFIXES


def _license_summary(value: str | None) -> str:
    if not value:
        return "not declared in wheel metadata"
    first_line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    return first_line[:157] + "..." if len(first_line) > 160 else first_line


def collect(manifest_path: Path, output: Path) -> None:
    config = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    external = set(config["licenses_from_source_archives"])
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)

    index_lines = ["ProbeFlow bundled Python distribution licenses", ""]
    for requested_name in config["runtime_distributions"]:
        try:
            dist = distribution(requested_name)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Required distribution is not installed: {requested_name}") from exc

        name = dist.metadata.get("Name", requested_name)
        declared_license = _license_summary(
            dist.metadata.get("License-Expression") or dist.metadata.get("License")
        )
        copied: list[str] = []
        destination_root = output / "Python" / f"{name}-{dist.version}"
        for item in dist.files or ():
            relative = PurePosixPath(str(item))
            if not _is_notice_file(relative):
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe distribution path in {name}: {relative}")
            source = Path(dist.locate_file(item))
            if not source.is_file():
                continue
            destination = destination_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied.append(relative.as_posix())

        if not copied and requested_name not in external:
            raise RuntimeError(f"No license or notice files found in {name} {dist.version}")

        index_lines.extend(
            [
                f"{name} {dist.version}",
                f"  Declared license: {declared_license}",
                (
                    f"  Included files: {len(copied)}"
                    if copied
                    else "  Full license texts: see the Qt source-license directory"
                ),
                "",
            ]
        )

    (output / "PYTHON_DISTRIBUTIONS.txt").write_text(
        "\n".join(index_lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    collect(args.manifest, args.output)


if __name__ == "__main__":
    main()
