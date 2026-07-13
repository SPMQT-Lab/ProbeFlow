#!/usr/bin/env python3
"""Verify Qt source archives and extract their license/attribution material."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tomllib


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required to download Qt source archives")
    subprocess.run(
        [
            curl,
            "--location",
            "--fail",
            "--show-error",
            "--output",
            str(temporary),
            url,
        ],
        check=True,
    )
    temporary.replace(destination)


def _wanted_member(path: PurePosixPath) -> bool:
    return (
        "LICENSES" in path.parts
        or path.name == "REUSE.toml"
        or path.name == "qt_attribution.json"
    )


def prepare(manifest_path: Path, download_dir: Path, license_output: Path) -> None:
    config = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    download_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(license_output, ignore_errors=True)
    license_output.mkdir(parents=True)

    source_index = [
        "Qt corresponding-source archives for the ProbeFlow desktop release",
        "",
        "These unmodified, checksum-pinned archives must accompany binary releases.",
        "",
    ]
    for archive in config["qt_source_archives"]:
        path = download_dir / archive["filename"]
        if not path.exists():
            print(f"Downloading Qt source archive: {archive['filename']}")
            _download(archive["url"], path)
        actual = _sha256(path)
        if actual != archive["sha256"]:
            raise RuntimeError(
                f"Checksum mismatch for {path}: expected {archive['sha256']}, found {actual}"
            )

        component_root = license_output / "Qt" / (
            f"{archive['component']}-{archive['version']}"
        )
        extracted = 0
        with tarfile.open(path, mode="r:xz") as bundle:
            for member in bundle.getmembers():
                relative = PurePosixPath(member.name)
                if not member.isfile() or not _wanted_member(relative):
                    continue
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError(f"Unsafe path in {path}: {relative}")
                source = bundle.extractfile(member)
                if source is None:
                    continue
                destination = component_root.joinpath(*relative.parts[1:])
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
                extracted += 1
        if not extracted:
            raise RuntimeError(f"No Qt license material found in {path}")

        source_index.extend(
            [
                f"{archive['component']} {archive['version']}",
                f"  Archive: {archive['filename']}",
                f"  SHA-256: {archive['sha256']}",
                f"  Source: {archive['url']}",
                "",
            ]
        )

    (license_output / "QT_CORRESPONDING_SOURCE.txt").write_text(
        "\n".join(source_index), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--download-dir", required=True, type=Path)
    parser.add_argument("--license-output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.manifest, args.download_dir, args.license_output)


if __name__ == "__main__":
    main()
