"""Safe filesystem actions used by the browse UI."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import shutil


def create_folder(parent: Path, name: str) -> Path:
    parent = Path(parent)
    name = name.strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("Enter a single folder name.")
    target = parent / name
    target.mkdir()
    return target


def move_files(sources: Iterable[Path], destination: Path) -> list[Path]:
    destination = Path(destination)
    if not destination.is_dir():
        raise NotADirectoryError(destination)

    unique_sources = list(dict.fromkeys(Path(source) for source in sources))
    if not unique_sources:
        return []
    destination_resolved = destination.resolve()
    if any(source.resolve().parent == destination_resolved for source in unique_sources):
        raise ValueError("The destination is already the source folder.")
    if any(not source.is_file() for source in unique_sources):
        raise FileNotFoundError("One or more selected files no longer exist.")

    targets = [destination / source.name for source in unique_sources]
    collisions = [target.name for target in targets if target.exists()]
    if collisions:
        raise FileExistsError(
            "Destination already contains: " + ", ".join(collisions)
        )

    for source, target in zip(unique_sources, targets):
        shutil.move(str(source), str(target))
    return targets
