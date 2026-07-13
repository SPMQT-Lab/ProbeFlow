"""Stable source identity helpers for decoded ProbeFlow data."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def build_source_identity(
    path,
    *,
    source_format: str,
    item_type: str,
    data_offset: int | None = None,
) -> dict[str, Any]:
    """Return a JSON-serialisable identity record for a source file."""

    p = Path(path)
    stat = p.stat()
    return {
        "source_path": str(p),
        "source_format": source_format,
        "item_type": item_type,
        "file_size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(p),
        "data_offset": int(data_offset) if data_offset is not None else None,
    }


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Backward-compatible private spelling retained for existing internal callers.
_sha256_file = sha256_file
