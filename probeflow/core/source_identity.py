"""Stable source identity helpers for decoded ProbeFlow data."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


_PATH_FIELD_NAMES = {
    "path",
    "source_path",
    "source_file",
    "source_filename",
    "output_path",
    "export_path",
    "bundle_dir",
    "data_dir",
    "image_id",
    "file",
    "filename",
}
_IDENTITY_FIELD_NAMES = {
    "user",
    "username",
    "operator",
    "experimenter",
    "scientist",
    "author",
    "owner",
    "hostname",
    "computername",
    "workstation",
    "email",
    "phone",
    "address",
}


def privacy_safe_path(value: Any) -> str | None:
    """Return only the final component of a filesystem path.

    Both POSIX and Windows separators are handled regardless of the platform
    running ProbeFlow.  Exported metadata uses this representation by default
    so account names and directory layouts do not escape in sidecars.
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return text
    return text.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def sanitize_export_data(value: Any, *, field_name: str | None = None) -> Any:
    """Recursively remove local identity and directory information."""

    key = "" if field_name is None else str(field_name)
    normalized = "".join(ch for ch in key.lower() if ch.isalnum() or ch == "_")
    collapsed = normalized.replace("_", "")
    if (
        collapsed in _IDENTITY_FIELD_NAMES
        or "username" in collapsed
    ):
        return "" if value is not None else None
    is_path_field = (
        normalized in _PATH_FIELD_NAMES
        or collapsed in {name.replace("_", "") for name in _PATH_FIELD_NAMES}
        or collapsed.endswith((
            "path",
            "directory",
            "filename",
            "scanfile",
            "sourcefile",
            "memofile",
            "linkfile",
        ))
        or collapsed.endswith("dir")
    )
    if is_path_field and isinstance(value, (str, Path)):
        return privacy_safe_path(value)
    if isinstance(value, Path):
        return privacy_safe_path(value)
    if isinstance(value, dict):
        return {
            str(child_key): sanitize_export_data(child, field_name=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_export_data(item) for item in value]
    return value


def sanitize_header_for_export(header: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an instrument header safe for exported artifacts."""

    return sanitize_export_data(dict(header))


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
        "source_path": privacy_safe_path(p),
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
