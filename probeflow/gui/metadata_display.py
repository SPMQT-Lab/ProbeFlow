"""Helpers for presenting structured acquisition metadata in tables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def metadata_rows(metadata: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return nested metadata as readable key/value table rows.

    SM4 headers contain a list of per-page dictionaries.  Rendering that list
    with ``str`` produces one very long, unreadable value.  Recursing through
    mappings and sequences keeps each metadata field in its own row while
    preserving the original values.
    """
    rows: list[tuple[str, str]] = []
    for key in sorted(metadata, key=str):
        value = metadata[key]
        if key == "pages" and isinstance(value, list):
            for index, page in enumerate(value, start=1):
                if not isinstance(page, Mapping):
                    rows.extend(_metadata_rows(f"Page {index}", page))
                    continue
                label = page.get("label") or page.get("page_type_label") or "Page"
                direction = page.get("scan_dir_label")
                title = f"Page {index} — {label}"
                if direction:
                    title += f" [{direction}]"
                rows.append((title, ""))
                for field in sorted(page, key=str):
                    rows.extend(_metadata_rows(str(field), page[field]))
        else:
            rows.extend(_metadata_rows(str(key), value))
    return rows


def _metadata_rows(path: str, value: Any) -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        if not value:
            return [(path, "{}")]
        rows: list[tuple[str, str]] = []
        for key in sorted(value, key=str):
            child_path = f"{path}.{key}" if path else str(key)
            rows.extend(_metadata_rows(child_path, value[key]))
        return rows

    if isinstance(value, (list, tuple)):
        if not value:
            return [(path, "[]")]
        rows = []
        for index, child in enumerate(value, start=1):
            rows.extend(_metadata_rows(f"{path}[{index}]", child))
        return rows

    return [(path, str(value))]
