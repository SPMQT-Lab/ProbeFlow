"""Dataset Builder export validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def validate_dataset(path: str | Path) -> dict:
    """Validate a Dataset Builder export directory."""

    root = Path(path)
    manifest = root / "manifest.csv"
    errors: list[str] = []
    if not manifest.exists():
        return {"ok": False, "errors": [f"missing manifest: {manifest}"], "n_rows": 0}

    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    for index, row in enumerate(rows, 1):
        for col in ("array_path", "preview_path", "provenance_path"):
            rel = row.get(col) or ""
            if rel and not (root / rel).exists():
                errors.append(f"row {index}: missing {col} {rel}")
        mask_paths_json = row.get("mask_paths_json") or "{}"
        try:
            mask_paths = json.loads(mask_paths_json)
        except json.JSONDecodeError:
            errors.append(f"row {index}: invalid mask_paths_json")
            mask_paths = {}
        for rel in mask_paths.values():
            if rel and not (root / rel).exists():
                errors.append(f"row {index}: missing mask path {rel}")
    return {"ok": not errors, "errors": errors, "n_rows": len(rows)}

