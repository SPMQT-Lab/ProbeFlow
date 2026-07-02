"""Persistent bank of labelled CLIP feature embeddings for classification.

Phase 1 (this module): accumulate human-labelled sample embeddings — class name +
512-d CLIP vector + provenance — into a JSON file, so future classification can
draw on examples gathered across many scans instead of only the ones labelled in
the current image. Adding to the bank is a deliberate, human-confirmed action
(the GUI double-checks before writing). Loading/using the bank to actually drive
classification is Phase 2.

GUI-free and dependency-light (json + numpy only): the embeddings are produced by
``probeflow.analysis.features._embed_clip`` upstream and passed in as plain lists,
so this module never imports torch/clip.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

BANK_SCHEMA_VERSION = 1


def default_bank_path() -> Path:
    """Default on-disk location for the shared feature bank."""
    return Path.home() / ".probeflow" / "feature_bank.json"


def _empty_bank() -> dict:
    return {"schema_version": BANK_SCHEMA_VERSION, "encoder": "clip", "entries": []}


def load_bank(path) -> dict:
    """Load a bank file, or an empty bank when it is missing / unreadable."""
    p = Path(path)
    if not p.exists():
        return _empty_bank()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return _empty_bank()
    if not isinstance(data, dict):
        return _empty_bank()
    data.setdefault("schema_version", BANK_SCHEMA_VERSION)
    data.setdefault("encoder", "clip")
    entries = data.get("entries")
    data["entries"] = entries if isinstance(entries, list) else []
    return data


def make_entry(
    embedding,
    class_name: str,
    *,
    source_path,
    particle_index: int,
    bbox_px=None,
) -> dict:
    """Build one bank entry (a labelled CLIP embedding + provenance)."""
    return {
        "class_name": str(class_name),
        "embedding": [float(x) for x in np.asarray(embedding, dtype=float).ravel()],
        "source_path": str(source_path) if source_path is not None else None,
        "particle_index": int(particle_index),
        "bbox_px": [int(v) for v in bbox_px] if bbox_px is not None else None,
        "added_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def class_counts(entries: Sequence[dict]) -> dict:
    """Count entries per class name (for confirmation dialogs / summaries)."""
    counts: dict = {}
    for e in entries:
        name = e.get("class_name", "?")
        counts[name] = counts.get(name, 0) + 1
    return counts


def append_entries(path, new_entries: Sequence[dict]) -> dict:
    """Append entries to the bank file, creating it (and its parent dir) as needed.

    Duplicates — same ``(source_path, particle_index)`` as an entry already in the
    bank — are skipped so re-banking the same labelled sample from the same scan
    doesn't accumulate copies. Returns a summary dict:
    ``{"added", "skipped", "total", "path"}``.
    """
    p = Path(path)
    bank = load_bank(p)
    seen = {
        (e.get("source_path"), e.get("particle_index"))
        for e in bank["entries"]
    }
    added = 0
    for entry in new_entries:
        key = (entry.get("source_path"), entry.get("particle_index"))
        if key in seen:
            continue
        bank["entries"].append(entry)
        seen.add(key)
        added += 1
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    return {
        "added": added,
        "skipped": len(list(new_entries)) - added,
        "total": len(bank["entries"]),
        "path": str(p),
    }
