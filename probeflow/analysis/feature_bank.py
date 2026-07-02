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

KNOWN GAP (physics review 2026-07-02, not yet fixed): entries record no pixel
size, but crops are a fixed 48 px (see ``embed_particles_clip`` in
``probeflow.analysis.features``), so a banked sample's physical field of view
depends on its source scan's resolution. CLIP embeddings are not scale
invariant, so cross-scan classification is scale-blind and there is currently
no way to detect the mismatch after the fact. See ``make_entry`` below for
where to add the fix.
"""

from __future__ import annotations

import json
import os
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


def load_bank(path, *, strict: bool = False) -> dict:
    """Load a bank file, or an empty bank when it is missing / unreadable.

    ``strict=True`` raises :class:`ValueError` instead of returning an empty
    bank when the file exists but cannot be parsed as a bank. Read-only
    consumers (classification) want the permissive default; writers must use
    strict so a corrupt file is never silently replaced, destroying every
    previously banked sample.
    """
    p = Path(path)
    if not p.exists():
        return _empty_bank()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        if strict:
            raise ValueError(
                f"feature bank {p} exists but is not readable as JSON ({exc}); "
                "refusing to overwrite it — repair or remove the file first"
            ) from exc
        return _empty_bank()
    if not isinstance(data, dict):
        if strict:
            raise ValueError(
                f"feature bank {p} does not contain a JSON object; "
                "refusing to overwrite it — repair or remove the file first"
            )
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
    """Build one bank entry (a labelled CLIP embedding + provenance).

    TODO(scale-blind bank, see module docstring): while the schema is still
    young (``BANK_SCHEMA_VERSION`` above), consider adding the source scan's
    pixel size (or the crop's physical size in nm) here so Phase-2
    classification can warn on or weight by a scale mismatch between the bank
    entry and the particle being classified.
    """
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
    # Materialise up front: a generator would be consumed by the loop below,
    # making the ``skipped`` count in the summary wrong (negative).
    new_entries = list(new_entries)
    # strict: a corrupt existing bank must abort the append, not be silently
    # replaced by an empty bank plus the new entries.
    bank = load_bank(p, strict=True)
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
    # Atomic replace: an interrupted write must never leave a truncated bank
    # in place of the previous good one.
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return {
        "added": added,
        "skipped": len(new_entries) - added,
        "total": len(bank["entries"]),
        "path": str(p),
    }


def bank_to_samples(bank: dict) -> list:
    """Return ``(class_name, embedding)`` pairs from a loaded bank.

    Malformed entries (missing name or empty embedding) are skipped rather than
    raised, so one corrupt row can't block classification with the rest of the
    bank. This is the Phase-2 read path: the pairs feed straight into
    ``classify_particles(bank_samples=...)``.
    """
    out = []
    for e in bank.get("entries", []):
        name = e.get("class_name")
        emb = e.get("embedding")
        if isinstance(name, str) and name and isinstance(emb, list) and emb:
            out.append((name, emb))
    return out
