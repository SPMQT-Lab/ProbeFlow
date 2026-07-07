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

SCALE GAP FIX (schema 2, 2026-07-07): entries now record the physical crop
scale — ``pixel_size_nm``, ``fov_nm``, ``out_px``, ``area_nm2`` — and the
``embed_version`` of the crop/mask/encode pipeline that produced the embedding
(``probeflow.analysis.features.EMBED_VERSION``). Classification uses these to
only compare embeddings computed at the same scale by the same pipeline (see
``select_bank_samples``); mismatched or legacy (schema-1, unscaled) entries are
excluded rather than silently mis-matched. Legacy banks are re-embedded by
``scripts/migrate_feature_bank.py`` where the source scan is still reachable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

BANK_SCHEMA_VERSION = 2


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
    embed_version: Optional[str] = None,
    pixel_size_nm: Optional[float] = None,
    fov_nm: Optional[float] = None,
    out_px: Optional[int] = None,
    area_nm2: Optional[float] = None,
) -> dict:
    """Build one bank entry (a labelled CLIP embedding + provenance + scale).

    The scale fields (schema 2) let classification only compare embeddings
    computed at the same physical scale by the same pipeline:

    * ``embed_version`` — the crop/mask/encode pipeline id
      (``probeflow.analysis.features.EMBED_VERSION``). Entries whose version
      differs from the current classifier are excluded, never mis-matched.
    * ``pixel_size_nm`` / ``fov_nm`` / ``out_px`` — the physical crop geometry.
    * ``area_nm2`` — the labelled particle's physical area, for the optional
      size gate in ``classify_particles``.

    All scale fields default to ``None`` for back-compat; an entry without
    ``embed_version`` is treated as legacy/unscaled and skipped by
    ``select_bank_samples`` until re-embedded by the migration script.
    """
    return {
        "class_name": str(class_name),
        "embedding": [float(x) for x in np.asarray(embedding, dtype=float).ravel()],
        "source_path": str(source_path) if source_path is not None else None,
        "particle_index": int(particle_index),
        "bbox_px": [int(v) for v in bbox_px] if bbox_px is not None else None,
        "embed_version": str(embed_version) if embed_version is not None else None,
        "pixel_size_nm": float(pixel_size_nm) if pixel_size_nm is not None else None,
        "fov_nm": float(fov_nm) if fov_nm is not None else None,
        "out_px": int(out_px) if out_px is not None else None,
        "area_nm2": float(area_nm2) if area_nm2 is not None else None,
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

    Malformed entries (missing name or empty embedding) and entries marked
    ``stale`` (migration could not re-embed them) are skipped rather than
    raised, so one bad row can't block classification with the rest of the
    bank. Note: this does NOT enforce scale/pipeline compatibility — use
    :func:`select_bank_samples` for the classifier read path.
    """
    out = []
    for e in bank.get("entries", []):
        if e.get("stale"):
            continue
        name = e.get("class_name")
        emb = e.get("embedding")
        if isinstance(name, str) and name and isinstance(emb, list) and emb:
            out.append((name, emb))
    return out


def select_bank_samples(
    bank: dict,
    *,
    embed_version: str,
    fov_nm: float,
    out_px: int,
    fov_tol: float = 0.05,
) -> dict:
    """Pick bank entries comparable to the current classifier configuration.

    Only entries produced by the same crop/mask/encode pipeline
    (``embed_version``) at the same physical FOV and output grid are safe to
    compare against the current scan's embeddings. Everything else — legacy
    (schema-1, no ``embed_version``), a different pipeline, a different FOV, or
    a ``stale`` entry — is excluded and counted by reason, so the caller can
    tell the operator *why* N banked samples were skipped instead of silently
    mis-matching them (the original scale-blind bug).

    Returns ``{"names", "embeddings", "areas", "kept", "skipped", "reasons"}``
    where ``names``/``embeddings``/``areas`` are aligned lists ready to feed
    ``classify_particles(bank_samples=zip(names, embeddings), bank_areas=areas)``.
    """
    names: list[str] = []
    embeddings: list[list] = []
    areas: list = []
    reasons = {"legacy": 0, "pipeline": 0, "fov": 0, "stale": 0, "malformed": 0}

    for e in bank.get("entries", []):
        name = e.get("class_name")
        emb = e.get("embedding")
        if not (isinstance(name, str) and name and isinstance(emb, list) and emb):
            reasons["malformed"] += 1
            continue
        if e.get("stale"):
            reasons["stale"] += 1
            continue
        ev = e.get("embed_version")
        if ev is None:
            reasons["legacy"] += 1
            continue
        if ev != embed_version:
            reasons["pipeline"] += 1
            continue
        e_fov, e_out = e.get("fov_nm"), e.get("out_px")
        if (e_out != int(out_px) or e_fov is None
                or abs(float(e_fov) - float(fov_nm)) > fov_tol * float(fov_nm)):
            reasons["fov"] += 1
            continue
        names.append(name)
        embeddings.append(emb)
        areas.append(e.get("area_nm2"))

    return {
        "names": names,
        "embeddings": embeddings,
        "areas": areas,
        "kept": len(names),
        "skipped": sum(reasons.values()),
        "reasons": reasons,
    }
