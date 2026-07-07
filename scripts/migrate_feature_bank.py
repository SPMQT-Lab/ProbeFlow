"""Re-embed a legacy (scale-blind) feature bank into the physical-FOV schema.

Why this exists
---------------
Schema-1 bank entries stored a CLIP embedding computed from a fixed 48-*pixel*
crop, whose physical field of view varied per scan — so cross-scan matching was
scale-blind (see ``probeflow/analysis/feature_bank.py``). Schema-2 embeddings are
computed at a fixed physical FOV and carry their scale + pipeline id
(``EMBED_VERSION``). This script re-embeds each legacy entry at the new FOV from
its source scan, preserving the accumulated human labels instead of wiping the
bank.

Behaviour
---------
* Entries already at the current ``EMBED_VERSION`` are left untouched (idempotent).
* Legacy entries with a reachable ``source_path`` + ``bbox_px`` are re-embedded
  from the scan's topography plane and rewritten with full scale metadata.
* Entries whose source scan is missing/unreadable, or that lack a ``bbox_px``,
  are marked ``"stale": true`` and excluded by readers — never deleted.
* A ``.bak`` copy of the original bank is written before the (atomic) rewrite.

Limitation: re-embedding uses the raw topography plane, which may differ slightly
from the flattened array the sample was originally banked from. That is inherent
(the exact processing isn't recorded) and still far better than a scale-blind
match; migrated entries are re-labelled with the current pipeline id.

Usage
-----
    python -m scripts.migrate_feature_bank [--bank PATH] [--dry-run]

Requires the CLIP extra (``pip install 'probeflow[clip]'``) — re-embedding runs
the encoder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from probeflow.analysis import feature_bank
from probeflow.analysis.features import (
    DEFAULT_CROP_FOV_NM,
    DEFAULT_OUT_PX,
    EMBED_VERSION,
    clip_available,
    embed_particles_clip,
)
from probeflow.core.scan_loader import load_scan


def _load_plane(source_path: str):
    """Return ``(topo_plane, px_x_m, px_y_m)`` for a scan, or None if unreadable."""
    p = Path(source_path)
    if not p.exists():
        return None
    try:
        scan = load_scan(p)
        arr = np.asarray(scan.planes[0], dtype=np.float64)
        ny, nx = arr.shape
        rx, ry = scan.scan_range_m
        px_x = float(rx) / max(nx, 1)
        px_y = float(ry) / max(ny, 1)
        if px_x <= 0 or px_y <= 0:
            return None
        return arr, px_x, px_y
    except Exception as exc:  # unreadable / wrong format / corrupt
        print(f"    ! could not load {source_path}: {exc}")
        return None


def _migrate_entry(entry: dict) -> dict:
    """Return a re-embedded entry, or the entry marked stale when impossible."""
    if entry.get("embed_version") == EMBED_VERSION:
        return entry  # already current
    source = entry.get("source_path")
    bbox = entry.get("bbox_px")
    name = entry.get("class_name")
    if not source or not bbox or not name:
        return {**entry, "stale": True}
    loaded = _load_plane(source)
    if loaded is None:
        return {**entry, "stale": True}
    arr, px_x, px_y = loaded
    x0, y0, x1, y1 = (int(v) for v in bbox)
    particle = SimpleNamespace(
        index=int(entry.get("particle_index", 0)),
        bbox_px=(x0, y0, x1, y1),
        contour_xy_m=[],  # mask falls back to the bbox rectangle
    )
    try:
        embs = embed_particles_clip(
            arr, [particle],
            pixel_size_x_m=px_x, pixel_size_y_m=px_y,
            fov_nm=DEFAULT_CROP_FOV_NM, out_px=DEFAULT_OUT_PX,
        )
    except ValueError as exc:  # e.g. scan too coarse for the FOV
        print(f"    ! {source}: {exc}")
        return {**entry, "stale": True}
    area_nm2 = abs((x1 - x0) * px_x * (y1 - y0) * px_y) * 1e18
    return feature_bank.make_entry(
        embs[0], name,
        source_path=source,
        particle_index=int(entry.get("particle_index", 0)),
        bbox_px=(x0, y0, x1, y1),
        embed_version=EMBED_VERSION,
        pixel_size_nm=float(np.sqrt(px_x * px_y)) * 1e9,
        fov_nm=DEFAULT_CROP_FOV_NM,
        out_px=DEFAULT_OUT_PX,
        area_nm2=area_nm2,
    )


def migrate(bank_path: Path, *, dry_run: bool = False) -> dict:
    bank = feature_bank.load_bank(bank_path, strict=True)
    entries = bank.get("entries", [])
    new_entries = []
    counts = {"already_current": 0, "migrated": 0, "stale": 0}
    for e in entries:
        out = _migrate_entry(e)
        if out.get("embed_version") == EMBED_VERSION and not out.get("stale"):
            if e.get("embed_version") == EMBED_VERSION:
                counts["already_current"] += 1
            else:
                counts["migrated"] += 1
        else:
            counts["stale"] += 1
        new_entries.append(out)

    if dry_run:
        return {**counts, "path": str(bank_path), "written": False}

    bank["schema_version"] = feature_bank.BANK_SCHEMA_VERSION
    bank["entries"] = new_entries
    if Path(bank_path).exists():
        shutil.copy2(bank_path, str(bank_path) + ".bak")
    tmp = Path(str(bank_path) + ".tmp")
    tmp.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    os.replace(tmp, bank_path)
    return {**counts, "path": str(bank_path), "written": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default=str(feature_bank.default_bank_path()),
                    help="Path to the feature bank JSON (default: ~/.probeflow/…).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args(argv)

    if not clip_available():
        print("CLIP not available — install 'probeflow[clip]' to re-embed.",
              file=sys.stderr)
        return 2
    bank_path = Path(args.bank)
    if not bank_path.exists():
        print(f"No bank at {bank_path} — nothing to migrate.")
        return 0

    print(f"Migrating feature bank: {bank_path}"
          + ("  (dry run)" if args.dry_run else ""))
    result = migrate(bank_path, dry_run=args.dry_run)
    print(f"  already current : {result['already_current']}")
    print(f"  re-embedded     : {result['migrated']}")
    print(f"  marked stale    : {result['stale']}")
    if result["written"]:
        print(f"  wrote {bank_path} (backup: {bank_path}.bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
