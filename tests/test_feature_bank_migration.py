"""Migration of a legacy (schema-1, scale-blind) bank to the physical-FOV schema.

The migration re-embeds legacy entries from their source scans at the fixed
physical FOV, preserving human labels; entries whose source scan is gone are
marked stale (never deleted). CLIP and real scan I/O are stubbed so the test
runs anywhere.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from probeflow.analysis import feature_bank
from probeflow.analysis.features import EMBED_VERSION

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_feature_bank.py"


def _load_migration_module():
    spec = importlib.util.spec_from_file_location("pf_migrate_bank_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _legacy_entry(name, source_path, particle_index=0):
    # Schema-1 style: no embed_version / scale fields.
    return {
        "class_name": name,
        "embedding": [0.1, 0.2, 0.3],
        "source_path": str(source_path) if source_path else None,
        "particle_index": particle_index,
        "bbox_px": [10, 10, 30, 30],
        "added_utc": "2026-01-01T00:00:00Z",
    }


def test_migrate_reembeds_reachable_and_marks_missing_stale(tmp_path, monkeypatch):
    mod = _load_migration_module()

    # A reachable source scan (file must exist for _load_plane) and a missing one.
    good_src = tmp_path / "good.dat"
    good_src.write_bytes(b"stub")
    missing_src = tmp_path / "gone.dat"  # never created

    bank_path = tmp_path / "feature_bank.json"
    bank_path.write_text(json.dumps({
        "schema_version": 1,
        "encoder": "clip",
        "entries": [
            _legacy_entry("disk", good_src, 0),
            _legacy_entry("ring", missing_src, 1),
        ],
    }), encoding="utf-8")

    # Stub scan loading + CLIP embedding (no real files / torch needed).
    def _fake_load_scan(path):
        return SimpleNamespace(
            planes=[np.zeros((256, 256), dtype=np.float64)],
            scan_range_m=(50e-9, 50e-9),   # 0.195 nm/px → 15 nm FOV ≈ 77 px
        )

    def _fake_embed(arr, particles, **kw):
        return np.ones((len(list(particles)), 512), dtype=np.float64)

    monkeypatch.setattr(mod, "load_scan", _fake_load_scan)
    monkeypatch.setattr(mod, "embed_particles_clip", _fake_embed)

    result = mod.migrate(bank_path, dry_run=False)
    assert result["migrated"] == 1
    assert result["stale"] == 1
    assert result["written"] is True

    # Backup of the original written before the atomic rewrite.
    assert (tmp_path / "feature_bank.json.bak").exists()

    bank = feature_bank.load_bank(bank_path)
    assert bank["schema_version"] == feature_bank.BANK_SCHEMA_VERSION
    by_name = {e["class_name"]: e for e in bank["entries"]}

    # Reachable entry: re-embedded with full scale metadata, usable by readers.
    disk = by_name["disk"]
    assert disk["embed_version"] == EMBED_VERSION
    assert disk["fov_nm"] is not None and disk["out_px"] is not None
    assert disk["area_nm2"] and disk["area_nm2"] > 0
    assert not disk.get("stale")

    # Missing-source entry: kept but marked stale (excluded by readers).
    ring = by_name["ring"]
    assert ring.get("stale") is True
    assert "ring" not in [n for n, _ in feature_bank.bank_to_samples(bank)]

    # A second run is idempotent — the already-current entry isn't re-embedded.
    result2 = mod.migrate(bank_path, dry_run=True)
    assert result2["already_current"] == 1
    assert result2["migrated"] == 0
