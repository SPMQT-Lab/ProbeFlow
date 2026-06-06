"""Tests for the mask sidecar (on-disk persistence)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from probeflow.core.mask import ImageMask, MaskSet
from probeflow.io.mask_sidecar import (
    default_mask_sidecar_path,
    load_mask_set_sidecar,
    mask_sidecar_candidates,
    save_mask_set_sidecar,
)


def test_default_sidecar_path_keeps_dotted_stem(tmp_path):
    scan = tmp_path / "A250320.191933.dat"
    assert default_mask_sidecar_path(scan).name == "A250320.191933.masks.json"


def test_sidecar_candidates_include_provenance_fallbacks(tmp_path):
    scan = tmp_path / "A250320.191933.dat"
    names = [p.name for p in mask_sidecar_candidates(scan)]
    assert names == [
        "A250320.191933.masks.json",
        "A250320.191933.probeflow.json",
        "A250320.191933.provenance.json",
    ]


def test_load_masks_from_exported_provenance_sidecar(tmp_path):
    """Masks embedded in an exported .probeflow.json are auto-discoverable."""
    scan = tmp_path / "A250320.191933.dat"
    ms = MaskSet(image_id="img")
    ms.add(ImageMask.new(np.ones((4, 4), bool), name="m1"))
    prov = tmp_path / "A250320.191933.probeflow.json"
    prov.write_text(json.dumps({"masks": ms.to_dict()}), encoding="utf-8")

    loaded, path_used = load_mask_set_sidecar(scan, missing_ok=True)
    assert loaded is not None
    assert loaded.image_id == "img"
    assert path_used.name == "A250320.191933.probeflow.json"


def test_canonical_masks_json_preferred_over_provenance(tmp_path):
    scan = tmp_path / "scan.sxm"
    canonical = MaskSet(image_id="canonical")
    canonical.add(ImageMask.new(np.ones((4, 4), bool), name="c"))
    save_mask_set_sidecar(canonical, scan)
    (tmp_path / "scan.probeflow.json").write_text(
        json.dumps({"masks": MaskSet(image_id="prov").to_dict()}), encoding="utf-8"
    )
    loaded, path_used = load_mask_set_sidecar(scan)
    assert loaded.image_id == "canonical"
    assert path_used.name == "scan.masks.json"


def test_save_load_roundtrip(tmp_path):
    scan = tmp_path / "scan.sxm"
    ms = MaskSet(image_id="scan")
    data = np.zeros((12, 9), dtype=bool)
    data[2:6, 3:7] = True
    m = ImageMask.new(data, method="canny", parameters={"sigma": 1.0})
    ms.add(m)
    ms.set_active(m.id)

    saved = save_mask_set_sidecar(ms, scan)
    assert saved.exists()

    loaded, path_used = load_mask_set_sidecar(scan)
    assert path_used == saved
    assert loaded.image_id == "scan"
    assert loaded.active_mask_id == m.id
    assert np.array_equal(loaded.get(m.id).data, data)


def test_missing_ok_returns_none(tmp_path):
    scan = tmp_path / "absent.sxm"
    loaded, path_used = load_mask_set_sidecar(scan, missing_ok=True)
    assert loaded is None
    assert path_used == default_mask_sidecar_path(scan)


def test_missing_raises_without_ok(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mask_set_sidecar(tmp_path / "absent.sxm")
