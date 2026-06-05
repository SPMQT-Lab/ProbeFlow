"""Tests for the mask sidecar (on-disk persistence)."""
from __future__ import annotations

import numpy as np
import pytest

from probeflow.core.mask import ImageMask, MaskSet
from probeflow.io.mask_sidecar import (
    default_mask_sidecar_path,
    load_mask_set_sidecar,
    save_mask_set_sidecar,
)


def test_default_sidecar_path_keeps_dotted_stem(tmp_path):
    scan = tmp_path / "A250320.191933.dat"
    assert default_mask_sidecar_path(scan).name == "A250320.191933.masks.json"


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
