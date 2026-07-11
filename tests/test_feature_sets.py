"""Tests for the named feature-set store."""

from __future__ import annotations

import numpy as np

from probeflow.measurements.feature_sets import FeatureSet, FeatureSetStore


def _set(name: str, n: int, seed: int) -> FeatureSet:
    rng = np.random.default_rng(seed)
    xy_nm = rng.uniform(0.0, 100.0, size=(n, 2))
    return FeatureSet.from_points(
        name=name,
        points_px=xy_nm,  # treat 1 px = 1 nm here for simplicity
        points_m=xy_nm * 1e-9,
        scan_range_m=(100e-9, 100e-9),
        image_shape=(256, 256),
        image_label=name,
        metadata={"detection_mode": "maxima"},
    )


def test_feature_set_store_add_rename_remove():
    store = FeatureSetStore()
    a = store.add(_set("A", 10, 1))
    store.add(_set("B", 12, 2))

    assert len(store) == 2
    assert store.rename(a, "A renamed")
    assert store.get(a).name == "A renamed"
    assert store.remove(a)
    assert len(store) == 1
    assert store.get(a) is None


def test_feature_set_store_json_round_trip(tmp_path):
    store = FeatureSetStore()
    store.add(_set("A", 8, 1))
    store.add(_set("B", 9, 2))
    path = tmp_path / "sets.json"
    store.save(path)

    loaded = FeatureSetStore.load(path)
    assert [fs.name for fs in loaded.all()] == ["A", "B"]
    assert [fs.point_count for fs in loaded.all()] == [8, 9]
    # points survive the round trip
    np.testing.assert_allclose(
        loaded.all()[0].points_m, store.all()[0].points_m
    )


def test_feature_set_load_missing_file_is_empty(tmp_path):
    assert len(FeatureSetStore.load(tmp_path / "nope.json")) == 0
