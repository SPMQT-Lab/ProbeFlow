"""Tests for the named feature-set store and its AdStat hand-off."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_feature_set_to_point_set_record():
    pytest.importorskip("adstat")
    record = _set("A", 20, 1).to_point_set_record()
    assert len(record.table) == 20
    assert record.region.area_nm2 == pytest.approx(100.0 * 100.0)


def test_single_record_view_spec_has_verdict():
    pytest.importorskip("adstat")
    from probeflow.analysis.adstat_adapter import compare_point_set_record_view_spec

    spec = compare_point_set_record_view_spec(
        _set("A", 120, 1).to_point_set_record(),
        models=("poisson",),
        n_simulations=6,
        random_seed=0,
    )
    stats = {getattr(p, "statistic", "") for p in spec.panels}
    assert "pair_correlation_g_r" in stats
    assert spec.verdict_rows


def test_combined_records_view_spec_pools_replicates():
    pytest.importorskip("adstat")
    from probeflow.analysis.adstat_adapter import compare_point_set_records_view_spec

    records = [_set(f"img{i}", 110 + i, i).to_point_set_record() for i in range(3)]
    spec = compare_point_set_records_view_spec(
        records, models=("poisson",), n_simulations=6, random_seed=0
    )
    stats = {getattr(p, "statistic", "") for p in spec.panels}
    assert "pair_correlation_g_r" in stats
    assert spec.verdict_rows


def test_feature_set_to_feature_layer_is_independent():
    layer = _set("edges", 12, 5).to_feature_layer()
    assert layer["kind"] == "points"
    assert layer["xy_nm"].shape[1] == 2
    assert layer["provenance"]["measured_independently"] is True
    assert layer["provenance"]["derived_from_particles"] is False


def test_measured_feature_record_view_spec_uses_feature_layer():
    pytest.importorskip("adstat")
    from probeflow.analysis.adstat_adapter import compare_point_set_record_view_spec

    particles = _set("particles", 120, 1)
    feature = _set("step edges", 15, 99)
    spec = compare_point_set_record_view_spec(
        particles.to_point_set_record(),
        models=("measured_feature_poisson",),
        feature_layers=[feature.to_feature_layer()],
        n_simulations=6,
        random_seed=7,
    )
    assert spec.metadata.get("active_model") == "measured_feature_poisson"
    assert {
        comparison.ensemble.base_seed
        for comparison in spec.metadata["comparison_results"]
    } == {7}
    assert spec.verdict_rows
    stats = {getattr(p, "statistic", "") for p in spec.panels}
    assert "pair_correlation_g_r" in stats
