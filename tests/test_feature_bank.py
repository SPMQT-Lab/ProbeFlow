"""Tests for the persistent labelled-embedding feature bank."""

import json

import pytest

from probeflow.analysis import feature_bank


def _entry(i, name="disk"):
    return feature_bank.make_entry(
        [0.1 * i, 0.2, 0.3],
        name,
        source_path=f"/scans/scan_{i}.dat",
        particle_index=i,
        bbox_px=(1, 2, 3, 4),
    )


def test_append_and_reload_roundtrip(tmp_path):
    bank_path = tmp_path / "bank" / "feature_bank.json"
    summary = feature_bank.append_entries(bank_path, [_entry(0), _entry(1, "ring")])

    assert summary["added"] == 2
    assert summary["skipped"] == 0
    assert summary["total"] == 2

    bank = feature_bank.load_bank(bank_path)
    assert bank["schema_version"] == feature_bank.BANK_SCHEMA_VERSION
    samples = feature_bank.bank_to_samples(bank)
    assert [name for name, _ in samples] == ["disk", "ring"]
    # No stray temp file left behind by the atomic write.
    assert list(bank_path.parent.iterdir()) == [bank_path]


def test_append_skips_duplicate_source_and_index(tmp_path):
    bank_path = tmp_path / "feature_bank.json"
    feature_bank.append_entries(bank_path, [_entry(0)])
    summary = feature_bank.append_entries(bank_path, [_entry(0), _entry(1)])

    assert summary["added"] == 1
    assert summary["skipped"] == 1
    assert summary["total"] == 2


def test_load_bank_missing_file_is_empty(tmp_path):
    bank = feature_bank.load_bank(tmp_path / "nope.json")
    assert bank["entries"] == []


@pytest.mark.parametrize("payload", ["{not json", '["a", "list"]'])
def test_corrupt_bank_is_empty_for_readers_but_never_overwritten(tmp_path, payload):
    bank_path = tmp_path / "feature_bank.json"
    bank_path.write_text(payload, encoding="utf-8")

    # Read path (classification) degrades gracefully…
    assert feature_bank.load_bank(bank_path)["entries"] == []

    # …but the write path must refuse: silently replacing a corrupt bank
    # would destroy every previously banked sample.
    with pytest.raises(ValueError, match="refusing to overwrite"):
        feature_bank.append_entries(bank_path, [_entry(0)])
    assert bank_path.read_text(encoding="utf-8") == payload


def test_bank_to_samples_skips_malformed_entries(tmp_path):
    bank_path = tmp_path / "feature_bank.json"
    good = _entry(0)
    bank_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "encoder": "clip",
                "entries": [
                    good,
                    {"class_name": "", "embedding": [1.0]},
                    {"class_name": "x", "embedding": []},
                    {"embedding": [1.0]},
                    {"class_name": "y"},
                ],
            }
        ),
        encoding="utf-8",
    )
    samples = feature_bank.bank_to_samples(feature_bank.load_bank(bank_path))
    assert len(samples) == 1
    assert samples[0][0] == "disk"


def test_make_entry_records_scale_fields():
    """Schema-2 entries carry the crop scale + pipeline id used to gate matching."""
    e = feature_bank.make_entry(
        [1.0, 2.0], "disk",
        source_path="/s/a.dat", particle_index=3,
        embed_version="physfov-mask-v1",
        pixel_size_nm=0.3, fov_nm=15.0, out_px=96, area_nm2=12.5,
    )
    assert e["embed_version"] == "physfov-mask-v1"
    assert e["pixel_size_nm"] == 0.3
    assert e["fov_nm"] == 15.0
    assert e["out_px"] == 96
    assert e["area_nm2"] == 12.5


def test_make_entry_scale_fields_default_none_for_legacy():
    """Old-style calls still work; scale fields default to None (legacy)."""
    e = feature_bank.make_entry([1.0], "disk", source_path="/s/a.dat",
                                particle_index=0)
    assert e["embed_version"] is None
    assert e["fov_nm"] is None


def test_bank_to_samples_skips_stale_entries():
    bank = {"entries": [
        feature_bank.make_entry([1.0], "keep", source_path="/s/a.dat",
                                particle_index=0),
        {**feature_bank.make_entry([1.0], "drop", source_path="/s/b.dat",
                                   particle_index=1), "stale": True},
    ]}
    names = [n for n, _ in feature_bank.bank_to_samples(bank)]
    assert names == ["keep"]


def test_class_counts():
    entries = [_entry(0), _entry(1), _entry(2, "ring"), {"embedding": [1.0]}]
    assert feature_bank.class_counts(entries) == {"disk": 2, "ring": 1, "?": 1}
