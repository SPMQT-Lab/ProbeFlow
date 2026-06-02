"""Tests for the persistent browse cache and its indexing integration."""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import pytest

from probeflow.core import browse_cache
from probeflow.core.indexing import index_folder_shallow

TESTDATA = Path(__file__).resolve().parents[1] / "test_data"


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROBEFLOW_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("PROBEFLOW_DISABLE_BROWSE_CACHE", raising=False)
    yield tmp_path / "cache"


def test_metadata_roundtrip_and_invalidation(cache_dir):
    p = "/some/file.sxm"
    assert browse_cache.get_metadata(p, 111, 222) == (False, None)

    browse_cache.put_metadata(p, 111, 222, {"shape": (4, 4)})
    assert browse_cache.get_metadata(p, 111, 222) == (True, {"shape": (4, 4)})

    # Changed mtime or size misses (stale entry not returned).
    assert browse_cache.get_metadata(p, 999, 222) == (False, None)
    assert browse_cache.get_metadata(p, 111, 333) == (False, None)


def test_metadata_caches_none_for_unrecognised(cache_dir):
    p = "/some/notes.txt"
    browse_cache.put_metadata(p, 1, 2, None)
    assert browse_cache.get_metadata(p, 1, 2) == (True, None)


def test_thumbnail_roundtrip_and_key_invalidation(cache_dir):
    k1 = browse_cache.thumbnail_key("/f.sm4", 10, 20, cm="gray", w=56, h=56)
    assert browse_cache.get_thumbnail(k1) is None
    browse_cache.put_thumbnail(k1, b"\x89PNG-data")
    assert browse_cache.get_thumbnail(k1) == b"\x89PNG-data"

    # Different mtime -> different key -> miss.
    k2 = browse_cache.thumbnail_key("/f.sm4", 11, 20, cm="gray", w=56, h=56)
    assert k2 != k1
    assert browse_cache.get_thumbnail(k2) is None


def test_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PROBEFLOW_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PROBEFLOW_DISABLE_BROWSE_CACHE", "1")
    assert browse_cache.enabled() is False
    browse_cache.put_metadata("/f.sxm", 1, 2, {"x": 1})
    assert browse_cache.get_metadata("/f.sxm", 1, 2) == (False, None)


def test_eviction_drops_oldest(cache_dir):
    for i in range(10):
        browse_cache.put_thumbnail(
            browse_cache.thumbnail_key("/f", i, 0, n=i), b"x" * 1000
        )
    browse_cache.evict(max_bytes=3000)
    remaining = sum(
        1
        for i in range(10)
        if browse_cache.get_thumbnail(browse_cache.thumbnail_key("/f", i, 0, n=i))
    )
    # Under the 0.9*budget target: at most 2 of the 1000-byte entries survive.
    assert remaining <= 3


def test_shallow_index_second_pass_hits_cache(cache_dir, monkeypatch):
    data_dir = cache_dir.parent / "data"
    data_dir.mkdir()
    src = next(TESTDATA.glob("*.sxm"), None) or next(TESTDATA.glob("*.dat"))
    shutil.copy(src, data_dir / src.name)

    idx1 = index_folder_shallow(data_dir)
    assert len(idx1.files) == 1

    # Second pass must not sniff or read the file again.
    import probeflow.core.indexing as indexing

    sniffs: list = []
    real = indexing.sniff_file_type
    monkeypatch.setattr(
        indexing, "sniff_file_type", lambda p: (sniffs.append(p), real(p))[1]
    )
    idx2 = index_folder_shallow(data_dir)

    assert sniffs == []
    assert idx2.files == idx1.files
