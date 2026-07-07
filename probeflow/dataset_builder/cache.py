"""In-memory Dataset Builder caches for current-sample and QuickSeg work."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from probeflow.dataset_builder.quickseg import (
    QuickSegParams,
    QuickSegPrepared,
    QuickSegSeed,
)


@dataclass(frozen=True)
class SampleCacheKey:
    path: Path
    plane_index: int
    mtime_ns: Optional[int]
    size_bytes: Optional[int]


@dataclass(frozen=True)
class LoadedSampleRaw:
    key: SampleCacheKey
    scan: Any
    arr: np.ndarray
    px_x_m: float
    px_y_m: float
    flatten_enabled: bool = False
    display_arr: np.ndarray | None = None


@dataclass(frozen=True)
class QuickSegPreprocKey:
    sample_key: SampleCacheKey
    params_fingerprint: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class QuickSegWatershedKey:
    preproc_key: QuickSegPreprocKey
    seed_fingerprint: tuple[tuple[int, int, int, int], ...]


def sample_cache_key(path: str | Path, plane_index: int) -> SampleCacheKey:
    p = Path(path)
    try:
        st = p.stat()
        mtime_ns = st.st_mtime_ns
        size_bytes = st.st_size
    except OSError:
        mtime_ns = None
        size_bytes = None
    return SampleCacheKey(
        path=p.resolve(),
        plane_index=int(plane_index),
        mtime_ns=mtime_ns,
        size_bytes=size_bytes,
    )


def quickseg_params_fingerprint(params: QuickSegParams) -> tuple[tuple[str, Any], ...]:
    payload = dict(params.to_dict())
    items: list[tuple[str, Any]] = []
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, list):
            value = tuple(value)
        elif isinstance(value, dict):
            value = tuple(sorted(value.items()))
        items.append((key, value))
    return tuple(items)


def quickseg_seed_fingerprint(seeds: list[QuickSegSeed]) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (int(seed.order), int(seed.terrace_label_id), int(seed.x), int(seed.y))
        for seed in sorted(seeds, key=lambda s: (int(s.order), int(s.terrace_label_id), int(s.x), int(s.y)))
    )


class DatasetBuilderCache:
    """Small in-memory LRU caches for the Dataset Builder cockpit."""

    def __init__(
        self,
        *,
        max_samples: int = 4,
        max_preproc: int = 4,
        max_watershed: int = 8,
        max_display: int = 8,
    ) -> None:
        self._max_samples = max(1, int(max_samples))
        self._max_preproc = max(1, int(max_preproc))
        self._max_watershed = max(1, int(max_watershed))
        self._max_display = max(1, int(max_display))
        self._raw: OrderedDict[SampleCacheKey, LoadedSampleRaw] = OrderedDict()
        self._display: OrderedDict[tuple[SampleCacheKey, bool], np.ndarray] = OrderedDict()
        self._preproc: OrderedDict[QuickSegPreprocKey, QuickSegPrepared] = OrderedDict()
        self._watershed: OrderedDict[QuickSegWatershedKey, np.ndarray] = OrderedDict()

    def clear(self) -> None:
        self._raw.clear()
        self._display.clear()
        self._preproc.clear()
        self._watershed.clear()

    # Raw sample cache -------------------------------------------------
    def get_sample(self, key: SampleCacheKey) -> LoadedSampleRaw | None:
        value = self._raw.get(key)
        if value is not None:
            self._raw.move_to_end(key)
        return value

    def put_sample(self, value: LoadedSampleRaw) -> None:
        self._raw[value.key] = value
        self._raw.move_to_end(value.key)
        while len(self._raw) > self._max_samples:
            self._raw.popitem(last=False)

    # Display cache ----------------------------------------------------
    def get_display(self, key: SampleCacheKey, flatten_enabled: bool) -> np.ndarray | None:
        value = self._display.get((key, bool(flatten_enabled)))
        if value is not None:
            self._display.move_to_end((key, bool(flatten_enabled)))
        return value

    def put_display(self, key: SampleCacheKey, flatten_enabled: bool, arr: np.ndarray) -> None:
        self._display[(key, bool(flatten_enabled))] = np.asarray(arr, dtype=np.float64)
        self._display.move_to_end((key, bool(flatten_enabled)))
        while len(self._display) > self._max_display:
            self._display.popitem(last=False)

    # QuickSeg preprocess cache ---------------------------------------
    def get_preproc(self, key: QuickSegPreprocKey) -> QuickSegPrepared | None:
        value = self._preproc.get(key)
        if value is not None:
            self._preproc.move_to_end(key)
        return value

    def put_preproc(self, key: QuickSegPreprocKey, value: QuickSegPrepared) -> None:
        self._preproc[key] = value
        self._preproc.move_to_end(key)
        while len(self._preproc) > self._max_preproc:
            self._preproc.popitem(last=False)

    # QuickSeg watershed cache ----------------------------------------
    def get_watershed(self, key: QuickSegWatershedKey) -> np.ndarray | None:
        value = self._watershed.get(key)
        if value is not None:
            self._watershed.move_to_end(key)
        return value

    def put_watershed(self, key: QuickSegWatershedKey, value: np.ndarray) -> None:
        self._watershed[key] = np.asarray(value, dtype=np.int32)
        self._watershed.move_to_end(key)
        while len(self._watershed) > self._max_watershed:
            self._watershed.popitem(last=False)
