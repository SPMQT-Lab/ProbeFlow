"""Named, persistable feature-point sets for multi-image Particle Statistics.

A :class:`FeatureSet` is a compact snapshot of one image's detected points
(pixel + metre coordinates) plus the calibration needed to analyse them. Unlike
point ROIs (one ROI per point), a set stores its points as arrays, so hundreds of
points stay lightweight and many sets — e.g. one per image in a study — can
coexist in a :class:`FeatureSetStore` and be pooled in Particle Statistics.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np


def _xy(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("feature points must have shape (N, 2)")
    return arr


@dataclass(frozen=True)
class FeatureSet:
    """One named set of detected points with its image calibration."""

    name: str
    points_px: np.ndarray
    points_m: np.ndarray
    scan_range_m: tuple[float, float]
    image_shape: tuple[int, int]
    source_type: str = "feature_finder"
    image_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    set_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)

    @property
    def point_count(self) -> int:
        return int(len(self.points_m))

    # ---- construction ------------------------------------------------------
    @classmethod
    def from_points(
        cls,
        *,
        name: str,
        points_px: Any,
        points_m: Any,
        scan_range_m: tuple[float, float],
        image_shape: tuple[int, int],
        source_type: str = "feature_finder",
        image_label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "FeatureSet":
        return cls(
            name=str(name),
            points_px=_xy(points_px),
            points_m=_xy(points_m),
            scan_range_m=(float(scan_range_m[0]), float(scan_range_m[1])),
            image_shape=(int(image_shape[0]), int(image_shape[1])),
            source_type=str(source_type),
            image_label=str(image_label),
            metadata=dict(metadata or {}),
        )

    # ---- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "set_id": self.set_id,
            "name": self.name,
            "points_px": self.points_px.tolist(),
            "points_m": self.points_m.tolist(),
            "scan_range_m": list(self.scan_range_m),
            "image_shape": list(self.image_shape),
            "source_type": self.source_type,
            "image_label": self.image_label,
            "metadata": dict(self.metadata),
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSet":
        return cls(
            name=str(data.get("name", "Feature set")),
            points_px=_xy(data.get("points_px", [])),
            points_m=_xy(data.get("points_m", [])),
            scan_range_m=tuple(float(v) for v in data.get("scan_range_m", (1.0, 1.0)))[:2],
            image_shape=tuple(int(v) for v in data.get("image_shape", (1, 1)))[:2],
            source_type=str(data.get("source_type", "feature_finder")),
            image_label=str(data.get("image_label", "")),
            metadata=dict(data.get("metadata", {}) or {}),
            set_id=str(data.get("set_id") or uuid.uuid4().hex[:12]),
            created=float(data.get("created", time.time())),
        )


class FeatureSetStore:
    """An ordered, named collection of :class:`FeatureSet` records.

    Lives at the application/viewer level so sets survive switching images; can be
    saved/loaded so they survive sessions.
    """

    def __init__(self, sets: list[FeatureSet] | None = None):
        self._sets: list[FeatureSet] = list(sets or [])

    def __len__(self) -> int:
        return len(self._sets)

    def all(self) -> tuple[FeatureSet, ...]:
        return tuple(self._sets)

    def get(self, set_id: str) -> FeatureSet | None:
        for fs in self._sets:
            if fs.set_id == set_id:
                return fs
        return None

    def add(self, feature_set: FeatureSet) -> str:
        self._sets.append(feature_set)
        return feature_set.set_id

    def remove(self, set_id: str) -> bool:
        before = len(self._sets)
        self._sets = [fs for fs in self._sets if fs.set_id != set_id]
        return len(self._sets) != before

    def rename(self, set_id: str, name: str) -> bool:
        for index, fs in enumerate(self._sets):
            if fs.set_id == set_id:
                self._sets[index] = replace(fs, name=str(name))
                return True
        return False

    def clear(self) -> None:
        self._sets = []

    # ---- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "feature_sets": [fs.to_dict() for fs in self._sets]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSetStore":
        items = data.get("feature_sets", []) if isinstance(data, dict) else []
        return cls([FeatureSet.from_dict(item) for item in items])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSetStore":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        return cls.from_dict(data)
