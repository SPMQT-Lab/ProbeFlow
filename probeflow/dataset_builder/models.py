"""Small data models for Dataset Builder orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REVIEW_STATUSES = ("blank", "draft", "accepted", "uncertain", "rejected", "exported")
TASK_PRESETS = (
    "step_edge_mask",
    "point_labels",
    "particle_instances",
    "terrace_segmentation",
    "image_classification",
    "denoising_pairs",
    "custom",
)


@dataclass(frozen=True)
class DatasetTaskConfig:
    """One Dataset Builder task configuration."""

    task: str = "step_edge_mask"
    label_name: str = "step_edge"
    label_type: str = "mask"
    proposal_method: str = "step_edge"
    plane_index: int = 0
    annotator: str | None = None
    proposal_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task not in TASK_PRESETS:
            raise ValueError(f"Unknown Dataset Builder task {self.task!r}")


@dataclass(frozen=True)
class DatasetQueueItem:
    """One scan-plane entry in the Dataset Builder queue."""

    source_path: Path
    plane_index: int
    display_id: str
    status: str = "blank"
    has_mask_sidecar: bool = False
    has_roi_sidecar: bool = False
    reviewed_at: str | None = None
    exported_at: str | None = None
    load_error: str | None = None


@dataclass(frozen=True)
class ProposalResult:
    """Proposal output from an automatic labelling method."""

    label_type: str
    method: str
    parameters: dict[str, Any] = field(default_factory=dict)
    mask: np.ndarray | None = None
    points: tuple[Any, ...] = ()
    objects: tuple[dict[str, Any], ...] = ()

    def require_mask(self) -> np.ndarray:
        if self.mask is None:
            raise ValueError(f"Proposal {self.method!r} did not produce a mask")
        return np.asarray(self.mask, dtype=bool)


@dataclass(frozen=True)
class ReviewRecord:
    """Review metadata stored under ``dataset_builder`` in ``*.probeflow.json``."""

    source_path: str
    plane_index: int
    task: str
    label_type: str
    label_name: str
    status: str
    annotator: str | None = None
    notes: str = ""
    proposal_method: str | None = None
    proposal_parameters: dict[str, Any] = field(default_factory=dict)
    updated_at: str | None = None
    exported_at: str | None = None

    def __post_init__(self) -> None:
        if self.status not in REVIEW_STATUSES:
            raise ValueError(f"Unknown review status {self.status!r}")

    @property
    def key(self) -> str:
        return record_key(self.task, self.plane_index, self.label_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "plane_index": int(self.plane_index),
            "task": self.task,
            "label_type": self.label_type,
            "label_name": self.label_name,
            "status": self.status,
            "annotator": self.annotator,
            "notes": self.notes,
            "proposal_method": self.proposal_method,
            "proposal_parameters": dict(self.proposal_parameters),
            "updated_at": self.updated_at,
            "exported_at": self.exported_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewRecord":
        return cls(
            source_path=str(data.get("source_path") or ""),
            plane_index=int(data.get("plane_index") or 0),
            task=str(data.get("task") or "custom"),
            label_type=str(data.get("label_type") or "mask"),
            label_name=str(data.get("label_name") or "label"),
            status=str(data.get("status") or "blank"),
            annotator=data.get("annotator"),
            notes=str(data.get("notes") or ""),
            proposal_method=data.get("proposal_method"),
            proposal_parameters=dict(data.get("proposal_parameters") or {}),
            updated_at=data.get("updated_at"),
            exported_at=data.get("exported_at"),
        )


@dataclass(frozen=True)
class DatasetExportSpec:
    """Settings for a frozen Dataset Builder export snapshot."""

    source: Path
    output_dir: Path
    task: str = "step_edge_mask"
    label_name: str = "step_edge"
    plane_index: int = 0
    include_statuses: tuple[str, ...] = ("accepted",)
    array_dtype: str = "float32"
    overwrite: bool = False
    clip_low: float = 1.0
    clip_high: float = 99.0
    colormap: str = "gray"


def record_key(task: str, plane_index: int, label_name: str) -> str:
    return f"{task}:plane{int(plane_index)}:{label_name}"

