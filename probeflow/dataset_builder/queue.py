"""Queue discovery for Dataset Builder."""

from __future__ import annotations

from pathlib import Path

from probeflow.core.indexing import ProbeFlowItem, image_browser_items, index_folder
from probeflow.dataset_builder.loading import plane_sample_id
from probeflow.dataset_builder.models import DatasetQueueItem
from probeflow.dataset_builder.sidecar_state import load_review_record
from probeflow.io.mask_sidecar import default_mask_sidecar_path
from probeflow.io.roi_sidecar import default_roi_sidecar_path


def _queue_item_for_path(
    path: Path,
    *,
    task: str,
    label_name: str,
    plane_index: int,
    load_error: str | None = None,
) -> DatasetQueueItem:
    record = load_review_record(
        path,
        task=task,
        plane_index=plane_index,
        label_name=label_name,
    )
    status = record.status if record is not None else "blank"
    return DatasetQueueItem(
        source_path=path,
        plane_index=plane_index,
        display_id=plane_sample_id(path, plane_index),
        status=status,
        has_mask_sidecar=default_mask_sidecar_path(path).exists(),
        has_roi_sidecar=default_roi_sidecar_path(path).exists(),
        reviewed_at=record.updated_at if record is not None else None,
        exported_at=record.exported_at if record is not None else None,
        load_error=load_error,
    )


def build_queue_from_indexed_items(
    items: list[ProbeFlowItem] | tuple[ProbeFlowItem, ...],
    *,
    task: str = "step_edge_mask",
    label_name: str = "step_edge",
    plane_index: int = 0,
) -> list[DatasetQueueItem]:
    """Build queue rows from a folder index without re-indexing the folder."""

    return [
        _queue_item_for_path(
            Path(item.path),
            task=task,
            label_name=label_name,
            plane_index=plane_index,
            load_error=item.load_error,
        )
        for item in items
        if item.item_type == "scan" and not item.load_error
    ]


def build_queue(
    source: str | Path,
    *,
    task: str = "step_edge_mask",
    label_name: str = "step_edge",
    plane_index: int = 0,
    recursive: bool = True,
) -> list[DatasetQueueItem]:
    """Build a scan-plane labelling queue from a folder or single scan file."""

    src = Path(source)
    if src.is_file():
        return [
            _queue_item_for_path(
                src.resolve(),
                task=task,
                label_name=label_name,
                plane_index=plane_index,
            )
        ]

    items = image_browser_items(index_folder(src, recursive=recursive, include_errors=True))
    return build_queue_from_indexed_items(items, task=task, label_name=label_name, plane_index=plane_index)


def queue_counts(queue: list[DatasetQueueItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in queue:
        counts[item.status] = counts.get(item.status, 0) + 1
    counts["total"] = len(queue)
    return counts
