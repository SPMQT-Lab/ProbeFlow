"""Queue discovery for Dataset Builder."""

from __future__ import annotations

from pathlib import Path

from probeflow.core.indexing import image_browser_items, index_folder
from probeflow.dataset_builder.loading import plane_sample_id
from probeflow.dataset_builder.models import DatasetQueueItem
from probeflow.dataset_builder.sidecar_state import load_review_record
from probeflow.io.mask_sidecar import default_mask_sidecar_path
from probeflow.io.roi_sidecar import default_roi_sidecar_path


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
        paths = [src.resolve()]
        errors: dict[Path, str | None] = {src.resolve(): None}
    else:
        items = image_browser_items(
            index_folder(src, recursive=recursive, include_errors=True)
        )
        paths = [item.path for item in items]
        errors = {item.path: item.load_error for item in items}

    queue: list[DatasetQueueItem] = []
    for path in paths:
        record = load_review_record(
            path,
            task=task,
            plane_index=plane_index,
            label_name=label_name,
        )
        status = record.status if record is not None else "blank"
        queue.append(
            DatasetQueueItem(
                source_path=path,
                plane_index=plane_index,
                display_id=plane_sample_id(path, plane_index),
                status=status,
                has_mask_sidecar=default_mask_sidecar_path(path).exists(),
                has_roi_sidecar=default_roi_sidecar_path(path).exists(),
                reviewed_at=record.updated_at if record is not None else None,
                exported_at=record.exported_at if record is not None else None,
                load_error=errors.get(path),
            )
        )
    return queue


def queue_counts(queue: list[DatasetQueueItem]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in queue:
        counts[item.status] = counts.get(item.status, 0) + 1
    counts["total"] = len(queue)
    return counts

