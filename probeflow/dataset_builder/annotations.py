"""Annotation save helpers for Dataset Builder."""

from __future__ import annotations

from pathlib import Path

from probeflow.core.mask import ImageMask, MaskSet
from probeflow.dataset_builder.models import DatasetTaskConfig, ProposalResult, ReviewRecord
from probeflow.dataset_builder.sidecar_state import save_review_record, utc_now
from probeflow.io.mask_sidecar import load_mask_set_sidecar, save_mask_set_sidecar


def proposal_to_mask(
    proposal: ProposalResult,
    *,
    config: DatasetTaskConfig,
    source_path: str | Path,
    source_channel: str | None = None,
) -> ImageMask:
    """Convert a mask proposal into an ``ImageMask`` with Dataset Builder context."""

    data = proposal.require_mask()
    params = dict(proposal.parameters)
    params.setdefault("source_path", str(source_path))
    params.setdefault("source_channel", source_channel)
    params.setdefault("dataset_builder_task", config.task)
    params.setdefault("label_name", config.label_name)
    params.setdefault("data_basis", "scan_plane")
    return ImageMask.new(
        data,
        method=proposal.method,
        parameters=params,
        name=config.label_name,
    )


def save_mask_annotation(
    scan_path: str | Path,
    image_mask: ImageMask,
    *,
    config: DatasetTaskConfig,
    status: str = "draft",
    notes: str = "",
    task_data: dict | None = None,
) -> tuple[Path, Path]:
    """Upsert one Dataset Builder mask and save review metadata."""

    scan_path = Path(scan_path)
    mask_set, _ = load_mask_set_sidecar(scan_path, missing_ok=True)
    if mask_set is None:
        mask_set = MaskSet(image_id=str(scan_path))

    existing = mask_set.get_by_name(config.label_name)
    # REVIEW(2026-07-06, low): upsert-by-name into the shared .masks.json can
    # silently overwrite a mask the user made in the viewer with the same
    # name (e.g. "step_edge"). Sharing the sidecar is the right call, but
    # consider a dataset_builder marker in parameters + a warning when the
    # existing mask was not created by this tool.
    if existing is not None:
        mask_set.replace(existing.id, image_mask.data)
        existing.method = image_mask.method
        existing.parameters = dict(image_mask.parameters)
        image_mask = existing
    else:
        mask_set.add(image_mask)
    mask_set.set_active(image_mask.id)
    mask_path = save_mask_set_sidecar(mask_set, scan_path)

    record = ReviewRecord(
        source_path=str(scan_path),
        plane_index=config.plane_index,
        task=config.task,
        label_type=config.label_type,
        label_name=config.label_name,
        status=status,
        annotator=config.annotator,
        notes=notes,
        proposal_method=image_mask.method,
        proposal_parameters=dict(image_mask.parameters),
        task_data=dict(task_data or {}),
        updated_at=utc_now(),
    )
    state_path = save_review_record(scan_path, record)
    return mask_path, state_path


def save_review_annotation(
    scan_path: str | Path,
    *,
    config: DatasetTaskConfig,
    status: str,
    notes: str = "",
    task_data: dict | None = None,
) -> Path:
    """Save review metadata for labels that have no raster/object artifact."""

    scan_path = Path(scan_path)
    record = ReviewRecord(
        source_path=str(scan_path),
        plane_index=config.plane_index,
        task=config.task,
        label_type=config.label_type,
        label_name=config.label_name,
        status=status,
        annotator=config.annotator,
        notes=notes,
        proposal_method=None,
        proposal_parameters={},
        task_data=dict(task_data or {}),
        updated_at=utc_now(),
    )
    return save_review_record(scan_path, record)
