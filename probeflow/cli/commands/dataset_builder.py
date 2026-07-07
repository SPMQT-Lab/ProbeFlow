"""Dataset Builder CLI commands."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from probeflow.dataset_builder.annotations import proposal_to_mask, save_mask_annotation
from probeflow.dataset_builder.export import export_dataset
from probeflow.dataset_builder.loading import load_scan_plane
from probeflow.dataset_builder.models import DatasetExportSpec, DatasetTaskConfig
from probeflow.dataset_builder.proposals import generate_proposal
from probeflow.dataset_builder.queue import build_queue, queue_counts
from probeflow.dataset_builder.validation import validate_dataset
from probeflow.io.common import setup_logging

log = logging.getLogger(__name__)


def _parse_params(items: list[str] | None) -> dict:
    params: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE proposal parameter, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        else:
            try:
                value = float(raw)
            except ValueError:
                value = raw
        params[key] = value
    return params


def _cmd_dataset_summary(args) -> int:
    setup_logging(args.verbose)
    queue = build_queue(
        args.source,
        task=args.task,
        label_name=args.label_name,
        plane_index=args.plane,
    )
    payload = {"counts": queue_counts(queue), "items": [item.__dict__ for item in queue]}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        counts = payload["counts"]
        print(f"Dataset Builder queue: {counts.get('total', 0)} item(s)")
        for key in ("blank", "draft", "accepted", "uncertain", "rejected", "exported"):
            print(f"  {key}: {counts.get(key, 0)}")
    return 0


def _cmd_dataset_propose(args) -> int:
    setup_logging(args.verbose)
    try:
        params = _parse_params(args.param)
        config = DatasetTaskConfig(
            task=args.task,
            label_name=args.label_name,
            label_type="mask",
            proposal_method=args.method,
            plane_index=args.plane,
            annotator=args.annotator,
            proposal_params=params,
        )
        scan, arr, px_x_m, px_y_m = load_scan_plane(args.input, args.plane)
        channel = (
            scan.plane_names[args.plane]
            if args.plane < len(scan.plane_names)
            else f"plane {args.plane}"
        )
        proposal = generate_proposal(
            arr,
            px_x_m=px_x_m,
            px_y_m=px_y_m,
            config=config,
            source_channel=channel,
        )
        image_mask = proposal_to_mask(
            proposal,
            config=config,
            source_path=args.input,
            source_channel=channel,
        )
        mask_path, state_path = save_mask_annotation(
            args.input,
            image_mask,
            config=config,
            status=args.status,
            notes=args.notes or "",
        )
    except Exception as exc:
        log.error("Dataset proposal failed: %s", exc)
        return 1
    if args.json:
        print(json.dumps({
            "mask_sidecar": str(mask_path),
            "state_sidecar": str(state_path),
            "mask_pixels": image_mask.count(),
        }, indent=2))
    else:
        print(f"Saved {image_mask.name!r} mask ({image_mask.count()} px)")
        print(f"  masks: {mask_path}")
        print(f"  state: {state_path}")
    return 0


def _cmd_dataset_export(args) -> int:
    setup_logging(args.verbose)
    spec = DatasetExportSpec(
        source=Path(args.source),
        output_dir=Path(args.output),
        task=args.task,
        label_name=args.label_name,
        plane_index=args.plane,
        include_statuses=tuple(args.status),
        overwrite=bool(args.force),
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        colormap=args.colormap,
    )
    try:
        summary = export_dataset(spec)
    except Exception as exc:
        log.error("Dataset export failed: %s", exc)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_dataset_validate(args) -> int:
    result = validate_dataset(args.dataset)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


__all__ = [
    "_cmd_dataset_export",
    "_cmd_dataset_propose",
    "_cmd_dataset_summary",
    "_cmd_dataset_validate",
]
