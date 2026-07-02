"""Frozen ML-ready Dataset Builder export snapshots."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from probeflow.core.mask import MaskSet
from probeflow.dataset_builder.loading import load_scan_plane, plane_sample_id
from probeflow.dataset_builder.manifest import write_manifest
from probeflow.dataset_builder.models import DatasetExportSpec
from probeflow.dataset_builder.queue import build_queue
from probeflow.dataset_builder.sidecar_state import (
    default_state_sidecar_path,
    load_review_record,
)
from probeflow.io.mask_sidecar import default_mask_sidecar_path, load_mask_set_sidecar
from probeflow.io.roi_sidecar import default_roi_sidecar_path, load_roi_set_sidecar
from probeflow.io.writers.png import write_png
from probeflow.provenance.export import (
    build_scan_export_provenance,
    png_display_state,
    write_provenance_sidecars,
)


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _file_hash(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _check_available(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output path already exists: {path}")


def _mask_png(mask: np.ndarray, path: Path, *, overwrite: bool) -> None:
    _check_available(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), mode="L").save(path)


def _copy_json_sidecar(src: Path, dst: Path, *, overwrite: bool) -> str:
    if not src.exists():
        return ""
    _check_available(dst, overwrite)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst)


def export_dataset(spec: DatasetExportSpec) -> dict[str, Any]:
    """Export a frozen ML-ready dataset snapshot and return a summary."""

    out = Path(spec.output_dir)
    for name in ("arrays", "previews", "masks", "rois", "objects", "provenance"):
        (out / name).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    queue = build_queue(
        spec.source,
        task=spec.task,
        label_name=spec.label_name,
        plane_index=spec.plane_index,
    )
    for item in queue:
        if item.status not in spec.include_statuses:
            continue
        record = load_review_record(
            item.source_path,
            task=spec.task,
            plane_index=spec.plane_index,
            label_name=spec.label_name,
        )
        if record is None:
            continue

        scan, arr, px_x_m, px_y_m = load_scan_plane(item.source_path, spec.plane_index)
        sample_id = plane_sample_id(item.source_path, spec.plane_index)

        array_path = out / "arrays" / f"{sample_id}.npy"
        _check_available(array_path, spec.overwrite)
        with array_path.open("wb") as fh:
            np.save(fh, arr.astype(spec.array_dtype), allow_pickle=False)

        preview_path = out / "previews" / f"{sample_id}.png"
        write_png(
            scan,
            preview_path,
            plane_idx=spec.plane_index,
            colormap=spec.colormap,
            clip_low=spec.clip_low,
            clip_high=spec.clip_high,
            add_scalebar=False,
            provenance=None,
            overwrite=spec.overwrite,
            overwrite_sidecars=spec.overwrite,
        )

        mask_paths: dict[str, str] = {}
        mask_set: MaskSet | None = None
        native_mask_sidecar = default_mask_sidecar_path(item.source_path)
        try:
            mask_set, _ = load_mask_set_sidecar(item.source_path, missing_ok=True)
        except Exception:
            mask_set = None
        if mask_set is not None:
            for mask in mask_set.masks:
                if mask.name != spec.label_name:
                    continue
                mask_npy = out / "masks" / f"{sample_id}_{mask.name}.npy"
                mask_png = out / "masks" / f"{sample_id}_{mask.name}.png"
                _check_available(mask_npy, spec.overwrite)
                with mask_npy.open("wb") as fh:
                    np.save(fh, mask.data.astype(bool), allow_pickle=False)
                _mask_png(mask.data, mask_png, overwrite=spec.overwrite)
                mask_paths[mask.name] = _rel(mask_npy, out)
                mask_paths[f"{mask.name}_png"] = _rel(mask_png, out)

        native_roi_sidecar = default_roi_sidecar_path(item.source_path)
        roi_export_path = out / "rois" / f"{sample_id}.rois.json"
        roi_path_value = _copy_json_sidecar(
            native_roi_sidecar,
            roi_export_path,
            overwrite=spec.overwrite,
        )
        roi_set = None
        try:
            roi_set, _ = load_roi_set_sidecar(item.source_path, missing_ok=True)
        except Exception:
            roi_set = None

        prov_artifact_path = out / "provenance" / f"{sample_id}.json"
        export_prov = build_scan_export_provenance(
            scan,
            channel_index=spec.plane_index,
            export_kind="dataset_builder_sample",
            output_path=prov_artifact_path,
            roi_set=roi_set,
            mask_set=mask_set,
        )
        write_provenance_sidecars(
            prov_artifact_path,
            export_prov,
            legacy=False,
            probeflow=True,
            export_format="json",
            overwrite=spec.overwrite,
        )
        written_prov_path = prov_artifact_path.with_suffix(".probeflow.json")

        channel_name = (
            scan.plane_names[spec.plane_index]
            if spec.plane_index < len(scan.plane_names)
            else f"plane {spec.plane_index}"
        )
        channel_unit = (
            scan.plane_units[spec.plane_index]
            if spec.plane_index < len(scan.plane_units)
            else ""
        )
        rows.append(
            {
                "dataset_schema_version": 1,
                "sample_id": sample_id,
                "source_path": str(item.source_path),
                "source_format": scan.source_format,
                "source_hash": _file_hash(item.source_path),
                "scan_stem": item.source_path.stem,
                "plane_index": spec.plane_index,
                "channel_name": channel_name,
                "channel_unit": channel_unit,
                "array_shape": json.dumps(list(arr.shape)),
                "scan_range_m": json.dumps(list(scan.scan_range_m)),
                "pixel_size_x_m": px_x_m,
                "pixel_size_y_m": px_y_m,
                "task": record.task,
                "label_type": record.label_type,
                "label_names": record.label_name,
                "review_status": record.status,
                "annotator": record.annotator or "",
                "reviewed_at": record.updated_at or "",
                "proposal_method": record.proposal_method or "",
                "proposal_parameters_json": json.dumps(record.proposal_parameters, default=str),
                "processing_state_hash": export_prov.processing_state_hash or "",
                "array_path": _rel(array_path, out),
                "preview_path": _rel(preview_path, out),
                "mask_paths_json": json.dumps(mask_paths, sort_keys=True),
                "roi_path": _rel(Path(roi_path_value), out) if roi_path_value else "",
                "objects_path": "",
                "provenance_path": _rel(written_prov_path, out),
                "native_mask_sidecar": str(native_mask_sidecar),
                "native_roi_sidecar": str(native_roi_sidecar),
                "native_probeflow_sidecar": str(default_state_sidecar_path(item.source_path)),
                "notes": record.notes,
            }
        )

    manifest_csv, manifest_json = write_manifest(out, rows)
    return {
        "output_dir": str(out),
        "n_exported": len(rows),
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
    }
