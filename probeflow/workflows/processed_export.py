"""One Qt-free workflow for writing a processed scan plane."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from probeflow.io.writers import SUPPORTED_OUTPUT_SUFFIXES
from probeflow.workflows.processed_export_model import (
    ProcessedExportRequest,
    ProcessedExportResult,
)


def _copy_with_export_values(request: ProcessedExportRequest):
    """Return the original scan unless the request supplies replacement data."""
    if (
        request.processed_plane is None
        and request.scan_range_m is None
        and request.processing_state is None
    ):
        return request.scan

    scan = copy.copy(request.scan)
    scan.planes = list(request.scan.planes)
    if request.processed_plane is not None:
        scan.planes[request.plane_idx] = np.asarray(
            request.processed_plane,
            dtype=np.float64,
        ).copy()
    if request.scan_range_m is not None:
        scan.scan_range_m = request.scan_range_m
    if request.processing_state is not None:
        from probeflow.core.processing_state import ProcessingState

        state = request.processing_state
        if not isinstance(state, ProcessingState):
            if hasattr(state, "to_dict"):
                state = state.to_dict()
            state = ProcessingState.from_dict(dict(state))
        scan.processing_state = state
    return scan


def _build_provenance(scan, request: ProcessedExportRequest, suffix: str):
    if request.provenance is not None:
        return request.provenance
    should_build = (
        request.include_provenance
        if request.build_provenance is None
        else request.build_provenance
    )
    if not should_build or suffix == ".sxm":
        return None

    from probeflow.provenance.export import build_scan_export_provenance

    channel_name = (
        scan.plane_names[request.plane_idx]
        if request.plane_idx < len(scan.plane_names)
        else None
    )
    return build_scan_export_provenance(
        scan,
        channel_index=request.plane_idx,
        channel_name=channel_name,
        processing_state=request.processing_state,
        display_state=request.display_state,
        export_kind=request.export_kind or suffix.lstrip("."),
        output_path=request.destination,
        warnings=request.warnings,
        roi_set=request.roi_set,
        mask_set=request.mask_set,
        processing_history=request.processing_history,
    )


def _overwrite_options(request: ProcessedExportRequest) -> dict[str, bool]:
    options: dict[str, bool] = {}
    if request.overwrite:
        options["overwrite"] = True
    if request.overwrite_sidecars:
        options["overwrite_sidecars"] = True
    return options


def _write_with_existing_writer(
    scan,
    request: ProcessedExportRequest,
    suffix: str,
    provenance,
) -> None:
    options: dict[str, Any] = dict(request.writer_options)
    options.update(_overwrite_options(request))
    if suffix == ".png":
        scan.save_png(
            request.destination,
            plane_idx=request.plane_idx,
            provenance=provenance,
            **options,
        )
    elif suffix == ".pdf":
        scan.save_pdf(
            request.destination,
            plane_idx=request.plane_idx,
            provenance=provenance,
            include_provenance=request.include_provenance,
            **options,
        )
    elif suffix == ".csv":
        scan.save_csv(
            request.destination,
            plane_idx=request.plane_idx,
            provenance=provenance,
            **options,
        )
    elif suffix == ".gwy":
        scan.save_gwy(
            request.destination,
            plane_idx=request.plane_idx,
            provenance=provenance,
            include_provenance=request.include_provenance,
            include_meta=request.include_provenance,
            **options,
        )
    else:
        state = getattr(scan, "processing_state", None)
        processed_plane_idx = (
            request.plane_idx if getattr(state, "steps", None) else None
        )
        scan.save_sxm(
            request.destination,
            processed_plane_idx=processed_plane_idx,
            include_provenance=request.include_provenance,
            **options,
        )


def write_processed_export(
    request: ProcessedExportRequest,
) -> ProcessedExportResult:
    """Write one artifact using the existing writer for its suffix."""
    suffix = request.destination.suffix.lower()
    if suffix not in SUPPORTED_OUTPUT_SUFFIXES:
        supported = ", ".join(SUPPORTED_OUTPUT_SUFFIXES)
        raise ValueError(
            f"Unsupported processed image format {suffix!r}. Use {supported}."
        )
    if request.plane_idx < 0 or request.plane_idx >= request.scan.n_planes:
        raise ValueError(
            f"plane_idx={request.plane_idx} out of range for Scan with "
            f"{request.scan.n_planes} plane(s)"
        )

    scan = _copy_with_export_values(request)
    provenance = _build_provenance(scan, request, suffix)
    _write_with_existing_writer(scan, request, suffix, provenance)
    return ProcessedExportResult(
        destination=request.destination,
        export_format=suffix.lstrip("."),
        plane_idx=request.plane_idx,
        provenance=provenance,
    )


__all__ = ["write_processed_export"]
