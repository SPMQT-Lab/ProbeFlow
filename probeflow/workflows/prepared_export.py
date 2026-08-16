"""Prepared PNG handoff built on the shared processed-export workflow."""

from __future__ import annotations

from typing import Any

from probeflow.workflows.processed_export import write_processed_export
from probeflow.workflows.processed_export_model import ProcessedExportRequest


def _state_dict(processing_state: Any | None) -> dict:
    if processing_state is None:
        return {"steps": []}
    if hasattr(processing_state, "to_dict"):
        return processing_state.to_dict()
    return dict(processing_state)


def write_prepared_png(
    scan,
    out_path,
    *,
    plane_idx: int = 0,
    processing_state=None,
    display_state=None,
    colormap: str = "gray",
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    add_scalebar: bool = False,
    overwrite: bool = False,
    overwrite_sidecars: bool = False,
):
    """Write the existing downstream-analysis PNG handoff."""
    from probeflow.provenance.export import (
        background_processing_warnings,
        png_display_state,
    )

    state = _state_dict(processing_state)
    display = png_display_state(
        display_state,
        clip_low=clip_low,
        clip_high=clip_high,
        colormap=colormap,
        add_scalebar=add_scalebar,
        scalebar_unit="nm",
        scalebar_pos="bottom-right",
    )
    return write_processed_export(
        ProcessedExportRequest(
            scan=scan,
            destination=out_path,
            plane_idx=plane_idx,
            processing_state=state,
            display_state=display,
            export_kind="prepared_png",
            warnings=tuple(background_processing_warnings(state)),
            overwrite=overwrite,
            overwrite_sidecars=overwrite_sidecars,
            writer_options={
                "colormap": colormap,
                "clip_low": clip_low,
                "clip_high": clip_high,
                "add_scalebar": add_scalebar,
            },
        )
    )


__all__ = ["write_prepared_png"]
