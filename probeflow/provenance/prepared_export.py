"""Prepared handoff exports for downstream STM/qPlus analysis tools.

These helpers are intentionally small: they do not create a full provenance
graph, but they do make "send this image to another package" a first-class,
trackable export with the same ProcessingState/DisplayState split as the rest
of ProbeFlow.
"""

from __future__ import annotations

from probeflow.processing.display_state import DisplayRangeState
from probeflow.provenance.export import (
    background_processing_warnings,
    build_scan_export_provenance,
    png_display_state,
)
from probeflow.processing.state import ProcessingState
from probeflow.core.scan_model import Scan
from probeflow.io.writers.png import write_png


def write_prepared_png(
    scan: Scan,
    out_path,
    *,
    plane_idx: int = 0,
    processing_state: ProcessingState | dict | None = None,
    display_state: DisplayRangeState | dict | None = None,
    colormap: str = "gray",
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    add_scalebar: bool = False,
) -> None:
    """Write a PNG plus provenance sidecar for AISurf-style handoff.

    The scan should already contain the prepared numerical plane if processing
    has been applied. ``processing_state`` records how that plane was prepared;
    if no background/line correction is recorded, the sidecar includes a
    warning because PNG-consuming tools are often sensitive to tilt/background.
    """
    if processing_state is None:
        ps_dict = {"steps": []}
    elif hasattr(processing_state, "to_dict"):
        ps_dict = processing_state.to_dict()
    else:
        ps_dict = dict(processing_state)

    warnings = background_processing_warnings(ps_dict)
    prov = build_scan_export_provenance(
        scan,
        channel_index=plane_idx,
        processing_state=ps_dict,
        display_state=png_display_state(
            display_state,
            clip_low=clip_low,
            clip_high=clip_high,
            colormap=colormap,
            add_scalebar=add_scalebar,
            scalebar_unit="nm",
            scalebar_pos="bottom-right",
        ),
        export_kind="prepared_png",
        output_path=out_path,
        warnings=warnings,
    )
    write_png(
        scan,
        out_path,
        plane_idx=plane_idx,
        colormap=colormap,
        clip_low=clip_low,
        clip_high=clip_high,
        add_scalebar=add_scalebar,
        provenance=prov,
    )
