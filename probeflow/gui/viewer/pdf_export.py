"""Viewer-specific, borderless PDF export helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_viewer_pdf(
    arr: np.ndarray,
    out_path: str,
    entry_path: Path,
    colormap: str,
    clip_low: float,
    clip_high: float,
    drs,
    processing: dict,
    image_roi_set,
    ch_idx: int,
    ch_name: str | None,
    processing_history=None,
    add_scalebar: bool = True,
    include_provenance: bool = True,
    image_mask_set=None,
    scan_range_m: tuple[float, float] | None = None,
) -> str:
    """Save the current viewer pixels as a borderless, single-image PDF."""

    from probeflow.core.scan_loader import load_scan
    from probeflow.gui.rendering import _get_lut
    from probeflow.processing.gui_adapter import processing_state_from_gui
    from probeflow.processing.pdf_export import export_image_pdf
    from probeflow.provenance.export import (
        build_scan_export_provenance,
        png_display_state,
    )

    try:
        load_error = None
        try:
            scan = load_scan(entry_path)
        except Exception as exc:
            scan = None
            load_error = exc
        if include_provenance and scan is None:
            return (
                "Export error: source scan could not be loaded for provenance: "
                f"{load_error}"
            )

        effective_range = scan_range_m
        if effective_range is None and scan is not None:
            effective_range = scan.scan_range_m
        if effective_range is None:
            effective_range = (0.0, 0.0)
        width_m, height_m = (
            float(effective_range[0]),
            float(effective_range[1]),
        )

        vmin, vmax = drs.resolve(arr)
        provenance = None
        if include_provenance and scan is not None:
            import copy

            provenance_scan = copy.copy(scan)
            provenance_scan.scan_range_m = (width_m, height_m)
            provenance = build_scan_export_provenance(
                provenance_scan,
                channel_index=ch_idx,
                channel_name=ch_name,
                processing_state=processing_state_from_gui(processing or {}),
                display_state=png_display_state(
                    drs,
                    colormap=colormap,
                    add_scalebar=add_scalebar,
                    scalebar_unit="nm",
                    scalebar_pos="bottom-right",
                ),
                export_kind="viewer_pdf",
                output_path=out_path,
                roi_set=image_roi_set,
                mask_set=image_mask_set,
                processing_history=processing_history,
            )

        export_image_pdf(
            arr,
            out_path,
            colormap,
            clip_low,
            clip_high,
            lut_fn=_get_lut,
            scan_range_m=(width_m, height_m),
            add_scalebar=add_scalebar,
            vmin=vmin,
            vmax=vmax,
            provenance=provenance,
        )
        return f"Saved → {Path(out_path).name}"
    except Exception as exc:
        return f"Export error: {exc}"


__all__ = ["save_viewer_pdf"]
