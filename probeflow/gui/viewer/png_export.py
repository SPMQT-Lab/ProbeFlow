"""Viewer PNG export logic extracted from ImageViewerDialog._on_save_png."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def save_viewer_png(
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
    """Write *arr* to *out_path* as a PNG with embedded provenance metadata.

    Returns a status string for the caller to display (success or error).
    All pre-flight validation (ROI reference checks, file-dialog interaction)
    is handled by the caller before this function is called.
    """
    from probeflow import processing as _proc
    from probeflow.core.scan_loader import load_scan
    from probeflow.gui.rendering import _get_lut
    from probeflow.processing.gui_adapter import processing_state_from_gui
    from probeflow.provenance.export import build_scan_export_provenance, png_display_state

    try:
        load_error = None
        try:
            _scan = load_scan(entry_path)
        except Exception as exc:
            _scan = None
            load_error = exc
        if include_provenance and _scan is None:
            return f"Export error: source scan could not be loaded for provenance: {load_error}"

        effective_range = scan_range_m
        if effective_range is None and _scan is not None:
            effective_range = _scan.scan_range_m
        if effective_range is None:
            effective_range = (0.0, 0.0)
        w_m, h_m = float(effective_range[0]), float(effective_range[1])

        vmin, vmax = drs.resolve(arr)
        provenance = None
        if include_provenance and _scan is not None:
            try:
                # Provenance describes the exported processed array. Keep the
                # source Scan itself untouched while giving the export record
                # the same calibrated range used by the scale bar.
                import copy
                provenance_scan = copy.copy(_scan)
                provenance_scan.scan_range_m = (w_m, h_m)
                ps = processing_state_from_gui(processing or {})
                provenance = build_scan_export_provenance(
                    provenance_scan,
                    channel_index=ch_idx,
                    channel_name=ch_name,
                    processing_state=ps,
                    display_state=png_display_state(
                        drs,
                        colormap=colormap,
                        add_scalebar=add_scalebar,
                        scalebar_unit="nm",
                        scalebar_pos="bottom-right",
                    ),
                    export_kind="viewer_png",
                    output_path=out_path,
                    roi_set=image_roi_set,
                    mask_set=image_mask_set,
                    processing_history=processing_history,
                )
            except Exception as exc:
                return f"Export error: provenance could not be built: {exc}"

        _proc.export_png(
            arr, out_path, colormap,
            clip_low, clip_high,
            lut_fn=_get_lut,
            scan_range_m=(w_m, h_m),
            add_scalebar=add_scalebar,
            vmin=vmin, vmax=vmax,
            provenance=provenance,
        )
        return f"Saved → {Path(out_path).name}"
    except Exception as exc:
        return f"Export error: {exc}"
