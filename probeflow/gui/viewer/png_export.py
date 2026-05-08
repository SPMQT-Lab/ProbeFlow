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
        try:
            _scan = load_scan(entry_path)
            w_m, h_m = _scan.scan_range_m
        except Exception:
            _scan = None
            w_m = h_m = 0.0

        vmin, vmax = drs.resolve(arr)
        provenance = None
        if _scan is not None:
            try:
                ps = processing_state_from_gui(processing or {})
                provenance = build_scan_export_provenance(
                    _scan,
                    channel_index=ch_idx,
                    channel_name=ch_name,
                    processing_state=ps,
                    display_state=png_display_state(
                        drs,
                        colormap=colormap,
                        add_scalebar=True,
                        scalebar_unit="nm",
                        scalebar_pos="bottom-right",
                    ),
                    export_kind="viewer_png",
                    output_path=out_path,
                    roi_set=image_roi_set,
                )
            except Exception:
                pass

        _proc.export_png(
            arr, out_path, colormap,
            clip_low, clip_high,
            lut_fn=_get_lut,
            scan_range_m=(w_m, h_m),
            vmin=vmin, vmax=vmax,
            provenance=provenance,
        )
        return f"Saved → {Path(out_path).name}"
    except Exception as exc:
        return f"Export error: {exc}"
