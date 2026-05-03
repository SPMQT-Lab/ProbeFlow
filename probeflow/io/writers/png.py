"""Render a :class:`probeflow.core.scan_model.Scan` plane to a colourised PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from probeflow.io.common import check_overwrite
from probeflow.provenance.export import build_scan_export_provenance, png_display_state
from probeflow.processing import export_png
from probeflow.core.scan_model import Scan


def lut_from_matplotlib(name: str) -> np.ndarray:
    """Return a (256, 3) uint8 LUT from a matplotlib colormap name.

    Shared by ``probeflow.io.writers.png`` and ``probeflow.cli``.  Uses the Agg
    backend so importing this function from a CLI context does not try to
    start a GUI.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        from matplotlib import colormaps
        cmap = colormaps.get_cmap(name)
    except Exception:
        return np.stack([np.arange(256, dtype=np.uint8)] * 3, axis=1)
    return (cmap(np.linspace(0, 1, 256))[:, :3] * 255.0).astype(np.uint8)


def write_png(
    scan: Scan,
    out_path,
    plane_idx: int = 0,
    *,
    colormap: str = "gray",
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    lut_fn: Optional[Callable[[str], np.ndarray]] = None,
    add_scalebar: bool = True,
    scalebar_unit: str = "nm",
    scalebar_pos: str = "bottom-right",
    provenance=None,  # ExportProvenance | None
) -> None:
    """Write a single plane of ``scan`` as a colourised PNG."""
    if scan.source_path is not None:
        check_overwrite(scan.source_path, out_path)
    if plane_idx < 0 or plane_idx >= scan.n_planes:
        raise ValueError(
            f"plane_idx={plane_idx} out of range for Scan with "
            f"{scan.n_planes} plane(s)"
        )
    if provenance is None:
        provenance = build_scan_export_provenance(
            scan,
            channel_index=plane_idx,
            display_state=png_display_state(
                clip_low=clip_low,
                clip_high=clip_high,
                colormap=colormap,
                add_scalebar=add_scalebar,
                scalebar_unit=scalebar_unit,
                scalebar_pos=scalebar_pos,
            ),
            export_kind="png",
            output_path=out_path,
        )
    arr = scan.planes[plane_idx]
    export_png(
        arr, out_path, colormap, clip_low, clip_high,
        lut_fn=lut_fn if lut_fn is not None else lut_from_matplotlib,
        scan_range_m=scan.scan_range_m,
        add_scalebar=add_scalebar,
        scalebar_unit=scalebar_unit,
        scalebar_pos=scalebar_pos,
        provenance=provenance,
    )
