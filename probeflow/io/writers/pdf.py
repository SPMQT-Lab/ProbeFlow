"""Render a :class:`probeflow.core.scan_model.Scan` plane to a publication-ready PDF.

Uses matplotlib's vector backend so the output is scale-bar-ready for papers.
Includes a colorbar with physical-unit ticks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from probeflow.io.common import check_overwrite
from probeflow.core.scan_model import Scan


def write_pdf(
    scan: Scan,
    out_path,
    plane_idx: int = 0,
    *,
    colormap: str = "gray",
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    title: Optional[str] = None,
    figsize: tuple = (6.0, 5.5),
    dpi: int = 300,
    show_colorbar: bool = True,
    show_scalebar: bool = True,
    scalebar_length_nm: Optional[float] = None,
) -> None:
    """Write a PDF figure of one Scan plane with axes, colorbar, and scale bar.

    Parameters
    ----------
    scan : Scan
    out_path : str or Path
    plane_idx : which plane to render (default 0 = Z forward)
    colormap : any matplotlib colormap name
    clip_low / clip_high : percentile contrast clipping [0-100]
    title : figure title; defaults to the source file stem
    figsize : (width, height) in inches
    dpi : rasterised elements only — vector parts stay scalable
    show_colorbar : render a colorbar with physical-unit ticks
    show_scalebar : render a white bar in the bottom-right with a length label
    scalebar_length_nm : override the auto-picked scale bar length
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if scan.source_path is not None:
        check_overwrite(scan.source_path, out_path)
    if plane_idx < 0 or plane_idx >= scan.n_planes:
        raise ValueError(
            f"plane_idx={plane_idx} out of range for Scan with "
            f"{scan.n_planes} plane(s)"
        )

    arr = scan.planes[plane_idx]
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Plane contains no finite values.")

    vmin = float(np.percentile(finite, clip_low))
    vmax = float(np.percentile(finite, clip_high))
    if vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    w_m, h_m = scan.scan_range_m
    Ny, Nx = arr.shape

    # Show the axes in nm — the most common physical unit for STM.
    extent_nm = (0.0, w_m * 1e9, 0.0, h_m * 1e9) if w_m > 0 and h_m > 0 else None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(
        arr,
        origin="upper",
        cmap=colormap,
        vmin=vmin, vmax=vmax,
        extent=extent_nm,
        aspect="equal",
        interpolation="nearest",
    )

    if title is None:
        title = scan.source_path.stem
    ax.set_title(title)
    if extent_nm is not None:
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
    else:
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    if show_colorbar:
        unit = scan.plane_units[plane_idx] if plane_idx < len(scan.plane_units) else ""
        label = scan.plane_names[plane_idx] if plane_idx < len(scan.plane_names) else ""
        cbar_label = f"{label} [{unit}]" if unit else label
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label)

    if show_scalebar and w_m > 0 and extent_nm is not None:
        if scalebar_length_nm is None:
            scalebar_length_nm = _pick_nice_nm(w_m * 1e9 * 0.25)
        bar_color = "white"
        text_color = "white"
        pad_nm = 0.03 * w_m * 1e9
        x0 = w_m * 1e9 - pad_nm - scalebar_length_nm
        y0 = pad_nm
        bar_h = 0.02 * h_m * 1e9
        ax.add_patch(Rectangle(
            (x0, y0), scalebar_length_nm, bar_h,
            facecolor=bar_color, edgecolor="black", linewidth=0.5, zorder=5,
        ))
        ax.text(
            x0 + scalebar_length_nm / 2.0, y0 + bar_h * 1.8,
            f"{scalebar_length_nm:g} nm",
            color=text_color, ha="center", va="bottom", fontsize=10, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", edgecolor="none",
                      alpha=0.4),
        )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


_NICE_NM = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def _pick_nice_nm(target_nm: float) -> float:
    """Pick a round scale-bar length close to target_nm."""
    best = _NICE_NM[0]
    for s in _NICE_NM:
        if abs(s - target_nm) < abs(best - target_nm):
            best = s
        if s > target_nm * 2:
            break
    return float(best)
