"""Matplotlib plotting functions for Createc spectroscopy data.

All functions accept an optional ``ax`` argument so they can be embedded
in the GUI or used standalone in scripts and notebooks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from .spec_io import SpecData
from .spec_processing import current_histogram as _hist


def plot_spectrum(
    spec: SpecData,
    channel: str = "Z",
    ax=None,
    label: Optional[str] = None,
    **plot_kwargs,
):
    """Plot a single spectrum with labelled axes.

    Parameters
    ----------
    spec : SpecData
        Parsed spectroscopy data.
    channel : str
        Channel name to plot, e.g. 'I', 'Z', 'V'.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; created if None.
    label : str, optional
        Legend label; defaults to the filename stem.
    **plot_kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    y = spec.channels[channel]
    unit = spec.y_units.get(channel, "")
    lbl = label if label is not None else Path(spec.metadata.get("filename", "")).stem
    ax.plot(spec.x_array, y, label=lbl, **plot_kwargs)
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(f"{channel} ({unit})" if unit else channel)
    return ax


def plot_spectra(
    specs: list[SpecData],
    channel: str = "Z",
    offset: float = 0.0,
    ax=None,
    **plot_kwargs,
):
    """Overlay multiple spectra; a non-zero offset produces a waterfall.

    Parameters
    ----------
    specs : list[SpecData]
        List of parsed spectroscopy files.
    channel : str
        Channel name to plot.
    offset : float
        Vertical shift applied to each successive spectrum (in channel units).
        Set to 0 for a plain overlay.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; created if None.
    **plot_kwargs
        Forwarded to each ``ax.plot`` call.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    for i, spec in enumerate(specs):
        y = spec.channels[channel] + i * offset
        lbl = Path(spec.metadata.get("filename", f"#{i}")).stem
        ax.plot(spec.x_array, y, label=lbl, **plot_kwargs)

    if specs:
        unit = specs[0].y_units.get(channel, "")
        ax.set_xlabel(specs[0].x_label)
        ax.set_ylabel(f"{channel} ({unit})" if unit else channel)

    return ax


def plot_spec_positions(
    image_path: str,
    specs: list[SpecData],
    ax=None,
):
    """Display an .sxm topography with spectroscopy tip positions marked.

    Each spectrum's (x, y) tip position is drawn as a numbered marker
    on the topography image.

    Parameters
    ----------
    image_path : str
        Path to a Nanonis .sxm topography file.
    specs : list[SpecData]
        Spectroscopy files whose positions should be marked.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; created if None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from .sxm_io import parse_sxm_header, read_sxm_plane, sxm_scan_range, sxm_dims

    if ax is None:
        _, ax = plt.subplots()

    hdr = parse_sxm_header(image_path)
    arr = read_sxm_plane(image_path, plane_idx=0)
    w_m, h_m = sxm_scan_range(hdr)

    vmin = float(np.nanpercentile(arr, 1))
    vmax = float(np.nanpercentile(arr, 99))
    ax.imshow(
        arr,
        origin="upper",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        extent=[0.0, w_m * 1e9, 0.0, h_m * 1e9],
    )
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")

    # Parse scan centre offset from the header (metres).
    ox_m, oy_m = _parse_sxm_offset(hdr)

    for i, spec in enumerate(specs):
        px, py = spec.position
        # Position relative to image lower-left corner.
        rel_x = (px - ox_m + w_m / 2.0) * 1e9
        rel_y = (h_m / 2.0 - (py - oy_m)) * 1e9
        ax.plot(rel_x, rel_y, "o", markersize=7, color="yellow",
                markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(
            str(i + 1),
            (rel_x, rel_y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="yellow",
        )

    return ax


def plot_current_histogram(
    spec: SpecData,
    channel: str = "I",
    bins: int = 100,
    ax=None,
    **plot_kwargs,
):
    """Bar histogram of current values for telegraph-noise analysis.

    Parameters
    ----------
    spec : SpecData
        Parsed spectroscopy file.
    channel : str
        Channel whose values are histogrammed (typically 'I').
    bins : int
        Number of histogram bins.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; created if None.
    **plot_kwargs
        Forwarded to ``ax.bar``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    bin_edges, counts = _hist(spec.channels[channel], bins=bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    widths = np.diff(bin_edges)
    ax.bar(centers, counts, width=widths, **plot_kwargs)
    unit = spec.y_units.get(channel, "")
    ax.set_xlabel(f"{channel} ({unit})" if unit else channel)
    ax.set_ylabel("Counts")
    return ax


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_sxm_offset(hdr: dict) -> tuple[float, float]:
    """Extract the scan centre offset from an .sxm header dict (metres)."""
    raw = hdr.get("SCAN_OFFSET", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", raw)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return 0.0, 0.0
