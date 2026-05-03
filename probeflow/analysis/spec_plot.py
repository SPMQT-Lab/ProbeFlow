"""Matplotlib plotting functions for Createc spectroscopy data.

All functions accept an optional ``ax`` argument so they can be embedded
in the GUI or used standalone in scripts and notebooks.
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from probeflow.io.spectroscopy import SpecData
from probeflow.processing.spectroscopy import current_histogram as _hist

log = logging.getLogger(__name__)


def plot_spectrum(
    spec: SpecData,
    channel: str = "Z",
    ax=None,
    label: Optional[str] = None,
    **plot_kwargs,
):
    """Plot a single spectrum with labelled axes and a legend.

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

    # Pop 'label' from kwargs to avoid passing it twice.
    kw_label = plot_kwargs.pop("label", None)
    lbl = label if label is not None else (
        kw_label if kw_label is not None else
        Path(spec.metadata.get("filename", "")).stem
    )

    if channel == "V" and spec.metadata.get("sweep_type") == "bias_sweep":
        warnings.warn(
            "plot_spectrum: 'V' channel is redundant for bias sweeps — "
            "it equals x_array. Consider plotting 'I' or 'Z' instead.",
            stacklevel=2,
        )
    y = spec.channels[channel]
    unit = spec.y_units.get(channel, "")
    ax.plot(spec.x_array, y, label=lbl, **plot_kwargs)
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(f"{channel} ({unit})" if unit else channel)
    ax.legend()
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
        Vertical shift applied to each successive spectrum, in the channel's
        SI units (e.g. A for current, m for Z). Set to 0 for a plain overlay.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; created if None.
    **plot_kwargs
        Forwarded to each ``ax.plot`` call. A 'label' key is applied to all
        curves; omit it to use per-file names.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    # A caller-supplied label overrides per-spec names for all curves.
    user_label = plot_kwargs.pop("label", None)

    for i, spec in enumerate(specs):
        y = spec.channels[channel] + i * offset
        lbl = user_label if user_label is not None else (
            Path(spec.metadata.get("filename", f"#{i}")).stem
        )
        ax.plot(spec.x_array, y, label=lbl, **plot_kwargs)

    if specs:
        unit = specs[0].y_units.get(channel, "")
        ax.set_xlabel(specs[0].x_label)
        ax.set_ylabel(f"{channel} ({unit})" if unit else channel)
        ax.legend()

    return ax


def plot_spec_positions(
    image_path: str,
    specs: list[SpecData],
    ax=None,
):
    """Display an .sxm topography with spectroscopy tip positions marked.

    Each spectrum's (x, y) tip position is drawn as a numbered marker on the
    topography. Scan-frame rotation (SCAN_ANGLE) is applied so markers land
    correctly on rotated images.

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
    from probeflow.io.sxm_io import parse_sxm_header, read_sxm_plane, sxm_scan_range

    if ax is None:
        _, ax = plt.subplots()

    hdr = parse_sxm_header(image_path)
    arr = read_sxm_plane(image_path, plane_idx=0)
    w_m, h_m = sxm_scan_range(hdr)

    vmin = float(np.nanpercentile(arr, 1))
    vmax = float(np.nanpercentile(arr, 99))
    # origin="upper": row 0 at top; extent maps y=h_nm to top, y=0 to bottom.
    # Nanonis .sxm files scan from top to bottom, so row 0 is the physically
    # highest y position — consistent with origin="upper" here.
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

    ox_m, oy_m = _parse_sxm_offset(hdr)
    theta = _parse_sxm_angle_rad(hdr)
    if theta != 0.0:
        log.warning(
            "plot_spec_positions: scan angle %.3f rad applied — "
            "verify markers if the image appears rotated.",
            theta,
        )
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i, spec in enumerate(specs):
        px, py = spec.position
        dx, dy = px - ox_m, py - oy_m
        # Rotate physical offset into the scan frame.
        dx_rot = cos_t * dx + sin_t * dy
        dy_rot = -sin_t * dx + cos_t * dy
        # With origin="upper" and extent [0,W,0,H]: x=0 is left, x=W is right,
        # y=H is the top of the axes (where image row 0 sits), y=0 is the bottom.
        rel_x = (dx_rot + w_m / 2.0) * 1e9
        rel_y = (dy_rot + h_m / 2.0) * 1e9
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


def spec_position_to_pixel(
    x_m: float,
    y_m: float,
    scan_shape: tuple[int, int],
    scan_range_m: tuple[float, float],
    scan_offset_m: Optional[tuple[float, float]] = None,
    scan_angle_deg: float = 0.0,
) -> Optional[tuple[float, float]]:
    """Convert a tip position to fractional image coordinates.

    Parameters
    ----------
    x_m, y_m : float
        Tip position in world coordinates (metres).
    scan_shape : tuple[int, int]
        (Ny, Nx) pixel dimensions of the image (currently unused in the
        fractional result, kept for API symmetry with pixel-based callers).
    scan_range_m : tuple[float, float]
        (width_m, height_m) physical size of the scan.
    scan_offset_m : tuple[float, float] | None
        (ox_m, oy_m) scan-frame centre in world coordinates. Defaults to
        (0, 0) — image centred at the origin.
    scan_angle_deg : float
        Scan rotation angle in degrees (counter-clockwise from +x-axis).

    Returns
    -------
    tuple[float, float] | None
        (frac_x, frac_y) where (0, 0) is the top-left corner and (1, 1) is
        the bottom-right corner of the image, or ``None`` if the position
        falls outside the scan extent.
    """
    w_m, h_m = scan_range_m
    ox_m, oy_m = scan_offset_m if scan_offset_m is not None else (0.0, 0.0)
    theta = np.radians(scan_angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    dx = x_m - ox_m
    dy = y_m - oy_m
    dx_rot = cos_t * dx + sin_t * dy
    dy_rot = -sin_t * dx + cos_t * dy

    # Fractional position in the scan frame (0 = left/bottom, 1 = right/top)
    frac_x = (dx_rot + w_m / 2.0) / w_m
    frac_y_from_bottom = (dy_rot + h_m / 2.0) / h_m

    if not (0.0 <= frac_x <= 1.0 and 0.0 <= frac_y_from_bottom <= 1.0):
        return None

    # Image row 0 is the top (highest physical y), so invert y.
    frac_y = 1.0 - frac_y_from_bottom
    return frac_x, frac_y


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
        Channel whose values are histogrammed. Defaults to 'I' for
        telegraph-noise analysis, but any channel name is accepted.
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

    counts, bin_edges = _hist(spec.channels[channel], bins=bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    widths = np.diff(bin_edges)
    ax.bar(centers, counts, width=widths, **plot_kwargs)
    unit = spec.y_units.get(channel, "")
    ax.set_xlabel(f"{channel} ({unit})" if unit else channel)
    ax.set_ylabel("Counts")
    return ax


# ── Helpers ──────────────────────────────────────────────────────────────────

# Matches floats with or without a leading digit: 1.5E-3, .5e+2, -14.0, etc.
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# (scale, prefix) options per base SI unit, ordered from smallest to largest
# prefix — the heuristic in ``choose_display_unit`` walks this list and picks
# the prefix that puts the typical magnitude into [0.1, 1000].
_UNIT_PREFIX_TABLE: dict[str, list[tuple[float, str]]] = {
    "m": [(1e12, "pm"), (1e10, "Å"), (1e9, "nm"), (1e6, "µm"), (1.0, "m")],
    "A": [(1e15, "fA"), (1e12, "pA"), (1e9, "nA"), (1e6, "µA"), (1.0, "A")],
    "V": [(1e6, "µV"), (1e3, "mV"), (1.0, "V")],
}

_ZERO_VALUE_DISPLAY_DEFAULTS: dict[str, tuple[float, str]] = {
    "m": (1e9, "nm"),
    "A": (1e12, "pA"),
    "V": (1e3, "mV"),
}


def lookup_unit_scale(si_unit: str, label: str) -> Optional[tuple[float, str]]:
    """Return ``(scale, label)`` for an explicit user choice of display unit.

    ``si_unit`` is the underlying SI unit (e.g. ``"m"``); ``label`` is the
    desired display label (e.g. ``"nm"``, ``"Å"``, ``"pm"``). Returns
    ``None`` if the label is unknown for that base unit, so callers can
    fall back to ``choose_display_unit``.
    """
    table = _UNIT_PREFIX_TABLE.get(si_unit)
    if table is None:
        return None
    for scale, lbl in table:
        if lbl == label:
            return scale, lbl
    return None


def choose_display_unit(si_unit: str, values: np.ndarray) -> tuple[float, str]:
    """Pick a sensible display unit and scale factor.

    Returns ``(scale_factor, display_unit_string)`` where multiplying the raw
    SI values by ``scale_factor`` gives numbers in the returned display unit.

    Heuristic: compute the median absolute value of non-zero samples and
    pick the SI prefix that brings that magnitude into ``[0.1, 1000]``.
    For units without a prefix table (Hz, rad, dimensionless, unknown),
    returns ``(1.0, si_unit)`` with no scaling.
    """
    if values is None:
        return 1.0, si_unit
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 1.0, si_unit

    prefixes = _UNIT_PREFIX_TABLE.get(si_unit)
    if prefixes is None:
        return 1.0, si_unit

    nonzero = arr[arr != 0]
    if nonzero.size == 0:
        return _ZERO_VALUE_DISPLAY_DEFAULTS.get(si_unit, (1.0, si_unit))
    magnitude = float(np.median(np.abs(nonzero)))
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return 1.0, si_unit

    # Pick the smallest (most fine-grained) prefix whose scaled magnitude is
    # still < 1000 — this walks the table from smallest to largest prefix.
    chosen = prefixes[-1]  # default: no-prefix SI unit
    for scale, label in prefixes:
        scaled = magnitude * scale
        if 0.1 <= scaled < 1000:
            chosen = (scale, label)
            break
    else:
        # Nothing matched the [0.1, 1000] window; pick the prefix whose scaled
        # value is closest to the centre (30) in log-space.
        best = None
        best_dist = float("inf")
        for scale, label in prefixes:
            scaled = magnitude * scale
            if scaled <= 0:
                continue
            dist = abs(np.log10(scaled) - np.log10(30.0))
            if dist < best_dist:
                best_dist = dist
                best = (scale, label)
        if best is not None:
            chosen = best

    return chosen


def _parse_sxm_offset(hdr: dict) -> tuple[float, float]:
    """Extract the scan centre offset from an .sxm header dict (metres)."""
    raw = hdr.get("SCAN_OFFSET", "")
    nums = _FLOAT_RE.findall(raw)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return 0.0, 0.0


def _parse_sxm_angle_rad(hdr: dict) -> float:
    """Extract the scan rotation angle from an .sxm header dict (radians)."""
    raw = hdr.get("SCAN_ANGLE", "0").strip()
    nums = _FLOAT_RE.findall(raw)
    try:
        return float(nums[0]) if nums else 0.0
    except (ValueError, IndexError):
        return 0.0
