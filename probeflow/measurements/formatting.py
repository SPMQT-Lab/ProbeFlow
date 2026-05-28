"""Shared unit-formatting helpers for measurement displays.

Review arch-backend #5 (2026-05-28): the codebase previously had six
parallel metre→pm/Å/nm/µm formatters, each with slightly different
breakpoints — ``analysis.simple_measurements._fmt_m``,
``analysis.line_periodicity._fmt_m`` + ``format_period``,
``analysis.roi_statistics._fmt_z`` (wrapping ``_fmt_m``),
``analysis.lattice_grid._choose_unit`` + ``_period_str`` + ``_length_str``,
``gui.viewer.image_measurements._format_period_bound_nm``, and
``analysis.spec_plot.choose_display_unit``/``lookup_unit_scale``.

This module is the single home for those helpers.  Existing public
APIs in the analysis package keep their signatures via thin wrappers
so external code does not break.

The canonical heuristic for scalar lengths:

    |value_m| < 5e-11  m  →  pm   (sub-Å scale)
    |value_m| < 1e-9   m  →  Å    (atomic-scale, < 1 nm)
    |value_m| < 1e-6   m  →  nm
    else                  →  µm

These thresholds match the ``simple_measurements`` choices, which were
the most carefully tuned for STM length scales.

For ARRAYS of SI values (where a single representative scale is more
useful than picking per-element), ``choose_display_unit`` performs a
median-based prefix walk — re-exported here for convenience and the
canonical implementation of the multi-unit prefix table.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


__all__ = [
    "choose_length_unit",
    "scale_length_m",
    "format_length_m",
    "format_period_m",
    "format_height_m",
    "choose_display_unit",
    "lookup_unit_scale",
]


# Prefix table — entries are ``(scale_factor, display_label)`` ordered from
# smallest physical scale (largest scale factor) to largest.  Multiplying
# the SI value by a scale factor gives the magnitude in the labelled unit.
#
# Moved here from ``analysis.spec_plot`` (review arch-backend #5).
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


def choose_length_unit(value_m: float) -> tuple[float, str]:
    """Return ``(scale_factor, unit_label)`` for a length in metres.

    Multiplying ``value_m`` by ``scale_factor`` gives the magnitude in the
    returned unit.  The choice matches the STM-tuned thresholds used by
    ``analysis.simple_measurements`` (was the canonical implementation
    before this consolidation).

    Special case: NaN inputs return ``(1e9, "nm")`` so callers can
    safely format the NaN as ``"nan nm"``.
    """
    v = float(value_m) if value_m is not None else float("nan")
    if not math.isfinite(v):
        return (1e9, "nm")
    av = abs(v)
    if av < 5e-11:        # < 0.05 Å
        return (1e12, "pm")
    if av < 1e-9:         # < 1 nm
        return (1e10, "Å")
    if av < 1e-6:         # < 1 µm
        return (1e9, "nm")
    return (1e6, "µm")


def scale_length_m(value_m: float) -> tuple[float, str]:
    """Return ``(scaled_value, unit_label)`` for a length in metres.

    Convenience wrapper around :func:`choose_length_unit` that applies
    the scale factor for the caller — matches the legacy ``_fmt_m``
    contract used in ``simple_measurements`` and ``roi_statistics``.
    """
    scale, unit = choose_length_unit(value_m)
    if value_m is None or not math.isfinite(float(value_m)):
        return float("nan"), unit
    return float(value_m) * scale, unit


def format_length_m(
    value_m: float,
    *,
    precision: int = 3,
    nan_repr: str = "—",
) -> str:
    """Return a single formatted string like ``"0.246 nm"`` for a length.

    ``nan_repr`` is the literal string returned when ``value_m`` is
    NaN or non-finite (default ``"—"``).
    """
    if value_m is None or not math.isfinite(float(value_m)):
        return nan_repr
    scaled, unit = scale_length_m(value_m)
    return f"{scaled:.{precision}g} {unit}"


def format_period_m(
    period_m: float,
    *,
    precision: int = 3,
    nan_repr: str = "—",
) -> tuple[str, str]:
    """Return ``(value_str, unit_str)`` for a period (spacing).

    The split form is convenient for tables that put the unit in a
    separate column.  ``format_length_m`` (returning the joined string)
    is more convenient for plot annotations.
    """
    if period_m is None or not math.isfinite(float(period_m)):
        return nan_repr, ""
    scaled, unit = scale_length_m(period_m)
    return f"{scaled:.{precision}g}", unit


def format_height_m(
    height_m: float,
    *,
    precision: int = 3,
    nan_repr: str = "—",
) -> str:
    """Return a formatted string for a z-axis (height) value in metres.

    Alias for :func:`format_length_m` — kept as a distinct name so
    height-vs-distance call sites stay self-documenting.
    """
    return format_length_m(height_m, precision=precision, nan_repr=nan_repr)
