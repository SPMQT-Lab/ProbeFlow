"""Single source of truth for the geometric processing-operation vocabulary.

Historically the long↔short operation aliases (``rotate_90_cw`` ↔ ``rot90_cw``),
the "lossless" set, and the "dimension-swapping" set were duplicated across
``core/roi.py``, ``processing/state.py``, ``processing/gui_adapter.py`` and the
CLI. That spread had no single owner and drifted (review arch-backend #9).

This module owns the vocabulary; the de-risking plan (Phase 1, see
``docs/core_derisk_plan.md``) points those call sites here. It is intentionally
tiny, dependency-free, and importable from any layer (``core`` included), so it
does not reintroduce a layering cycle.

Naming conventions
------------------
* **Long form** is canonical and public — it matches the entries in
  ``core.processing_state._SUPPORTED_OPS`` and the function names in
  ``processing.geometry`` (e.g. ``rotate_90_cw``).
* **Short form** is an internal convenience used by ROI / overlay geometry
  (``rot90_cw``). Use :func:`to_short` / :func:`to_long` to convert.
"""

from __future__ import annotations

# ── long ↔ short aliases (only the rotations differ) ────────────────────────
LONG_TO_SHORT: dict[str, str] = {
    "rotate_90_cw": "rot90_cw",
    "rotate_180": "rot180",
    "rotate_270_cw": "rot270_cw",
}
SHORT_TO_LONG: dict[str, str] = {short: long for long, short in LONG_TO_SHORT.items()}


def to_short(operation: str) -> str:
    """Return the internal short form of *operation* (identity if already short
    or has no alias)."""
    return LONG_TO_SHORT.get(operation, operation)


def to_long(operation: str) -> str:
    """Return the canonical long form of *operation* (identity if already long
    or has no alias)."""
    return SHORT_TO_LONG.get(operation, operation)


# ── op classes (short form — the vocabulary ROI/overlay/scan logic switches on)
# Exact pixel-coordinate transforms: ROI geometry survives them unchanged.
LOSSLESS_OPS: frozenset[str] = frozenset({
    "flip_horizontal",
    "flip_vertical",
    "rot90_cw",
    "rot180",
    "rot270_cw",
})

# Lossless ops that exchange the physical X and Y extents (transpose dims), so
# ``scan_range_m`` must be swapped.
DIMENSION_SWAPPING_OPS: frozenset[str] = frozenset({"rot90_cw", "rot270_cw"})

# ── simple geometric op names in LONG form (the flip/rotate family) ──────────
# These are the names listed by the grouped dispatch branches and the GUI
# adapter; they are members of ``_SUPPORTED_OPS`` and map 1:1 to
# ``processing.geometry`` functions.
SIMPLE_GEOMETRIC_OPS: frozenset[str] = frozenset({
    "flip_horizontal",
    "flip_vertical",
    "rotate_90_cw",
    "rotate_180",
    "rotate_270_cw",
})
