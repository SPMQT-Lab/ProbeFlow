"""Generic conversion utilities shared across ``core`` and ``io``.

These helpers are deliberately format-agnostic: they handle common parsing
edge cases (European decimal notation, safe type coercion) that arise when
reading vendor file headers.
"""

from __future__ import annotations


def _f(x, default=None):
    """Safe float conversion; replaces commas with dots for European locales."""
    try:
        return float(str(x).replace(",", "."))
    except (TypeError, ValueError):
        return default


def _i(x, default=None):
    """Safe int conversion."""
    try:
        return int(float(str(x).replace(",", ".")))
    except (TypeError, ValueError):
        return default
