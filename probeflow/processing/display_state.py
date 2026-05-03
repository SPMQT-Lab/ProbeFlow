"""Per-image display range state for ProbeFlow.

Separates two distinct modes:
  percentile  — vmin/vmax derived from data percentiles on each render
  manual      — explicit vmin/vmax set directly by the user (e.g., bar drag)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DisplayRangeState:
    """Tracks whether display limits come from percentile clipping or manual input.

    In *percentile* mode, :meth:`resolve` computes (vmin, vmax) from the
    supplied array each time.  In *manual* mode, the stored (vmin, vmax) are
    returned directly without touching the array.
    """

    mode: Literal["percentile", "manual"] = "percentile"
    low_pct: float = 1.0
    high_pct: float = 99.0
    vmin: float | None = None
    vmax: float | None = None

    # ── Mode transitions ──────────────────────────────────────────────────────

    def set_percentile(self, low_pct: float, high_pct: float) -> None:
        """Switch to percentile mode and update clip percentiles."""
        self.mode     = "percentile"
        self.low_pct  = float(low_pct)
        self.high_pct = float(high_pct)

    def set_manual(self, vmin: float, vmax: float) -> None:
        """Switch to manual mode with explicit display limits.

        If *vmin >= vmax*, *vmax* is nudged up by a relative amount so
        rendering never receives a degenerate range at any magnitude.
        """
        vmin = float(vmin)
        vmax = float(vmax)
        if vmax <= vmin:
            sep = max(abs(vmin) * 1e-9, 1e-30)
            vmax = vmin + sep
        self.mode = "manual"
        self.vmin = vmin
        self.vmax = vmax

    def reset(self, low_pct: float = 1.0, high_pct: float = 99.0) -> None:
        """Return to percentile mode with the given (or default) percentiles."""
        self.mode     = "percentile"
        self.low_pct  = float(low_pct)
        self.high_pct = float(high_pct)
        self.vmin     = None
        self.vmax     = None

    # ── Limit resolution ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-compatible dictionary of the current display state.

        ``vmin`` and ``vmax`` are always present (as ``None`` in percentile mode)
        so downstream consumers can rely on a fixed key set.
        """
        return {
            "mode":     self.mode,
            "low_pct":  self.low_pct,
            "high_pct": self.high_pct,
            "vmin":     self.vmin,
            "vmax":     self.vmax,
        }

    def resolve(self, arr) -> tuple[float | None, float | None]:
        """Return (vmin, vmax) for rendering.

        In *manual* mode: return the stored limits directly.
        In *percentile* mode: compute from *arr* using
        :func:`probeflow.processing.display.clip_range_from_array`.

        Returns ``(None, None)`` if limits cannot be determined (e.g. all-NaN
        array in percentile mode).

        Parameters
        ----------
        arr:
            The array that will be rendered.  Only used in percentile mode.
        """
        if self.mode == "manual":
            if self.vmin is not None and self.vmax is not None:
                return self.vmin, self.vmax
            # fall through to percentile if manual state is incomplete
        # percentile mode (or manual fallback)
        from probeflow.processing.display import clip_range_from_array
        try:
            return clip_range_from_array(arr, self.low_pct, self.high_pct)
        except ValueError:
            return None, None
