"""Reactive wrapper around DisplayRangeState for the image viewer.

Emits ``rangeChanged`` whenever the display range is mutated, so the viewer
can connect its refresh methods once rather than sprinkling explicit refresh
calls throughout every handler.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from probeflow.processing.display_state import DisplayRangeState


class DisplayRangeController(QObject):
    """Wraps :class:`~probeflow.processing.display_state.DisplayRangeState`.

    Emits :attr:`rangeChanged` after each mutating call so connected slots
    (e.g. ``_refresh_display_range``) fire automatically.
    """

    rangeChanged = Signal()

    def __init__(
        self,
        *,
        clip_low: float = 1.0,
        clip_high: float = 99.0,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = DisplayRangeState(low_pct=clip_low, high_pct=clip_high)

    # ── Read-only state access ─────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._state.mode

    @property
    def low_pct(self) -> float:
        return self._state.low_pct

    @property
    def high_pct(self) -> float:
        return self._state.high_pct

    @property
    def vmin(self) -> float | None:
        return self._state.vmin

    @property
    def vmax(self) -> float | None:
        return self._state.vmax

    # ── Delegated operations (emit rangeChanged after each) ────────────────────

    def resolve(self, arr) -> tuple[float | None, float | None]:
        """Compute (vmin, vmax) for rendering — pure read, no signal."""
        return self._state.resolve(arr)

    def to_dict(self) -> dict:
        """Return a JSON-compatible dict of the current state."""
        return self._state.to_dict()

    def set_manual(self, vmin: float, vmax: float) -> None:
        """Switch to manual mode with explicit limits and emit rangeChanged."""
        self._state.set_manual(vmin, vmax)
        self.rangeChanged.emit()

    def set_percentile(self, low_pct: float, high_pct: float) -> None:
        """Switch to percentile mode and emit rangeChanged."""
        self._state.set_percentile(low_pct, high_pct)
        self.rangeChanged.emit()

    def reset(self, low_pct: float = 1.0, high_pct: float = 99.0) -> None:
        """Return to percentile mode with default percentiles and emit rangeChanged."""
        self._state.reset(low_pct, high_pct)
        self.rangeChanged.emit()

    def reset_silent(self, low_pct: float = 1.0, high_pct: float = 99.0) -> None:
        """Reset without emitting rangeChanged — for callers that manage refresh."""
        self._state.reset(low_pct, high_pct)
