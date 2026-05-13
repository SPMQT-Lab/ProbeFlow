"""Data models for non-destructive spectroscopy display transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpectrumTrace:
    """Raw spectroscopy trace used as the source for display transforms."""

    source_file: str
    spectrum_id: str
    x_channel: str
    y_channel: str
    x_raw: np.ndarray
    y_raw: np.ndarray
    x_label: str = "x"
    y_label: str = "signal"
    x_unit: str = ""
    y_unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpectrumDisplayOptions:
    """Display transform choices for a spectroscopy trace."""

    smoothing_mode: str = "none"
    smoothing_points: int | None = None
    savgol_polyorder: int = 2
    derivative: bool = False
    normalize_mode: str = "none"
    normalize_constant: float | None = None
    normalize_channel: str | None = None
    outlier_mode: str = "none"
    outlier_threshold: float | None = None
    vertical_offset: float = 0.0


@dataclass(frozen=True)
class DisplayedSpectrum:
    """Derived spectroscopy data that exactly represents a plotted/exported trace."""

    source_file: str
    spectrum_id: str
    label: str
    x_channel: str
    y_channel: str
    x_display: np.ndarray
    y_display: np.ndarray
    mask: np.ndarray | None
    options: SpectrumDisplayOptions
    metadata: dict[str, Any]
    x_label: str = "x"
    y_label: str = "signal"
    x_unit: str = ""
    y_unit: str = ""

    @property
    def excluded_indices(self) -> list[int]:
        if self.mask is None:
            return []
        return np.flatnonzero(~self.mask).astype(int).tolist()
