"""Pure browse metadata filters and completion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FolderFilterState:
    """Session-local browse folder filters."""

    size_enabled: bool = False
    min_width_nm: Optional[float] = 0.0
    max_width_nm: Optional[float] = None
    min_height_nm: Optional[float] = 0.0
    max_height_nm: Optional[float] = None
    completion_enabled: bool = False
    min_completion_pct: Optional[float] = None
    bias_enabled: bool = False
    min_bias_mv: Optional[float] = None
    max_bias_mv: Optional[float] = None

    def has_metadata_filters(self) -> bool:
        return bool(self.size_enabled or self.completion_enabled or self.bias_enabled)


@dataclass(frozen=True)
class FilterMatchCounts:
    total_folders: int = 0
    visible_folders: int = 0
    total_scans: int = 0
    visible_scans: int = 0
    total_spectra: int = 0
    visible_spectra: int = 0

    @property
    def hidden_items(self) -> int:
        return max(
            0,
            (self.total_folders - self.visible_folders)
            + (self.total_scans - self.visible_scans)
            + (self.total_spectra - self.visible_spectra),
        )


def completion_pct_from_visible_range(
    width_m: Optional[float],
    height_m: Optional[float],
) -> Optional[float]:
    """Return completion from visible physical dimensions."""
    if width_m is None or height_m is None:
        return None
    width = float(width_m)
    height = float(height_m)
    if width <= 0.0 or height <= 0.0:
        return None
    largest = max(width, height)
    if largest <= 0.0:
        return None
    return 100.0 * ((width * height) / (largest * largest))


def createc_visible_height_m(
    full_height_m: Optional[float],
    original_ny: Optional[int],
    decoded_ny: Optional[int],
) -> Optional[float]:
    """Return the visible Createc height for a possibly partial scan."""
    if full_height_m is None:
        return None
    full_height = float(full_height_m)
    if full_height <= 0.0:
        return None
    try:
        original = int(original_ny) if original_ny is not None else 0
        decoded = int(decoded_ny) if decoded_ny is not None else 0
    except (TypeError, ValueError):
        return full_height
    if original <= 0 or decoded <= 0:
        return full_height
    return full_height * (decoded / original)


def scan_matches_folder_filters(
    *,
    width_nm: Optional[float],
    height_nm: Optional[float],
    completion_pct: Optional[float],
    bias_mv: Optional[float],
    state: FolderFilterState,
) -> bool:
    """Return True when one scan satisfies all active metadata filters."""
    if state.size_enabled:
        if (
            width_nm is None
            or height_nm is None
            or state.min_width_nm is None
            or state.max_width_nm is None
            or state.min_height_nm is None
            or state.max_height_nm is None
        ):
            return False
        if float(width_nm) < float(state.min_width_nm):
            return False
        if float(width_nm) > float(state.max_width_nm):
            return False
        if float(height_nm) < float(state.min_height_nm):
            return False
        if float(height_nm) > float(state.max_height_nm):
            return False

    if state.completion_enabled:
        if completion_pct is None or state.min_completion_pct is None:
            return False
        if float(completion_pct) < float(state.min_completion_pct):
            return False

    if state.bias_enabled:
        if (
            bias_mv is None
            or state.min_bias_mv is None
            or state.max_bias_mv is None
        ):
            return False
        bias = float(bias_mv)
        if bias < float(state.min_bias_mv) or bias > float(state.max_bias_mv):
            return False

    return True
