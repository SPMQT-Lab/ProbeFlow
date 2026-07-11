"""Pure browse metadata filters and completion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# A scan counts as "incomplete" when it recorded less than this fraction of
# its frame (stopped early). Used by the hide-incomplete browse filter.
INCOMPLETE_COMPLETION_PCT = 50.0

# Two bias values within this distance (mV) are treated as the same setpoint
# when grouping scans for the browse bias picker.
BIAS_MATCH_TOLERANCE_MV = 0.5


@dataclass(frozen=True)
class FolderFilterState:
    """Session-local browse folder filters.

    ``bias_value_mv`` — show only scans acquired at this bias (within
    ``BIAS_MATCH_TOLERANCE_MV``); ``None`` shows every bias.
    ``hide_incomplete`` — hide scans that recorded less than
    ``INCOMPLETE_COMPLETION_PCT`` of their frame; scans without completion
    metadata are left visible.
    """

    bias_value_mv: Optional[float] = None
    hide_incomplete: bool = False

    def has_metadata_filters(self) -> bool:
        return bool(self.bias_value_mv is not None or self.hide_incomplete)


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
    completion_pct: Optional[float],
    bias_mv: Optional[float],
    state: FolderFilterState,
) -> bool:
    """Return True when one scan satisfies all active metadata filters."""
    if state.bias_value_mv is not None:
        if bias_mv is None:
            return False
        if abs(float(bias_mv) - float(state.bias_value_mv)) > BIAS_MATCH_TOLERANCE_MV:
            return False

    if state.hide_incomplete and completion_pct is not None:
        if float(completion_pct) < INCOMPLETE_COMPLETION_PCT:
            return False

    return True


def bias_options_from_values(
    bias_values_mv: list[Optional[float]],
) -> list[tuple[float, int]]:
    """Group scan biases into distinct picker options.

    Returns ``(bias_mv, count)`` pairs sorted by bias, grouping values within
    ``BIAS_MATCH_TOLERANCE_MV`` of each other (each group is represented by
    its first-seen value). ``None`` entries are ignored.
    """
    options: list[tuple[float, int]] = []
    for value in bias_values_mv:
        if value is None:
            continue
        bias = float(value)
        for i, (existing, count) in enumerate(options):
            if abs(bias - existing) <= BIAS_MATCH_TOLERANCE_MV:
                options[i] = (existing, count + 1)
                break
        else:
            options.append((bias, 1))
    return sorted(options, key=lambda pair: pair[0])
