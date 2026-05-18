"""Bad scan-line segment detection and repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._image_utils import _nonnegative_finite


@dataclass(frozen=True)
class BadSegment:
    """Detected bad fast-scan segment on one image row.

    ``end_col`` is exclusive, matching normal NumPy slicing:
    ``image[line_index, start_col:end_col]``.
    """

    line_index: int
    start_col: int
    end_col: int
    score: float
    method: str


@dataclass(frozen=True)
class BadLineCorrectionInfo:
    """Summary returned by bad scan-line segment correction."""

    segments: tuple[BadSegment, ...]
    skipped_segments: tuple[BadSegment, ...]
    method: str
    threshold: float
    corrected_segments: tuple[BadSegment, ...] = ()
    polarity: str = "bright"
    min_segment_length_px: int = 2
    max_adjacent_bad_lines: int = 1


def _normalise_bad_segment_method(method: str) -> str:
    method = str(method or "step").lower().replace("-", "_")
    aliases = {
        "step": "step",
        "step_segments": "step",
        "segments": "step",
        "mad": "mad",
        "mad_segments": "mad",
        "outlier": "mad",
        "outlier_segments": "mad",
    }
    if method not in aliases:
        raise ValueError(f"method must be 'step' or 'mad', got {method!r}")
    return aliases[method]


def _normalise_bad_segment_polarity(polarity: str) -> str:
    polarity = str(polarity or "bright").lower()
    if polarity not in {"bright", "dark"}:
        raise ValueError(f"polarity must be 'bright' or 'dark', got {polarity!r}")
    return polarity


def _robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    if mad > 0.0:
        return 1.4826 * mad
    fallback = float(np.max(np.abs(finite))) * 1e-3
    if fallback > 0.0:
        return fallback
    return np.finfo(np.float64).eps


def _nearest_line_baseline(
    a: np.ndarray,
    row: int,
    radius: int = 2,
) -> np.ndarray | None:
    Ny, _Nx = a.shape
    rows: list[np.ndarray] = []
    for offset in range(1, radius + 1):
        above = row - offset
        below = row + offset
        if above >= 0 and np.isfinite(a[above]).any():
            rows.append(a[above])
        if below < Ny and np.isfinite(a[below]).any():
            rows.append(a[below])
    if rows:
        with np.errstate(all="ignore"):
            return np.nanmedian(np.vstack(rows), axis=0)
    return None


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, int(mask.size)))
    return runs


def _split_segments_by_adjacent_limit(
    segments: list[BadSegment] | tuple[BadSegment, ...],
    max_adjacent_bad_lines: int,
) -> tuple[list[BadSegment], list[BadSegment]]:
    """Return (accepted, skipped) after applying adjacent-line safety limit."""
    max_adjacent_bad_lines = max(1, int(max_adjacent_bad_lines))
    by_line: dict[int, list[BadSegment]] = {}
    for seg in segments:
        by_line.setdefault(int(seg.line_index), []).append(seg)
    accepted: list[BadSegment] = []
    skipped: list[BadSegment] = []
    lines = sorted(by_line)
    i = 0
    while i < len(lines):
        group = [lines[i]]
        i += 1
        while i < len(lines) and lines[i] == group[-1] + 1:
            group.append(lines[i])
            i += 1
        group_segments = [seg for line in group for seg in by_line[line]]
        if len(group) > max_adjacent_bad_lines:
            skipped.extend(group_segments)
        else:
            accepted.extend(group_segments)
    return accepted, skipped


def detect_bad_scanline_segments(
    image: np.ndarray,
    threshold: float = 5.0,
    *,
    method: str = "step",
    polarity: str = "bright",
    min_segment_length_px: int = 2,
    max_adjacent_bad_lines: int = 1,
    min_length: int | None = None,
    max_segment_fraction: float = 0.80,
) -> list[BadSegment]:
    """Detect bad fast-scan row segments without modifying image data."""
    method = _normalise_bad_segment_method(method)
    polarity = _normalise_bad_segment_polarity(polarity)
    threshold = _nonnegative_finite(threshold, "threshold")
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("image must be a 2-D array")
    Ny, Nx = a.shape
    if Ny == 0 or Nx == 0:
        return []
    _ = max(1, int(max_adjacent_bad_lines))  # accepted for API parity with repair
    if min_length is not None:
        min_segment_length_px = min_length
    min_length = max(1, int(min_segment_length_px))
    max_len = max(min_length, int(round(float(max_segment_fraction) * Nx)))

    residuals = np.full_like(a, np.nan, dtype=np.float64)
    for row in range(Ny):
        baseline = _nearest_line_baseline(a, row)
        if baseline is not None:
            residuals[row] = a[row] - baseline

    if method == "step":
        diffs = np.diff(residuals, axis=1)
        scale = _robust_scale(diffs)
        cutoff = threshold * scale
        segments: list[BadSegment] = []
        if not np.isfinite(cutoff):
            return []
        for row in range(Ny):
            row_diffs = diffs[row]
            finite = np.isfinite(row_diffs)
            edge_idx = np.where(finite & (np.abs(row_diffs) > cutoff))[0]
            k = 0
            while k < len(edge_idx) - 1:
                left = int(edge_idx[k])
                right = int(edge_idx[k + 1])
                left_step = float(row_diffs[left])
                right_step = float(row_diffs[right])
                if polarity == "bright":
                    is_candidate = left_step > cutoff and right_step < -cutoff
                else:
                    is_candidate = left_step < -cutoff and right_step > cutoff
                if not is_candidate:
                    k += 1
                    continue
                start = left + 1
                end = right + 1
                length = end - start
                if min_length <= length <= max_len:
                    score = min(abs(left_step), abs(right_step)) / max(scale, 1e-15)
                    segments.append(BadSegment(row, start, end, float(score), method))
                k += 2
        return segments

    scale = _robust_scale(residuals)
    cutoff = threshold * scale
    if not np.isfinite(cutoff):
        return []
    segments = []
    for row in range(Ny):
        residual = residuals[row]
        finite = np.isfinite(residual)
        if not finite.any():
            continue
        centre = float(np.median(residual[finite]))
        if polarity == "bright":
            bad = finite & ((residual - centre) > cutoff)
        else:
            bad = finite & ((residual - centre) < -cutoff)
        for start, end in _contiguous_true_runs(bad):
            length = end - start
            if not (min_length <= length <= max_len):
                continue
            local = np.abs(residual[start:end] - centre)
            score = float(np.nanmedian(local) / max(scale, 1e-15))
            segments.append(BadSegment(row, start, end, score, method))
    return segments


def _valid_neighbour_for_segment(
    a: np.ndarray,
    bad_mask: np.ndarray,
    row: int,
    j0: int,
    j1: int,
    direction: int,
) -> int | None:
    candidate = row + direction
    Ny = a.shape[0]
    while 0 <= candidate < Ny:
        values = a[candidate, j0:j1]
        if np.isfinite(values).any() and not bad_mask[candidate, j0:j1].any():
            return candidate
        candidate += direction
    return None


def repair_bad_scanline_segments(
    image: np.ndarray,
    segments: list[BadSegment] | tuple[BadSegment, ...],
    *,
    max_adjacent_bad_lines: int = 1,
    threshold: float = float("nan"),
    polarity: str = "bright",
    min_segment_length_px: int = 2,
) -> tuple[np.ndarray, BadLineCorrectionInfo]:
    """Repair only the provided bad scan-line segments."""
    polarity = _normalise_bad_segment_polarity(polarity)
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("image must be a 2-D array")
    corrected = a.copy()
    Ny, Nx = corrected.shape
    accepted_segments, unsafe_segments = _split_segments_by_adjacent_limit(
        segments,
        max_adjacent_bad_lines,
    )
    valid_segments: list[BadSegment] = []
    skipped: list[BadSegment] = list(unsafe_segments)
    bad_mask = np.zeros(corrected.shape, dtype=bool)
    for seg in accepted_segments:
        row = int(seg.line_index)
        j0 = max(0, min(Nx, int(seg.start_col)))
        j1 = max(0, min(Nx, int(seg.end_col)))
        if not (0 <= row < Ny and j1 > j0):
            skipped.append(seg)
            continue
        bad_mask[row, j0:j1] = True
        if j0 == seg.start_col and j1 == seg.end_col:
            valid_segments.append(seg)
        else:
            valid_segments.append(BadSegment(row, j0, j1, seg.score, seg.method))

    corrected_segments: list[BadSegment] = []
    for seg in valid_segments:
        row = int(seg.line_index)
        j0 = int(seg.start_col)
        j1 = int(seg.end_col)
        above = _valid_neighbour_for_segment(corrected, bad_mask, row, j0, j1, -1)
        below = _valid_neighbour_for_segment(corrected, bad_mask, row, j0, j1, 1)
        if above is None and below is None:
            skipped.append(seg)
            continue
        if above is not None and below is not None:
            replacement = 0.5 * (corrected[above, j0:j1] + corrected[below, j0:j1])
        elif above is not None:
            replacement = corrected[above, j0:j1]
        else:
            replacement = corrected[below, j0:j1]
        finite_replacement = np.isfinite(replacement)
        if finite_replacement.any():
            target = corrected[row, j0:j1]
            target[finite_replacement] = replacement[finite_replacement]
            corrected[row, j0:j1] = target
            corrected_segments.append(seg)
        else:
            skipped.append(seg)

    method = valid_segments[0].method if valid_segments else "step"
    info = BadLineCorrectionInfo(
        segments=tuple(valid_segments),
        skipped_segments=tuple(skipped),
        method=method,
        threshold=float(threshold),
        corrected_segments=tuple(corrected_segments),
        polarity=polarity,
        min_segment_length_px=int(min_segment_length_px),
        max_adjacent_bad_lines=int(max_adjacent_bad_lines),
    )
    return corrected, info


def correct_bad_scanline_segments(
    image: np.ndarray,
    threshold: float = 5.0,
    *,
    method: str = "step",
    polarity: str = "bright",
    min_segment_length_px: int = 2,
    max_adjacent_bad_lines: int = 1,
    min_length: int | None = None,
    max_segment_fraction: float = 0.80,
) -> tuple[np.ndarray, BadLineCorrectionInfo]:
    """Detect and repair bad fast-scan row segments."""
    method = _normalise_bad_segment_method(method)
    polarity = _normalise_bad_segment_polarity(polarity)
    threshold = _nonnegative_finite(threshold, "threshold")
    if min_length is not None:
        min_segment_length_px = min_length
    segments = detect_bad_scanline_segments(
        image,
        threshold,
        method=method,
        polarity=polarity,
        min_segment_length_px=min_segment_length_px,
        max_adjacent_bad_lines=max_adjacent_bad_lines,
        max_segment_fraction=max_segment_fraction,
    )
    corrected, repair_info = repair_bad_scanline_segments(
        image,
        segments,
        max_adjacent_bad_lines=max_adjacent_bad_lines,
        threshold=threshold,
        polarity=polarity,
        min_segment_length_px=min_segment_length_px,
    )
    info = BadLineCorrectionInfo(
        segments=tuple(segments),
        skipped_segments=repair_info.skipped_segments,
        method=method,
        threshold=threshold,
        corrected_segments=repair_info.corrected_segments,
        polarity=polarity,
        min_segment_length_px=int(min_segment_length_px),
        max_adjacent_bad_lines=int(max_adjacent_bad_lines),
    )
    return corrected, info


def remove_bad_lines(
    arr: np.ndarray,
    threshold_mad: float = 5.0,
    *,
    method: str = "mad",
    polarity: str = "bright",
    min_segment_length_px: int = 2,
    max_adjacent_bad_lines: int = 1,
) -> np.ndarray:
    """Correct bad fast-scan line segments by neighbour-line interpolation."""
    threshold_mad = _nonnegative_finite(threshold_mad, "threshold_mad")
    corrected, _info = correct_bad_scanline_segments(
        arr,
        threshold=threshold_mad,
        method=method,
        polarity=polarity,
        min_segment_length_px=min_segment_length_px,
        max_adjacent_bad_lines=max_adjacent_bad_lines,
    )
    return corrected
