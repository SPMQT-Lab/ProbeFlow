"""Bad scan-line segment detection and repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

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


# Maximum sub-threshold gap (px) bridged inside a MAD segment so per-pixel
# noise holes don't shatter a coherent defect into fragments.  Kept small and
# fixed (a noise-hole scale), independent of the minimum feature length.
_NOISE_GAP_TOL_PX = 4

# Along-row smoothing window (px) applied to the residual before MAD detection.
# Acts as a matched filter for *extended* scan-line defects: a coherent line is
# preserved while per-pixel noise averages down (SNR gain ~ sqrt(window)), so a
# faint-but-long bad line clears the threshold without texture flooding in.
_MAD_SMOOTH_WINDOW_PX = 11


def _smooth_rows_nanaware(arr: np.ndarray, window: int) -> np.ndarray:
    """Moving average along each row (axis 1), ignoring non-finite pixels."""
    if window <= 1:
        return arr
    finite = np.isfinite(arr)
    vals = np.where(finite, arr, 0.0)
    num = uniform_filter1d(vals, size=window, axis=1, mode="nearest")
    den = uniform_filter1d(finite.astype(np.float64), size=window, axis=1, mode="nearest")
    out = np.full_like(arr, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 1e-6)
    return out


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


def _close_small_gaps(mask: np.ndarray, gap_tol: int) -> np.ndarray:
    """Bridge ``False`` gaps no longer than ``gap_tol`` between two ``True`` runs.

    Per-pixel noise punches sub-threshold holes through an otherwise coherent
    bright/dark segment; closing those holes lets the segment be recovered as a
    single run instead of many short fragments.
    """
    if gap_tol <= 0:
        return mask
    out = np.asarray(mask, dtype=bool).copy()
    runs = _contiguous_true_runs(out)
    for (_s0, e0), (s1, _e1) in zip(runs, runs[1:]):
        if s1 - e0 <= gap_tol:
            out[e0:s1] = True
    return out


def _split_segments_by_adjacent_limit(
    segments: list[BadSegment] | tuple[BadSegment, ...],
    max_adjacent_bad_lines: int,
) -> tuple[list[BadSegment], list[BadSegment]]:
    """Return (accepted, skipped) after applying the adjacent-line safety limit.

    The limit guards against repairing a vertical *block* of bad pixels that has
    no clean neighbour row to copy from.  Crucially it is **column-aware**: two
    segments on adjacent lines only belong to the same stack when their column
    ranges overlap.  Segments that merely share a line index with unrelated bad
    lines elsewhere in the image are independently repairable and are kept.  A
    stack taller than ``max_adjacent_bad_lines`` consecutive overlapping lines is
    skipped; everything else is repaired.
    """
    max_adjacent_bad_lines = max(1, int(max_adjacent_bad_lines))
    segs = list(segments)
    n = len(segs)
    if n == 0:
        return [], []

    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_line: dict[int, list[int]] = {}
    for i, seg in enumerate(segs):
        by_line.setdefault(int(seg.line_index), []).append(i)

    def _overlaps(a: int, b: int) -> bool:
        return (int(segs[a].start_col) < int(segs[b].end_col)
                and int(segs[b].start_col) < int(segs[a].end_col))

    # Link a segment to any column-overlapping segment on the next line; the
    # connected components are the vertical stacks of contiguous bad pixels.
    for i, seg in enumerate(segs):
        for j in by_line.get(int(seg.line_index) + 1, ()):
            if _overlaps(i, j):
                union(i, j)

    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    accepted: list[BadSegment] = []
    skipped: list[BadSegment] = []
    for members in components.values():
        lines = {int(segs[i].line_index) for i in members}
        # Members are linked only through ±1-line overlaps, so the line set is a
        # consecutive run; its span is the stack thickness.
        stack_height = max(lines) - min(lines) + 1
        target = accepted if stack_height <= max_adjacent_bad_lines else skipped
        target.extend(segs[i] for i in members)

    accepted.sort(key=lambda s: (int(s.line_index), int(s.start_col)))
    skipped.sort(key=lambda s: (int(s.line_index), int(s.start_col)))
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

    # The residual is already referenced to the *neighbour rows*, so good pixels
    # sit at ~0 and a bright/dark defect is a direct deviation from zero.  We do
    # NOT subtract each row's own median (the previous behaviour): once a defect
    # covers more than ~half the row it becomes the row median, nets its own
    # pixels to ~0, and vanishes — so lowering the threshold only surfaced noise.
    #
    # Scan-line defects are *extended*, so we smooth the residual along each row
    # first (a matched filter): this preserves a coherent line's amplitude while
    # averaging per-pixel noise down, letting a faint-but-long line clear the
    # threshold without texture flooding in.  ``step`` remains the detector for
    # sharp, short scars.
    window = max(1, min(_MAD_SMOOTH_WINDOW_PX, Nx))
    smoothed = _smooth_rows_nanaware(residuals, window)
    scale = _robust_scale(smoothed)
    cutoff = threshold * scale
    if not np.isfinite(cutoff):
        return []

    sign = 1.0 if polarity == "bright" else -1.0
    segments = []
    for row in range(Ny):
        sm = smoothed[row]
        finite = np.isfinite(sm)
        if not finite.any():
            continue
        bad = finite & ((sign * sm) > cutoff)
        # Bridge short noise-sized holes (independent of ``min_length`` so
        # raising the minimum feature length to reject texture does not start
        # merging unrelated noise spikes into spurious long runs).
        bad = _close_small_gaps(bad, _NOISE_GAP_TOL_PX) & finite
        for start, end in _contiguous_true_runs(bad):
            # No upper-length cap: a long (even full-width) coherent defect is
            # exactly the target, and the neighbour-referenced baseline stays
            # valid no matter how much of the row is affected.
            if end - start < min_length:
                continue
            seg = sm[start:end]
            seg = seg[np.isfinite(seg)]
            if seg.size == 0:
                continue
            seg_median = float(np.median(seg))
            if (sign * seg_median) <= cutoff:
                continue  # not a defect on the median
            score = float(abs(seg_median) / max(scale, 1e-15))
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
