"""Pure outlier masking helpers for spectroscopy display traces."""

from __future__ import annotations

import numpy as np


def apply_outlier_mask(
    x: np.ndarray,
    y: np.ndarray,
    *,
    mode: str = "none",
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x/y with outliers omitted plus a keep-mask over the input arrays.

    ``threshold`` (default 6.0) is in **raw MAD multiples**, not σ
    equivalents.  For normally distributed data the MAD-to-σ scaling
    factor is ~1.4826, so a threshold of 6 MAD corresponds to roughly
    ~9σ.  If you want a σ-equivalent threshold pass ``threshold = N * 1.4826``
    where ``N`` is your desired number of standard deviations.  Review
    physics #15 / spectroscopy outlier semantics clarified 2026-05-28
    (no behaviour change).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have matching shapes")

    mode = (mode or "none").strip().lower()
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mode in {"none", "off"}:
        return x_arr[keep].copy(), y_arr[keep].copy(), keep

    limit = float(threshold if threshold is not None else 6.0)
    if limit <= 0:
        raise ValueError("outlier threshold must be positive")

    if mode == "mad":
        scores = robust_scores(y_arr)
        keep &= scores <= limit
    elif mode in {"jump", "derivative_jump", "derivative-jump"}:
        jumps = np.diff(y_arr)
        bad = np.zeros(y_arr.shape, dtype=bool)
        abs_jumps = np.abs(jumps)
        finite_jumps = abs_jumps[np.isfinite(abs_jumps)]
        typical = float(np.nanpercentile(finite_jumps, 25)) if finite_jumps.size else 0.0
        jump_limit = limit * typical if typical > 0 else 0.0
        for i in range(1, y_arr.size - 1):
            left = y_arr[i] - y_arr[i - 1]
            right = y_arr[i + 1] - y_arr[i]
            if (
                np.isfinite(left)
                and np.isfinite(right)
                and np.sign(left) != np.sign(right)
                and min(abs(left), abs(right)) > jump_limit
            ):
                bad[i] = True
        if not np.any(bad):
            jump_scores = robust_scores(jumps)
            bad_jumps = jump_scores > limit
            bad[:-1] |= bad_jumps
            bad[1:] |= bad_jumps
        keep &= ~bad
    else:
        raise ValueError(f"Unknown outlier mode: {mode!r}")

    return x_arr[keep].copy(), y_arr[keep].copy(), keep


def robust_scores(values: np.ndarray) -> np.ndarray:
    """Return ``|x - median| / MAD`` scores (in raw MAD multiples).

    No 1.4826 σ-equivalence scaling is applied — the score is the raw
    deviation-over-MAD ratio.  Non-finite inputs return inf so a
    threshold comparison naturally excludes them.  See
    :func:`apply_outlier_mask` for the threshold-units note.
    """
    arr = np.asarray(values, dtype=np.float64)
    scores = np.zeros(arr.shape, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        scores[:] = np.inf
        return scores

    med = float(np.nanmedian(arr[finite]))
    dev = np.abs(arr - med)
    mad = float(np.nanmedian(dev[finite]))
    if mad > 0:
        scores = dev / mad
        scores[~finite] = np.inf
        return scores

    scores[:] = 0.0
    scores[(dev > 0) | ~finite] = np.inf
    return scores
