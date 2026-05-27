# ProbeFlow Numerical Stability Review
Date: 2026-05-27
Reviewer: numerical-stability agent (parallel deep review pass)

## Summary
- 18 findings: S0=1, S1=8, S2=7, S3=2
- Top 3 concerns:
  1. Polar decomposition in `lattice_distortion._polar_decompose` silently produces a non-SPD "stretch" matrix when the input has a reflection (det T < 0), so "preserve orientation" mode mis-applies the correction.
  2. `tv_denoise` runs an extra iteration past `max_iter` and propagates NaNs back to *every* originally-NaN pixel even after they were mean-filled (mask reshape sanity OK), but the convergence/iter-count bug means callers asking for 1 iteration get 2.
  3. `quantize_bit_depth` quantises with `vmax/vmin` from the live finite data, so two physically-identical pixel values quantised in different scans land on different reconstruction levels — a hidden non-deterministic mapping that breaks before/after image arithmetic and corrupts provenance reproducibility.

## Findings

### 1. [S0] Polar decomposition: stretch matrix becomes indefinite under reflection
**Location**: `probeflow/analysis/lattice_distortion.py:193-200`
**Problem**: When `det(T) < 0` the SVD fix flips `U[:, -1] *= -1` *and* `sigma[-1] *= -1`. Singular values must remain non-negative; making one negative breaks the standard polar decomposition `S = V·Σ·Vᵀ`. The resulting `S` is symmetric but no longer positive semidefinite. Realistic trigger: measured lattice vectors swapped (rare but possible when extracted from FFT picks that flip handedness, or when scan_y is reversed) → `det(T) < 0` is exactly the case the fix is meant to handle.
**Why it matters**: "Preserve image orientation" mode (`stretch_matrix`) silently applies a reflective/indefinite transform that shrinks the image along one axis with a sign error — pixels collapse or invert with no warning. This is the canonical use case for the polar decomposition and the code only triggers when the user most needs it.
**Suggested fix**: Standard convention is `U' = U · diag(1,…,1,det(U·Vᵀ))`, then `R = U' · Vᵀ` and `S = V · Σ · Vᵀ` with the **original (positive) Σ**. Do not modify `sigma`.

### 2. [S1] `tv_denoise` runs `max_iter + 1` iterations
**Location**: `probeflow/processing/tv.py:127`
**Problem**: The loop is `for it in range(max_iter + 1):`. The convergence check only runs at multiples of 50 and even then a hard cap of `max_iter` is exceeded by one iteration in the trivial case (`max_iter=1` → executes twice). For `max_iter=0` it still runs once.
**Why it matters**: Off-by-one in iteration count; not catastrophic on its own, but interacts badly with the convergence test (which is only sampled every 50 iterations, so for any `max_iter < 50` no convergence test ever runs and the function ignores `tol`). Documented behaviour ("Hard cap on iterations") is violated.
**Suggested fix**: Use `range(max_iter)` and check `tol` every iteration once the first 5–10 iterations have warmed up.

### 3. [S1] `quantize_bit_depth` uses data-dependent scale, not a fixed physical range
**Location**: `probeflow/processing/geometry.py:664-678`
**Problem**: `vmin, vmax` are computed from the live finite pixels. Applying the same "convert to 8-bit" step to the same physical SI quantity on two different scans (or on a cropped vs. uncropped view) maps it to a different reconstruction level. Combined with the new processing-ops pipeline, this means the *quantized values themselves depend on which crop/ROI was active when the step ran*.
**Why it matters**: (a) Image arithmetic between two quantised scans produces noise instead of structural differences; (b) provenance is not reproducible — re-running the same op on the same source gives a different output if any prior crop/ROI step changed the finite-data extent; (c) NaN pixels that re-enter via a later inverse op land outside the captured `[vmin, vmax]` and become out-of-range. The docstring promises only "the result contains 2**bits distinct values" but the GUI sells this as Image > Type > 8-bit, which users will assume is a calibrated quantization to a fixed physical range.
**Suggested fix**: Either (a) take an explicit `vmin, vmax` argument (default to current behaviour but record it in the op params for reproducibility), or (b) snap to a calibrated SI range (e.g. nanometres for topography) and clamp out-of-range pixels to ±vmax with a warning.

### 4. [S1] `_polar_decompose` rotation angle wrong when det was originally negative
**Location**: `probeflow/analysis/lattice_distortion.py:194-201`
**Problem**: After the broken sigma flip (finding 1), `R = U @ Vt` is recomputed but `np.linalg.det(R) < 0` is never re-checked. Even with a correct sign-fix (finding 1), `rotation_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))` assumes R is a proper rotation; under reflection the angle is ambiguous and the reported `polar_rotation_deg` becomes meaningless.
**Why it matters**: Provenance records `polar_rotation_deg` (lattice_correction_workflow.py:59); a wrong value silently misleads downstream analysis and any human reading the export.
**Suggested fix**: After the determinant fix, assert `det(R) ≈ +1`; if the input was reflective, document `polar_rotation_deg = NaN` instead of computing a misleading angle.

### 5. [S1] `find_bragg_peaks_in_annulus`: equal magnitudes never picked
**Location**: `probeflow/processing/filters.py:707-708`
**Problem**: Candidate selection uses `mag == local_max_img` with floating-point equality. After `maximum_filter`, a plateau (two adjacent equal peaks — common in highly symmetric Bragg spots, or in log-spaced magnitudes after quantisation) leaves only **one** sub-pixel as exactly equal; ties on a 2-pixel plateau give *both* picked because they both match `local_max_img`, but on a longer plateau they all match → density-of-candidates explodes and `expected_count` runs out before reaching far-side peaks. More importantly, the equality test against `maximum_filter` output is brittle for float64 inputs that were typed back from log1p.
**Why it matters**: For perfectly symmetric square / hex lattices on noise-free synthetic test data, the function can return all 6 picks at one azimuth (one plateau) and zero at the others. Real-world impact is moderate but the asymmetry is silent.
**Suggested fix**: Use `mag >= local_max_img - eps`, where eps is a small fraction of the local magnitude, then break plateau ties by centroid.

### 6. [S1] `_modal_shift` returns `mid-bin` when no pixels lie inside the modal bin
**Location**: `probeflow/processing/background.py:645-648`
**Problem**: `in_peak = (values >= edges[peak]) & (values <= edges[peak + 1])`. With `np.histogram`'s closed-right convention on the last bin only, pixels exactly at the bin boundary land in two bins depending on float jitter. Worse, the function falls back to `0.5 * (edges[peak] + edges[peak + 1])` when `np.any(in_peak)` is False — but np.histogram guarantees ≥1 element in any non-empty bin, so this branch only triggers under extreme float jitter. The mid-bin fallback is fine, but if a single NaN slips through `_modal_shift` callers (such as `stm_line_background`) it becomes `prev_shift = mid-bin = NaN` and propagates.
**Why it matters**: For typical STM data this is harmless; for data with extreme dynamic range a single pixel can drift `prev_shift` enough that subsequent rows compound into a visible baseline drift.
**Suggested fix**: Tighten bin edges with `np.searchsorted(edges, values, side='right') == peak + 1`.

### 7. [S1] `align_rows(method='linear')` ignores explicit NaN restoration
**Location**: `probeflow/processing/alignment.py:48-56`
**Problem**: `arr[r] -= np.polyval(coeffs, xs)` subtracts the polynomial across *every* column including the originally-NaN ones. After the subtraction `nan - finite = nan` for those columns, which is OK by accident — but `_finite_mean` / `_finite_median` callers in the rest of background processing assume NaN-preservation invariants and there is no explicit `arr[r, ~fin] = np.nan` step. The subtraction also rewrites the row in-place on the `arr.astype(np.float64, copy=True)` copy, so when `polyfit` is rank-deficient (e.g. all finite values at the same x), numpy's silent fallback can return a degenerate slope and dump it into the row.
**Why it matters**: Pathological row (e.g. only the centre column is finite) gets a slope from polyfit on 1 point + 1 NaN. `polyfit` raises only on ≤degree points; with `fin.sum() == 2` and both x-values nearly equal, the slope is near-singular and the row gets multiplied by a huge value.
**Suggested fix**: Add an explicit `arr[r, ~fin] = np.nan` after subtraction; warn or skip rows where `np.std(xs[fin])` is below a small threshold.

### 8. [S1] `_eliminate_profile_jumps` is order-dependent and irreversible
**Location**: `probeflow/processing/background.py:152-172`
**Problem**: The jump-removal scan starts at `finite_idx[0]` and adjusts forward. A single jump near the start of the profile shifts the offset for *all subsequent rows* permanently; the median-preserve step later cannot undo this since each row was already moved. With a `jump_threshold` value comparable to the true row-to-row variation, transient noise spikes early in the scan get treated as steps and the entire downstream scan is offset.
**Why it matters**: User-facing parameter `jump_threshold` has no documented sensitivity guidance; a threshold slightly too low produces a smoothly-wrong image that looks plausible but with all heights shifted by an arbitrary amount near the start of the scan.
**Suggested fix**: Either (a) apply offsets symmetrically (scan forward and backward, take the median), or (b) cap the cumulative offset and warn when it exceeds N·threshold.

### 9. [S1] `gmm_autoclip` initialisation can produce NaN means
**Location**: `probeflow/processing/analysis.py:46-49`
**Problem**: `mu1 = data[data <= med].mean()`, `mu2 = data[data > med].mean()`. When the data is extremely peaked at one value (common for binary masks, flat references, or after thresholding), `data > med` can be empty → `mu2 = NaN` → all subsequent EM iterations propagate NaN through `_gauss`, then the `n1 < 1e-6 or n2 < 1e-6` check fires only *after* `r1/r2` are already NaN. The function then returns NaN-derived percentiles or, more often, the clamped `(0, 100)` fallback, but the histogram count `np.sum(full_data < low_val)` with `low_val = NaN` is zero — so `clip_low = 0.0` always, which may not be what the user intended.
**Why it matters**: GUI auto-clip silently returns the default percentiles on degenerate data; no warning is shown.
**Suggested fix**: Detect `(data > med).sum() == 0` and fall back to `(1.0, 99.0)` immediately. Add explicit `NaN` checks before each M-step update.

### 10. [S2] Sub-pixel parabolic refinement guarded only against zero denominator
**Location**: `probeflow/analysis/line_periodicity.py:229-234`
**Problem**: `denom = 2 * y1 - y0 - y2`; the guard `if denom > 0` prevents division by zero but allows tiny positive denominators that produce |delta| > 0.5 (the parabola's max is outside `[i-1, i+1]`). This corresponds to the central sample not actually being a peak.
**Why it matters**: Reported `period_m = lag_m[first_idx] + delta * sample_spacing` can be off by more than one sample; documented uncertainty (FWHM/2 · spacing) under-reports the true error. Compounded with the FFT method's frequency-bin uncertainty `(period_m**2)*freq_res` which already underestimates when the dominant bin shares power with neighbours.
**Suggested fix**: Clamp `delta` to ±0.5 and only apply the correction when `abs(delta) <= 0.5`.

### 11. [S2] Bilinear interp + `mode="reflect"` near border in `line_profile`
**Location**: `probeflow/processing/geometry.py:107-132`
**Problem**: When `width_px > 1` the perpendicular swath is sampled with `map_coordinates(..., mode="reflect")` and **mean-averaged across `n_perp` offsets**. Offsets that fall outside the array are reflected back into the interior, which biases the perpendicular average towards interior values. Worse: `n_perp = int(round(width_px))` can equal 1 when `width_px ∈ [1, 1.5)` while the else branch was taken (`width_px > 1.0`), making `accum / n_perp == accum / 1` — silently identical to the width=1 case but with extra cost.
**Why it matters**: User-perceivable: setting width=1.0 vs 1.4 gives identical output; setting width=1.5 vs 2.0 may produce different smoothing than expected. For line ROIs near image borders the reflected swath compresses against the edge.
**Suggested fix**: Use `mode='constant', cval=NaN` and ignore non-finite offsets in the perpendicular mean; document the minimum `width_px` step that produces a visible change.

### 12. [S2] `_robust_scale` reverts to `eps` when MAD is zero, then threshold is meaningless
**Location**: `probeflow/processing/bad_lines.py:64-76`
**Problem**: When all values are equal (or all zero), `mad == 0`. The fallback `np.max(abs(finite)) * 1e-3` works for non-zero data; for an all-zero image the final fallback is `np.finfo(np.float64).eps`. Then `cutoff = threshold * eps ≈ 1e-14`, and *every* finite difference > 1e-14 is marked as a bad-segment edge.
**Why it matters**: Running bad-line detection on a flat synthetic test image, or on the difference of two identical scans (a common QC check), spuriously detects bad segments everywhere.
**Suggested fix**: When MAD == 0 *and* the max absolute residual is < some tolerance, return `inf` so no segments get flagged.

### 13. [S2] `fft_soft_border` division by tapered window amplifies edge noise
**Location**: `probeflow/processing/filters.py:316-318`
**Problem**: After the inverse FFT, `out = out / safe_win + mean_val` divides by the *original* Tukey window. For pixels exactly at the boundary, `win2d` is 0; `safe_win = np.where(win2d > 1e-6, win2d, 1.0)` substitutes 1.0 — but for pixels just inside the boundary where `win2d ≈ 0.01`, the division *multiplies* any FFT ringing by 100. The "soft border" intent is to suppress wrap-around, but the post-division reintroduces it amplified.
**Why it matters**: A periodic signal near an edge gets a bright halo after high-pass via this function. Visible on small images (≲ 128 px) where `border_frac=0.12` puts more than a few rows in the steep ramp.
**Suggested fix**: Clip the division so it cannot exceed a maximum gain (e.g. 5×); document that interior pixels are recovered but the inner ≲ `border_frac/4` of the image is not trustworthy after this filter.

### 14. [S2] Empty-array `apply_outlier_mask` silently returns empty arrays
**Location**: `probeflow/spectroscopy/outliers.py:22-24`
**Problem**: `keep = np.isfinite(x_arr) & np.isfinite(y_arr)`. If *all* values are non-finite, `keep` is all-False, returned arrays are empty, and the keep-mask is all-False. Downstream `make_displayed_spectrum` puts `int(y_display.size) = 0` in metadata, and any subsequent normalization division-by-mean steps explode (`np.nanmax` on empty raises a RuntimeWarning and returns NaN).
**Why it matters**: GUI shows an empty trace silently; no diagnostic message that "all samples were excluded".
**Suggested fix**: Raise `ValueError("no finite samples after outlier mask")` when `keep.sum() == 0`.

### 15. [S2] `subtract_background` step-tolerance: zero candidates when threshold too tight
**Location**: `probeflow/processing/background.py:591-602`
**Problem**: The step-tolerance branch only *replaces* `fit_mask_acc` when `candidate.sum() >= n_terms`; otherwise it silently keeps the wide mask, which is correct. But the gradient is computed on `np.where(np.isfinite(arr), arr, _finite_median(arr))`, which substitutes the *global* median into NaN regions. This creates artificial sharp gradients at the NaN boundary that exceed the threshold, falsely excluding the boundary pixels from the fit even when they are physically valid neighbours.
**Why it matters**: Background fit on a scan with masked/NaN strips produces a tilted plane biased by the inability to use boundary pixels; the fit residuals show an asymmetric darkening near the NaN region.
**Suggested fix**: Estimate gradient with `np.nan_to_num` only at points where the central-difference neighbours are both finite; treat boundary gradients as missing.

### 16. [S2] `feature_lattice.compare_features_to_lattice` round() rounds ties to even
**Location**: `probeflow/analysis/feature_lattice.py:102-103`
**Problem**: `i = int(round(u))`, `j = int(round(v))`. Python 3 `round()` uses banker's rounding (half to even). For features lying *exactly* on a half-cell boundary the assignment depends on the parity of the integer part, not on physical position. With sub-pixel feature coordinates this is rare, but for synthetic/exact lattice data (e.g. test fixtures) it makes adjacent-feature matching depend on which side of zero the index falls on.
**Why it matters**: Test reproducibility and edge-of-image features can flip between sites; `n_duplicate_sites` can increment from a tie that should be deterministic.
**Suggested fix**: Use `int(math.floor(u + 0.5))` (always-up for half) and document the convention.

### 17. [S3] `_polar_decompose` ignores tiny negative singular values from numerical error
**Location**: `probeflow/analysis/lattice_distortion.py:193-200`
**Problem**: When `T` is near-rank-deficient (one measured vector very small), the smallest singular value can be returned as a tiny negative number by LAPACK (≈ -1e-16). This bypasses the `det(R) < 0` branch (since R is then nearly singular and `det(R) ≈ 0`), and the `S = Vt.T @ diag(sigma) @ Vt` carries a tiny negative element. The downstream `affine_lattice_correction` (which uses `np.linalg.cond(matrix) > 1e10`) catches this only sometimes.
**Why it matters**: Edge case; only triggers when the lattice basis is borderline degenerate (e.g. one extracted vector close to zero).
**Suggested fix**: `sigma = np.clip(sigma, 0.0, None)` after the SVD.

### 18. [S3] `np.histogram` rounding in `current_histogram` empty-array case
**Location**: `probeflow/processing/spectroscopy.py:228-230`
**Problem**: If `data` is all non-finite, `finite` is empty and `np.histogram(empty, bins=bins)` returns `counts = zeros(bins)` and `edges = linspace(0, 1, bins+1)` silently. Callers cannot distinguish "no data" from "all-zero data".
**Why it matters**: Telegraph-noise analysis on an all-NaN trace silently shows an empty histogram with the [0,1] axis range.
**Suggested fix**: Raise `ValueError` (or return a sentinel) when `finite.size == 0`.

---

## Items consciously *not* flagged

- `_nan_normalized_gaussian` uses `weights > 1e-12` which is correct.
- `affine_lattice_correction` checks `np.linalg.cond > 1e10` — good.
- `set_zero_plane` checks `np.linalg.matrix_rank(A) < 3` — good.
- `fit_axis_aligned_ellipse` rejects non-positive `u, v` — good.
- `subtract_background` clamps `pixel_size` to `1e-30` — already documented.
- `tv_denoise` NaN handling looks correct apart from finding 2.
- `compute_pair_correlation` masks `annulus > 0` before division — good.
- `apply_normalization` rejects NaN/zero denominator channels — good.

## Cross-references
- Physics-correctness agent may have flagged finding 1 from a different angle (polar decomp semantics) and findings 3, 11 (calibrated bit-depth, perpendicular profile averaging) as physics-of-measurement issues. Findings 4, 6, 8 also have a physics-of-provenance angle.
- Image-processing review (if any) should overlap on findings 5, 13, 15.
