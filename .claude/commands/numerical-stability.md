Perform a numerical stability review of ProbeFlow code. $ARGUMENTS

Check the specified files or, if none given, recently changed files (`git diff HEAD~1 --name-only`) for these numerical hazards:

## 1. Catastrophic Cancellation

- Subtraction of two nearly-equal large values where the result is small: `a - b` when |a| ≈ |b| >> |a−b|
- Common in ProbeFlow: plane subtraction, background correction, line-by-line levelling on raw STM heights
- Flag: `data - reference` where reference is computed from `data` itself (e.g., polyfit background)
- Prefer: reformulate algebraically, or accumulate differences in a stable basis

## 2. Float Accumulation

- `sum(python_list)` accumulates error O(N) — flag on large arrays; prefer `np.sum(..., dtype=np.float64)`
- Running averages in `float32`: STM data is typically float64; check explicit `dtype=np.float32` casts in accumulators
- `np.mean(stack, axis=0)` is stable; manual `total += arr; total / N` is not — flag the manual form

## 3. Division Safety

- `x / y` without prior zero-check — flag on any physical denominator
- `np.log(x)` where x could be ≤ 0 — should use `np.log1p` or `np.clip(x, eps, None)` first
- `np.sqrt(x)` where x could be negative due to float noise — should clip to zero
- Pattern in codebase for safe division: check `finite and non-zero`, then divide — new code should follow this

## 4. Singular and Ill-conditioned Operations

- `np.linalg.inv(A)` — flag; prefer `np.linalg.solve(A, b)` or check `np.linalg.cond(A) < 1/np.finfo(float).eps`
- `np.linalg.lstsq` result: always check `rank < matrix.shape[1]` before trusting the solution
- `scipy.optimize.curve_fit`: check `np.any(np.isinf(pcov))` before using fit parameters — infinite covariance = failed fit
- FFT of all-NaN or near-zero masked array: result is technically valid but physically meaningless; verify `n_finite > 0` before transforming

## 5. Array Indexing Edge Cases

- DC bin after fftshift: `N//2` is correct for even N; for odd N the DC lands at `(N-1)//2` — flag hardcoded `N//2` on odd-size arrays
- `np.argmax` on an all-equal or all-NaN array: returns 0 (tie-breaks to first element) — verify this is the intended fallback
- `arr[mask]` where mask could be all-False: downstream `.mean()`, `.std()`, `.max()` return nan or raise; check `if mask.any()` before use
- `arr[np.nanargmax(arr)]` on all-NaN input: raises ValueError — flag without prior `np.any(np.isfinite(arr))` guard

## 6. Gradient and Smoothing Stability

- Savitzky-Golay with `window_length` near `len(data)`: edge values are severely distorted; `smooth_spectrum` handles this but downstream callers should not assume perfect edge quality
- `np.gradient` at endpoints uses first-order differences (less accurate than interior central differences) — flag use of endpoint values as authoritative without acknowledging this
- Very short arrays (< `polyorder+2` points): `smooth_spectrum` returns input unchanged (documented) — flag callers that don't handle the no-op case
- `np.gradient(y, x)` with non-uniform x: correct, but x values must be strictly monotonic — flag if monotonicity isn't checked beforehand

## 7. Masking and NaN Propagation

- `np.nanmean`, `np.nanmax` etc. silently ignore NaN — acceptable for display, but flag if the result feeds into a physical measurement where coverage matters
- Boolean mask inversion: `~np.isnan(arr)` vs `np.isfinite(arr)` — the latter also catches Inf; prefer `np.isfinite` on physical data
- `np.where(mask, val, arr)` preserving NaN structure: verify the mask correctly selects the intended subset

---

For each issue found:

```
[CRITICAL|WARN|INFO] path/to/file.py:line
What: one-line description
Why: the numerical reason this is problematic
Fix: specific correction or defensive pattern to apply
```

- **CRITICAL** — will silently produce wrong numbers for common inputs (catastrophic cancellation, division by zero, failed-fit parameters used unchecked)
- **WARN** — will produce wrong numbers for edge-case inputs that are plausible in STM data (very short spectra, all-NaN masked regions, non-square pixels)
- **INFO** — not wrong but inconsistent with the defensive patterns used elsewhere in the codebase

If no issues are found, state "No numerical stability issues found" and list which files and checks were applied.
