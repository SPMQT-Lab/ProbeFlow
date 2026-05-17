Perform a physics and mathematical correctness review of ProbeFlow code. $ARGUMENTS

ProbeFlow processes STM/AFM data (Nanonis SXM, Createc DAT, RHK SM4). Check the specified files or, if none given, recently changed files (`git diff HEAD~1 --name-only`) against these invariants:

## 1. FFT Axis Convention

- `np.fft.fftfreq(N, d=pixel_size_m)` with metres produces cycles/m; multiply by 1e-9 to get nm⁻¹
- After `fftshift`, DC must be at index N//2; verify `qx[N//2] ≈ 0`
- Missing `d=` argument defaults to d=1.0 (cycles/pixel, not physical) — flag this
- `fft_magnitude` converts pixel_size_x_m → nm internally; any new FFT code should follow the same convention
- Flag: `fftfreq` called without the `d` keyword on physical data

## 2. Spectroscopy Pipeline Order

The physically correct order for `make_displayed_spectrum` is:
**raw copy → smoothing → derivative → normalization → outlier mask → offset**

- Smoothing BEFORE derivative (noise amplification if reversed)
- Normalization AFTER derivative (normalizing before derivative changes the meaning of dI/dV)
- Flag: any code path that reorders these steps, especially derivative after normalization

## 3. Numerical Derivative

- `np.gradient(y, x)` with explicit x is correct for non-uniform spacing
- `np.gradient(y)` without x assumes unit spacing — flag when x has physical units (V, s, etc.)
- `np.diff(y)/np.diff(x)` loses an endpoint — flag without explicit edge handling
- x must be strictly monotonic; `numeric_derivative` checks this — any new derivative code should too
- Flag: derivative computed on a forward+backward sweep without splitting first

## 4. Normalization Safety

- Division by setpoint, constant, or channel must guard: check finite AND non-zero before dividing
- Pattern in the codebase: `if not np.isfinite(val) or val == 0.0: raise ValueError(...)`
- Flag: bare `y / value` on physical data without a prior finite+nonzero check

## 5. Coordinate System (scan geometry)

- `scan_range_m = (width_m, height_m)` where width ↔ x-axis ↔ columns ↔ `arr.shape[1]`
- Correct: `pixel_size_x_m = width_m / shape[1]`, `pixel_size_y_m = height_m / shape[0]`
- Flag: width used as `shape[0]` denominator or height used as `shape[1]` denominator (rows/cols swapped)
- Flag: `scan_range_m[0]` (width) divided by `arr.shape[0]` (rows)

## 6. Rectangular Pixel Handling in Lattice Analysis

- For non-square scans, physical lattice vectors require per-axis pixel sizes:
  `a_physical = (a_px_x * pixel_size_x_m, a_px_y * pixel_size_y_m)` — NOT `a_px * scalar`
- `LatticeResult.a_vector_m` and `b_vector_m` must reflect this
- Flag: lattice pixel→physical conversion using a single scalar `pixel_size_m` when the scan is non-square

## 7. Unit Label Propagation

- After numerical derivative on I(V): `y_unit` must update from A → A/V (or nA/V, pA/V)
- After normalization: `y_unit` should become "relative" or "a.u."
- After FFT: axis unit must state "nm⁻¹" or "cycles/nm", not retain "m" or "pixels"
- Flag: unit string not updated after an operation that changes physical units

---

For each issue found:

```
[CRITICAL|WARN|INFO] path/to/file.py:line
What: one-line description
Why: the physical or mathematical reason this is wrong
Fix: specific correction
```

- **CRITICAL** — will produce wrong physical values (wrong axis scale, wrong pipeline order, wrong pixel size axis)
- **WARN** — likely wrong depending on context, or missing guard that could silently corrupt data
- **INFO** — convention deviation that won't affect correctness but breaks consistency with the rest of the codebase

If no issues are found, state "No physics issues found" and list which files and invariants were checked.
