# ProbeFlow Review Status

**Updated**: 2026-06-04

This is the single, consolidated record of ProbeFlow's code-review history. The
detailed per-angle review files (`docs/reviews/2026-05-27-*.md`) and the earlier
staged-review stage files were pruned once their conclusions were captured here,
following the project convention: keep a living summary, prune the detail once
findings are enacted. The conclusions and the remaining open items survive below.

## Summary

Two review efforts have run on ProbeFlow:

1. **Staged review (2026-05, "Stage 1/2/3")** — scientific-workflow/physics pass,
   an architecture/maintainability pass with bounded refactor slices, and a
   compatibility/stability/release-safety pass. All concrete findings were enacted.
2. **Deep review (2026-05-27)** — six parallel angles (physics, numerical
   stability, image-processing pipeline, IO/instrument format, backend
   architecture, GUI architecture) producing **114 findings (S0=1, S1=36, S2=61,
   S3=16)**.

**Status: essentially complete.** Every S0 and S1 finding, and every physics /
numerical-correctness finding, has been enacted. What remains (listed below) is
lower-severity S2/S3 polish — mostly GUI-architecture tidiness, large-file
splits, and a handful of numerical/IO edge cases. None block users.

## What was enacted (highlights)

Scientific correctness:

- **Lattice correction** — `_polar_decompose` no longer flips singular values on
  reflective input (the S0); reflective transforms are now flagged, not silently
  applied. Calibrated physical lattice vectors and per-axis residuals throughout.
- **FFT / window cluster** — unified window vocabulary, unbiased masked-ROI DC
  subtraction, coherent-gain normalisation, periodic Hann semantics, isotropic
  cutoffs, plateau-safe Bragg-peak finding.
- **Calibration threading** — `apply_processing_state` now takes scan calibration;
  pixel sizes reach `subtract_background(step_tolerance)`/`facet_level`, and
  `scan_range_m` is updated by shape-changing geometric ops so scale bars, FFT
  k-axes, and pixel→nm conversions stay correct.
- **`quantize_bit_depth`** — explicit reproducible `vmin`/`vmax`/`quantum`, run
  after arithmetic ops, preserving the "discrete level" export guarantee.
- Periodicity/autocorrelation bias, pair-correlation edge correction, row-align
  ordering, and assorted estimator fixes.

IO robustness:

- SXM writer overwrite/source-equality guard; ROI sidecar path no longer collapses
  dotted Createc filenames; Createc channel-count trusts the header; Nanonis
  missing tip position → NaN (not origin); DAT→SXM `SCAN_DIR` orientation; SI unit
  normalisation; decode warnings surfaced on `Scan.warnings`; structured CSV and
  provenance-sidecar coverage for all exporters.

Architecture / maintainability:

- Canonical `measurements.models.MeasurementResult` with a single legacy adapter;
  unified `FeaturePoint`; shared kernels under `measurements/` (formatting,
  `roi_resolve`, `raster`); `ProcessingStep`→`ProvenanceStep` (alias kept).
- `ScanGraph` and `plugins/` shelved as explicitly experimental (dropped from the
  public `provenance` namespace); boundary-rules docstrings rewritten to describe
  the real architecture (Scan → ProcessingState → ProcessingHistory + ExportRecord).
- GUI modeless-dialog lifecycle: `ImageViewerDialog` now owns a
  `_modeless_children` registry closed on `closeEvent`; worker-signals race fixed;
  central viewer command/shortcut registry (`gui/viewer/shortcuts.py`).

Release safety: Qt headless test-harness skip; toolbar assets verified in the
wheel; CLI surface verified; backend imports remain GUI-free.

## 2026-06-04 re-check of the open backlog

Every item that was still listed as open was re-verified against the current
code. The result: **the overwhelming majority were already resolved** by the
refactoring that landed after the review docs were written, or were intentional
design choices rather than defects. Details below.

### Already resolved / not a defect (verified 2026-06-04)

Physics / numerical:
- `_polar_decompose` is the reworked PSD form (S always PSD, R inherits sign of
  det, NaN + `UserWarning` on reflection). NumPy SVD returns non-negative
  singular values, so the "tiny negative singular value" concern cannot arise.
- `compute_correction`'s `tol = 1e-6·|a||b|` compared against `|det|` **is** a
  `sin(angle)` test, since `det = |a||b|·sinθ`. Scale-invariant by design.
- `line_profile` already uses `mode="constant", cval=NaN` (no `reflect` leakage)
  and a symmetric perpendicular `linspace` (no `width_px=1` asymmetry).
- `_robust_scale` already has a graded fallback (max-abs·1e-3, then `eps`), not a
  bare `eps`. `subtract_background` step-tolerance already falls back to the full
  fit mask when too few candidates pass (`candidate.sum() >= n_terms`).
- `_subtract_scanline_background` `preserve_level` intentionally restores a single
  global reference level; per-row "preservation" would undo the subtraction.
- `extract_lattice` `cluster_choice=1` selects the *most populated* cluster (a
  sensible, user-tunable default), not an arbitrary one.
- `current_histogram` handles empty input via NumPy's defined empty-array behaviour
  (no crash, no bad rounding).

Image processing:
- `_nan_normalized_gaussian` uses normalized convolution (`mode="constant"` + weight
  division) — no `reflect` boundary leakage.
- `threshold_image(mode="binarize")` returns float64 deliberately so non-finite
  pixels stay NaN (bool/uint8 cannot) and stay pipeline-compatible.

Backend / GUI architecture:
- `_legacy.py` no longer exists (removed).
- `FeatureCountingWindow` was extracted into the `gui/features/` package with a
  dedicated `controller.py` (the ~150 LOC dedup) and now takes/propagates a theme
  via `apply_theme` (no more `t={}`).
- `gui.processing` no longer uses a wildcard import (explicit imports now).
- A central viewer command/shortcut registry exists (`gui/viewer/shortcuts.py`).

### Fixed in this pass (2026-06-04)

- `feature_points_to_csv` now rejects non-finite / non-positive `pixel_size_*_nm`
  instead of silently emitting meaningless nm columns.
- **Large-file splits — done.** The three monoliths were broken up along their
  natural seams (behaviour preserved; methods/functions moved verbatim and resolve
  via the class MRO / module re-exports, verified by ruff `F821`, class assembly,
  and the runnable backend tests):
  - `fft_viewer.py` 3044 → 1769 LOC, with `fft_viewer_lattice_mixin.py`,
    `fft_viewer_mains_mixin.py`, `fft_viewer_reconstruct_mixin.py`.
  - `image_viewer.py` 2293 → 627 LOC, with `image_viewer_build_mixin.py`
    (the `_build` UI assembly) and `image_viewer_chrome_mixin.py` (menu bar,
    sidebar/tool hosting, modal overlays, window layout).
  - `filters.py` 1214 → 577 LOC, with the Bragg/reciprocal-lattice analysis moved
    to `bragg.py` and re-exported for backward compatibility.

### 2026-06-04 re-check #2 — the rest of the backlog was already enacted

A second deliberate re-verification (prompted by how many earlier "open" items
turned out already-done) confirmed that essentially the entire deferred list had
been fixed during the deep-review campaign without being logged against its
finding number. Verified resolved / intentional in the current code:

- **Layering**: `apply_processing_state` no longer hides disk I/O — it takes an
  `operand_resolver` injection point with the file-load isolated in a documented
  default (cites arch-backend #13). `Scan`'s lazy `processing.*` imports are the
  intended way to break the core←processing cycle.
- **IO (RHK SM4 + SXM)**: `_dtype_from_data_size` now uses the `data_size` header
  with a warned payload-length fallback (#7); `_parse_string_data` warns and stops
  on a truncated block instead of silently dropping (#12); `_parse_page_index_array`
  advances by `PAGE_INDEX_ARRAY_SIZE + object_count·OBJECT_SIZE`, not a fixed
  stride (#13); the SXM cushion cache was cleanly refactored into `_cushion_tail_lens`
  + `_data_offset_in_file` with accurately-named byte caches (#22).
- **Image processing**: `affine_lattice_correction` now uses weight-normalized
  NaN-aware OOB handling, matching the rotate path (#8); the threshold dialog's
  display range is percentile-based, not raw extrema (#13).
- **GUI**: `ThresholdDialog` owns its own `HistogramPanel` and uses its public API
  (#6); the three thumbnail setters share `_apply_thumbnail_setting` (#14); saved
  processing is keyed by path **and** mtime (#10); `closeEvent` writes config via
  the centralized `_save_viewer_desktop_layout_into` helper (#16); survey
  availability is probed before `_build_ui` (#17); `desktop_layout` restore
  falls back gracefully when splitter counts differ (#18);
  `FeatureCountingWindow` now has `test_feature_counting_controller.py` +
  `test_gui_features.py` (#21).

### Genuinely remaining (architectural opinions — not bugs)

What is left is purely structural preference: no defect, no functional benefit,
and a real import-churn/regression cost to change. Deliberately not pursued.

- Spectroscopy logic spread across three packages; `processing/analysis.py` holds
  analysis-style measurements despite living in `processing/`; many-argument
  measurement constructors could take a context object.
- `_collect_point_sources_m` reads the measurement controller's (non-underscore)
  `feature_points` / `feature_metadata` via `getattr` defaults — minor coupling,
  acceptable as-is.

## Deferred (not in code-review scope)

- A true Python 3.11/3.12 test matrix (only the local interpreter was available).
- GUI smoke tests on a working Qt platform (local Qt cannot init `QApplication`;
  GUI tests skip under the preflight guard).

## Current user-facing / reference docs (kept)

- `docs/cli.md`
- `docs/createc_dat_reader.md`
- `docs/roi_manual_test_checklist.md`
- `docs/notes/roi-display-notes.md`
