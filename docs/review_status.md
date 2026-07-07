# ProbeFlow Review Status

**Updated**: 2026-07-07 (GUI general-fixes round added)

This is the single, consolidated record of ProbeFlow's code-review history. The
detailed per-angle review files (`docs/reviews/2026-05-27-*.md`) and the earlier
staged-review stage files were pruned once their conclusions were captured here,
following the project convention: keep a living summary, prune the detail once
findings are enacted. The conclusions and the remaining open items survive below.

## Summary

Three review efforts have run on ProbeFlow:

1. **Staged review (2026-05, "Stage 1/2/3")** — scientific-workflow/physics pass,
   an architecture/maintainability pass with bounded refactor slices, and a
   compatibility/stability/release-safety pass. All concrete findings were enacted.
2. **Deep review (2026-05-27)** — six parallel angles (physics, numerical
   stability, image-processing pipeline, IO/instrument format, backend
   architecture, GUI architecture) producing **114 findings (S0=1, S1=36, S2=61,
   S3=16)**.
3. **Adversarial review campaign (2026-06-09 → 06-12, PRs #20–#36)** — targeted
   seam-by-seam passes over the areas the earlier reviews could not exercise
   (GUI timing, async loading, replay fidelity, parsers under corruption). See
   the dedicated section below.

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

## Adversarial review campaign (2026-06-09 → 06-12, PRs #20–#36)

A third effort ran as short adversarial passes, each shipping its fixes with
regression tests before moving on. One PR per pass; the PR descriptions hold
the per-finding detail.

- **GUI robustness (#20)** — QPixmap built off the GUI thread, worker-signal
  lifetime (`SIGSEGV` class), thread-pool drain at exit.
- **Browse loading at scale (#21)** — network-drive freezes: async preview /
  index workers, metadata peek budget, sliced card building, thumbnail
  priority.
- **Seam review, two passes (#22–#26)** — scoped-filter replay ordering,
  browse async/navigation seams, sidecar discovery fallback, corrupt-sidecar
  visibility, quick-selection lifecycle, scale/shear overlay transforms, and
  the PySide wrapper-recycling test flake (root-caused and fixed).
- **Physics reviews (#27, #29)** — spectroscopy: the qPlus setpoint gate
  (Δf misread as amps), time-axis sanitisation, derivative units; FFT:
  window-envelope compensation and odd-size soft-border centring.
- **User-feedback batch (#28, #30)** — mains custom streak pairs, notch-width
  visualisation, background notch fill, streak/overlay decoupling.
- **Workflow-replay harness (#31)** — a permanent integration harness
  asserting display == export == provenance-replay for representative
  pipelines (plus a set-zero frame fix it caught).
- **Parser adversarial review (#32–#34)** — mutated-fixture corpus tests for
  every reader; strict VERT metadata summaries (fast path now agrees with the
  full parse), partial-load warnings surfaced in the viewer, Nanonis
  single-column fix, and recognition of the normal Createc trailing appendix
  (no more spurious warnings on healthy files).
- **Docs (#35–#36)** — GUI guide with offscreen-generated screenshots.

Suite grew from ~2,260 to 2,449 tests across the campaign; main is green.

## Scheduled deep review, 2026-07-03 (new-code pass: .dat→.npy converter, feature bank, CLIP classify, contrast fix)

Targets: everything merged since the adversarial campaign that earlier passes
had not covered — the Createc `.dat`→`.npy` converter (+ the #46 scan-range
fix), the feature bank Phases 1–2, the CLIP classify encoder, and the
viewer-black contrast fix (8a2950d). Particle Statistics (#45) was itself a
review and was not re-reviewed.

### Fixed in this pass (committed and merged to main in PR #48, suite green: 2508 passed / 3 skipped)

1. **Feature-bank silent data loss (the one real find)** —
   `analysis/feature_bank.py`: a corrupt/unparseable bank file made
   `load_bank` silently return an empty bank, so the next "Add to bank"
   overwrote the user's entire cross-scan labelled-sample collection with just
   the new entries. `append_entries` now loads strict (`ValueError`, surfaced
   on the GUI status bar, original file untouched) and writes atomically
   (tmp + `os.replace`), closing the truncated-write corruption source too.
   Read path (classification) stays permissive. New `tests/test_feature_bank.py`
   (the module previously had **zero** tests).
2. **Createc complete-scan trim hazard** — `io/readers/createc_dat.py`
   `_trim_createc_stack`: when `ImageYPosMax` confirmed the scan completed
   (`= Num.Y + 1`), the code still fell through to the legacy channel-0
   nonzero heuristic, which silently drops genuine trailing rows whose DAC
   values are zero (and drops the last row if the bottom-right pixel is 0/NaN).
   Header-confirmed-complete now returns untrimmed. Pinned in
   `tests/test_createc_dat_decode.py`.
3. **Converter `main()` partial-kwarg crash** —
   `io/converters/createc_dat_to_npy.py`: programmatic calls that passed only
   some keywords (e.g. `basis=` without `src=`) skipped the CLI arg-parse
   branch and crashed on `Path(None)`. Defaults now resolved unconditionally.
4. **Mojibake log line** — `gui/workers.py` NumPy-conversion log printed
   `â”€â”€` (double-encoded box-drawing chars) in the convert dialog.

### Verified sound (no change needed)

- `decoded_scan_range_m` (#46): per-pixel step from the programmed frame ×
  decoded pixel count — physically correct for first-column removal and
  partial-scan trims; metadata path consistent (`shape` (Ny, Nx) + decoded range).
- CLIP classify + bank integration (`analysis/features.py`): bank-dimension
  guard, zero-norm cosine guard, bank-only degenerate sample stack, sharpness
  z-norm padding for bank entries, encoder gating in the GUI controller.
- Contrast revalidation (8a2950d): sound; known tradeoff — a deliberate manual
  window capturing <2% of finite data is reset by the next processing op.
- Previously-flagged gaps now confirmed fixed: shear ROI/mask invalidation
  (`_on_shear_applied`), quick-selection menu sync after transform-drop.

### Larger items — outlined fixes (not done; ordered by value)

1. **Feature-bank scale blindness** (tracked TODO in `feature_bank.py`): crops
   are fixed 48 px, so a banked embedding's physical field of view depends on
   the source scan resolution and cross-scan matches are scale-blind.
   *Plan:* bump `BANK_SCHEMA_VERSION` → 2; `make_entry` gains
   `pixel_size_m` (or crop physical size in nm) threaded from the features
   panel's scan; `bank_to_samples` returns it; `classify_particles` warns when
   bank-entry crop size differs from the current crop's physical size by more
   than ~2× (later: resample crops to a canonical nm/px before embedding).
   Old v1 entries: treat as unknown scale, warn once.
2. **Background-dominated CLIP embedding** (carried from UniMR review):
   similarities crush to ~0.998–1.0 because the 48 px crop is mostly flat
   background, so outlier rejection is unreliable in general. *Plan:* mask or
   weight the crop by the particle contour before embedding (contours already
   exist on `Particle`); re-verify `_threshold_similarities` guards afterwards
   — the spread guard may become unnecessary.
3. **Bundle-provenance pixel-key convention**: in the `.npy` bundle sidecars,
   `original_shape`/`decoded_shape` are `(Nx, Ny)` while each plane's `shape`
   is `(rows, cols)` — same word, opposite conventions in one JSON (e.g.
   A250407: `decoded_shape: [255, 9]` vs plane `shape: [9, 255]`). *Plan:*
   rename the bundle keys to `original_pixels`/`decoded_pixels` (matching
   `scan_pixels`) while the format is young; regenerate the committed sample
   bundles under `test_data/output_raw_npy/`; grep for external consumers first.
4. **Bank rotation-augmentation asymmetry**: in-image samples get 36 rotated
   copies under `rotate_augment`; bank vectors are single unrotated embeddings,
   so banked classes are disadvantaged for rotated instances. *Plan:* either
   bank all 36 rotated embeddings per sample (36× file size) or store the raw
   crop in the entry and re-embed with rotations at classify time (slower,
   smaller file). Decide with real usage data.

### Next-run candidates (unexplored, carried forward)

- `processed_export.py` / provenance replay **without** canonical sidecars, and
  a CLI replay smoke test of new-format sidecars.
- `image_canvas.py` selection internals (~1500 lines, only ever grepped).
- ~~Spectroscopy display wiring (`gui/spec_viewer/`), `analysis/spec_plot.py`~~ —
  done 2026-07-06, see the dedicated section below.
- Partial `.sxm`/`.dat` thumbnail decode in browse (open since the browse
  loading review).
- Feature-bank GUI flows under Qt (add-to-bank dialog, `refresh_bank_status`)
  have no GUI-level tests.

Environment note for future runs: the full suite needs
`QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PySide6,pathlib; print(pathlib.Path(PySide6.__file__).parent/'Qt'/'plugins'/'platforms')")`
alongside `QT_QPA_PLATFORM=offscreen` in sandboxed shells, or `qapp` creation
aborts (SIGABRT) at `test_feature_finder.py`; `test_fft_viewer_utils.py` and
`test_lattice_grid.py` still segfault under offscreen Qt on anaconda py3.13
(environmental, pre-existing).

## GUI clarity/layout review, 2026-07-06 (fixes applied, uncommitted)

A structural review of the GUI code and layouts (clarity + logical
organisation, not bug-hunting). All four finding groups were fixed in the
same session; full suite green (2716 passed / 3 skipped, with adstat + cv2
installed).

1. **Feature Counting misroute (real bug)** — the Browse card context action
   "Send to Feature Counting" loaded the scan into a hidden duplicate
   Features workspace wired into the main window's content/sidebar stacks,
   so the floating `FeatureCountingWindow` opened empty. The action now loads
   through the floating window's own off-thread path (saved viewer processing
   applied), and the dead duplicate workspace (`_features_panel`,
   `_features_sidebar`, `_features_ctrl`, its pools and handlers) was removed;
   content/sidebar stacks renumbered to 0 browse / 1 convert / 2 tv / 3 dev /
   4 survey. Regression test:
   `test_gui_features.py::test_card_context_features_loads_floating_window`.
   The `test_gui_features.py` workspace tests now exercise
   `FeatureCountingWindow` (the real workspace) instead of `ProbeFlowWindow`'s
   dead copy.
2. **Main-window menu regroup + navbar removal** — all workspace pages live in
   one **Workspace** menu (Browse Ctrl+1, Convert Ctrl+2, TV denoise Ctrl+4,
   Survey, Developer tools Ctrl+5); Feature Counting is a plain action under
   **Tools** (Ctrl+3) since it opens a floating window, not a page. "Align
   rows" moved from the one-item top-level "Processing" menu to View →
   "Thumbnail row alignment" (it is a thumbnail display setting); the
   redundant Convert menu and the permanently-disabled placeholders (Open
   recent, Export image..., Preferences...) were dropped. The vestigial
   `Navbar` (logo strip with never-emitted signals and styling for buttons
   that no longer existed) was deleted outright — `gui/navbar.py`, its compat
   re-exports, and the `NAVBAR_*`/`#navBtn` styling. Docs screenshots
   regenerated.
3. **Large-file splits (continuing the established convention)** —
   `gui/features/__init__.py` (2,520 LOC of implementation in a package
   `__init__`) became `features/panel.py` with a thin re-exporting
   `__init__`; `dialogs/particle_statistics.py` (5,845 LOC) was split
   verbatim into `particle_statistics_content.py` (labels, statistic
   metadata/guides, tutorial lessons — pure data, no Qt),
   `particle_field_view.py` (point-field renderer + drawing helpers), and
   `particle_statistics_workers.py` (pooled workers), leaving the dialog at
   ~4,080 LOC with backward-compatible re-imports of the moved private names.
4. **Tooltip system unification + label fixes** — `_tooltips.tip()` no longer
   hard-wraps at 50 columns (the app-wide rich-text wrapper in
   `gui/tooltips.py` owns wrapping; the double system gave those tooltips a
   ragged narrower column); the three tests that pinned the ≤52-column
   pre-wrap now assert tooltips exist. FFT viewer: "⚡ Mains" tab renamed
   "Mains" (only emoji tab label); Grid tab group retitled "Known structure
   & grid overlay" since the grid-extent spinner lives in it. A "where the
   lattice code lives" module map was added to the `gui/__init__` docstring.

5. **Truncation follow-ups from user screenshots** — the viewer sidebar's
   hardcoded minimum width (340/360 px) elided every tab label at
   Medium/Large GUI fonts ("Vi…/Proc…/Meas…"); the minimum is now derived
   from the tab bar's size hint so all five labels fit un-elided at any
   font (measured need: 323/390/439 px at Small/Medium/Large). The Particle
   Statistics bottom pane could be splitter-crushed until the Setup page
   scrolled, cutting the three column headers mid-glyph; the workflow tab
   widget now has a minimum height equal to the Setup page's natural height
   (capped at 460 px), while the Results tab still scrolls freely.

Known-good follow-ups not taken: `particle_statistics.py` is still ~4,080
LOC (dialog + tutorial flow); the viewer's `_build()` (~980-line method) and
`_build_viewer_menu_bar` (~475 lines) would benefit from a
method-per-tab/menu pass; `_build_fft_column` still builds three tabs inline
while the other three have `_build_*_tab()` methods.

## Spectroscopy display wiring review, 2026-07-06 (fixes applied, uncommitted)

The twice-carried-forward "spectroscopy display wiring" target from the
adversarial campaign: `gui/spec_viewer/` (single + overlay dialogs, shared
helpers), `analysis/spec_plot.py`, `gui/viewer/spec_overlay.py` (marker
controller), `gui/dialogs/spec_mapping.py`. The decoders (`createc_vert.py`,
`nanonis_spec.py`) were already covered by the parser adversarial pass (#32)
and `io/spectroscopy.py` by the physics pass (#27); they were not re-reviewed.

### Fixed in this pass (all with regression tests)

1. **Per-trace display-unit scaling on shared axes (the big one)** — both
   spectroscopy dialogs chose the SI display prefix per trace from that
   trace's *own* values (`choose_display_unit`), then plotted the rescaled
   traces on one axis. A 50 pA and a 2 nA spectrum overlaid as "50 vs 2" —
   relative magnitudes silently distorted up to 1000×, with only the y-label
   quietly degrading to "value". Fixed in `SpecOverlayDialog` (same channel
   across files) and `SpecViewerDialog` Overlay/Waterfall modes (channels
   sharing a base unit): traces co-plotted on one axis now share one scale
   per unit group (`_shared_scale_values` in `spec_viewer/shared.py`), and
   the CSV/JSON/TXT export paths reuse the same rows so display == export.
   Separate-mode per-axis auto units are unchanged (pinned by test).
   Tests: `tests/test_spec_viewer_display.py` (GUI module).
2. **SCAN_ANGLE unit inconsistency (three-way)** — Nanonis stores SCAN_ANGLE
   in degrees. The Createc→SXM converter wrote **radians**
   (`createc_dat_to_sxm.py`), `spec_plot._parse_sxm_angle_rad` read the value
   as radians (wrong for real Nanonis files), while the marker controller and
   mapping dialog read degrees (wrong for ProbeFlow-converted files). All
   sides now agree on degrees in the header; `_parse_sxm_angle_rad` converts
   deg→rad on read. Angled-scan spec markers were off by 180/π before.
3. **Spec markers fabricated at image centre** — `SpecOverlayController.load`
   placed a confident, ordinary-looking marker at (0.5, 0.5) whenever the
   spectrum's recorded position fell outside the scan frame. Out-of-frame
   specs are now skipped, collected in `controller.out_of_frame`, and
   reported on the viewer status bar ("N mapped spectrum position(s) outside
   this scan frame").
4. **Createc `.dat` scans never applied their frame offset** — the marker
   controller and the mapping dialog's "Suggest all" assumed origin-centred
   frames for non-sxm scans, but Createc `.VERT` positions and `.dat`
   OffsetX/OffsetY share one DAC coordinate system. Both now derive the frame
   centre via `_position_from_createc_header` (and parse the `Rotation`
   header). Before, every Createc marker was mislocated or (with the old
   centre fallback) pinned to the middle of the image.
   Tests: `tests/test_spec_marker_positions.py` (Qt-free).

Also removed two dead helpers in `spec_viewer/single.py`
(`_display_values_for_channel`, `_selected_channels_in_display_units`).

### Verified sound (no change)

- `make_displayed_spectrum` pipeline order + derivative unit propagation
  (matches the #27 conventions); vertical offset applied last in displayed
  units, consistently in plot and export.
- Measurement machinery reuse between the two dialogs (unbound-method
  delegation is odd but correct); measurement state reset on every redraw;
  second measurement point correctly pinned to the first point's trace.
- `spec_position_to_pixel` frame math and y-inversion (readers normalise
  planes to top-left origin); mapping dialogs' one-to-one map semantics.
- Overlay x-compatibility gate (label/unit/length/allclose) before mixing
  files on one axis.

### Noted, unchanged

- Checkbox labels in the single viewer show per-channel auto units even in
  Overlay mode where the axis shares a scale (cosmetic).
- `SpecMappingDialog._on_suggest` loads full scan data per image just for
  header/shape (slow on big folders; acknowledged in a comment there).
- Createc `Rotation` sign convention vs the sxm SCAN_ANGLE rotation sense is
  applied symmetrically but not verified against an angled fixture — no
  angled scans in `test_data`.

## Provenance replay + canvas selection review, 2026-07-06 (same session, uncommitted)

Second half of the session: the remaining carried-forward targets —
``processed_export.py`` / replay-without-canonical-sidecars, and the
``image_canvas.py`` selection internals (full read; previously "only ever
grepped").

### Fixed (with regression tests)

1. **New-format provenance sidecars were not replayable (the one real find
   in the export chain)** — the ``.probeflow.json`` ExportRecord is the
   declared "reliable current record", but its ``processing_history.steps``
   use ``operation_id``/``parameters`` keys and
   ``processing_state_from_history`` silently returned an **empty** state
   for them — a replay would reproduce the raw image while claiming to be
   processed. The helper now accepts both formats and skips provenance
   bookkeeping steps (``file_load``, ``export_*``, ``dat_to_sxm``).
   ``tests/test_provenance_replay.py`` pins the reconstruction and a full
   end-to-end round trip: export with an ROI-scoped step → recover the state
   AND the ROI set from the exported ``.probeflow.json`` alone (no
   ``.rois.json``, legacy ``.provenance.json`` deleted) → replay equals the
   exported array bit-for-bit.
2. **Hidden-overlay handle hijack** — resize handles are data-model
   geometry, not scene items, so with ``set_rois_visible(False)`` a pan
   click near an *invisible* active-ROI handle still started a resize and
   committed a geometry change (plus sidecar write) the user could not see.
   Handle hit test and hover cursor now honour overlay visibility.
3. **Marker clicks stole drawing clicks** — the spec-marker click intercept
   ran before the drawing-tool dispatch, so starting a rectangle near a
   visible marker opened the spectrum dialog instead. Markers now only
   intercept clicks in pan mode.
4. **Fit-to-view could not fit large images** — ``_compute_fit_zoom``
   clamped to the manual-zoom floor (0.25), so images more than 4× the
   viewport never fit. Fit mode now clamps at 0.01.
   Tests: ``tests/test_canvas_selection_interactions.py`` (all three
   verified discriminating via stash).

### Verified sound (no change)

- ``processed_export.py`` itself: SXM multi-plane block, PNG-vs-sidecar
  routing, collision checks, provenance threading.
- ROI recovery from provenance exports (`load_roi_set_sidecar(sidecar=…)`
  with an ExportRecord payload) — worked as designed, now pinned end-to-end.
- Canvas selection state machine: tool transitions, Escape ordering
  (cancel drawing → clear selection), drag-move reset-to-original +
  model-rebuild pattern, polygon double-click duplicate-vertex pop,
  quick-selection marquee lifecycle, wrapper-lifetime destroyed hook
  (already pinned by test_canvas_item_lifetime.py).

### User-reported follow-up (same day): quick-selection editing restored

Peter reported the yellow quick-selection marquee could no longer be resized
("I used to be able to grab the edges or sides") and that its outline was
too thick. Root causes: when the area tools switched from creating ROIs to
ephemeral selections (2f4d0d3), the marquee never received the resize
handles managed ROIs have; and the marquee/ROI pens were non-cosmetic
scene-unit pens, so outlines fattened with zoom (only the lattice-grid
overlay had received the cosmetic-pen treatment). Fixed in image_canvas.py:

- Rectangle/ellipse selections render gold resize handles (same
  ``resize_handles``/``resize_roi`` geometry as active ROIs, Shift keeps
  aspect); dragging the outline (±6 px band, interior still pans) moves any
  selection kind via ``core.roi.translate``; commits re-emit
  ``selection_drawn`` so the viewer sees edited geometry through the normal
  pipeline. Hover cursors + status hint added.
- Selection/preview pens and the ROI item pens are now cosmetic (constant
  ~1 px on screen at any zoom).
  Tests: TestQuickSelectionEditing + TestOutlinePensStayThin in
  ``tests/test_canvas_selection_interactions.py``.

### Noted, unchanged

- No CLI command replays a ``.probeflow.json`` end-to-end (the library path
  is now tested; a ``probeflow replay`` command would be a feature, not a
  fix).
- ``_movable_overlay_at`` is O(n²) in scene items (fine at current scales).
- Delete/copy keyboard shortcuts still act on the active ROI while overlays
  are hidden (explicit user action; left as-is).

## GUI general-fixes round, 2026-07-07 (Rohan's "General" list)

Implemented in this round (see git history for detail):

- **Scrollbar tracks** — the `add-page`/`sub-page` sub-controls were unstyled,
  so Qt painted its native textured fill in the groove either side of the
  handle. Now transparent in the global QSS (`styling.py`).
- **Menu hover** — `QMenuBar`/`QMenu` had no styling anywhere and fell through
  to the native white-on-black hover. A themed Menus section in the global QSS
  now covers all four menu bars (main window, image editor, particle
  statistics, FFT viewer).
- **Breadcrumb back/up buttons** hide when they cannot navigate instead of
  rendering as inert grey squares next to the folder title.
- **Workspaces are independent windows** — Convert, TV denoise, Dataset
  Builder, Survey and Developer tools now open as top-level
  `WorkspaceWindow`s (generalising `FeatureCountingWindow`); the main window
  hosts Browse only. Windows are lazily created, hidden (not destroyed) on
  close so state survives reopen, geometry persists per window in the config,
  and workspace shortcuts use `Qt.ApplicationShortcut` so they work from any
  window. The converter workspace was renamed **"STM File Converter"**.
- **Theme sweep (first pass)** — fixed dark-assuming literals that broke light
  themes: spec-viewer plot background/foreground now follow the theme
  (`spec_viewer/single.py` `_BG`/`_FG`), image-viewer rulers and line-profile
  measurement label take theme colours (`viewer/widgets.py`), and
  `particle_statistics.py`/`adstat_results.py` borders, tutorial text and the
  selected-statistic highlight route through the theme dict.
- **Tooltips** — Convert and Browse tool panels now have hover descriptions on
  every parameter input (house style: what it does, plus the effect of
  increasing/decreasing for numeric inputs).

### Reversible file conversions — investigated, deferred

Rohan requested `.sxm → .dat` and general format interchange. Findings from a
feasibility pass (2026-07-07):

- There is **no Createc `.dat` writer** (writers cover `.sxm`, `.gwy`, and
  rendered `.png`/`.pdf`/`.csv` only), and the `.dat` **reader is deliberately
  lossy** in exactly the dimensions a faithful writer needs: it drops the
  artifact first column and the trailing appendix, trims partial rows
  (mutating `Num.X`/`Num.Y`), reorders channels to the canonical public order,
  and converts DAC→SI (so a writer must re-quantise SI→DAC with loss).
- The existing `.sxm` writer is itself not a general Nanonis serialiser — it
  reproduces one reverse-engineered reference layout via the captured byte
  fixtures in `probeflow/data/file_cushions/`.
- Verdict: a *behaviourally* faithful `.dat` writer (re-readable, values
  correct, not byte-identical) is achievable but is a standalone project:
  cushion-style captured layout, regeneration of the dropped column/appendix,
  SI→DAC re-quantisation, plus a reader policy that retains currently-dropped
  raw bytes behind a "lossless" flag. Deferred until there is a concrete use
  case that `.sxm`/`.gwy` export does not cover.

### Remaining sweep backlog (follow-up sessions)

- **Theme**: most of the ~300 remaining hex literals turned out to be benign
  (theme-`get` fallbacks or data-encoding colours such as class overlays and
  plot palettes — checked: `definitions.py`, `fft_viewer.py`,
  `features/panel.py`). Real remaining offenders are the smaller per-widget
  sheets in `dataset_builder/view_tray.py`, `roi_items.py`,
  `lattice_grid/fft_overlay.py`, `image_canvas.py` and a long tail; convert
  opportunistically when touching those files. Status colours (green/red/amber
  chips that define both bg and fg) are self-contained and deliberately left
  theme-invariant.
- **Tooltips**: Convert + Browse set the template; remaining panels are the
  image-viewer sidebars, Feature Counting, TV denoise, and the Dataset Builder
  tuning tray (the latter is on Rohan's Dataset Builder list anyway).
- **Rohan's Dataset Builder list** — deferred to its own round; the pop-out
  window item is now free (Dataset Builder already opens as a window).

## Deferred (not in code-review scope)

- A true Python 3.11/3.12 test matrix (only the local interpreter was available).
- GUI smoke tests on a working Qt platform (local Qt cannot init `QApplication`;
  GUI tests skip under the preflight guard).

## Current user-facing / reference docs (kept)

- `docs/gui.md`
- `docs/cli.md`
- `docs/createc_dat_reader.md`
- `docs/roi_manual_test_checklist.md`
- `docs/notes/roi-display-notes.md`
- `docs/core_derisk_plan.md` (completed plan, kept as a design record)
