# ProbeFlow Review Status

**Updated**: 2026-07-11 (condensed after the refocus; see the note below)

The single, consolidated record of ProbeFlow's code-review history, following
the project convention: keep a living summary, prune the detail once findings
are enacted. Per-review detail lives in the git history and PR descriptions.

> **2026-07 condensation.** The 2026-07 refocus removed several subsystems
> (Particle Statistics/AdStat, Feature Counting, ML classification, Dataset
> Builder, Developer Terminal, ScanFlow integration — see
> `docs/refocus_plan.md`). Review findings and backlog items that applied only
> to those modules were pruned from this document along with the modules.

## Review history — all findings enacted

1. **Staged review (2026-05)** — physics/workflow, architecture, and
   release-safety passes. All concrete findings enacted.
2. **Deep review (2026-05-27)** — six parallel angles, 114 findings
   (S0=1, S1=36, S2=61, S3=16). Every S0/S1 and every physics/numerical
   finding enacted; re-verification passes (2026-06-04) confirmed the
   remaining backlog had been resolved en route or was intentional design.
   Highlights that still shape the code: PSD-safe `_polar_decompose` with
   reflection flagging; unified FFT window semantics and coherent-gain
   normalisation; scan calibration threaded through `apply_processing_state`
   and shape-changing ops; reproducible `quantize_bit_depth`; the
   `measurements/` shared-kernel layer; the central viewer command registry
   (`gui/viewer/shortcuts.py`); the large-file mixin splits (`fft_viewer`,
   `image_viewer`, `filters`).
3. **Adversarial campaign (2026-06-09 → 06-12, PRs #20–#36)** — seam-by-seam
   passes: GUI thread/lifetime robustness (#20), browse loading at scale
   (#21), replay/navigation/sidecar seams (#22–#26), spectroscopy + FFT
   physics (#27, #29), the permanent workflow-replay harness asserting
   display == export == provenance-replay (#31), parser adversarial corpus
   tests for every reader (#32–#34), GUI guide with generated screenshots
   (#35–#36).
4. **Scheduled pass (2026-07-03, PR #48)** — new-code review of the
   `.dat`→`.npy` converter and related work. Fixed: Createc complete-scan
   trim hazard, converter partial-kwarg crash, a mojibake log line.
5. **GUI clarity/layout review (2026-07-06)** — menu regroup (Workspace
   menu), navbar removal, large-file splits, tooltip-system unification,
   sidebar tab-width fixes.
6. **Spectroscopy display review (2026-07-06)** — per-trace display-unit
   scaling on shared axes (relative magnitudes were silently distorted up to
   1000×), the three-way SCAN_ANGLE deg/rad inconsistency, fabricated
   centre markers for out-of-frame spectra, Createc frame-offset handling.
7. **Provenance replay + canvas review (2026-07-06)** — new-format
   `.probeflow.json` sidecars made replayable; hidden-overlay handle hijack,
   marker-click steal, fit-zoom floor; quick-selection resize/move restored
   with cosmetic pens.
8. **General-fixes round (2026-07-07)** — independent workspace windows,
   themed menus/scrollbars, theme sweep pass 1, Convert/Browse tooltips.
9. **Refocus reviews (2026-07-11)** — a 4-pass review of the paring itself
   (no dangling imports/registrations/docs/dead code from removed modules;
   all cosmetic strays fixed), then a usability round from screenshot review
   and a loose-ends pass (per-image overlays now cleared on navigation,
   spin-box arrows made visible, button font floor, browse filter redesign,
   stored lattice-grid layers, adaptive zoom caps, tooltip natural sizing).
   Post-merge: `nn_histogram_nm` hardened against near-degenerate NN
   distances (numpy-version-sensitive binning failure).

Suite: ~2,530 tests, green on `main`; CI runs Python 3.11 and 3.12 with the
`dev,lattice` extras, plus a weekly dependency-canary job on latest deps.

## Open items (none block users)

Deliberate design positions — documented so they aren't "tidied" later:

- `Scan.save_*` lazy imports and the `processing/state.py` if/elif dispatcher
  are load-bearing (import cycles + heavy-dep deferral); guarded by tests
  instead of refactored. Rationale in `docs/core_derisk_plan.md`.
- Structural preferences not pursued (no defect, real churn cost):
  spectroscopy logic spread across packages; `processing/analysis.py` holds
  analysis-style measurements; many-argument measurement constructors;
  `_collect_point_sources_m` reads controller attributes via `getattr`.
- Delete/copy shortcuts act on the active ROI while overlays are hidden
  (explicit user action; left as-is). `_movable_overlay_at` is O(n²) in
  scene items (fine at current scales).

Known gaps / follow-ups, ordered roughly by value:

1. **Partial `.sxm`/`.dat` thumbnail decode in browse** — carried since the
   browse-loading review; thumbnails currently need a full decode.
2. **`.npy` bundle sidecar key convention** — `original_shape` /
   `decoded_shape` are `(Nx, Ny)` while each plane's `shape` is
   `(rows, cols)`; rename to `*_pixels` while the format is young and
   regenerate the sample bundles under `test_data/output_raw_npy/`.
3. **Reversible `.dat` conversion — investigated, deferred (2026-07-07).**
   There is no Createc `.dat` writer, and the reader is deliberately lossy
   (drops the artifact column and appendix, trims partial rows, converts
   DAC→SI). A behaviourally faithful writer is achievable but is a
   standalone project; deferred until a use case exists that `.sxm`/`.gwy`
   export does not cover.
4. **Viewer build monoliths** — `_build()` (~980 lines),
   `_build_viewer_menu_bar` (~475), `_build_fft_column` builds three tabs
   inline; method-per-tab split candidates.
5. **Theme long tail** — remaining hard-coded hex literals in
   `roi_items.py`, `lattice_grid/fft_overlay.py`, `image_canvas.py`;
   convert opportunistically when touching those files (most others were
   verified benign). Tooltip coverage remains for the image-viewer sidebars
   and TV denoise panels.
6. **Per-region display levels are session-only** — per-ROI contrast
   (`_region_drs`) resets on reload/channel change; persistence would key by
   ROI id (see `docs/notes/roi-display-notes.md`, "B5").
7. **`save_sxm` requires the source `.sxm` on disk** for its header cushion,
   so a synthetic `Scan` cannot round-trip; making the source optional is a
   nice-to-have.
8. Small noted items: no CLI command replays a `.probeflow.json` end-to-end
   (would be a feature); Createc `Rotation` sign vs SCAN_ANGLE rotation
   sense is applied symmetrically but unverified against an angled fixture
   (no angled scans in `test_data`); spec-viewer checkbox labels show
   per-channel units in Overlay mode (cosmetic);
   `SpecMappingDialog._on_suggest` loads full scans just for header/shape
   (slow on big folders).

## Running the suite (environment note)

In sandboxed/headless shells the full suite needs both:

```bash
QT_QPA_PLATFORM=offscreen \
QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import PySide6, pathlib; print(pathlib.Path(PySide6.__file__).parent/'Qt'/'plugins'/'platforms')") \
pytest
```

Without the plugin path, `QApplication` creation aborts (SIGABRT) in the
first GUI-fixture test.

## Current user-facing / reference docs

- `docs/gui.md` — GUI walkthrough
- `docs/cli.md` — command-line guide
- `docs/createc_dat_reader.md` — reader conventions
- `docs/roi_manual_test_checklist.md` — manual regression checklist
- `docs/notes/roi-display-notes.md` — ROI display/interaction architecture
- `docs/core_derisk_plan.md` — completed de-risking plan (design record)
- `docs/refocus_plan.md` — record of the 2026-07 refocus removals
- `docs/maintenance.md` — the "check regularly" watch-list
