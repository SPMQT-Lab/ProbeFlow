# Dataset Builder branch review — 2026-07-06

Read-only review of `dataset_builder` at `f7bbc6f` (28 commits, branched from
the PR #45 merge `850a443`). Reviewer: Peter + Claude. No functional code was
changed — this document and the matching `# REVIEW(2026-07-06, …)` inline
comments are the whole deliverable. Grep for `REVIEW(2026-07-06` to find every
anchored site.

## What the branch is (for whoever integrates it)

An ML dataset creation cockpit: queue → review → export of scan planes.

- **Backend** `probeflow/dataset_builder/` (Qt-free): queue discovery over a
  folder index, per-sample label proposals (`step_edge` reusing
  `analysis/step_edges`, `canny`, `quickseg`), review records persisted in
  sidecars next to the data, frozen export snapshots with a
  `manifest.csv`/`.json` (pixel sizes, units, source hashes, proposal params,
  per-sample provenance).
- **QuickSeg** (`quickseg.py`): seeded-watershed terrace segmentation with an
  eight-knob preprocessing stack (TV denoise → bilateral → anisotropic blur →
  multi-scale Scharr gradient → hysteresis edges → oriented-line closing →
  barrier elevation) plus horizontal scan-line defect suppression.
- **GUI** `probeflow/gui/dataset_builder/`: a new workspace page (queue tray,
  sample canvas, QuickSeg controls), pooled workers following the house
  `_PooledWorker` pattern.
- Plus browse metadata filters/export and CLI `dataset` commands.

The design instinct — reuse ProbeFlow loaders/masks/ROIs/provenance instead of
a parallel data system — is right, and is also the source of finding 1.

## Findings (severity-ordered; each has an inline anchor)

1. **HIGH — sidecar namespace collision can wipe labelling work.**
   `sidecar_state.default_state_sidecar_path` stores review state in
   `<stem>.probeflow.json` *next to the scan* — the exact path
   `write_provenance_sidecars` produces when a user exports `<stem>.csv/.gwy`
   beside the data. Non-forced export → confusing `FileExistsError`; with
   `--force`/overwrite the provenance writer replaces the file wholesale and
   **every review record for that scan is gone**. Rename the state file (e.g.
   `<stem>.dataset.json`) while the format is young; keep a one-shot
   migration read from the old name.
2. **HIGH — sample-ID collisions across subfolders.**
   `loading.plane_sample_id` is `stem_planeN` and the queue walks
   recursively: same-stem scans in different folders overwrite each other's
   exported arrays/masks (overwrite=True) or kill the export mid-run
   (overwrite=False). Disambiguate with a relative-path hash or parent
   folder name.
3. **MED — accepted samples can export without their label.**
   In `export.py`, if no mask named `label_name` remains in the sidecar, the
   row still exports with empty `mask_paths_json` — silently unlabeled
   training data. Skip with a warning (or error) for `label_type == "mask"`.
4. **MED — export-time QuickSeg recompute.** When the result `.npy` is
   missing, `export.py` re-runs the watershed with current code/params — the
   exported labels may not be what the reviewer accepted. Record
   `recomputed_at_export` in the manifest at minimum.
5. **MED — `quickseg.reorder_labels_area` breaks the seed↔label id
   contract.** Output ids are area-ranked while the exported seeds JSON keeps
   the user's `terrace_label_id`s; GUI seed-dot colours don't match region
   fills; ids flip between parameter tweaks.
6. **MED — `loading.load_scan_plane` crashes on `scan_range_m=None`**
   (unguarded tuple unpack) and takes the whole export loop down.
7. **LOW — `sidecar_state.mark_exported` is dead code**, so the "exported"
   status/`exported_at` in `REVIEW_STATUSES` never happens.
8. **LOW — partial export dirs.** A collision mid-loop leaves a snapshot
   with no manifest and no cleanup; also the QuickSeg `.npy` is written
   before its `.png` collision check.
9. **LOW — mask upsert-by-name** (`annotations.save_mask_annotation`) can
   silently overwrite a viewer-made mask of the same name.

## Pre-existing test failure

`test_dataset_builder_view_tray.py::test_dataset_builder_panel_refresh_uses_display_only_flatten`
fails on some numpy versions: the display-flattened synthetic image is
~constant and `histogram_from_array` (`probeflow/processing/display.py`) has
no degenerate-range guard ("Too many bins for data range"). The proper fix is
a span epsilon in `histogram_from_array` (main code — coordinate, since it
touches shared display logic).

## Integration notes for the merge to main (post `a6e48d0` / `7a0a0dc`)

Confirmed via merge-tree; both need real reconciliation, not a textual pick:

1. **`gui/app.py`** — the branch adds the Dataset Builder page at
   content-stack **index 6** assuming the old 0–5 layout, and registers its
   mode action in the old Tools menu. Main has renumbered the stacks to
   0 browse / 1 convert / 2 tv / 3 dev / 4 survey and moved workspace pages
   into the **Workspace** menu. On merge: the page becomes **index 5** and
   the action belongs in the Workspace menu (suggest Ctrl+Shift+D — Ctrl+3
   is taken by Feature Counting under Tools).
2. **`core/metadata.py`** — SEMANTIC conflict: main (PR #48) switched
   `scan_range` to `decoded_scan_range_m(report)` (decoded-plane extent,
   correct for first-column removal and partial-scan trims); the branch
   builds `visible_scan_range`/`completion_pct` on the old `Length x[A]`
   header range. The branch's completion computation should be rebased onto
   the decoded values (completion = decoded rows / programmed rows), not the
   header range.
3. House conventions to apply before/at merge: `gui/dataset_builder/tab.py`
   is 1,700 lines (main splits at natural seams); no `docs/gui.md`/`cli.md`
   sections yet; new GUI test modules must be registered in
   `tests/conftest.py::GUI_TEST_MODULES` for the sandboxed-Qt preflight.
4. `ruff check probeflow/dataset_builder` currently reports 6 errors (all
   auto-fixable unused imports) — main's CI runs ruff, so the merge will
   fail lint until `ruff --fix` is run on the branch.

## Verified sound (no action needed)

- `manifest.py` atomic writes; `cache.py` mtime+size cache keys and
  parameter/seed fingerprints; `sidecar_state.py` JSON hygiene + atomic
  writes (modulo the filename, finding 1).
- Worker threading follows the house `_PooledWorker` + signals pattern; no
  QPixmap construction off the GUI thread (the PR #20 lessons hold).
- QuickSeg axis conventions: `smooth_along_scan` correctly maps to axis 1
  (fast axis) and `smooth_across_scan` to axis 0 in the anisotropic blur.
- Proposal parameter provenance is carried into review records and the
  manifest.

## Status addendum (2026-07-07, post-merge)

The branch was merged to main (`cac4646`) the day after this review; the
notes above were written against the pre-merge branch. Where things stand:

- Integration notes 1–4: **done** — landed via Rohan's conflict resolution
  (`core/metadata.py` reconciled onto decoded ranges, correctly) plus the
  `post-merge-stabilisation` branch (menu regroup, ruff, conftest
  registration, `histogram_from_array` degenerate-range guard). The
  resurrected Feature Counting page removal is in progress separately.
- Findings 1–9 (sidecar namespace collision, sample-ID collisions,
  accepted-without-mask exports, export-time recompute, seed↔label ids,
  `scan_range_m=None` crash, dead `mark_exported`, partial export dirs,
  mask upsert) are **still open on main** — the inline
  `REVIEW(2026-07-06, …)` comments now mark them in the merged code.
