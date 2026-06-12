# Core de-risking plan

> **Status: completed (2026-06).** All phases are done or were deliberately
> re-assessed and closed with guard tests instead (Phases 2–3 below explain
> why). Kept as a design record: it documents *why* the `Scan` lazy imports
> and the `state.py` dispatcher look the way they do, so they aren't
> "tidied" into regressions later.

**Goal:** reduce the highest-risk structural brittleness in ProbeFlow's core —
without changing behaviour — by working in small, independently-mergeable steps
on short-lived branches.

Background: the Feature Finder subsystem reviewed very cleanly because it is new,
isolated, and GUI-free. The core is healthy *today* but only after a large
remediation campaign, and it still carries entanglement that concentrates risk.
This plan targets that entanglement. See the structural review for context.

## Risk targets (highest leverage first)

1. **`Scan` god-object.** `core/scan_model.py` reaches *up* into `processing`
   and `io` (`processing_history`, six `save_*` methods) via ~31 function-local
   imports that dodge `core↔processing` / `core↔io` import cycles. These lazy
   imports are invisible to static analysis, so a renamed function/kwarg only
   fails at call time.
2. **Scattered op vocabulary.** The geometric-op names + long↔short alias maps
   (`rot90_cw` vs `rotate_90_cw`, …) are duplicated across ~10 files with ≥3
   independent alias maps and no single source of truth (`core/roi.py`,
   `processing/state.py`, `processing/gui_adapter.py`, `processing/geometry.py`,
   `core/processing_state.py`, three `cli/*`, two `gui/viewer/*`).
3. **Multiple op dispatchers.** `processing/state.py` has a ~25-branch `if/elif`
   plus a separate `apply_geometric_op_to_scan` dispatcher and the calibration
   variant — each new op must be wired into several places + the `_SUPPORTED_OPS`
   / `_ROI_ELIGIBLE_OPS` constants.
4. **Blanket `F401/F403` lint ignore** hides dead/duplicate exports across the
   whole tree (same blind spot that let recent lint slips reach `main`).

Partial existing guard: `tests/test_pipeline_connectivity.py` already asserts
op-dispatch completeness + GUI-adapter coverage, so the op *set* is guarded; the
*alias-map consistency* and the *dependency direction* are not.

## Working principles

- **Test-first at every seam that changes**; several of those tests become
  permanent drift guards (part of the fix, not just scaffolding).
- **Each step is behaviour-preserving and independently mergeable.**
- **Per-step verification harness** (local Qt aborts on dialog construction, so
  GUI is validated statically): `ruff check` (F821 catches lazy-import/name
  breakage) + an import-assembly check + the backend test suite. GUI paths get
  static checks + manual confirmation in the app.
- **Short-lived branch per phase**, branched off `main`, rebased before merge —
  a parallel contributor pushes to `main`, so we avoid a long-lived branch.

## Decisions taken

- **Phase 2 end-state:** keep `Scan.save_*` / history as *thin,
  statically-checkable delegations* (lower risk, ~80% of the benefit) rather than
  fully moving them off `Scan`. Revisit only if we want the full inversion.

## Phases

### Phase 0 — Safety net (no production code change) — _in progress_
- [x] 0.1 Op-vocabulary alias-equivalence guard: long-form and short-form op
      names must produce identical results via `ROI.transform` and
      `apply_geometric_op_to_scan`. (`tests/test_core_op_vocab_invariants.py`)
- [x] 0.2 `Scan` save/history round-trip characterization
      (`tests/test_scan_model_roundtrip.py`).
- [ ] 0.3 Dispatcher behaviour snapshot — largely already covered by
      `test_pipeline_connectivity`; extend only if gaps appear.

### Phase 1 — Single op-vocabulary source of truth
- [x] 1.1 New op-vocab module: `LONG_TO_SHORT`/`SHORT_TO_LONG` + `to_short`/
      `to_long`, `LOSSLESS_OPS`, `DIMENSION_SWAPPING_OPS`, `SIMPLE_GEOMETRIC_OPS`;
      consistency test (`tests/test_op_vocab.py`).
- [x] 1.2 Point `processing/state.py` at it (dropped its private `_LONG_TO_SHORT`
      / `_LOSSLESS` and the inline dim-swap + grouped-branch name lists).
- [x] 1.3 Point `core/roi.py` at it (`transform` alias map, the lossless check,
      and `_post_transform_shape` dim-swap). `transform_all` delegates to
      `transform`, so no separate map there.
- [x] 1.5 (done early) The module lives in **`core/op_vocab.py`**, not
      `processing/`: `core.roi` needs it, and hosting it in `processing` would
      force a `core → processing` import inversion. It sits alongside
      `_SUPPORTED_OPS` (already in `core/processing_state.py`), so the op
      vocabulary now has a single owner in the lowest layer.
- [x] 1.4 Point `processing/gui_adapter.py` at it (its `SIMPLE_GEOMETRIC_OPS`
      membership list). The `cli/*` files use individual op names in one-off
      factory functions / argparse wiring — legitimate single-use, not a
      consolidatable set or alias map, so they were left as-is.

**Phase 1 complete.** The geometric-op vocabulary now has a single owner
(`core/op_vocab.py`); `core/roi.py`, `processing/state.py`, and
`processing/gui_adapter.py` all source it from there. The drift hazard
(arch-backend #9) is closed and guarded by `tests/test_op_vocab.py` +
`tests/test_core_op_vocab_invariants.py`.

### Phase 2 — `Scan` god-object — _re-assessed & addressed_

Re-assessment (from tracing the actual dependencies): the function-local
imports in `Scan.save_*` / `processing_history` are **not** removable without a
regression. Two independent reasons:

1. **Real runtime cycle.** `io.writers.{sxm,png,pdf,gwy}` import this module at
   runtime (`Scan`, and `PLANE_CANON_*` in sxm, used in an `isinstance`-style
   layout check) — so hoisting the writer imports into `scan_model` would cycle.
2. **Heavy-dep deferral.** `png`/`pdf` pull `matplotlib`. A plain
   `import probeflow.core` currently pulls **no** heavy deps (verified); making
   the writer imports top-level would drag `matplotlib` into every CLI/headless/
   batch import. The lazy imports are *load-bearing*, not accidental.

So the lazy delegations are kept by design. The actual risk they posed
("renamed writer fails only at the caller's first save") is mitigated instead by:
- [x] A `core de-risk Phase 2` comment in `scan_model` documenting *why* the
      imports are local, so they aren't "tidied" into top-level imports later.
- [x] Full delegation test coverage in `tests/test_scan_model_roundtrip.py`:
      `save_png` / `save_csv` / `save_pdf` round-trip; `save()` suffix dispatch;
      a resolve-guard that imports every writer (incl. `write_sxm`) and asserts
      each `Scan.save_*` method exists; `save_gwy` skips without optional
      `gwyfile`; history getter/setter idempotency. A renamed writer now fails
      in CI, not at the user's call.

Carried forward (not blocking): `save_sxm` reads the source `.sxm` on disk for
its header cushion, so it can't round-trip a synthetic Scan — making that source
optional remains a nice-to-have, tracked here.

### Phase 3 — dispatcher — _re-assessed & addressed_

Re-assessment (same discipline as Phase 2): a name→handler **registry was not
worth the risk** for this dispatcher.

- The `if/elif` branches are genuinely heterogeneous (ROI resolution, params
  dataclasses, calibration threading, nested-ROI recursion, operand resolver),
  so a registry would have to thread a shared **context object** through ~25
  handlers — the deferred arch-backend #18 — i.e. a large rewrite of the single
  most central function.
- The error it would guard against is already covered: `_SUPPORTED_OPS` ↔
  dispatch drift is caught by `test_pipeline_connectivity`, and there is **no
  duplicated op-classification** to drift — the "second/third dispatcher" worry
  from the review was overstated: `apply_processing_state_with_calibration`
  loops over `apply_processing_state` and detects shape changes *empirically*
  (not via an op-name set), and `apply_geometric_op_to_scan` is a distinct
  Scan-level op (now sourcing names from `op_vocab`).

Delivered instead (low-risk, genuinely closes a gap):
- [x] 3.1 Calibration-path op coverage: `test_pipeline_connectivity` now also
      runs **every** `_SUPPORTED_OPS` op through
      `apply_processing_state_with_calibration` (the path the GUI/CLI use), plus
      a `_ROI_ELIGIBLE_OPS ⊆ _SUPPORTED_OPS` invariant.
- [x] 3.2 Documented the dispatcher design decision in `state.py` so the
      if/elif isn't "registry-ified" later without weighing the context-object
      cost.

Deferred (tracked, not a defect): a context-object refactor (arch-backend #18)
would be the prerequisite to a clean registry if op count grows substantially.

### Phase 4 — Lint hygiene (low risk) — _done_
- [x] 4.1 Removed the global `F401/F403/F405` ignore from `pyproject.toml`.
      Dead-import / star-import detection is now ON tree-wide; the genuine
      re-export shims (package `__init__`s + `gui/processing.py`) keep per-file
      ignores, and intentional single re-exports use `# noqa: F401`. ~150 dead
      imports were removed (the bulk were post-split residue in
      `image_viewer.py`). Verified: all touched modules import cleanly, umbrella
      package imports succeed, and the backend/test suites pass — so no implicit
      re-export was broken.

## Follow-ups discovered

### Pooled-worker signals destroyed off the main thread (one fix landed)
A hard `SIGSEGV` opening Feature Finder/Counting traced to a `QThreadPool`
QRunnable auto-deleting on the *worker* thread while it solely owned a
parentless `*Signals` QObject — destroying that QObject (with cross-thread
connections) off the main thread corrupts Qt internals; the crash surfaced in
the app-level tooltip event filter.

- **Fixed (FC crash path):** `_FeaturesWorker` (features) and `_ScanLoadWorker`
  (app) now parent their auto-created signals to the `QApplication` and
  `deleteLater()` them after emit. Guarded by `tests/test_worker_signals_lifetime.py`
  (runs in a real GUI env; this headless box can't construct a `QApplication`).
- **Fixed (same antipattern, swept):** introduced a shared `_PooledWorker` base
  (`gui/workers.py`) that parents its signals to the `QApplication` and
  `deleteLater()`s them after `work()`. Migrated `ThumbnailLoader`,
  `FolderThumbnailLoader`, `SpecThumbnailLoader`, `ViewerLoader`,
  `ConversionWorker`, and `_ScanLoaderWorker` (`image_arithmetic`) onto it.
  `ChannelLoader` was confirmed safe (its signals are created and retained by the
  caller, `browse/panels.py`) and left as a plain `QRunnable`; `_TVWorker` is
  likewise window-owned. So every fire-and-forget pooled worker now owns its
  signals on the main thread.

## Suggested merge order
Phase 0 → Phase 1 → Phase 4 → Phase 2 (per-method) → Phase 3 (batched).
Each phase leaves the codebase fully working; stopping after any one banks real
risk reduction.
