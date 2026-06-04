# Core de-risking plan

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
- [ ] 1.1 New `processing/op_vocab.py`: canonical names, `LONG_TO_SHORT` /
      `SHORT_TO_LONG`, `LOSSLESS_OPS`, `GEOMETRIC_OPS`, `SHAPE_CHANGING_OPS`;
      internal-consistency test. No callers yet.
- [ ] 1.2 Point `processing/state.py` at it.
- [ ] 1.3 Point `core/roi.py` `transform`/`transform_all` at it. ⚠ highest care.
- [ ] 1.4 Point `processing/gui_adapter.py` and `cli/*` at it.
- [ ] 1.5 Resolve the layer of `_SUPPORTED_OPS` (keep in core + re-export, lean).

### Phase 2 — Untangle the `Scan` god-object (smallest steps)
- [ ] 2.1 History getter/setter → statically-checkable delegation
      (`TYPE_CHECKING` import).
- [ ] 2.2–2.6 Each `save_*` method → thin delegation to `io.writers.x(scan,…)`,
      one commit each; internal call sites use the writer directly.
- [ ] 2.7 Optionally drop shims once call sites are migrated.

Finding surfaced during Phase 0: `save_sxm` → `write_sxm` reads the *source*
`.sxm` file on disk to reuse its header cushion, so it raises
`FileNotFoundError` for an in-memory `Scan` whose `source_path` doesn't exist.
Consider, during Phase 2, making the header-cushion source explicit/optional so
synthetic Scans can be written without a real source file.

### Phase 3 — Consolidate the dispatcher
- [ ] 3.1 Introduce an op registry alongside the `if/elif`; test coverage ==
      `_SUPPORTED_OPS`.
- [ ] 3.2 Migrate ops into the registry in small batches.
- [ ] 3.3 Fold `apply_geometric_op_to_scan` + calibration variant onto the same
      registry + op metadata.
- [ ] 3.4 Remove the dead `if/elif`.

### Phase 4 — Lint hygiene (low risk, anytime)
- [ ] 4.1 Narrow the global `F401/F403` ignore to per-file ignores on the
      genuine re-export shims; fix/annotate dead imports that surface.

## Suggested merge order
Phase 0 → Phase 1 → Phase 4 → Phase 2 (per-method) → Phase 3 (batched).
Each phase leaves the codebase fully working; stopping after any one banks real
risk reduction.
