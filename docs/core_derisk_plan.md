# Core de-risking plan

> **Status: completed (2026-06).** Kept as a design record: it documents *why*
> the `Scan` lazy imports and the `state.py` dispatcher look the way they do,
> so they aren't "tidied" into regressions later. The step-by-step checklists
> were pruned once every phase landed; outcomes below.

**Goal (achieved):** reduce the highest-risk structural brittleness in
ProbeFlow's core — without changing behaviour — in small, independently
mergeable steps.

## Outcomes

### Op vocabulary — single source of truth (Phase 1, done)

The geometric-op names and long↔short alias maps were duplicated across ~10
files. They now have one owner, **`core/op_vocab.py`**
(`LONG_TO_SHORT`/`SHORT_TO_LONG`, `LOSSLESS_OPS`, `DIMENSION_SWAPPING_OPS`,
`SIMPLE_GEOMETRIC_OPS`); `core/roi.py`, `processing/state.py`, and
`processing/gui_adapter.py` all source it from there. Drift is guarded by
`tests/test_op_vocab.py` and `tests/test_core_op_vocab_invariants.py`. It
lives in `core/` (not `processing/`) because `core.roi` needs it and hosting
it higher would invert the layering.

### `Scan` lazy imports are load-bearing — do not hoist (Phase 2)

`Scan.save_*` / `processing_history` use function-local imports. Tracing the
dependencies showed they are **not** removable:

1. **Real runtime cycle** — `io.writers.{sxm,png,pdf,gwy}` import
   `core.scan_model` at runtime, so hoisting writer imports into `scan_model`
   would cycle.
2. **Heavy-dep deferral** — `png`/`pdf` pull matplotlib; a plain
   `import probeflow.core` pulls **no** heavy deps today, and top-level writer
   imports would drag matplotlib into every headless/batch import.

The risk the lazy imports posed (a renamed writer failing only at the user's
first save) is mitigated by full delegation coverage in
`tests/test_scan_model_roundtrip.py` (round-trips, suffix dispatch, a
resolve-guard importing every writer). Carried forward, not blocking:
`save_sxm` reads the source `.sxm` for its header cushion, so a synthetic
`Scan` cannot round-trip.

### The `state.py` dispatcher stays an if/elif — do not registry-ify (Phase 3)

The ~25 branches are genuinely heterogeneous (ROI resolution, params
dataclasses, calibration threading, nested-ROI recursion, operand resolver);
a name→handler registry would force a shared context object through every
handler — a large rewrite of the single most central function for no caught
bug. The drift it would guard against is already covered:
`test_pipeline_connectivity` asserts every `_SUPPORTED_OPS` op dispatches
through **both** `apply_processing_state` and the calibration path, plus
`_ROI_ELIGIBLE_OPS ⊆ _SUPPORTED_OPS`. A context-object refactor remains the
prerequisite if the op count ever grows substantially.

### Lint hygiene (Phase 4, done)

The global `F401/F403/F405` ignore was removed from `pyproject.toml`;
dead-import detection is on tree-wide, with per-file ignores only on genuine
re-export shims. ~150 dead imports were removed in the process.

### Pooled-worker signal ownership (follow-up, done)

A hard SIGSEGV traced to `QThreadPool` auto-deleting a QRunnable on the
worker thread while it solely owned a parentless `*Signals` QObject with
cross-thread connections. All fire-and-forget pooled workers now inherit the
shared `_PooledWorker` base (`gui/workers.py`), which parents signals to the
`QApplication` and `deleteLater()`s them after `work()`. Guarded by
`tests/test_worker_signals_lifetime.py`.
