# Maintenance watch-list — things to check regularly

Living list of the known ways ProbeFlow degrades over time and the routine
that keeps them in check. Review this quarterly, and immediately whenever the
weekly **dependency-canary** CI job goes red.

## Dependency drift (the main one)

ProbeFlow's riskiest dependencies are the two whose breaking releases take
the GUI down wholesale: **PySide6** and **numpy**. The machinery in place:

- `constraints.txt` — exact pins of the last verified set (header carries the
  verification date).
- **Weekly dependency-canary CI** — installs unpinned latest and runs the
  suite; a red canary means "latest deps have drifted", not "your change
  broke something".
- `probeflow/core/env_check.py` — startup check that prints clear terminal
  notes ("ProbeFlow needs numpy >= 1.26; you have 1.24.3"; "PySide6 X is
  newer than the last verified Y") and a crash banner listing installed
  versions on any unhandled exception, so pasted bug reports name the
  suspect.

**When bumping pins** (after a green full suite on the new versions), update
*all three together* — `constraints.txt` (pins + date),
`env_check.VERIFIED` (the "newest verified" column), and, if the ruff pin
moved, the `ruff-pre-commit` rev in `.pre-commit-config.yaml`.
`tests/test_env_check.py` fails if `VERIFIED` drifts from
`pyproject.toml`/`constraints.txt`, so a forgotten update is caught in CI.

Specific packages to watch:

- **PySide6 / Qt** — the QSS-heavy styling (spin-box sub-controls, cosmetic
  pens, menu/scrollbar theming) is exactly what shifts between Qt minor
  releases and across macOS/Windows/Linux. After any PySide6 bump, smoke the
  GUI *visually* (browse → viewer → FFT viewer → a dialog with spin boxes) —
  almost none of the visual layer is test-covered.
- **numpy** — version-sensitive numeric edge cases pass on new numpy and fail
  on old (precedent: the `nn_histogram_nm` "Too many bins" failure, fixed
  2026-07-11 by passing explicit bin edges). The dev machines run newer numpy
  than most users; treat "works here" with suspicion. A CI leg pinned to the
  *oldest* supported versions would close this asymmetry — not set up yet.
- **scikit-image** — deprecates and relocates APIs aggressively; check the
  release notes on each minor.
- **gwyfile** — effectively unmaintained upstream; the optional `.gwy` writer
  may need vendoring or replacing if it breaks on a future Python.
- **Python versions** — CI runs 3.11 and 3.12; the dev machines already run
  3.13. Add 3.13 to the CI matrix (cheap) and drop 3.11 when its usage fades.

## File-format and interchange fragility

- **`.sxm` writer cushion** — exported `.sxm` files reproduce one
  reverse-engineered reference layout via captured byte fixtures
  (`probeflow/data/file_cushions/`). After a Nanonis or Gwyddion update,
  verify an exported `.sxm` still opens there. `save_sxm` also still needs
  the source file on disk for its header cushion.
- **Other labs' raw files** — the parser fixtures cover one lab's Createc
  vintage and a handful of `.sxm`/`.sm4` files. RHK `.sm4` has many page
  types unseen here; Nanonis headers vary by software version. The dangerous
  failure is plausible data with wrong calibration, not a crash — grow the
  fixture corpus from every external bug report, and keep decode warnings
  loud (`Scan.warnings` surfaces in the viewer).

## Held invariants (guarded by tests — keep them green)

- **Sidecars never crash the app.** Corrupt/old `.rois.json` /
  `.masks.json` / `.probeflow.json` files must load as an error message on
  the status bar, never an exception and never silently as "empty" (which
  would let the next save overwrite the user's damaged-but-recoverable
  data). Pinned by `tests/test_sidecar_discovery.py`; extend it when adding
  any new sidecar format.
- **Display == export == provenance-replay** for representative pipelines
  (`tests/test_workflow_replay.py`) — the core provenance promise.
- **Op vocabulary + dispatcher completeness**
  (`tests/test_op_vocab.py`, `tests/test_pipeline_connectivity.py`).
- **Raw files are read-only** — processing and export always write new
  files.

## Before 1.0

- **Sidecar schema versioning** — `.rois.json` / `.masks.json` /
  `.probeflow.json` need explicit schema versions and a written migration
  policy before the formats calcify in users' data. (The replay code already
  accepts two step formats — the pattern exists.)
- **Distribution** — a PyPI release and a double-click installer are the gap
  between "a repo" and "a community tool"; a locked distribution also
  protects the pared codebase from monolithic re-growth.
- **`CITATION.cff` / DOI** so academic users can cite the tool.

## Structural guard-rails (see also `docs/core_derisk_plan.md`)

- `Scan.save_*` lazy imports and the `processing/state.py` if/elif
  dispatcher are **deliberate** — do not "tidy" them (rationale in the
  de-risk record).
- Every new feature passes the mission test from `docs/refocus_plan.md`:
  *does it help someone browse or process an STM image while keeping
  physical units?* Prefer deletion over abstraction.
