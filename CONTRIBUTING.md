# Contributing to ProbeFlow

Thanks for contributing. This document captures the working agreements
that keep the codebase navigable as it grows.

## Local setup

```bash
git clone https://github.com/SPMQT-Lab/ProbeFlow.git
cd ProbeFlow
python -m pip install -e ".[dev,features]" -c constraints.txt
pip install pre-commit
pre-commit install
```

`pre-commit install` wires the repo's hooks (ruff, trailing whitespace,
file size, merge-conflict markers) so they run on every `git commit`.

`constraints.txt` pins the exact dependency versions the suite was last
verified against (notably PySide6 and numpy — the two whose breaking
releases would take the GUI down wholesale). Omit `-c constraints.txt` to
live on latest; the weekly **dependency-canary** CI job does exactly that,
so check its status before blaming your own changes for a fresh-environment
breakage. When bumping a pin: run the full suite, then refresh the pin and
the "last verified" date in the file header.

## Running the test suite

```bash
pytest                       # run everything
pytest tests/test_roi.py     # one file
pytest -k "lattice"          # by name
QT_QPA_PLATFORM=offscreen pytest   # if you don't have a display
```

CI runs `pytest` on Python 3.11 and 3.12 with the `dev` and `features`
extras installed (so OpenCV / scikit-learn paths are exercised).

## Linting

```bash
ruff check probeflow tests
```

Currently we only enforce **pyflakes** (real bugs: undefined names,
shadowed imports, etc.). The full pycodestyle / pep8-naming /
import-sorting suite will be turned on after a one-time format pass.
Until then, follow the existing style of the file you are editing.

## Architectural boundaries

Each top-level subpackage has a docstring describing what belongs there
and what does **not**. Read these before adding new code:

- `probeflow/core/` — `Scan` model, loaders, metadata, ROI geometry,
  validation. **Not** GUI, **not** numerical kernels.
- `probeflow/io/` — file sniffing, vendor-specific readers and writers,
  `.sxm` byte layout. **Not** processing, **not** display.
- `probeflow/processing/` — Qt-free numerical kernels and the
  `ProcessingState` model.
- `probeflow/analysis/` — particles, lattice, spectroscopy plotting,
  feature counting.
- `probeflow/provenance/` — export provenance now; future scan-graph
  dataclasses (`ImageNode`, `MeasurementNode`, etc.).
- `probeflow/gui/` — PySide6 widgets and dialogs only. **Not** numerical
  kernels, **not** vendor parsers, **not** model definitions.
- `probeflow/cli/` — orchestration over the canonical APIs above.
  **Not** model definitions, **not** numerical kernels, **not** GUI.
- `probeflow/plugins/` — future plugin registry; do not migrate in-tree
  ops here yet.

If a change crosses a boundary, prefer adding a small adapter in the
caller over moving domain code into a foreign package.

## Compatibility shims

The decomposition of the original monolithic GUI and CLI files is
complete: the directory layout is the real code layout, and every class
lives in its proper submodule (`gui/dialogs/`, `gui/viewer/`,
`gui/browse/`, `cli/commands/`, …).

Two small shims remain only to keep the historical import surface
stable:

- `probeflow/gui/compat.py` — re-exports consumed via
  `gui/__init__.py` so `from probeflow.gui import X` keeps working.
- `probeflow/cli/_legacy.py` — re-exports every public CLI name into
  the canonical `cli/parser.py`, `cli/processing_ops.py`, and
  `cli/commands/*` modules.

Do not add new code to either shim — new GUI or CLI code goes directly
in the appropriate submodule. When a re-export stops having external
users, it can simply be deleted.

## Commit style

- Imperative mood, lower-case first word ("add foo", "fix bar").
- One logical change per commit. The diff and the message should agree.
- If a commit fixes a bug introduced earlier in the same branch,
  squash it before opening a PR.

## Pull requests

- Describe the *why*, not the *what* — the diff already says what.
- If the change touches a GUI dialog, include a screenshot or a
  short note on what visually changed.
- If the change adds a public API or CLI command, update the relevant docs
  (`README.md` for broad user-facing changes, `docs/cli.md` for CLI details).
