# Contributing to ProbeFlow

Thanks for contributing. This document captures the working agreements
that keep the codebase navigable as it grows.

## Local setup

```bash
git clone https://github.com/SPMQT-Lab/ProbeFlow.git
cd ProbeFlow
python -m pip install -e ".[dev,features]"
pip install pre-commit
pre-commit install
```

`pre-commit install` wires the repo's hooks (ruff, trailing whitespace,
file size, merge-conflict markers) so they run on every `git commit`.

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

## Where the implementation actually lives

The package directory layout looks fully decomposed, but two large
files still contain the bulk of the GUI and CLI implementation:

- `probeflow/gui/_legacy.py` (~6,200 LoC) — main window, all dialogs,
  panels, sidebars, the developer terminal.
- `probeflow/cli/_legacy.py` (~2,200 LoC) — every command parser and
  dispatcher.

The other modules in `gui/` and `cli/` are mostly thin re-exports back
into these files. We are decomposing `_legacy.py` opportunistically:
when you touch a class for a feature, pull it out into its proper
submodule. Avoid standalone refactor sprints — they have no stopping
condition and break tests for too long.

## Commit style

- Imperative mood, lower-case first word ("add foo", "fix bar").
- One logical change per commit. The diff and the message should agree.
- If a commit fixes a bug introduced earlier in the same branch,
  squash it before opening a PR.

## Pull requests

- Describe the *why*, not the *what* — the diff already says what.
- If the change touches a GUI dialog, include a screenshot or a
  short note on what visually changed.
- If the change adds a public API or CLI command, update the README.
