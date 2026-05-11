<p align="center">
  <img src="assets/logo.gif" alt="ProbeFlow logo" width="100%"/>
</p>

# ProbeFlow

ProbeFlow is a focused STM/SPM browser, converter, and analysis tool for
Createc and Nanonis image and point-spectroscopy data.

The current goal is practical lab use: open a folder, browse scans and spectra,
apply standard STM image corrections, draw ROIs, make profiles/histograms/FFT
checks, and export files with enough metadata to understand what happened.

ProbeFlow is not trying to replace Gwyddion, Fiji/ImageJ, WSXM, or AISurf. It is
intended to be the workflow layer around them: reliable loading, routine cleanup,
simple measurements, conversion, and traceable handoff.

## Status

ProbeFlow is beta software. The GUI, CLI, Python API, and sidecar formats may
change between commits. Raw microscope input files are not modified.

Working today:

- Createc `.dat` scan loading and `.VERT` spectroscopy loading.
- Nanonis `.sxm` scan loading and Nanonis spectroscopy `.dat` loading.
- Folder browsing with image and spectroscopy thumbnails.
- Createc `.dat` to Nanonis-compatible `.sxm` conversion.
- PNG, PDF, CSV, JSON, `.sxm`, and `.gwy` export paths.
- Processing operations such as row alignment, bad-line removal, plane/polynomial
  background subtraction, STM line background, smoothing, FFT filters, periodic
  notch filtering, TV denoising, zero-point/zero-plane tools, and geometric
  flips/rotations.
- Analysis paths for line profiles, histograms, FFT spectra, particles, feature
  counts, lattice extraction, unit-cell averaging, grain detection, and
  spectroscopy plotting.
- ROI drawing and ROI-aware analysis/processing for the main image viewer.
- Export provenance for PNG/JSON-style outputs.

Still evolving:

- The main window (`ProbeFlowWindow`) and most CLI command runners still live
  in `probeflow/gui/_legacy.py` and `probeflow/cli/_legacy.py`. Many widgets
  and dialogs have already been extracted to dedicated submodules (`gui/browse/`,
  `gui/workers/`, `gui/models/`, `gui/terminal/`, `gui/dialogs/`, `gui/viewer/`,
  `gui/rendering/`, `gui/image_canvas/`, `gui/roi_items/`, `gui/features/`);
  extraction is ongoing and opportunistic.
- CLI command runners are being moved to `probeflow/cli/commands/` submodules.
  The submodules exist as extraction targets and re-export the runners from
  `_legacy` while the migration is in progress; the active parser entry point
  is still `cli/_legacy.py`.
- `ProcessingState` is the canonical processing model, but
  `Scan.processing_history` remains as a compatibility representation for some
  writers and CLI paths.
- `ScanGraph` dataclasses exist in `probeflow/provenance/graph.py` and are tested,
  but the GUI/CLI do not yet use the graph as the runtime source of truth.
- The plugin registry and `PluginSpec` API exist as a foundation
  (`probeflow/plugins/`), but in-tree operations are not yet discovered from
  the registry at runtime. `plugins/manifest.py` is scaffolding for the wiring
  step.
- Full session persistence, a provenance panel, a measurement table, and
  DisplayLayer-style per-region display settings are planned, not implemented.

## Installation

```bash
git clone https://github.com/SPMQT-Lab/ProbeFlow.git
cd ProbeFlow
python -m pip install -e .
```

Python 3.11+ is required.

For development:

```bash
python -m pip install -e ".[dev,features]"
pytest
```

Optional extras:

```bash
python -m pip install -e ".[features]"  # OpenCV / scikit-learn feature tools
python -m pip install -e ".[gwyddion]"  # optional Gwyddion writer dependency
```

## Quick Start

Launch the GUI:

```bash
probeflow gui
```

Inspect a scan:

```bash
probeflow info scan.dat
probeflow info scan.sxm --json
```

Convert Createc `.dat` to Nanonis `.sxm`:

```bash
probeflow convert scan.dat scan.sxm
```

Apply a small processing pipeline and export a PNG:

```bash
probeflow pipeline scan.dat \
    --steps align-rows:median plane-bg:1 \
    --png --colormap gray \
    -o scan_processed.png
```

Prepare a PNG handoff with provenance:

```bash
probeflow prepare-png scan.dat aisurf_input.png \
    --steps align-rows:median plane-bg:1 \
    --colormap gray
```

## Supported Files

Input:

| File type | Use |
|---|---|
| Createc `.dat` | STM/SPM image scan |
| Createc `.VERT` | Point spectroscopy |
| Nanonis `.sxm` | STM/SPM image scan |
| Nanonis `.dat` | Point spectroscopy |

Output:

| File type | Use |
|---|---|
| `.sxm` | Converted or processed scan data |
| `.png` | Image export |
| `.pdf` | Figure-style export |
| `.csv` | Numerical data export |
| `.json` | Metadata, provenance, or analysis output |
| `.gwy` | Gwyddion handoff/export |

## CLI Overview

The top-level command is `probeflow`. Run `probeflow <command> --help` for the
full options.

Common commands:

| Command | Purpose |
|---|---|
| `gui` | Launch the PySide6 GUI |
| `convert` | Suffix-driven scan conversion/export |
| `pipeline` | Apply ordered processing steps |
| `prepare-png` | PNG handoff with provenance sidecar |
| `plane-bg` | Polynomial background subtraction, including ROI-aware modes |
| `align-rows` | Per-row median/mean/linear offset correction |
| `remove-bad-lines` | Detect and interpolate bad scan lines |
| `smooth` | Gaussian smoothing |
| `fft` | Fourier-domain low/high-pass filtering |
| `histogram` | Pixel-value histogram, optionally ROI-aware |
| `fft-spectrum` | FFT magnitude spectrum, optionally ROI-aware |
| `profile` | Line profile from endpoints or a named line ROI |
| `autoclip` | Suggest display clip percentiles |
| `particles`, `count`, `classify` | Feature detection workflows |
| `lattice`, `unit-cell` | Lattice extraction and unit-cell averaging |
| `spec-info`, `spec-plot`, `spec-overlay`, `spec-positions` | Spectroscopy tools |

Pipeline step syntax is `name[:param1,param2,...]`, for example:

```bash
probeflow pipeline scan.sxm \
    --steps remove-bad-lines align-rows:median plane-bg:1 smooth:1.2 \
    -o scan_processed.sxm
```

## GUI Overview

The GUI currently has six tabs:

- **Browse**: folder scan, thumbnails, full image viewer, histogram controls,
  ROI tools, line profiles, FFT viewer, spectroscopy markers, and PNG export.
- **Convert**: folder-in/folder-out Createc conversion to `.sxm` and/or PNG.
- **FeatureCounting**: particles, template matching, lattice workflows, result
  table, and JSON export.
- **TV-denoise**: Chambolle-Pock TV denoising with axis-selective options.
- **Dev**: embedded developer terminal.
- **Defs**: short reference notes for processing operations.

Preferences such as theme, last folders, clip values, and font size are stored
in `~/.probeflow_config.json`.

## ROI Manager Status

ROI support is now a first-class part of the processing and GUI model, but it is
still being hardened.

Supported drawing tools:

- rectangle
- ellipse
- polygon
- freehand
- line
- point

Implemented ROI-aware paths include:

- line profile from a line ROI;
- ROI histogram;
- ROI FFT spectrum/crop;
- ROI background subtraction fit and exclusion regions;
- ROI sidecars loaded by both GUI and CLI named ROI lookup;
- ROI transform updates for lossless flip/90-degree rotation;
- ROI invalidation for arbitrary-angle rotation.

Current limitations:

- ROIs are stored in pixel coordinates, not physical coordinates.
- The GUI stores one `.rois.json` sidecar per scan file. Channels/planes that
  share the same displayed pixel frame currently share that ROISet.
- Geometric display operations transform the ROISet in the displayed frame.
  Arbitrary-angle rotation removes existing ROIs because exact transformed
  masks would be misleading.
- Session-level ROI persistence does not exist yet; the `.rois.json` sidecar is
  the durable store for now.
- Composite/inverted ROIs are supported through polygon/multipolygon geometry,
  but the UI for complex ROI algebra is intentionally light.

### ROI Coordinate Rules

ROI coordinates use image pixels with `(x, y) = (column, row)` and origin at the
top-left of the displayed image. Masks are generated with array shape
`(Ny, Nx)` and applied as `array[row, col]`.

The sidecar path for a scan is:

```text
<scan-stem>.rois.json
```

CLI named ROI lookup checks that GUI sidecar first, then falls back to a
matching `.provenance.json` sidecar if needed.

## Provenance

PNG/JSON-style exports can include a provenance sidecar recording:

- source file and format;
- selected channel;
- array shape and scan range;
- display state;
- processing state;
- ProbeFlow version;
- warnings;
- ROISet data when available.

This is not yet a complete lab notebook or graph-backed session model. It is a
linear export record intended to make routine image preparation auditable.

## Python Use

ProbeFlow can be used without launching the GUI:

```python
from probeflow import load_scan, processing

scan = load_scan("scan.dat")
scan.planes[0] = processing.align_rows(scan.planes[0], method="median")
scan.planes[0] = processing.subtract_background(scan.planes[0], order=1)
scan.save("processed.sxm")
scan.save("processed.png", colormap="gray")
```

Spectroscopy:

```python
from probeflow.io.spectroscopy import read_spec_file
from probeflow.processing.spectroscopy import smooth_spectrum, numeric_derivative

spec = read_spec_file("spectrum.VERT")
z_smooth = smooth_spectrum(spec.channels["Z"], method="savgol")
dzdv = numeric_derivative(spec.x_array, z_smooth)
```

## Repository Layout

```text
probeflow/
|-- core/        # Scan model, loading dispatch, metadata, ROI, validation
|-- io/          # File sniffing, readers, writers, converters, sidecar helpers
|-- processing/  # GUI-free numerical processing and ProcessingState
|-- analysis/    # Particles, lattice, spectroscopy plotting, feature tools
|-- provenance/  # Export provenance and tested ScanGraph dataclasses
|-- gui/         # PySide6 GUI package
|-- cli/         # Command-line interface
`-- plugins/     # Plugin API/registry foundation

tests/           # pytest suite
test_data/       # sample/manual input data
assets/          # logo artwork
```

### Current Architecture Reality

The package tree shows the intended decomposition.  The bulk of the main window
and CLI parser remain in the two `_legacy` files, but significant extraction has
already happened:

| Area | Extracted to |
|---|---|
| Browse cards, thumbnail grid | `gui/browse/` |
| Background workers | `gui/workers/` |
| File-list models | `gui/models/` |
| Developer terminal | `gui/terminal/` |
| Dialogs (FFT, ROI, spec, about…) | `gui/dialogs/` |
| Viewer controllers and panels | `gui/viewer/` |
| Colormap rendering helpers | `gui/rendering/` |
| Image canvas and ROI graphics items | `gui/image_canvas/`, `gui/roi_items/` |
| Feature-counting and TV-denoise panels | `gui/features/` |
| Processing control panel | `gui/processing/` |

What remains in `gui/_legacy.py`: `ProbeFlowWindow`, `Navbar`,
`BrowseInfoPanel`, `BrowseToolPanel`, `ConvertPanel`, `ConvertSidebar`, and
config helpers (`load_config`, `save_config`).

The `_legacy` suffix is historical. These files are not deprecated. Cleanup is
happening opportunistically: extract one class or helper when a real feature or
bug fix already touches it.

CLI command runners are moving to `probeflow/cli/commands/` submodules as they
are touched.  Until a runner is fully moved the submodule re-exports it from
`_legacy`; see `cli/commands/__init__.py` for the migration protocol.

## Testing And Development

```bash
python -m pip install -e ".[dev,features]"
pytest
ruff check probeflow tests
```

The repository includes GitHub Actions for pytest on Python 3.11/3.12 and a
permissive ruff check. Pre-commit hooks are available:

```bash
pip install pre-commit
pre-commit install
```

Dead-code checks (install `vulture` via `pip install -e ".[dev]"`):

```bash
vulture probeflow/ tests/ whitelist.py --min-confidence 80
python scripts/find_orphan_modules.py
```

`whitelist.py` at the repo root suppresses known false positives (Qt override
methods, plugin API entry points). `scripts/find_orphan_modules.py` reports
`.py` files that no other file imports. `docs/dead_code_audit.md` documents
the current findings and per-symbol verdicts.

## Notes

- Failed batch conversions are logged to `errors.json`.
- The `.sxm` writer uses a reference Nanonis byte layout captured under
  `src/file_cushions/`; converted files should still be checked in target
  software when scientific output depends on interoperability.
- Sidecar formats are intentionally JSON and version-light while the project is
  beta. Avoid treating them as a permanent project file format.

## Acknowledgements

ProbeFlow is developed at [SPMQT-Lab](https://github.com/SPMQT-Lab) at The
University of Queensland.

The original Createc-decoding work was written by
[Rohan Platts](https://github.com/rohanplatts). ProbeFlow builds on that
foundation with browsing, conversion, processing, ROI workflows, spectroscopy
handling, and export provenance.
