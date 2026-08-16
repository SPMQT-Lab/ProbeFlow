# ProbeFlow Architecture

## Purpose and boundary

ProbeFlow is post-acquisition STM/SPM software. It browses microscope files,
loads calibrated data, applies processing, measures images, and exports derived
artifacts. It does not control instruments. Raw microscope files are read-only;
all changes are held in memory or written to new files.

This document maps the current implementation. It does not certify scientific
correctness. Proposed refactors are listed in [`proposal.md`](../proposal.md).

## System map

```text
Desktop GUI                 Command line                 Python API
    |                            |                           |
    +---------------- interface/orchestration ----------------+
                                 |
        +------------------------+------------------------+
        |                        |                        |
   core domain             scientific code          provenance
 Scan, state, ROI,       processing, analysis,     history, records,
 mask, metadata, index   measurements, spectra     export metadata
        |                        |                        |
        +------------------------+------------------------+
                                 |
                         I/O and sidecars
                 vendor readers, writers, converters
```

The diagram is conceptual. Workflow orchestration currently exists in several
packages rather than in one application layer.

## Package ownership

| Package | Current responsibility |
|---|---|
| `probeflow` | Version and small public, vendor-neutral API. |
| `core` | `Scan`, `ProcessingState`, ROIs, masks, validation, format sniffing, metadata, and folder indexing. |
| `io` | Vendor readers, writers, converters, format byte layouts, spectroscopy file decoding, and JSON sidecars. |
| `processing` | Qt-free image transformations, display preparation, GUI-state translation, and pipeline replay. |
| `measurements` | Measurement result models, calibrated measurement kernels, feature sets, and tabular export. |
| `analysis` | Higher-level feature, grain, periodicity, lattice, point-pattern, and plotting routines. |
| `spectroscopy` | Display-oriented spectrum models and transformations after file decoding. |
| `provenance` | Processing-history records, export records, sidecar construction, and replay metadata. |
| `gui` | PySide6 windows, dialogs, canvases, controllers, rendering adapters, and background workers. |
| `cli` | Argument parsing and command orchestration over the other packages. |
| `data` | Packaged byte fixtures used by the Nanonis-compatible SXM writer. |

`core`, `io`, `processing`, `measurements`, `analysis`, `spectroscopy`, and
`provenance` are intended to remain Qt-free. The GUI may depend on all backend
packages. The CLI may depend on backend packages and imports the GUI only for
the `gui` command.

### Actual dependency exceptions

The current backend is not a strict acyclic layer stack:

- `core` calls `io` lazily for loading, metadata, and `Scan.save_*`, and calls
  `processing.history` to expose compatibility history.
- `io` depends on `core` models and provenance; rendered writers also use
  processing display/export helpers.
- `processing` depends on `core`; compatibility shims forward to `analysis`
  and `spectroscopy`, while PNG/PDF export helpers call `io` and provenance.
- `measurements` uses core identity helpers and spectroscopy models;
  `analysis` uses measurement models and some I/O helpers.
- `provenance` uses core and processing models; `prepared_export` also invokes
  the PNG writer.

These are existing seams, not the desired extension API. Some function-local
imports prevent runtime cycles and defer heavy dependencies.

## Domain models

### Images

`core.scan_model.Scan` is the common image model. It contains:

- equally shaped 2-D NumPy planes in display orientation;
- parallel channel names, units, and synthetic-channel flags;
- raw vendor headers and interpreted experiment metadata;
- physical `(width_m, height_m)` in `scan_range_m`;
- source path, source format, reader warnings, and processing state.

Readers normally convert known physical channels to SI units. `validate_scan`
checks plane shapes, names, units, finite data, source format, and selected
vendor-specific invariants immediately after full loading.

`ProcessingState` and `ProcessingStep` live in `core.processing_state` so
`Scan` can own processing state without importing numerical kernels. The
historical import from `processing.state` is retained as a re-export.

### Regions and masks

`ROI`/`ROISet` and `ImageMask`/`MaskSet` live in `core`. They hold geometry or
raster selections independently of Qt. GUI graphics items are disposable views
of these models. Geometry operations update or invalidate selections through
the common operation vocabulary in `core.op_vocab`.

### Browsing

`ScanMetadata` is a lightweight scan summary. `ProbeFlowItem` is the common
folder-index result for either a scan or spectrum. Neither contains full image
or spectrum arrays. The GUI adapts these into the older `SxmFile` and `VertFile`
view models, even when the source is RHK or Nanonis spectroscopy.

### Measurements and spectra

`measurements.models.MeasurementResult` is the common quantitative result.
Feature points and feature-set storage also live in `measurements`.

Decoded spectroscopy uses `io.spectroscopy.SpecData` and `SpecMetadata`.
Display processing uses `spectroscopy.SpectrumTrace`, `DisplayedSpectrum`, and
`SpectrumDisplayOptions`. This image/spectrum split is functional but not a
single unified domain model.

## File identification and loading

Files are identified by content because `.dat` is shared by Createc images and
Nanonis spectra.

| Input | Content identity | Full reader | Index identity |
|---|---|---|---|
| Createc `.dat` image | `CREATEC_IMAGE` | `io.readers.createc_scan` | `createc_dat` |
| Nanonis `.sxm` image | `NANONIS_IMAGE` | `io.readers.nanonis_sxm` | `nanonis_sxm` |
| RHK `.sm4` image pages | `RHK_SM4_IMAGE` | `io.readers.rhk_sm4` | `rhk_sm4` |
| Createc `.VERT` spectrum | `CREATEC_SPEC` | `io.spectroscopy` / `io.readers.createc_vert` | `createc_vert` |
| Nanonis `.dat` spectrum | `NANONIS_SPEC` | `io.readers.nanonis_spec` | `nanonis_dat_spectrum` |

Image loading follows:

```text
path -> sniff_file_type -> identify_scan_file -> LoadSignature
     -> load_scan_from_signature -> vendor reader -> validate_scan -> Scan
```

`core.file_type` owns signatures and `FileType`. `core.loaders` converts those
to `LoadSignature`. `core.scan_loader` dispatches to readers. Metadata dispatch
is separately implemented in `core.metadata`; spectroscopy has another
dispatcher in `io.spectroscopy`.

RHK can decode a selected image page for thumbnails. Createc and Nanonis image
thumbnails currently use a full scan decode.

## Browse workflow

```text
ProbeFlowWindow
  -> FolderIndexLoader
  -> index_folder_shallow / index_folder
  -> content sniff + lightweight vendor metadata
  -> ProbeFlowItem + browse cache
  -> SxmFile / VertFile GUI adapter
  -> thumbnail worker -> render_scan_thumbnail -> card grid
```

Indexing uses bounded directory peeks and thread pools. GUI work uses
`QThreadPool` workers and signal objects. `_PooledWorker` owns the shared Qt
object-lifetime pattern. Cache entries are invalidated by file metadata and a
schema tag. Filtering and sorting are Qt-free where practical.

## Processing workflow

Numerical kernels are grouped under `processing` by purpose: alignment,
background correction, bad-line repair, filters, FFT operations, geometry,
masks, arithmetic, repair, display conversion, and experimental TV processing.

```text
GUI controls or CLI syntax
  -> ProcessingStep list
  -> apply_processing_state (array-only)
     or apply_processing_state_with_calibration (array + physical range)
  -> numerical kernels -> processed result
```

The GUI stores a dictionary-shaped control state. `processing.gui_adapter`
translates it into canonical steps. `processing.state` executes those steps in
an explicit dispatcher and resolves nested ROI/mask scopes. Shape-changing
operations pass through the calibration-aware wrapper so the physical extent
can follow the output array.

Display settings such as colourmap and contrast are separate from numerical
processing. The workflow-replay tests compare display, export, and replay for
representative pipelines.

## Measurements and analysis

Simple calibrated measurements use `measurements` kernels: ROI statistics,
step heights, line profiles, spectrum deltas, feature points, and table export.
`analysis` builds higher-level workflows such as lattice extraction,
periodicity, grains, and pair correlation. Some historical helpers remain as
thin compatibility wrappers between these packages and `processing`.

The GUI supplies selected arrays, pixel sizes, and ROI/mask models to these
backend functions, then displays `MeasurementResult` or specialised result
objects. The CLI invokes many of the same kernels through command-specific
orchestration.

## GUI architecture

`gui.app.ProbeFlowWindow` is the main folder browser. Conversion and TV tools
open as separate workspace windows. Image and spectrum files open modeless
viewers retained by explicit strong-reference registries.

`ImageViewerDialog` is composed from build, chrome, display, ROI, mask,
selection, toolbar, tools, and processing/export mixins. These mixins share
state through attributes on the dialog. Focused controllers handle some
concerns, including display ranges, undo, zero-plane selection, overlays, and
deferred actions. `ImageCanvas` owns Qt graphics interaction; core ROI and mask
objects remain the persisted source of truth.

`FFTViewerDialog` uses separate mixins for lattice, mains suppression, inverse
FFT, and symmetrisation. The interactive lattice grid has its own controller,
graphics item, FFT panel, real-space panel, and stored-grid model.

`gui.workers` contains indexing, thumbnail, viewer, preview, conversion, and
filtered-export tasks. Worker results cross back to widgets through Qt signals;
tokens prevent stale results from replacing newer state.

The GUI source is grouped as follows:

| Area | Contents |
|---|---|
| `browse` | Breadcrumbs, cards, filter panels, and thumbnail grid. |
| `convert` | Conversion workspace and sidebar. |
| `dialogs` | Image/FFT/spectrum viewers and task dialogs. |
| `viewer` | Image-viewer controllers, mixins, export helpers, and commands. |
| `lattice_grid` | Interactive lattice model, controller, graphics, and panels. |
| `spec_viewer` | Single, overlay, and shared spectrum-viewer code. |
| `widgets` | Reusable measurement and feature panels. |

## CLI architecture

`cli.parser` defines commands and dispatches to `cli.commands`. Shared
processing syntax and helpers live in `cli.processing_ops`. Commands load data,
call processing or analysis functions, construct provenance, and write outputs.
The CLI is an adapter, but some workflows are duplicated with Qt-free helpers
under `gui`.

Console entry points are `probeflow`, `dat-sxm`, `dat-png`, and `dat-npy`.
`probeflow-gui` is the desktop entry point.

## Export and provenance

Writers under `io.writers` support SXM, GWY, PNG, PDF, and CSV. JSON helpers and
Createc-to-NPY conversion provide other structured outputs. Writers refuse to
overwrite raw input and normally refuse output collisions unless explicitly
allowed.

The current provenance path contains several related models:

- `Scan.processing_state`: canonical numerical steps attached to a scan;
- `Scan.processing_history`: list-shaped compatibility view of those steps;
- `ProcessingHistory` and `ExportRecord`: timestamped production records;
- `ExportProvenance`: artifact-oriented metadata used by writers.

Exports can write `.probeflow.json` and legacy `.provenance.json` sidecars.
ROIs use `.rois.json`; masks use `.masks.json`. ROI and mask loaders also search
provenance sidecars. Sidecar writes use a sibling temporary file followed by an
atomic replace. Corrupt preferred sidecars raise instead of silently loading a
stale fallback.

The processed-image GUI path reloads the source `Scan`, inserts the displayed
processed plane, attaches processing state, builds provenance, and delegates to
the relevant writer. Raw source files are not modified.

## Public API and compatibility

The root package explicitly exports `Scan`, loaders, metadata, indexing, and
version information without importing Qt. Historical APIs remain through:

- `processing.image` and `processing.__init__` wildcard re-exports;
- `processing.analysis` and `processing.spectroscopy` forwarding shims;
- `gui.compat` and lazy exports in `gui.__init__`;
- `cli._legacy` and the proxy behaviour in `cli.__init__`;
- selected measurement and analysis adapters.

New internal code should import the canonical defining module. The shims exist
for compatibility and are not extension points.

## Current extension seams

There is no third-party plugin loader. Optional OpenCV/scikit-learn lattice
support and `gwyfile` export are dependency extras loaded lazily. New formats
currently require edits to sniffing, loading, metadata, indexing, thumbnail,
GUI mapping, tests, and documentation. New processing operations require edits
to the supported-operation set, dispatcher, parameter translation, provenance
handling, tests, and any interface that exposes them.

Viewer commands are centrally described in `gui.viewer.shortcuts`, but command
registration is internal to the GUI.

## Deliberate exceptions and invariants

- `Scan.save_*` imports writers lazily to avoid cycles and heavy imports.
- The processing dispatcher remains explicit because operations have different
  context, ROI, mask, operand, shape, and calibration needs.
- Backend imports must not load PySide6.
- Processing and display state remain separate.
- All supported image planes in one `Scan` have the same shape.
- Reader warnings stay attached to the `Scan` and are surfaced by the viewer.
- Raw files remain read-only; exports are new artifacts with provenance.
- Scientific correctness requires human validation beyond automated tests.

## Architecture tests and maintenance

Repository support code is separated from the installed package:

| Path | Responsibility |
|---|---|
| `tests` | Unit, integration, workflow, GUI, format, and release tests. |
| `test_data` | Small real Createc, Nanonis, and RHK fixtures. |
| `scripts` | Packaging, validation, screenshots, licensing, indexing, and adversarial-render utilities. |
| `packaging` | PyInstaller, macOS, Windows, icons, constraints, and licence material. |
| `.github/workflows` | CI, dependency canary, and Windows release build. |
| `docs` | User guides, format notes, maintenance records, and architecture decisions. |
| `scope.md` | Agreed narrow JOSS publication scope. |
| `proposal.md` | Prioritised architecture-refactor prompts. |

`tests/test_layout_compatibility.py` checks canonical import locations,
Qt-free backend imports, and compatibility surfaces.
`tests/test_pipeline_connectivity.py` checks operation dispatch and selected
import boundaries. `tests/test_architecture_boundaries.py` statically enforces
the package rules and confines each existing back-edge to the files declared in
`tests/architecture_policy.py`. Reader, writer, calibration, provenance,
replay, GUI, and packaging tests cover their respective seams.

CI runs tests on Python 3.11 and 3.12 plus linting; a weekly canary checks newer
dependencies. Any accepted boundary change must update this document,
`CONTRIBUTING.md`, relevant user documentation, and architecture tests in the
same pull request.
