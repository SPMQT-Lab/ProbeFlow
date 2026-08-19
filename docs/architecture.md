# ProbeFlow Architecture

A document outlining how the code for ProbeFlow is structured, what the fundamental objects the code uses are, and what rules / boundaries have governed the development of the current architecture. 

## Purpose

ProbeFlow is post-acquisition STM/SPM software. It browses data, loads
calibrated scans and spectra, processes and measures them, and exports derived
artifacts. It does not control instruments. Raw microscope files are read-only.

## System map

```text
GUI / CLI / Python API
        |
        +-- workflows (shared application orchestration)
        |
        +-- core -------- Scan, state, ROIs, masks, formats, metadata, index
        +-- processing -- numerical kernels and replay
        +-- provenance -- history, export records, sidecars
        +-- io ---------- readers, writers, converters, byte layouts
        +-- analysis / measurements / spectroscopy
```

`gui` and `cli` collect input and present results. Backend packages, including
`workflows`, must remain Qt-free.

## Ownership

| Area | Owns |
|---|---|
| `core` | `Scan`, `ProcessingState`, ROI/mask models, validation, formats, metadata, and indexing. |
| `io` | Vendor decoding, file encoding, conversion, and sidecars. |
| `processing` | Numerical image operations, replay, display preparation, and GUI-state translation. |
| `provenance` | Source records, processing history, export records, and schema migration. |
| `workflows` | Shared Qt-free orchestration over backend models, provenance, and writers. |
| `measurements`, `analysis`, `spectroscopy` | Quantitative kernels and higher-level scientific workflows. |
| `gui`, `cli` | Interface adapters; not owners of backend rules. |

`Scan` is the common image model: equally shaped display-oriented planes,
channel metadata, calibrated physical range, source identity, warnings, and
processing state. GUI graphics are views over the persisted `core` ROI and mask
models.

## Fundamental objects

| Object | Role | Owned by |
|---|---|---|
| `Scan` | One loaded image dataset: planes, channel metadata, calibration, source identity, warnings, and processing state. | `core.scan_model` |
| `ProcessingState` / `ProcessingStep` | Ordered, serialisable numerical recipe. A step is an operation ID plus sparse parameters. | `core.processing_state` |
| `ROI` / `ROISet`; `ImageMask` / `MaskSet` | Persisted geometric and raster selections. | `core.roi`, `core.mask` |
| `FileType`, `FormatDefinition`, `FormatCatalog`, `LoadSignature` | Content identification and the route from a path to a reader. | `core.formats`, `core.loaders` |
| `ScanMetadata` / `ProbeFlowItem` | Lightweight metadata and folder-browser records; neither holds image arrays. | `core.metadata`, `core.indexing` |
| `ProcessingHistory` / `ProvenanceStep` / `ExportRecord` | Source context, ordered history, warnings, and the persisted export record. | `provenance.records` |
| `ProcessedExportRequest` / `ProcessedExportResult` | Input and result for shared processed-image export. | `workflows` |
| `MeasurementResult` / `FeaturePoint` | Backend measurement output and detected features. | `measurements.models` |

The ownership rule is simple: `core` holds durable domain data, `processing`
changes arrays, `provenance` describes artifacts, `io` reads or writes bytes,
and `workflows` composes those capabilities.

## Supported input structures

Formats are declared once in `core.formats` and resolved by sniffing, loading,
metadata, indexing, thumbnails, and GUI adapters.

| Structure | Canonical ID | Reader |
|---|---|---|
| Createc `.dat` image | `createc_dat` | `io.readers.createc_scan` |
| Nanonis `.sxm` image | `nanonis_sxm` | `io.readers.nanonis_sxm` |
| RHK `.sm4` image pages | `rhk_sm4` | `io.readers.rhk_sm4` |
| Createc `.VERT` spectrum | `createc_vert` | `io.readers.createc_vert` |
| Nanonis `.dat` spectrum | `nanonis_dat_spectrum` | `io.readers.nanonis_spec` |

`.dat` is identified by content because it is shared by Createc images and
Nanonis spectra. Reader imports are lazy; vendor decoding remains in `io`.

## Construction paths

### Load

```text
path -> sniff_file_type -> identify_scan_file -> LoadSignature
     -> load_scan_from_signature -> vendor reader -> validate_scan -> Scan
```

`FileType` is a public classification. `FormatDefinition` supplies the stable
format ID, aliases, capabilities, and reader binding. `LoadSignature` records
the selected route without loading the full file.

### Process and calibrate

`core.operations` defines the 36 supported operations without numerical
callables. A contract records canonical IDs, aliases, display names, defaults,
parameter roles, scope, calibration inputs, shape/range policy, and handler
keys.

```text
GUI controls or CLI syntax
  -> ProcessingState
  -> processing.state replay
  -> numerical kernels
  -> calibrated processed result
```

GUI and CLI adapters, replay defaults, ROI/mask rules, aliases, calibration
updates, and provenance labels use the same catalog. The explicit dispatcher
remains deliberate: operations have different inputs and scope semantics.
Display settings are separate from numerical processing.

### Record and export

- `ProcessingState` is the numerical recipe attached to a `Scan`.
- `ProcessingHistory` owns source context, timestamps, and warnings.
- `ExportRecord` is the canonical persisted export-sidecar record.
- `Scan.processing_history` and `ExportProvenance` are compatibility views.

`workflows.processed_export` accepts a scan, processed plane, calibrated range,
state, selections, warnings, destination, and overwrite policy. It builds
provenance once and calls an existing writer. Prepared PNG, CLI processing
output, and viewer processed-image export use this workflow.

Writers support SXM, GWY, PNG, PDF, and CSV. They protect raw sources and
existing outputs by default. Sidecars are written atomically.

```text
Scan + processed plane + state + display/selections
  -> ProcessedExportRequest
  -> workflow provenance builder
  -> existing writer
  -> artifact + ExportRecord sidecar
```

## Interfaces

`ProbeFlowWindow` is the folder browser. Image and spectrum viewers are
modeless. `ImageViewerDialog` is composed from focused mixins and controllers;
its shared dialog state is a known future simplification target.

`cli.parser` dispatches commands to `cli.commands`; shared processing syntax
lives in `cli.processing_ops`. The `gui` command is the only permitted
CLI-to-GUI dependency.

## Boundaries and exceptions

New code must follow these directions:

- `core` has no GUI or CLI dependency.
- `io`, `processing`, `provenance`, and scientific packages have no GUI or CLI
  dependency.
- `workflows` has no PySide6, widget, dialog, or CLI-parser dependency.
- Interfaces call backend APIs; they do not become the source of backend rules.

Some compatibility seams remain and are explicitly limited by
`tests/architecture_policy.py`: lazy `core -> io` facades; low-level PNG/PDF
renderer crossings; historical analysis and spectroscopy shims; version lookup;
the prepared-export compatibility facade; and the explicit CLI GUI launcher.
They are not extension points.

## Invariants and maintenance

- Raw files are never modified.
- Backend imports do not load PySide6.
- All planes in a `Scan` share a shape.
- Reader warnings stay attached to the `Scan`.
- New operations require one catalog entry, an explicit executor binding,
  scientific tests, and deliberate interface exposure.
- There is no third-party plugin loader. New formats and scientific features
  require separate approval.
- Automated tests do not establish scientific correctness; human validation is
  required before publication.

`tests/test_architecture_boundaries.py` enforces import ownership and named
exceptions. `tests/test_layout_compatibility.py` protects public import paths
and Qt-free backend imports. Update this document, the policy, and relevant
tests whenever an accepted boundary changes.
