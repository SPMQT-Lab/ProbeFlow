# Provenance Ownership and Compatibility

This document defines the provenance contract used during the issue #52
refactor. It describes software structure, not scientific validation.

## Canonical hierarchy

```text
core.processing_state.ProcessingState
        |
        v
provenance.records.ProcessingHistory
        |
        v
provenance.records.ExportRecord
```

`ProcessingState` is the replayable numerical recipe attached to `Scan`.
`ProcessingHistory` adds the source, timestamps, warnings, versions, and state
identifiers. `ExportRecord` adds artifact, display, ROI, mask, and export
information. Only `ExportRecord` is the current `.probeflow.json` sidecar
schema.

## Compatibility models

| Model | Current purpose | Canonical source |
|---|---|---|
| `Scan.processing_history` | Older list-of-dictionaries API | `Scan.processing_state` plus retained timestamps |
| `processing.history` | Converts the old list API to and from `ProcessingState` | `ProcessingState` |
| `provenance.export.ExportProvenance` | Older writer-facing object and `.provenance.json` payload | An `ExportRecord` when available |
| `provenance.records.ProcessingStep` | Deprecated class name | Alias of `ProvenanceStep` |

Compatibility models remain readable and callable during the JOSS cycle. New
code must not treat them as independent stores.

## Field ownership

| Information | Canonical location | Compatibility projection |
|---|---|---|
| Ordered numerical operations and parameters | `ProcessingState.steps` | `Scan.processing_history`, `ExportProvenance.processing_state` |
| Source path, format, channel, loader, metadata, file hash | `ProcessingHistory.source_record` | Flat `ExportProvenance` source and channel fields |
| Step timestamps, versions, warnings, state IDs | `ProcessingHistory.steps` | `ExportProvenance.processing_history` |
| Export path, format, time, warnings, summary | `ExportRecord` | Flat `ExportProvenance` export fields |
| Display range and rendering settings | `ExportRecord.display_settings` | `ExportProvenance.display_state` |
| ROIs and masks | `ExportRecord.rois` and `ExportRecord.masks` | Flat `ExportProvenance` fields |
| Legacy writer identity hashes | `ExportProvenance` only | Retained for compatibility; not duplicated into the current record |

Paths are passed through `core.source_identity` before serialization. Sidecars
must not expose machine-specific absolute paths or user names.

## Construction paths

The canonical construction path is:

```text
Scan + ProcessingState
  -> processing_history_from_scan
  -> build_export_record
  -> write_provenance_sidecars
```

`build_scan_export_provenance` wraps the resulting history and record in an
`ExportProvenance` adapter for writers that still need the older shape.
`export_record_dict_from_provenance` is the single compatibility gateway used
before writing `.probeflow.json`.

Direct calls to `ExportProvenance.from_scan_export` must also project their
`processing_state` into the current record. Otherwise the legacy sidecar would
contain the recipe while the current sidecar silently omitted it.

## Sidecar contracts

- `<artifact>.probeflow.json` contains a versioned `ExportRecord` and is the
  reliable current record.
- `<artifact>.provenance.json` contains the legacy `ExportProvenance` shape when
  a compatibility object is available.
- Existing names, discovery order, collision checks, corrupt-file behaviour,
  privacy filtering, and atomic replacement remain unchanged.
- Existing sidecars remain readable. Schema migration is additive and explicit;
  unknown scientific meaning must not be guessed.
- ROI and mask lookup must continue to find data embedded in the current
  sidecar.

## Refactor invariants

- Converting to `ExportRecord` must preserve processing steps, source identity,
  display settings, warnings, ROIs, and masks.
- Conversion must not mutate the source `Scan`, history, or compatibility
  object.
- Writers may override the final output path and format, but no other field.
- Timestamps are metadata and must not alter replayed numerical results.
- Writer migration must not alter artifact bytes except where the format already
  embeds a human-readable provenance comment.
- Automated round trips prove field preservation, not scientific correctness.
