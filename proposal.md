# ProbeFlow Architecture Proposal

This proposal explains the current architecture and the smallest useful
refactors. The detailed module inventory is in
[`docs/architecture.md`](docs/architecture.md).

This is an architecture audit, not scientific validation. Existing tests show
that code behaves as encoded; they do not independently prove calibration,
processing, or measurement correctness.

## Current state

ProbeFlow is post-acquisition STM/SPM software with three entry points:

- `probeflow.gui` provides the folder browser, thumbnails, image and spectrum
  viewers, processing controls, measurements, and exports.
- `probeflow.cli` provides command-line versions of many workflows.
- the root `probeflow` package exposes a small Python API.

The main data path is:

```text
GUI / CLI / Python API
          |
          v
core: identify, index, model, validate
          |
          v
io: vendor readers and writers
          |
          v
Scan + calibrated arrays
          |
          +----------+-------------+
          v          v             v
     processing  measurements   analysis / spectroscopy
          |          |             |
          +----------+-------------+
                     v
              provenance + export
```

In practice, the arrows also run backwards and sideways:

```text
gui ----------> core, io, processing, measurements, analysis,
 |              spectroscopy, provenance
 |
 +-- contains some Qt-free workflow logic that CLI-like callers could use

cli ----------> all backend packages
 +------------> gui, but only to launch the desktop application

core ---------> io through lazy imports for loading, metadata, and Scan.save_*
 +------------> processing.history for an older history interface

io -----------> core models
 +------------> processing display helpers and provenance during export

processing ---> core models
 +------------> io and provenance in its export helpers
 +------------> analysis and spectroscopy through compatibility modules

provenance ---> core and processing models
 +------------> io from prepared_export

analysis -----> measurements and some io helpers
measurements -> core and spectroscopy models
```

The result is a **dependency web**, not a strict sequence of layers. Lazy
imports prevent some runtime cycles, but they do not establish clear ownership.

### Key areas

- `core.scan_model.Scan` is the common image object. `core.processing_state`,
  `core.roi`, and `core.mask` hold Qt-free state.
- File recognition and loading are divided across `core.file_type`,
  `core.loaders`, `core.scan_loader`, `core.metadata`, `core.indexing`, and
  vendor implementations under `io.readers` and `io.spectroscopy`.
- Processing definitions are divided across `core.op_vocab`,
  `core.processing_state`, `processing.gui_adapter`, `processing.state`, GUI
  controls, and CLI helpers. Numerical kernels live under `processing`.
- Quantitative functions are spread across `measurements`, `analysis`, and
  `spectroscopy`. Their present separation works, but ownership is not always
  obvious from package names alone.
- Export is shared by `io.writers`, `provenance`, `processing` export helpers,
  `gui.viewer`, and CLI commands.
- `gui.dialogs.image_viewer.ImageViewerDialog` is assembled from many mixins.
  The mixins communicate through attributes on the dialog rather than through
  one explicit viewer-state object.
- Compatibility modules in `processing`, `gui`, and `cli` preserve older import
  paths. They make the dependency graph look broader than the intended API.

## Why the current shape is risky

- Adding one format requires coordinated edits to identification, loading,
  metadata, indexing, thumbnails, GUI mapping, tests, and documentation.
- Adding one operation requires coordinated edits to its vocabulary,
  dispatcher, parameter conversion, provenance, GUI, CLI, and tests.
- Several provenance objects can describe the same export, so fields can be
  dropped or disagree during conversion.
- Similar export workflows exist in different packages, so GUI and CLI results
  can drift even when they use the same numerical kernels.
- Shared, undocumented GUI attributes make state transitions hard to test
  without creating the complete Qt widget tree.
- Import tests cover selected boundaries, but most intended package boundaries
  are conventions only. A new dependency can silently make the web worse.

This is not evidence that current scientific results are wrong. It means the
code is harder to review, validate, extend, and keep consistent than necessary.

## Proposed changes

### 1. Enforce package boundaries

**Change:** Add automated architecture tests.

**What it is:** Turn the ownership rules in `docs/architecture.md` into import
tests. Record each required lazy import or compatibility exception in a small,
named allow-list.

**Why the current state is bad:** Current tests check that backend imports do
not load Qt and cover a few public import paths. They do not prevent `core`,
`io`, `processing`, or `provenance` from gaining new cross-dependencies.

**If implemented:** CI will reject an unapproved dependency, such as `core`
importing GUI code or a processing kernel taking ownership of a vendor parser.
Accepted exceptions will be visible and reviewable.

**Implementation prompt:** Add tests that inspect imports under `probeflow/`
and enforce the rules in `docs/architecture.md`. Begin with the current graph,
explicitly allow documented exceptions, and tighten the allow-list only when a
later refactor removes an exception. Do not alter production behaviour.

### 2. Establish one provenance hierarchy

**Change:** Give each provenance model one distinct role.

**What it is:** Use one direction:

```text
ProcessingState -> ProcessingHistory -> ExportRecord
```

- `ProcessingState`: replayable numerical steps.
- `ProcessingHistory`: the steps plus source identity, time, and warnings.
- `ExportRecord`: a versioned sidecar with history, display settings, ROIs,
  masks, and artifact details.

`Scan.processing_history` and `ExportProvenance` should remain compatibility
views, not separate sources of truth.

**Why the current state is bad:** `Scan.processing_state`,
`Scan.processing_history`, `ProcessingHistory`, `ExportRecord`, and
`ExportProvenance` overlap. Passing between them can omit timestamps, warnings,
source identity, display settings, ROIs, or masks. Writers do not all begin from
one authoritative record.

**If implemented:** Every writer will receive the same complete export record.
One builder will create sidecars, and documented migrations will keep existing
sidecars readable.

**Implementation prompt:** Inventory every provenance construction, read, and
write path. Make `ProcessingState`, `ProcessingHistory`, and `ExportRecord` the
canonical hierarchy. Convert other models into compatibility adapters. Add
golden sidecar, round-trip, and replay tests before migrating writers. Do not
remove support for old sidecars during the JOSS cycle.

### 3. Define each supported format once

**Change:** Add one typed `FormatDefinition` for each existing format.

**What it is:** A definition should contain one stable ID, image or spectrum
kind, suffixes, content signature, metadata reader, full reader, thumbnail
reader, and supported export routes. Vendor byte decoding stays in `io`.

**Why the current state is bad:** Format knowledge is repeated in
`core.file_type`, `core.loaders`, `core.scan_loader`, `core.metadata`,
`core.indexing`, `io.spectroscopy`, `gui.models`, and `gui.rendering`. The code
also uses both short and vendor-qualified IDs for the same formats. A partial
update can make a file load in one workflow but disappear or fail in another.

**If implemented:** Createc, Nanonis, and RHK capabilities will each be declared
once. Browsing, loading, thumbnails, and interfaces will query the same
definition instead of maintaining parallel dispatch tables.

**Implementation prompt:** Introduce a typed registry for the five currently
supported input structures. Migrate one format at a time and compare detection,
metadata, arrays, warnings, calibration, thumbnails, and error behaviour. Do
not add formats or a third-party plugin system.

### 4. Define each processing operation once

**Change:** Add one typed `OperationSpec` for each existing operation.

**What it is:** Each specification should define its operation ID, kernel
version, parameter schema and defaults, ROI and mask rules, and effect on array
shape and physical calibration.

**Why the current state is bad:** These facts are split across
`core.op_vocab`, `core.processing_state`, `processing.gui_adapter`,
`processing.state`, CLI helpers, GUI controls, and tests. `ProcessingStep`
checks the operation name but does not fully validate its parameters. An
interface can therefore encode an operation differently from replay.

**If implemented:** GUI, CLI, validation, replay, and provenance will use the
same contract. Invalid state will fail before a numerical kernel is called.

**Implementation prompt:** Add operation specifications around the existing
dispatcher. Migrate one operation at a time. Compare arrays, calibration,
warnings, defaults, ROI/mask behaviour, and serialized state. Do not change a
kernel, default, or scientific result.

### 5. Centralise processed-image export

**Change:** Add a Qt-free processed-export workflow.

**What it is:** A service under `probeflow.workflows` should receive a source
scan, processed array, calibrated range, processing and display state, ROIs,
masks, output path, and overwrite policy. It should construct provenance and
call the selected writer.

**Why the current state is bad:** Export orchestration is divided among CLI
commands, `provenance.prepared_export`, processing export helpers, and
`gui.viewer.processed_export`. Each path can make different choices about
calibration, provenance, warnings, or output collisions.

**If implemented:** GUI, CLI, and Python callers will use one tested workflow.
Interface code will only collect inputs and report the result; `io.writers`
will remain responsible for encoding files.

**Implementation prompt:** Extract only the existing processed-image export
workflow. Keep old entry points as delegates. Prove equivalence for output
arrays, physical ranges, sidecars, warnings, and overwrite protection before
migrating another caller.

### 6. Make viewer state explicit

**Change:** Add a Qt-free `ViewerSession` after JOSS-critical work.

**What it is:** The session should own the current scan and channel, calibrated
range, processing state, display state, ROIs, masks, and undo state.

**Why the current state is bad:** `ImageViewerDialog` mixins read and write
shared dialog attributes without one declared state contract. A change in one
mixin can break another, and meaningful state tests require much of the GUI.

**If implemented:** State transitions such as channel change, processing,
undo, ROI/mask updates, navigation, and export preparation can be tested in
plain Python. Qt classes will bind widgets and signals to that session.

**Implementation prompt:** Characterise current viewer behaviour first. Add a
plain-Python session and migrate one transition at a time. Preserve signals,
shortcuts, window behaviour, arrays, calibration, and exports. Do not redesign
the interface or split files merely because they are large.

## Target shape

```text
GUI / CLI / Python API
          |
          v
shared workflows (only where orchestration is currently duplicated)
          |
          +-------------------+
          v                   v
core domain contracts    scientific capability packages
          |              processing / measurements /
          |              analysis / spectroscopy
          +-------------------+
                    |
                    v
          provenance contracts
                    |
                    v
             io readers/writers
```

This target does not require every dependency to point downward immediately.
It requires new dependencies to be controlled, duplicated definitions to gain
one owner, and compatibility paths to be visibly separate from canonical APIs.

## Priority and limits

1. Package-boundary tests and provenance consolidation come before scientific
   validation.
2. Format and operation definitions come before extending those areas.
3. Shared export orchestration is useful but not required to enlarge the JOSS
   feature scope.
4. `ViewerSession` is optional and should not delay scientific validation.

Do not add a general plugin framework, new formats, scientific tools, or
instrument-control features. Do not reorganise `measurements`, `analysis`, or
`spectroscopy` merely for symmetry. Do not remove compatibility imports until
the public API and real user dependencies are known. Most importantly, do not
change numerical algorithms before independent human scientific validation.

## Implementation plan

### Progress

| Phase | Status | Evidence |
|---|---|---|
| 0. Behavioural baseline | Complete | `7e17a8b`; five-format read-only checks and recorded local limitations |
| 1. Package boundaries | Complete | `f3fd44b`; static policy, named exceptions, and Qt confinement |
| 2. Provenance | Complete | `78efeb9` to `eba0eed`; canonical field map, lossless legacy adapter, and removal of the `core -> processing.history` back-edge |
| 3. Format definitions | Not started | Must preserve the five-format baseline |
| 4. Operation contracts | Not started | Must preserve every existing processing state and result |
| 5. Export workflow | Not started | Depends on the completed provenance contract |
| 6. Viewer session | Optional; not started | Must not delay JOSS-critical validation |

“Complete” here means the architectural work and automated equivalence checks
for that phase are complete. It does not mean independent scientific validation
has occurred.

### Delivery rules

Every refactor must follow the same order:

```text
characterise current behaviour
        -> add the new contract beside the old path
        -> compare old and new results
        -> migrate one caller
        -> keep a compatibility delegate
        -> update architecture documentation
```

- One issue should describe one independently reviewable change.
- A test-only pull request should establish the baseline before production code
  moves.
- Refactor pull requests must not also change algorithms, defaults, file
  support, UI behaviour, or output schemas.
- Exact equality is required for identifiers, metadata, units, channel order,
  shapes, warnings, and serialized state. Numerical arrays should be exactly
  equal where deterministic; any tolerance must be justified in the test.
- Existing sidecars, commands, imports, and public entry points remain supported
  throughout the JOSS cycle.
- A phase stops if equivalence cannot be shown. The discrepancy becomes a
  separate investigation; it must not be hidden by updating an expected value.

### Dependency and merge order

```text
Phase 0: behavioural baseline
                 |
Phase 1: architecture enforcement
        +--------+------------------+
        |        |                  |
Phase 2:         Phase 3:           Phase 4:
provenance       formats            operations
   |                                  |
   +-> Phase 5: export workflow ------+
                       |
               Phase 6: viewer session (optional)
```

Phases 2, 3, and 4 can be developed independently after Phase 1, but should not
be merged concurrently when they edit the same `core` files. Phase 5 depends on
the provenance contract. Phase 6 depends on stable processing and export
interfaces and must not block the JOSS-critical work.

The **minimum JOSS route** is Phases 0, 1, and 2 followed by independent human
scientific validation. Phases 3 and 4 are required before extending their
respective systems. Phase 5 is useful consolidation. Phase 6 is optional. If
any later phase is merged into the release candidate, validation of every
affected path must be repeated against the new commit.

### Target dependency rules

| Area | Allowed role | Forbidden direction |
|---|---|---|
| `core` | Vendor-neutral models, processing-operation contracts, state, identity, ROI/mask, validation, and public loading facades | No GUI or CLI dependency; existing local `io` and history delegates must stay named exceptions |
| `io` | Format definitions, vendor decoding, sidecars, and file encoding | No GUI or CLI dependency; writers must not orchestrate processing |
| `processing` | Operation execution, replay, and numerical kernels | No GUI, CLI, or vendor-reader dependency; export calls leave after Phase 5 |
| `measurements`, `analysis`, `spectroscopy` | Scientific capability packages | No GUI or CLI dependency; existing compatibility imports remain explicit |
| `provenance` | History and export records, builders, and schema migration | No GUI or CLI dependency; artifact writing leaves after Phase 5 |
| `workflows` | Shared Qt-free orchestration across backend capabilities | No GUI types, widgets, dialogs, or command-line parsing |
| `gui`, `cli` | Collect user input, call workflows or backend APIs, present results | Must not become the canonical owner of backend rules |

The first architecture test records current exceptions. Later phases remove
only the exceptions they directly replace. This prevents an attempted
all-at-once dependency rewrite.

### Phase 0: establish the behavioural baseline

**Goal:** Define what “no behaviour change” means before moving code.

**Work:**

1. Run the complete supported Python test matrix and record existing failures,
   skips, optional-dependency results, and platform-specific tests.
2. Build a fixture matrix for the five supported input structures: Createc DAT
   image, Nanonis SXM image, RHK SM4 image page, Createc VERT spectrum, and
   Nanonis DAT spectrum.
3. Characterise for each fixture: detected type, metadata, channel names and
   order, units, array orientation and values, physical range, reader warnings,
   thumbnail path, and error behaviour.
4. Add golden cases for `ProcessingState`, provenance sidecars, ROI/mask
   sidecars, processed exports, CLI arguments, and representative viewer state
   transitions. Normalise timestamps only; do not discard scientific fields.
5. Hash raw fixtures before and after tests to prove they remain read-only.

**Acceptance:** The baseline suite is repeatable, all unexplained failures are
resolved or documented, and no production file has changed. This phase creates
evidence of current behaviour, not evidence of scientific correctness.

### Phase 1: enforce package boundaries

**Goal:** Stop the dependency web from growing while later work reduces it.

**Files:** Add `tests/test_architecture_boundaries.py`; update
`docs/architecture.md` and `CONTRIBUTING.md` only if the written rule needs
clarification.

**Work:**

1. Parse imports statically rather than importing every module during the test.
2. Assign every `probeflow` package to the ownership table above.
3. Fail on any backend import of `probeflow.gui` or `probeflow.cli`.
4. Fail on Qt imports outside `gui` and packaging/launcher boundaries.
5. Add a named allow-list for each current back-edge described in
   `docs/architecture.md`. Each entry must state its reason and removal phase,
   or state that it is a retained compatibility facade.
6. Keep the existing runtime checks in `test_layout_compatibility.py` and
   `test_pipeline_connectivity.py`; static rules complement rather than replace
   them.

**Acceptance:** CI fails when a synthetic forbidden import is introduced, the
current tree passes only through explicit exceptions, and the test imports no
optional GUI dependency.

### Phase 2: consolidate provenance

**Goal:** Make one lossless route from processing state to an exported record.

**Canonical ownership:**

- `core.processing_state.ProcessingState` remains the serialized numerical
  recipe attached to `Scan`.
- `provenance.records.ProcessingHistory` owns source context, timestamps, and
  warnings around that recipe.
- `provenance.records.ExportRecord` is the canonical versioned sidecar record.
- `Scan.processing_history` and `provenance.export.ExportProvenance` become
  adapters over those canonical objects.

**Work:**

1. Inventory every field and conversion in `core.scan_model`,
   `processing.history`, `provenance.records`, `provenance.export`,
   `provenance.prepared_export`, and the writers that currently consume
   provenance.
2. Write a field-mapping table. Every old field must map to a canonical field,
   a compatibility-only field, or an explicitly documented deprecation. No
   field may disappear implicitly.
3. Define schema-version handling and pure conversion functions for legacy
   `.probeflow.json` and `.provenance.json` inputs.
4. Make one builder produce `ExportRecord`. Adapt `ExportProvenance` from this
   record where an old writer still expects it.
5. Migrate only writers that already accept or produce provenance, one at a
   time. Do not force provenance output onto unrelated writers.
6. Change `Scan.processing_history` into a compatibility view while preserving
   getter, setter, timestamp, and import behaviour.
7. Retain sidecar names, discovery order, corrupt-file errors, and atomic write
   behaviour.

**Acceptance:**

- New records round-trip without field loss.
- Every checked-in legacy sidecar still loads with the same meaning.
- Processing replay produces the baseline arrays and warnings.
- GUI, CLI, and writer sidecars are semantically identical after normalising
  timestamps and output paths.
- SXM comments and other embedded provenance remain unchanged unless a separate
  documented schema decision is approved.

### Phase 3: centralise format definitions

**Goal:** Give each supported input structure one declaration without rewriting
its reader.

**Canonical ownership:** Use the small `probeflow/core/formats/` package for
`FormatDefinition` and the immutable built-in catalog. This keeps the loading
contract beside `FileType` without putting vendor decoding in `core`.
`core.file_type`, `core.loaders`, `core.scan_loader`, `core.metadata`, and
`core.indexing` remain public facades; actual readers remain under `io` and are
loaded lazily. Catalog functions must return Qt-free data.

**Work:**

1. Define the typed contract and stable canonical IDs. Preserve all current ID
   spellings through an alias map at input and serialization boundaries.
2. Register existing readers rather than moving vendor decoding in the first
   pull request.
3. Migrate formats in this risk-controlled order:
   - RHK SM4 image, including page selection;
   - Nanonis SXM image;
   - Createc VERT spectrum;
   - Createc DAT image and Nanonis DAT spectrum together, because content must
     disambiguate their shared suffix.
4. For each format, migrate sniffing, metadata, full loading, indexing, and
   thumbnail selection before removing its old dispatch branch.
5. Make `gui.models` and `gui.rendering` branch on neutral image/spectrum
   capability information, not vendor-specific IDs where possible.
6. Keep the old functions as delegates until every existing caller is covered.

**Acceptance for every format:**

- Detection reads no more bytes and performs no additional full decode compared
  with the baseline.
- Metadata, arrays, channel order, units, orientation, physical calibration,
  warnings, RHK page behaviour, and errors match the baseline.
- Folder indexing and thumbnail creation return the same items and cache keys.
- Ambiguous `.dat`, empty, corrupt, truncated, missing, and unsupported files
  retain their existing result or exception.
- No new format, writer, entry-point discovery, or plugin API is introduced.

### Phase 4: centralise processing-operation contracts

**Goal:** Make interfaces, replay, validation, and provenance agree on every
existing operation.

**Canonical ownership:** Add `probeflow/core/operation_specs.py` containing
kernel-free `OperationSpec` instances and a built-in registry. A specification
contains identifiers, parameter rules, scope rules, and calibration policy, but
never a numerical callable. `ProcessingStep` can therefore validate against it
without making `core` import `processing`. Numerical handlers remain in
`processing`, and `processing.state` remains the executor during migration.

**Work:**

1. Inventory every name in `core.processing_state._SUPPORTED_OPS`, aliases in
   `core.op_vocab`, dispatcher branch in `processing.state`, GUI encoding in
   `processing.gui_adapter`, and CLI construction in `cli.processing_ops`.
2. Define the `OperationSpec` fields and a validation error contract. Preserve
   canonical long names and accept existing short aliases only where currently
   accepted. Give each spec a handler key; the executor's handler binding must
   contain no second copy of defaults, scope, shape, or calibration metadata.
3. Add specs in four reviewable groups:
   - shape-preserving spatial and background operations;
   - FFT, notch, inverse, and symmetrisation operations;
   - ROI, mask, arithmetic, and other scoped or multi-input operations;
   - geometry, crop, scale, shear, undistortion, and other operations that can
     change shape or physical range.
4. Make `ProcessingStep`, the GUI adapter, CLI helpers, replay dispatcher,
   ROI/mask rules, and provenance query the registry. Remove duplicated sets
   only after no caller reads them directly; retain compatibility exports where
   public.
5. Do not replace the explicit dispatcher unless the registry can delegate to
   it without duplicate metadata. The purpose is one contract, not a clever
   execution framework.

**Acceptance for every operation:**

- Existing serialized states load and serialize identically.
- Defaults, validation failures, warnings, and GUI/CLI parameter encodings
  match the baseline.
- Fixed inputs produce the same array, dtype, shape, ROI/mask effect, and
  physical range.
- Undo/replay and provenance preserve step order and parameters.
- The operation kernel itself has no scientific or numerical edit in the same
  pull request.

### Phase 5: centralise processed-image export

**Goal:** Use one application workflow without changing any writer.

**Canonical ownership:** Add `probeflow/workflows/__init__.py` and
`probeflow/workflows/processed_export.py`. The workflow accepts plain backend
models and values; it must not import PySide6, GUI classes, or CLI parsers.

**Work:**

1. Characterise the existing paths in CLI commands,
   `provenance.prepared_export`, processing export helpers, and
   `gui.viewer.processed_export`.
2. Define typed request and result objects covering source identity, processed
   plane, calibrated range, processing/display state, ROIs, masks, destination,
   format, warnings, and overwrite policy.
3. Implement the workflow by composing the Phase 2 provenance builder with the
   existing `io.writers`; do not copy writer logic.
4. Migrate one non-GUI caller first, then the GUI caller. Keep every old helper
   as a thin delegate so import and call compatibility remains.
5. Remove `processing -> io/provenance` and `provenance -> io` exceptions only
   when no canonical path uses them.

**Acceptance:** Reopened output arrays, channel metadata, calibration, sidecar
content, embedded comments, warnings, collision behaviour, destination names,
and raw-file protection match the baseline for each currently supported export
path. Cancellation and GUI message presentation remain interface concerns.

### Phase 6: introduce `ViewerSession`

**Goal:** Make viewer behaviour testable without redesigning the GUI.

**Canonical ownership:** Add a PySide6-free
`probeflow/gui/viewer/session.py`. It owns state only; widgets, signals,
rendering, file dialogs, and window lifetime remain in GUI classes.

**Work:**

1. List every attribute shared by `ImageViewerDialog` and its build, chrome,
   display, ROI, mask, selection, toolbar, tools, and processing/export mixins.
   Mark each as session state, widget reference, controller, cache, or transient
   UI state.
2. Add plain-Python transition tests before moving each state group.
3. Migrate in order: current scan/channel/range; processing and undo; ROI/mask
   state; navigation and export preparation.
4. Let mixins call session methods rather than mutate session-owned attributes.
   Provide temporary compatibility properties on the dialog during migration.
5. Remove a compatibility property only after all mixins and tests use the
   session. Do not change widget layout, signals, shortcuts, or modeless window
   retention.

**Acceptance:** Plain-Python tests cover every session transition; existing Qt
tests still pass; arrays, ranges, undo, selection, sidecars, navigation, and
exports match the baseline; importing the session does not import PySide6.

### Issue and pull-request breakdown

| Work item | Scope | Depends on |
|---|---|---|
| `ARCH-0` | Baseline fixture and behaviour matrix | None |
| `ARCH-1` | Static boundary test and current exception list | `ARCH-0` |
| `PROV-1` | Field map, canonical builders, legacy adapters | `ARCH-1` |
| `PROV-2` | Migrate existing provenance-producing writers | `PROV-1` |
| `FMT-0` | `FormatDefinition`, IDs, aliases, empty registry path | `ARCH-1` |
| `FMT-1` to `FMT-4` | RHK, Nanonis image, Createc spectrum, ambiguous DAT pair | `FMT-0` |
| `OPS-0` | `OperationSpec`, validation contract, registry | `ARCH-1` |
| `OPS-1` to `OPS-4` | The four operation groups above | `OPS-0` |
| `EXPORT-1` | Qt-free workflow and non-GUI migration | `PROV-2` |
| `EXPORT-2` | GUI migration and obsolete dependency removal | `EXPORT-1` |
| `VIEW-1` | Viewer attribute inventory and session skeleton | `OPS-4`, `EXPORT-2` |
| `VIEW-2` to `VIEW-5` | The four viewer state groups above | `VIEW-1` |

Each work item may use more than one pull request if its review diff becomes
large. It must not absorb an unrelated cleanup discovered during the work.

### Review and validation gates

1. **Automated equivalence:** Relevant baseline, unit, integration, CLI, GUI,
   replay, reader/writer, and packaging tests pass on supported Python versions.
2. **Architecture review:** A human reviewer confirms ownership, dependency
   direction, compatibility, and documentation. The author of a change should
   not be its only reviewer.
3. **Scientific review:** Human authors independently inspect calibration
   formulas, orientation, processing semantics, measurement units, and export
   reconstruction against known data or established tools. AI-written tests do
   not satisfy this gate.
4. **JOSS scope review:** Only behaviour inside the agreed scope is claimed.
   Experimental algorithms may remain present but must not become validated
   claims merely because their architecture was cleaned up.

### Completion criteria

The proposal is complete only when:

- the current and target graphs in `docs/architecture.md` match the code;
- CI enforces package boundaries and lists every retained exception;
- formats, operations, and provenance each have one canonical definition;
- processed exports use one Qt-free workflow;
- old public imports, commands, supported files, and sidecars still work;
- every migrated path has baseline-equivalence evidence;
- separate implementation issues and decisions are linked from issue #52;
- developer and affected user documentation are updated in the same pull
  requests; and
- independent scientific validation remains an explicit, unfinished gate until
  the human authors complete it.
