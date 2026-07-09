# ProbeFlow refocus plan — core vs. advanced package split

**Status:** direction agreed 2026-07-09 (Peter Jacobson). Design phase — no code
moved yet. This document is written to be read cold by a reviewer who was not
part of the planning conversation.

**Reviewer:** you are asked to pressure-test the boundary choices, the phase
ordering, and the extension-seam design, and to answer the open questions in
§11. Assume every line reference below was verified against `main` on
2026-07-09 but may drift; re-check before acting.

---

## 1. Why

ProbeFlow began as a way to **browse and process STM images while retaining
their physical meaning** — a faster, multi-format alternative to Gwyddion /
WSxM / the TU-Wien ImageJ (Michael Schmid) tools, without falling back to bare
8/16-bit PNGs that lose calibration. It solved multi-format browsing (`.sxm`,
`.dat`, `.sm4`), added standard processing (background subtraction, smoothing,
edge detection, FFT/lattice, bad-line, mains, symmetrize), and `.dat`↔`.sxm`
conversion.

It then grew three layers that fail the mission test and now dilute it:
**Feature Counting** (UniMR-style segmentation/template/classify), the
**Particle Statistics simulation engine** (null-model hypothesis testing), and
**ML classification** (scikit-learn / CLIP-torch). At ~95k lines, these plus a
Dataset Builder, a Developer Terminal, and a Survey/ScanFlow campaign mode make
the app confusing as a *community STM tool*.

**The sorting test for every feature:** *does it help someone browse or process
an STM image while keeping physical units?*

We are **not deleting** any working code. We are deciding what ships in a
downloadable `.exe` / `.dmg`, and physically separating the rest so the core
codebase and its dependency surface shrink.

---

## 2. Target architecture — three homes

### 2a. `probeflow-core` — the shipped distribution
Browse; image viewer + processing; FFT/reciprocal-lattice; `.dat`↔`.sxm`
conversion; spectroscopy viewer; **basic particle statistics** (§7);
**lightweight feature/grain detection** (the on-ramp to positions, §6).

Dependencies (unchanged floor): `numpy`, `scipy`, `pillow>=12.2`, `PySide6`,
`matplotlib`, `shapely`, `scikit-image`. **No** `opencv-python`,
`scikit-learn`, `torch`, `adstat`.

### 2b. `probeflow-advanced` — `pip install` locally, registers into core
Full Feature Counting (template matching, classify, feature-lattice), Dataset
Builder, Developer Terminal, Survey/ScanFlow. Depends on `probeflow-core` plus
the heavy extras. Discovered at runtime through the `probeflow/plugins/`
registry (§5), so core has **zero compile-time import** of it.

### 2c. `AdStat` — standalone, already exists at `~/AdStat`
The simulation / null-model workbench: load a JSON/CSV/TXT of positions,
compute observables, compare against baseline simulations. ProbeFlow's Particle
Statistics **simulate / compare / tutorial** modes rehome here.

**Important finding:** AdStat *already has* a viz workbench —
`~/AdStat/src/adstat/viz/{workbench,comparison,sandbox,series,realspace,
diagnostics,captions,glossary,app}.py` plus an `adstat/education/` package. So
the rehome is largely **ProbeFlow deleting duplication**, not porting UI. The
statistical engine already lives in `adstat.{stats,models}`; ProbeFlow's
`analysis/adstat_adapter.py` (1,359 lines) is only a bridge to it. Phase 5 is
therefore mostly deletion on the ProbeFlow side plus reconciling any
presentation AdStat still lacks. **Open question 11.4** covers what, if
anything, is unique to ProbeFlow's version and worth moving rather than
dropping.

---

## 3. File inventory

Line counts are approximate (2026-07-09). "→ relocate" means the *code* is core
but currently sits under an advanced-named directory and should move.

### CORE (ships)
- **Backend engine:** `core/`, `io/` (incl. `io/converters/` — verified no
  heavy deps), `processing/` (incl. `processing/tv.py` — pure-numpy TV kernel),
  `provenance/`, `spectroscopy/`, most of `cli/`.
- **GUI shell + browse + viewer:** `gui/app.py`, `gui/styling.py`,
  `gui/workers.py`, `gui/rendering.py`, `gui/workspace_window.py`,
  `gui/browse/`, `gui/viewer/`, `gui/image_canvas.py`, `gui/roi_*`,
  `gui/mask_*`, `gui/dialogs/image_viewer*`, `gui/spec_viewer/`,
  `gui/convert/`.
- **FFT / lattice (core science):** `gui/dialogs/fft_viewer*.py`,
  `gui/lattice_grid/`, `analysis/lattice*.py`, `processing/bragg.py`,
  `processing/inverse_fft.py`, `processing/symmetrize.py`.
- **Basic particle stats backend (already dep-clean — numpy/scipy only):**
  `analysis/pair_correlation.py` (`compute_pair_correlation`),
  `analysis/point_summary.py` (`summarize_point_pattern`, `nn_histogram_nm`),
  `analysis/simple_measurements.py`, `analysis/roi_statistics.py`, and all of
  `measurements/` (incl. `measurements/feature_sets.py::FeatureSetStore` — the
  shared point-set container, and `measurements/point_table_io.py` — CSV/JSON
  import/export). **Verified:** none import `cv2`/`sklearn`/`torch`/`adstat`.
- **Lightweight detection (scipy-only, → core):** `analysis/grains.py`
  (`detect_grains`), `analysis/feature_finder.py` (`find_image_features`).
- **→ relocate:** `gui/features/tv.py` (TV-denoise GUI) — the code is core but
  the file sits under `gui/features/`. Move to e.g. `gui/tv/` so the
  `gui/features/` directory can leave with the advanced package.

### ADVANCED (`probeflow-advanced`)
- **Feature Counting GUI:** `gui/features/panel.py` (2,510),
  `gui/features/controller.py` (779), `gui/features/window.py` (147),
  `gui/features/__init__.py`. (Minus `tv.py`, relocated above.)
- **Feature dialogs:** `gui/dialogs/feature_finder.py` (the GUI, 493),
  `gui/dialogs/feature_lattice_dialog.py`, `gui/widgets/feature_detection_panel.py`.
- **Feature backend needing heavy deps:** `analysis/features.py` (1,172 — cv2
  template matching + sklearn classify), `analysis/feature_bank.py`,
  `analysis/feature_lattice.py`. **Note:** `analysis/features.py` mixes core-
  worthy helpers with cv2/sklearn ones — needs a split, see Open question 11.2.
- **Particle Statistics simulation + presentation** (total ~9,081 lines):
  `gui/dialogs/particle_statistics.py` (4,109),
  `gui/dialogs/particle_statistics_content.py` (1,232),
  `gui/dialogs/particle_statistics_workers.py`,
  `gui/dialogs/adstat_results.py` (1,734),
  `gui/dialogs/particle_field_view.py`, `gui/dialogs/pair_correlation.py`
  (the *dialog*; the analysis stays core), `analysis/adstat_adapter.py` (1,359),
  `measurements/adstat_export.py`. **Most of this is AdStat-bound, not just
  advanced — see §2c and Phase 5.**
- **Other extras:** `gui/dataset_builder/` + `dataset_builder/` +
  `cli/commands/dataset_builder.py` (~5,084 total), `gui/terminal/` (423),
  `gui/survey/` (485).

### DROP (not moved anywhere)
- Feature Counting **lattice-from-features** mode — redundant with the core
  FFT/reciprocal-lattice tools. Remove the mode; keep any shared helper only if
  a core caller needs it.
- Particle Statistics **tutorial** mode content once AdStat's `education/`
  covers it (confirm parity first — Open question 11.4).

---

## 4. Coupling analysis (what must be cut)

The good news: **most advanced imports in `app.py` are already lazy** (inside
methods), so the hard-import surface is small.

| Coupling point | Location | Nature | Cut strategy |
|---|---|---|---|
| `FeatureCountingWindow` | `app.py:1266` (lazy) | opens FC workspace | register as a plugin workspace (§5) |
| `ParticleStatisticsDialog` | `app.py:1391` (lazy) | opened from FC | moves with advanced; core keeps the *basic* panel |
| `SurveyPanel` | `app.py:879` (lazy) | survey workspace | plugin workspace |
| `DeveloperTerminalWidget`, `_DevSidebar` | `app.py:46,96` (**top-level**) | dev workspace | make lazy, then plugin workspace |
| `DatasetBuilderPanel/Sidebar` | `app.py:94` (**top-level**) | dataset workspace | make lazy, then plugin workspace |
| `features.tv` | `app.py:40` (**top-level**) | TV denoise | **core** — relocate file out of `features/` (§3) |
| `FeatureSetStore` | `app.py:327` (lazy) | shared point store | **stays core** — both basic panel and advanced consume it |
| Viewer "Send to Feature Counting" | `viewer/context_menus.py:156`, `image_viewer_processing_export_mixin.py:709` via `immediate_action_requested`/`DeferredPlaneAction` | viewer → FC/TV | core emits a generic "send-to" signal; advanced registers a handler (§5). "Send to TV" stays core. |
| Tools-menu FC entry (Ctrl+3) | `app.py:522` | menu action | core renders menu items the registry supplies |

`FeatureSetStore` (`measurements/feature_sets.py`) is the shared-state hub —
imported by core (`viewer/image_viewer_tools_mixin.py`, `dialogs/import_points.py`)
and advanced (`features/*`, `particle_statistics*`). It **stays in core** as the
point-set container; the basic panel and any advanced consumer both read it. No
cut needed — just confirm it carries no advanced-only assumptions.

---

## 5. The extension seam (`probeflow/plugins/`)

Current state (verified): `plugins/registry.py` defines `PluginRegistry` with
`register(spec)` and `operations(kind=…)`; `plugins/api.py` defines frozen
`PluginSpec` / `PluginOperation` dataclasses for parser/transform/measurement/
writer operation kinds. The package `__init__` docstring says plainly:
"experimental, not wired to any caller." **Nothing dispatches through it.**

The current registry models *data operations*, not *GUI extensions*. The split
needs a **GUI/app extension surface** in addition (or instead). It must let an
installed advanced package contribute, without core importing it:

1. **Workspace windows** — register a factory `() -> WorkspaceWindow` plus a
   menu label + shortcut. Core's Workspace menu iterates the registry.
   (`WorkspaceWindow` already exists — `gui/workspace_window.py`.)
2. **Tools-menu / dialog entries** — register `label -> callable(parent, theme)`.
3. **Viewer "send-to" targets** — register a handler for a named action so the
   viewer context menu ("Send to …") is populated from the registry rather than
   hard-wired to FC.
4. **Discovery** — advanced registers via an entry-point group (e.g.
   `probeflow.extensions`) so `pip install probeflow-advanced` is sufficient; no
   core code change. Core calls `importlib.metadata.entry_points(...)` at startup
   and invokes each extension's `register(registry)`.

**This is the load-bearing refactor (Phase 2).** Until it exists, the packages
cannot physically separate because `app.py` and the viewer reference advanced
symbols by name.

---

## 6. Feature Counting — keep only the detection on-ramp

Feature Counting bundles four modes with different profiles:

| Mode | Backend | Deps | Disposition |
|---|---|---|---|
| Particle/grain **detection** | `grains.py::detect_grains`, `feature_finder.py::find_image_features` | scipy | **CORE.** Same class as edge detection; outputs positions/areas in nm; the on-ramp that feeds basic stats without a second tool. Surface as a viewer action → `FeatureSetStore`. |
| **Template** matching | `analysis/features.py` | opencv | Advanced. |
| **Classify** (GMM / CLIP) | `analysis/features.py`, torch | sklearn, torch | Advanced ("ML guff"). |
| **Lattice** from features | `analysis/feature_lattice.py` | — | **DROP** — redundant with core FFT/reciprocal-lattice. |

Net: lift lightweight detection into core (a viewer action, not a workspace);
the full UniMR-style workspace becomes `probeflow-advanced`.

---

## 7. Basic particle statistics — core panel spec (Phase 1 detail)

**Decision:** keep as its own lightweight panel (not folded into measurements).

**Keeps:** g(r) (`compute_pair_correlation`), particle count, average
nearest-neighbour distance and NN histogram (`summarize_point_pattern`,
`nn_histogram_nm`), density, other computed scalars already surfaced. Input from
the shared `FeatureSetStore` (ROI-marked, detection-produced, or file-imported
points). **Headline capability: one-click CSV/JSON export** of both the point
table and the computed quantities, via the existing `measurements/point_table_io.py`
(already handles `probeflow_csv`, `probeflow_json`, `feature_set_store_json`).

**Drops:** the `generated` (tutorial) and `sandbox` (Model simulations) data
modes; the simulated-envelope comparison; every call into `adstat_adapter`
(`adstat_sandbox_*`, `compare_*`, `workbench_view_spec`) and `adstat_results.py`.

**Build approach:** strip-down of the current `particle_statistics.py`
`data_mode="real"` path, or a fresh compact panel reusing the same clean backend
— **Open question 11.1**. Either way, no new science: the computation already
exists and is dependency-clean.

---

## 8. Phasing — each phase leaves `main` green and shippable

**Phase 1 — Basic-stats core panel.** Build the stripped panel (§7): compute +
export, no sim/tutorial/adstat. No plugin seam required. Valuable standalone;
proves the dependency boundary in practice. *Ships green independently.*

**Phase 2 — Wire the extension seam (§5).** Add the GUI extension registry +
entry-point discovery; expose registration hooks for workspace windows,
tools-menu entries, and viewer send-to targets. Convert Feature Counting (still
in-repo) to register through it instead of being hard-imported by `app.py`.
Relocate `gui/features/tv.py` out of `features/`. Make the top-level
terminal/dataset_builder imports lazy/registered. *The hard refactor.*

**Phase 3 — Physically split packages.** Create `probeflow-advanced` (own
`pyproject.toml`, depends on `probeflow-core` + heavy extras). Move the advanced
files (§3) into it; it registers via entry points. Core then imports none of
them. Split `analysis/features.py` (Open question 11.2). *Two installable
packages; core suite green without advanced installed.*

**Phase 4 — Distribution.** PyInstaller or briefcase build from `probeflow-core`
only → `.exe` / `.dmg`. Hide advanced menus when the registry is empty. Verify a
clean-room core install has no heavy deps and the binary launches. *Shippable
artifact.*

**Phase 5 — AdStat rehome** (in the `~/AdStat` repo). Move any presentation
AdStat lacks; retire ProbeFlow's `adstat_adapter.py` + `adstat_results.py` +
particle-stats sim modes. Mostly deletion on ProbeFlow's side (§2c). *AdStat
absorbs the sim workbench; ProbeFlow drops the `adstat` dependency entirely.*

---

## 9. Verification strategy

- **Per phase:** full `pytest` green (2,814 pass baseline), `ruff check .`
  clean, and a manual smoke of the affected surface. GUI tests need
  `QT_QPA_PLATFORM=offscreen` + `QT_QPA_PLATFORM_PLUGIN_PATH` (see CONTRIBUTING).
- **Test blast radius (verified):** ~14 test files reference
  particle_statistics / feature_counting / adstat; ~10 reference the basic-stats
  backend (`pair_correlation`, `point_summary`). Phase 1 must keep the
  basic-stats tests green; Phases 3–5 will relocate the advanced tests into the
  advanced package's own suite.
- **Boundary guard (recommend adding in Phase 3):** a CI check that
  `probeflow-core` imports cleanly in an environment with **only** the core
  deps installed (no cv2/sklearn/torch/adstat) — this is the regression that
  would otherwise silently re-couple the packages. Model it on the existing
  weekly `dependency-canary` workflow.

---

## 10. Risks & mitigations

- **`analysis/features.py` is mixed** (core-worthy helpers + cv2/sklearn). A
  sloppy move takes core helpers with it or leaves a cv2 import in core. →
  Split the file deliberately in Phase 3 (Open question 11.2).
- **Shared `FeatureSetStore` re-coupling.** If the advanced package extends the
  store's schema, core may need to read fields it doesn't own. → Keep the store
  core-owned and schema-stable; advanced attaches its own metadata out-of-band.
- **Entry-point discovery adds startup cost / failure modes.** A broken
  extension shouldn't crash core. → Wrap each extension's `register()` in a
  guarded try/except with a status-bar notice (mirror the existing optional-
  Survey `_probe_survey_available` pattern).
- **Phase 2 is genuinely large** and touches `app.py` (the 1,900-line shell). →
  Do it behind the still-in-repo advanced code (nothing moves yet), so it can be
  validated with the full suite before any package split.
- **AdStat parity unknown.** Dropping ProbeFlow's tutorial/presentation assumes
  AdStat covers it. → Audit AdStat `viz/` + `education/` before Phase 5 deletes
  anything (Open question 11.4).

---

## 11. Open questions for the reviewer

1. **Phase 1 build approach:** strip the existing 4,109-line
   `particle_statistics.py` down to the `real` path, or write a fresh compact
   panel on the same backend? Strip = less risk of behavioural drift but inherits
   a large file; fresh = clean small file but must re-derive layout/plot code.
2. **`analysis/features.py` split:** where exactly is the line between the
   core-worthy detection/crop helpers and the cv2/sklearn template+classify code?
   Should the scipy-only helpers move to `feature_finder.py`/`grains.py`?
3. **Extension seam shape:** extend the existing `plugins/` data-operation
   registry with a parallel GUI-extension registry, or add GUI extension as a new
   `kind` in the same registry? Entry-point group name?
4. **AdStat parity:** what, if anything, in ProbeFlow's particle-statistics
   presentation / tutorial is *not* already in AdStat `viz/` + `education/` and
   worth porting rather than dropping?
5. **Dataset Builder / Terminal / Survey:** confirmed all three go to
   `probeflow-advanced` (none ship in the `.exe`/`.dmg`)? Or does any belong in a
   third bucket (e.g. Dataset Builder as its own separate companion, given it is
   an ML-labelling workflow distinct from analysis)?
6. **Repo topology:** monorepo with `probeflow-core` + `probeflow-advanced` as
   separate distributions built from one tree, or two repos? Monorepo keeps
   shared-history and one CI; two repos enforce the boundary harder.

---

## 12. Explicitly out of the core distribution

Feature Counting (template/classify/lattice), Particle Statistics simulation
sandbox + tutorial, ML/CLIP classify, Dataset Builder, Developer Terminal,
Survey/ScanFlow. All retained in-repo / in `probeflow-advanced`; none shipped in
the `.exe` / `.dmg`. The community download is the STM image viewer/processor
with basic particle statistics and easy export — nothing more.
