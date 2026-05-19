# ProbeFlow Stage 2 Architecture And Repetition Review Findings

Date: 2026-05-20

Scope: architecture, repetition, and maintainability review after the Stage 1 scientific workflow fixes, followed by the first bounded implementation slice for the highest-priority findings.

## 1. Executive Summary

ProbeFlow now has a stronger scientific workflow foundation than the Stage 1 report described. The most important numerical backends are GUI-free, and several recently added workflows already delegate calculations to `analysis`, `processing`, `measurements`, or `spectroscopy` modules.

Top architecture risks:

1. Two measurement result models are active. `probeflow.analysis.measurements.MeasurementResult` and `probeflow.measurements.models.MeasurementResult` both claim to be the compact exportable result model. The image viewer converts between them in `ImageViewerDialog._to_dock_result`, which creates a high-risk maintenance point for units, provenance, IDs, and table/export behavior.
2. `ImageViewerDialog` is still the central tool launcher and source collector. It is 3123 lines and still owns ROI statistics legacy flow, point-source collection, pair-correlation launch, feature-lattice launch, lattice-grid docking, export checks, and processing application.
3. `LatticeGridPanel` mixes user controls, calibrated physical-vector math, affine correction matrix construction, preview/apply state, and PNG/PDF grid export. The math is now physically improved, but the file is a high-blast-radius place to make future changes.
4. Repeated patterns are now clear enough to extract surgically: unit formatting, ROI selection/mask construction, point-source collection, file export/copy plumbing, and preview/apply state.
5. Spectroscopy has good backend separation but duplicated single-spectrum and overlay export/control methods inside one 1944-line dialog file.

Implementation status for this slice:

- Added `probeflow.measurements.adapters.legacy_measurement_to_result` as the single compatibility adapter from legacy analysis results to the canonical measurement model.
- Updated pair-correlation and feature-to-lattice dialogs to emit canonical measurement results directly.
- Added `probeflow.gui.roi_context` for point-source collection and active area ROI area calculation.
- Added `probeflow.analysis.lattice_correction_workflow` for pixel-space lattice correction matrices and processing/provenance operation parameters.
- Added targeted regression coverage for the adapter, ROI context helper, and lattice correction operation helper.

Recommended next stage: continue with measured, test-backed extractions. Do not split large files mechanically. The remaining high-value work is to migrate the remaining legacy simple/ROI analysis measurement producers, unify feature maxima/minima point-source semantics, then add a shared unit-formatting helper.

## 2. Current Architecture Map

| Layer | Current role | Assessment |
|---|---|---|
| `probeflow.analysis` | GUI-free scientific analysis such as line periodicity, pair correlation, feature-to-lattice, lattice geometry, ROI stats, and simple measurements. | Mostly healthy. No Qt imports found. Some modules still return the older measurement model. |
| `probeflow.processing` | GUI-free array transformations and canonical processing state/history. | Mostly healthy. No Qt imports found. Processing history and geometric-op metadata are now strong. |
| `probeflow.measurements` | Canonical generic measurement model, feature points, image measurement adapters, point-mask FFT, export helpers, and the legacy result adapter. | Good direction. Some legacy analysis producers still need migration. |
| `probeflow.spectroscopy` | Display transforms, normalization, smoothing, measurement, and export helpers for spectroscopy workflows. | Strong backend separation; GUI file duplicates orchestration/export code. |
| `probeflow.gui.viewer` | Extracted controllers/helpers for image viewer measurements, ROI analysis, bad-line preview, exports, and processing. | Good extraction direction. `image_measurements.py` is now near the next split boundary. |
| `probeflow.gui.dialogs` | Tool dialogs and the main image/spectroscopy viewers. | Functional, but the largest files still combine unrelated tool orchestration. |

## 3. Module Risk Table

| Module | Role | Evidence | Layering risk | Refactor priority | Confidence |
|---|---|---:|---|---|---|
| `probeflow/gui/dialogs/image_viewer.py` | Main image viewer, tool launcher, processing/export coordinator. | 3123 lines before this slice; still owns tool launch, but legacy conversion and point-source gathering now delegate to helpers. | High | Critical | Surgical |
| `probeflow/gui/lattice_grid_tool.py` | Real-space/FFT lattice tools and correction panel. | 1985 lines before this slice; preview/apply UI remains in panel, while correction matrix/operation params now delegate to a backend helper. | High | Important | Needs design |
| `probeflow/gui/dialogs/spec_viewer.py` | Single and overlay spectroscopy viewer. | 1944 lines; single/overlay export methods repeat CSV/JSON/TXT path/write/status logic. | Medium | Cleanup | Safe |
| `probeflow/gui/viewer/image_measurements.py` | Image measurement controller. | 952 lines; good extraction, but now owns periodicity export, feature maxima, point masks, point FFT, line profile, ROI stats, and file dialogs. | Medium | Important | Surgical |
| `probeflow/gui/dialogs/feature_finder.py` | Legacy maxima/minima feature finder. | Separate backend and ROI mask support, but not aligned with the newer measure-tab feature maxima workflow. | Medium | Important | Surgical |
| `probeflow/gui/dialogs/pair_correlation.py` | Pair-correlation UI and plot. | Backend calculation delegated; dialog now emits canonical measurement results but still formats summary/context locally. | Medium | Important | Safe |
| `probeflow/gui/dialogs/feature_lattice_dialog.py` | Feature-to-lattice UI and plot. | Backend calculation delegated; dialog now emits canonical measurement results but still formats values locally. | Medium | Important | Safe |
| `probeflow/gui/dialogs/stm_background.py` | STM background preview/apply dialog. | Good backend delegation; preview/apply state resembles other tools but is local. | Low/Medium | Cleanup | Safe |
| `probeflow/gui/viewer/bad_line_preview.py` | Bad-line preview controller. | Good extraction example; no direct dialog status dependency. | Low | Keep pattern | Safe |
| `probeflow/analysis/*`, `probeflow/processing/*`, `probeflow/measurements/*` | Backends and exports. | No Qt imports found in analysis/processing/measurements. | Low | Guardrail | Safe |

## 4. Workflow And Result-Flow Traces

| Workflow | GUI path | Backend path | Measurement/export path | Data/units path | Layering risk |
|---|---|---|---|---|---|
| Line profile and periodicity | Toolbar/Measure tab calls `ImageMeasurementController.find_periodicity_for_active_line_roi`. | `processing.image.line_profile`; `analysis.line_periodicity.estimate_line_periodicity`; `measurements.image.line_periodicity_measurement`. | Measurement table uses the newer result model; profile/autocorrelation CSV generated in `image_measurements.py`. | Distance is computed in metres using per-axis pixel size; displayed/exported context records method, width, bounds, quality. | Medium. Good backend split, but controller owns export formatting and dialog launch. |
| ROI statistics | New controller path calls `measurements.image.roi_statistics`; legacy viewer path calls `analysis.roi_statistics.compute_roi_statistics`. | Two backend implementations/models remain active. | Legacy path converts via `_to_dock_result`; new path records directly. | Area uses both pixel axes in current backends. | High. Duplicate route and model conversion should be collapsed. |
| Feature finder, masks, FFT | Measure tab detects maxima in `ImageMeasurementController`; legacy dialog supports maxima/minima. | New: `measurements.features.detect_local_maxima`; legacy: `analysis.feature_finder.find_image_features`; mask/FFT: `measurements.fft_points`. | New path writes self-describing CSV/JSON and table rows; legacy CSV remains narrower. | New path stores nm coordinates and detection metadata; legacy result is pixel-first and dialog-local. | Medium/High. Functionally useful but split feature semantics still create maintenance drag. |
| Pair correlation | Image viewer gets point sources from `gui.roi_context`, then opens `PairCorrelationDialog`. | `analysis.pair_correlation.compute_pair_correlation`. | Dialog emits canonical `measurements.models.MeasurementResult` rows. | Sources are physical metre coordinates; ROI area uses active area ROI mask and per-axis pixel sizes. | Medium. Calculation and model path are separated; source metadata still needs richer unification. |
| Lattice/grid measurement | Image viewer docks `LatticeGridPanel`. | `analysis.lattice_grid` models/formatters; `analysis.lattice_distortion.compute_correction`. | Grid export via `gui.lattice_export`; correction applied as a geometric processing op. | Calibrated physical vectors are used in correction display and operation params. | High. UI panel also builds correction matrices and operation payloads. |
| Linear lattice correction | Distortion tab in `LatticeGridPanel`. | `analysis.lattice_distortion.compute_correction`; `analysis.lattice_correction_workflow`; `processing.image.affine_lattice_correction`; `processing.state` records operation. | Apply callback adds canonical `affine_lattice_correction` params to processing state. | Matrix conversion between nm-space and pixel-space is isolated in a GUI-free helper. | Medium. Scientifically important conversion is now testable outside the panel; labels and UI state remain panel-owned. |
| Feature-to-lattice comparison | Image viewer gets point sources from `gui.roi_context`, then opens `FeatureLatticeDialog`. | `analysis.feature_lattice.compare_features_to_lattice`. | Dialog emits canonical `measurements.models.MeasurementResult` rows. | Backend returns physical displacements when per-axis pixel sizes are supplied. | Medium. Backend and model path are good; feature-source semantics still need unification. |
| Spectroscopy trace processing | `SpecViewerDialog` and overlay viewer. | `spectroscopy.transforms`, `spectroscopy.normalization`, `spectroscopy.smoothing`, `spectroscopy.measurement`, `spectroscopy.export`. | Measurement table uses newer result model; CSV/JSON/TXT/xmgrace export helpers are backend-driven. | Displayed data is explicitly transformed copy of raw traces. | Medium. Backend split is sound; GUI duplication can be reduced. |

## 5. Repetition Table

| Pattern | Repeated locations | Count estimate | Refactor recommendation |
|---|---|---:|---|
| Distance/unit formatting | `analysis.simple_measurements`, `analysis.line_periodicity`, `analysis.lattice_grid`, `analysis.spec_plot`, `gui.lattice_grid_tool`, `gui.dialogs.pair_correlation`, `gui.dialogs.feature_lattice_dialog`, FFT/periodic filter dialogs. | 8+ | Add a small `probeflow.utils.units` helper and migrate only measurement/reporting paths first. |
| ROI selection and mask construction | `image_viewer.py`, `image_measurements.py`, `feature_finder.py`, `measurements.features`, `measurements.image`, `processing.state`, `processing.background`, `processing.filters`. | 8+ | Add a GUI-side ROI context helper for active/selected ROI and masks; keep backend validation local. |
| Point-source collection | `image_viewer._collect_point_sources_m`, `_collect_point_sources_px`, feature maxima controller, feature finder dialog, point ROI collection. | 3+ | Extract a point-source registry/context object so pair correlation and feature-lattice share source metadata. |
| Measurement model conversion | `analysis.measurements`, `measurements.models`, `image_viewer._to_dock_result`, pair/feature-lattice dialogs, old `MeasurementResultsPanel`, new `MeasurementResultsTable`. | 5+ | Make one canonical result model and move any legacy adapter to one module. |
| File save/copy boilerplate | Measurement tables, image measurements, line profile export, feature finder, point FFT, lattice export, spectroscopy viewer, FFT viewer, app metadata export. | 10+ | Add a tiny GUI export helper after the measurement-model work. |
| Preview/apply/clear-preview state | STM background, bad lines, lattice correction, periodic filter, feature finder preview, point-mask FFT display. | 5+ | Do not abstract first. Normalize naming/state contracts after ROI/source extraction. |
| Matplotlib theming and diagnostic plot setup | Pair correlation, feature-lattice, periodicity plot, FFT viewer, feature finder, spectroscopy viewer, lattice preview fallback. | 6+ | Optional cleanup helper for themed axes/canvas only after functional refactors. |
| NaN/finite handling | Many backends and display/export helpers. | 10+ | Keep local unless a concrete bug appears; broad abstraction would add little value. |

## 6. High-Priority Findings

### PF-STAGE2-001

Priority: Critical

Location: `probeflow.analysis.measurements`, `probeflow.measurements.models`, `probeflow.gui.dialogs.image_viewer._to_dock_result`, `probeflow.gui.widgets.measurement_results_panel`, `probeflow.gui.widgets.measurement_table`.

Problem: ProbeFlow has two active measurement result schemas and two table widgets. Older analysis tools and pair/feature-lattice dialogs create `analysis.measurements.MeasurementResult`, while newer image/spectrum workflows use `measurements.models.MeasurementResult`. The image viewer converts the older form to the newer form in `_to_dock_result`.

Why it matters: This is the central architecture risk for reproducibility. Units and provenance can be lost or renamed at the conversion seam, new result kinds need ad hoc mapping, and export behavior depends on which table/model a workflow touches.

Recommended change: Make `probeflow.measurements.models.MeasurementResult` canonical. Keep one compatibility adapter in a backend or measurement module, not in `ImageViewerDialog`. Migrate simple measurements, ROI statistics, pair correlation, and feature-to-lattice to return the canonical model. Retire `MeasurementResultsPanel` or make it a thin wrapper around the canonical table.

Suggested verification: Add targeted tests that pair correlation, feature-to-lattice, ROI stats, distance/angle, line periodicity, feature maxima, point FFT, and spectroscopy all export through `measurements.export` with source path, channel, units, values, and context intact.

Implementation status: Partially completed in this slice. The compatibility adapter now lives in `probeflow.measurements.adapters`, and pair-correlation plus feature-to-lattice dialogs emit the canonical measurement model directly. Remaining migration targets are simple measurements, legacy ROI statistics, and retirement or wrapping of the old measurement panel.

### PF-STAGE2-002

Priority: Critical

Location: `probeflow/gui/dialogs/image_viewer.py`.

Problem: The main image viewer is still the integration point for too many workflows. It owns measurement result conversion, ROI stats legacy calculation, point-source collection, pair-correlation launch, feature-lattice launch, lattice-grid docking, processing/export checks, and status updates.

Why it matters: The file is large enough that unrelated tools can accidentally affect each other. It also makes test targeting difficult: source collection, ROI validation, and dialog launch behavior are hard to exercise without constructing the full viewer.

Recommended change: Extract two narrow GUI-side services before any file split: an ROI/point-source context helper and a tool-launch coordinator for pair correlation, feature-lattice, and lattice grid. Keep the viewer as the owner of widgets and callbacks, but move decision logic out.

Suggested verification: Existing GUI smoke tests plus focused tests for active/selected ROI fallback, point-source source names, pair-correlation source availability, and feature-to-lattice lattice-required behavior.

Implementation status: Partially completed in this slice. `probeflow.gui.roi_context` now owns point-source collection and active area ROI area calculation, and the image viewer delegates those decisions. Remaining work is a small tool-launch coordinator and broader ROI context coverage for line/area validation.

### PF-STAGE2-003

Priority: Important

Location: `probeflow/gui/lattice_grid_tool.py`, especially `LatticeGridPanel._refresh_correction_label`, `_correction_matrix_px`, `_on_preview`, and `_on_apply`.

Problem: The lattice panel computes calibrated measured vectors, constructs ideal/measured correction objects, converts correction matrices between physical and pixel bases, previews transformed arrays, and builds processing operation payloads inside GUI callbacks.

Why it matters: This workflow is scientifically sensitive and already has non-square-pixel pitfalls. Keeping correction operation construction in the GUI increases the chance that future changes update labels but not applied/exported matrices, or vice versa.

Recommended change: Extract a backend helper that accepts grid vectors, calibration, ideal settings, and correction options, then returns a small operation object: measured/ideal metadata, displayed label values, physical matrix, applied pixel matrix, and processing-state params. Leave widget state and button enabling in the GUI.

Suggested verification: Preserve existing lattice-grid and lattice-correction tests; add one test comparing generated operation params for square and non-square pixels, including preserve-orientation true/false.

Implementation status: Partially completed in this slice. Pixel-space matrix conversion and processing operation parameter construction now live in `probeflow.analysis.lattice_correction_workflow`, with targeted non-square-pixel tests. Correction label construction and preview/apply widget state remain in the GUI panel.

### PF-STAGE2-004

Priority: Important

Location: ROI handling across `image_viewer.py`, `image_measurements.py`, `feature_finder.py`, `measurements.features`, `measurements.image`, and `processing.state`.

Problem: Active ROI, selected ROI, line ROI, area ROI, point ROI, and ROI mask rules are repeated in multiple places. The code is mostly correct, but tool-specific fallbacks differ: for example feature maxima over a non-area active ROI falls back to full image in one path, while direct ROI action reports an area-ROI error.

Why it matters: In STM workflows, ROI scope is part of the scientific result. Repeated selection and mask logic makes it easy for one workflow to use full-image data while another uses the active ROI under similar UI conditions.

Recommended change: Add a GUI-only ROI context helper that returns explicit objects such as active area ROI, selected point ROIs, active line ROI, and mask-for-image. Do not move backend mask validation out of backend modules.

Suggested verification: Tests for area ROI required, line ROI required, point-source collection, non-area fallback behavior, empty mask handling, and active ROI area calculation.

### PF-STAGE2-005

Priority: Important

Location: unit formatting and physical-scale conversion in `analysis.simple_measurements`, `analysis.line_periodicity`, `analysis.lattice_grid`, `analysis.spec_plot`, `gui.lattice_grid_tool`, pair/feature-lattice dialogs, FFT/periodic filter dialogs.

Problem: ProbeFlow formats metres, nanometres, Angstrom-scale values, reciprocal units, densities, and areas in several local helpers with slightly different thresholds and return shapes.

Why it matters: Unit display is not just UI polish here. Mixed thresholds or labels can make results appear inconsistent across line periodicity, lattice measurements, pair correlation, FFT, and feature-lattice comparisons.

Recommended change: Introduce a small `probeflow.utils.units` module with distance, area, reciprocal-space, density, and scalar formatting helpers. Migrate only result summaries and export/display text first; leave low-level SI values unchanged.

Suggested verification: Unit-format tests for pm/Angstrom/nm/m, area nm^2, density nm^-2, reciprocal nm^-1, and signed values.

### PF-STAGE2-006

Priority: Important

Location: `probeflow.gui.viewer.image_measurements.ImageMeasurementController`, `probeflow.gui.dialogs.feature_finder.FeatureFinderDialog`, `probeflow.measurements.features`, `probeflow.analysis.feature_finder`.

Problem: Feature detection has two user-facing implementations: new measure-tab maxima and legacy maxima/minima feature finder. Both are useful, but they use different result types, metadata/export paths, and downstream integration.

Why it matters: Features are downstream inputs for masks, FFT, pair correlation, and lattice comparison. Maintaining two point-result semantics makes it harder to guarantee that source, ROI scope, thresholds, smoothing, and coordinate units survive every downstream operation.

Recommended change: Create one canonical point-source/result representation and let both UI paths populate it. Preserve the legacy dialog if needed, but route its accepted result through the same source registry and export helpers as measure-tab maxima.

Suggested verification: Tests that legacy minima and measure-tab maxima can both become point sources for pair correlation and feature-to-lattice comparison, with ROI scope and detection settings recorded.

### PF-STAGE2-007

Priority: Cleanup

Location: file export/copy methods across measurement tables, image measurements, feature finder, line profile export, spectroscopy viewer, point FFT, lattice export, and app metadata export.

Problem: Save-dialog, empty-data guard, `Path.write_text`, clipboard copy, and status-message patterns are repeated.

Why it matters: This repetition is not scientifically dangerous by itself, but it makes provenance/export fixes tedious and inconsistent.

Recommended change: After canonical measurement and ROI-source work, add a tiny `probeflow.gui.export_helpers` module for `save_text_with_dialog`, `copy_text`, and common status/error handling. Keep binary/image export custom.

Suggested verification: Tests should focus on one helper with monkeypatched dialogs and one representative caller.

### PF-STAGE2-008

Priority: Cleanup

Location: `probeflow/gui/dialogs/spec_viewer.py`.

Problem: Single-spectrum and overlay spectroscopy viewers duplicate CSV/JSON/TXT export flows and several crosshair/measurement delegations while sharing the same backend transform/export concepts.

Why it matters: Spectroscopy is currently scientifically clear, but the GUI file is large and future export/provenance changes need to be duplicated.

Recommended change: Extract a spectroscopy export mixin/helper or a small local function group that accepts current displayed spectra, default filename, and status callback. Avoid changing spectroscopy transform semantics.

Suggested verification: Keep `tests/test_spectroscopy_display.py` and GUI integration export tests passing; add one overlay/single export helper test if implementation touches both paths.

### PF-STAGE2-009

Priority: Cleanup

Location: preview/apply workflows in STM background, bad-line preview, lattice correction, feature finder, and periodic filter.

Problem: Each workflow owns its own preview-active state, invalidation, status text, and apply behavior.

Why it matters: This is manageable today, but as more tools gain previews, inconsistent "parameters changed" and "apply stale preview" behavior will be easy to introduce.

Recommended change: Do not introduce a framework immediately. First normalize naming and contracts: preview computes from current params, apply either recomputes or requires fresh preview explicitly, and parameter changes invalidate preview where relevant.

Suggested verification: One test per preview tool is enough: stale preview invalidates or recomputes; apply records current params.

### PF-STAGE2-010

Priority: Cleanup

Location: themed Matplotlib plot setup in pair correlation, feature-lattice, periodicity, FFT/periodic filter, feature finder, spectroscopy, and lattice preview fallback.

Problem: Plot theming and axis setup are repeated.

Why it matters: Low scientific risk, but it creates maintenance noise.

Recommended change: Only extract a themed-axes helper if a future UI cleanup already touches these dialogs. This should stay below model/ROI/lattice work.

Suggested verification: Screenshot or Qt smoke coverage only if visual behavior changes.

## 7. Candidate Helper Extractions

These helpers meet the "3+ call sites or concrete risk" threshold:

| Helper | Initial contents | Do not include yet |
|---|---|---|
| `probeflow.measurements.adapters` | Implemented: single adapter from legacy analysis results to canonical measurement results during migration. | UI table code or dialog logic. |
| `probeflow.gui.roi_context` | Implemented initial slice: active area ROI area plus point sources from legacy feature result, measure-tab feature maxima, selected point ROIs, and all point ROIs in physical/pixel coordinates. | Backend mask validation or processing-state ROI resolution. |
| `probeflow.analysis.lattice_correction_workflow` | Implemented: pixel-space correction matrix conversion and processing/provenance operation params. | Preview/apply widget state or image transformation kernel code. |
| `probeflow.utils.units` | Display-only formatting for distance, area, density, reciprocal-space, and period bounds. | Changing stored SI values or backend APIs. |
| `probeflow.gui.export_helpers` | Small text-save/copy wrappers with status callbacks. | Image export, processed scan export, or provenance serialization. |

## 8. Refactor Queue

Critical:

1. Finish canonicalizing measurement results.
   - Completed in this slice: one compatibility adapter plus canonical pair-correlation and feature-to-lattice results.
   - Remaining target: migrate ROI stats and simple measurements or adapt them at their call boundaries; retire or wrap the old result panel.

2. Continue extracting ROI and point-source context from `ImageViewerDialog`.
   - Completed in this slice: point-source gathering and active area ROI area calculation.
   - Remaining target: active/selected line and area ROI context plus a small launch coordinator.

Important:

1. Finish lattice correction extraction.
   - Completed in this slice: matrix conversion and processing params.
   - Remaining target: optional correction label/value object if future changes touch the panel.

2. Unify feature point result/source handling.
   - Target: measure-tab maxima and legacy maxima/minima can both feed point masks, FFT, pair correlation, and feature-lattice comparison through the same source representation.

3. Add unit formatting helper.
   - Target: remove repeated Angstrom/nm/pm/nm^-1 threshold logic from user-facing summaries.

Cleanup:

1. Add text export/copy helper for GUI dialogs.
2. Reduce duplicated single/overlay spectroscopy export methods.
3. Optionally extract themed Matplotlib plot setup.
4. Keep `_legacy.py` from growing; do not put new scientific workflow logic there.

## 9. Compatibility Guardrails

- Keep Python 3.11 and 3.12 compatibility; avoid 3.12-only syntax.
- Keep analysis, processing, measurements, and spectroscopy backends free of PySide6/Qt imports.
- Preserve existing public imports with shims during model/module moves.
- Avoid breaking `probeflow.processing.image` compatibility re-exports without explicit migration.
- Do not move large GUI files mechanically. Extract around real tool boundaries with tests.
- Keep processing-state and provenance keys stable for existing exports.
- Prefer additive adapters first, then remove old paths only after tests prove no caller remains.

## 10. Recommended Stage 3 Implementation Plan

Stage 3A: Finish measurement model consolidation.
- Migrate remaining legacy analysis result producers or adapt them at one non-GUI boundary.
- Remove `_to_dock_result` as a workflow dependency once no old producers reach the viewer.
- Retire or wrap the legacy `MeasurementResultsPanel`.

Stage 3B: Complete ROI context.
- Extend `probeflow.gui.roi_context` to active/selected line and area ROI selection.
- Add focused tests for line-required, area-required, empty-mask, and non-area fallback behavior.

Stage 3C: Feature source unification.
- Route legacy maxima/minima and measure-tab maxima through a canonical point-source representation.
- Preserve source, ROI scope, threshold/smoothing settings, and coordinate units.

Stage 3D: Cleanup helpers.
- Unit formatting helper.
- GUI text export/copy helper.
- Spectroscopy export duplication cleanup.

This order keeps the scientifically sensitive seams first and leaves cosmetic repetition for after the result/provenance paths are simpler.
