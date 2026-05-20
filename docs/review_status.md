# ProbeFlow Review Status

Date: 2026-05-20

This document preserves the useful status from completed review artifacts after pruning superseded detailed stage files.

## Status Summary

| Review stage | Status | Current artifact | Outcome |
|---|---|---|---|
| Stage 1 scientific workflow and physics review | Completed | Pruned after summary preservation | Identified unit, provenance, feature workflow, lattice correction, pair-correlation, point-mask FFT, periodicity export, and spectroscopy wording issues. |
| Stage 1 workflow physics fixes | Completed | Pruned after summary preservation | Concrete PF-STAGE1-001 through PF-STAGE1-010 fixes were implemented. Latest relevant commit: `2724b77 Improve scientific workflow provenance`. |
| Stage 2 architecture, repetition, and maintainability review plus bounded implementation slices | Completed slices | Pruned after summary preservation | Produced the architecture/refactor queue and implemented bounded cleanup: measurement adapter, canonical pair/feature-lattice results, ROI point-source helper, launch-context helper, feature-source metadata propagation, and lattice correction operation helper. |
| Stage 3 compatibility, stability, and release-safety review | Completed locally with release caveats | Pruned after summary preservation | Verified import/CLI/processing/provenance surfaces locally, fixed a Qt headless test-harness abort, verified a temp wheel includes toolbar assets, and documented remaining Python 3.11/3.12 matrix and GUI-smoke checks. |

## Stage 1 Preserved Outcome

The Stage 1 review was a review-only pass over ProbeFlow's user-facing STM workflows. It found several risks that could make output scientifically misleading or difficult to reproduce, especially around calibrated non-square pixels, feature-point workflow continuity, lattice correction provenance, pair/feature-lattice measurement context, point-mask FFT exports, line-periodicity exports, and spectroscopy derivative wording.

Follow-up fixes addressed the concrete Stage 1 findings:

- Lattice/grid angle and reciprocal-space reporting now use calibrated physical vectors.
- Feature-to-lattice residuals now use per-axis physical pixel calibration.
- Measure-tab feature maxima are available as downstream point sources.
- Active area ROI scope is passed into the legacy Feature Finder.
- Affine lattice correction provenance retains measured/ideal lattice metadata, matrix data, preserve-orientation, and rotation metadata.
- Pair-correlation, feature-lattice, point-mask, point-mask FFT, periodicity, and spectroscopy exports record stronger context.

Remaining Stage 1-derived long-term work is architectural rather than a direct defect fix: unify the feature maxima/minima workflows and add richer per-assignment or per-bin exports where detailed downstream analysis needs them.

## Stage 2 Preserved Outcome

The bounded implementation slices from Stage 2 addressed the highest-priority seams without a broad refactor:

- `probeflow.measurements.adapters.legacy_measurement_to_result` is the single compatibility adapter for old analysis measurement rows.
- Pair-correlation and feature-to-lattice dialogs now emit canonical `probeflow.measurements.models.MeasurementResult` rows.
- Simple distance/angle and ROI-statistics producers now emit canonical measurement rows.
- The legacy `MeasurementResultsPanel` public name now wraps the canonical measurement table.
- `probeflow.gui.roi_context` now gathers downstream point sources, preserves source metadata, resolves line/area ROI context, and calculates active area ROI physical area outside the main viewer.
- `probeflow.gui.viewer.tool_launch` now owns pair-correlation, feature-to-lattice, and lattice-grid launch precondition checks and source/context assembly.
- Pair-correlation and feature-to-lattice table rows now record point-source type, selection scope, and available detection settings.
- `probeflow.analysis.lattice_correction_workflow` now builds pixel-space lattice correction matrices and processing/provenance operation parameters outside the lattice GUI panel.
- Focused regression tests cover the adapter, ROI context helper, launch-context helper, and lattice correction helper.

The recommended next implementation stage is to address the remaining Stage 2 findings in this order:

1. Continue feature-source unification for point masks, point FFT, and detailed feature exports where source provenance is needed.
2. Remove compatibility measurement shims only after no supported caller depends on legacy result rows.
3. Add small unit-formatting and text-export helpers.
4. Clean up spectroscopy and plotting duplication.

## Stage 3 Preserved Outcome

The local compatibility/stability pass found no application-code S0/S1 blocker in the inspected import, CLI, optional dependency, processing-state, provenance, or package metadata surfaces. The current advertised CLI entry through `probeflow.cli.main` works for top-level, pipeline, info, convert, and GUI help paths. Backend imports remain GUI-free, and optional OpenCV, scikit-learn, and Gwyddion writer dependencies are lazy or guarded.

One release-validation blocker was fixed in the test harness: mixed backend/GUI tests that instantiate `QApplication` now use the same subprocess Qt preflight skip as the main GUI test modules, preventing local headless Qt aborts from killing the full pytest run while preserving backend coverage in those files.

Local validation completed with:

- `ruff check probeflow tests`
- `pytest tests/test_layout_compatibility.py tests/test_processing_state.py tests/test_export_provenance.py`
- `pytest tests/test_feature_lattice.py tests/test_pair_correlation.py tests/test_lattice_grid.py -rs`
- full `pytest`
- temp wheel build from a copied worktree, including installed-wheel import/resource/CLI smoke checks

The release caveats are:

1. Python 3.11 and Python 3.12 were not available locally, so the Stage 3 check set still needs a true 3.11/3.12 matrix before release-complete status.
2. Local Qt cannot initialize `QApplication`, so GUI tests skip under the preflight guard here; manual GUI smoke or CI on a working Qt platform is still needed.
3. Toolbar assets and package-data metadata passed local wheel verification, but they still need to be included in the final release commit so clean checkouts reproduce that package result.

## Housekeeping Notes

Detailed root-level Stage 2/3 review outputs and the dated dead-code audit were pruned after this summary captured their useful conclusions. Future review or audit passes should update this file only when the outcome needs to remain visible to users or contributors.

## Current Non-Review Docs

The following docs remain current user-facing or technical references and were intentionally kept:

- `docs/cli.md`
- `docs/createc_dat_reader.md`
- `docs/roi_manual_test_checklist.md`
