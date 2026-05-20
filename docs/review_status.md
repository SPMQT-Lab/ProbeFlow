# ProbeFlow Review Status

Date: 2026-05-20

This document preserves the useful status from completed review artifacts after pruning superseded detailed stage files.

## Status Summary

| Review stage | Status | Current artifact | Outcome |
|---|---|---|---|
| Stage 1 scientific workflow and physics review | Completed | Pruned after summary preservation | Identified unit, provenance, feature workflow, lattice correction, pair-correlation, point-mask FFT, periodicity export, and spectroscopy wording issues. |
| Stage 1 workflow physics fixes | Completed | Pruned after summary preservation | Concrete PF-STAGE1-001 through PF-STAGE1-010 fixes were implemented. Latest relevant commit: `2724b77 Improve scientific workflow provenance`. |
| Stage 2 architecture, repetition, and maintainability review plus bounded implementation slices | Completed slices | `review_stage2_architecture_repetition_findings.md` | Produced the architecture/refactor queue and implemented bounded cleanup: measurement adapter, canonical pair/feature-lattice results, ROI point-source helper, launch-context helper, feature-source metadata propagation, and lattice correction operation helper. |

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

The Stage 2 review artifact is:

- `review_stage2_architecture_repetition_findings.md`

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

## Current Non-Review Docs

The following docs remain current user-facing or technical references and were intentionally kept:

- `docs/cli.md`
- `docs/createc_dat_reader.md`
- `docs/dead_code_audit.md`
- `docs/roi_manual_test_checklist.md`
