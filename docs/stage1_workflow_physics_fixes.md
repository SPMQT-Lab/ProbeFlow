# Stage 1 Workflow Physics Fixes

Date: 2026-05-20

This note tracks code changes made after the Stage 1 scientific workflow review. The goal of these fixes is to make user-facing STM workflows more defensible, reproducible, and interpretable without changing numerical behaviour beyond the reviewed unit/provenance defects.

## Fixed Findings

| Finding | Area | Implemented fix |
|---|---|---|
| PF-STAGE1-001 | Lattice/grid angles | Real-space and reciprocal-space lattice angles now use calibrated physical vectors for non-square pixels. |
| PF-STAGE1-002 | Feature-to-lattice residuals | Matched feature displacements now use per-axis physical pixel calibration instead of average pixel size. |
| PF-STAGE1-003 | Feature point workflow | Measure-tab feature maxima are now available as point sources for pair correlation and feature-to-lattice comparison. |
| PF-STAGE1-004 | Feature Finder ROI scope | Opening the legacy Feature Finder with an active area ROI now passes that ROI mask into the dialog. |
| PF-STAGE1-005 | Lattice correction provenance | Canonical affine lattice correction steps now preserve measured/ideal lattice metadata, full matrix, preserve-orientation, and rotation metadata. |
| PF-STAGE1-006 | Pair/feature-lattice measurement context | Pair-correlation and feature-to-lattice rows now record point source, source path, calibration, lattice/match/bin settings, and relevant assumptions in export context. |
| PF-STAGE1-007 | Point mask/FFT exports | Standalone point-mask and point-mask FFT CSV exports now include metadata headers for source, ROI, detector settings, mask settings, pixel size, and FFT units. |
| PF-STAGE1-008 | Periodicity secondary exports | Periodicity copy text and profile/autocorrelation CSVs now include source, ROI, method, width, search bounds, quality, and message context. |
| PF-STAGE1-009 | Pair-correlation assumptions | Pair-correlation measurement exports now retain binning, ROI area, point source, and the explicit edge-correction-not-applied warning. |
| PF-STAGE1-010 | Spectroscopy derivative wording | Spectroscopy controls and exports now label computed derivatives as numerical derivatives, for example `Numerical dy/dx` and `numerical dI/dV`. |

## Validation Coverage

Targeted regression tests were added for:

- Non-square-pixel lattice angles and feature-to-lattice residuals.
- Measure-tab feature maxima as downstream point sources.
- Full affine lattice correction metadata retention.
- Active ROI masking in the legacy Feature Finder path.
- Numerical spectroscopy derivative labels.
- Periodicity copy/profile export context.
- Pair-correlation and feature-lattice export context.
- Point-mask and point-mask FFT metadata headers.

## Remaining Work

These fixes close the concrete Stage 1 defects above, but they do not redesign the feature workflow. The broader long-term work is still to unify the Measure-tab feature maxima workflow with the legacy maxima/minima Feature Finder, and to add richer per-assignment/per-bin exports where users need detailed downstream analysis rather than summary measurements.
