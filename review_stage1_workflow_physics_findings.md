# ProbeFlow Stage 1 Scientific Workflow Review Findings

Date: 2026-05-19
Scope: review-only pass over ProbeFlow STM analysis workflows, GUI paths, backend logic, measurement/export paths, and provenance/parameter recording.

No application code was changed for this stage. Existing tests were used as supporting evidence, not as the main deliverable.

Targeted validation run:

```text
env QT_QPA_PLATFORM=offscreen /opt/anaconda3/bin/python -m pytest \
  tests/test_line_periodicity.py tests/test_pair_correlation.py \
  tests/test_feature_lattice.py tests/test_lattice_grid.py \
  tests/test_lattice_correction_kernel.py tests/test_spectroscopy_display.py \
  tests/test_measurement_seams.py -q

176 passed in 1.71s
```

## 1. Executive Summary

Top risks:

1. Calibrated non-square pixels can silently corrupt lattice angles and feature-to-lattice displacements. The numerical kernels often keep pixel and physical units separate, but several UI summaries use pixel-space angles or average pixel size conversions where anisotropic calibration requires per-axis scaling.
2. Feature-point workflows are split between two UI paths. The newer Measure tab produces feature maxima, masks, FFTs, and measurement rows, but pair correlation and feature-to-lattice comparison only see the legacy Feature Finder dialog result or point ROIs. A user can complete one visible feature workflow and then be blocked by the next tool.
3. Lattice correction is applied with a clear measured-to-ideal direction and good controls, but the canonical processing/provenance adapter drops much of the rich correction metadata that the dialog constructs.
4. Measurement recording is split between legacy `analysis.measurements.MeasurementResult` and newer `measurements.models.MeasurementResult`; conversion is lossy for pair-correlation and feature-lattice results.
5. Export/provenance is relatively strong for processed image exports and spectroscopy display exports. The weaker areas are derived CSVs for masks/FFTs, legacy feature-finder CSVs, and copy/profile exports that omit method context.

Overall assessment: ProbeFlow is scientifically promising and many numerical kernels are solid, but Stage 2 should first fix the unit/provenance seams that can produce plausible-looking but wrong measurements.

## 2. Workflow Summary Table

| Workflow | Functional status | Scientific risk | Primary data source | Key parameters exposed | Parameters recorded/exported | Notes |
|---|---:|---:|---|---|---|---|
| Open folder -> browse images -> choose scan -> adjust display -> export | Works | Low | Raw scan plus current display-scaled view | Channel, colormap, clipping/display range, scalebar | PNG/export provenance includes display state via `png_display_state`; processed exports record processing state | Export is generally defensible when provenance is saved with the image. |
| Open image -> remove bad lines -> STM background -> export with provenance | Works | Medium | Processed image derived from raw scan | Bad-line method/threshold/polarity/adjacent-line guard; STM model/fit region/statistic/blur/jump threshold | Canonical processing state records bad-line and STM background params | Good preview/validation; scientific risk is mainly whether users understand whole-image subtraction after ROI fitting. |
| Draw line ROI -> line profile -> periodicity -> measurement table | Works | Low/Medium | Processed/display image along line ROI | Width, method, background, smoothing, period bounds | Measurement row records method/background/smoothing/width/quality; profile CSV is thinner | The main measurement is good; copy/profile export should include method and width. |
| Area ROI -> statistics -> measurement table | Works | Low/Medium | Processed/display image inside area ROI | ROI geometry | Measurement row records area, height stats, finite/non-finite counts, units | Area uses both x and y pixel sizes. Duplicate legacy/new paths should be unified eventually. |
| Find maxima/minima -> threshold -> feature points -> feature mask -> FFT | Partly works | Medium/High | Processed/display image and derived binary point mask | New Measure tab: maxima only. Legacy Feature Finder: maxima/minima and feature image | New path records better metadata; legacy CSV lacks source/params; point-mask CSV/FFT CSV lacks self-contained metadata | Two overlapping workflows create inconsistent results and provenance. |
| Feature points -> pair correlation | Partly works | Medium/High | Physical coordinates from legacy feature result or point ROIs | Point source, bin controls, optional ROI area | Table row records counts/density/NN/peak/quality but loses some context via conversion | New Measure tab maxima are not available to this tool. Edge correction warning is present. |
| Feature points -> lattice comparison | Partly works | High/Dangerous | Pixel coordinates from legacy feature result or point ROIs plus active lattice grid | Point source, match radius, overlay toggles | Table row records counts/RMS/occupancy but not full lattice/match context; non-square displacement conversion is wrong | Highest-risk derived workflow after lattice angle issue. |
| Fit lattice grid -> ideal lattice -> linear correction -> apply/export | Partly works | High | Active processed image and lattice grid in pixel coordinates | Grid type/locks, ideal a/b/angle, preserve orientation, expand canvas, interpolation, fill | Applied operation stores rich GUI params, but canonical export state keeps only matrix/expand/interpolation/fill | Apply direction is clear; measured angle display and exported correction metadata need repair. |
| Spectroscopy file -> inspect trace -> smooth/derivative/export | Works | Medium | Raw spectrum copied into displayed transform pipeline | Channel, smoothing, derivative, normalization, outlier masking, vertical offset | CSV/JSON/TXT exports record options, source, units, excluded indices, displayed arrays | Non-destructive design is good. Risk is interpretive: displayed exports are transformed data and derivative assumptions need visible context. |

## 3. High-Priority Findings

### P0 Findings

#### PF-STAGE1-001

Severity: P0
Workflow: Fit lattice grid -> ideal lattice -> linear correction -> apply/export
Location: UI path `Image Viewer -> Grid`; modules `probeflow.analysis.lattice_grid.format_real_space_measurements`, `probeflow.gui.lattice_grid_tool.LatticeGridPanel._refresh_correction_label`.

Reproduction steps:

1. Open or construct a calibrated image with non-square pixels, for example `px_size_x = 1 nm/px` and `px_size_y = 2 nm/px`.
2. Open the real-space lattice/grid tool.
3. Create a grid with `a_px = (10, 0)` and `b_px = (10, 10)`.
4. Inspect the displayed real-space lattice angle and measured lattice angle in the distortion/correction panel.

Observed result:

The displayed angle is computed from pixel vectors via `grid.angle_deg()` (`probeflow/analysis/lattice_grid.py:221` and `probeflow/analysis/lattice_grid.py:426`; distortion panel at `probeflow/gui/lattice_grid_tool.py:1036`). For the example above it reports 45 deg.

Expected scientific/user result:

Real-space angle should be computed after applying calibration to both vector components: `a_m = (ax * px_x, ay * px_y)`, `b_m = (bx * px_x, by * px_y)`. For the example above, the physical angle is about 63.4 deg, not 45 deg.

Why it matters:

This can silently report a wrong lattice angle for anisotropic scans or non-square pixels while other displayed values, such as `|a|`, `|b|`, and unit-cell area, are calibrated. That mix makes the output look internally consistent when it is not.

Recommended fix class: physics/math, UI, tests.

Stage 2 target:

Compute real-space and reciprocal-space grid angles from calibrated vectors, not pixel vectors. Add regression tests with non-square calibration showing angle changes while square-pixel results remain unchanged.

#### PF-STAGE1-002

Severity: P0
Workflow: Feature points -> lattice comparison
Location: UI path `Image Viewer -> Feature-to-lattice comparison`; modules `probeflow.analysis.feature_lattice.compare_features_to_lattice`, `probeflow.gui.dialogs.feature_lattice_dialog.FeatureLatticeDialog`.

Reproduction steps:

1. Use a calibrated image with `px_size_x != px_size_y`, for example `1 nm/px` in x and `2 nm/px` in y.
2. Open a lattice grid and compare features where matched displacements are predominantly horizontal or vertical.
3. Inspect RMS/mean displacement in the dialog and after adding to the measurement table.

Observed result:

The backend computes `displacement_px` as Euclidean distance in pixel coordinates (`probeflow/analysis/feature_lattice.py:96`). The dialog converts RMS and mean displacement using the average pixel size, `0.5 * (px_x_m + px_y_m)` (`probeflow/gui/dialogs/feature_lattice_dialog.py:296` and `probeflow/gui/dialogs/feature_lattice_dialog.py:314`).

Expected scientific/user result:

Displacement should be computed in physical coordinates per assignment, for example `sqrt((dx_px * px_x)^2 + (dy_px * px_y)^2)`. A 1 px vertical residual in the example above is 2 nm, while a 1 px horizontal residual is 1 nm; both are currently reported as 1.5 nm.

Why it matters:

This can silently overstate or understate feature-lattice residuals depending on direction. It affects RMS displacement, mean displacement, match-radius interpretation, and any conclusions about lattice distortion or adsorbate registry.

Recommended fix class: physics/math, validation, export, tests.

Stage 2 target:

Store `dx_px`, `dy_px`, `dx_m`, `dy_m`, `displacement_m` per assignment. Summaries should use physical displacements. The match radius should either be explicitly pixel-space or exposed in physical units with anisotropic handling.

### P1 Findings

#### PF-STAGE1-003

Severity: P1
Workflow: Feature points -> pair correlation; Feature points -> lattice comparison
Location: UI paths `Measure tab -> Feature maxima -> Point mask/FFT`, then toolbar/menu `Pair correlation` or `Feature-to-lattice`; module `probeflow.gui.dialogs.image_viewer`.

Reproduction steps:

1. Open an image.
2. Use the Measure tab `Feature maxima` panel to detect maxima.
3. Confirm the overlay and measurement row exist; optionally compute the point-mask FFT.
4. Open pair correlation or feature-to-lattice comparison.

Observed result:

Pair correlation and feature-to-lattice source collectors only read:

- legacy `_feature_finder_dlg.result`
- selected point ROIs
- all point ROIs

They do not read the newer `ImageMeasurementController._feature_points` list created by the Measure tab (`probeflow/gui/dialogs/image_viewer.py:2511` and `probeflow/gui/dialogs/image_viewer.py:2546`). If no legacy feature dialog result or point ROIs exist, the user sees "Run Feature finder or select point ROIs first" (`probeflow/gui/dialogs/image_viewer.py:2578` and `probeflow/gui/dialogs/image_viewer.py:2619`).

Expected scientific/user result:

Feature points detected in the Measure tab should be a first-class point source for pair correlation and feature-to-lattice comparison, with their detection parameters and ROI scope preserved.

Why it matters:

This blocks a natural end-to-end workflow: detect features, make a point mask/FFT, then quantify ordering or compare to the lattice. It also encourages users to rerun a different feature finder with different parameters, creating inconsistent downstream results.

Recommended fix class: UI, data lineage, provenance, tests.

Stage 2 target:

Create a central point-source registry or expose Measure-tab feature points through `_collect_point_sources_m/_px`. Preserve source name, detection settings, ROI scope, coordinate units, and source path in downstream measurements.

#### PF-STAGE1-004

Severity: P1
Workflow: Find maxima/minima -> threshold -> feature points -> feature mask -> FFT
Location: UI paths `Measure tab -> Feature maxima` and legacy `Feature finder`; modules `probeflow.gui.widgets.feature_detection_panel`, `probeflow.gui.dialogs.feature_finder`, `probeflow.gui.dialogs.image_viewer`.

Reproduction steps:

1. Open an image with a meaningful dark feature population.
2. Try to detect minima from the Measure tab.
3. Try again with the legacy Feature Finder dialog.
4. Try to restrict the legacy dialog to the active ROI launched from the image viewer.

Observed result:

The Measure tab is explicitly maxima-only (`probeflow/gui/widgets/feature_detection_panel.py:76`; detection calls `detect_local_maxima` at `probeflow/gui/viewer/image_measurements.py:488`). The legacy Feature Finder supports maxima/minima and feature images (`probeflow/analysis/feature_finder.py:39`, `probeflow/gui/dialogs/feature_finder.py:425`), but the image viewer opens it without passing an active ROI mask (`probeflow/gui/dialogs/image_viewer.py:2389`). The legacy CSV export contains `index,x_px,y_px,x_nm,y_nm,value` only, without source file, channel, ROI, mode, threshold, smoothing, or units metadata (`probeflow/analysis/feature_finder.py:217`).

Expected scientific/user result:

There should be one coherent feature workflow, or the UI should clearly distinguish "quick maxima measurement" from "legacy exploratory feature finder." Minima, ROI scope, threshold mode, smoothing, and point-mask/FFT provenance should be available consistently.

Why it matters:

For STM data, maxima and minima can correspond to different physical objects depending on bias/channel. Splitting them across two paths with different metadata makes it easy to mix results with different assumptions.

Recommended fix class: UI, provenance, export, docs, tests.

Stage 2 target:

Unify feature detection settings behind one controller, or make the legacy dialog feed the same measurement/source registry. Ensure active ROI scope is passed when requested and included in point exports.

#### PF-STAGE1-005

Severity: P1
Workflow: Fit lattice grid -> ideal lattice -> linear correction -> apply/export
Location: UI path `Image Viewer -> Grid -> Correction`; modules `probeflow.gui.lattice_grid_tool`, `probeflow.processing.gui_adapter`.

Reproduction steps:

1. Open an image and fit a lattice grid.
2. Set an ideal lattice, enable/disable preserve orientation, choose interpolation/fill, and apply correction.
3. Save a processed export/provenance JSON.
4. Inspect the canonical processing state in the export.

Observed result:

The correction dialog constructs rich params including `matrix`, `full_matrix`, `preserve_orientation`, `polar_rotation_deg`, measured vectors, and ideal lattice values (`probeflow/gui/lattice_grid_tool.py:1231`). The canonical GUI adapter keeps only `matrix`, `expand_canvas`, `interpolation`, `fill_mode`, and optional `fill_value` for `affine_lattice_correction` (`probeflow/processing/gui_adapter.py:237`).

Expected scientific/user result:

Processed exports should preserve enough correction context to interpret and reproduce the transform: measured and ideal lattice vectors, physical matrix, pixel matrix, preserve-orientation setting, rotation removed, interpolation, fill, and canvas expansion.

Why it matters:

The corrected pixels can be reproduced from the matrix, but the scientific reason for the transform and the measured-to-ideal lattice mapping are lost or harder to audit. This is a provenance gap for one of the highest-impact operations in the app.

Recommended fix class: provenance, export, docs, tests.

Stage 2 target:

Extend canonical `ProcessingStep("affine_lattice_correction", ...)` to retain the full correction metadata already built by the dialog. Add an export regression test that asserts measured/ideal vectors and preserve-orientation survive.

## 4. Medium-Priority Findings

#### PF-STAGE1-006

Severity: P2
Workflow: Feature points -> pair correlation; Feature points -> lattice comparison
Location: modules `probeflow.gui.dialogs.pair_correlation`, `probeflow.gui.dialogs.feature_lattice_dialog`, `probeflow.gui.dialogs.image_viewer._to_dock_result`.

Reproduction steps:

1. Run pair correlation or feature-to-lattice comparison from a legacy feature result or point ROI source.
2. Add the result to the measurement table.
3. Export measurements as CSV or JSON.

Observed result:

Pair correlation and feature-lattice dialogs create legacy `analysis.measurements.MeasurementResult` rows (`probeflow/gui/dialogs/pair_correlation.py:285`, `probeflow/gui/dialogs/feature_lattice_dialog.py:347`). They are converted by `_to_dock_result`, which maps `source_path` to the short source label and only carries a minimal context of summary and ROI id (`probeflow/gui/dialogs/image_viewer.py:1730`).

Expected scientific/user result:

The exported measurement should include the actual source file, point source name, ROI/area source, binning or match radius, lattice basis, pixel calibration, and assumptions.

Why it matters:

The exported table may contain plausible metrics without enough context to reproduce or interpret them later.

Recommended fix class: provenance, export, UI, tests.

#### PF-STAGE1-007

Severity: P2
Workflow: Find maxima/minima -> threshold -> feature points -> feature mask -> FFT
Location: modules `probeflow.measurements.fft_points`, `probeflow.gui.viewers.image_measurements`.

Reproduction steps:

1. Detect feature maxima in the Measure tab.
2. Export the point mask CSV and point-mask FFT CSV.
3. Open the CSVs without the original GUI session.

Observed result:

The point mask CSV is only a 0/1 matrix (`probeflow/measurements/fft_points.py:88`). The FFT CSV contains columns `qx,qy,magnitude,unit` (`probeflow/measurements/fft_points.py:103`). The summary measurement records some context, but the standalone CSVs do not include source path, image shape, pixel size, radius, shape mode, point count, or threshold settings.

Expected scientific/user result:

Derived mask and FFT exports should be self-describing or accompanied by a JSON sidecar.

Why it matters:

A binary feature mask and its FFT are not interpretable without knowing how the mask was generated, how coordinates map to physical units, and which image/channel/ROI was used.

Recommended fix class: export, provenance, tests.

#### PF-STAGE1-008

Severity: P2
Workflow: Line ROI -> line profile -> periodicity -> measurement table
Location: module `probeflow.gui.viewer.image_measurements`.

Reproduction steps:

1. Draw a line ROI.
2. Estimate periodicity with non-default method, width, background, smoothing, and period bounds.
3. Copy the periodicity result or export the periodicity profile CSV.

Observed result:

The measurement row records method/background/smoothing/quality/width (`probeflow/measurements/image.py:240`). The copy path stores only background and smoothing in `_last_periodicity_settings` (`probeflow/gui/viewer/image_measurements.py:262`), and the profile CSV writes only `s_m,s_nm,z_raw,z_processed` (`probeflow/gui/viewer/image_measurements.py:317`).

Expected scientific/user result:

Copied text and profile CSV should include method, line width, period bounds, ROI id/name, source/channel, and quality/message.

Why it matters:

The measurement table is defensible, but users often share copied text or profile CSVs. Those secondary exports should not lose the assumptions needed to interpret the period estimate.

Recommended fix class: export, provenance, UI.

#### PF-STAGE1-009

Severity: P2
Workflow: Feature points -> pair correlation
Location: modules `probeflow.analysis.pair_correlation`, `probeflow.gui.dialogs.pair_correlation`.

Reproduction steps:

1. Compute pair correlation with fewer than 20 points.
2. Compute again with an active area ROI and no explicit edge correction.
3. Add the result to the measurement table and export it.

Observed result:

The backend correctly warns that edge correction is not applied (`probeflow/analysis/pair_correlation.py:121`), and the dialog displays the result message (`probeflow/gui/dialogs/pair_correlation.py:242`). The measurement row includes `quality`, but not the edge-correction limitation, bin width/range, or ROI area in exported context.

Expected scientific/user result:

The warning and binning/area context should survive into the measurement table and export, not only the dialog label.

Why it matters:

Pair correlation is very sensitive to point count, ROI geometry, and boundary correction. Without those fields, exported `g(r)`-derived summaries can be overinterpreted.

Recommended fix class: provenance, export, docs.

#### PF-STAGE1-010

Severity: P2
Workflow: Spectroscopy file -> inspect trace -> smooth/derivative/export
Location: modules `probeflow.gui.dialogs.spec_viewer`, `probeflow.spectroscopy.transforms`, `probeflow.spectroscopy.export`.

Reproduction steps:

1. Open spectra in the overlay viewer.
2. Apply smoothing, derivative, normalization, outlier masking, and vertical offsets.
3. Export CSV/JSON/TXT.

Observed result:

The displayed-transform design is non-destructive and exports options/source/units/excluded indices (`probeflow/spectroscopy/transforms.py:38`, `probeflow/spectroscopy/export.py:15`). However, derivative uses generic `np.gradient` after smoothing and only enforces strict monotonic x (`probeflow/processing/spectroscopy.py:72`); the UI labels the derivative option as `dI/dV` even when the selected signal may not literally be current-vs-bias (`probeflow/gui/dialogs/spec_viewer.py:1388`).

Expected scientific/user result:

The UI/export should make clear that this is a numerical derivative of the displayed channel with respect to the x-axis, after smoothing and before normalization/outlier masking. The exported option metadata is good; the visible label should be less likely to imply lock-in dI/dV unless the channel metadata supports that interpretation.

Why it matters:

In STM spectroscopy, measured lock-in dI/dV and numerical dI/dV from I(V) are not equivalent. The app should help prevent accidental overinterpretation.

Recommended fix class: UI, docs, provenance.

## 5. Physics and Unit Concerns

- Real-space lattice angles: affected output is grid measurement angle and distortion-panel measured angle. Use calibrated vectors before angle calculation.
- Reciprocal-space lattice angles: `format_reciprocal_measurements` also uses `grid.angle_deg()` after computing calibrated q vectors (`probeflow/analysis/lattice_grid.py:471`). Use q-space vectors for the reciprocal angle.
- Feature-to-lattice displacement: affected outputs are RMS displacement, mean displacement, match interpretation, and measurement export. Use per-axis physical residuals.
- Feature-to-lattice occupancy: occupancy is reported from sites inside pixel image bounds (`probeflow/analysis/feature_lattice.py:120`). It is meaningful only when the point source covers the same image/ROI region as the lattice. Export should record the region used for occupancy.
- Point-mask FFT units: q axes use cycles/nm from mask pixel size (`probeflow/measurements/fft_points.py:54`). This is reasonable, but exports need mask-generation metadata and should name whether values are cycles/nm or angular wavevector if future FFT tools use rad/nm.
- ROI statistics: area uses both pixel axes and finite-pixel counts are recorded (`probeflow/measurements/image.py:37`). RMS roughness is calculated as population standard deviation around the finite-pixel mean (`probeflow/measurements/image.py:60`); label/export should keep that definition visible.
- Line profiles: physical line length and swath averaging are handled through calibrated x/y pixel sizes; measurement rows preserve width and method. Secondary copy/profile exports are the weak point.
- Spectroscopy: x/y units survive displayed exports; numerical derivative units are derived as `y_unit/x_unit` (`probeflow/spectroscopy/transforms.py:63`). UI language should distinguish numerical derivative from measured derivative channels.

## 6. Misleading-Output Risks

- A calibrated non-square-pixel scan can report a wrong lattice angle without warning.
- Feature-to-lattice residuals can be directionally wrong on non-square pixels while shown in physical units.
- Pair correlation and feature-lattice tools can ignore the latest visible detected maxima, causing users to rerun another detector and accidentally compare different point sets.
- Legacy feature-finder CSV exports include coordinates but not enough metadata to distinguish maxima from minima or thresholds.
- Lattice correction exports may not explain what physical lattice was corrected to, even when the transform matrix is present.
- Point-mask FFT CSVs can outlive their context and become uninterpretable.
- Spectroscopy `dI/dV` UI wording can imply a lock-in measurement when the tool is computing a numerical derivative.

## 7. Workflow-Specific Notes

### Open Folder, Browse, Display, Export

Purpose: load STM scans, choose channels/images, adjust display, and create interpretable visual exports.

User path: browser/open folder -> image viewer -> channel/display controls -> `Export -> Save PNG`, `Save processed image`, or `Save provenance`.

Expected result: image export with display settings, scan range, processing state, channel, and optional ROI/provenance.

Findings: no P0/P1 found. Export blocking for stale ROI references and processing errors is good (`probeflow/gui/dialogs/image_viewer.py:2808`, `probeflow/gui/dialogs/image_viewer.py:2847`).

### Bad Lines and STM Background

Purpose: remove scan-line artifacts and subtract row/STM background while preserving auditable parameters.

User path: processing panel bad-line controls -> preview/apply; toolbar/menu `STM BG` -> background dialog -> preview/apply -> export.

Expected result: corrected image with bad-line and STM background parameters recorded.

Findings: no P0/P1 found. The bad-line preview reports skipped adjacent-line corrections (`probeflow/gui/viewer/bad_line_preview.py:105`). STM background exposes fit region/model/statistic/jump/blur and records canonical params (`probeflow/gui/dialogs/stm_background.py:229`, `probeflow/processing/gui_adapter.py:124`). Keep the ROI-fit/full-image-apply distinction visible.

### Line Profile and Periodicity

Purpose: measure physical profile length/height and periodicity along a line ROI.

User path: draw line ROI -> profile/periodicity controls -> measurement table/export.

Expected result: calibrated line length/profile, period estimate, quality flag, and method metadata.

Findings: main measurement path works. Secondary copy/CSV export should carry full method context (PF-STAGE1-008).

### Area ROI Statistics

Purpose: summarize height distribution/roughness in a physical area.

User path: draw area ROI -> ROI statistics -> measurement table/export.

Expected result: finite-pixel stats, area, units, ROI id/name, data basis.

Findings: no high-priority issue found. Both legacy and newer implementations are scientifically reasonable, but should eventually be unified to avoid table/export drift.

### Feature Detection, Mask, and FFT

Purpose: detect atomic/defect/feature positions, inspect them as points, derive a binary mask, and inspect ordering in Fourier space.

User path: Measure tab `Feature maxima`; legacy Feature Finder for maxima/minima; Point mask/FFT panel.

Expected result: visible point overlay, source/ROI/detection metadata, point exports, mask/FFT exports.

Findings: split maxima/minima workflows and incomplete standalone mask/FFT exports (PF-STAGE1-004, PF-STAGE1-007).

### Pair Correlation

Purpose: estimate radial ordering statistics from feature points.

User path: feature points or point ROIs -> Pair correlation dialog -> Add to measurements.

Expected result: physical distances, density/NN/peak, clear quality and edge-correction limits.

Findings: blocked from newer feature maxima source and exported context is thin (PF-STAGE1-003, PF-STAGE1-009).

### Lattice/Grid and Linear Correction

Purpose: measure lattice vectors, compare measured lattice to ideal lattice, and apply measured-to-ideal correction.

User path: Grid tool -> tune lattice -> correction panel -> preview/apply -> export.

Expected result: calibrated a/b/angle/area, clear correction direction, preserved correction metadata.

Findings: non-square pixel lattice angle bug and correction provenance truncation (PF-STAGE1-001, PF-STAGE1-005). Correction direction/options are otherwise clear: measured -> ideal label, preserve-orientation default, expand-canvas default, interpolation/fill controls, grid hidden after apply.

### Feature-to-Lattice Comparison

Purpose: compare detected feature points to an active lattice, report registry, duplicates, off-lattice points, displacement, and occupancy.

User path: run feature finder or point ROIs -> active lattice grid -> Feature-to-lattice dialog -> Add to measurements.

Expected result: matched/off-lattice/duplicate counts and calibrated residuals.

Findings: non-square displacement bug, missing Measure-tab source, and thin exported context (PF-STAGE1-002, PF-STAGE1-003, PF-STAGE1-006).

### Spectroscopy

Purpose: inspect point spectra, apply non-destructive display transforms, measure deltas, and export displayed traces.

User path: spectroscopy browser/overlay viewer -> choose signal -> smoothing/derivative/normalization/outliers -> export CSV/JSON/TXT.

Expected result: transformed displayed traces with source, units, options, and excluded indices.

Findings: export path is strong; UI wording around numerical derivative should be tightened (PF-STAGE1-010).

## 8. Recommended Stage 2 Fix List

1. Fix calibrated lattice angle calculations in real and reciprocal grid formatting and the distortion panel. Add non-square-pixel tests.
2. Fix feature-to-lattice physical residuals by computing per-axis physical displacement per assignment. Include residual vectors in exported assignments/measurement context.
3. Add a central feature-point source registry used by point mask/FFT, pair correlation, and feature-to-lattice comparison.
4. Unify legacy Feature Finder and Measure-tab feature detection, or make their separation explicit and ensure both emit full metadata.
5. Preserve full affine lattice correction metadata in canonical processing state and processed export provenance.
6. Enrich pair-correlation and feature-lattice measurement exports with point source, ROI/area, bin/match settings, lattice basis, pixel calibration, and assumptions.
7. Add JSON sidecars or metadata headers for point-mask CSV and point-mask FFT CSV exports.
8. Add full method/context headers to periodicity copy/profile CSV exports.
9. Tighten spectroscopy derivative labeling: "numerical dy/dx" unless metadata confirms a current-vs-bias interpretation.
10. Consolidate legacy and new measurement result models so table/export conversion is not lossy.

## 9. Low-Priority Polish and Backlog

- Add human-friendly labels for `pair_corr` and `feat_lattice` in `KIND_LABELS`; `line_periodicity` is handled by `measurement_main_value` but not by the label map.
- Give pair-correlation and feature-lattice rows better default main values instead of falling back to the first dict key.
- In ROI statistics, include a short RMS roughness definition in details/export context.
- In feature exports, include the image coordinate origin convention (`x=column`, `y=row`, origin upper-left) and whether y increases downward.
- Consider adding a warning when pair-correlation point count is low but still above the hard failure threshold.
- Consider displaying point-mask radius in physical units alongside pixels.
- Consider exporting feature-to-lattice per-assignment tables, not only summary measurement rows.
- Add a "data basis" badge in dialogs when a tool is operating on raw, processed, displayed, ROI-derived, or binary-mask data.
- Add regression tests for stale ROI references through all derived export paths, not only image exports.
