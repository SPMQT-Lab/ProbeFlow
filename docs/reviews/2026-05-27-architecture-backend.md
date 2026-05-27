# ProbeFlow Backend Architecture & Maintainability Review
Date: 2026-05-27 (written 2026-05-28 retry, sequential mode)
Reviewer: backend-architecture agent (deep review pass)

## Summary

- 18 findings: S0=0, S1=9, S2=7, S3=2
- Top 3 concerns:
  1. Three competing provenance/history structures (`ProcessingState`, `ProcessingHistory`, `ScanGraph`) coexist; the most ambitious one (`ScanGraph`) has zero production callers but is re-exported as public API. This is the largest source of contributor confusion.
  2. Stage 2's measurement-result canonicalisation produced two identically named `MeasurementResult` (in `probeflow.analysis.measurements` and `probeflow.measurements.models`) and two identically named `FeaturePoint` (in `analysis.feature_finder` and `measurements.models`). The legacy `analysis.measurements` is now used only by tests and is ready for removal.
  3. Re-emerged duplication: two parallel local-maxima detectors, two `roi_statistics` flavours, six metre→nm/Å formatters, two-name geometric-op vocabulary inside one module. The Stage 2 seams (canonical measurement adapter, ROI context helper, lattice correction helper) were correctly used by recent code, but the older parallel kernels were never folded in.

**Overall posture**: The backend layering is genuinely clean — outside of `cli/commands/gui.py`, no backend module imports `probeflow.gui`, and the boundary-rules docstrings in subpackage `__init__.py` files accurately describe a layered architecture. The trouble is that those docstrings describe an _intended_ architecture (Session → Probe → ScanGraph, plugin registry) that exists in dataclass form but is not yet wired to any production code path. Meanwhile, the actually-running architecture uses linear `ProcessingState` + `ProcessingHistory`, and the package is growing duplicates inside that working model (two `MeasurementResult`, two `FeaturePoint`, two local-maxima kernels, two `roi_statistics`, six unit formatters). The package is reaching the size where ambiguity about "which seam is canonical" will cost real contributor time. The Stage 2 surgical-seam approach is the right tool to address this; the aspirational graph/plugin surface should either be wired in or hidden.

## Findings

### 1. [S1] Two parallel `MeasurementResult` dataclasses with incompatible schemas

**Location**: `probeflow/analysis/measurements.py:18-32` (legacy) vs `probeflow/measurements/models.py:10-24` (canonical).

**Problem**: `probeflow.analysis.measurements.MeasurementResult` and `probeflow.measurements.models.MeasurementResult` both exist with different field names (`id` vs `measurement_id`, `source` vs `source_label`/`source_path`, `units: dict` vs `x_unit`/`y_unit`/`z_unit`). The Stage 2 adapter (`measurements/adapters.legacy_measurement_to_result`) bridges them. Production code (analysis, GUI, measurements, CLI) only imports the canonical one; the legacy `analysis.measurements.MeasurementResult` and its `results_to_csv` / `result_to_text` helpers are now only synthesised by `tests/test_measurements_core.py:53` and `tests/test_gui_processing_panel.py:1226`. The GUI mixin `probeflow/gui/viewer/image_viewer_roi_mixin.py:163-167` still imports the adapter, but a non-canonical result reaches `_add_dialog_measurement_result` only via the `else:` branch — no in-tree caller is known to exercise it.

**Why it matters**: Two classes with the same name in different packages is a contributor trap (autocomplete will offer both). Stage 2's release-plan item "Remove compatibility measurement shims only after no supported caller depends on legacy result rows" appears already satisfied — only the tests still synthesise legacy rows.

**Suggested fix**: Confirm via grep that nothing in `probeflow/gui/`, `probeflow/cli/`, `probeflow/analysis/` constructs `probeflow.analysis.measurements.MeasurementResult`, then delete `probeflow/analysis/measurements.py` and update the two tests to construct the canonical `MeasurementResult` directly (preferred) or inline the legacy minimal class as a test fixture. Also delete `measurements/adapters.py:legacy_measurement_to_result` once `_to_dock_result` in `image_viewer_roi_mixin.py` is shown to be dead.

### 2. [S1] Two parallel `FeaturePoint` dataclasses with different field schemas

**Location**: `probeflow/analysis/feature_finder.py:18-23` (3 fields: `x_px`, `y_px`, `value`, `label`) vs `probeflow/measurements/models.py:28-39` (9 fields: `point_id`, `x_px`, `y_px`, `x_phys`, `y_phys`, `z_value`, `channel`, `source_label`, `roi_id`).

**Problem**: `analysis.feature_finder.FeaturePoint` is a thin detection record. `measurements.models.FeaturePoint` is the richer point-source record produced by `measurements/features.detect_local_maxima` and consumed by `measurements/fft_points.py`. Both are part of the public surface — the first is reachable via `from probeflow.analysis.feature_finder import FeaturePoint`; the second is re-exported from `probeflow.measurements`. A reader sees both names and cannot tell which is canonical.

**Why it matters**: The Stage 1 follow-up explicitly listed feature-source unification as an outstanding action. Two `FeaturePoint` classes plus a `gui/roi_context.PointSource` (a third shape for the same concept) make the seam where downstream point sources should agree harder to enforce. Bug fixes to attribute names (e.g. renaming `value` → `z_value`) will silently divide the codebase.

**Suggested fix**: Make `analysis.feature_finder.find_image_features` return the canonical `measurements.models.FeaturePoint`. Most fields are already computable inside `find_image_features` (`x_phys`/`y_phys` need pixel sizes, but the function already does enough I/O for that to be a reasonable kwarg). Delete the local 3-field `FeaturePoint` dataclass and re-export the canonical one for backward compatibility.

### 3. [S1] Duplicate local-maxima detection algorithms

**Location**: `probeflow/analysis/feature_finder.py:39-165` (`find_image_features`) vs `probeflow/measurements/features.py:12-95` (`detect_local_maxima`).

**Problem**: Both perform the same operation: NaN-fill, optional Gaussian smooth, ROI mask, threshold, `maximum_filter` peak finding, descending-value sort, NMS-by-radius. The two implementations have diverged in details — `find_image_features` supports `between`/`minima` modes and a single absolute threshold, while `detect_local_maxima` supports `mean_offset`/`median_offset`/`percentile` thresholding modes and a `min_prominence` filter — but they share their core. A bug fix to one (e.g. the NMS implementation in `_far_enough`) will not propagate to the other.

**Why it matters**: Two NMS implementations of the same physics-relevant kernel guarantee future divergence in detection results between the feature-finder dialog and the measurement-tab feature-maxima tool, contradicting Stage 1's explicit goal of unifying feature workflows.

**Suggested fix**: Pick one as the kernel (`detect_local_maxima` is closer to the canonical result schema). Refactor the other into a thin wrapper that forwards to the kernel and adapts the return type (`FeatureDetectionResult` can carry the canonical `FeaturePoint` tuple plus the diagnostic strings). Add a single shared `_far_enough` and a `_threshold_value` that supports both threshold-mode vocabularies.

### 4. [S1] Two `roi_statistics` flavours producing the same `kind="roi_stats"`

**Location**: `probeflow/measurements/image.py:12-70` (`roi_statistics`) vs `probeflow/analysis/roi_statistics.py:14-99` (`compute_roi_statistics`).

**Problem**: Both compute mean/median/std/RMS/peak-to-peak/area for an image region and emit a canonical `MeasurementResult` with `kind="roi_stats"`. Their key/unit choices differ:
- `measurements/image.py` returns keys `mean_height`, `median_height`, `std_height`, `rms_roughness`, `min_height`, `max_height`, `peak_to_peak`, `area`, `n_finite_pixels`, `n_nonfinite_pixels`. `area` is in pixel² unless `pixel_size_x/y` are passed.
- `analysis/roi_statistics.py` returns `area_m2`, `area_nm2`, `n_finite_pixels`, `mean_height`, `median_height`, `std_height`, `rms_roughness`, `peak_to_peak` and a pre-formatted `summary` string in `context`. Always requires SI pixel sizes.

The first is used by `tests/test_processing_seams.py:24`; the second is used by `probeflow/gui/viewer/image_viewer_tools_mixin.py:291`.

**Why it matters**: Two functions emitting the same `kind` with overlapping but non-identical value keys make the measurement table's column layout depend on which producer fired. Downstream consumers (CSV export, clipboard) silently get different schemas.

**Suggested fix**: Keep `measurements/image.py:roi_statistics` as the kernel (it's already in the canonical `measurements/` package). Move the `summary`-formatting and `area_nm2` convenience into `measurements/image.py` (or the proposed shared unit-formatting helper from finding 5). Delete `analysis/roi_statistics.py` after updating `image_viewer_tools_mixin.py:291` to call the kernel and format the summary at the GUI seam.

### 5. [S2] Six parallel metre-to-display unit formatters

**Location**:
- `probeflow/analysis/simple_measurements.py:132-140` (`_fmt_m`)
- `probeflow/analysis/line_periodicity.py:409-421` (`_fmt_m` + `format_period`)
- `probeflow/analysis/roi_statistics.py:102-110` (`_fmt_z` wrapping `_fmt_m`)
- `probeflow/analysis/lattice_grid.py:445-526` (`_choose_unit` + inline pm/Å/nm logic in `_period_str`/`_length_str`)
- `probeflow/gui/viewer/image_measurements.py:76-80` (`_format_period_bound_nm`)
- `probeflow/gui/widgets/measurement_table.py:332-335` (`_fmt_value`)
- `probeflow/analysis/spec_plot.py:317-399` (`_UNIT_PREFIX_TABLE` + `choose_display_unit` + `lookup_unit_scale`)

**Problem**: Seven different metre→pm/Å/nm/µm formatters exist. Each has slightly different breakpoints (`< 5e-11`, `< 1e-9`, `< 1.0 nm`, `< 0.1 nm`, etc.). `spec_plot.choose_display_unit` is the most thoughtful (prefix table; handles `A` and `V` too) but lives in a matplotlib plotting module so other backend code does not find it.

**Why it matters**: Inconsistent display across tabs/dialogs; Stage 2 explicitly listed "Add small unit-formatting and text-export helpers" as the next step. Risk of silently miscounting orders of magnitude when copy-pasting between formatters. The combined complexity of the six helpers is larger than the proposed shared one.

**Suggested fix**: Promote `spec_plot.choose_display_unit` + `lookup_unit_scale` (and add `format_length_m`, `format_height_m`, `format_period_m` convenience returning `(value_str, unit_str)`) into a small new module `probeflow/measurements/formatting.py` (or `probeflow/core/units.py`). Replace each `_fmt_m`/`_fmt_value` with a call to it. Keep `format_period` as a thin wrapper exposed for the public periodicity API.

### 6. [S1] `provenance/graph.py` (`ScanGraph`/`ImageNode`/`MeasurementNode`) is exported as public API but has zero non-test callers

**Location**: `probeflow/provenance/__init__.py:21-30`, `probeflow/provenance/graph.py` (325 LOC).

**Problem**: The graph model is fully implemented (root/derived nodes, materialize, JSON round-trip, `OpRegistry` indirection) and re-exported from `probeflow.provenance`, but `grep` finds only `tests/test_graph.py` consuming it. Production processing/provenance uses the linear `ProcessingHistory` in `probeflow/provenance/records.py` instead. Module docstrings everywhere (in `processing/__init__.py:1-18`, `analysis/__init__.py:1-17`, `plugins/__init__.py:1-17`, `core/__init__.py:1-17`) describe the scan-owned graph as the intended home for nodes — so a contributor reading those docstrings will believe the graph is the canonical home, then discover the actual home is `ProcessingState` + `ProcessingHistory`, which together form a second, parallel structure.

**Why it matters**: This is the largest "competing structures" issue. Either the graph is the future and the linear history is the placeholder (then the boundary-rules docstrings are accurate), or the graph is dead aspirational code (then those docstrings actively mislead). Both states cannot coexist quietly.

**Suggested fix**: Decide explicitly:
- If keep: add at least one in-tree call site that converts `ProcessingHistory` → `ScanGraph` (e.g. a `ProcessingHistory.to_graph()` helper used by the export sidecar writer), or wire `materialize_image` into the processing dispatch. Document the migration plan as a top-level TODO and a `docs/provenance_graph_roadmap.md`.
- If shelve: move `graph.py` to `probeflow/provenance/_experimental.py`, drop the public re-exports from `provenance/__init__.py`, and replace the "intended graph architecture" paragraphs in the subpackage docstrings with a one-line note that the linear `ProcessingHistory` model is the production source of truth.

### 7. [S1] `probeflow/plugins/` is a 127-LOC scaffold with no production consumer

**Location**: `probeflow/plugins/` (5 files, 127 LOC total; `plugins/adapters/__init__.py:1` is one line of docstring).

**Problem**: `PluginRegistry`, `PluginSpec`, `PluginOperation`, and `manifest_from_spec` are implemented but the only references outside the package are `whitelist.py:60-66` (vulture quieting) and `tests/test_layout_compatibility.py:129` (asserting the names import). No CLI command, GUI panel, or processing path discovers operations from the registry. `plugins/manifest.py:5` openly states "Status: scaffolding — not yet wired". The empty `plugins/adapters/` subpackage is even more misleading.

**Why it matters**: A 4-class plugin API in the public package signals to contributors that a plugin system exists. They will not realise their plugin cannot actually be loaded by ProbeFlow. The presence of an `adapters/` subdirectory with only a docstring `__init__.py` reinforces the impression of a real system.

**Suggested fix**: Either (a) wire it minimally: register at least the in-tree processing operations from `processing/state.py:_SUPPORTED_OPS` into a default `PluginRegistry` at import time, and have one CLI/GUI path enumerate it — even read-only is enough to show the system is real; or (b) hide it: rename to `probeflow/_plugins/` (leading underscore), remove the empty `adapters/` directory, and add a one-line README note that plugin support is not yet shipped.

### 8. [S2] Spectroscopy split across three packages with overlapping responsibilities

**Location**: `probeflow/processing/spectroscopy.py` (numeric kernels — `smooth_spectrum`, `numeric_derivative`, `normalize`, `crop`, `average_spectra`, `current_histogram`); `probeflow/spectroscopy/` (display helpers and trace transforms, imports from `processing.spectroscopy`); `probeflow/analysis/spec_plot.py` (matplotlib plotting + the unit-prefix table, also imports `processing.spectroscopy`); `probeflow/io/spectroscopy.py` (parser).

**Problem**: The current split is layered correctly in the abstract — kernels at the bottom, display helpers and plotting on top — but the *names* do not tell a reader where to look:
- `processing/spectroscopy.py` is a single file inside the otherwise-image-focused `processing/` package.
- The `spectroscopy/` package handles display-state of traces.
- `analysis/spec_plot.py` lives in `analysis/` despite being neither a measurement algorithm nor an export.
- The unit-prefix table (`_UNIT_PREFIX_TABLE`) is a generic helper that has nothing to do with spectroscopy plotting yet lives in `spec_plot.py`.

**Why it matters**: A future contributor adding a new spectroscopy normalization or trace transform has three plausible homes and no rule to pick one. Stage 2 explicitly listed "Clean up spectroscopy and plotting duplication" as a remaining item.

**Suggested fix**: Move `processing/spectroscopy.py` into `probeflow/spectroscopy/_kernels.py` and re-export the names from `probeflow.spectroscopy` so the package is the single home. Keep `analysis/spec_plot.py` (the matplotlib layer) but move its unit-prefix helpers to the shared formatting module from finding 5. The `io/spectroscopy.py` parser is fine where it is.

### 9. [S1] Inconsistent geometric-op naming vocabulary inside one module

**Location**: `probeflow/processing/state.py:52-54` and `:698-699` use `rotate_90_cw`/`rotate_180`/`rotate_270_cw`. `probeflow/processing/state.py:791-806` (`apply_geometric_op_to_scan`) uses the shorter `rot90_cw`/`rot180`/`rot270_cw`. `probeflow/processing/state.py:819` mixes both vocabularies in a single tuple: `("rot90_cw", "rot270_cw", "rotate_90_cw", "rotate_270_cw")`. The kernel functions in `probeflow/processing/geometry.py:475-485` are named `rotate_90_cw` etc.

**Problem**: Two name vocabularies for the same op live in the same file. `apply_processing_state` step names use `rotate_*_cw`; `apply_geometric_op_to_scan` operation names use `rot*_cw`. The `_LOSSLESS` frozenset and the body of `apply_geometric_op_to_scan` only handle `rot*_cw`, so passing `rotate_90_cw` would raise `ValueError` even though it's the canonical name elsewhere. The scan-range swap block defensively accepts both, but the per-plane loop just above does not — that asymmetry is fragile.

**Why it matters**: Caller-side confusion (which name does this function accept?) and a latent path-dependent bug. `apply_geometric_op_to_scan` is currently only exercised by `tests/test_roi.py` and `tests/test_geometric_ops.py`, all of which use the short-form `rot*_cw`. `gui_adapter.py:236` emits only the long-form `rotate_*_cw`. If any future production caller routes a `processing_state` op through `apply_geometric_op_to_scan`, it would fail. The mixed tuple at line 819 looks like defensive code anticipating that case, but it does not work because the per-plane loop above raises first.

**Suggested fix**: Pick one vocabulary (`rotate_90_cw` matches the kernel function names and the `_SUPPORTED_OPS` set). Convert `apply_geometric_op_to_scan` to use the long names internally; remove the short-form `rot*_cw` branches. Keep a deprecation shim if any external CLI/script may still pass short names.

### 10. [S2] Duplicate point-rasterization helpers

**Location**: `probeflow/analysis/feature_finder.py:168-214` (`feature_points_to_image`) vs `probeflow/measurements/fft_points.py:26-51` (`points_to_mask`).

**Problem**: Both rasterize a list of `(x, y)` feature points to a 2-D array with disk dilation. `feature_points_to_image` returns a `float64` image with optional Gaussian smoothing; `points_to_mask` returns a `bool` mask with `disk`/`square` shape modes. The kernels overlap on the disk-dilation loop and the bounds checks.

**Why it matters**: Two implementations of the same routine; bug fixes (e.g. handling sub-pixel `x_px`, anti-aliased disk edges) will diverge.

**Suggested fix**: Factor the shared inner loop (`disk_pixels_for_point(cx, cy, r, ny, nx)`) into `measurements/fft_points.py` or a new `measurements/raster.py`. Have both wrappers call it.

### 11. [S2] Duplicate ROI→mask resolution helpers

**Location**: `probeflow/measurements/image.py:291-308` (`_mask_from_roi_or_mask`), `probeflow/measurements/features.py:146-162` (`_roi_mask`), `probeflow/processing/state.py:405-442` (`_resolve_mask_roi_param` — lookup by id + warning), plus per-call `roi.to_mask(image_shape)` invocations in `probeflow/measurements/features.py`, `probeflow/measurements/fft_points.py`, and `probeflow/processing/state.py:684`.

**Problem**: Four near-identical patterns for "given a ROI or a precomputed mask, return a bool 2-D array of the image's shape, raising if dimensions disagree". The Stage 2 ROI context helper (`probeflow.gui.roi_context`) is the GUI-side counterpart, but no backend-side equivalent exists.

**Why it matters**: Each call site reimplements the shape-mismatch guard differently (some `raise ValueError`, others log and skip with a `UserWarning`). The contract "what does a backend measurement do if a mask is the wrong shape" is not centralised.

**Suggested fix**: Add a `probeflow/measurements/roi_resolve.py` (or extend `core/roi.py`) with one `resolve_roi_to_mask(roi_or_mask, image_shape, *, on_empty="raise"|"return_none") -> np.ndarray` function. Use it from `measurements/image.py`, `measurements/features.py`, `measurements/fft_points.py`, and `processing/state.py`.

### 12. [S2] Two `ProcessingStep` dataclasses (same name, different schemas)

**Location**: `probeflow/processing/state.py:101-117` (`op`, `params`) vs `probeflow/provenance/records.py:127-169` (`step_id`, `operation_id`, `operation_name`, `operation_version`, `parameters`, `input_state_id`, `output_state_id`, `timestamp`, `warnings`).

**Problem**: Two dataclasses both named `ProcessingStep`. The `provenance` one is re-exported via `from probeflow.provenance import ProcessingStep`. `tests/test_provenance_records.py:11` already has to rename one of them at import time (`from probeflow.processing.state import ProcessingState, ProcessingStep as StateStep`) to avoid collision. Production code does not currently import `provenance.ProcessingStep` by name outside `provenance/` itself, but the public `__all__` exposes both.

**Why it matters**: Same-name dataclasses in adjacent packages is a confused name space, even if only the simpler one (`processing.state.ProcessingStep`) is widely imported.

**Suggested fix**: Rename `probeflow.provenance.records.ProcessingStep` to `ProvenanceStep` (or `HistoryStep`). Update `provenance/__init__.py:__all__`. Production code paths are unaffected because nobody imports `provenance.ProcessingStep` by name.

### 13. [S2] `apply_processing_state` opens files on disk (layering surprise in a "pure array" function)

**Location**: `probeflow/processing/state.py:445-469` (`_load_arithmetic_operand_image`), called from `apply_processing_state` at lines 614-633.

**Problem**: `apply_processing_state` is documented as a function that "Apply ``state`` steps in order to ``arr``" and accepts only an ndarray + state + ROI set. But when a step has `op="arithmetic"` and `operand_type="image"`, it transparently calls `load_scan(Path(source_path))` from `probeflow.core.scan_loader`, opening an arbitrary file on disk and reading another scan's plane. There is no warning or escape hatch for sandboxed/offline use.

**Why it matters**: Filesystem access hidden inside a "pure" transform pipeline breaks expectations for batch scripts, tests, and any code that wants deterministic behaviour from a `ProcessingState`. Also makes worker-thread / sandboxed callers indirectly dependent on `probeflow.core.scan_loader` (and its lazy chain into `io/readers/*`). Note: `processing/__init__.py:11-18` explicitly says "Do not add … vendor parsers".

**Suggested fix**: Either (a) accept an `operand_resolver: Callable[[dict], np.ndarray]` kwarg on `apply_processing_state` so callers explicitly opt-in to file resolution (CLI/GUI inject `load_scan`-based default; tests inject a memory dict), or (b) move the file-loading wrapper into `probeflow.processing.gui_adapter.apply_processing_state_to_scan` and have the kernel raise if it sees `operand_type="image"` without an operand pre-attached.

### 14. [S2] `Scan` model (in `core/`) lazy-imports `processing.state` and `processing.history`

**Location**: `probeflow/core/scan_model.py:110, 115, 147, 161, 174` (five lazy `from probeflow.processing.state import ProcessingState` / `from probeflow.processing.history import ...` inside `Scan` methods).

**Problem**: The `core/__init__.py:9-17` boundary docstring says the `core` package "may attach a provenance graph to a probe object, but it must not define ... `ScanGraph`" and that "core may depend on the Scan model without creating import cycles" — but it does not address that `core.Scan` itself depends on `processing.state`. The lazy imports are a circular-import workaround: `processing.state` imports `core.roi`, and `core.scan_model` cannot import `processing.state` at module level. This is a known pattern but it suggests the data model's home is ambiguous: is `ProcessingState` part of the core domain (then it should live in `core/`) or part of `processing/` (then `Scan` should not own a `processing_state` attribute by name)?

**Why it matters**: Five lazy-import sites are a sign that the layer boundary between "the data model" and "what you can do with the data" was drawn in the wrong place. New contributors will write similar lazy imports rather than recognise the underlying issue.

**Suggested fix**: Move `ProcessingState`, `ProcessingStep`, and `apply_processing_state` to `probeflow/core/processing_state.py` (the *state data class* is a core domain object; the kernels in `processing/filters.py` etc. remain in `processing/`). Update `core/scan_model.py` to import normally. `processing.state` becomes a thin re-export shim for compatibility. This also clears the path for `ScanGraph` (if kept) to live in `core/` per the docstring's intent.

### 15. [S2] `processing/analysis.py` lives in `processing/` but contains analysis-style measurements

**Location**: `probeflow/processing/analysis.py:1-220` — defines `gmm_autoclip`, `detect_grains`, `measure_periodicity`.

**Problem**: The `processing/__init__.py:1-18` docstring says "``processing`` contains array-in/array-out numerical transformations" but `processing/analysis.py` exposes operations that produce *measurements* (clip percentiles, grain labels, periodicity peaks) rather than transformed images. `gmm_autoclip` returns a `(low_pct, high_pct)` tuple; `detect_grains` returns `(label_map, n_grains, stats_dict)`; `measure_periodicity` returns a list of peak tuples. These look like they belong in `analysis/` (which already has `features.py`, `lattice.py`, `line_periodicity.py`, `pair_correlation.py`).

**Why it matters**: The naming "analysis inside processing" actively contradicts the boundary docstring. The CLI calls them via `_proc.detect_grains(...)` (see `cli/commands/analysis.py:34`) — that line itself shows the confusion (an `analysis` CLI command importing from a `processing` module called `analysis`).

**Suggested fix**: Move `gmm_autoclip`, `detect_grains`, `measure_periodicity` to `probeflow/analysis/grains.py` (or split). Update `processing/image.py` to drop the `from .analysis import *` star-import. Add a thin compatibility shim if external `probeflow.processing.detect_grains` callers exist (none in the in-tree code; only tests).

### 16. [S2] `processing/filters.py` is 1,106 LOC and covers four unrelated concerns

**Location**: `probeflow/processing/filters.py` — sections (per top-level comments) cover `fourier_filter`, `gaussian_smooth`, `edge_detect`, `fft_soft_border`, `fft_magnitude`, **Bragg shell physics** (`BraggShell`, `bragg_shells`, `first_bragg_q`), **predicted Bragg radius**, **Bragg peak finding in q-annulus** (~650-950), **ellipse fitting** (`fit_axis_aligned_ellipse`), and **piezo correction** (`piezo_correction`).

**Problem**: A file titled "Spatial and frequency-domain image filters" is also the home of Bragg-peak physics, ellipse fitting, and piezo correction — none of which are "filters". The Bragg/ellipse section is ~470 LOC of crystallography-flavoured code that would more naturally live in `analysis/lattice.py` or a new `analysis/fft_lattice.py`.

**Why it matters**: 1,100-LOC files are difficult to navigate, slow to import (everything is pulled even for simple `gaussian_smooth`), and signal that the conceptual unit has overflowed. The Bragg-finding code was added relatively recently and its physics-y density makes the file an unintended mixed-concern landing pad.

**Suggested fix**: Split into `processing/filters.py` (filters only: fourier_filter, gaussian_smooth, edge_detect, fft_soft_border, gaussian_high_pass, periodic_notch_filter) and `analysis/fft_lattice.py` (BraggShell, bragg_shells, first_bragg_q, predicted_bragg_radius, find_bragg_peaks_in_annulus, find_bragg_peaks_in_q_annulus, snap_to_compact_peak_q, fit_axis_aligned_ellipse, piezo_correction). The first becomes ~600 LOC; the second is the new analysis kernel.

### 17. [S3] `cli/commands/__init__.py` docstring is stale (claims migration in progress)

**Location**: `probeflow/cli/commands/__init__.py:1-18`.

**Problem**: The docstring says "Each submodule re-exports the ``_cmd_*`` runner functions that currently live in ``probeflow/cli/_legacy.py``. The submodules are not yet wired into the parser; the active entry point is still ``cli/_legacy.py:_build_parser``." But the migration is done: `probeflow/cli/parser.py:1-58` imports the `_cmd_*` directly from `cli.commands.*`, and `cli/_legacy.py` is now itself a re-export shim that just forwards to `parser.py` and `commands/*`. The only consumer of `_legacy` is `cli/__init__.py:28`, which uses it for backwards compatibility on private dotted imports (e.g. `from probeflow.cli import _cmd_spec_info` in `tests/test_spec_io.py:692`).

**Why it matters**: A new contributor reading this docstring will think the CLI is mid-migration and may add new commands "to _legacy first" per the migration recipe — which is the wrong direction.

**Suggested fix**: Replace the docstring with a one-paragraph statement: "CLI commands. Add new commands here, in the appropriate submodule. The legacy `cli/_legacy.py` is a backward-compatibility re-export shim and should not gain new entries."

### 18. [S3] Many-argument measurement constructors signal need for a context object

**Location**: e.g. `probeflow/measurements/features.py:98-117` (`feature_maxima_result`: 14 keyword args), `probeflow/measurements/image.py:73-87` (`step_height_from_rois`: 11 args), `probeflow/measurements/image.py:132-150` (`line_profile_measurement`: 15 args), `probeflow/measurements/image.py:240-254` (`line_periodicity_measurement`: 13 args).

**Problem**: Each measurement helper takes a long tail of "context" parameters: `measurement_id`, `source_label`, `source_path`, `channel`, `x_unit`, `y_unit`, `z_unit`/`height_unit`, `roi_id`, `roi_name`, `data_basis`, `notes`. Every call site must thread these through. The GUI mixin and dialog code (e.g. `probeflow/gui/dialogs/pair_correlation.py:29`) duplicates the assembly.

**Why it matters**: Adding a new context field (e.g. `selection_scope` was added to `feature_maxima_result` per Stage 2) requires touching every measurement helper. The call sites are already inconsistent: `feature_maxima_result` carries `selection_scope`, `roi_statistics` does not.

**Suggested fix**: Introduce `probeflow/measurements/context.py:MeasurementContext` dataclass (frozen) carrying `measurement_id`, `source_label`, `source_path`, `channel`, `roi_id`, `roi_name`, `data_basis`, `selection_scope`, `notes`. Each measurement helper takes `*, context: MeasurementContext, ...measurement-specific-args`. The GUI / CLI seam builds the context once (mirrors Stage 2's `roi_context` helper).
