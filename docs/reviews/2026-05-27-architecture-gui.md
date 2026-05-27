# ProbeFlow GUI Architecture & Maintainability Review
Date: 2026-05-27 (written 2026-05-28 retry, sequential mode)
Reviewer: GUI-architecture agent (deep review pass)

## Summary
- 23 findings: S0=0, S1=4, S2=17, S3=2
- Top 3 concerns:
  1. **Floating Feature Counting window duplicates ~150 LOC of orchestration from `ProbeFlowWindow` (finding 1)** — biggest maintainability hit on the GUI right now.
  2. **`_FeaturesWorker` shared `_FeaturesWorkerSignals` instance + no "running" button-disable** — last-result-wins race when a user re-clicks Run on a slow classify (finding 2).
  3. **`f6aac4a`-class RuntimeError still possible on the lattice-grid dock and on the ImageViewer modeless-dialog teardown path** (findings 3, 22).
- Overall posture: the GUI is structurally sound — Stage 2 extractions for `roi_context`, `tool_launch`, controllers, and feature/tv sidebar split are working. Backend boundary is clean: no `probeflow.processing`/`analysis`/`measurements`/`io`/`core`/`provenance` module imports from `probeflow.gui` (verified by grep), and the new dialogs (Threshold, FeatureLattice, PairCorrelation, STMBackground) correctly delegate scientific logic to backend modules. The pain is concentrated in three places: (a) the floating-window duplication described above; (b) inconsistent modeless-dialog lifecycle (some use `WA_DeleteOnClose=True`, some `False`, only two are explicitly closed from `closeEvent`); (c) the `image_viewer.py` mixin chain plus `_legacy.py` shim layer — these don't block anything but they're the path-of-most-effort for new contributors. None of the issues are user-data corrupting; all are addressable as bounded Stage-2-style extractions.

## Findings

### 1. [S1] FeatureCountingWindow duplicates ~150 LOC of orchestration logic from ProbeFlowWindow
**Location**: `probeflow/gui/features/window.py:147-258` and `probeflow/gui/app.py:1037-1142`
**Problem**: The floating Feature Counting window (`FeatureCountingWindow`) reimplements `_on_run`, `_on_finished`, `_on_export`, `_on_segment_for_classify`, `_on_mode_changed`, `_on_classify_params_changed`. The body is nearly identical to `ProbeFlowWindow._on_features_run`, `_on_features_finished`, `_on_features_export`, `_on_features_segment_for_classify`, `_on_features_mode_changed`, `_on_classify_params_changed` in `app.py`. The classification summary builder (≈25 LOC computing `_BIN=15` angle buckets, "other" handling, `parts` joining) is copy-pasted verbatim.
**Why it matters**: Two places to update for every new analysis mode, new export field, or new classify behaviour. Any drift produces subtle "tab vs window" disagreements that the user perceives as a bug. The recent Feature Counting button commit (`58636e7`) already inherited the divergent surface.
**Suggested fix** (Stage-2-style): extract a `FeatureCountingController` (plain `QObject`, no Qt widgets) that owns the panel + sidebar + thread-pool wiring and the `_on_run / _on_finished / _on_export` slots. Have both `ProbeFlowWindow` and `FeatureCountingWindow` instantiate one. Specifically: move the body of `app.py:1037-1219` into a method on the controller, parameterise on `panel` and `sidebar`, leave the `_status_bar` callback as an injected callable.
**Cross-refs**: backend report independent.

### 2. [S1] `_FeaturesWorker` uses `_signals` shared across multiple in-flight calls — last-arrives-wins races
**Location**: `probeflow/gui/features/window.py:59-60`, `probeflow/gui/app.py:240-249, 1083-1087`
**Problem**: `FeatureCountingWindow` constructs *one* `_FeaturesWorkerSignals()` and reuses it for every `_FeaturesWorker` it spawns. Same pattern in `ProbeFlowWindow._features_signals`. If the user clicks "Run" twice rapidly (or while a slow classify is still running), the second worker shares the same `finished` signal as the first. The `finished` slot dispatches on `mode`, so two concurrent runs of the same mode (e.g. quickly re-clicking Run on Classify after a parameter tweak) can deliver the *first* result *after* the second, silently corrupting which params produced which displayed result.
**Why it matters**: Subtle classify/template result mismatches with no error. Hard to reproduce in tests. Becomes more likely as Classify runs slow down with larger images.
**Suggested fix**: either disable the Run button while a worker is in flight (`self._sidebar.set_running(True)` — TV-sidebar already has this, Features does not), or make each `_FeaturesWorker` carry its own `_FeaturesWorkerSignals` instance the way `ConversionWorker`/`ThumbnailLoader` do (each instantiates its own signals object in `__init__`).
**Cross-refs**: numerical-stability report likely independent.

### 3. [S1] Sibling RuntimeError-on-close risk in lattice-grid dock and FFT-tools dock (same class as fixed `f6aac4a`)
**Location**: `probeflow/gui/viewer/image_viewer_tools_mixin.py:120-124` (lattice-grid dock visibilityChanged), `probeflow/gui/dialogs/fft_viewer.py:976-990` (already fixed)
**Problem**: The `f6aac4a` fix added a `try/except RuntimeError` around the `dlg.close()` callback in `FFTViewerDialog` because the lattice panel can be deleted before the parent FFT dialog fires `finished`. The same lifecycle pattern exists in `image_viewer_tools_mixin._on_open_lattice_grid` at line 124: `dock.visibilityChanged.connect(lambda v: _on_dock_closed() if not v else None)`. When the image viewer is closed (with `WA_DeleteOnClose,False` not set on its docks) Qt destroys the C++ object before this lambda's closure can fire — and `getattr(self, "_lattice_grid_dock", None) is dock` will compare a dead C++ object reference. Similarly the `dock.blockSignals(True)` / `dock.close()` block at lines 50-58 of the same file catches `RuntimeError` but `dock.blockSignals(False)` in the `finally` will *also* raise `RuntimeError` if the object is already gone — the inner try/except handles it, but the pattern shows the lifecycle reasoning is fragile.
**Why it matters**: Hidden crash class. The `f6aac4a` fix shipped exactly because this pattern bit users in production.
**Suggested fix**: use `QObject.destroyed` instead of `visibilityChanged` as the "dock gone" signal — destroyed fires *before* the C++ object dies and is the canonical lifecycle hook. Or: hold `self._lattice_grid_dock = dock` only via a `QPointer`, which auto-nulls on C++ delete and removes the `is dock` comparison footgun.
**Cross-refs**: none; pure GUI lifecycle.

### 4. [S2] `_legacy.py` is no longer legacy code — comment and module name are misleading
**Location**: `probeflow/gui/_legacy.py:1-16`, `probeflow/gui/main_window.py:1-6`
**Problem**: The docstring says "this is the working GUI implementation, not deprecated code" and explains the `_legacy` suffix is a refactor-in-progress marker. After commits up through `def9b71`, `_legacy.py` is now a 211-line re-export shim — `ProbeFlowWindow` and `main` live in `app.py`. `main_window.py` is a 6-line re-export from `_legacy`. The bulk has already moved. The misleading filename actively discourages new contributors from poking inside (because "legacy" reads "don't touch").
**Why it matters**: New code wires through opaque indirection (`probeflow.gui.main_window → _legacy → app`) for no benefit. The `_GuiCompatModule` `__setattr__` hook in `__init__.py:148-154` also pushes attribute writes into `_legacy` for back-compat — but if nothing else still imports from `_legacy` directly, this is invisible bookkeeping.
**Suggested fix**: rename `_legacy.py` → `compat.py` (or delete entirely after a deprecation cycle), inline the re-exports, and replace `main_window.py` with the trivial `from .app import ProbeFlowWindow` to remove the double hop. Keep the `_LEGACY_EXPORTS` table in `__init__.py` for one release if external tests use those names.
**Cross-refs**: architecture-backend report — neither block landing.

### 5. [S2] `ROIManagerDock` does not set its own `objectName`; relies on call-site
**Location**: `probeflow/gui/roi_manager_dock.py:38-52`, vs `probeflow/gui/dialogs/image_viewer.py:892`
**Problem**: `ROIManagerDock.__init__` never calls `setObjectName(…)`. The call site at `image_viewer.py:892` immediately patches the name in after construction (`self._roi_dock.setObjectName("imageViewerRoiManagerDock")`). The `f6aac4a` fix message ("missing dock objectName") shows this was previously an actual `QMainWindow.saveState/restoreState` warning. If any future caller forgets the setObjectName line, the same warning returns silently.
**Why it matters**: This is a class invariant ("ROI Manager docks must have a stable name for state save/restore") that the class itself does not enforce — easy to regress when a second caller appears.
**Suggested fix**: in `ROIManagerDock.__init__`, after `super().__init__("ROI Manager", parent)`, add `self.setObjectName("roiManagerDock")`. Call sites can override if they need a more specific id but at least the default is non-empty.
**Cross-refs**: none.

### 6. [S2] `ThresholdDialog` reads `HistogramPanel` private attrs to hide brightness/contrast/auto/reset
**Location**: `probeflow/gui/dialogs/threshold_dialog.py:167-170`
**Problem**: `self._hist._brightness_w.hide() / self._hist._contrast_w.hide() / self._hist._auto_btn.hide() / self._hist._reset_btn.hide()`. Four single-underscore private accesses into a sibling viewer module to remove UI affordances. If `HistogramPanel` is refactored — e.g. these get combined into a `_controls_widget` — the threshold dialog will silently render with the unwanted controls visible.
**Why it matters**: Hidden coupling that doesn't show up in imports/grep for public symbols. The exact private name is the contract.
**Suggested fix**: add a `HistogramPanel.set_threshold_mode(enabled: bool)` method that hides those four widgets internally and is the documented entry point. Or add an opt-in keyword to `HistogramPanel.__init__` like `controls=False`. Either way, move the hide-list inside the panel.
**Cross-refs**: none.

### 7. [S2] `ImageViewerDialog` is multiply-mixed-in across five mixin modules; method-resolution surface is hard to audit
**Location**: `probeflow/gui/dialogs/image_viewer.py:133-140`
**Problem**: `class ImageViewerDialog(ImageViewerRoiMixin, ImageViewerToolbarMixin, ImageViewerDisplayMixin, ImageViewerToolsMixin, ImageViewerProcessingExportMixin, QDialog)`. Five mixins, no protocols/typing for the assumed attributes. Each mixin reaches into `self._zoom_lbl`, `self._processing`, `self._display_arr`, `self._raw_arr`, `self._image_roi_set`, `self._status_lbl`, `self._scan_range_m` etc. with no formal interface. The MRO order matters (e.g. RoiMixin first), but nothing enforces it.
**Why it matters**: Adding a method to a mixin can shadow a method on another mixin or `QDialog` without warning. Tests are hard to write because instantiating any mixin requires the full QDialog + 30+ instance attrs. Recent file size: `image_viewer.py` 1883 LOC, `image_viewer_tools_mixin.py` 520 LOC, `..._processing_export_mixin.py` 519 LOC, `..._roi_mixin.py` 452 LOC.
**Suggested fix**: Stage 2 already extracted Controllers (`DisplayRangeController`, `DisplaySliderController`, etc.) and that pattern works. Continue: replace mixin methods that don't need `self` Qt-event state with controller objects that take the dialog as a dependency. Concretely, `_on_open_pair_correlation`, `_on_open_feature_lattice`, `_on_open_fft_viewer` could become free functions taking a small dataclass of context (the same shape that `tool_launch.py` already produces).
**Cross-refs**: architecture-backend report on tool_launch — this is the natural place to converge.

### 8. [S2] Image-viewer dialog is 1883 LOC; FFT-viewer dialog is 2044 LOC — both above monolith threshold
**Location**: `probeflow/gui/dialogs/image_viewer.py` (1883), `probeflow/gui/dialogs/fft_viewer.py` (2044), `probeflow/gui/app.py` (1509), `probeflow/gui/image_canvas.py` (1324), `probeflow/gui/spec_viewer/single.py` (1240), `probeflow/gui/features/__init__.py` (1239), `probeflow/gui/lattice_grid/real_space_panel.py` (1150), `probeflow/gui/viewer/image_measurements.py` (1029)
**Problem**: Eight files over 1000 LOC. The pain isn't the size itself — it's that `_build()` methods string together hundreds of widgets in linear order with no sub-grouping. `image_viewer.py:212-944` is one `_build()` method covering toolbar, channels, colormap, histogram, processing controls, ROI dock, measurement dock, navigation, and controllers — 730 LOC of setup.
**Why it matters**: Touching any widget in `_build()` carries fear of breaking siblings; PRs are hard to review; merge conflicts on this file are routine.
**Suggested fix** (Stage 2 style, not a rewrite): split `_build()` into `_build_toolbar()`, `_build_image_area()`, `_build_right_panel()`, `_build_docks()`, `_build_nav_row()`. Each becomes ≤200 LOC. The mixin pattern already does this for behaviour; do the same for layout.
**Cross-refs**: none.

### 9. [S2] `FeatureCountingWindow` ignores theme — passes empty dict `t = {}` to its panels
**Location**: `probeflow/gui/features/window.py:64-66`
**Problem**: `t: dict = {} # theme dict — window owns its own styling`. `FeaturesPanel(t)` and `FeaturesSidebar(t)` are constructed with no theme. Meanwhile every other panel inherits the current theme from `ProbeFlowWindow._dark`. Re-opening the FC window after a Light/Dark toggle yields the *first-launch* styling.
**Why it matters**: User-visible: floating FC window has different appearance from the main app after a theme switch. The comment "window owns its own styling" suggests this is intentional but in practice the app stylesheet is set on `QApplication`, so the floating window mostly follows; the gap is only the per-widget theme tweaks that the panels read from `t`.
**Suggested fix**: pass the parent's theme dict on construction (`FeatureCountingWindow(parent=self, theme=THEMES["dark" if self._dark else "light"])`) and reapply on `_apply_theme` by tracking the open FC window in `_open_viewers`-style list.
**Cross-refs**: none.

### 10. [S2] Per-scan saved processing keyed by absolute path string — invalidated on file move/rename, never on file mtime change
**Location**: `probeflow/gui/app.py:132-134, 982-988, 1340-1355`
**Problem**: `self._saved_processing: dict = {}` keys by `str(entry.path)`. When the same file changes on disk (re-acquired in ScanFlow, edited by a writer, etc.) the cached processing replays *on top of stale path*. Conversely if the user moves the file then re-opens, the cache key changes and saved processing is silently dropped.
**Why it matters**: With Survey/ScanFlow integration (`open_survey`) the same scan path can be re-saved as part of polishing; the cache may apply a now-irrelevant `align_rows` to fresh data and confuse the user (they think the file changed by itself).
**Suggested fix**: capture `(path, mtime_at_open)` as the key. On open, look up by path and invalidate if mtime differs. Or wire an explicit "forget saved processing for this entry" right-click action on the grid card.
**Cross-refs**: io report mentions reader invariants — neither blocks the other.

### 11. [S2] Tool launch is split between `viewer/tool_launch.py` (pair-correlation, feature-lattice, lattice-grid) and ad-hoc dialog imports for everything else
**Location**: `probeflow/gui/viewer/tool_launch.py` (extracted in Stage 2) vs `image_viewer_tools_mixin.py:140-150` (FeatureFinder), `:158-208` (ImageArithmetic), `:312-343` (ImageInfo), `:481-493` (FFTViewer), `:500-520` (PeriodicFilter)
**Problem**: Stage 2 extracted launch-context helpers for three tools (pair-correlation, feature-lattice, lattice-grid). The other five tools (FeatureFinder, ImageArithmetic, ImageInfo, FFTViewer, PeriodicFilter) still construct dialogs inline inside `image_viewer_tools_mixin`, each with their own one-off preview/apply/clear closures. The closures `_get_image / _preview_fft_correction / _clear_fft_correction_preview / _apply_fft_correction` at `image_viewer_tools_mixin.py:463-479` are duplicated almost line-for-line with `lattice_grid` handlers at the same file:81-103.
**Why it matters**: New tools that need preview/apply behavior copy the closure block again. Tool-specific knowledge bleeds back into the mixin.
**Suggested fix**: in `tool_launch.py`, add a small `ProcessingPreviewBridge` dataclass with `get_image / preview / clear_preview / apply_geometric_op / apply_arithmetic_op` callables, build it once on dialog init, hand the same object to FFT, lattice-grid, periodic-filter dialogs. Their dialog constructors already accept the same callable shape.
**Cross-refs**: architecture-backend extracted tool_launch — continue from there.

### 12. [S2] `image_viewer_tools_mixin._collect_point_sources_m` reaches into `_image_measurements.feature_points/feature_metadata` private attrs
**Location**: `probeflow/gui/viewer/image_viewer_tools_mixin.py:352-378`
**Problem**: `getattr(ff_dlg, "result", None)` and `getattr(measure_ctrl, "feature_points", [])` and `getattr(measure_ctrl, "feature_metadata", {})` — three back-channel reads, two through `getattr` defaults to mask the case where the controller didn't initialise the attr yet. Same pattern at three call sites (`_collect_point_sources_m`, `_collect_point_sources_px`, `_collect_point_source_metadata`).
**Why it matters**: When `ImageMeasurementController` is refactored or one of these is renamed, the GUI silently behaves as if no point sources exist (because the getattr default is the empty list/dict). No traceback, no warning — just "Pair Correlation says: no sources available" mysteriously.
**Suggested fix**: add `ImageMeasurementController.point_source_records(px_x, px_y)` that returns the records the same way `collect_point_source_records` builds them. Move the `feature_finder_result` read into the same controller method. The mixin then calls one method instead of poking three attrs.
**Cross-refs**: architecture-backend.

### 13. [S2] `app.py._open_viewers` Python-keepalive list never garbage-collects spec viewers
**Location**: `probeflow/gui/app.py:127-130, 1334-1339`
**Problem**: `self._open_viewers.append(dlg)` and `_on_closed` removes via `self._open_viewers.remove(d)`. The remove call is wrapped in try/except ValueError. The `_on_closed` lambda is attached to `dlg.finished` — and `finished` fires for `accept()`, `reject()`, and Qt-driven close, so this should work. However spec viewers go through the same `_open_viewers.append(dlg)` (line 1316 + 1334) but `_on_closed` only checks `if not spec` — meaning spec viewers stay attached to `_on_closed` (the `finished.connect`) but if they're closed with the window-close X they will still trigger `_on_closed`, which only removes from the list. Reading carefully this works, but the parallel for-spec path is implicit and hard to verify. The thread safety issue is: `finished` fires from the Qt event loop; if a spec viewer is held by `_open_viewers` and the closeEvent path also triggers `closeEvent` on the *parent* (browser closing while a viewer is open), the order isn't deterministic.
**Why it matters**: Borderline S2 — under normal use everything cleans up, but the pattern is one mistake away from a leak (e.g., someone adds an early-return in `_on_closed` for an exception, the list grows unbounded).
**Suggested fix**: switch to `WeakSet` keyed on the dialog, or set `dlg.setAttribute(Qt.WA_DeleteOnClose, True)` for non-modal viewers and observe `destroyed` to clean the list.
**Cross-refs**: none.

### 14. [S2] `_set_thumbnail_colormap / _set_thumbnail_channel / _set_thumbnail_align` use the combo box's `currentText` as proxy state — duplicated logic with 3 near-identical methods
**Location**: `probeflow/gui/app.py:572-591`
**Problem**: Three methods each do `if hasattr(self, "_browse_tools") and self._browse_tools.<cb>.currentText() != value: setCurrentText(value) else: <call handler>(value); self._sync_menu_actions()`. The pattern is identical except for the combo name and handler name. View state is read out of the combo widget rather than from an explicit model.
**Why it matters**: Every new thumbnail setting (e.g., a future "thumbnail aspect-ratio" menu) needs another 6-line clone. Bugs around "menu/combo out of sync" — exactly the lattice_grid commit class.
**Suggested fix**: extract `_apply_thumbnail_setting(combo_attr, value, handler_attr, mode_name)` taking the four names; the three menu-handler methods become one-liners.
**Cross-refs**: none.

### 15. [S2] No central keyboard-shortcut registry — shortcuts defined inline at point of menu construction
**Location**: `probeflow/gui/app.py:343-466` (~13 inline `setShortcut(QKeySequence(…))` calls), and viewer `command_finder.py`+`shortcuts.py` for the viewer-command set only
**Problem**: Main window menu shortcuts (`Ctrl+1` browse, `Ctrl+2` convert, `Ctrl+3` features, `Ctrl+4` tv, `Ctrl+5` dev, `Ctrl+6` definitions, `Ctrl+Shift+T` theme, `Ctrl+Shift+S` survey, `Ctrl+Shift+F` floating FC window, `Ctrl+O` open, `Ctrl+Q` quit) are scattered inline. The viewer has a proper registry (`viewer/shortcuts.py`) but it doesn't cover the main window. No way to detect a shortcut clash without manually scanning. The FC window adds `Ctrl+Shift+F` — there's no check that this doesn't clash with an OS-level "Find" shortcut on macOS.
**Why it matters**: As Tools menu grows (already 6 items + Survey + FC window), shortcut collisions become hard to spot. The viewer pattern (registered commands) is already the right answer — the main window should reuse it.
**Suggested fix**: extend `viewer/shortcuts.py` to a top-level `gui/shortcuts.py` that registers main-window commands the same way, then dump-on-startup any conflicts to stderr.
**Cross-refs**: none.

### 16. [S2] `closeEvent` writes config keys ad-hoc; no centralized config-write surface
**Location**: `probeflow/gui/app.py:1475-1493`
**Problem**: `closeEvent` does a literal `cfg.update({…})` with 11 keys from 6 widget attributes. Each new persisted preference adds a line here. Meanwhile `desktop_layout.py` has its own `_save_desktop_layout_into(cfg)` and config keys are loaded in `__init__`. There's no single "what does ProbeFlow persist?" surface.
**Why it matters**: Easy to introduce a key in `__init__` that loads from cfg but forget to add the matching `closeEvent` write — silent setting-doesn't-stick bug. Already happened twice in git history per commit messages (per-scan processing memory `61495b7` and floating FC window — neither persists state across launches).
**Suggested fix**: add a `Preferences` dataclass with `from_cfg(dict) -> Preferences` and `to_cfg(self, dict) -> None`. `__init__` reads via `from_cfg`, `closeEvent` writes via `to_cfg`. Adding a new preference touches exactly the dataclass plus the consuming widget.
**Cross-refs**: none.

### 17. [S2] Survey import is inline inside `_build_ui` — not optional in any meaningful sense
**Location**: `probeflow/gui/app.py:194-195`
**Problem**: `from probeflow.gui.survey import SurveyPanel` is inside `_build_ui()` (delayed import). But `SurveyPanel` is *always* constructed and added to the content stack at idx 5, regardless of whether ScanFlow is installed. The docstring at `__init__.py:31` of the gui-survey module probably warns about optional dependencies — let me check.
**Why it matters**: If `probeflow/gui/survey/__init__.py` imports ScanFlow eagerly, the GUI fails to launch when ScanFlow isn't installed — even though Survey is described as an optional integration.
**Suggested fix**: wrap the import in `try/except ImportError`, hide the Survey tab if absent, disable the `Ctrl+Shift+S` shortcut. Or push the construction to the first switch to Survey mode (lazy panel construction) — same pattern as `_fc_window = None` until first open.
**Cross-refs**: io report may have noted optional-dep handling for ScanFlow.

### 18. [S2] `desktop_layout.py` save/restore expects splitter counts to match exactly — silently fails otherwise
**Location**: `probeflow/gui/app.py:518-527`
**Problem**: `if main_sizes and len(main_sizes) == self._splitter.count(): self._splitter.setSizes(...)` else falls back to default. If a future refactor adds or removes a splitter pane (e.g., merging two sidebars), users with old saved layouts silently get the default. There's no migration path.
**Why it matters**: Users who customize splitter sizes lose them silently after an upgrade.
**Suggested fix**: version the layout dict (`"version": 2`). On mismatch, attempt a documented best-effort migration or print a one-line status-bar notice "Window layout reset because pane structure changed."
**Cross-refs**: none.

### 19. [S3] `from PySide6.QtWidgets import …` walls in `_legacy.py` are unused after extraction
**Location**: `probeflow/gui/_legacy.py:41-48`
**Problem**: `QAbstractItemView, QButtonGroup, QCheckBox, …, QTableWidget, QTableWidgetItem, QHeaderView, QVBoxLayout, QWidget` are imported but `_legacy.py` no longer contains any code that uses them — it just re-exports from submodules.
**Why it matters**: Dead imports; minor lint pain.
**Suggested fix**: trim to only the names re-exported.
**Cross-refs**: none.

### 20. [S3] `ImageViewerDialog._open_viewers`-equivalent list uses `_open_viewers` and `_feature_finder_dlg` attr held alongside `_open_viewers` parent list — naming is inconsistent
**Location**: `probeflow/gui/app.py:130 (_open_viewers)`, `probeflow/gui/viewer/image_viewer_tools_mixin.py:149 (_feature_finder_dlg)`, `probeflow/gui/dialogs/image_viewer.py` various controller attrs
**Problem**: Some dialogs are kept-alive by storing on `self._feature_finder_dlg`, others by being added to `self._open_viewers`. Some panels are referenced via `self._roi_dock` etc. No documented convention.
**Why it matters**: Style smell only — affects on-ramp speed for new contributors.
**Suggested fix**: document the convention in `gui/__init__.py` docstring: "non-modal dialogs go through a single `_open_children` list on the parent; ROI/measurement docks are stored as `_<name>_dock`."
**Cross-refs**: none.

### 21. [S2] No tests cover `FeatureCountingWindow` lifecycle or the `_features_signals` shared-signal pattern
**Location**: `tests/test_gui_*.py` (none reference `FeatureCountingWindow`)
**Problem**: The floating window has zero test coverage despite being new code (`82cd4d8`, `58636e7`). The race condition described in finding 2 is not testable without a test, and the shared-signals pattern is repeated in two places.
**Why it matters**: Structurally hard-to-test code accumulates bugs that only surface in user reports. The Qt preflight skip in stage-3 already prevents headless local runs, so even adding a test won't run in this environment — but it would run in CI.
**Suggested fix**: add a `tests/test_features_window.py` that constructs a `FeatureCountingWindow` against a fake panel/sidebar and verifies (a) `load_entry` calls through, (b) running while a worker is in-flight is blocked, (c) close + reopen does not leak signal connections.
**Cross-refs**: stage-3 GUI test caveats already noted.

### 22. [S1] Inconsistent modeless-dialog lifecycle: only Threshold and STMBackground get explicitly closed in `ImageViewerDialog.closeEvent`
**Location**: `probeflow/gui/viewer/image_viewer_processing_export_mixin.py:501-519` (close handler), plus the seven other modeless dialogs created via `dlg.show()` in `image_viewer_tools_mixin.py:149,170,343,414,456,493,510` and `image_viewer_processing_export_mixin.py:376,394,419`
**Problem**: `closeEvent` only closes `_stm_background_dialog` and `_threshold_dialog`. The other modeless dialogs created by the viewer (FeatureFinderDialog, ImageArithmeticDialog (modal-exec, OK), ImageInfoDialog, PairCorrelationDialog, FeatureLatticeDialog, FFTViewerDialog, ScaleDialog, ShearDialog, PeriodicFilterDialog (modal-exec), and the lattice grid panel via dock) all use `WA_DeleteOnClose=False` (PairCorrelation, FeatureLattice, FFTViewer, FeatureFinder, PointFFT, LinePeriodicityPlot) or `WA_DeleteOnClose=True` (Threshold, ImageInfo, Scale, Shear) inconsistently. With `WA_DeleteOnClose=False`, closing the viewer first leaves these C++ objects as orphan children of the viewer until Qt garbage-collects the parent; their signal handlers may still fire on partially-torn-down state. Several of these dialogs hold references to `self._raw_arr` / `self._scan_range_m` / preview callbacks that close over the viewer.
**Why it matters**: Same class of bug as `f6aac4a`. A user who has a Pair Correlation or Feature Lattice panel open and closes the parent image viewer can land in code paths that trigger the C++-object-already-deleted RuntimeError. The hand-rolled `try/except RuntimeError` patches won't scale across nine dialog types.
**Suggested fix**: collect modeless dialog refs into a `self._modeless_children: list[QDialog]` (the same way `app._open_viewers` does), append on creation, and iterate-close in `closeEvent`. Plus standardise on `WA_DeleteOnClose=True` for non-result dialogs (Threshold/Scale/Shear/Info already do this; Pair Correlation, Feature Lattice, Feature Finder, Point FFT, Line Periodicity Plot still set `False`).
**Cross-refs**: directly extends `f6aac4a`.

### 23. [S2] `probeflow.gui.processing` uses `from probeflow.processing.gui_adapter import *` wildcard import
**Location**: `probeflow/gui/processing.py:12`
**Problem**: `from probeflow.processing.gui_adapter import *` pulls in everything `gui_adapter` exports without an `__all__`-controlled surface. Anything the adapter happens to import becomes a name on `probeflow.gui.processing` — including private helpers.
**Why it matters**: Two indirect harms — (a) name collisions if `gui_adapter` ever re-exports a name that shadows a local one in `processing.py`; (b) the GUI's public surface becomes coupled to whatever the backend adapter chooses to export, making the boundary fuzzier than the rest of the GUI/backend split (which is otherwise clean per the directory grep). Adds noise to IDE auto-imports and linter passes.
**Suggested fix**: replace the wildcard with explicit names — `from probeflow.processing.gui_adapter import processing_state_from_gui, …` (whatever `processing.py` actually consumes). If nothing is used directly, drop the import.
**Cross-refs**: architecture-backend report — neither blocks.

## Appendix: not flagged but worth tracking

- `tool_manager.py` is just a constants module (8 lines + dict) — the name "manager" overstates it but it's not worth renaming.
- `gui/dialogs/__init__.py` omits `ThresholdDialog` from its re-exports, but `image_viewer_processing_export_mixin.py` imports directly from `probeflow.gui.dialogs.threshold_dialog`, so nothing actually depends on the export — cosmetic only.
- The viewer-specific `shortcuts.py` registry and `command_finder.py` are a good pattern; extending it to the main window (finding 15) would close a real gap.
- `gui/_legacy.py` still does `import copy, import json, import sys` at module level — they're no longer used in this file (everything moved to `app.py`). Same dead-import situation as finding 19.
- `survey/__init__.py` imports `SurveyPanel` eagerly, which itself does lazy `from scanflow.automation import …` inside methods — so the optional-dep boundary is fine in practice. The finding 17 risk is around hard-coding the Survey tab in `_build_ui` rather than around the import itself.
- Backend boundary verified clean: `grep -rn "from probeflow.gui" probeflow/{processing,analysis,measurements,io,core,provenance}/` returned no results.
