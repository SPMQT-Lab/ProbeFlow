# Dead-Code Audit — ProbeFlow

Run date: 2026-05-11  
Tools: custom orphan script, vulture 2.x, ruff F401, AST grep  
Command:
```
vulture probeflow/ tests/ whitelist.py --min-confidence 80
ruff check probeflow/ --select F401 --no-cache
python scripts/find_orphan_modules.py
```

---

## Layer 1 — Orphan modules (no inbound import)

These files exist but are never imported by any other `.py` file.

| File | Lines | Verdict | Notes |
|---|---|---|---|
| `probeflow/cli/commands/analysis.py` | 25 | **defer** | Stub module for future CLI decomposition; referenced by intent in `cli/parser.py` but not yet wired |
| `probeflow/cli/commands/conversion.py` | 5 | **defer** | Same |
| `probeflow/cli/commands/gui.py` | 5 | **defer** | Same |
| `probeflow/cli/commands/processing.py` | 5 | **defer** | Same |
| `probeflow/cli/commands/scan.py` | 5 | **defer** | Same |
| `probeflow/cli/commands/spectroscopy.py` | 15 | **defer** | Same |
| `probeflow/plugins/manifest.py` | 20 | **defer** | Empty plugin-manifest placeholder; keep until plugin wiring is implemented |

All 7 are intentional scaffolding. No removal recommended yet.  
Add `probeflow/cli/__main__.py` to `ENTRY_POINTS` in the script if it ever becomes a real entry point.

---

## Layer 2 — Vulture dead code (fixed in this audit)

Items vulture flagged at ≥80% confidence. All resolved during the audit run.

| Symbol | File | Fix applied |
|---|---|---|
| `QEvent` import | `gui/_legacy.py:36` | Removed — was never used after Qt5→6 migration |
| `QWheelEvent` import | `gui/_legacy.py:40` | Removed — was never used |
| `option` parameter in `paint()` | `gui/image_canvas.py:83,103` | Renamed `_option` — required Qt signature, not accessed |
| `option` parameter in `paint()` | `gui/roi_items.py:97` | Renamed `_option` — same reason |
| `exit_status` parameter | `gui/terminal/__init__.py:353` | Renamed `_exit_status` — required by `QProcess.finished` signal |
| `image_px` parameter | `processing/image.py:1713` | Removed — accepted but never used in `_pick_scalebar_length`; call site updated |
| `kw` in test helpers | `tests/test_graph.py:180,200` | Renamed `**_kw` — variadic kwarg accepted but not accessed |

Vulture now exits cleanly (exit 0) at `--min-confidence 80`.

---

## Layer 3 — Unused imports (ruff F401)

61 hits across 14 files. Suppressed globally in pyproject.toml today; this is a one-off audit view.

### `gui/_legacy.py` (~38 hits) — **defer**

Large legacy file still being incrementally extracted. Most unused imports are refactored-out code paths where the import was left behind. The right fix is continued extraction, not bulk deletion, to avoid breaking the large surface this file exposes to `gui/dialogs/`, `gui/viewer/`, etc.

Specific clusters:
- `probeflow.gui.browse.*` (ScanCard, SpecCard, _BrowseCard) — moved to `gui/browse/`; legacy file no longer uses them but still imports for backward compat
- `probeflow.gui.workers.*` (5 classes) — moved to `gui/workers/`; same pattern
- `probeflow.gui.models.*` (4 helpers) — moved to `gui/models/`; same pattern
- `probeflow.gui.terminal.*` — moved to `gui/terminal/`; same
- `PySide6.QtCore.*` (QObject, QRect, QRunnable, QSize, QTimer) — Qt widgets whose feature was extracted
- `PySide6.QtGui.*` (QBrush, QColor, QImage, QMovie, QPainter, QPen) — same
- `PySide6.QtWidgets.*` (QGraphicsPixmapItem, QGroupBox, QLineEdit, QTabWidget, QToolTip) — same

**Action:** Fix opportunistically as each widget class is extracted from `_legacy.py`.

### `cli/_legacy.py` (7 hits) — **defer**

| Import | Note |
|---|---|
| `typing.Tuple`, `typing.List` | Legacy 3.8-era style; replace with `tuple`, `list` builtins when touching the file |
| `io` (stdlib) | Left over from a removed `io.StringIO` usage |
| `probeflow.io.sxm_io.{parse_sxm_header, read_all_sxm_planes, read_sxm_plane, sxm_dims, sxm_scan_range}` | Imported for CLI commands that were re-pointed to higher-level `load_scan` API; safe to remove |
| `probeflow.io.sxm_io.write_sxm_with_planes` | Same |

**Action:** Remove the 5 sxm_io and the `io` import when next touching `cli/_legacy.py`. The `typing` aliases are style — defer.

### `analysis/xmgrace_export.py` (2 hits) — **fix soon**

`Iterable` and `Optional` from `typing` — both superseded by builtin equivalents in Python 3.11. Simple cleanup.

### Other files (1 hit each) — **fix soon**

| File | Import | Note |
|---|---|---|
| `processing/gui_adapter.py` | `dataclasses.field` | Unused after a recent refactor |
| `processing/gui_adapter.py` | `dataclasses.dataclass` | Same |
| `processing/display_state.py` | `typing.Any` | Unused type alias |
| `processing/image.py` | `typing.Any` | Left over |
| `io/readers/createc_vert.py` | `pathlib.Path` | Unused |
| `io/converters/createc_dat_to_sxm.py` | `dataclasses.field` | Unused |
| `io/common.py` | `probeflow.core.roi.ROI` + `find_hdr` | Imported for a function that was removed |
| `io/writers/png.py` | `PIL.Image` | Unused after refactor |
| `processing/history.py` (via gui_adapter) | `processing_history_entries_from_state` | Function re-routed |
| `gui/roi_items.py` | `probeflow.core.scan_loader.SUPPORTED_SUFFIXES` | Copy-paste remnant |
| `gui/viewer/deferred_action.py` | `PIL.Image` | Unused |
| `gui/viewer/spec_overlay.py` | `typing.Union` | Old-style annotation |

**Action:** Clean these up in the next general pass — each is a one-liner removal.

---

## Layer 4 — Intra-file private helpers (processing/image.py)

AST grep found two public-looking names only ever called within `image.py` itself:

| Function | Line | Called by | Verdict |
|---|---|---|---|
| `fit_scanline_background` | 614 | `preview_stm_background` (same file) | **rename** — prefix `_` |
| `subtract_scanline_background` | 724 | `preview_stm_background` (same file) | **rename** — prefix `_` |

Neither is exported via `processing/__init__.py` or referenced in tests. Both are private implementation details of the STM line-background preview workflow. Renaming to `_fit_scanline_background` / `_subtract_scanline_background` makes their scope explicit without removing any functionality.

---

## Summary by priority

| Priority | Action | Count |
|---|---|---|
| **Done** | Vulture false positives suppressed + real issues fixed | 7 |
| **Fix soon** | One-liner unused imports in non-legacy files | ~12 |
| **Next refactor pass** | `cli/_legacy.py` sxm_io import cleanup | 6 |
| **Ongoing** | `gui/_legacy.py` imports cleared as classes are extracted | ~38 |
| **Rename** | `fit_scanline_background`, `subtract_scanline_background` → `_` prefix | 2 |
| **Defer** | `cli/commands/*.py` stubs, `plugins/manifest.py` orphans | 7 |

---

## Tooling added

| File | Purpose |
|---|---|
| `scripts/find_orphan_modules.py` | Layer 1: reports modules with zero inbound imports |
| `whitelist.py` | Vulture false-positive suppression (Qt overrides, plugin API, etc.) |
| `pyproject.toml` | Added `vulture` to `[project.optional-dependencies] dev` |

Run the full check with:
```bash
vulture probeflow/ tests/ whitelist.py --min-confidence 80
python scripts/find_orphan_modules.py
ruff check probeflow/ --select F401 --no-cache
```
