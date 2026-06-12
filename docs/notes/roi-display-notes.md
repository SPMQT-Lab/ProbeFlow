# ROI display & interaction — implementation notes

> Point-in-time implementation notes (2026-05); cited line numbers drift.
> The described architecture (hover/selection flow, `_ROI_HINTS`,
> `DisplayRangeController`, per-region composite) remains current.

Working notes for the per-region brightness/contrast + line-ROI click-fix work.
Captures how the touched subsystems actually function so the in-app help/tooltip
text (Part C) can be corrected against reality rather than assumption.

## ROI hover / click / handle flow (`probeflow/gui/image_canvas.py`)

- **Hover state**: `_hover_roi_id` tracks the ROI currently under the cursor.
  Updated in `_update_hover(view_pos)` (~802), called from `mouseMoveEvent` only
  in pan mode without a drag (~1107). `_roi_at_pos(view_pos)` resolves which ROI
  is under the cursor. `_hover_roi_id` is set even when it equals the active ROI,
  but the yellow hover *style* is only applied when `roi_id != active_id`
  (active ROI keeps its cyan dashed style).
- **Pens** (`roi_items.py` 26-36): inactive = blue solid `#89b4fa` 1.0pt;
  active = cyan dashed `#22D3EE` 1.5pt (handles visible); hover = yellow
  `#f9e2af` 1.5pt. Handles only visible on the active ROI.
- **mousePressEvent pan branch** (~928): resolves `roi_id` under cursor and
  `active_id`. Resize-handle hit test uses `resize_handles(cand_roi)` with a 12px
  (Chebyshev) view-space box, nearest handle wins.
  - **Part A change**: selection target now derives from `_hover_roi_id` first
    (falling back to `_roi_at_pos`) so the visible highlight and the click target
    always agree; the active-ROI handle fallback only fires when the cursor is not
    over a *different* (non-active) ROI (`roi_id is None or roi_id == active_id`).
    Net behaviour: first click selects the highlighted ROI (handles then appear);
    a second click on the now-active ROI drags it / its endpoints.
- **Activation signal**: `roi_activate_requested.emit(roi_id)` →
  `_on_canvas_roi_activate` (`image_viewer_roi_mixin.py` ~279) → `set_active` +
  `_on_image_roi_set_changed` (rebuilds canvas, refreshes line profile via
  `_sync_line_profile_visibility`).

## Hint / tooltip text sources (for Part C)

- **Per-kind hover hints**: `_hover_message_at` (`image_canvas.py` 824-839) emits
  `object_hovered(kind, message)` → `_on_canvas_object_hovered`
  (`image_viewer_roi_mixin.py` 112) → `self._status_lbl.setText(message)` (bottom
  status label). Long single-line strings.
- **Item tooltips**: duplicate per-kind strings in `roi_items.py` 143-148, set via
  `item.setToolTip(...)`. These float near the cursor and currently render as one
  wide row (Qt only word-wraps tooltips for rich text `<qt>…</qt>`).
- **Empty-state help**: `image_viewer.py` ~809 "ROI tools live in the ROI Manager
  dock…" (`roi_empty_lbl`, already `setWordWrap(True)`).

### Part C — implemented

- Single source of truth: `_ROI_HINTS` in `roi_items.py` with `roi_hint_text(kind)`
  (concise one-line status hint) and `roi_tooltip_html(kind)` (`<qt>…<br>…</qt>`
  rich text → wraps onto several short rows near the cursor instead of one wide
  line). `_tooltip_for_roi` now delegates to `roi_tooltip_html`.
- `image_canvas._hover_message_at` uses `roi_hint_text` (no more duplicated
  per-kind strings). Status label already word-wraps.
- Wording updated to select-then-edit ("Click to select … drag the active …").
  `definitions.py` "Line ROI actions" gains "click a line = make it the active
  line". `roi_empty_lbl` now points users at the View-tab Contrast scope +
  Hide ROI overlays controls.

## Display range / render pipeline (Part B — implemented)

- Global display levels: `DisplayRangeState` (`processing/display_state.py`),
  wrapped by `DisplayRangeController` (`gui/viewer/display_range.py`), driven by
  four sliders (`gui/viewer/display_sliders.py`). `resolve(arr)` returns
  `(vmin, vmax)` (manual or percentile).
- Render: `render_scan_image` (`gui/rendering.py`) →
  `colored = lut[_array_to_uint8(arr, vmin, vmax)]`. Shared by browse thumbnails,
  channel previews, and the viewer (via `ViewerLoader` in `gui/workers.py`).
- ROI mask: `ROI.to_mask(shape)` (`core/roi.py` ~155) → bool (Ny, Nx);
  `area_roi_mask` helper in `gui/roi_context.py` wraps it for area ROIs.

### Per-region composite (how it works now)

- **Compositing**: `render_scan_image(..., region_levels=[(mask, vmin, vmax), …])`.
  After the global `colored`, each region re-maps its masked pixels:
  `colored[mask] = lut[_array_to_uint8(arr, rvmin, rvmax)[mask]]`. `region_levels`
  defaults to `None` so the shared thumbnail/preview path is unchanged. Threaded
  through `ViewerLoader(region_levels=…)`.
- **Per-ROI ranges**: `ImageViewerDialog._region_drs: dict[roi_id, DisplayRangeController]`
  plus `_display_scope` ("global"|"roi"). `_target_drs()` returns the controller
  the contrast sliders edit (active area ROI's own when scope=="roi"). The four
  slider handlers + auto/reset clip all route through `_target_drs()`. The
  `DisplaySliderController` now resolves its target lazily (its `drs` arg may be a
  callable) — passed `self._target_drs`.
- **What renders**: `_region_levels_for_render()` includes a region only when its
  controller is in **manual** mode (i.e. the user actually tuned it), so untouched
  regions render identically to the global mapping. Resolves levels against the
  whole `_display_arr` to stay consistent with the slider readout.
- **Active-ROI follow**: `_on_image_roi_set_changed` refreshes sliders + re-renders
  when scope=="roi". Channel change clears `_region_drs` (units differ).
- **Hide overlay**: `ImageCanvas.set_rois_visible(bool)` toggles `setVisible` on all
  `_roi_items`/`_point_items`, persists across `_rebuild_roi_items` via
  `_add_roi_item_internal`, and short-circuits `_update_hover`. UI: View-tab
  "Contrast scope" combo + "Hide ROI overlays" checkbox.
- **Per-region background**: not part of this change — use the existing ROI-scoped
  processing pipeline (apply STM BG within the active ROI) as before.

### Deferred — B5 persistence

Per-region `_region_drs` levels are **session-only** (not yet serialized). They
reset on reload / channel change. Persisting them (keyed by ROI id, dropping
stale ids on load) is a follow-up; it does not affect B1–B4 behaviour.
