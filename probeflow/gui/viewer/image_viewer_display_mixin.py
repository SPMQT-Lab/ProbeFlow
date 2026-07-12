"""Display, histogram, and processing-control handlers for ImageViewerDialog."""

from __future__ import annotations

import copy

import numpy as np
from PySide6.QtCore import Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QMenu

from probeflow.gui.viewer import export_histogram


class ImageViewerDisplayMixin:
    def _on_set_zero_plane_mode_toggled(self, checked: bool):
        if checked:
            # The three clicks are one logical processing action. Capture the
            # state before the first point, then push it only when point 3
            # commits the plane.
            self._set_zero_undo_snapshot = self._capture_proc_undo_snapshot()
            self._set_zero_undo_snapshot["zero_mode"] = False
        msg = self._zero_ctrl.toggle(checked, self._set_selection_tool)
        self._zoom_lbl.set_set_zero_mode(checked)
        if msg:
            self._status_lbl.setText(msg)
        if not checked:
            self._set_zero_undo_snapshot = None
            self._refresh_zero_markers()

    def _on_set_zero_pick(self, frac_x: float, frac_y: float):
        """Handle image clicks while manual zero-plane mode is active."""
        # The user is clicking on the *displayed* array; the controller maps
        # the click fraction into that frame and stamps the geometric-op
        # count so replay anchors the plane on the clicked features.
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        undo_snapshot = getattr(self, "_set_zero_undo_snapshot", None)
        if undo_snapshot is None:
            undo_snapshot = self._capture_proc_undo_snapshot()
        rerender, msg = self._zero_ctrl.on_canvas_pick(
            frac_x, frac_y,
            arr,
            self._processing,
            self._set_zero_plane_btn.isChecked(),
        )
        if msg:
            self._status_lbl.setText(msg)
        if rerender:
            self._push_proc_undo_snapshot(undo_snapshot)
            self._set_zero_undo_snapshot = None
            if self._set_zero_plane_btn.isChecked():
                self._set_zero_plane_btn.setChecked(False)
            # Re-levelling moves the data baseline; a manual contrast window
            # tuned to the old baseline can leave the whole image clamped to
            # one end of the colour scale (current channels especially shift
            # far enough to render solid black). Reset to auto so the
            # re-levelled result is visible. reset_silent avoids a double
            # refresh — _refresh_processing_display re-renders below.
            self._drs.reset_silent(self._clip_low, self._clip_high)
            self._refresh_processing_display()

    def _refresh_zero_markers(self):
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        self._zero_ctrl.refresh_markers(arr, self._processing)

    def _on_clear_set_zero(self):
        if self._set_zero_plane_btn.isChecked():
            self._set_zero_plane_btn.setChecked(False)
        msg = self._zero_ctrl.clear()
        self._status_lbl.setText(msg)

    # ── Per-region display range (composite brightness/contrast) ──────────────
    def _target_drs(self):
        """Return the DisplayRangeController the contrast sliders should edit.

        In ``roi`` scope, edits go to the active area ROI's own range so each
        region can be scaled independently; otherwise the global range.
        """
        if getattr(self, "_display_scope", "global") == "roi":
            from probeflow.gui.roi_context import active_area_roi_context
            ctx = active_area_roi_context(getattr(self, "_image_roi_set", None))
            if ctx.roi_id is not None:
                return self._region_drs_for(ctx.roi_id)
        return self._drs

    def _region_drs_for(self, roi_id: str):
        """Get (or lazily create) the per-ROI DisplayRangeController."""
        from probeflow.gui.viewer.display_range import DisplayRangeController
        drs = self._region_drs.get(roi_id)
        if drs is None:
            drs = DisplayRangeController(
                clip_low=self._clip_low, clip_high=self._clip_high, parent=self
            )
            drs.rangeChanged.connect(self._refresh_display_range)
            self._region_drs[roi_id] = drs
        return drs

    def _region_levels_for_render(self):
        """Build the ``region_levels`` list for render_scan_image.

        Only area ROIs whose range has been *manually* tuned contribute, so an
        untouched region renders identically to the global mapping.
        """
        if (
            self._display_arr is None
            or not getattr(self, "_region_drs", None)
            or getattr(self, "_image_roi_set", None) is None
        ):
            return None
        from probeflow.gui.roi_context import area_roi_mask
        shape = self._display_arr.shape[:2]
        out = []
        for roi_id, drs in self._region_drs.items():
            if drs.mode != "manual":
                continue
            roi = self._image_roi_set.get(roi_id)
            if roi is None:
                continue
            mask = area_roi_mask(roi, shape)
            if mask is None:
                continue
            vmin, vmax = drs.resolve(self._display_arr)
            if vmin is None:
                continue
            out.append((mask, vmin, vmax))
        return out or None

    def _on_display_scope_changed(self, index: int) -> None:
        """Switch the contrast sliders between global and active-ROI scope."""
        self._display_scope = "roi" if index == 1 else "global"
        self._update_display_sliders()
        if self._display_scope == "roi":
            from probeflow.gui.roi_context import active_area_roi_context
            ctx = active_area_roi_context(getattr(self, "_image_roi_set", None))
            if ctx.roi_id is None:
                self._status_lbl.setText(
                    "Per-ROI contrast: select an area ROI to tune its own "
                    "brightness/contrast."
                )
            else:
                self._status_lbl.setText(
                    f"Per-ROI contrast: editing '{ctx.roi.name}'."
                )
        else:
            self._status_lbl.setText("Contrast: editing whole image.")

    def _on_toggle_rois_hidden(self, checked: bool) -> None:
        """Hide/show all ROI overlays so the composited image can be inspected."""
        self._rois_hidden = bool(checked)
        self._zoom_lbl.set_rois_visible(not self._rois_hidden)
        self._status_lbl.setText(
            "ROI overlays hidden." if self._rois_hidden else "ROI overlays shown."
        )

    # ── Histogram range and clip handlers ─────────────────────────────────────
    def _on_hist_range_released(self, lo_phys: float, hi_phys: float) -> None:
        """Receive drag-release from HistogramPanel and update display range."""
        scale, _, _ = self._channel_unit()
        if not scale:
            return
        self._target_drs().set_manual(lo_phys / scale, hi_phys / scale)

    # Auto-contrast cycles through progressively tighter central windows of the
    # data range (low %, high %, label). Clicking Auto steps to the next one.
    _AUTO_CLIP_PRESETS = (
        (1.0, 99.0, "99%"),
        (16.0, 84.0, "68%"),
        (33.5, 66.5, "33%"),
    )

    def _on_auto_clip(self):
        """Cycle the autoscale window: 99% → 68% → 33% → 99% of the data range."""
        presets = self._AUTO_CLIP_PRESETS
        idx = (getattr(self, "_auto_clip_idx", -1) + 1) % len(presets)
        self._auto_clip_idx = idx
        self._clip_low, self._clip_high, label = presets[idx]
        self._target_drs().reset()
        self._status_lbl.setText(f"Auto contrast: central {label} of range.")

    def _on_reset_display(self):
        """Reset display range to default percentile state."""
        self._auto_clip_idx = 0  # next Auto click starts the cycle at 99%
        self._target_drs().reset()
        self._status_lbl.setText("Display range reset.")

    # ── Per-image colormap ────────────────────────────────────────────────────
    def _on_viewer_colormap_changed(self, label: str) -> None:
        """Update the viewer colormap without touching browser thumbnails."""
        from probeflow.gui.rendering import CMAP_KEY as _CMAP_KEY
        self._viewer_colormap = _CMAP_KEY.get(label, label)
        self._refresh_viewer_pixmap(reset_zoom=False)

    def _set_viewer_colormap_by_key(self, mpl_key: str, label: str) -> None:
        """Set the viewer colormap directly by matplotlib key.

        Tries to sync the colormap combo-box by matching *label*.  Falls back
        to a direct pixmap refresh if the label is not found in the combo.
        """
        self._viewer_colormap = mpl_key
        if hasattr(self, "_viewer_cmap_cb"):
            idx = self._viewer_cmap_cb.findText(label)
            if idx >= 0:
                # Temporarily block signals to avoid a double-refresh
                self._viewer_cmap_cb.blockSignals(True)
                try:
                    self._viewer_cmap_cb.setCurrentIndex(idx)
                finally:
                    self._viewer_cmap_cb.blockSignals(False)
        self._refresh_viewer_pixmap(reset_zoom=False)

    def _on_colormap_picker(self) -> None:
        """Switch to the display panel and open the colormap combo drop-down."""
        self._show_sidebar_tab("display")
        if hasattr(self, "_viewer_cmap_cb"):
            self._viewer_cmap_cb.showPopup()

    # ── Stale manual contrast after a re-levelling op ─────────────────────────
    def _revalidate_manual_display_range(self) -> None:
        """Drop a manual contrast range that no longer overlaps the data.

        A processing op such as Set Zero Plane re-levels the image, so its
        value range can shift entirely away from the manual vmin/vmax the user
        set on the *previous* pipeline output.  Manual ``resolve()`` returns the
        stored limits without looking at the array, so every pixel would clamp
        to one end of the colour scale — the image goes flat and a colormap swap
        merely recolours that flat field (the "display error" symptom).  When
        the manual window no longer intersects the finite data range, fall back
        to percentile (auto) so the image — and the colormap — read correctly
        again.  Mirrors the manual-limit reset already done on channel change.

        Each per-region (per-ROI) range is revalidated against its own masked
        sub-array for the same reason.
        """
        arr = self._display_arr
        drs = getattr(self, "_drs", None)
        if arr is None or drs is None:
            return
        clip_low = getattr(self, "_clip_low", 1.0)
        clip_high = getattr(self, "_clip_high", 99.0)

        def _stale(drs, sub) -> bool:
            # A manual contrast window is "stale" when the (changed) data has
            # moved so far out from under it that the window now captures almost
            # no pixels — the image renders as a near-flat clamped field. Using
            # a small capture fraction rather than strict non-overlap also
            # catches the partial-overlap case where the window clips a thin
            # sliver of the data (e.g. a re-levelled current channel). This runs
            # only on processing changes, not on slider edits, so it never
            # fights a contrast the user is actively setting.
            if drs.mode != "manual" or drs.vmin is None or drs.vmax is None:
                return False
            finite = sub[np.isfinite(sub)]
            if finite.size == 0:
                return False
            inside = int(np.count_nonzero((finite >= drs.vmin) & (finite <= drs.vmax)))
            return inside < 0.02 * finite.size   # window captures <2% → stale

        if _stale(drs, arr):
            drs.reset_silent(clip_low, clip_high)

        region_drs = getattr(self, "_region_drs", None)
        if region_drs and getattr(self, "_image_roi_set", None) is not None:
            from probeflow.gui.roi_context import area_roi_mask
            shape = arr.shape[:2]
            for roi_id, drs in region_drs.items():
                roi = self._image_roi_set.get(roi_id)
                mask = area_roi_mask(roi, shape) if roi is not None else None
                if mask is None or not mask.any():
                    continue
                if _stale(drs, arr[mask]):
                    drs.reset_silent(clip_low, clip_high)

    # ── Display range sliders ─────────────────────────────────────────────────

    def _update_display_sliders(self) -> None:
        self._display_slider_ctrl.update()

    def _on_min_slider_changed(self, v: int) -> None:
        self._display_slider_ctrl.on_min_changed(v)

    def _on_max_slider_changed(self, v: int) -> None:
        self._display_slider_ctrl.on_max_changed(v)

    def _on_brightness_slider_changed(self, v: int) -> None:
        self._display_slider_ctrl.on_brightness_changed(v)

    def _on_contrast_slider_changed(self, v: int) -> None:
        self._display_slider_ctrl.on_contrast_changed(v)

    # ── Simple background subtraction ─────────────────────────────────────────
    def _on_simple_background(self) -> None:
        """Apply automated plane subtraction (order-1 polynomial fit, whole image)."""
        if self._display_arr is None:
            return
        self._push_proc_undo_snapshot()
        self._processing["plane_bg"] = {"order": 1}
        self._clear_bad_line_preview()
        self._refresh_processing_display()
        self._status_lbl.setText("Simple background: plane subtracted.")

    def _on_hist_context_menu(self, pos):
        menu = QMenu(self)
        auto_action = menu.addAction("Auto display range")
        export_action = menu.addAction("Export histogram...")
        chosen = menu.exec(self._hist_panel._canvas.mapToGlobal(pos))
        if chosen is auto_action:
            self._on_auto_clip()
        elif chosen is export_action:
            self._on_export_histogram()

    def _on_export_histogram(self):
        """Save the current histogram (bin centres + counts) as a TSV file."""
        ok, msg = export_histogram(
            self._hist_panel.flat_phys,
            self._entries[self._idx].stem,
            self._hist_panel.unit or "",
            self._ch_cb.currentText(),
            parent=self,
        )
        if msg:
            self._status_lbl.setText(msg)

    def _on_channel_changed(self, _: int):
        # Different channels have different physical units — reset manual limits.
        # Use reset_silent to avoid a premature refresh with stale channel data.
        self._drs.reset_silent(self._clip_low, self._clip_high)
        # Per-region manual limits are channel-specific; drop them too.
        self._region_drs.clear()
        self._hist_panel.clear(self._t)
        self._load_current(reset_zoom=True)

    @Slot(QImage, object)
    def _on_loaded(self, image: QImage, token):
        # ViewerLoader emits QImage (QPixmap is GUI-thread-only); convert here.
        if token is not self._token:
            return
        self._zoom_lbl.setText("")
        reset_zoom = self._reset_zoom_on_next_pixmap
        self._reset_zoom_on_next_pixmap = False
        self._zoom_lbl.set_source(QPixmap.fromImage(image), reset_zoom=reset_zoom)
        self._zoom_lbl.set_raw_array(self._display_arr)
        self._refresh_zero_markers()
        self._refresh_scale_bar()

    @Slot(str, object)
    def _on_viewer_pixmap_failed(self, message: str, token) -> None:
        if token is not self._token:
            return
        self._zoom_lbl.setText("Image render failed")
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(message)

    def _scan_extent_nm(self) -> tuple[float, float]:
        """Return (width_nm, height_nm) for the *displayed* scan, or (0,0).

        When the processing pipeline contains a shape-changing step that
        expanded the canvas (``rotate_arbitrary``, ``shear``, or
        ``affine_lattice_correction`` with canvas expansion), the displayed
        array is larger than the raw plane; the displayed scan_range must
        grow with it so the scale bar and rulers stay calibrated (review
        image-proc #4).  ``_display_scan_range_m`` is the post-processing
        extent (set by :meth:`_refresh_display_array`); fall back to
        ``_scan_range_m`` when unset.
        """
        scan_range = getattr(self, "_display_scan_range_m", None)
        if scan_range is None:
            scan_range = self._scan_range_m
        if scan_range is None:
            return 0.0, 0.0
        try:
            w_nm = float(scan_range[0]) * 1e9
            h_nm = float(scan_range[1]) * 1e9
        except (TypeError, ValueError, IndexError):
            return 0.0, 0.0
        return max(0.0, w_nm), max(0.0, h_nm)

    def _refresh_scale_bar(self):
        """Re-bind the scale bar + axes rulers to current scan/pixmap dimensions."""
        w_nm, h_nm = self._scan_extent_nm()
        pw = self._zoom_lbl.width()
        ph = self._zoom_lbl.height()
        self._scale_bar.set_scan_size(w_nm, pw)
        self._ruler_top.set_extent(w_nm, pw)
        self._ruler_left.set_extent(h_nm, ph)
        # The scroll area hosts a container (rulers + image), not the image
        # label directly. When the pixmap/ruler fixed sizes change, Qt does not
        # automatically resize that non-resizable scroll widget; without this,
        # the container can stay at its tiny construction-time size and show
        # only a postage-stamp slice of the large image.
        self._ruler_container.adjustSize()

    def _on_pixmap_resized(self, new_width_px: int):
        new_h = self._zoom_lbl.height()
        w_nm, h_nm = self._scan_extent_nm()
        self._scale_bar.set_scan_size(w_nm, new_width_px)
        self._ruler_top.set_extent(w_nm, new_width_px)
        self._ruler_left.set_extent(h_nm, new_h)
        self._ruler_container.adjustSize()

    # ── Controls ───────────────────────────────────────────────────────────────
    def _on_align_rows_changed(self, _index: int) -> None:
        """Apply row-alignment changes immediately without committing queued filters."""
        if not hasattr(self, "_processing_panel"):
            return
        align_value = self._processing_panel.state().get("align_rows")
        current_value = self._processing.get("align_rows")
        if current_value == align_value:
            self._sync_viewer_menu_actions()
            return
        if align_value is None and "align_rows" not in self._processing:
            self._sync_viewer_menu_actions()
            return
        base_state = copy.deepcopy(self._processing)
        base_state.pop("align_rows", None)
        base_snapshot = self._capture_proc_undo_snapshot(processing=base_state)
        coalesced_align_undo = self._proc_undo_ctrl.try_coalesce(base_snapshot)
        if not coalesced_align_undo:
            self._push_proc_undo_snapshot()
        if align_value is None:
            self._processing.pop("align_rows", None)
            label = "None"
        else:
            self._processing["align_rows"] = align_value
            label = str(align_value).replace("_", " ").title()
        if coalesced_align_undo:
            self._proc_undo_ctrl.discard_last_undo_if_eq(
                self._capture_proc_undo_snapshot()
            )
        self._clear_bad_line_preview()
        self._refresh_processing_display()
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(f"Align rows: {label}.")

    def _advanced_processing_state(self) -> dict:
        if not hasattr(self, "_undistort_shear_spin"):
            return {}
        shear_x = float(self._undistort_shear_spin.value())
        scale_y = float(self._undistort_scale_spin.value())
        state = {
            "linear_undistort": (shear_x != 0.0 or scale_y != 1.0),
            "undistort_shear_x": shear_x,
            "undistort_scale_y": scale_y,
        }
        if hasattr(self, "_advanced_fft_combo"):
            fft_map = {0: None, 1: "low_pass", 2: "high_pass"}
            fft_mode = fft_map.get(self._advanced_fft_combo.currentIndex())
            fft_cutoff = float(self._advanced_fft_cutoff_spin.value())
            if fft_mode is not None:
                state.update({
                    "fft_mode": fft_mode,
                    "fft_cutoff": fft_cutoff,
                    "fft_window": "hanning",
                })
            if self._advanced_fft_soft_cb.isChecked():
                state.update({
                    "fft_soft_border": True,
                    "fft_soft_mode": fft_mode or "low_pass",
                    "fft_soft_cutoff": fft_cutoff,
                    "fft_soft_border_frac": 0.12,
                })
        return state

    def _set_advanced_processing_state(self, state: dict | None) -> None:
        if not hasattr(self, "_undistort_shear_spin"):
            return
        state = state or {}
        if hasattr(self, "_advanced_fft_combo"):
            fft_mode = state.get("fft_mode") or state.get("fft_soft_mode")
            self._advanced_fft_combo.setCurrentIndex(
                {None: 0, "low_pass": 1, "high_pass": 2}.get(fft_mode, 0)
            )
            cutoff = state.get("fft_cutoff", state.get("fft_soft_cutoff", 0.10))
            self._advanced_fft_cutoff_spin.setValue(float(cutoff))
            self._advanced_fft_soft_cb.setChecked(bool(state.get("fft_soft_border", False)))
        self._undistort_shear_spin.setValue(float(state.get("undistort_shear_x", 0.0)))
        self._undistort_scale_spin.setValue(float(state.get("undistort_scale_y", 1.0)))
