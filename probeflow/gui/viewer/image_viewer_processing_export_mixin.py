"""Processing, export, geometric-operation, and close handlers for ImageViewerDialog."""

from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtWidgets import QFileDialog

from probeflow.gui.config import load_config, save_config
from probeflow.gui.roi_context import active_area_roi_context
from probeflow.gui.viewer import (
    export_line_profile,
    save_viewer_png,
    transform_roi_set_for_display_op,
)
from probeflow.gui.viewer.processed_export import (
    build_processed_scan_for_export,
    save_processed_image,
    save_provenance_json,
)
from probeflow.processing.gui_adapter import processing_state_from_gui
from probeflow.processing.state import (
    apply_processing_state,
    assert_roi_references_resolved,
)
from probeflow.provenance import build_export_record, display_lines


class ImageViewerProcessingExportMixin:
    def _on_apply_processing(self):
        panel_state = self._processing_panel.state()
        panel_state.update(self._advanced_processing_state())
        has_roi_aware_local_filter = self._processing_has_roi_aware_local_filter(panel_state)
        active_roi = self._active_image_roi()
        active_area_roi_id = active_area_roi_context(self._image_roi_set).roi_id
        wants_filter_roi = self._scope_cb.currentIndex() == 1
        if (
            wants_filter_roi
            and active_roi is not None
            and active_area_roi_id is None
            and has_roi_aware_local_filter
        ):
            self._status_lbl.setText(
                f"Active {active_roi.kind} ROI is not valid for area processing; "
                "select an area ROI or delete/deselect it before applying local filters."
            )
            return
        if wants_filter_roi:
            if active_area_roi_id is None:
                self._status_lbl.setText("Select an active area ROI before using ROI filters.")
                return
        # Snapshot for undo before any mutation. Validation has passed; this
        # apply is going to change the state.
        self._push_proc_undo_snapshot()
        preserve = {
            key: self._processing[key]
            for key in (
                "set_zero_xy",
                "set_zero_plane_points",
                "set_zero_patch",
                "periodic_notches",
                "periodic_notch_radius",
                "geometric_ops",
                "arithmetic_ops",
                "stm_background",
                "plane_bg",
            )
            if key in self._processing
        }
        self._processing = panel_state
        self._processing.update(preserve)
        if wants_filter_roi and active_area_roi_id is not None:
            self._processing["processing_scope"] = "roi"
            self._processing["processing_roi_id"] = active_area_roi_id
        else:
            self._processing.pop("processing_scope", None)
            self._processing.pop("processing_roi_id", None)
        self._clear_bad_line_preview()
        self._refresh_processing_display()

    def _on_reset_processing(self):
        """Clear all processing for the current image and reload raw data."""
        has_zero = bool(self._zero_ctrl.points)
        if not self._processing and not has_zero:
            self._status_lbl.setText("Already showing the original — nothing to reset.")
            return
        # Snapshot for undo before clearing.
        self._push_proc_undo_snapshot()
        self._processing = {}
        self._processing_panel.set_state({})
        self._set_advanced_processing_state({})
        self._clear_bad_line_preview()
        # Untoggle any active set-zero pick modes so we don't re-pick on reload.
        if self._set_zero_plane_btn.isChecked():
            self._set_zero_plane_btn.setChecked(False)
        self._zero_ctrl.clear()
        self._set_selection_tool("none")
        self._scope_cb.setCurrentIndex(0)
        self._roi_status_lbl.setText("ROI filter scope: whole image")
        self._refresh_zero_markers()
        self._status_lbl.setText("Reset: showing original on-disk data.")
        self._refresh_processing_display()

    # ── Processing undo / redo ────────────────────────────────────────────────

    def _push_proc_undo_snapshot(self) -> None:
        self._proc_undo_ctrl.push(self._processing)

    def _restore_processing_state(self, state: dict) -> None:
        """Apply a snapshot to ``self._processing`` and resync the GUI."""
        self._processing = copy.deepcopy(state)
        self._processing_panel.set_state(self._processing)
        self._set_advanced_processing_state(self._processing)
        self._refresh_processing_display()

    def _on_undo_processing(self) -> None:
        state = self._proc_undo_ctrl.undo(self._processing)
        if state is None:
            return
        self._restore_processing_state(state)
        self._status_lbl.setText("Undo: restored previous processing.")

    def _on_redo_processing(self) -> None:
        state = self._proc_undo_ctrl.redo(self._processing)
        if state is None:
            return
        self._restore_processing_state(state)
        self._status_lbl.setText("Redo: reapplied processing.")

    def _update_undo_redo_buttons(self) -> None:
        if self._proc_undo_ctrl is not None:
            self._proc_undo_ctrl.update_buttons()

    def _on_save_png(self):
        entry = self._entries[self._idx]
        if not self._assert_exportable_processing():
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save PNG", str(Path.home() / f"{entry.stem}_viewer.png"),
            "PNG images (*.png)")
        if not out_path:
            return
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No data to save.")
            return
        msg = save_viewer_png(
            arr, out_path, entry.path,
            self._colormap, self._clip_low, self._clip_high,
            self._drs, self._processing, self._image_roi_set,
            self._ch_cb.currentIndex(), self._ch_cb.currentText() or None,
            processing_history=(
                self._processing_history.to_dict()
                if self._processing_history is not None else None
            ),
        )
        if msg.startswith("Saved") and self._processing_history is not None:
            self._mark_history_export(out_path, export_parameters={"export_kind": "viewer_png"})
        self._status_lbl.setText(msg)

    def _processing_state_has_image_arithmetic_operand(self, state) -> bool:
        for step in state.steps:
            if (
                step.op == "arithmetic"
                and step.params.get("operand_type") == "image"
            ):
                return True
            if step.op == "roi":
                nested = step.params.get("step")
                if not isinstance(nested, dict):
                    continue
                params = nested.get("params", {})
                if (
                    nested.get("op") == "arithmetic"
                    and isinstance(params, dict)
                    and params.get("operand_type") == "image"
                ):
                    return True
        return False

    def _assert_exportable_processing(self) -> bool:
        if getattr(self, "_processing_roi_error", ""):
            self._status_lbl.setText(
                f"Cannot export while processing has stale ROI references. {self._processing_roi_error}"
            )
            return False
        if getattr(self, "_processing_error", ""):
            self._status_lbl.setText(f"Export blocked: {self._processing_error}")
            return False
        try:
            ps = processing_state_from_gui(self._processing or {})
            assert_roi_references_resolved(ps, self._image_roi_set)
        except ValueError as _roi_err:
            self._status_lbl.setText(f"Export blocked: {_roi_err}")
            return False
        if (
            self._raw_arr is not None
            and self._processing_state_has_image_arithmetic_operand(ps)
        ):
            try:
                apply_processing_state(self._raw_arr, ps, self._image_roi_set)
            except Exception as exc:
                self._status_lbl.setText(f"Export blocked: Processing failed: {exc}")
                return False
        return True

    def _current_display_settings(self) -> dict:
        from probeflow.provenance.export import png_display_state

        return png_display_state(
            self._drs,
            clip_low=self._clip_low,
            clip_high=self._clip_high,
            colormap=self._viewer_colormap,
            add_scalebar=True,
            scalebar_unit="nm",
            scalebar_pos="bottom-right",
        )

    def _processed_scan_for_export(self):
        entry = self._entries[self._idx]
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        return build_processed_scan_for_export(
            entry.path, self._ch_cb.currentIndex(), arr, self._processing or {},
        )

    def _on_save_processed_image(self):
        if not self._assert_exportable_processing():
            return
        entry = self._entries[self._idx]
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save processed image",
            str(Path.home() / f"{entry.stem}_processed.sxm"),
            (
                "Supported images (*.sxm *.png *.csv *.pdf *.gwy);;"
                "Nanonis SXM (*.sxm);;PNG images (*.png);;"
                "CSV grids (*.csv);;PDF figures (*.pdf);;Gwyddion (*.gwy)"
            ),
        )
        if not out_path:
            return
        out = Path(out_path)
        if not out.suffix:
            out = out.with_suffix(".sxm")
        try:
            scan, plane_idx = self._processed_scan_for_export()
        except ValueError as exc:
            self._status_lbl.setText(str(exc))
            return
        msg = save_processed_image(
            scan, plane_idx, out,
            colormap=self._viewer_colormap,
            clip_low=self._clip_low,
            clip_high=self._clip_high,
            display_settings=self._current_display_settings(),
            roi_set=self._image_roi_set,
            processing_history=(
                self._processing_history.to_dict()
                if self._processing_history is not None else None
            ),
        )
        self._status_lbl.setText(msg)

    def _on_save_provenance(self):
        if not self._assert_exportable_processing():
            return
        if self._processing_history is None:
            self._status_lbl.setText("No provenance available to save.")
            return
        entry = self._entries[self._idx]
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save provenance",
            str(Path.home() / f"{entry.stem}.probeflow.json"),
            "ProbeFlow provenance (*.probeflow.json *.json)",
        )
        if not out_path:
            return
        out = Path(out_path)
        if not out.suffix:
            out = out.with_suffix(".probeflow.json")
        try:
            msg, record = save_provenance_json(
                self._processing_history,
                out,
                display_settings=self._current_display_settings(),
            )
            self._last_export_record = record
            self._history_text.setText(
                "\n".join(display_lines(record.processing_history))
            )
            self._status_lbl.setText(msg)
        except Exception as exc:
            self._status_lbl.setText(f"Save provenance error: {exc}")

    def _mark_history_export(self, out_path: str, export_parameters: dict | None = None) -> None:
        try:
            record = build_export_record(
                self._processing_history,
                export_path=out_path,
                export_format="png",
                display_settings=self._current_display_settings(),
                export_parameters=export_parameters,
            )
            self._last_export_record = record
            self._history_text.setText(
                "\n".join(display_lines(record.processing_history))
            )
        except Exception:
            pass

    def _on_send_to_features(self):
        self._deferred.action = "features"
        self._deferred.plane_idx = self._ch_cb.currentIndex()
        self.accept()

    def _on_send_to_tv(self):
        self._deferred.action = "tv"
        self._deferred.plane_idx = self._ch_cb.currentIndex()
        self.accept()

    def _on_image_context_menu(self, pos):
        from probeflow.gui.viewer.context_menus import build_blank_image_context_menu

        menu = build_blank_image_context_menu(self)
        menu.exec(pos)

    def _on_threshold(self) -> None:
        """Open the Threshold dialog (modeless) for the current image."""
        from probeflow.gui.dialogs.threshold_dialog import ThresholdDialog
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return

        # Snapshot of the pipeline-output array at dialog-open time.
        # _preview must restore _display_arr to this value immediately after
        # submitting the render job so that other concurrent modeless operations
        # (Scale, Shear, etc.) read the real pipeline state, not a transient
        # preview result.
        _pipeline_arr = arr

        def _preview(result_arr):
            # Temporarily install the preview array so _refresh_viewer_pixmap
            # reads its range for vmin/vmax and passes it to the ViewerLoader.
            self._display_arr = result_arr
            self._refresh_viewer_pixmap(reset_zoom=False)
            # Restore immediately — ViewerLoader captured its own reference to
            # result_arr at construction; restoring here is safe and prevents
            # other handlers from seeing the transient preview data.
            self._display_arr = _pipeline_arr

        def _preview_pixmap(pixmap):
            # Coloured highlight preview — set pixmap directly, bypass async loader.
            self._zoom_lbl.setText("")
            self._zoom_lbl.set_source(pixmap, reset_zoom=False)

        def _clear_preview():
            # Restore the full processing-pipeline display.
            self._refresh_processing_display()

        dlg = ThresholdDialog(
            arr,
            preview_fn=_preview,
            preview_pixmap_fn=_preview_pixmap,
            clear_preview_fn=_clear_preview,
            theme=getattr(self, "_t", None),
            parent=self,
        )
        dlg.applied.connect(self._on_threshold_applied)
        # Track so closeEvent can explicitly close the dialog before the viewer
        # tears down, preventing queued HistogramPanel signals from firing into
        # partially-destroyed viewer widgets.
        self._threshold_dialog = dlg
        dlg.show()

    def _on_threshold_applied(self, params: dict) -> None:
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": "image_threshold", "params": params})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()
        mode = params.get("mode", "clip")
        self._status_lbl.setText(f"Threshold applied ({mode} mode).")

    def _on_scale_image(self) -> None:
        """Open the Scale dialog to resample the image to new pixel dimensions."""
        from probeflow.gui.dialogs.scale_dialog import ScaleDialog
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        dlg = ScaleDialog(
            arr.shape,
            scan_range_m=self._scan_range_m,
            parent=self,
        )
        dlg.applied.connect(self._on_scale_image_applied)
        dlg.exec()

    def _on_scale_image_applied(self, params: dict) -> None:
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": "scale_image", "params": params})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()
        w, h = params["new_width"], params["new_height"]
        self._status_lbl.setText(
            f"Scaled to {w} × {h} px (ROI coordinates may be invalid — use Reset to undo)."
        )

    def _on_shear(self) -> None:
        """Open the Shear dialog to apply a 2-component shear correction."""
        from probeflow.gui.dialogs.shear_dialog import ShearDialog
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        dlg = ShearDialog(parent=self)
        dlg.applied.connect(self._on_shear_applied)
        dlg.exec()

    def _on_shear_applied(self, params: dict) -> None:
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": "shear", "params": params})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()
        sx, sy = params.get("shear_x", 0.0), params.get("shear_y", 0.0)
        self._status_lbl.setText(f"Shear applied (x={sx:.4f}, y={sy:.4f}).")

    def _on_convert_bit_depth(self, bits: int) -> None:
        """Quantize the current image to *bits*-bit precision as a processing step."""
        arr = self._display_arr if self._display_arr is not None else self._raw_arr
        if arr is None:
            self._status_lbl.setText("No image loaded.")
            return
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": "quantize_bit_depth", "params": {"bits": bits}})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()
        n = 2 ** bits
        self._status_lbl.setText(f"Converted to {bits}-bit ({n:,} levels).")

    def _on_geometric_op(self, op_name: str) -> None:
        self._transform_image_roi_set_for_display_op(op_name)
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": op_name, "params": {}})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()

    def _on_rotate_arbitrary(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        angle, ok = QInputDialog.getDouble(
            self, "Rotate Arbitrary",
            "Angle (degrees, positive = counter-clockwise):",
            0.0, -360.0, 360.0, 1,
        )
        if not ok:
            return
        self._transform_image_roi_set_for_display_op(
            "rotate_arbitrary",
            {"angle_degrees": angle},
        )
        ops = list(self._processing.get("geometric_ops") or [])
        ops.append({"op": "rotate_arbitrary", "params": {"angle_degrees": angle}})
        self._processing["geometric_ops"] = ops
        self._refresh_processing_display()

    def _transform_image_roi_set_for_display_op(
        self,
        op_name: str,
        params: dict | None = None,
    ) -> None:
        transform_roi_set_for_display_op(
            self._image_roi_set,
            op_name,
            params,
            self._current_array_shape(),
            status_fn=(
                self._status_lbl.setText if hasattr(self, "_status_lbl") else None
            ),
            roi_changed_fn=self._on_image_roi_set_changed,
        )

    def _on_export_line_profile_csv(self):
        prof = self._line_profile_panel.profile_data()
        if prof is None:
            self._status_lbl.setText("No line profile to export (draw a line first).")
            return
        x_vals, y_vals, x_label, y_label = prof
        entry = self._entries[self._idx]
        ok, msg = export_line_profile(
            x_vals, y_vals, x_label, y_label,
            entry.stem,
            self._scan_header or {},
            parent=self,
        )
        if msg:
            self._status_lbl.setText(msg)

    def closeEvent(self, event):
        try:
            cfg = load_config()
            self._save_viewer_desktop_layout_into(cfg)
            save_config(cfg)
        except Exception:
            pass
        # Invalidate the in-flight worker token so any pending loaded() signal
        # is dropped rather than delivered to widgets that are being torn down.
        self._token = object()
        # Close any modeless child dialogs that hold a reference to self or to
        # the currently displayed Scan.  Without this they outlive the viewer.
        stm_dlg = getattr(self, "_stm_background_dialog", None)
        if stm_dlg is not None and stm_dlg.isVisible():
            stm_dlg.close()
        thr_dlg = getattr(self, "_threshold_dialog", None)
        if thr_dlg is not None and thr_dlg.isVisible():
            thr_dlg.close()
        super().closeEvent(event)
