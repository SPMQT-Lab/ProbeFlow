"""Processing, export, geometric-operation, and close handlers for ImageViewerDialog."""

from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog, QMenu

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
from probeflow.processing.state import assert_roi_references_resolved
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
        if getattr(self, "_processing_roi_error", ""):
            self._status_lbl.setText(
                f"Cannot export while processing has stale ROI references. {self._processing_roi_error}"
            )
            return
        if getattr(self, "_processing_error", ""):
            self._status_lbl.setText(f"Export blocked: {self._processing_error}")
            return
        try:
            ps = processing_state_from_gui(self._processing or {})
            assert_roi_references_resolved(ps, self._image_roi_set)
        except ValueError as _roi_err:
            self._status_lbl.setText(f"Export blocked: {_roi_err}")
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
        menu = QMenu(self)
        a_feat = QAction("→ Feature Counting", self)
        a_feat.setToolTip("Send processed image to Feature Counting tab")
        a_feat.triggered.connect(self._on_send_to_features)
        menu.addAction(a_feat)
        a_tv = QAction("→ TV Denoising", self)
        a_tv.setToolTip("Send processed image to TV Denoising tab")
        a_tv.triggered.connect(self._on_send_to_tv)
        menu.addAction(a_tv)
        menu.addSeparator()
        from PySide6.QtWidgets import QMenu as _QMenu
        transform_menu = _QMenu("Transform", self)
        for label, op in [
            ("Flip Horizontal", "flip_horizontal"),
            ("Flip Vertical",   "flip_vertical"),
            ("Rotate 90° CW",   "rotate_90_cw"),
            ("Rotate 180°",     "rotate_180"),
            ("Rotate 270° CW",  "rotate_270_cw"),
        ]:
            act = transform_menu.addAction(label)
            act.triggered.connect(
                (lambda _op=op: lambda: self._on_geometric_op(_op))()
            )
        arb_act = transform_menu.addAction("Rotate Arbitrary…")
        arb_act.triggered.connect(self._on_rotate_arbitrary)
        menu.addMenu(transform_menu)
        menu.addSeparator()
        a_png = QAction("⬇ Save PNG copy…", self)
        a_png.triggered.connect(self._on_save_png)
        menu.addAction(a_png)
        prof = self._line_profile_panel.profile_data()
        if prof is not None:
            a_csv = QAction("Export line profile as CSV…", self)
            a_csv.triggered.connect(self._on_export_line_profile_csv)
            menu.addAction(a_csv)
        menu.exec(pos)

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
        # Invalidate the in-flight worker token so any pending loaded() signal
        # is dropped rather than delivered to widgets that are being torn down.
        self._token = object()
        # Close any modeless child dialogs that hold a reference to self or to
        # the currently displayed Scan.  Without this they outlive the viewer.
        stm_dlg = getattr(self, "_stm_background_dialog", None)
        if stm_dlg is not None and stm_dlg.isVisible():
            stm_dlg.close()
        super().closeEvent(event)
