"""Toolbar, menu, and measurement visibility handlers for ImageViewerDialog."""

from __future__ import annotations

from probeflow.core import AREA_ROI_KINDS
from probeflow.gui.dialogs.about import AboutDialog
from probeflow.gui.dialogs.definitions import _DefinitionsDialog
from probeflow.gui.roi_context import (
    active_line_roi_context,
    selected_or_active_area_roi_context,
)
from probeflow.gui.viewer.shortcuts import viewer_command


class ImageViewerToolbarMixin:
    def _show_viewer_about(self) -> None:
        dlg = AboutDialog(self._t, self)
        dlg.exec()

    def _show_viewer_howto(self) -> None:
        dlg = getattr(self, "_definitions_dialog", None)
        if dlg is None:
            dlg = _DefinitionsDialog(self._t, self, initial_tab="howto")
            self._definitions_dialog = dlg
        else:
            dlg.set_reference_tab("howto")
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _show_viewer_definitions(self) -> None:
        dlg = getattr(self, "_definitions_dialog", None)
        if dlg is None:
            dlg = _DefinitionsDialog(self._t, self, initial_tab="processing")
            self._definitions_dialog = dlg
        else:
            dlg.set_reference_tab("processing")
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _show_viewer_roi_reference(self) -> None:
        dlg = getattr(self, "_definitions_dialog", None)
        if dlg is None:
            dlg = _DefinitionsDialog(self._t, self, initial_tab="roi")
            self._definitions_dialog = dlg
        else:
            dlg.set_reference_tab("roi")
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _show_image_viewer_shortcuts(self) -> None:
        from probeflow.gui.viewer.onboarding import ImageViewerShortcutsDialog

        dlg = ImageViewerShortcutsDialog(self)
        dlg.exec()

    def _show_command_finder(self) -> None:
        from probeflow.gui.viewer.command_finder import CommandFinderDialog

        self._sync_viewer_menu_actions()
        dlg = getattr(self, "_command_finder_dialog", None)
        if dlg is None:
            dlg = CommandFinderDialog(self._viewer_command_actions, parent=self)
            self._command_finder_dialog = dlg
        else:
            dlg.set_actions(self._viewer_command_actions)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        dlg.focus_search()

    def _sync_viewer_menu_actions(self) -> None:
        if hasattr(self, "_viewer_processing_actions"):
            for key, value in self._viewer_processing_actions.items():
                if isinstance(value, dict):
                    title = key.removeprefix("combo:")
                    combo = {
                        "Align rows": self._processing_panel._align_combo,
                        "Bad line correction": self._processing_panel._bad_lines_combo,
                        "Smooth": self._processing_panel._smooth_combo,
                        "Hi-pass": self._processing_panel._highpass_combo,
                        "Edge filter": self._processing_panel._edge_combo,
                    }.get(title)
                    current = combo.currentText() if combo is not None else ""
                    for label, action in value.items():
                        action.blockSignals(True)
                        action.setChecked(label == current)
                        action.blockSignals(False)
                    continue
                if key == "zero_plane":
                    value.blockSignals(True)
                    value.setChecked(self._set_zero_plane_btn.isChecked())
                    value.blockSignals(False)
                elif key == "undo":
                    value.setEnabled(self._proc_undo_ctrl.can_undo)
                elif key == "redo":
                    value.setEnabled(self._proc_undo_ctrl.can_redo)

        if hasattr(self, "_viewer_roi_tool_actions"):
            tool = self._zoom_lbl.tool()
            for key, action in self._viewer_roi_tool_actions.items():
                action.blockSignals(True)
                action.setChecked(key == tool)
                action.blockSignals(False)
            if tool in {"rectangle", "ellipse", "polygon", "freehand", "point"}:
                self._show_sidebar_tab("roi")
            elif tool == "line":
                self._show_sidebar_tab("measurements")
                if hasattr(self, "_measurement_panel"):
                    self._measurement_panel.set_measurement_type("line_profile")

        roi = None
        is_area = False
        if hasattr(self, "_image_roi_set"):
            roi_id = self._selected_or_active_image_roi_id()
            roi = self._image_roi_set.get(roi_id) if (self._image_roi_set and roi_id) else None
            is_area = roi is not None and roi.kind in AREA_ROI_KINDS

        if hasattr(self, "_viewer_roi_actions"):
            for key, action in self._viewer_roi_actions.items():
                action.setEnabled(roi is not None)
                if key in ("invert", "mask"):
                    action.setEnabled(is_area)

        measurement_states = {}
        if hasattr(self, "_viewer_measurement_actions"):
            measurement_states = self._image_measurements.action_enabled_state()
            for key, action in self._viewer_measurement_actions.items():
                action.setEnabled(measurement_states.get(key, True))

        if hasattr(self, "_viewer_command_actions"):
            for command_id, action in self._viewer_command_actions.items():
                key = viewer_command(command_id).enabled_state_key
                if key in measurement_states:
                    action.setEnabled(measurement_states[key])
                elif key == "lattice_grid":
                    action.setEnabled(getattr(self, "_lattice_grid_item", None) is not None)
                elif key == "undo":
                    action.setEnabled(self._proc_undo_ctrl.can_undo)
                elif key == "redo":
                    action.setEnabled(self._proc_undo_ctrl.can_redo)

        if hasattr(self, "_quick_toolbar"):
            is_line = bool(self._active_line_roi_id())
            self._quick_toolbar.set_action_enabled(
                "line_periodicity",
                is_line,
                enabled_tip="Estimate periodicity from the active line ROI.",
                disabled_tip="Draw or select a line ROI to estimate periodicity.",
            )
            self._quick_toolbar.set_action_enabled(
                "line_profile",
                is_line,
                enabled_tip="Show a line profile for the active line ROI.",
                disabled_tip="Draw or select a line ROI to show a line profile.",
            )
            self._quick_toolbar.set_action_enabled(
                "mask_selection",
                is_area,
                enabled_tip="Use the active area ROI as a processing mask.",
                disabled_tip="Draw or select an area ROI to enable mask-based processing.",
            )
            self._quick_toolbar.set_action_enabled(
                "invert_selection",
                is_area,
                enabled_tip="Invert the active area ROI.",
                disabled_tip="Draw or select an area ROI to invert the selection.",
            )

    def _set_selection_tool(self, kind: str) -> None:
        """Compat shim: delegates to _set_drawing_tool, mapping 'none' → 'pan'."""
        self._set_drawing_tool(kind if kind and kind != "none" else "pan")

    def _set_drawing_tool(self, kind: str) -> None:
        """Activate a drawing tool both on the canvas and in the toolbar."""
        kind = str(kind or "pan")
        from probeflow.gui.tool_manager import TOOLS
        if kind not in TOOLS:
            kind = "pan"
        if hasattr(self, "_quick_toolbar"):
            self._quick_toolbar.set_active_mode(kind)
        self._zoom_lbl.set_tool(kind)
        self._sync_line_profile_visibility(kind)
        from probeflow.gui.tool_manager import _TOOL_HINTS
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(_TOOL_HINTS.get(kind, ""))
        self._sync_viewer_menu_actions()

    def _on_quick_toolbar_mode(self, key: str) -> None:
        """Handle a drawing-mode request from the quick toolbar."""
        if self._set_zero_plane_btn.isChecked():
            self._set_zero_plane_btn.setChecked(False)
        self._set_drawing_tool(key)

    def _on_quick_toolbar_action(self, key: str) -> None:
        """Dispatch an action request from the quick toolbar to existing handlers."""
        dispatch = {
            "clear_selection":   self._clear_all_image_marks,
            "auto_contrast":     self._on_auto_clip,
            "plane_background":  self._on_simple_background,
            "stm_background":    self._on_open_stm_background,
            "bad_lines":         self._on_preview_bad_lines,
            "open_fft":          self._on_open_fft_viewer,
            "open_lattice_grid": self._on_open_lattice_grid,
            "line_periodicity":  self._image_measurements.find_periodicity_for_active_line_roi,
            "line_profile":      self._image_measurements.add_current_line_profile_measurement,
            "mask_selection":    self._on_mask_selection,
            "invert_selection":  self._invert_active_image_roi,
        }
        handler = dispatch.get(key)
        if handler is not None:
            handler()

    def _on_mask_selection(self) -> None:
        """Apply ROI-scoped filter mask from the active area ROI."""
        roi_ctx = selected_or_active_area_roi_context(
            self._image_roi_set,
            getattr(self, "_roi_panel", None),
        )
        if roi_ctx.roi is None:
            if hasattr(self, "_status_lbl"):
                self._status_lbl.setText(
                    "Select an area ROI first to use mask-based processing."
                )
            return
        self._scope_cb.setCurrentText("ROI filters only")
        self._show_sidebar_tab("processing")
        if hasattr(self, "_status_lbl"):
            self._status_lbl.setText(
                f"ROI filter scope set to '{roi_ctx.roi.name}'. "
                "Filters now apply inside the ROI only."
            )

    def _active_line_roi_id(self) -> "str | None":
        """Return the active ROI id if it is a line ROI, else None."""
        return active_line_roi_context(getattr(self, "_image_roi_set", None)).roi_id

    def _sync_line_profile_visibility(self, kind: str | None = None) -> None:
        if not hasattr(self, "_line_profile_panel"):
            return
        tool_is_line = (kind or self._zoom_lbl.selection_tool()) == "line"
        active_line_id = self._active_line_roi_id()
        is_line = tool_is_line or (active_line_id is not None)
        self._line_profile_panel.setVisible(is_line)
        if is_line and active_line_id is not None:
            self._on_roi_line_profile(active_line_id)
        else:
            self._line_profile_panel.show_empty(theme=self._t)
            if hasattr(self, "_measurement_panel"):
                self._measurement_panel.clear_line_profile_live()

    def _pixel_size_xy_m(self) -> tuple[float, float]:
        shape = self._current_array_shape()
        if shape is None or self._scan_range_m is None:
            return 1e-10, 1e-10
        Ny, Nx = shape
        try:
            w_m = float(self._scan_range_m[0])
            h_m = float(self._scan_range_m[1])
        except (TypeError, ValueError, IndexError):
            return 1e-10, 1e-10
        px_x = w_m / Nx if Nx > 0 and w_m > 0 else 1e-10
        px_y = h_m / Ny if Ny > 0 and h_m > 0 else 1e-10
        return px_x, px_y
