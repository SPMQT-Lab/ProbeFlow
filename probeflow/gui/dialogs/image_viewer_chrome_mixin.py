"""Menu bar, window-layout, sidebar/tool, and modal chrome for the image viewer.

Mixin split out of ``image_viewer.py``. Methods here build the dialog's menu bar
and manage the sidebar tabs, in-sidebar tool hosting, dismissible modal overlays,
desktop-layout persistence, and window docking. They rely on attributes and
handlers owned by ``ImageViewerDialog`` and its other mixins (resolved via the
class MRO).
"""

from __future__ import annotations

import logging

from probeflow.gui.config import GITHUB_URL, load_config, save_config

_log = logging.getLogger(__name__)
from probeflow.gui.desktop_layout import (
    apply_screen_fraction_geometry,
    b64_to_qbytearray,
    qbytearray_to_b64,
    restore_geometry_or_default,
)
from probeflow.gui.utils import _open_url
from probeflow.gui.viewer.floating_panel import ModalOverlay
from probeflow.gui.viewer.shortcuts import viewer_command
from probeflow.gui.viewer.window_menu import populate_window_menu
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QComboBox, QDockWidget, QLabel, QMenu, QVBoxLayout, QWidget,
)


class ImageViewerChromeMixin:
    """Menu bar, sidebar/tool hosting, modal overlays, and window layout."""

    def _configure_viewer_action(self, action: QAction, command_id: str) -> QAction:
        command = viewer_command(command_id)
        action.setText(command.label)
        if command.shortcuts:
            action.setShortcuts([QKeySequence(s) for s in command.shortcuts])
        if command.status_tip:
            action.setStatusTip(command.status_tip)
            action.setToolTip(command.status_tip)
        self._viewer_command_actions[command_id] = action
        return action

    def _viewer_action(
        self,
        command_id: str,
        handler=None,
        *,
        register: dict[str, QAction] | None = None,
    ) -> QAction:
        action = self._configure_viewer_action(QAction(self), command_id)
        if handler is not None:
            action.triggered.connect(handler)
        command = viewer_command(command_id)
        if register is not None:
            key = command.enabled_state_key or command.command_id
            register[key] = action
        return action

    def _build_viewer_menu_bar(self) -> None:
        menu_bar = self._viewer_main.menuBar()
        self._viewer_processing_actions: dict[str, QAction | dict[str, QAction]] = {}
        self._viewer_roi_tool_actions: dict[str, QAction] = {}
        self._viewer_roi_actions: dict[str, QAction] = {}
        self._viewer_measurement_actions: dict[str, QAction] = {}
        self._viewer_command_actions: dict[str, QAction] = {}

        file_menu = menu_bar.addMenu("File")
        close_action = QAction("Close", self)
        close_action.setShortcut(QKeySequence.Close)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

        view_menu = menu_bar.addMenu("View")
        auto_contrast_action = self._viewer_action(
            "view.auto_contrast",
            self._on_auto_clip,
        )
        view_menu.addAction(auto_contrast_action)
        reset_contrast_action = self._viewer_action(
            "view.reset_contrast",
            self._on_reset_display,
        )
        view_menu.addAction(reset_contrast_action)
        view_menu.addSeparator()
        fit_action = self._viewer_action("view.fit", self._zoom_lbl.fit_to_view)
        view_menu.addAction(fit_action)
        native_action = self._viewer_action("view.one_to_one", self._zoom_lbl.reset_zoom)
        view_menu.addAction(native_action)
        view_menu.addSeparator()
        reset_layout_action = self._viewer_action(
            "view.reset_layout",
            self._reset_viewer_window_layout,
        )
        view_menu.addAction(reset_layout_action)
        view_menu.addSeparator()
        display_panel_action = self._viewer_action(
            "panel.view",
            lambda: self._show_sidebar_tab("display"),
        )
        view_menu.addAction(display_panel_action)
        processing_panel_action = self._viewer_action(
            "panel.process",
            lambda: self._show_sidebar_tab("processing"),
        )
        view_menu.addAction(processing_panel_action)
        roi_panel_action = self._viewer_action(
            "panel.roi",
            lambda: self._show_sidebar_tab("roi"),
        )
        view_menu.addAction(roi_panel_action)
        measurements_panel_action = self._viewer_action(
            "panel.measure",
            lambda: self._show_sidebar_tab("measurements"),
        )
        view_menu.addAction(measurements_panel_action)
        export_panel_action = self._viewer_action(
            "panel.export",
            lambda: self._show_sidebar_tab("export"),
        )
        view_menu.addAction(export_panel_action)
        view_menu.addSeparator()
        dock_panels_action = self._viewer_action(
            "view.dock_panels",
            self._dock_panels_into_window,
        )
        view_menu.addAction(dock_panels_action)

        # ── Image menu ────────────────────────────────────────────────────────
        image_menu = menu_bar.addMenu("Image")

        info_action = self._viewer_action("image.info", self._on_show_image_info)
        image_menu.addAction(info_action)
        image_menu.addSeparator()

        # Color submenu
        color_menu = image_menu.addMenu("Color")
        _QUICK_CMAPS = [
            ("Gray",         "gray"),
            ("Gray (inv.)",  "gray_r"),
            ("Hot",          "hot"),
            ("AFM Hot",      "afmhot"),
            ("Viridis",      "viridis"),
            ("Plasma",       "plasma"),
            ("Inferno",      "inferno"),
            ("Cividis",      "cividis"),
        ]
        for _cmap_label, _mpl_key in _QUICK_CMAPS:
            _act = QAction(_cmap_label, self)
            _act.triggered.connect(
                lambda _c=False, k=_mpl_key, lbl=_cmap_label:
                    self._set_viewer_colormap_by_key(k, lbl)
            )
            color_menu.addAction(_act)
        color_menu.addSeparator()
        colormap_action = self._viewer_action("image.colormap", self._on_colormap_picker)
        color_menu.addAction(colormap_action)
        image_menu.addSeparator()

        # Adjust submenu
        adjust_menu = image_menu.addMenu("Adjust")
        bc_action = self._viewer_action(
            "panel.view",
            lambda: self._show_sidebar_tab("display"),
        )
        adjust_menu.addAction(bc_action)
        threshold_action = self._viewer_action("image.threshold", self._on_threshold)
        adjust_menu.addAction(threshold_action)

        scale_action = self._viewer_action("image.scale", self._on_scale_image)
        image_menu.addAction(scale_action)

        # Type submenu (bit-depth quantization)
        type_menu = image_menu.addMenu("Type")
        type_menu.addAction(
            self._viewer_action("image.type_8bit",
                                lambda: self._on_convert_bit_depth(8))
        )
        type_menu.addAction(
            self._viewer_action("image.type_16bit",
                                lambda: self._on_convert_bit_depth(16))
        )
        image_menu.addSeparator()

        # Transform submenu
        transform_menu = image_menu.addMenu("Transform")
        transform_menu.addAction(
            self._viewer_action("image.flip_h",
                                lambda: self._on_geometric_op("flip_horizontal"))
        )
        transform_menu.addAction(
            self._viewer_action("image.flip_v",
                                lambda: self._on_geometric_op("flip_vertical"))
        )
        transform_menu.addSeparator()
        transform_menu.addAction(
            self._viewer_action("image.rotate_90_cw",
                                lambda: self._on_geometric_op("rotate_90_cw"))
        )
        transform_menu.addAction(
            self._viewer_action("image.rotate_180",
                                lambda: self._on_geometric_op("rotate_180"))
        )
        transform_menu.addAction(
            self._viewer_action("image.rotate_270_cw",
                                lambda: self._on_geometric_op("rotate_270_cw"))
        )
        transform_menu.addAction(
            self._viewer_action("image.rotate_arbitrary", self._on_rotate_arbitrary)
        )
        transform_menu.addSeparator()
        transform_menu.addAction(
            self._viewer_action("image.shear", self._on_shear)
        )

        processing_menu = menu_bar.addMenu("Processing")
        plane_action = self._viewer_action(
            "processing.plane_background",
            self._on_simple_background,
        )
        processing_menu.addAction(plane_action)
        stm_bg_top_action = self._viewer_action(
            "processing.stm_background",
            self._on_open_stm_background,
        )
        processing_menu.addAction(stm_bg_top_action)
        bad_lines_top_action = self._viewer_action(
            "processing.bad_lines",
            self._on_preview_bad_lines,
        )
        processing_menu.addAction(bad_lines_top_action)
        image_ops_action = self._viewer_action(
            "processing.image_operations",
            self._on_open_image_operations,
        )
        processing_menu.addAction(image_ops_action)
        processing_menu.addSeparator()
        self._add_combo_menu(
            processing_menu, "Align rows", self._processing_panel._align_combo,
            ["None", "Median", "Mean"],
        )
        self._add_combo_menu(
            processing_menu, "Bad line correction", self._processing_panel._bad_lines_combo,
            ["None", "Step segments", "MAD/outlier segments"],
        )
        self._add_combo_menu(
            processing_menu, "Smooth", self._processing_panel._smooth_combo,
            ["None", "Gaussian"],
        )
        self._add_combo_menu(
            processing_menu, "Hi-pass", self._processing_panel._highpass_combo,
            ["None", "Gaussian"],
        )
        self._add_combo_menu(
            processing_menu, "Edge filter", self._processing_panel._edge_combo,
            ["None", "Laplacian", "LoG", "DoG"],
        )
        processing_menu.addSeparator()

        zero_action = self._viewer_action(
            "processing.zero_plane",
            self._set_zero_plane_btn.setChecked,
            register=self._viewer_processing_actions,
        )
        zero_action.setCheckable(True)
        self._set_zero_plane_btn.toggled.connect(self._sync_viewer_menu_actions)
        processing_menu.addAction(zero_action)
        clear_zero_action = self._viewer_action(
            "processing.clear_zero",
            self._on_clear_set_zero,
        )
        processing_menu.addAction(clear_zero_action)
        processing_menu.addSeparator()

        apply_action = self._viewer_action(
            "processing.apply",
            self._on_apply_processing,
        )
        processing_menu.addAction(apply_action)
        undo_action = self._viewer_action(
            "processing.undo",
            self._on_undo_processing,
            register=self._viewer_processing_actions,
        )
        processing_menu.addAction(undo_action)
        redo_action = self._viewer_action(
            "processing.redo",
            self._on_redo_processing,
            register=self._viewer_processing_actions,
        )
        processing_menu.addAction(redo_action)
        reset_action = self._viewer_action(
            "processing.reset",
            self._on_reset_processing,
        )
        processing_menu.addAction(reset_action)

        roi_menu = menu_bar.addMenu("ROI")
        show_roi_manager_action = QAction("Show ROI Manager", self)
        show_roi_manager_action.triggered.connect(self._show_roi_manager)
        roi_menu.addAction(show_roi_manager_action)
        roi_reference_action = self._viewer_action(
            "help.roi_reference",
            self._show_viewer_roi_reference,
        )
        roi_menu.addAction(roi_reference_action)
        roi_menu.addSeparator()
        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)
        for key, label in (
            ("pan", "Pan"),
            ("rectangle", "Rectangle"),
            ("ellipse", "Ellipse"),
            ("polygon", "Polygon"),
            ("freehand", "Freehand"),
            ("line", "Line"),
            ("point", "Point"),
        ):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=key: self._set_drawing_tool(value)
            )
            tool_group.addAction(action)
            self._viewer_roi_tool_actions[key] = action
            roi_menu.addAction(action)
        roi_menu.addSeparator()

        rename_action = QAction("Rename ROI", self)
        rename_action.triggered.connect(self._rename_active_image_roi)
        self._viewer_roi_actions["rename"] = rename_action
        roi_menu.addAction(rename_action)
        delete_action = QAction("Delete ROI", self)
        delete_action.triggered.connect(self._delete_active_image_roi)
        self._viewer_roi_actions["delete"] = delete_action
        roi_menu.addAction(delete_action)
        set_active_action = QAction("Set active ROI", self)
        set_active_action.triggered.connect(self._set_selected_or_active_image_roi)
        self._viewer_roi_actions["set_active"] = set_active_action
        roi_menu.addAction(set_active_action)
        invert_action = QAction("Invert ROI", self)
        invert_action.triggered.connect(self._invert_active_image_roi)
        self._viewer_roi_actions["invert"] = invert_action
        roi_menu.addAction(invert_action)
        mask_action = QAction("Mask from selection", self)
        mask_action.triggered.connect(self._on_mask_selection)
        self._viewer_roi_actions["mask"] = mask_action
        roi_menu.addAction(mask_action)
        # Planned ROI menu additions: Grow ROI, Shrink ROI, Specify ROI.
        # These should remain hidden or disabled until implemented in the ROI backend.

        measurements_menu = menu_bar.addMenu("Measurements")
        ruler_action = self._viewer_action("measure.distance", self._on_measure_distance)
        measurements_menu.addAction(ruler_action)
        angle_action = self._viewer_action("measure.angle", self._on_measure_angle)
        measurements_menu.addAction(angle_action)
        update_angle_action = self._viewer_action(
            "measure.update_angle", self._on_update_angle_measurement,
        )
        measurements_menu.addAction(update_angle_action)
        roi_stats_new_action = self._viewer_action(
            "measure.roi_stats",
            self._on_measure_roi_stats,
        )
        measurements_menu.addAction(roi_stats_new_action)
        measurements_menu.addSeparator()
        add_roi_stats_action = self._viewer_action(
            "measure.add_roi_stats",
            self._roi_stats_active_and_show,
            register=self._viewer_measurement_actions,
        )
        measurements_menu.addAction(add_roi_stats_action)
        add_step_height_action = self._viewer_action(
            "measure.step_height",
            self._image_measurements.add_selected_step_height_measurement,
            register=self._viewer_measurement_actions,
        )
        measurements_menu.addAction(add_step_height_action)
        add_line_profile_action = self._viewer_action(
            "measure.line_profile",
            self._image_measurements.add_current_line_profile_measurement,
            register=self._viewer_measurement_actions,
        )
        measurements_menu.addAction(add_line_profile_action)
        find_periodicity_action = self._viewer_action(
            "measure.line_periodicity",
            self._image_measurements.find_periodicity_for_active_line_roi,
            register=self._viewer_measurement_actions,
        )
        measurements_menu.addAction(find_periodicity_action)
        # Niche / feature-analysis tools live under a Features submenu.
        features_menu = measurements_menu.addMenu("Features")
        feature_finder_action = self._viewer_action(
            "measure.feature_finder",
            self._on_open_feature_finder,
        )
        features_menu.addAction(feature_finder_action)
        detect_maxima_action = self._viewer_action(
            "measure.feature_maxima",
            lambda: self._open_measure_tool("feature_maxima"),
        )
        features_menu.addAction(detect_maxima_action)
        point_fft_action = self._viewer_action(
            "measure.point_fft",
            lambda: self._open_measure_tool("point_fft"),
        )
        features_menu.addAction(point_fft_action)
        pair_corr_action = self._viewer_action(
            "measure.pair_correlation",
            self._on_open_pair_correlation,
        )
        features_menu.addAction(pair_corr_action)
        feat_lat_action = self._viewer_action(
            "measure.feature_lattice",
            self._on_open_feature_lattice,
        )
        features_menu.addAction(feat_lat_action)
        measurements_menu.addSeparator()
        self._image_measurements.add_detected_point_menu_actions(
            measurements_menu,
            self,
            self._viewer_measurement_actions,
        )
        measurements_menu.addSeparator()
        lattice_grid_action = self._viewer_action(
            "measure.lattice_grid",
            self._on_open_lattice_grid,
        )
        measurements_menu.addAction(lattice_grid_action)
        clear_lattice_grid_action = self._viewer_action(
            "measure.clear_lattice_grid",
            self._on_clear_lattice_grid,
        )
        measurements_menu.addAction(clear_lattice_grid_action)
        measurements_menu.addSeparator()
        open_fft_action = self._viewer_action("fft.open", self._on_open_fft_viewer)
        measurements_menu.addAction(open_fft_action)
        periodic_filter_action = self._viewer_action(
            "fft.periodic_filter",
            self._on_periodic_filter,
        )
        measurements_menu.addAction(periodic_filter_action)
        measurements_menu.addSeparator()
        show_measure_tab_action = self._viewer_action(
            "measure.show_panel",
            lambda: self._show_sidebar_tab("measurements"),
        )
        measurements_menu.addAction(show_measure_tab_action)

        export_menu = menu_bar.addMenu("Export")
        save_png_action = self._viewer_action("export.save_png", self._on_save_png)
        export_menu.addAction(save_png_action)
        save_processed_action = self._viewer_action(
            "export.save_processed",
            self._on_save_processed_image,
        )
        export_menu.addAction(save_processed_action)
        save_provenance_action = self._viewer_action(
            "export.save_provenance",
            self._on_save_provenance,
        )
        export_menu.addAction(save_provenance_action)

        window_menu = menu_bar.addMenu("Window")
        self._viewer_window_menu = window_menu
        window_menu.aboutToShow.connect(lambda: populate_window_menu(window_menu, self))
        populate_window_menu(window_menu, self)

        # Persistent shortcut for cycling windows (Cmd+` on macOS, Ctrl+` elsewhere).
        # Registered as a QShortcut on self so it fires even when a floating tool
        # window (e.g. FFT viewer) has keyboard focus, not just the main window.
        from probeflow.gui.viewer.window_menu import cycle_viewer_windows
        from PySide6.QtGui import QShortcut
        _cycle_shortcut = QShortcut(QKeySequence("Ctrl+`"), self)
        _cycle_shortcut.setContext(Qt.WindowShortcut)
        _cycle_shortcut.activated.connect(lambda: cycle_viewer_windows(self))

        help_menu = menu_bar.addMenu("Help")
        command_finder_action = self._viewer_action(
            "viewer.command_finder",
            self._show_command_finder,
        )
        help_menu.addAction(command_finder_action)
        shortcuts_action = self._viewer_action(
            "help.shortcuts",
            self._show_image_viewer_shortcuts,
        )
        help_menu.addAction(shortcuts_action)
        help_menu.addSeparator()
        github_action = QAction("GitHub", self)
        github_action.triggered.connect(lambda: _open_url(GITHUB_URL))
        help_menu.addAction(github_action)
        about_action = self._viewer_action("help.about", self._show_viewer_about)
        help_menu.addAction(about_action)
        howto_action = self._viewer_action(
            "help.howto",
            self._show_viewer_howto,
        )
        help_menu.insertAction(github_action, howto_action)
        definitions_action = self._viewer_action(
            "help.definitions",
            self._show_viewer_definitions,
        )
        help_menu.insertAction(github_action, definitions_action)
        roi_reference_help_action = self._viewer_action(
            "help.roi_reference",
            self._show_viewer_roi_reference,
        )
        help_menu.insertAction(github_action, roi_reference_help_action)
        help_menu.insertSeparator(github_action)

        self._sync_viewer_menu_actions()

    def _restore_viewer_desktop_layout(self) -> None:
        if not hasattr(self, "_viewer_splitter"):
            return
        cfg = load_config()
        layout = cfg.get("layout", {}).get("image_viewer", {})
        restore_geometry_or_default(self, layout.get("geometry"), 0.90)
        state = layout.get("state")
        if state and hasattr(self, "_viewer_main"):
            try:
                self._viewer_main.restoreState(b64_to_qbytearray(state))
            except Exception:
                _log.warning("Could not restore saved viewer layout; using "
                             "defaults", exc_info=True)

        sizes = layout.get("splitter_sizes")
        if sizes and len(sizes) == self._viewer_splitter.count():
            self._viewer_splitter.setSizes([max(1, int(x)) for x in sizes])
        else:
            self._viewer_splitter.setSizes([900, 400])

        tab_key = layout.get("sidebar_tab")
        if tab_key and hasattr(self, "_sidebar_tabs"):
            idx = self._sidebar_tab_indices.get(tab_key)
            if idx is not None:
                self._sidebar_tabs.setCurrentIndex(idx)

        zoom_mode = layout.get("zoom_mode", "fit")
        if zoom_mode in {"fit", "one_to_one", "manual"} and hasattr(self, "_zoom_lbl"):
            self._zoom_lbl._view_scale_mode = zoom_mode

        if layout.get("sidebar_collapsed") and hasattr(self, "_sidebar_panel"):
            self._set_sidebar_collapsed(True)

        self._show_maximized_on_start = bool(layout.get("was_maximized"))
        if self._show_maximized_on_start:
            self.setWindowState(self.windowState() | Qt.WindowMaximized)

    def _save_viewer_desktop_layout_into(self, cfg: dict) -> None:
        if not hasattr(self, "_viewer_splitter"):
            return
        layout_root = cfg.setdefault("layout", {})
        layout = layout_root.setdefault("image_viewer", {})
        layout["geometry"] = qbytearray_to_b64(self.saveGeometry())
        if hasattr(self, "_viewer_main"):
            layout["state"] = qbytearray_to_b64(self._viewer_main.saveState())
        layout["was_maximized"] = self.isMaximized()
        # When collapsed the live splitter sizes are not the expanded layout, so
        # persist the remembered expanded sizes instead.
        if getattr(self, "_sidebar_collapsed", False):
            layout["splitter_sizes"] = getattr(
                self, "_expanded_sidebar_sizes", self._viewer_splitter.sizes()
            )
        else:
            layout["splitter_sizes"] = self._viewer_splitter.sizes()
        layout["sidebar_collapsed"] = bool(getattr(self, "_sidebar_collapsed", False))
        layout["zoom_mode"] = getattr(getattr(self, "_zoom_lbl", None), "_view_scale_mode", "fit")

        if hasattr(self, "_sidebar_tabs") and hasattr(self, "_sidebar_tab_indices"):
            current = self._sidebar_tabs.currentIndex()
            for key, idx in self._sidebar_tab_indices.items():
                if idx == current:
                    layout["sidebar_tab"] = key
                    break

    def _dock_panels_into_window(self) -> None:
        """Re-dock any floating tool docks (e.g. the lattice grid) back in place.

        A QDockWidget dragged out to float has no obvious way back for most
        users; this pulls every floating tool dock back into the right dock area.
        The ROI and Measurements panels are no longer docks (they live in the
        sidebar tabs), so they are unaffected.
        """
        if not hasattr(self, "_viewer_main"):
            return
        restored = 0
        for dock in self._viewer_main.findChildren(QDockWidget):
            if dock.isFloating():
                self._viewer_main.addDockWidget(Qt.RightDockWidgetArea, dock)
                dock.setFloating(False)
                restored += 1
        if hasattr(self, "_status_lbl"):
            if restored:
                self._status_lbl.setText("Tool panels docked back into the window.")
            else:
                self._status_lbl.setText("No floating tool panels to dock.")

    def _reset_viewer_window_layout(self) -> None:
        cfg = load_config()
        if isinstance(cfg.get("layout"), dict):
            cfg["layout"].pop("image_viewer", None)
        save_config(cfg)
        apply_screen_fraction_geometry(self, 0.90)
        self._dock_panels_into_window()
        self._set_sidebar_collapsed(False)
        self._viewer_splitter.setSizes([900, 400])
        self._sidebar_tabs.setCurrentIndex(self._sidebar_tab_indices.get("display", 0))
        self._zoom_lbl._view_scale_mode = "fit"
        self._zoom_lbl.fit_to_view()
        self._show_maximized_on_start = False
        self._status_lbl.setText("Image viewer layout reset.")

    def _add_combo_menu(
        self,
        parent_menu: QMenu,
        title: str,
        combo: QComboBox,
        labels: list[str],
    ) -> None:
        menu = parent_menu.addMenu(title)
        group = QActionGroup(self)
        group.setExclusive(True)
        action_map: dict[str, QAction] = {}
        for label in labels:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=label, cb=combo: cb.setCurrentText(value)
            )
            group.addAction(action)
            menu.addAction(action)
            action_map[label] = action
        self._viewer_processing_actions[f"combo:{title}"] = action_map
        combo.currentTextChanged.connect(self._sync_viewer_menu_actions)

    def _show_sidebar_tab(self, key: str) -> None:
        if not hasattr(self, "_sidebar_tabs"):
            return
        # A request to show a tab implies the panel should be visible and that
        # any in-sidebar tool view yields back to the tabs.
        if getattr(self, "_sidebar_collapsed", False):
            self._set_sidebar_collapsed(False)
        if hasattr(self, "_sidebar_stack"):
            self._sidebar_stack.setCurrentIndex(0)
        idx = self._sidebar_tab_indices.get(key)
        if idx is not None:
            self._sidebar_tabs.setCurrentIndex(idx)

    def _show_sidebar_tool(self, title: str, widget, on_close=None) -> None:
        """Host an interactive tool's controls in the sidebar column (page 1).

        Replaces the tab strip with *widget* + a Back button, so a tool like the
        lattice grid gets the whole right column instead of opening a 2nd dock.
        ``on_close`` runs when the user clicks Back (e.g. to clear an overlay).
        """
        self._close_sidebar_tool()
        if getattr(self, "_sidebar_collapsed", False):
            self._set_sidebar_collapsed(False)
        self._sidebar_tool_title.setText(title)
        self._sidebar_tool_content.layout().addWidget(widget)
        self._sidebar_tool_widget = widget
        self._sidebar_tool_on_close = on_close
        self._sidebar_stack.setCurrentIndex(1)

    def _on_sidebar_tool_back(self) -> None:
        cb = self._sidebar_tool_on_close
        if cb is not None:
            cb()
        self._close_sidebar_tool()

    def _close_sidebar_tool(self) -> None:
        widget = getattr(self, "_sidebar_tool_widget", None)
        if widget is not None:
            widget.setParent(None)
            self._sidebar_tool_widget = None
        self._sidebar_tool_on_close = None
        if hasattr(self, "_sidebar_stack"):
            self._sidebar_stack.setCurrentIndex(0)

    # ── ROI statistics (recorded to the table + shown in a modal) ──────────────
    def _roi_stats_active_and_show(self) -> None:
        self._image_measurements.add_active_roi_stats_measurement()
        self._present_roi_stats_modal()

    def _roi_stats_roi_and_show(self, roi_id: str) -> None:
        self._image_measurements.add_roi_stats_measurement(roi_id)
        self._present_roi_stats_modal()

    def _present_roi_stats_modal(self) -> None:
        """Pop the latest ROI-statistics result in a dismissible modal card."""
        results = self._measurement_table.results()
        if not results or results[-1].kind != "roi_stats":
            return
        from probeflow.gui.widgets.measurement_table import _format_details
        card = QWidget()
        card.setMinimumWidth(300)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)
        title = QLabel("ROI statistics")
        title.setStyleSheet("font-weight: 700; font-size: 14pt;")
        lay.addWidget(title)
        body = QLabel(_format_details(results[-1]))
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(body)
        self._present_modal_tool(card)

    def _set_sidebar_collapsed(self, collapsed: bool) -> None:
        """Collapse the task sidebar to a thin rail (image takes full width)."""
        if not hasattr(self, "_sidebar_panel"):
            return
        collapsed = bool(collapsed)
        if collapsed:
            sizes = self._viewer_splitter.sizes()
            total = sum(sizes)
            if len(sizes) == 2 and sizes[1] > 0:
                self._expanded_sidebar_sizes = sizes
            self._sidebar_panel.setVisible(False)
            self._sidebar_rail.setVisible(True)
            # Shrink the sidebar pane to the rail width so the image reclaims the
            # freed space (otherwise the splitter leaves an empty band).
            rail_w = self._SIDEBAR_RAIL_W
            if total > rail_w:
                self._viewer_splitter.setSizes([total - rail_w, rail_w])
        else:
            self._sidebar_rail.setVisible(False)
            self._sidebar_panel.setVisible(True)
            sizes = getattr(self, "_expanded_sidebar_sizes", None)
            if sizes:
                self._viewer_splitter.setSizes(sizes)
        self._sidebar_collapsed = collapsed

    def _toggle_sidebar_collapsed(self) -> None:
        self._set_sidebar_collapsed(not getattr(self, "_sidebar_collapsed", False))

    def _open_sidebar_from_rail(self, key: str) -> None:
        self._set_sidebar_collapsed(False)
        self._show_sidebar_tab(key)

    def _present_modal_tool(self, dialog, *, persistent: bool = False) -> "ModalOverlay":
        """Show *dialog* as a dimmed, click-outside-to-dismiss overlay.

        The dialog is reused whole (reparented as a child of a scrim overlay) instead
        of opening as a separate window that can slip behind the viewer.  Transient
        dialogs are tracked as modeless children so viewer close tears them down;
        ``persistent`` widgets are reused across opens (state preserved) and are owned
        by the viewer, so they are not tracked/destroyed here.
        """
        if not persistent:
            self._track_modeless_child(dialog)
        return ModalOverlay(self, dialog, persistent=persistent)

    def _open_advanced_tools(self) -> None:
        """Open the advanced-processing controls as a dismissible overlay."""
        self._present_modal_tool(self._advanced_widget, persistent=True)

    def _on_apply_advanced_tools(self) -> None:
        """Apply the pipeline from inside the Advanced overlay, then close it.

        The overlay scrim hides the Process tab's own Apply button, so
        committing from here must be a single action. Hiding the hosted
        widget dismisses the ModalOverlay (it watches Hide/Close on its
        content); the persistent widget survives for the next open.
        """
        self._on_apply_processing()
        self._advanced_widget.hide()

    def _show_roi_manager(self) -> None:
        self._show_sidebar_tab("roi")

    def _show_measurements(self) -> None:
        self._show_sidebar_tab("measurements")

    def _open_measure_tool(self, key: str) -> None:
        """Show the Measure tab and focus a tool's detail (controls) by key.

        Used by the Measurements → Features menu so the demoted setup tools
        (feature maxima, point/FFT) remain fully usable without a tab button.
        """
        self._show_sidebar_tab("measurements")
        if hasattr(self, "_measurement_panel"):
            self._measurement_panel.set_measurement_type(key)
