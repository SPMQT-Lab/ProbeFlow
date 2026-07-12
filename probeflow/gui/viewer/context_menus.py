"""Context-menu builders for the image viewer canvas."""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from probeflow.core import AREA_ROI_KINDS


def build_blank_image_context_menu(viewer) -> QMenu:
    """Build the context menu shown when right-clicking blank image space."""
    menu = QMenu(viewer)

    act_fit = menu.addAction("Fit image to view")
    act_fit.triggered.connect(viewer._zoom_lbl.fit_to_view)

    act_one_to_one = menu.addAction("View at 1:1")
    act_one_to_one.triggered.connect(viewer._zoom_lbl.reset_zoom)

    menu.addSeparator()

    act_auto = menu.addAction("Auto contrast")
    act_auto.triggered.connect(viewer._on_auto_clip)

    act_reset_display = menu.addAction("Reset display range")
    act_reset_display.triggered.connect(viewer._on_reset_display)

    menu.addSeparator()

    act_fft = menu.addAction("Open FFT viewer")
    act_fft.triggered.connect(viewer._on_open_fft_viewer)

    act_clear = menu.addAction("Clear ROIs and overlays")
    act_clear.triggered.connect(viewer._clear_all_image_marks)

    # Quick-selection actions (only when a selection is active).
    if getattr(viewer, "_has_quick_selection", None) and viewer._has_quick_selection():
        menu.addSeparator()
        act_promote = menu.addAction("Promote selection to ROI")
        act_promote.triggered.connect(viewer._promote_selection_to_roi)
        act_drop = menu.addAction("Clear selection")
        act_drop.triggered.connect(viewer._clear_quick_selection)

    menu.addSeparator()

    act_save = menu.addAction("Save PNG copy...")
    act_save.triggered.connect(viewer._on_save_png)

    if hasattr(viewer, "_on_map_spectra_here"):
        act_spec = menu.addAction("Map spectroscopy markers to this image...")
        act_spec.triggered.connect(viewer._on_map_spectra_here)

    menu.addSeparator()
    _add_send_to_tool_actions(menu, viewer)
    _add_transform_menu(menu, viewer)
    _add_line_profile_export_action(menu, viewer)

    return menu


def build_roi_context_menu(viewer, roi_id: str) -> QMenu:
    """Build the context menu shown when right-clicking an ROI."""
    roi_set = viewer._image_roi_set
    roi = roi_set.get(roi_id) if roi_set else None
    menu = QMenu(viewer)

    if roi is None:
        return menu

    is_area = roi.kind in AREA_ROI_KINDS
    is_line = roi.kind == "line"
    is_point = roi.kind == "point"

    title = menu.addAction(f"{roi.name} ({roi.kind})")
    title.setEnabled(False)
    menu.addSeparator()

    act_active = menu.addAction("Set active")
    act_active.triggered.connect(lambda: viewer._set_active_image_roi(roi_id))

    act_rename = menu.addAction("Rename...")
    act_rename.triggered.connect(lambda: viewer._rename_image_roi(roi_id))

    act_copy = menu.addAction("Copy ROI")
    act_copy.triggered.connect(lambda: viewer._on_canvas_roi_copy(roi_id))

    act_duplicate = menu.addAction("Duplicate ROI")
    act_duplicate.triggered.connect(lambda: viewer._duplicate_image_roi(roi_id))

    act_delete = menu.addAction("Delete ROI")
    act_delete.triggered.connect(lambda: viewer._delete_image_roi(roi_id))

    menu.addSeparator()

    if is_area:
        _add_area_roi_actions(menu, viewer, roi_id)
    elif is_line:
        _add_line_roi_actions(menu, viewer, roi_id)
    elif is_point:
        _add_point_roi_actions(menu, viewer, roi_id)

    return menu


def _add_area_roi_actions(menu: QMenu, viewer, roi_id: str) -> None:
    act_scope = menu.addAction("Set as ROI filter scope")
    act_scope.triggered.connect(lambda: viewer._set_roi_filter_scope(roi_id))

    act_invert = menu.addAction("Invert area")
    act_invert.triggered.connect(lambda: viewer._invert_image_roi(roi_id))

    menu.addSeparator()

    act_stm_bg = menu.addAction("STM background fit from ROI...")
    act_stm_bg.triggered.connect(lambda: viewer._open_stm_background_for_roi(roi_id))

    act_hist = menu.addAction("Histogram of this region")
    act_hist.triggered.connect(lambda: viewer._on_roi_histogram(roi_id))

    act_fft = menu.addAction("FFT this region")
    act_fft.triggered.connect(lambda: viewer._on_roi_fft(roi_id))

    act_stats = menu.addAction("Add ROI statistics to measurements")
    act_stats.triggered.connect(
        lambda: viewer._image_measurements.add_roi_stats_measurement(roi_id)
    )

    act_maxima = menu.addAction("Detect maxima in this region")
    act_maxima.triggered.connect(
        lambda: viewer._image_measurements.detect_feature_maxima_for_roi(roi_id)
    )

    menu.addSeparator()

    act_repair = menu.addAction("Remove spots (interpolate this region)")
    act_repair.setToolTip(
        "Replace the data inside this ROI with a smooth surface interpolated "
        "from the surroundings — for tip changes, dirt, and glitches."
    )
    act_repair.triggered.connect(lambda: viewer._commit_repair_under_roi(roi_id))

    act_crop = menu.addAction("Crop image to this region")
    act_crop.triggered.connect(lambda: viewer._on_crop_to_roi(roi_id))


def _add_line_roi_actions(menu: QMenu, viewer, roi_id: str) -> None:
    act_profile = menu.addAction("Show line profile")
    act_profile.triggered.connect(lambda: viewer._on_roi_line_profile(roi_id))

    act_add_profile = menu.addAction("Add line profile to measurements")
    act_add_profile.triggered.connect(
        lambda: viewer._image_measurements.add_line_profile_measurement_for_roi(roi_id)
    )

    act_period = menu.addAction("Estimate periodicity")
    act_period.triggered.connect(lambda: viewer._find_periodicity_for_roi(roi_id))

    act_width = menu.addAction("Set line width...")
    act_width.triggered.connect(lambda: viewer._set_line_roi_width_dialog(roi_id))


def _add_point_roi_actions(menu: QMenu, viewer, roi_id: str) -> None:
    act_copy = menu.addAction("Copy point coordinates")
    act_copy.triggered.connect(lambda: viewer._copy_point_roi_coordinates(roi_id))


def _add_send_to_tool_actions(menu: QMenu, viewer) -> None:
    a_tv = QAction("Send to TV Denoising", viewer)
    a_tv.setToolTip("Send processed image to TV Denoising tab")
    a_tv.triggered.connect(viewer._on_send_to_tv)
    menu.addAction(a_tv)


def _add_transform_menu(menu: QMenu, viewer) -> None:
    """Build the Transform submenu using VIEWER_COMMANDS for labels/shortcuts."""
    from probeflow.gui.viewer.shortcuts import viewer_command as _vc
    from PySide6.QtGui import QKeySequence

    def _cmd_action(cmd_id: str) -> QAction:
        """Create a QAction labelled from VIEWER_COMMANDS without registering it."""
        cmd = _vc(cmd_id)
        act = QAction(cmd.label, viewer)
        if cmd.shortcuts:
            act.setShortcuts([QKeySequence(s) for s in cmd.shortcuts])
        if cmd.status_tip:
            act.setStatusTip(cmd.status_tip)
        return act

    transform_menu = QMenu("Transform", viewer)

    for cmd_id, op_name in [
        ("image.flip_h", "flip_horizontal"),
        ("image.flip_v", "flip_vertical"),
    ]:
        act = _cmd_action(cmd_id)
        act.triggered.connect(
            lambda _checked=False, op=op_name: viewer._on_geometric_op(op)
        )
        transform_menu.addAction(act)

    transform_menu.addSeparator()

    for cmd_id, op_name in [
        ("image.rotate_90_cw",  "rotate_90_cw"),
        ("image.rotate_180",    "rotate_180"),
        ("image.rotate_270_cw", "rotate_270_cw"),
    ]:
        act = _cmd_action(cmd_id)
        act.triggered.connect(
            lambda _checked=False, op=op_name: viewer._on_geometric_op(op)
        )
        transform_menu.addAction(act)

    arb_act = _cmd_action("image.rotate_arbitrary")
    arb_act.triggered.connect(viewer._on_rotate_arbitrary)
    transform_menu.addAction(arb_act)

    transform_menu.addSeparator()
    shear_act = _cmd_action("image.shear")
    shear_act.triggered.connect(viewer._on_shear)
    transform_menu.addAction(shear_act)

    transform_menu.addSeparator()
    crop_act = _cmd_action("image.crop_selection")
    crop_act.triggered.connect(viewer._on_crop_to_selection)
    transform_menu.addAction(crop_act)

    menu.addMenu(transform_menu)


def _add_line_profile_export_action(menu: QMenu, viewer) -> None:
    prof = viewer._line_profile_panel.profile_data()
    if prof is None:
        return
    menu.addSeparator()
    a_csv = QAction("Export line profile as CSV...", viewer)
    a_csv.triggered.connect(viewer._on_export_line_profile_csv)
    menu.addAction(a_csv)
