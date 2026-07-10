"""Image-viewer command metadata and keyboard shortcuts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewerCommand:
    command_id: str
    label: str
    menu_group: str
    shortcuts: tuple[str, ...] = ()
    status_tip: str = ""
    enabled_state_key: str | None = None
    finder_visible: bool = True
    aliases: tuple[str, ...] = ()


VIEWER_COMMANDS: tuple[ViewerCommand, ...] = (
    ViewerCommand(
        "viewer.command_finder", "Command finder...", "Help", ("Ctrl+K",),
        "Search and run image-viewer commands.", finder_visible=False,
        aliases=("palette", "finder", "imagej"),
    ),
    ViewerCommand("panel.view", "Histogram / Contrast", "View", ("Ctrl+1",), "Show the View sidebar tab.", aliases=("display", "contrast", "histogram")),
    ViewerCommand("panel.process", "Processing panel", "View", ("Ctrl+2",), "Show the Process sidebar tab.", aliases=("process", "processing")),
    ViewerCommand("panel.measure", "Measurements panel", "View", ("Ctrl+3",), "Show the Measure sidebar tab.", aliases=("measure", "measurement")),
    ViewerCommand("panel.roi", "ROI / Mask panel", "View", ("Ctrl+4",), "Show the ROI/Mask sidebar tab.", aliases=("roi", "mask", "selection")),
    ViewerCommand("panel.export", "Export panel", "View", ("Ctrl+5",), "Show the Export sidebar tab.", aliases=("save", "write")),
    ViewerCommand("view.fit", "Fit image to window", "View", ("Ctrl+0",), "Fit the image to the visible canvas.", aliases=("zoom", "fit")),
    ViewerCommand("view.one_to_one", "View at 1:1", "View", ("Ctrl+Shift+0",), "View the image at native raster size.", aliases=("native", "actual size")),
    ViewerCommand("view.auto_contrast", "Auto contrast", "View", ("Ctrl+Shift+A",), "Autoscale the display range.", aliases=("clip", "autoscale")),
    ViewerCommand("view.reset_contrast", "Reset contrast", "View", status_tip="Reset the display range controls.", aliases=("display", "clip")),
    ViewerCommand("view.dock_panels", "Dock tool windows", "View", status_tip="Re-dock any floating tool window (e.g. the lattice grid) back into the viewer.", aliases=("float", "restore", "panels", "redock")),
    ViewerCommand("view.reset_layout", "Reset viewer layout", "View", status_tip="Restore the default viewer window layout.", aliases=("desktop", "panels")),
    ViewerCommand("processing.apply", "Apply processing", "Processing", ("Ctrl+Return",), "Apply queued processing settings.", aliases=("run", "process")),
    ViewerCommand("processing.reset", "Reset processing", "Processing", ("Ctrl+Shift+Backspace",), "Reset processing to the on-disk data.", aliases=("clear processing", "raw")),
    ViewerCommand("processing.plane_background", "Plane/background subtraction...", "Processing", ("Ctrl+Shift+B",), "Run simple plane/background subtraction.", aliases=("plane", "background", "level")),
    ViewerCommand("processing.stm_background", "STM scan-line background...", "Processing", ("Ctrl+Alt+B",), "Open STM scan-line background subtraction.", aliases=("stm", "scanline", "line background")),
    ViewerCommand("processing.bad_lines", "Bad scan-line correction...", "Processing", ("Ctrl+Shift+L",), "Open bad scan-line correction.", aliases=("bad lines", "stripes", "line correction")),
    ViewerCommand("processing.image_operations", "Image Operations...", "Processing", status_tip="Open image arithmetic operations.", aliases=("arithmetic", "calculator", "image math", "add", "subtract", "checkerboard", "noise", "speckle", "test pattern")),
    ViewerCommand("processing.zero_plane", "Zero plane", "Processing", status_tip="Set a zero reference plane from an ROI.", enabled_state_key="zero_plane", aliases=("zero", "reference")),
    ViewerCommand("processing.clear_zero", "Clear zero plane", "Processing", status_tip="Clear the current zero reference plane.", aliases=("zero", "reference")),
    ViewerCommand("processing.undo", "Undo", "Processing", ("Ctrl+Z",), "Restore the previous processing state.", "undo", aliases=("history",)),
    ViewerCommand("processing.redo", "Redo", "Processing", ("Ctrl+Y", "Ctrl+Shift+Z"), "Reapply a processing state that was undone.", "redo", aliases=("history",)),
    ViewerCommand("fft.open", "FFT viewer...", "Measurements", ("Ctrl+Shift+F",), "Open the FFT viewer: inspect the power spectrum, fit reciprocal Bragg peaks, and preview lattice undistortion.", aliases=("fft viewer", "viewer", "spectrum", "fourier", "reciprocal", "bragg", "lattice correction")),
    ViewerCommand("fft.periodic_filter", "Periodic notch filter...", "Measurements", ("Ctrl+Alt+F",), "Suppress selected FFT peaks; this is separate from lattice correction.", aliases=("notch", "lattice", "periodic")),
    ViewerCommand(
        "image.info", "Show image info...", "Image", ("Ctrl+I",),
        "Show acquisition metadata and processing history for this image.",
        aliases=("metadata", "header", "acquisition", "info", "properties"),
    ),
    ViewerCommand(
        "image.colormap", "Change colormap...", "Image",
        status_tip="Open the colormap selector.",
        aliases=("lut", "palette", "colormap", "color"),
    ),
    ViewerCommand(
        "image.threshold", "Threshold...", "Image", ("Ctrl+T",),
        "Apply a lower/upper value threshold (clip or binarize height data).",
        aliases=("clip values", "binarize", "mask values", "threshold"),
    ),
    ViewerCommand(
        "image.scale", "Scale...", "Image",
        status_tip="Resample the image to new pixel dimensions.",
        aliases=("resize", "resample", "zoom pixels"),
    ),
    ViewerCommand(
        "image.flip_h", "Flip Horizontal", "Image",
        status_tip="Mirror the image about the vertical axis.",
        aliases=("mirror", "flip x", "flip horizontal"),
    ),
    ViewerCommand(
        "image.flip_v", "Flip Vertical", "Image",
        status_tip="Mirror the image about the horizontal axis.",
        aliases=("flip y", "mirror vertical", "flip vertical"),
    ),
    ViewerCommand(
        "image.rotate_90_cw", "Rotate 90° CW", "Image",
        status_tip="Rotate the image 90° clockwise.",
        aliases=("rotate", "cw", "rotate 90"),
    ),
    ViewerCommand(
        "image.rotate_180", "Rotate 180°", "Image",
        status_tip="Rotate the image 180°.",
        aliases=("rotate", "rotate 180"),
    ),
    ViewerCommand(
        "image.rotate_270_cw", "Rotate 270° CW", "Image",
        status_tip="Rotate the image 270° clockwise (= 90° CCW).",
        aliases=("rotate", "ccw", "rotate 270"),
    ),
    ViewerCommand(
        "image.rotate_arbitrary", "Rotate Arbitrary...", "Image",
        status_tip="Rotate the image by an arbitrary angle.",
        aliases=("rotate", "arbitrary angle", "rotate angle"),
    ),
    ViewerCommand(
        "image.shear", "Shear...", "Image",
        status_tip="Apply a 2-component shear correction to the image.",
        aliases=("shear x", "shear y", "distort", "skew"),
    ),
    ViewerCommand(
        "image.type_8bit", "8-bit (256 levels)", "Image",
        status_tip="Quantize image to 8-bit precision (256 levels, same physical range).",
        aliases=("8 bit", "8bit", "quantize 8", "reduce precision"),
    ),
    ViewerCommand(
        "image.type_16bit", "16-bit (65 536 levels)", "Image",
        status_tip="Quantize image to 16-bit precision (65 536 levels, same physical range).",
        aliases=("16 bit", "16bit", "quantize 16"),
    ),
    ViewerCommand("measure.distance", "Ruler / distance...", "Measurements", ("Ctrl+Alt+D",), "Measure distance from the active line ROI.", "distance", aliases=("ruler", "length")),
    ViewerCommand("measure.angle", "Angle measurement...", "Measurements", ("Ctrl+Alt+A",), "Measure an angle between selected line ROIs.", aliases=("angle",)),
    ViewerCommand("measure.update_angle", "Update angle measurement", "Measurements", status_tip="Rewrite the current angle measurement with the adjusted overlay value.", aliases=("angle", "refresh", "update")),
    ViewerCommand("measure.roi_stats", "ROI statistics...", "Measurements", ("Ctrl+Alt+S",), "Compute statistics for the active area ROI.", "roi_stats", aliases=("statistics", "area")),
    ViewerCommand("measure.add_roi_stats", "Add active ROI statistics", "Measurements", status_tip="Add statistics for the active area ROI to the measurements table.", enabled_state_key="roi_stats", aliases=("statistics", "area")),
    ViewerCommand("measure.step_height", "Add step height from selected ROIs", "Measurements", status_tip="Measure step height from selected ROIs.", enabled_state_key="step_height", aliases=("terrace", "height")),
    ViewerCommand("measure.line_profile", "Add current line profile", "Measurements", ("Ctrl+Shift+P",), "Add a line-profile measurement.", "line_profile", aliases=("profile", "line")),
    ViewerCommand("measure.line_periodicity", "Find spacing from line profile...", "Measurements", ("Ctrl+Alt+P",), "Estimate spacing from a line ROI and optionally save it as a known structure.", "line_periodicity", aliases=("period", "spacing", "profile")),
    ViewerCommand("measure.feature_maxima", "Feature maxima...", "Measurements", status_tip="Detect local maxima in the active area ROI (opens the controls).", aliases=("maxima", "peaks", "features")),
    ViewerCommand("measure.point_fft", "Point mask / FFT...", "Measurements", status_tip="Build a point mask from detected features and inspect its FFT.", aliases=("point mask", "fft", "mask fft", "selective fft")),
    ViewerCommand("measure.pair_correlation", "Pair correlation...", "Measurements", status_tip="Open pair correlation analysis.", aliases=("g(r)", "rdf", "radial distribution")),
    ViewerCommand("measure.lattice_grid", "Real-space lattice correction...", "Measurements", status_tip="Fit a real-space lattice grid to a known structure and preview undistortion.", aliases=("grid", "lattice")),
    ViewerCommand("measure.clear_lattice_grid", "Clear lattice grid", "Measurements", status_tip="Remove the active lattice grid overlay from the image.", enabled_state_key="lattice_grid", aliases=("grid", "lattice", "overlay", "clear")),
    ViewerCommand("measure.show_panel", "Show measurements table", "Measurements", status_tip="Show the Measure sidebar tab (results table).", aliases=("measure tab", "results", "measurement table")),
    ViewerCommand("export.save_png", "Save PNG copy", "Export", ("Ctrl+S",), "Save a PNG copy of the current view.", aliases=("image", "snapshot")),
    ViewerCommand("export.save_processed", "Save processed image", "Export", ("Ctrl+Shift+S",), "Save the processed image data.", aliases=("data", "processed")),
    ViewerCommand("export.save_provenance", "Save provenance", "Export", status_tip="Save processing provenance metadata.", aliases=("metadata", "history")),
    ViewerCommand("help.shortcuts", "Image viewer shortcuts", "Help", status_tip="Show image-viewer shortcut help.", aliases=("keyboard", "keys")),
    ViewerCommand("help.howto", "How-to guides", "Help", status_tip="Show step-by-step how-to walkthroughs for common tasks.", aliases=("tutorial", "guide", "walkthrough", "getting started")),
    ViewerCommand("help.definitions", "Definitions", "Help", status_tip="Show processing definitions and equations.", aliases=("reference", "math", "algorithms")),
    ViewerCommand(
        "help.measurements", "Measurements Reference", "Help",
        status_tip="Show what each measurement computes (distance, step height, roughness, …).",
        aliases=("measure", "measurement", "stats", "step height", "roughness"),
    ),
    ViewerCommand(
        "help.roi_reference", "ROI Reference", "Help",
        status_tip="Show ROI actions, selection rules, and tool interactions.",
        aliases=("roi", "mask", "active roi", "selection"),
    ),
    ViewerCommand("help.about", "About ProbeFlow", "Help", status_tip="Show ProbeFlow version and project information.", aliases=("version",)),
    ViewerCommand(
        "window.cycle_windows",
        "Cycle Windows",
        "Window",
        ("Ctrl+`",),
        "Focus the next tool window owned by this viewer.",
        aliases=("windows", "next window", "cycle"),
    ),
)

VIEWER_COMMAND_BY_ID = {command.command_id: command for command in VIEWER_COMMANDS}


def viewer_command(command_id: str) -> ViewerCommand:
    return VIEWER_COMMAND_BY_ID[command_id]


def viewer_finder_commands() -> tuple[ViewerCommand, ...]:
    """Return commands that should appear in the image-viewer command finder."""
    return tuple(command for command in VIEWER_COMMANDS if command.finder_visible)


def mac_shortcut_text(shortcut: str) -> str:
    """Return a stable macOS-style rendering for a portable Qt shortcut string."""
    pieces = [piece.strip() for piece in str(shortcut).split("+") if piece.strip()]
    if not pieces:
        return ""
    key = pieces[-1]
    modifiers = {piece.lower() for piece in pieces[:-1]}
    rendered = ""
    if "shift" in modifiers:
        rendered += "⇧"
    if "ctrl" in modifiers or "control" in modifiers:
        rendered += "⌘"
    if "alt" in modifiers or "option" in modifiers:
        rendered += "⌥"
    if "meta" in modifiers:
        rendered += "⌃"
    return f"{rendered}{key}"


def dual_platform_shortcut_text(shortcut: str) -> str:
    """Return shortcut text showing macOS and Windows/Linux forms."""
    portable = str(shortcut).strip()
    if not portable:
        return ""
    mac = mac_shortcut_text(portable)
    if mac == portable:
        return portable
    return f"{mac} / {portable}"


def display_shortcuts_for_all_platforms(shortcuts: tuple[str, ...]) -> str:
    """Return a help-friendly shortcut string for all desktop platforms."""
    return "; ".join(
        text for text in (dual_platform_shortcut_text(shortcut) for shortcut in shortcuts)
        if text
    )
