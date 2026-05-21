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
    ViewerCommand("panel.roi", "ROI panel", "View", ("Ctrl+3",), "Show the ROI sidebar tab.", aliases=("roi", "selection")),
    ViewerCommand("panel.measure", "Measurements panel", "View", ("Ctrl+4",), "Show the Measure sidebar tab.", aliases=("measure", "measurement")),
    ViewerCommand("panel.export", "Export panel", "View", ("Ctrl+5",), "Show the Export sidebar tab.", aliases=("save", "write")),
    ViewerCommand("dock.roi_manager", "ROI Manager", "View", ("Ctrl+Shift+R",), "Show or hide the ROI Manager dock.", aliases=("roi list", "regions")),
    ViewerCommand("dock.measurements", "Measurements", "View", ("Ctrl+Shift+M",), "Show or hide the Measurements dock.", aliases=("measurement table", "results")),
    ViewerCommand("view.fit", "Fit image to window", "View", ("Ctrl+0",), "Fit the image to the visible canvas.", aliases=("zoom", "fit")),
    ViewerCommand("view.one_to_one", "View at 1:1", "View", ("Ctrl+Shift+0",), "View the image at native raster size.", aliases=("native", "actual size")),
    ViewerCommand("view.auto_contrast", "Auto contrast", "View", ("Ctrl+Shift+A",), "Autoscale the display range.", aliases=("clip", "autoscale")),
    ViewerCommand("view.reset_contrast", "Reset contrast", "View", status_tip="Reset the display range controls.", aliases=("display", "clip")),
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
    ViewerCommand("fft.open", "Open FFT viewer...", "FFT", ("Ctrl+Shift+F",), "Open the FFT viewer.", aliases=("spectrum", "fourier")),
    ViewerCommand("fft.periodic_filter", "Periodic filter...", "FFT", ("Ctrl+Alt+F",), "Open periodic FFT filtering.", aliases=("notch", "lattice", "periodic")),
    ViewerCommand("measure.distance", "Ruler / distance...", "Measurements", ("Ctrl+Alt+D",), "Measure distance from the active line ROI.", "distance", aliases=("ruler", "length")),
    ViewerCommand("measure.angle", "Angle measurement...", "Measurements", ("Ctrl+Alt+A",), "Measure an angle between selected line ROIs.", aliases=("angle",)),
    ViewerCommand("measure.roi_stats", "ROI statistics...", "Measurements", ("Ctrl+Alt+S",), "Compute statistics for the active area ROI.", "roi_stats", aliases=("statistics", "area")),
    ViewerCommand("measure.add_roi_stats", "Add active ROI statistics", "Measurements", status_tip="Add statistics for the active area ROI to the measurements table.", enabled_state_key="roi_stats", aliases=("statistics", "area")),
    ViewerCommand("measure.step_height", "Add step height from selected ROIs", "Measurements", status_tip="Measure step height from selected ROIs.", enabled_state_key="step_height", aliases=("terrace", "height")),
    ViewerCommand("measure.line_profile", "Add current line profile", "Measurements", ("Ctrl+Shift+P",), "Add a line-profile measurement.", "line_profile", aliases=("profile", "line")),
    ViewerCommand("measure.line_periodicity", "Find periodicity from line profile...", "Measurements", ("Ctrl+Alt+P",), "Estimate periodicity from the active line ROI.", "line_periodicity", aliases=("period", "spacing", "profile")),
    ViewerCommand("measure.feature_maxima", "Detect maxima in active ROI", "Measurements", status_tip="Detect feature maxima in the active area ROI.", enabled_state_key="feature_maxima", aliases=("maxima", "peaks", "features")),
    ViewerCommand("measure.feature_finder", "Feature finder...", "Measurements", status_tip="Open the feature finder tool.", aliases=("maxima", "minima", "peaks")),
    ViewerCommand("measure.pair_correlation", "Pair correlation...", "Measurements", status_tip="Open pair correlation analysis.", aliases=("g(r)", "rdf", "radial distribution")),
    ViewerCommand("measure.feature_lattice", "Feature-to-lattice comparison...", "Measurements", status_tip="Compare detected features to a lattice.", aliases=("lattice", "features")),
    ViewerCommand("measure.lattice_grid", "Lattice/Grid tool...", "Measurements", status_tip="Open the lattice/grid analysis tool.", aliases=("grid", "lattice")),
    ViewerCommand("measure.clear_lattice_grid", "Clear lattice grid", "Measurements", status_tip="Remove the active lattice grid overlay from the image.", enabled_state_key="lattice_grid", aliases=("grid", "lattice", "overlay", "clear")),
    ViewerCommand("measure.show_table", "Show measurements", "Measurements", status_tip="Show the measurements dock.", aliases=("measurement table", "results")),
    ViewerCommand("measure.show_panel", "Measurement table (Measure tab)", "Measurements", status_tip="Show the Measure sidebar tab.", aliases=("measure tab", "results")),
    ViewerCommand("export.save_png", "Save PNG copy", "Export", ("Ctrl+S",), "Save a PNG copy of the current view.", aliases=("image", "snapshot")),
    ViewerCommand("export.save_processed", "Save processed image", "Export", ("Ctrl+Shift+S",), "Save the processed image data.", aliases=("data", "processed")),
    ViewerCommand("export.save_provenance", "Save provenance", "Export", status_tip="Save processing provenance metadata.", aliases=("metadata", "history")),
    ViewerCommand("help.shortcuts", "Image viewer shortcuts", "Help", status_tip="Show image-viewer shortcut help.", aliases=("keyboard", "keys")),
    ViewerCommand("help.definitions", "Definitions", "Help", status_tip="Show processing definitions and equations.", aliases=("reference", "math", "algorithms")),
    ViewerCommand("help.about", "About ProbeFlow", "Help", status_tip="Show ProbeFlow version and project information.", aliases=("version",)),
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
