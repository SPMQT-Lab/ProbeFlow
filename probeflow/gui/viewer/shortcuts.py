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


VIEWER_COMMANDS: tuple[ViewerCommand, ...] = (
    ViewerCommand("panel.view", "Histogram / Contrast", "View", ("Ctrl+1",), "Show the View sidebar tab."),
    ViewerCommand("panel.process", "Processing panel", "View", ("Ctrl+2",), "Show the Process sidebar tab."),
    ViewerCommand("panel.roi", "ROI panel", "View", ("Ctrl+3",), "Show the ROI sidebar tab."),
    ViewerCommand("panel.measure", "Measurements panel", "View", ("Ctrl+4",), "Show the Measure sidebar tab."),
    ViewerCommand("panel.export", "Export panel", "View", ("Ctrl+5",), "Show the Export sidebar tab."),
    ViewerCommand("dock.roi_manager", "ROI Manager", "View", ("Ctrl+Shift+R",), "Show or hide the ROI Manager dock."),
    ViewerCommand("dock.measurements", "Measurements", "View", ("Ctrl+Shift+M",), "Show or hide the Measurements dock."),
    ViewerCommand("view.fit", "Fit image to window", "View", ("Ctrl+0",), "Fit the image to the visible canvas."),
    ViewerCommand("view.one_to_one", "View at 1:1", "View", ("Ctrl+Shift+0",), "View the image at native raster size."),
    ViewerCommand("view.auto_contrast", "Auto contrast", "View", ("Ctrl+Shift+A",), "Autoscale the display range."),
    ViewerCommand("processing.apply", "Apply processing", "Processing", ("Ctrl+Return",), "Apply queued processing settings."),
    ViewerCommand("processing.reset", "Reset processing", "Processing", ("Ctrl+Shift+Backspace",), "Reset processing to the on-disk data."),
    ViewerCommand("processing.plane_background", "Plane/background subtraction...", "Processing", ("Ctrl+Shift+B",), "Run simple plane/background subtraction."),
    ViewerCommand("processing.stm_background", "STM scan-line background...", "Processing", ("Ctrl+Alt+B",), "Open STM scan-line background subtraction."),
    ViewerCommand("processing.bad_lines", "Bad scan-line correction...", "Processing", ("Ctrl+Shift+L",), "Open bad scan-line correction."),
    ViewerCommand("processing.undo", "Undo", "Processing", ("Ctrl+Z",), "Restore the previous processing state.", "undo"),
    ViewerCommand("processing.redo", "Redo", "Processing", ("Ctrl+Y", "Ctrl+Shift+Z"), "Reapply a processing state that was undone.", "redo"),
    ViewerCommand("fft.open", "Open FFT viewer...", "FFT", ("Ctrl+Shift+F",), "Open the FFT viewer."),
    ViewerCommand("fft.periodic_filter", "Periodic filter...", "FFT", ("Ctrl+Alt+F",), "Open periodic FFT filtering."),
    ViewerCommand("measure.distance", "Ruler / distance...", "Measurements", ("Ctrl+Alt+D",), "Measure distance from the active line ROI."),
    ViewerCommand("measure.angle", "Angle measurement...", "Measurements", ("Ctrl+Alt+A",), "Measure an angle between selected line ROIs."),
    ViewerCommand("measure.roi_stats", "ROI statistics...", "Measurements", ("Ctrl+Alt+S",), "Compute statistics for the active area ROI."),
    ViewerCommand("measure.line_profile", "Add current line profile", "Measurements", ("Ctrl+Shift+P",), "Add a line-profile measurement.", "line_profile"),
    ViewerCommand("measure.line_periodicity", "Find periodicity from line profile...", "Measurements", ("Ctrl+Alt+P",), "Estimate periodicity from the active line ROI.", "line_periodicity"),
    ViewerCommand("export.save_png", "Save PNG copy", "Export", ("Ctrl+S",), "Save a PNG copy of the current view."),
    ViewerCommand("export.save_processed", "Save processed image", "Export", ("Ctrl+Shift+S",), "Save the processed image data."),
)

VIEWER_COMMAND_BY_ID = {command.command_id: command for command in VIEWER_COMMANDS}


def viewer_command(command_id: str) -> ViewerCommand:
    return VIEWER_COMMAND_BY_ID[command_id]
