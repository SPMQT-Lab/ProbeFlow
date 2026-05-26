"""Public constructors for lattice-grid GUI tools."""

from __future__ import annotations

import numpy as np

from probeflow.analysis.lattice_grid import (
    LatticeGrid,
    RealSpaceCalibration,
    ReciprocalCalibration,
)

from .controller import LatticeGridController
from .fft_overlay import FFTLatticeOverlay
from .fft_panel import FFTLatticePanel
from .graphics_item import LatticeGridItem
from .real_space_panel import LatticeGridPanel

# ── public entry points ───────────────────────────────────────────────────────

def open_real_space_tool(
    canvas,
    scan_range_m: tuple,
    image_shape: tuple,
    parent=None,
    get_image_fn=None,
    apply_correction_fn=None,
    preview_image_fn=None,
    clear_preview_fn=None,
) -> tuple[LatticeGridItem, LatticeGridPanel]:
    """
    Create a lattice grid overlay on an ImageCanvas.

    Installs the interaction controller immediately with edit mode active.
    Returns (item, panel); caller adds the panel to the UI (e.g. as a dock).

    get_image_fn: callable() → np.ndarray | None — returns current image array
    apply_correction_fn: callable(op_name, op_params) — applies a geometric op
    preview_image_fn: callable(np.ndarray) → None — shows temporary preview in viewer
    clear_preview_fn: callable() → None — restores original display from processing
    """
    Ny, Nx = image_shape
    cx, cy = Nx / 2.0, Ny / 2.0
    size = min(Nx, Ny) * 0.15

    cal = RealSpaceCalibration.from_scan_range(scan_range_m, Nx, Ny)
    grid = LatticeGrid.make_square(cx, cy, size, space="real")

    item = LatticeGridItem(grid, Nx, Ny, cells=12)
    canvas.scene().addItem(item)

    controller = LatticeGridController(item, canvas)
    controller.install()

    panel = LatticeGridPanel(
        item, controller, cal, Nx, Ny, parent=parent,
        get_image_fn=get_image_fn,
        apply_correction_fn=apply_correction_fn,
        preview_image_fn=preview_image_fn,
        clear_preview_fn=clear_preview_fn,
    )
    controller.set_panel(panel)
    item.grid_changed.connect(panel.sync_from_model)

    return item, panel


def open_fft_tool(
    ax,
    canvas,
    qx_axis: np.ndarray,
    qy_axis: np.ndarray,
    image_shape: tuple,
    parent=None,
    on_change=None,
) -> tuple[FFTLatticeOverlay, FFTLatticePanel]:
    """
    Create a reciprocal-space lattice grid overlay on an FFT matplotlib axes.

    Returns (overlay, panel).
    """
    Ny, Nx = image_shape
    cx_px, cy_px = Nx / 2.0, Ny / 2.0
    size_px = min(Nx, Ny) * 0.12

    cal = ReciprocalCalibration(
        qx_axis=qx_axis, qy_axis=qy_axis,
        image_width=Nx, image_height=Ny,
    )
    grid = LatticeGrid.make_square(cx_px, cy_px, size_px, space="reciprocal")

    overlay = FFTLatticeOverlay(ax, canvas, qx_axis, qy_axis, Nx, Ny)
    overlay.set_grid(grid)

    panel = FFTLatticePanel(overlay, cal, Nx, Ny, parent=parent, on_change=on_change)
    return overlay, panel
