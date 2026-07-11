"""Tests for stored lattice-grid layers (real space + FFT) and tooltip sizing."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")
    app = QApplication.instance()
    if app is not None:
        return app
    return QApplication([])


# ── palette ───────────────────────────────────────────────────────────────────

def test_stored_palette_cycles_and_avoids_active_blue():
    from probeflow.gui.lattice_grid.constants import STORED_GRID_PALETTE
    from probeflow.gui.lattice_grid.stored_grids import next_stored_color

    n = len(STORED_GRID_PALETTE)
    assert n >= 4
    assert "#89b4fa" not in STORED_GRID_PALETTE  # active grid stays unique
    assert next_stored_color(0) == STORED_GRID_PALETTE[0]
    assert next_stored_color(n) == STORED_GRID_PALETTE[0]  # cycles
    assert next_stored_color(1) != next_stored_color(0)


# ── real-space tool ───────────────────────────────────────────────────────────

def _real_space_tool(qapp):
    from PySide6.QtGui import QPixmap
    from probeflow.gui.image_canvas import ImageCanvas
    from probeflow.gui.lattice_grid import open_real_space_tool

    canvas = ImageCanvas()
    pm = QPixmap(100, 100)
    pm.fill()
    canvas.set_source(pm, reset_zoom=True)
    item, panel = open_real_space_tool(canvas, (10e-9, 10e-9), (100, 100))
    return canvas, item, panel


def _grid_items(canvas):
    from probeflow.gui.lattice_grid.graphics_item import LatticeGridItem

    return [it for it in canvas.scene().items() if isinstance(it, LatticeGridItem)]


def test_real_space_store_edit_remove_cleanup(qapp):
    canvas, item, panel = _real_space_tool(qapp)
    try:
        assert len(_grid_items(canvas)) == 1  # just the active grid

        # Store the current grid, then rotate the active one so they differ.
        a_first = item.grid().a_px
        panel._on_store_grid()
        assert len(panel._stored) == 1
        assert len(_grid_items(canvas)) == 2
        stored_entry = panel._stored[0]
        assert stored_entry.grid.show_handles is False
        assert stored_entry.grid.a_px == pytest.approx(a_first)

        item.set_grid(item.grid().rotate(30.0))
        a_rotated = item.grid().a_px
        assert a_rotated != pytest.approx(a_first)

        # Edit swaps: stored grid becomes active, rotated grid takes the slot.
        panel._on_edit_stored(0)
        assert item.grid().a_px == pytest.approx(a_first)
        assert panel._stored[0].grid.a_px == pytest.approx(a_rotated)
        assert panel._stored[0].color == stored_entry.color  # slot keeps colour
        assert item.grid().show_handles is True  # editor keeps its handles

        # A second store gets a different palette colour.
        panel._on_store_grid()
        assert len(panel._stored) == 2
        assert panel._stored[1].color != panel._stored[0].color
        assert len(_grid_items(canvas)) == 3

        # Remove drops the layer and its scene item.
        panel._on_remove_stored(0)
        assert len(panel._stored) == 1
        assert len(_grid_items(canvas)) == 2

        # cleanup() clears every stored layer from the scene.
        panel.cleanup()
        assert panel._stored == []
        # Only the active item remains (its removal is the viewer's job).
        assert len(_grid_items(canvas)) == 1
    finally:
        panel.deleteLater()
        canvas.deleteLater()
        qapp.processEvents()


def test_real_space_stored_item_uses_layer_color(qapp):
    canvas, item, panel = _real_space_tool(qapp)
    try:
        panel._on_store_grid()
        stored_item = panel._stored_items[0]
        assert stored_item._color is not None
        assert stored_item._color.name() == panel._stored[0].color
        # The active editable item keeps the default (no custom colour).
        assert item._color is None
    finally:
        panel.cleanup()
        panel.deleteLater()
        canvas.deleteLater()
        qapp.processEvents()


# ── FFT tool ──────────────────────────────────────────────────────────────────

def _fft_tool(qapp):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from probeflow.gui.lattice_grid import open_fft_tool

    fig = Figure(figsize=(4, 4))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    qx = np.linspace(-5.0, 5.0, 64)
    qy = np.linspace(-5.0, 5.0, 64)
    overlay, panel = open_fft_tool(ax, canvas, qx, qy, (64, 64))
    return overlay, panel


def test_fft_store_edit_remove(qapp):
    overlay, panel = _fft_tool(qapp)
    try:
        a_first = overlay.grid().a_px
        panel._on_store_grid()
        assert len(panel._stored) == 1
        # 2*(2c+1) line artists per stored layer.
        cells = panel._stored[0].cells
        assert len(overlay._stored_artists) == 2 * (2 * cells + 1)

        overlay.set_grid(overlay.grid().rotate(15.0))
        a_rotated = overlay.grid().a_px

        panel._on_edit_stored(0)
        assert overlay.grid().a_px == pytest.approx(a_first)
        assert panel._stored[0].grid.a_px == pytest.approx(a_rotated)

        panel._on_remove_stored(0)
        assert panel._stored == []
        assert overlay._stored_artists == []
    finally:
        overlay.clear()
        panel.deleteLater()
        qapp.processEvents()


def test_fft_clear_drops_stored_layers(qapp):
    overlay, panel = _fft_tool(qapp)
    try:
        panel._on_store_grid()
        assert overlay._stored_artists
        overlay.clear()
        assert overlay._stored_artists == []
        assert overlay._stored == []
    finally:
        panel.deleteLater()
        qapp.processEvents()


# ── stored-grid list widget ───────────────────────────────────────────────────

def test_stored_grid_list_rows_and_signals(qapp):
    from probeflow.analysis.lattice_grid import LatticeGrid
    from probeflow.gui.lattice_grid.stored_grids import StoredGrid, StoredGridList

    widget = StoredGridList()
    try:
        grid = LatticeGrid.make_square(50, 50, 10)
        entries = [
            StoredGrid(grid=grid, cells=12, line_width_px=1.5,
                       color="#f9e2af", summary="one"),
            StoredGrid(grid=grid, cells=12, line_width_px=1.5,
                       color="#a6e3a1", summary="two"),
        ]
        widget.set_entries(entries)
        assert len(widget._row_widgets) == 2

        edits, removes = [], []
        widget.edit_requested.connect(edits.append)
        widget.remove_requested.connect(removes.append)
        # Buttons per row: [Edit, ✕] — click the second row's buttons.
        from PySide6.QtWidgets import QPushButton
        row_buttons = widget._row_widgets[1].findChildren(QPushButton)
        assert len(row_buttons) == 2
        row_buttons[0].click()
        row_buttons[1].click()
        assert edits == [1]
        assert removes == [1]

        widget.set_entries([])
        assert widget._row_widgets == []
    finally:
        widget.deleteLater()
        qapp.processEvents()


# ── tooltip sizing ────────────────────────────────────────────────────────────

def test_short_tooltip_keeps_natural_size(qapp):
    from probeflow.gui.tooltips import _wrap

    out = _wrap("Unit for the reference lattice spacing.")
    assert "table" not in out  # no fixed-width box around one short sentence
    assert "Unit for the reference lattice spacing." in out


def test_long_tooltip_is_width_capped(qapp):
    from probeflow.gui.tooltips import _WRAP_PX, _wrap

    long_text = (
        "This is a deliberately long tooltip explaining a subtle behaviour in "
        "enough detail that it would stretch across the whole screen if it "
        "were rendered as a single unwrapped plain-text line."
    )
    out = _wrap(long_text)
    assert f'table width="{_WRAP_PX}"' in out


def test_rich_text_tooltip_stays_capped(qapp):
    from probeflow.gui.tooltips import _WRAP_PX, _wrap

    out = _wrap("<b>g(r)</b> formula")
    assert f'table width="{_WRAP_PX}"' in out


def test_multiline_short_tooltip_unboxed_with_breaks(qapp):
    from probeflow.gui.tooltips import _wrap

    out = _wrap("Line one\nLine two")
    assert "table" not in out
    assert "<br>" in out
