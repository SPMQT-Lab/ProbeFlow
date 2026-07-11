"""Tests for angle-measurement update and floating-panel restore."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PySide6 unavailable: {exc}")
    app = QApplication.instance()
    if app is not None:
        return app
    try:
        return QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"QApplication unavailable: {exc}")


def _angle_result(mid, deg):
    from probeflow.measurements.models import MeasurementResult
    return MeasurementResult(
        measurement_id=mid, kind="angle", source_label="s", source_path="s",
        channel=None, x_unit="°", y_unit=None, z_unit=None,
        values={"angle_deg": deg}, context={}, notes="",
    )


def test_table_update_result_replaces_in_place(qapp):
    from probeflow.gui.widgets.measurement_table import MeasurementResultsTable

    table = MeasurementResultsTable()
    try:
        mid = table.next_measurement_id()
        table.add_result(_angle_result(mid, 90.0))
        assert len(table.results()) == 1

        ok = table.update_result(_angle_result(mid, 45.0))
        assert ok is True
        assert len(table.results()) == 1  # replaced, not appended
        assert table.results()[0].values["angle_deg"] == 45.0

        # Unknown id is not updated.
        assert table.update_result(_angle_result("M9999", 10.0)) is False
        assert len(table.results()) == 1
    finally:
        table.deleteLater()
        qapp.processEvents()


def _dialog(monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES
    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self, **kw: None)
    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    return ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])


def test_angle_placement_shows_live_value_without_storing(qapp, monkeypatch):
    """Placing the angle updates the live readout only; storing is explicit."""
    from PySide6.QtCore import QPointF

    dlg = _dialog(monkeypatch)
    try:
        # Place a right-angle overlay: P1=(0,0), vertex=(10,0), P3=(10,10) -> 90°.
        dlg._on_angle_points_ready(QPointF(0, 0), QPointF(10, 0), QPointF(10, 10))
        assert dlg._measurement_table.results() == []  # nothing auto-stored
        assert dlg._measurement_panel._angle_live_lbl.text() == "90.00°"

        # Dragging a handle updates the live readout with no clicks.
        # Move P3 to (0,10): arms (-10,0) and (-10,10) -> 45°.
        dlg._angle_overlay._h3.setPos(QPointF(0, 10))
        assert dlg._measurement_panel._angle_live_lbl.text() == "45.00°"
        assert dlg._measurement_table.results() == []

        # 'Add to results' stores a snapshot; a second click stores another.
        dlg._on_update_angle_measurement()
        results = dlg._measurement_table.results()
        assert len(results) == 1
        assert results[0].values["angle_deg"] == pytest.approx(45.0, abs=1e-6)

        dlg._angle_overlay._h3.setPos(QPointF(10, 10))
        dlg._on_update_angle_measurement()
        results = dlg._measurement_table.results()
        assert len(results) == 2
        assert results[1].values["angle_deg"] == pytest.approx(90.0, abs=1e-6)

        # Clearing the overlay resets the live readout.
        dlg._clear_angle_overlay()
        assert dlg._measurement_panel._angle_live_lbl.text() == "—"
        assert len(dlg._measurement_table.results()) == 2  # stored rows survive
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_add_angle_without_overlay_is_a_noop(qapp, monkeypatch):
    dlg = _dialog(monkeypatch)
    try:
        assert dlg._angle_overlay is None
        dlg._on_update_angle_measurement()
        assert dlg._measurement_table.results() == []
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_dock_panels_into_window_redocks_floating_panel(qapp, monkeypatch):
    # The ROI/Measurements panels now live in the sidebar tabs; only transient
    # tool docks (e.g. the lattice grid) remain dockable. _dock_panels_into_window
    # should re-dock any such floating dock hosted in the viewer's QMainWindow.
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDockWidget

    dlg = _dialog(monkeypatch)
    try:
        dock = QDockWidget("Tool", dlg._viewer_main)
        dlg._viewer_main.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setFloating(True)
        assert dock.isFloating()

        dlg._dock_panels_into_window()

        assert not dock.isFloating()
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()
