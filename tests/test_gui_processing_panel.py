"""Regression tests for the Browse/Viewer processing control ownership."""

from __future__ import annotations

import os
from pathlib import Path

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
    try:
        return QApplication([])
    except Exception as exc:
        pytest.skip(f"QApplication unavailable: {exc}")


def test_browse_quick_panel_emits_only_thumbnail_corrections(qapp):
    from probeflow.gui import ProcessingControlPanel

    panel = ProcessingControlPanel("browse_quick")
    panel.set_state({
        "align_rows": "median",
        "bg_order": 4,
        "stm_line_bg": "step_tolerant",
        "facet_level": True,
        "smooth_sigma": 3,
        "highpass_sigma": 12,
        "fft_mode": "high_pass",
    })

    assert panel.state() == {"align_rows": "median"}


def test_viewer_full_panel_round_trips_standard_processing_state(qapp):
    from probeflow.gui import ProcessingControlPanel

    panel = ProcessingControlPanel("viewer_full")
    panel.set_state({
        "align_rows": "mean",
        "bg_order": 4,
        "bg_step_tolerance": True,
        "stm_line_bg": "step_tolerant",
        "facet_level": True,
        "smooth_sigma": 3,
        "highpass_sigma": 12,
        "edge_method": "dog",
        "edge_sigma": 4,
        "fft_mode": "high_pass",
        "fft_cutoff": 0.25,
        "fft_soft_border": True,
    })

    state = panel.state()

    assert state["align_rows"] == "mean"
    assert state["bg_order"] == 4
    assert state["bg_step_tolerance"] is True
    assert state["stm_line_bg"] == "step_tolerant"
    assert state["facet_level"] is True
    assert state["smooth_sigma"] == 3
    assert state["highpass_sigma"] == 12
    assert state["edge_method"] == "dog"
    assert state["edge_sigma"] == 4
    assert state["edge_sigma2"] == 8
    assert state["fft_mode"] == "high_pass"
    assert state["fft_cutoff"] == 0.25
    assert state["fft_soft_border"] is True


def test_viewer_dialog_keeps_standard_processing_visible(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])

    assert dlg._processing_panel.isHidden() is False
    assert not hasattr(dlg, "_set_zero_btn")
    assert dlg._selection_widget.isVisible() is False
    assert dlg._advanced_widget.isVisible() is False
    assert dlg._spec_overlay_widget.isVisible() is False
    assert dlg._export_widget.isVisible() is False

    dlg.close()
    dlg.deleteLater()


def test_viewer_apply_merges_standard_and_advanced_processing(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ImageViewerDialog, "_refresh_processing_display", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._processing_panel.set_state({"align_rows": "median", "bg_order": 2})
    dlg._undistort_shear_sl.setValue(3)
    dlg._undistort_scale_sl.setValue(110)

    dlg._on_apply_processing()

    assert dlg._processing["align_rows"] == "median"
    assert dlg._processing["bg_order"] == 2
    assert dlg._processing["linear_undistort"] is True
    assert dlg._processing["undistort_shear_x"] == 3.0
    assert dlg._processing["undistort_scale_y"] == 1.10

    dlg.close()
    dlg.deleteLater()


def test_viewer_zero_plane_workflow_remains_available(qapp, monkeypatch):
    import numpy as np
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ImageViewerDialog, "_refresh_processing_display", lambda self: None)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
    dlg._raw_arr = np.zeros((10, 10), dtype=float)

    dlg._set_zero_plane_btn.setChecked(True)
    dlg._on_set_zero_pick(0.0, 0.0)
    dlg._on_set_zero_pick(0.5, 0.5)
    dlg._on_set_zero_pick(1.0, 1.0)

    assert dlg._processing["set_zero_plane_points"] == [(0, 0), (4, 4), (9, 9)]
    assert "set_zero_xy" not in dlg._processing
    assert dlg._set_zero_plane_btn.isChecked() is False

    dlg.close()
    dlg.deleteLater()


def test_viewer_dialog_initializes_panel_from_thumbnail_processing(qapp, monkeypatch):
    from probeflow.gui import ImageViewerDialog, ProcessingControlPanel, SxmFile

    captured = {}

    def fake_build(self):
        self._processing_panel = ProcessingControlPanel("viewer_full")

    def fake_set_state(self, state):
        captured.update(state)

    monkeypatch.setattr(ImageViewerDialog, "_build", fake_build)
    monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
    monkeypatch.setattr(ProcessingControlPanel, "set_state", fake_set_state)

    entry = SxmFile(path=Path("/tmp/example.sxm"), stem="example", Nx=8, Ny=8)
    ImageViewerDialog(
        entry,
        [entry],
        "gray",
        {},
        processing={"align_rows": "median", "bg_order": 2},
    )

    assert captured == {"align_rows": "median", "bg_order": 2}
