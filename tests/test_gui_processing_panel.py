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


def test_viewer_full_panel_round_trips_advanced_processing_state(qapp):
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
        "linear_undistort": True,
        "undistort_shear_x": 2,
        "undistort_scale_y": 1.10,
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
    assert state["linear_undistort"] is True
    assert state["undistort_shear_x"] == 2.0
    assert state["undistort_scale_y"] == 1.10


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
