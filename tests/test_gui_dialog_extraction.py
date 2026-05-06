"""Smoke tests for extracted standalone GUI dialogs."""

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
    try:
        return QApplication([])
    except Exception as exc:
        pytest.skip(f"QApplication unavailable: {exc}")


def _theme() -> dict[str, str]:
    return {
        "bg": "#1e1e2e",
        "fg": "#cdd6f4",
        "sub_fg": "#6c7086",
    }


def test_dialogs_import_from_dialogs_package():
    from probeflow.gui.dialogs import (
        AboutDialog,
        FFTViewerDialog,
        PeriodicFilterDialog,
        SpecMappingDialog,
        SpecViewerDialog,
        ViewerSpecMappingDialog,
    )

    assert AboutDialog.__name__ == "AboutDialog"
    assert FFTViewerDialog.__name__ == "FFTViewerDialog"
    assert PeriodicFilterDialog.__name__ == "PeriodicFilterDialog"
    assert SpecMappingDialog.__name__ == "SpecMappingDialog"
    assert SpecViewerDialog.__name__ == "SpecViewerDialog"
    assert ViewerSpecMappingDialog.__name__ == "ViewerSpecMappingDialog"


def test_dialogs_remain_available_from_gui_package():
    from probeflow.gui import (
        AboutDialog,
        FFTViewerDialog,
        PeriodicFilterDialog,
        SpecMappingDialog,
        SpecViewerDialog,
        ViewerSpecMappingDialog,
    )

    assert AboutDialog.__module__ == "probeflow.gui.dialogs.about"
    assert FFTViewerDialog.__module__ == "probeflow.gui.dialogs.fft_viewer"
    assert PeriodicFilterDialog.__module__ == "probeflow.gui.dialogs.periodic_filter"
    assert SpecMappingDialog.__module__ == "probeflow.gui.dialogs.spec_mapping"
    assert SpecViewerDialog.__module__ == "probeflow.gui.dialogs.spec_viewer"
    assert ViewerSpecMappingDialog.__module__ == "probeflow.gui.dialogs.spec_mapping"


def test_extracted_dialogs_smoke_construct(qapp):
    from probeflow.gui.dialogs import (
        AboutDialog,
        FFTViewerDialog,
        PeriodicFilterDialog,
        SpecMappingDialog,
        SpecViewerDialog,
        ViewerSpecMappingDialog,
    )

    arr = np.arange(64, dtype=float).reshape(8, 8)
    dialogs = [
        FFTViewerDialog(arr, (1e-9, 1e-9), theme=_theme()),
        PeriodicFilterDialog(arr),
        SpecMappingDialog([], [], {}),
        ViewerSpecMappingDialog("image", [], {}),
        AboutDialog(_theme()),
    ]

    for dlg in dialogs:
        assert dlg.windowTitle()

    # Keep Qt/Matplotlib-backed widgets alive for the test process so queued
    # draw_idle timers cannot fire against deleted canvas objects in later tests.
    qapp._probeflow_dialog_smoke_refs = dialogs
