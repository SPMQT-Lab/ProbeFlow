"""Tests for the processing definitions reference content."""

from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        return app
    return QApplication([])


def test_definitions_reference_has_equations_and_light_theme_contrast():
    from probeflow.gui.dialogs.definitions import (
        _DEFINITION_ENTRIES,
        render_definitions_html,
    )
    from probeflow.gui.styling import THEMES

    html = render_definitions_html(THEMES["light"])
    lowered = html.lower()

    assert html.count('class="equation"') >= len(_DEFINITION_ENTRIES)
    for operation in (
        "Bad-line correction",
        "Simple background subtraction",
        "STM background subtraction",
        "Gaussian high-pass",
        "Periodic notch filtering",
        "Manual zero reference",
        "Image arithmetic",
        "Thresholding and bit-depth conversion",
        "Geometric transforms and resampling",
        "FFT-derived correction tools",
        "Forward/backward scan blending",
    ):
        assert operation in html

    assert "color: #111827" in html
    assert "#cdd6f4" not in lowered
    assert "#a6adc8" not in lowered
    # Equation/reference blocks are preserved verbatim.
    assert "S = [j0, j1)" in html
    assert "slope &lt; tan(step_threshold_deg)" in html
    # Tutorial prose (reworded) still conveys the key physics.
    assert "subtracted before the FFT in both modes" in html
    assert "sharp kink" in html  # creep-model singularity explanation
    # Tutorial framing is present.
    assert "no valid measurement" in lowered  # NaN explained in the intro
    assert "fourier transform" in lowered      # FFT introduced in plain terms


def test_roi_reference_has_action_scope_and_tool_behaviour():
    from probeflow.gui.dialogs.definitions import (
        _ROI_REFERENCE_ENTRIES,
        render_roi_reference_html,
    )
    from probeflow.gui.styling import THEMES

    html = render_roi_reference_html(THEMES["light"])

    assert html.count('class="equation"') >= len(_ROI_REFERENCE_ENTRIES)
    for expected in (
        "ROI Actions Reference",
        "x = image column, y = image row",
        "active_roi_id",
        ".rois.json",
        "selected ROI(s) in ROI Manager dock win",
        "ROI filters only",
        "STM background from ROI",
        "correction is applied to the full image",
        "line = two endpoints plus optional averaging width",
        "Point ROI actions",
        "pair-correlation",
        # Tutorial behaviour the reference must teach.
        "select-then-edit",
        "Contrast scope",
    ):
        assert expected in html


def test_howto_reference_has_numbered_steps_and_key_workflows():
    from probeflow.gui.dialogs.definitions import (
        _HOWTO_ENTRIES,
        render_howto_html,
    )
    from probeflow.gui.styling import THEMES

    html = render_howto_html(THEMES["light"])

    # Every guide renders an ordered (numbered) step list.
    assert html.count("<ol>") >= len(_HOWTO_ENTRIES)
    assert "<li>" in html

    for expected in (
        "How-to Guides",
        "Open an image and flatten it",
        "Measure a height profile along a line",
        "Export line profile as CSV",
        "Create, select, and delete ROIs",
        "Show two regions at once (per-region contrast)",
        "Contrast scope",
        "Measure a step height between two regions",
        "Correct lattice distortion with the FFT viewer",
        "Bragg",
        "Apply correction",
        "JSON",
    ):
        assert expected in html, expected


def test_definitions_dialog_tabs_can_focus_howto_processing_and_roi(qapp):
    from probeflow.gui.dialogs.definitions import _DefinitionsDialog
    from probeflow.gui.styling import THEMES

    default = _DefinitionsDialog(THEMES["light"])
    roi_first = _DefinitionsDialog(THEMES["light"], initial_tab="roi")
    howto_first = _DefinitionsDialog(THEMES["light"], initial_tab="howto")
    try:
        assert default.current_reference_tab() == "processing"
        default.set_reference_tab("roi")
        assert default.current_reference_tab() == "roi"
        default.set_reference_tab("howto")
        assert default.current_reference_tab() == "howto"
        default.set_reference_tab("processing")
        assert default.current_reference_tab() == "processing"
        assert roi_first.current_reference_tab() == "roi"
        assert howto_first.current_reference_tab() == "howto"
    finally:
        for dlg in (default, roi_first, howto_first):
            dlg.close()
            dlg.deleteLater()
        qapp.processEvents()


def test_howto_help_command_is_registered():
    from probeflow.gui.viewer.shortcuts import VIEWER_COMMANDS

    ids = {c.command_id for c in VIEWER_COMMANDS}
    assert "help.howto" in ids
