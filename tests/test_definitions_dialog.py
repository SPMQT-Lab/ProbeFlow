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
        "Advanced edge detection (Canny",
        "Manual zero reference",
        "Image arithmetic",
        "Thresholding and bit-depth conversion",
        "Geometric transforms and resampling",
        "FFT-derived correction tools",
        "Forward/backward scan blending",
    ):
        assert operation in html
    # Advanced edge detection explains hysteresis and the mask/ROI outputs.
    assert "hysteresis" in html
    assert "Sobel / Scharr" in html

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
        "selected ROI(s) in the ROI Manager list when present",
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


def test_measurements_reference_has_entries_and_formulas():
    from probeflow.gui.dialogs.definitions import (
        _MEASUREMENT_ENTRIES,
        render_measurements_html,
    )
    from probeflow.gui.styling import THEMES

    html = render_measurements_html(THEMES["light"])

    # Every measurement entry renders an equation block and an "In practice" lead.
    assert html.count('class="equation"') >= len(_MEASUREMENT_ENTRIES)
    assert html.count("In practice:") >= len(_MEASUREMENT_ENTRIES)

    for expected in (
        "Measurements Reference",
        "Distance",
        "Angle",
        "Line profile",
        "Line periodicity",
        "ROI statistics",
        "Step height",
        "Feature maxima",
        "Pair correlation",
        "Feature → lattice",
        # A couple of formulas must match the implementation.
        "rms_roughness = sqrt(mean((z - mean(z))^2))",
        "height_difference = mean_b - mean_a",
        # The line tool's drawing/editing is cross-referenced to the ROI tab.
        "Line ROI actions",
    ):
        assert expected in html, expected


def test_definitions_dialog_tabs_can_focus_howto_processing_and_roi(qapp):
    from probeflow.gui.dialogs.definitions import _DefinitionsDialog
    from probeflow.gui.styling import THEMES

    default = _DefinitionsDialog(THEMES["light"])
    roi_first = _DefinitionsDialog(THEMES["light"], initial_tab="roi")
    howto_first = _DefinitionsDialog(THEMES["light"], initial_tab="howto")
    measure_first = _DefinitionsDialog(THEMES["light"], initial_tab="measurements")
    try:
        assert default.current_reference_tab() == "processing"
        default.set_reference_tab("roi")
        assert default.current_reference_tab() == "roi"
        default.set_reference_tab("measurements")
        assert default.current_reference_tab() == "measurements"
        default.set_reference_tab("howto")
        assert default.current_reference_tab() == "howto"
        default.set_reference_tab("processing")
        assert default.current_reference_tab() == "processing"
        assert roi_first.current_reference_tab() == "roi"
        assert howto_first.current_reference_tab() == "howto"
        assert measure_first.current_reference_tab() == "measurements"
    finally:
        for dlg in (default, roi_first, howto_first, measure_first):
            dlg.close()
            dlg.deleteLater()
        qapp.processEvents()


def test_particle_statistics_reference_has_models_and_formulas():
    from probeflow.gui.dialogs.definitions import (
        _PARTICLE_STATISTICS_ENTRIES,
        render_particle_statistics_html,
    )
    from probeflow.gui.styling import THEMES

    html = render_particle_statistics_html(THEMES["light"])

    # Every entry that defines an Operation block renders an equation.
    with_equations = sum(1 for entry in _PARTICLE_STATISTICS_ENTRIES if entry.equations)
    assert html.count('class="equation"') >= with_equations

    # The methodology, all three null models, and each statistic are documented.
    for heading in (
        "How a comparison works",
        "Homogeneous Poisson",
        "Hard-core random",
        "Measured-feature Poisson",
        "Pair correlation g(r)",
        "Nearest-neighbour distribution",
        "Ripley",  # apostrophe in "Ripley's L" is HTML-escaped
        "Cluster sizes",
        "Reading verdicts and limitations",
    ):
        assert heading in html, heading

    # The mathematical defence and honest caveats are present.
    assert "extreme rank length" in html.lower()
    assert "exchangeable" in html.lower()
    assert "lambda = N / A" in html  # CSR intensity
    assert "L(r) = sqrt( K(r) / pi )" in html  # Ripley L transform
    assert "non-equilibrium" in html.lower()  # hard-core honesty
    assert "least user-tested" in html.lower()  # maturity note in the intro


def test_definitions_dialog_can_focus_particle_statistics_tab(qapp):
    from probeflow.gui.dialogs.definitions import _DefinitionsDialog
    from probeflow.gui.styling import THEMES

    stats_first = _DefinitionsDialog(THEMES["light"], initial_tab="particle_statistics")
    default = _DefinitionsDialog(THEMES["light"])
    try:
        assert stats_first.current_reference_tab() == "particle_statistics"
        default.set_reference_tab("particle_statistics")
        assert default.current_reference_tab() == "particle_statistics"
        default.set_reference_tab("stats")  # alias
        assert default.current_reference_tab() == "particle_statistics"
    finally:
        for dlg in (stats_first, default):
            dlg.close()
            dlg.deleteLater()
        qapp.processEvents()


def test_howto_help_command_is_registered():
    from probeflow.gui.viewer.shortcuts import VIEWER_COMMANDS

    ids = {c.command_id for c in VIEWER_COMMANDS}
    assert "help.howto" in ids
