"""Tests for the processing definitions reference content."""

from __future__ import annotations

import pytest


pytest.importorskip("PySide6.QtWidgets")


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
        "Forward/backward scan blending",
    ):
        assert operation in html

    assert "color: #111827" in html
    assert "#cdd6f4" not in lowered
    assert "#a6adc8" not in lowered
    assert "S = [j0, j1)" in html
    assert "slope &lt; tan(step_threshold_deg)" in html
    assert "mean is subtracted before the FFT in both modes" in html
    assert "derivative cusp" in html
