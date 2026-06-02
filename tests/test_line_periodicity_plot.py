"""Headless tests for the line-periodicity diagnostic plot.

Regression cover for the 'profile shows flat lines' bug: the raw trace carries a
large absolute-height offset while the processed trace is the small,
background-removed corrugation. They must not share one y-axis, or the
corrugation collapses to a flat line.
"""

from __future__ import annotations

import os

import numpy as np
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


def _result():
    from probeflow.analysis.line_periodicity import PeriodicityResult
    return PeriodicityResult(
        method="autocorrelation",
        period_m=0.222e-9,
        uncertainty_m=4.4e-11,
        line_length_m=4.0e-9,
        n_periods=18.0,
        n_samples=400,
        quality="good",
        message="ok",
    )


def _diag(z_raw, z_proc, s_m):
    from probeflow.analysis.line_periodicity import PeriodicityDiagnostic
    return PeriodicityDiagnostic(s_m=s_m, z_raw=z_raw, z_processed=z_proc)


def _axis_by_ylabel(fig, label):
    for ax in fig.axes:
        if ax.get_ylabel() == label:
            return ax
    return None


def test_profile_uses_separate_axis_so_corrugation_is_visible(qapp):
    from probeflow.gui.dialogs.line_periodicity_plot import PeriodicityPlotDialog

    s_m = np.linspace(0.0, 4.0e-9, 400)
    corrugation = 2.0e-11 * np.sin(2 * np.pi * s_m / 0.222e-9)  # ~20 pm
    z_proc = corrugation
    # Raw: same corrugation sitting on a huge absolute offset + tilt (~140 nm).
    z_raw = corrugation - 1.4e-7 + 3.0e-9 * (s_m / s_m[-1])

    dlg = PeriodicityPlotDialog(_result(), _diag(z_raw, z_proc, s_m))
    try:
        prof_ax = _axis_by_ylabel(dlg._fig, "z processed (data units)")
        raw_ax = _axis_by_ylabel(dlg._fig, "z raw (data units)")
        assert prof_ax is not None, "processed trace should have its own axis"
        assert raw_ax is not None, "raw trace should be on a separate twin axis"

        # The processed axis is scaled to the pm corrugation, not the 140 nm
        # raw offset — so the oscillation is actually visible.
        assert max(abs(v) for v in prof_ax.get_ylim()) < 1e-9
        # The raw axis spans the large absolute offset.
        assert max(abs(v) for v in raw_ax.get_ylim()) > 1e-8
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_profile_single_axis_when_no_background_removal(qapp):
    from probeflow.gui.dialogs.line_periodicity_plot import PeriodicityPlotDialog

    s_m = np.linspace(0.0, 4.0e-9, 200)
    z = 2.0e-11 * np.sin(2 * np.pi * s_m / 0.222e-9)
    # method 'none' -> raw and processed coincide.
    dlg = PeriodicityPlotDialog(_result(), _diag(z.copy(), z.copy(), s_m))
    try:
        assert _axis_by_ylabel(dlg._fig, "z raw (data units)") is None
        assert _axis_by_ylabel(dlg._fig, "z (data units)") is not None
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()
