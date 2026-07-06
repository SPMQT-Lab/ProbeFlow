"""Shared display-unit scaling in the spectroscopy viewer dialogs (GUI).

2026-07-06 spectroscopy display review: traces co-plotted on one axis must
share one SI-prefix scale per unit.  Before the fix, each trace/channel got
its prefix from its own values, so a 50 pA and a 2 nA spectrum plotted as
50 vs 2 — relative magnitudes silently distorted up to 1000×.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from probeflow.gui.models import VertFile
from probeflow.io.spectroscopy import SpecData

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


def _spec(current_a: float, n: int = 32) -> SpecData:
    x = np.linspace(-0.5, 0.5, n)
    return SpecData(
        header={},
        channels={"I": np.full(n, current_a), "Z": np.full(n, 1e-9)},
        x_array=x,
        x_label="Bias (V)",
        x_unit="V",
        y_units={"I": "A", "Z": "m"},
        position=(0.0, 0.0),
        metadata={"sweep_type": "bias_sweep", "n_points": n},
        channel_order=["I", "Z"],
        default_channels=["I"],
    )


def _two_channel_spec(i1_a: float, i2_a: float, n: int = 32) -> SpecData:
    x = np.linspace(-0.5, 0.5, n)
    return SpecData(
        header={},
        channels={"I1": np.full(n, i1_a), "I2": np.full(n, i2_a)},
        x_array=x,
        x_label="Bias (V)",
        x_unit="V",
        y_units={"I1": "A", "I2": "A"},
        position=(0.0, 0.0),
        metadata={"sweep_type": "bias_sweep", "n_points": n},
        channel_order=["I1", "I2"],
        default_channels=["I1"],
    )


@pytest.fixture
def entries(tmp_path):
    paths = []
    for name in ("spec_a", "spec_b"):
        p = tmp_path / f"{name}.VERT"
        p.write_text("")
        paths.append(VertFile(path=p, stem=name))
    return paths


class TestOverlayDialogSharedScale:
    def _dialog(self, qapp, entries, monkeypatch, specs):
        lookup = {e.path: s for e, s in zip(entries, specs)}
        monkeypatch.setattr(
            "probeflow.io.spectroscopy.read_spec_file",
            lambda path: lookup[Path(path)],
        )
        from probeflow.gui.spec_viewer.overlay import SpecOverlayDialog

        return SpecOverlayDialog(entries, {})

    def test_same_channel_traces_share_one_unit_scale(
        self, qapp, entries, monkeypatch
    ):
        dlg = self._dialog(
            qapp, entries, monkeypatch, [_spec(50e-12), _spec(2e-9)]
        )
        try:
            traces = dlg._displayed_traces
            assert len(traces) == 2
            # One unit for the shared axis…
            assert traces[0].y_unit == traces[1].y_unit
            # …and true relative magnitude preserved (2 nA / 50 pA = 40).
            ratio = np.median(traces[1].y_display) / np.median(traces[0].y_display)
            assert ratio == pytest.approx(40.0)
        finally:
            dlg.deleteLater()

    def test_export_matches_display_scaling(self, qapp, entries, monkeypatch):
        dlg = self._dialog(
            qapp, entries, monkeypatch, [_spec(50e-12), _spec(2e-9)]
        )
        try:
            rows = dlg._current_displayed_spectra()
            assert [r.y_unit for r in rows] == [t.y_unit for t in dlg._displayed_traces]
            for row, trace in zip(rows, dlg._displayed_traces):
                np.testing.assert_allclose(row.y_display, trace.y_display)
        finally:
            dlg.deleteLater()


class TestSingleViewerSharedScale:
    def _dialog(self, qapp, entry, monkeypatch, spec):
        monkeypatch.setattr(
            "probeflow.io.spectroscopy.read_spec_file", lambda path: spec
        )
        from probeflow.gui.spec_viewer.single import SpecViewerDialog

        return SpecViewerDialog(entry, {})

    def test_overlay_mode_channels_share_scale(self, qapp, entries, monkeypatch):
        dlg = self._dialog(
            qapp, entries[0], monkeypatch, _two_channel_spec(50e-12, 2e-9)
        )
        try:
            for cb in dlg._checkboxes.values():
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
            dlg._plot_mode_cb.setCurrentText("Overlay")  # triggers _redraw
            traces = {t.y_channel: t for t in dlg._displayed_traces}
            assert set(traces) == {"I1", "I2"}
            assert traces["I1"].y_unit == traces["I2"].y_unit
            ratio = np.median(traces["I2"].y_display) / np.median(traces["I1"].y_display)
            assert ratio == pytest.approx(40.0)
        finally:
            dlg.deleteLater()

    def test_separate_mode_keeps_per_channel_auto_units(
        self, qapp, entries, monkeypatch
    ):
        dlg = self._dialog(
            qapp, entries[0], monkeypatch, _two_channel_spec(50e-12, 2e-9)
        )
        try:
            for cb in dlg._checkboxes.values():
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
            dlg._plot_mode_cb.setCurrentText("Separate")
            dlg._redraw()
            traces = {t.y_channel: t for t in dlg._displayed_traces}
            # Per-axis auto units: each channel picks its own prefix.
            assert traces["I1"].y_unit == "pA"
            assert traces["I2"].y_unit == "nA"
            assert np.median(traces["I1"].y_display) == pytest.approx(50.0)
            assert np.median(traces["I2"].y_display) == pytest.approx(2.0)
        finally:
            dlg.deleteLater()

    def test_overlay_export_matches_display(self, qapp, entries, monkeypatch):
        dlg = self._dialog(
            qapp, entries[0], monkeypatch, _two_channel_spec(50e-12, 2e-9)
        )
        try:
            for cb in dlg._checkboxes.values():
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
            dlg._plot_mode_cb.setCurrentText("Waterfall")
            rows = {r.y_channel: r for r in dlg._current_displayed_spectra()}
            traces = {t.y_channel: t for t in dlg._displayed_traces}
            assert set(rows) == set(traces)
            for ch in rows:
                assert rows[ch].y_unit == traces[ch].y_unit
                np.testing.assert_allclose(rows[ch].y_display, traces[ch].y_display)
        finally:
            dlg.deleteLater()


class TestSharedScaleValuesHelper:
    def test_groups_by_unit_and_filters_nonfinite(self):
        from types import SimpleNamespace

        from probeflow.gui.spec_viewer.shared import _shared_scale_values

        groups = _shared_scale_values([
            SimpleNamespace(y_unit="A", y_display=np.array([1e-12, np.nan])),
            SimpleNamespace(y_unit="A", y_display=np.array([2e-9])),
            SimpleNamespace(y_unit="m", y_display=np.array([1e-9])),
        ])
        np.testing.assert_allclose(groups["A"], [1e-12, 2e-9])
        np.testing.assert_allclose(groups["m"], [1e-9])
