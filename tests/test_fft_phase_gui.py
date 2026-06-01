"""GUI tests for the FFT viewer's Magnitude / Phase view option."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

N = 128


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


def _dialog(qapp, captured=None):
    from probeflow.gui.dialogs.fft_viewer import FFTViewerDialog
    yy, xx = np.mgrid[:N, :N]
    img = np.exp(-(((xx - 40) ** 2 + (yy - 70) ** 2) / 400.0)) + 0.5 * np.sin(2 * np.pi * 8 * xx / N)
    apply_fn = (lambda op, p: captured.update(op=op, params=p)) if captured is not None else None
    return FFTViewerDialog(img, (10e-9, 10e-9), apply_correction_fn=apply_fn,
                           get_image_fn=lambda: img)


def _set_view(dlg, text):
    dlg._fft_view_combo.setCurrentIndex(dlg._fft_view_combo.findText(text))


class TestPhaseView:
    def test_default_is_magnitude(self, qapp):
        dlg = _dialog(qapp)
        assert dlg._fft_display_mode == "magnitude"
        assert dlg._fft_view_combo.currentText() == "Magnitude"
        dlg.deleteLater()

    def test_phase_returns_angle_in_range(self, qapp):
        dlg = _dialog(qapp)
        _set_view(dlg, "Phase")
        disp = dlg._compute_display_fft()
        assert dlg._fft_display_mode == "phase"
        assert np.allclose(disp, dlg._fft_phase, equal_nan=True)
        finite = disp[np.isfinite(disp)]
        assert finite.min() >= -np.pi - 1e-9 and finite.max() <= np.pi + 1e-9
        assert dlg._disp_range == (-np.pi, np.pi)
        dlg.deleteLater()

    def test_phase_render_uses_cyclic_cmap_and_fixed_range(self, qapp):
        dlg = _dialog(qapp)
        _set_view(dlg, "Phase")
        assert dlg._fft_im.get_cmap().name == "twilight"
        lo, hi = dlg._fft_im.get_clim()
        assert lo == pytest.approx(-np.pi) and hi == pytest.approx(np.pi)
        dlg.deleteLater()

    def test_scale_and_lut_disabled_in_phase(self, qapp):
        dlg = _dialog(qapp)
        assert dlg._scale_combo.isEnabled() and dlg._cmap_combo.isEnabled()
        _set_view(dlg, "Phase")
        assert not dlg._scale_combo.isEnabled() and not dlg._cmap_combo.isEnabled()
        _set_view(dlg, "Magnitude")
        assert dlg._scale_combo.isEnabled() and dlg._cmap_combo.isEnabled()
        dlg.deleteLater()

    def test_toggle_back_restores_magnitude(self, qapp):
        dlg = _dialog(qapp)
        mag_before = dlg._compute_display_fft().copy()
        _set_view(dlg, "Phase")
        _set_view(dlg, "Magnitude")
        assert dlg._fft_display_mode == "magnitude"
        assert np.allclose(dlg._compute_display_fft(), mag_before, equal_nan=True)
        dlg.deleteLater()

    def test_phase_view_does_not_alter_data_or_tools(self, qapp):
        """The phase toggle is display-only: |F| is unchanged and an apply still
        routes the same op."""
        captured: dict = {}
        dlg = _dialog(qapp, captured=captured)
        mag = dlg._fft_mag.copy()
        _set_view(dlg, "Phase")
        assert np.array_equal(dlg._fft_mag, mag)   # magnitude data untouched
        # Reconstruct still applies in phase view.
        dlg._tab_widget.setCurrentIndex(dlg._reconstruct_tab_index)
        dlg._on_add_selection("circle")
        dlg._on_reconstruct_apply()
        assert captured.get("op") == "inverse_fft_filter"
        dlg.deleteLater()

    def test_view_combo_tooltip_wrapped(self, qapp):
        dlg = _dialog(qapp)
        tt = dlg._fft_view_combo.toolTip()
        assert tt and max(len(line) for line in tt.split("\n")) <= 52
        dlg.deleteLater()
