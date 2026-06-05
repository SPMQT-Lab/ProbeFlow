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


class TestReconstructStatus:
    def test_changing_reconstruct_mode_does_not_crash(self, qapp):
        # currentIndexChanged emits the int index; _update_reconstruct_status
        # treats its first positional arg as a reconstruction result, so the
        # connection must not pass the index straight through (it used to call
        # int.imag_residual_norm and raise AttributeError).
        dlg = _dialog(qapp)
        other = 1 - dlg._recon_mode_combo.currentIndex()
        dlg._recon_mode_combo.setCurrentIndex(other)  # fires currentIndexChanged
        assert dlg._recon_mode_combo.currentIndex() == other
        dlg.deleteLater()


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

    def test_phase_hides_dc_for_zero_and_mask_modes(self, qapp):
        dlg = _dialog(qapp)
        _set_view(dlg, "Phase")
        cy, cx = (n // 2 for n in dlg._fft_phase.shape)

        dlg._dc_combo.setCurrentIndex(1)  # Keep DC
        keep = dlg._compute_display_fft()
        assert np.isfinite(keep[cy, cx])

        dlg._dc_combo.setCurrentIndex(0)  # Zero DC
        zero = dlg._compute_display_fft()
        assert np.isnan(zero[cy, cx])

        dlg._dc_combo.setCurrentIndex(2)  # Mask DC
        mask = dlg._compute_display_fft()
        assert np.isnan(mask[cy, cx])
        dlg.deleteLater()

    def test_phase_range_controls_are_fixed(self, qapp):
        dlg = _dialog(qapp)
        _set_view(dlg, "Phase")
        assert not dlg._hist_panel.isEnabled()

        dlg._on_fft_hist_range_released(-1.0, 1.0)
        assert dlg._fft_drs.mode != "manual"

        dlg._fft_drs.set_manual(-1.0, 1.0)
        lo, hi = dlg._fft_im.get_clim()
        assert lo == pytest.approx(-np.pi)
        assert hi == pytest.approx(np.pi)
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
        from probeflow.gui.dialogs.fft_selection import FourierSelection
        ov = dlg._ensure_selection_overlay()
        ov._sels.append(FourierSelection("ellipse", cx_q=1.5, cy_q=0.0, rx_q=0.5, ry_q=0.5))
        ov._selected = 0
        dlg._on_selection_changed()
        dlg._on_reconstruct_apply()
        assert captured.get("op") == "inverse_fft_filter"
        dlg.deleteLater()

    def test_view_combo_tooltip_wrapped(self, qapp):
        dlg = _dialog(qapp)
        tt = dlg._fft_view_combo.toolTip()
        assert tt and max(len(line) for line in tt.split("\n")) <= 52
        dlg.deleteLater()
