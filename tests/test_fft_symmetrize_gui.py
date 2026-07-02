"""GUI tests for the FFT viewer's Symmetrize tab."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

N = 128
C = (N - 1) / 2.0
PERIOD_PX = 8.0          # FFT peaks at radius N / PERIOD_PX = 16 px
ANGLE0_DEG = 12.0
SCAN_NM = 10.0
# Hex first-order Bragg radius r = 2·scan/(a·√3) px; invert so the reference
# lattice constant lands the predicted annulus on the synthetic peaks.
A_NM = 2.0 * SCAN_NM * PERIOD_PX / (N * np.sqrt(3.0))


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


def _hex_scene(angle0_deg=ANGLE0_DEG):
    yy, xx = np.mgrid[:N, :N].astype(np.float64)
    q = 2.0 * np.pi / PERIOD_PX
    out = np.zeros((N, N))
    for k in range(3):
        th = np.radians(angle0_deg) + k * np.pi / 3.0
        out += np.cos(q * ((xx - C) * np.cos(th) + (yy - C) * np.sin(th)))
    return out


def _dialog(qapp, captured=None, scan_range=(SCAN_NM * 1e-9, SCAN_NM * 1e-9),
            roi=False, new_image_fn=None):
    from probeflow.gui.dialogs.fft_viewer import FFTViewerDialog
    img = _hex_scene()
    apply_fn = (lambda op, p: captured.update(op=op, params=p)) if captured is not None else None
    kw = {}
    if roi:
        kw = dict(roi_bounds_px=(10, 90, 10, 90), roi_id="roi1", roi_name="r")
    dlg = FFTViewerDialog(img, scan_range, apply_correction_fn=apply_fn,
                          get_image_fn=lambda: img, new_image_fn=new_image_fn, **kw)
    dlg._tab_widget.setCurrentIndex(dlg._symmetrize_tab_index)
    return dlg, img


class TestTab:
    def test_tab_registered(self, qapp):
        dlg, _ = _dialog(qapp)
        assert dlg._symmetrize_tab_index >= 0
        assert dlg._tab_widget.tabText(dlg._symmetrize_tab_index) == "Symmetrize"
        assert dlg._symm_fold_spin.value() == 6
        assert dlg._symm_register_cb.isChecked()
        dlg.close()

    def test_mirror_toggle_gates_axis_controls(self, qapp):
        dlg, _ = _dialog(qapp)
        assert not dlg._symm_axis_spin.isEnabled()
        assert not dlg._symm_align_btn.isEnabled()
        dlg._symm_mirror_cb.setChecked(True)
        assert dlg._symm_axis_spin.isEnabled()
        assert dlg._symm_align_btn.isEnabled()
        dlg.close()


class TestPreview:
    def test_preview_shows_result_and_metric(self, qapp):
        dlg, _ = _dialog(qapp)
        dlg._on_symmetrize_preview()
        assert dlg._symmetrize_preview_active
        assert dlg._symm_clear_btn.isEnabled()
        assert "Removed asymmetry" in dlg._symm_status_lbl.text()
        dlg._on_symmetrize_clear()
        assert not dlg._symmetrize_preview_active
        dlg.close()

    def test_centred_lattice_reports_low_asymmetry(self, qapp):
        dlg, img = _dialog(qapp)
        res = dlg._compute_symmetrization(img)
        assert res.symmetry_residual_norm < 0.06

    def test_anisotropic_pixels_warn(self, qapp):
        dlg, _ = _dialog(qapp, scan_range=(10e-9, 20e-9))
        assert "anisotropic" in dlg._symm_status_lbl.text()
        dlg.close()

    def test_square_pixels_do_not_warn(self, qapp):
        dlg, _ = _dialog(qapp)
        assert "anisotropic" not in dlg._symm_status_lbl.text()
        dlg.close()


class TestApply:
    def test_apply_emits_symmetrize_op(self, qapp):
        captured = {}
        dlg, _ = _dialog(qapp, captured)
        dlg._symm_fold_spin.setValue(4)
        dlg._on_symmetrize_apply()
        assert captured["op"] == "symmetrize_fft"
        p = captured["params"]
        assert p["n_fold"] == 4
        assert p["mirror"] is False
        assert p["register"] is True
        assert p["interpolation"] == "linear"
        assert p["strict_coverage"] is False
        assert p["fft_source"] == "whole_image"
        dlg.close()

    def test_apply_params_replay_through_processing_state(self, qapp):
        # The emitted op must round-trip through the real pipeline.
        from probeflow.processing.state import ProcessingState, ProcessingStep, \
            apply_processing_state
        from probeflow.processing.symmetrize import symmetrize_filter
        captured = {}
        dlg, img = _dialog(qapp, captured)
        dlg._on_symmetrize_apply()
        state = ProcessingState(
            steps=[ProcessingStep(captured["op"], captured["params"])])
        out = apply_processing_state(img, state)
        np.testing.assert_array_equal(out, symmetrize_filter(img, 6))
        dlg.close()

    def test_apply_hides_active_preview(self, qapp):
        captured = {}
        dlg, _ = _dialog(qapp, captured)
        dlg._on_symmetrize_preview()
        dlg._on_symmetrize_apply()
        assert not dlg._symmetrize_preview_active
        assert captured["op"] == "symmetrize_fft"
        dlg.close()

    def test_roi_source_creates_new_image(self, qapp):
        opened = {}

        def _new_image(arr, scan_range_m, provenance):
            opened.update(arr=arr, scan_range=scan_range_m, provenance=provenance)

        dlg, img = _dialog(qapp, roi=True, new_image_fn=_new_image)
        dlg._fft_source = "active_roi"
        dlg._on_symmetrize_apply()
        assert opened["provenance"]["op"] == "symmetrize_fft"
        assert opened["provenance"]["params"]["fft_source"] == "active_roi"
        assert opened["provenance"]["params"]["fft_roi_id"] == "roi1"
        assert opened["arr"].shape == np.asarray(dlg._arr).shape
        dlg.close()


class TestAlignToBragg:
    def _configured(self, qapp):
        dlg, img = _dialog(qapp)
        dlg._bragg_sym_combo.setCurrentIndex(1)          # Hexagonal
        dlg._bragg_unit_combo.setCurrentText("nm")
        dlg._bragg_a_spin.setValue(A_NM)
        dlg._symm_mirror_cb.setChecked(True)
        return dlg

    def test_axis_snaps_to_lattice_direction(self, qapp):
        dlg = self._configured(qapp)
        dlg._on_symm_align_axis()
        assert dlg._symm_axis_spin.value() == pytest.approx(ANGLE0_DEG, abs=2.5)
        assert "Mirror axis set" in dlg._symm_status_lbl.text()
        dlg.close()

    def test_bad_lattice_constant_reports_no_peaks(self, qapp):
        dlg = self._configured(qapp)
        dlg._bragg_a_spin.setValue(A_NM * 3.0)   # annulus far from real peaks
        before = dlg._symm_axis_spin.value()
        dlg._on_symm_align_axis()
        txt = dlg._symm_status_lbl.text()
        assert ("Could not detect" in txt) or ("do not share" in txt)
        assert dlg._symm_axis_spin.value() == before
        dlg.close()
