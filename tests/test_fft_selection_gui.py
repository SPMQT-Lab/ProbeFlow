"""GUI tests for the FFT viewer's Reconstruct tab + Fourier selection overlay."""

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


def _scene():
    yy, xx = np.mgrid[:N, :N]
    return np.exp(-(((xx - 40) ** 2 + (yy - 70) ** 2) / 400.0)) + 0.5 * np.sin(2 * np.pi * 8 * xx / N)


def _dialog(qapp, captured=None, roi=False):
    from probeflow.gui.dialogs.fft_viewer import FFTViewerDialog
    img = _scene()
    apply_fn = (lambda op, p: captured.update(op=op, params=p)) if captured is not None else None
    kw = {}
    if roi:
        kw = dict(roi_bounds_px=(10, 90, 10, 90), roi_id="roi1", roi_name="r")
    dlg = FFTViewerDialog(img, (10e-9, 10e-9), apply_correction_fn=apply_fn,
                          get_image_fn=lambda: img, **kw)
    dlg._tab_widget.setCurrentIndex(dlg._reconstruct_tab_index)
    return dlg, img


class TestSelectionOverlay:
    def test_add_move_resize_delete(self, qapp):
        dlg, _img = _dialog(qapp)
        dlg._on_add_selection("circle")
        ov = dlg._fft_selection_overlay
        assert ov.count() == 1 and len(ov._artists) > 0
        # to_fft_ellipses maps q→px; centre is off-DC, radius ≥ 1 px.
        e = ov.to_fft_ellipses()[0]
        assert e["dx"] > 0 and e["rx"] >= 1.0 and e["kind"] == "circle"
        # resize: enlarging rx_q grows the px radius
        ov._sels[0].rx_q *= 2
        assert ov.to_fft_ellipses()[0]["rx"] > e["rx"]
        ov.delete_selected()
        assert ov.count() == 0

    def test_ellipse_independent_axes(self, qapp):
        dlg, _ = _dialog(qapp)
        dlg._on_add_selection("ellipse")
        ov = dlg._fft_selection_overlay
        ov._sels[0].rx_q = 0.5
        ov._sels[0].ry_q = 0.2
        e = ov.to_fft_ellipses()[0]
        assert e["rx"] != e["ry"]

    def test_conjugate_drawn(self, qapp):
        dlg, _ = _dialog(qapp)
        dlg._on_add_selection("circle")
        ov = dlg._fft_selection_overlay
        # two ellipse patches per selection (feature + conjugate) + 3 handles
        from matplotlib.patches import Ellipse
        n_ell = sum(isinstance(a, Ellipse) for a in ov._artists)
        assert n_ell == 2

    def test_clear(self, qapp):
        dlg, _ = _dialog(qapp)
        dlg._on_add_selection("circle")
        dlg._on_add_selection("ellipse")
        assert dlg._fft_selection_overlay.count() == 2
        dlg._on_clear_selections()
        assert dlg._fft_selection_overlay.count() == 0


class TestReconstructTab:
    def test_tab_present(self, qapp):
        dlg, _ = _dialog(qapp)
        labels = [dlg._tab_widget.tabText(i) for i in range(dlg._tab_widget.count())]
        assert "Reconstruct" in labels
        assert dlg._reconstruct_active()

    def test_preview_does_not_mutate_and_sets_active(self, qapp):
        dlg, img = _dialog(qapp)
        dlg._on_add_selection("circle")
        before = img.copy()
        dlg._on_reconstruct_preview()
        assert dlg._reconstruct_preview_active is True
        assert np.array_equal(img, before)   # source untouched
        dlg._on_reconstruct_clear()
        assert dlg._reconstruct_preview_active is False

    def test_whole_image_apply_routes_op(self, qapp):
        captured: dict = {}
        dlg, _ = _dialog(qapp, captured=captured)
        dlg._on_add_selection("circle")
        dlg._recon_mode_combo.setCurrentIndex(1)        # Keep selected
        dlg._recon_soft_spin.setValue(2)
        dlg._on_reconstruct_apply()
        assert captured["op"] == "inverse_fft_filter"
        p = captured["params"]
        assert p["mode"] == "keep_selected"
        assert p["conjugate_symmetric"] is True
        assert p["soft_px"] == 2.0
        assert len(p["selections"]) == 1
        assert p["fft_source"] == "whole_image"

    def test_apply_without_selection_warns(self, qapp):
        captured: dict = {}
        dlg, _ = _dialog(qapp, captured=captured)
        dlg._on_reconstruct_apply()
        assert "op" not in captured
        assert "selection" in dlg._recon_status_lbl.text().lower()

    def test_roi_apply_degrades_without_new_image_host(self, qapp):
        captured: dict = {}
        dlg, _ = _dialog(qapp, captured=captured, roi=True)
        # switch FFT source to the ROI
        dlg._fft_source = "active_roi"
        dlg._arr, dlg._scan_range_m = dlg._resolve_source_array()
        dlg._on_add_selection("circle")
        dlg._on_reconstruct_apply()
        assert "op" not in captured                     # no whole-image op routed
        assert "export" in dlg._recon_status_lbl.text().lower()

    def test_export_writes_file(self, qapp, monkeypatch, tmp_path):
        dlg, _ = _dialog(qapp)
        dlg._on_add_selection("circle")
        out = tmp_path / "result.csv"
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(out), "CSV (*.csv)"))
        dlg._on_reconstruct_export("result")
        assert out.exists() and out.stat().st_size > 0

    def test_tooltips_wrapped(self, qapp):
        dlg, _ = _dialog(qapp)
        for w in (dlg._recon_mode_combo, dlg._recon_conj_cb, dlg._recon_soft_spin,
                  dlg._recon_preview_btn, dlg._recon_apply_btn):
            tt = w.toolTip()
            assert tt and max(len(line) for line in tt.split("\n")) <= 52
