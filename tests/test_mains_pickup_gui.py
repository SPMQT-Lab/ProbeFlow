"""GUI tests for the FFT viewer's ⚡ Mains tab (mains-pickup tool)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# A synthetic scan with injected 50 Hz mains: 10 nm / 160 px, v = 2e-8 m/s
# → T_line = 0.5 s, so 50 Hz sits at q = 2.5 nm⁻¹ (FFT bin 25).
W_M = 10e-9
NPX = 160
V = 2.0e-8


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


def _scan():
    yy, xx = np.mgrid[:NPX, :NPX]
    t = yy * (W_M / V) + xx * ((W_M / V) / NPX)
    base = 0.2e-9 * np.exp(-(((xx - 50) ** 2 + (yy - 60) ** 2) / 300))
    return base + 0.03e-9 * np.sin(2 * np.pi * 50.0 * t)


def _dialog(qapp, scan_speed=V, captured=None):
    from probeflow.gui.dialogs.fft_viewer import FFTViewerDialog
    arr = _scan()
    apply_fn = (lambda op, p: captured.update(op=op, params=p)) if captured is not None else None
    return FFTViewerDialog(
        arr, (W_M, W_M), scan_speed_m_per_s=scan_speed,
        apply_correction_fn=apply_fn, get_image_fn=lambda: arr,
    )


class TestMainsTab:
    def test_tab_present_and_speed_autofilled(self, qapp):
        dlg = _dialog(qapp)
        labels = [dlg._tab_widget.tabText(i) for i in range(dlg._tab_widget.count())]
        assert any("Mains" in t for t in labels)
        assert dlg._mains_speed_spin.value() == pytest.approx(V * 1e9)   # nm/s
        assert dlg._mains_auto_cb.isChecked() is True
        assert dlg._mains_harm_spin.isEnabled() is False
        assert dlg._mains_min_q_spin.value() == pytest.approx(0.0)
        dlg.deleteLater()

    def test_predicts_expected_positions(self, qapp):
        dlg = _dialog(qapp)
        preds = dlg._mains_predictions()
        assert [p["fft_index"] for p in preds] == [25, 50, 75]
        assert preds[0]["q_nm_inv"] == pytest.approx(2.5)
        dlg.deleteLater()

    def test_overlay_toggles_artists(self, qapp):
        dlg = _dialog(qapp)
        assert dlg._mains_artists == []
        dlg._mains_overlay_cb.setChecked(True)
        assert len(dlg._mains_artists) == 6        # 3 harmonics × 2 conjugates
        dlg._mains_overlay_cb.setChecked(False)
        assert dlg._mains_artists == []
        dlg.deleteLater()

    def test_overlay_obeys_minimum_q_floor(self, qapp):
        dlg = _dialog(qapp)
        dlg._mains_auto_cb.setChecked(False)
        dlg._mains_harm_spin.setValue(1)
        dlg._mains_min_q_spin.setValue(3.0)
        dlg._mains_overlay_cb.setChecked(True)

        assert len(dlg._mains_artists) == 4
        gap = np.sqrt(3.0 ** 2 - 2.5 ** 2)
        for art in dlg._mains_artists:
            y0, y1 = art.get_ydata()
            assert max(abs(float(y0)), abs(float(y1))) >= gap
            assert not (float(y0) < 0.0 < float(y1))
        dlg.deleteLater()

    def test_unavailable_speed_shows_note_and_no_overlay(self, qapp):
        dlg = _dialog(qapp, scan_speed=None)
        assert dlg._mains_speed_spin.value() == 0.0
        assert "unavailable" in dlg._mains_status_lbl.text().lower()
        dlg._mains_overlay_cb.setChecked(True)
        assert dlg._mains_artists == []            # nothing to draw without speed
        dlg.deleteLater()

    def test_apply_routes_default_streak_params(self, qapp):
        captured: dict = {}
        dlg = _dialog(qapp, captured=captured)
        dlg._on_mains_apply()
        assert captured["op"] == "mains_pickup_suppression"
        p = captured["params"]
        assert p["mains_frequency_hz"] == 50.0
        assert p["harmonics"] is None
        assert p["notch_shape"] == "streak"
        assert p["min_q_nm_inv"] == pytest.approx(0.0)
        assert p["scan_speed_m_per_s"] == pytest.approx(V)
        assert p["scan_range_m"] == [pytest.approx(W_M), pytest.approx(W_M)]
        dlg.deleteLater()

    def test_manual_harmonics_route_when_auto_is_off(self, qapp):
        captured: dict = {}
        dlg = _dialog(qapp, captured=captured)
        dlg._mains_auto_cb.setChecked(False)
        dlg._mains_freq_combo.setCurrentIndex(1)   # 60 Hz
        dlg._mains_harm_spin.setValue(2)
        dlg._mains_min_q_spin.setValue(1.25)
        dlg._on_mains_apply()

        p = captured["params"]
        assert p["mains_frequency_hz"] == 60.0
        assert p["harmonics"] == 2
        assert p["min_q_nm_inv"] == pytest.approx(1.25)
        assert dlg._mains_harm_spin.isEnabled() is True
        dlg.deleteLater()

    def test_preview_and_clear(self, qapp):
        dlg = _dialog(qapp)
        dlg._on_mains_preview()
        assert dlg._mains_preview_active is True
        dlg._on_mains_clear()
        assert dlg._mains_preview_active is False
        dlg.deleteLater()

    def test_controls_have_wrapped_tooltips(self, qapp):
        dlg = _dialog(qapp)
        for w in (
            dlg._mains_overlay_cb, dlg._mains_freq_combo, dlg._mains_auto_cb,
            dlg._mains_harm_spin, dlg._mains_speed_spin, dlg._mains_radius_spin,
            dlg._mains_min_q_spin, dlg._mains_apply_btn,
        ):
            tt = w.toolTip()
            assert tt, "control must have a tooltip"
            assert max(len(line) for line in tt.split("\n")) <= 52
        dlg.deleteLater()
