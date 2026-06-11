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
        from matplotlib.lines import Line2D

        dlg = _dialog(qapp)
        assert dlg._mains_artists == []
        dlg._mains_overlay_cb.setChecked(True)
        lines = [a for a in dlg._mains_artists if isinstance(a, Line2D)]
        assert len(lines) == 6                     # 3 harmonics × 2 conjugates
        dlg._mains_overlay_cb.setChecked(False)
        assert dlg._mains_artists == []
        dlg.deleteLater()

    def test_overlay_obeys_minimum_q_floor(self, qapp):
        dlg = _dialog(qapp)
        dlg._mains_auto_cb.setChecked(False)
        dlg._mains_harm_spin.setValue(1)
        dlg._mains_min_q_spin.setValue(3.0)
        dlg._mains_overlay_cb.setChecked(True)

        from matplotlib.lines import Line2D

        lines = [a for a in dlg._mains_artists if isinstance(a, Line2D)]
        assert len(lines) == 4
        gap = np.sqrt(3.0 ** 2 - 2.5 ** 2)
        for art in lines:
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


class TestCustomStreaksAndWidthViz:
    def test_overlay_includes_width_bands_that_track_radius(self, qapp):
        """Each overlay line carries a translucent band of half-width =
        notch radius; changing the radius rebuilds the overlay (the control
        previously had no visible effect)."""
        from matplotlib.collections import PolyCollection

        dlg = _dialog(qapp)
        dlg._mains_overlay_cb.setChecked(True)
        dlg._on_mains_changed()
        bands = [a for a in dlg._mains_artists if isinstance(a, PolyCollection)]
        assert bands, "no width bands drawn for the mains overlay"

        n_before = len(dlg._mains_artists)
        dlg._mains_radius_spin.setValue(8)   # triggers _on_mains_changed
        assert len(dlg._mains_artists) == n_before, "overlay not rebuilt"
        wide = [a for a in dlg._mains_artists if isinstance(a, PolyCollection)]
        assert wide, "bands vanished after radius change"
        dlg.deleteLater()

    def test_add_drag_and_remove_custom_pair(self, qapp):
        from types import SimpleNamespace

        dlg = _dialog(qapp)
        dlg._tab_widget.setCurrentIndex(dlg._mains_tab_index)
        dlg._on_mains_add_streak()
        assert len(dlg._mains_custom_streaks()) == 1
        q0 = dlg._mains_custom_streaks()[0]
        assert dlg._mains_remove_streak_btn.isEnabled()

        # Drag: press near the line, move to a new q, release.
        press = SimpleNamespace(inaxes=dlg._ax_fft, button=1, xdata=q0, ydata=0.0)
        assert dlg._mains_handle_press(press) is True
        move = SimpleNamespace(inaxes=dlg._ax_fft, button=1, xdata=q0 + 1.0, ydata=0.0)
        assert dlg._mains_handle_motion(move) is True
        assert dlg._mains_custom_streaks()[0] == pytest.approx(q0 + 1.0)
        assert dlg._mains_handle_release(SimpleNamespace()) is True

        # A press far from any line must not start a drag (panning wins).
        far = SimpleNamespace(inaxes=dlg._ax_fft, button=1,
                              xdata=q0 + 5.0, ydata=0.0)
        assert dlg._mains_handle_press(far) is False

        dlg._on_mains_remove_streak()
        assert dlg._mains_custom_streaks() == []
        assert not dlg._mains_remove_streak_btn.isEnabled()
        dlg.deleteLater()

    def test_params_carry_custom_streaks_and_fill(self, qapp):
        captured: dict = {}
        dlg = _dialog(qapp, captured=captured)
        dlg._tab_widget.setCurrentIndex(dlg._mains_tab_index)
        dlg._on_mains_add_streak()
        dlg._mains_fill_cb.setChecked(True)

        params = dlg._mains_op_params()
        assert params["extra_streaks_px"], "custom pair missing from params"
        assert all(isinstance(v, int) and v > 0 for v in params["extra_streaks_px"])
        assert params["notch_fill"] == "background"

        dlg._on_mains_apply()
        assert captured["op"] == "mains_pickup_suppression"
        assert captured["params"]["extra_streaks_px"] == params["extra_streaks_px"]
        dlg.deleteLater()

    def test_custom_pair_enables_preview_and_apply_without_scan_speed(self, qapp):
        captured: dict = {}
        dlg = _dialog(qapp, scan_speed=None, captured=captured)
        dlg._mains_speed_spin.setValue(0.0)
        dlg._tab_widget.setCurrentIndex(dlg._mains_tab_index)

        dlg._on_mains_apply()
        assert not captured, "apply must be blocked with no speed and no pairs"

        dlg._on_mains_add_streak()
        dlg._on_mains_apply()
        assert captured["op"] == "mains_pickup_suppression"
        assert captured["params"]["scan_speed_m_per_s"] is None
        assert captured["params"]["extra_streaks_px"]
        dlg.deleteLater()

    def test_fft_auto_contrast_cycles_presets(self, qapp):
        dlg = _dialog(qapp)
        seen = []
        for _ in range(4):
            dlg._reset_intensity()
            state = dlg._fft_drs._state
            seen.append((state.low_pct, state.high_pct))
        assert len(set(seen[:3])) == 3, "Auto must change the range each click"
        assert seen[3] == seen[0], "cycle must wrap back to the full range"
        dlg.deleteLater()
