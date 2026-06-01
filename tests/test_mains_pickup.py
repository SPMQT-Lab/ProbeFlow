"""Tests for probeflow.processing.mains_pickup — mains-pickup prediction/removal.

Physics: a mains tone at f Hz completes f·T_line cycles per scan line, so it
appears in the FFT at index round(f·T_line) from DC along the fast axis, i.e. at
q = f/v where v = scan_width/T_line is the fast-scan tip speed.
"""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.mains_pickup import (
    equivalent_frequency_hz,
    estimate_fast_scan_speed_m_per_s,
    mains_pickup_suppression,
    predict_mains_fft_positions,
)

# A representative slow scan: 10 nm wide, 160 px, 0.5 s/line → v = 2e-8 m/s.
W_M = 10e-9
NPX = 160
V = 2.0e-8                 # m/s  (T_line = W/v = 0.5 s)
T_LINE = W_M / V


# ─── estimate_fast_scan_speed_m_per_s ────────────────────────────────────────

class TestEstimateSpeed:
    def test_nanonis_explicit_speed(self):
        hdr = {"Scan>speed forw. (m/s)": "7.71605E-9", "Scan>speed backw. (m/s)": "7.71605E-9"}
        assert estimate_fast_scan_speed_m_per_s(hdr) == pytest.approx(7.71605e-9)

    def test_nanonis_from_line_time_and_range(self):
        # SCAN_TIME is per-line seconds; SCAN_RANGE is width (multi-field strings).
        hdr = {"SCAN_TIME": "1.296E+0   1.296E+0", "SCAN_RANGE": "1.0E-8   1.0E-8"}
        v = estimate_fast_scan_speed_m_per_s(hdr, scan_range_m=(1e-8, 1e-8))
        assert v == pytest.approx(1e-8 / 1.296)

    def test_createc_line_time(self):
        hdr = {"Sec/line:": "0.328", "Sec/Image:": "335.544"}
        v = estimate_fast_scan_speed_m_per_s(hdr, scan_range_m=(1.125e-8, 1.125e-8))
        assert v == pytest.approx(1.125e-8 / 0.328)

    def test_createc_frame_time_fallback(self):
        # No per-line key → frame_time / rows.
        hdr = {"Sec/Image:": "320.0"}
        v = estimate_fast_scan_speed_m_per_s(
            hdr, scan_range_m=(1e-8, 1e-8), image_shape=(160, 160))
        assert v == pytest.approx(1e-8 / (320.0 / 160))

    def test_missing_returns_none(self):
        assert estimate_fast_scan_speed_m_per_s({"Bias": "1.0"}, scan_range_m=(1e-8, 1e-8)) is None
        assert estimate_fast_scan_speed_m_per_s(None) is None

    @pytest.mark.parametrize("fname", [
        "sxm_moire_10nm.sxm",            # Nanonis: Scan>speed forw. (m/s)
        "createc_scan_atomic_11nm.dat",  # Createc: Sec/line:
    ])
    def test_real_sample_headers(self, fname):
        from pathlib import Path
        from probeflow.core.scan_loader import load_scan

        path = Path(__file__).parent.parent / "test_data" / fname
        if not path.exists():
            pytest.skip(f"test_data/{fname} not available")
        scan = load_scan(str(path))
        v = estimate_fast_scan_speed_m_per_s(
            scan.header or {}, scan_range_m=scan.scan_range_m,
            image_shape=scan.planes[0].shape,
        )
        assert v is not None and v > 0, "fast-scan speed should parse from the header"


# ─── predict_mains_fft_positions ─────────────────────────────────────────────

class TestPredict:
    def test_fundamental_and_harmonics(self):
        preds = predict_mains_fft_positions(NPX, W_M, V, mains_frequency_hz=50.0, harmonics=4)
        # n=1→25, n=2→50, n=3→75 ; n=4 (q=10 nm⁻¹) exceeds Nyquist (8) → dropped.
        assert [p["fft_index"] for p in preds] == [25, 50, 75]
        assert preds[0]["q_nm_inv"] == pytest.approx(2.5)
        assert preds[0]["dx"] == 25 and preds[0]["dy"] == 0

    def test_auto_harmonics_runs_to_fft_limit(self):
        preds = predict_mains_fft_positions(NPX, W_M, V, mains_frequency_hz=50.0, harmonics=None)
        assert [p["fft_index"] for p in preds] == [25, 50, 75]

    def test_fft_index_equals_f_times_line_time(self):
        preds = predict_mains_fft_positions(NPX, W_M, V, mains_frequency_hz=50.0, harmonics=1)
        assert preds[0]["fft_index"] == round(50.0 * T_LINE)

    def test_sixty_hz_differs_from_fifty(self):
        p50 = predict_mains_fft_positions(NPX, W_M, V, mains_frequency_hz=50.0, harmonics=1)[0]
        p60 = predict_mains_fft_positions(NPX, W_M, V, mains_frequency_hz=60.0, harmonics=1)[0]
        assert p60["fft_index"] == round(60.0 * T_LINE) == 30
        assert p60["fft_index"] != p50["fft_index"]

    def test_fast_axis_swap(self):
        px = predict_mains_fft_positions(NPX, W_M, V, fast_axis="x", harmonics=1)[0]
        py = predict_mains_fft_positions(NPX, W_M, V, fast_axis="y", harmonics=1)[0]
        assert (px["dx"], px["dy"]) == (25, 0)
        assert (py["dx"], py["dy"]) == (0, 25)

    def test_unknown_speed_returns_empty(self):
        assert predict_mains_fft_positions(NPX, W_M, None) == []
        assert predict_mains_fft_positions(NPX, W_M, 0.0) == []

    def test_equivalent_frequency_inverts_prediction(self):
        p = predict_mains_fft_positions(NPX, W_M, V, mains_frequency_hz=50.0, harmonics=1)[0]
        assert equivalent_frequency_hz(p["q_nm_inv"], V) == pytest.approx(50.0)
        assert equivalent_frequency_hz(2.5, None) is None


# ─── mains_pickup_suppression ────────────────────────────────────────────────

def _scan_with_mains(f_hz=50.0, amp=0.03e-9, phase_drift=0.0, seed=0):
    """Synthetic scan: smooth bumps + injected mains tone along the fast (x) axis."""
    yy, xx = np.mgrid[:NPX, :NPX]
    base = (0.2e-9 * np.exp(-(((xx - 50) ** 2 + (yy - 60) ** 2) / 300))
            + 0.15e-9 * np.exp(-(((xx - 110) ** 2 + (yy - 90) ** 2) / 400)))
    dt_px = T_LINE / NPX
    t = yy * T_LINE + xx * dt_px
    rng = np.random.default_rng(seed)
    drift = phase_drift * rng.standard_normal(NPX)[:, None]  # per-row phase jitter
    mains = amp * np.sin(2 * np.pi * f_hz * t + drift)
    return base + mains + rng.normal(0, 2e-12, (NPX, NPX))


def _column_power(a, col, halfw=2):
    m = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean())))
    return m[:, col - halfw:col + halfw + 1].sum()


def _fft_pixel_power(a, dx, dy, halfw=1):
    m = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean())))
    cy, cx = np.array(m.shape) // 2
    y0, y1 = cy + dy - halfw, cy + dy + halfw + 1
    x0, x1 = cx + dx - halfw, cx + dx + halfw + 1
    return m[y0:y1, x0:x1].sum()


def _vertical_streak_component(dx=25, dy=30):
    yy, xx = np.mgrid[:NPX, :NPX]
    return np.sin(2 * np.pi * dx * xx / NPX) * np.cos(2 * np.pi * dy * yy / NPX)


def _horizontal_streak_component(dy=25, dx=30):
    yy, xx = np.mgrid[:NPX, :NPX]
    return np.sin(2 * np.pi * dy * yy / NPX) * np.cos(2 * np.pi * dx * xx / NPX)


class TestSuppression:
    def test_suppresses_injected_mains_preserves_rest(self):
        img = _scan_with_mains()
        out = mains_pickup_suppression(
            img, scan_speed_m_per_s=V, scan_range_m=(W_M, W_M),
            mains_frequency_hz=50.0, harmonics=1, notch_radius_px=3.0,
        )
        col = NPX // 2 + 25
        before, after = _column_power(img, col), _column_power(out, col)
        assert after < 0.5 * before, "mains peak power must drop substantially"
        # Total FFT power largely preserved (only the notch removed).
        tot = lambda a: float(np.abs(np.fft.fft2(a - a.mean())).sum())
        assert tot(out) > 0.85 * tot(img)

    def test_unknown_speed_is_noop(self):
        img = _scan_with_mains()
        out = mains_pickup_suppression(img, scan_speed_m_per_s=None, scan_range_m=(W_M, W_M))
        assert np.array_equal(out, img.astype(np.float64))

    def test_deterministic(self):
        img = _scan_with_mains()
        kw = dict(scan_speed_m_per_s=V, scan_range_m=(W_M, W_M), harmonics=2)
        assert np.array_equal(mains_pickup_suppression(img, **kw),
                              mains_pickup_suppression(img, **kw))

    def test_snap_to_peak_when_prediction_off_by_one(self):
        # Inject mains one bin away from the nominal prediction; snapping should
        # still land the notch on the true peak and remove it.
        f_off = 50.0 * (26.0 / 25.0)   # → fft_index 26, prediction rounds to 25
        img = _scan_with_mains(f_hz=f_off)
        col_true = NPX // 2 + 26
        out = mains_pickup_suppression(
            img, scan_speed_m_per_s=V, scan_range_m=(W_M, W_M),
            mains_frequency_hz=50.0, harmonics=1, snap_window_px=2,
        )
        assert _column_power(out, col_true) < 0.6 * _column_power(img, col_true)

    def test_streak_mode_removes_off_axis_vertical_energy(self):
        img = _vertical_streak_component(dx=25, dy=30)
        before = _fft_pixel_power(img, dx=25, dy=30)
        legacy = mains_pickup_suppression(
            img, scan_speed_m_per_s=V, scan_range_m=(W_M, W_M),
            mains_frequency_hz=50.0, harmonics=1, snap_window_px=0,
        )
        out = mains_pickup_suppression(
            img, scan_speed_m_per_s=V, scan_range_m=(W_M, W_M),
            mains_frequency_hz=50.0, harmonics=1, snap_window_px=0,
            notch_shape="streak", notch_radius_px=2.0,
        )

        assert _fft_pixel_power(legacy, dx=25, dy=30) > 0.85 * before
        assert _fft_pixel_power(out, dx=25, dy=30) < 0.2 * before

    def test_streak_mode_respects_radial_min_q(self):
        img = _vertical_streak_component(dx=25, dy=0) + _vertical_streak_component(dx=25, dy=30)
        low_before = _fft_pixel_power(img, dx=25, dy=0)
        high_before = _fft_pixel_power(img, dx=25, dy=30)
        out = mains_pickup_suppression(
            img, scan_speed_m_per_s=V, scan_range_m=(W_M, W_M),
            mains_frequency_hz=50.0, harmonics=1, snap_window_px=0,
            notch_shape="streak", notch_radius_px=2.0, min_q_nm_inv=3.0,
        )

        assert _fft_pixel_power(out, dx=25, dy=0) > 0.85 * low_before
        assert _fft_pixel_power(out, dx=25, dy=30) < 0.2 * high_before

    def test_streak_mode_handles_vertical_fast_axis(self):
        img = _horizontal_streak_component(dy=25, dx=30)
        before = _fft_pixel_power(img, dx=30, dy=25)
        out = mains_pickup_suppression(
            img, scan_speed_m_per_s=V, scan_range_m=(W_M, W_M),
            mains_frequency_hz=50.0, harmonics=1, snap_window_px=0,
            notch_shape="streak", notch_radius_px=2.0, fast_axis="y",
        )

        assert _fft_pixel_power(out, dx=30, dy=25) < 0.2 * before

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            mains_pickup_suppression(np.zeros((4, 4, 2)),
                                     scan_speed_m_per_s=V, scan_range_m=(W_M, W_M))


# ─── ProcessingState op + provenance ─────────────────────────────────────────

class TestProcessingStateOp:
    def test_op_registered_and_applies(self):
        from probeflow.core.processing_state import ProcessingState, ProcessingStep
        from probeflow.processing.state import apply_processing_state

        params = {
            "scan_speed_m_per_s": V, "scan_range_m": (W_M, W_M),
            "mains_frequency_hz": 50.0, "harmonics": 1, "notch_radius_px": 3.0,
            "fast_axis": "x", "snap_window_px": 2,
        }
        state = ProcessingState(steps=[ProcessingStep("mains_pickup_suppression", params)])
        img = _scan_with_mains()
        out = apply_processing_state(img, state)
        col = NPX // 2 + 25
        assert _column_power(out, col) < 0.5 * _column_power(img, col)
        # Params round-trip through the canonical dict (provenance reproducibility).
        d = state.to_dict()
        step = d["steps"][0]
        assert step["op"] == "mains_pickup_suppression"
        assert step["params"]["mains_frequency_hz"] == 50.0

    def test_gui_apply_path_persists(self):
        """Regression: the Mains tab's Apply stores the op under
        _processing["geometric_ops"]; processing_state_from_gui must emit it
        (it was silently dropped by the geometric-ops allowlist)."""
        from probeflow.processing.gui_adapter import processing_state_from_gui
        from probeflow.processing.state import apply_processing_state

        gui_state = {"geometric_ops": [{
            "op": "mains_pickup_suppression",
            "params": {
                "scan_speed_m_per_s": V, "scan_range_m": [W_M, W_M],
                "mains_frequency_hz": 50.0, "harmonics": 1, "notch_radius_px": 3.0,
                "fast_axis": "x", "snap_window_px": 2,
            },
        }]}
        state = processing_state_from_gui(gui_state)
        assert [s.op for s in state.steps] == ["mains_pickup_suppression"]
        img = _scan_with_mains()
        out = apply_processing_state(img, state)
        assert _column_power(out, NPX // 2 + 25) < 0.5 * _column_power(img, NPX // 2 + 25)
