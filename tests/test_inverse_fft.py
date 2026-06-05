"""Tests for probeflow.processing.inverse_fft — Fourier reconstruction backend."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.inverse_fft import (
    FourierEllipse,
    FourierRect,
    FourierStrokes,
    fourier_ellipse_mask,
    fourier_region_from_dict,
    fourier_region_mask,
    inverse_fft_filter,
    inverse_fft_from_mask,
)

N = 128
CY, CX = N // 2, N // 2
KX = 8                       # injected sine: 8 cycles across the image, along x


def _scene(amp=0.5):
    yy, xx = np.mgrid[:N, :N]
    base = np.exp(-(((xx - 40) ** 2 + (yy - 70) ** 2) / 400.0))
    sine = amp * np.sin(2 * np.pi * KX * xx / N)
    return base, sine, base + sine


def _bin_power(a):
    F = np.fft.fftshift(np.fft.fft2(a - a.mean()))
    return abs(F[CY, CX + KX]) + abs(F[CY, CX - KX])


def _sine_circle():
    return [FourierEllipse(dx=KX, dy=0, rx=3, ry=3)]


# ─── fourier_ellipse_mask ────────────────────────────────────────────────────

class TestMask:
    def test_circle_covers_centre_and_conjugate(self):
        m = fourier_ellipse_mask((N, N), _sine_circle())
        assert m[CY, CX + KX] == 1.0
        assert m[CY, CX - KX] == 1.0          # conjugate added automatically
        assert m[CY, CX] == 0.0               # DC untouched

    def test_no_conjugate_when_disabled(self):
        m = fourier_ellipse_mask((N, N), _sine_circle(), conjugate=False)
        assert m[CY, CX + KX] == 1.0
        assert m[CY, CX - KX] == 0.0

    def test_soft_edge_is_graded(self):
        hard = fourier_ellipse_mask((N, N), _sine_circle(), soft_px=0.0)
        soft = fourier_ellipse_mask((N, N), _sine_circle(), soft_px=3.0)
        assert set(np.unique(hard)) <= {0.0, 1.0}
        assert ((soft > 0.0) & (soft < 1.0)).any()   # intermediate values exist

    def test_ellipse_anisotropy(self):
        m = fourier_ellipse_mask((N, N), [FourierEllipse(dx=0, dy=0, rx=10, ry=2)])
        assert m[CY, CX + 6] == 1.0           # wide along x
        assert m[CY + 6, CX] == 0.0           # narrow along y


# ─── fourier_region_mask: rect / paint ───────────────────────────────────────

class TestRegionMask:
    def test_ellipse_wrapper_matches_region(self):
        m_old = fourier_ellipse_mask((N, N), _sine_circle())
        m_new = fourier_region_mask((N, N), _sine_circle())
        assert np.array_equal(m_old, m_new)

    def test_rect_covers_box_and_conjugate(self):
        r = FourierRect(dx=KX, dy=0, half_w=4, half_h=2)
        m = fourier_region_mask((N, N), [r])
        assert m[CY, CX + KX] == 1.0
        assert m[CY, CX - KX] == 1.0                 # conjugate
        assert m[CY, CX] == 0.0                      # DC untouched
        assert m[CY, CX + KX + 5] == 0.0            # outside the half-width (4)

    def test_square_is_symmetric(self):
        sq = FourierRect(dx=0, dy=0, half_w=5, half_h=5)
        m = fourier_region_mask((N, N), [sq])
        assert m[CY, CX + 5] == 1.0 and m[CY + 5, CX] == 1.0
        assert m[CY, CX + 7] == 0.0 and m[CY + 7, CX] == 0.0

    def test_strokes_cover_stamps_and_conjugate(self):
        s = FourierStrokes(stamps=((KX, 0), (KX + 2, 0)), radius=3)
        m = fourier_region_mask((N, N), [s])
        assert m[CY, CX + KX] == 1.0
        assert m[CY, CX - KX] == 1.0                 # conjugate stamp
        assert m[CY + 20, CX] == 0.0                 # far away is untouched

    def test_strokes_no_conjugate_when_disabled(self):
        s = FourierStrokes(stamps=((KX, 0),), radius=3)
        m = fourier_region_mask((N, N), [s], conjugate=False)
        assert m[CY, CX + KX] == 1.0
        assert m[CY, CX - KX] == 0.0

    def test_soft_edge_graded_rect_and_strokes(self):
        rect = fourier_region_mask((N, N), [FourierRect(0, 0, 5, 5)], soft_px=3.0)
        strokes = fourier_region_mask((N, N), [FourierStrokes(((0, 0),), 5)], soft_px=3.0)
        for m in (rect, strokes):
            assert ((m > 0.0) & (m < 1.0)).any()

    def test_region_from_dict_kinds(self):
        assert isinstance(fourier_region_from_dict({"kind": "rect", "half_w": 2, "half_h": 3}),
                          FourierRect)
        assert isinstance(fourier_region_from_dict({"kind": "paint", "stamps": [[1, 2]], "radius": 2}),
                          FourierStrokes)
        # legacy dict without a kind reads as an ellipse
        assert isinstance(fourier_region_from_dict({"dx": 1, "dy": 0, "rx": 2, "ry": 2}),
                          FourierEllipse)

    def test_mixed_regions_via_filter(self):
        _b, _s, img = _scene()
        regions = [
            FourierEllipse(dx=KX, dy=0, rx=3, ry=3),
            FourierRect(dx=20, dy=10, half_w=2, half_h=2),
            FourierStrokes(stamps=((-15, -5),), radius=2),
        ]
        out = inverse_fft_filter(img, regions, mode="remove_selected")
        assert out.shape == img.shape and np.isfinite(out).all()
        assert _bin_power(out) < 1e-6 * _bin_power(img)   # the sine ellipse still removed


# ─── inverse_fft_from_mask ───────────────────────────────────────────────────

class TestInverse:
    def test_remove_drops_target_bin_keeps_rest(self):
        _base, _sine, img = _scene()
        mask = fourier_ellipse_mask((N, N), _sine_circle())
        res = inverse_fft_from_mask(img, mask, mode="remove_selected")
        assert _bin_power(res.result) < 1e-6 * _bin_power(img)   # sine gone
        # The rest of the image survives (correlate with the base bump).
        base = _base = _scene()[0]
        assert np.corrcoef(res.result.ravel(), base.ravel())[0, 1] > 0.9

    def test_keep_reconstructs_the_sine(self):
        _base, sine, img = _scene()
        mask = fourier_ellipse_mask((N, N), _sine_circle())
        res = inverse_fft_from_mask(img, mask, mode="keep_selected")
        assert np.corrcoef(res.result.ravel(), sine.ravel())[0, 1] > 0.99

    def test_conjugate_symmetric_is_real(self):
        _b, _s, img = _scene()
        sym = fourier_ellipse_mask((N, N), _sine_circle(), conjugate=True)
        asym = fourier_ellipse_mask((N, N), _sine_circle(), conjugate=False)
        assert inverse_fft_from_mask(img, sym, mode="remove_selected").imag_residual_norm < 1e-9
        assert inverse_fft_from_mask(img, asym, mode="remove_selected").imag_residual_norm > 1e-3

    def test_dc_preserved_on_remove(self):
        _b, _s, img = _scene()
        mask = fourier_ellipse_mask((N, N), _sine_circle())
        res = inverse_fft_from_mask(img, mask, mode="remove_selected")
        assert np.nanmean(res.result) == pytest.approx(img.mean(), abs=1e-9)

    def test_residual_is_original_minus_result_and_is_removed_component(self):
        _b, sine, img = _scene()
        mask = fourier_ellipse_mask((N, N), _sine_circle())
        res = inverse_fft_from_mask(img, mask, mode="remove_selected")
        assert np.allclose(res.residual, img - res.result)
        # In remove mode the residual IS the removed periodic component.
        assert np.corrcoef(res.residual.ravel(), sine.ravel())[0, 1] > 0.99

    def test_nan_preserved(self):
        _b, _s, img = _scene()
        img = img.copy()
        img[10:14, 20:24] = np.nan
        mask = fourier_ellipse_mask((N, N), _sine_circle())
        res = inverse_fft_from_mask(img, mask, mode="remove_selected")
        assert np.all(np.isnan(res.result[10:14, 20:24]))
        assert np.isfinite(res.result[60, 60])

    def test_deterministic(self):
        _b, _s, img = _scene()
        mask = fourier_ellipse_mask((N, N), _sine_circle())
        r1 = inverse_fft_from_mask(img, mask, mode="remove_selected").result
        r2 = inverse_fft_from_mask(img, mask, mode="remove_selected").result
        assert np.array_equal(r1, r2)

    def test_validation(self):
        _b, _s, img = _scene()
        with pytest.raises(ValueError):
            inverse_fft_from_mask(np.zeros((4, 4, 2)), np.zeros((4, 4, 2)))
        with pytest.raises(ValueError):
            inverse_fft_from_mask(img, np.zeros((N, N // 2)))
        with pytest.raises(ValueError):
            inverse_fft_from_mask(img, np.zeros((N, N)), mode="bogus")


# ─── inverse_fft_filter (op entry point) ─────────────────────────────────────

class TestFilterEntry:
    def test_returns_result_and_empty_is_noop(self):
        _b, _s, img = _scene()
        out = inverse_fft_filter(img, _sine_circle(), mode="remove_selected")
        assert out.shape == img.shape and _bin_power(out) < 1e-6 * _bin_power(img)
        assert np.array_equal(inverse_fft_filter(img, []), img)


class TestProcessingStateOp:
    def test_op_dispatches_and_removes_sine(self):
        from probeflow.core.processing_state import ProcessingState, ProcessingStep
        from probeflow.processing.state import apply_processing_state

        _b, _s, img = _scene()
        params = {
            "selections": [{"dx": KX, "dy": 0, "rx": 3.0, "ry": 3.0, "angle_deg": 0.0}],
            "mode": "remove_selected", "conjugate_symmetric": True, "soft_px": 0.0,
        }
        state = ProcessingState(steps=[ProcessingStep("inverse_fft_filter", params)])
        out = apply_processing_state(img, state)
        assert _bin_power(out) < 1e-6 * _bin_power(img)
        # Provenance round-trip.
        d = state.to_dict()["steps"][0]
        assert d["op"] == "inverse_fft_filter"
        assert d["params"]["selections"][0]["dx"] == KX

    def test_gui_apply_path_persists(self):
        """Regression: the FFT viewer's Apply stores the op under
        _processing["geometric_ops"]; processing_state_from_gui must emit it
        (it was silently dropped by the geometric-ops allowlist, so Apply did
        nothing and closing the viewer left the image unchanged)."""
        from probeflow.processing.gui_adapter import processing_state_from_gui
        from probeflow.processing.state import apply_processing_state

        _b, _s, img = _scene()
        gui_state = {
            "geometric_ops": [{
                "op": "inverse_fft_filter",
                "params": {
                    "selections": [{"dx": KX, "dy": 0, "rx": 3.0, "ry": 3.0, "angle_deg": 0.0}],
                    "mode": "remove_selected", "conjugate_symmetric": True,
                    "soft_px": 0.0, "fft_source": "whole_image",
                },
            }],
        }
        state = processing_state_from_gui(gui_state)
        assert [s.op for s in state.steps] == ["inverse_fft_filter"]
        out = apply_processing_state(img, state)
        assert _bin_power(out) < 1e-6 * _bin_power(img)
