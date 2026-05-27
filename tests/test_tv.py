"""Tests for probeflow.processing.tv_denoise."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing import tv_denoise


@pytest.fixture
def noisy_terrace():
    """Two flat terraces separated by a step, with additive Gaussian noise."""
    rng = np.random.default_rng(0)
    clean = np.zeros((64, 64), dtype=np.float64)
    clean[:, 32:] = 1.0
    noise = rng.normal(scale=0.2, size=clean.shape)
    return clean + noise, clean


class TestTvDenoiseHuberROF:
    def test_reduces_rms_vs_input(self, noisy_terrace):
        noisy, clean = noisy_terrace
        denoised = tv_denoise(noisy, method="huber_rof", lam=0.2, max_iter=200)
        in_rms = float(np.sqrt(np.mean((noisy - clean) ** 2)))
        out_rms = float(np.sqrt(np.mean((denoised - clean) ** 2)))
        assert out_rms < in_rms

    def test_preserves_shape(self, noisy_terrace):
        noisy, _ = noisy_terrace
        denoised = tv_denoise(noisy, method="huber_rof", max_iter=50)
        assert denoised.shape == noisy.shape

    def test_preserves_edge_location(self, noisy_terrace):
        noisy, _ = noisy_terrace
        denoised = tv_denoise(noisy, method="huber_rof", lam=0.5, max_iter=200)
        # The step is around x=32. Find max of absolute x-gradient.
        gx = np.abs(np.diff(denoised.mean(axis=0)))
        assert 28 <= int(np.argmax(gx)) <= 35


class TestTvDenoiseTvL1:
    def test_tv_l1_runs_and_reduces_rms(self, noisy_terrace):
        noisy, clean = noisy_terrace
        denoised = tv_denoise(noisy, method="tv_l1", lam=0.5, max_iter=200)
        in_rms = float(np.sqrt(np.mean((noisy - clean) ** 2)))
        out_rms = float(np.sqrt(np.mean((denoised - clean) ** 2)))
        assert out_rms < in_rms


class TestTvDenoiseErrors:
    def test_non_2d_raises(self):
        with pytest.raises(ValueError):
            tv_denoise(np.zeros((4, 4, 4)), method="huber_rof")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            tv_denoise(np.zeros((8, 8)), method="nope")

    def test_bad_nabla_raises(self):
        with pytest.raises(ValueError):
            tv_denoise(np.zeros((8, 8)), method="huber_rof", nabla_comp="z")


class TestIterationContract:
    """Regression for review numerical #2 — tv_denoise must respect
    max_iter as a hard cap (no off-by-one) and must honour tol even for
    max_iter < 50."""

    def test_max_iter_is_a_hard_cap(self, monkeypatch, noisy_terrace):
        """The loop must run at most ``max_iter`` iterations.  Before
        the fix it ran ``max_iter + 1``."""
        from probeflow.processing import tv
        noisy, _ = noisy_terrace
        call_count = [0]
        original = tv._nabla_T_apply

        def counting(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(tv, "_nabla_T_apply", counting)
        tv_denoise(noisy, method="huber_rof", max_iter=7, lam=0.05)
        # Each iteration calls _nabla_T_apply once.  Convergence is
        # very unlikely on noise within 7 iterations, so the loop
        # should run the full max_iter=7 times.
        assert call_count[0] == 7, (
            f"Expected at most 7 iterations (max_iter=7), counted "
            f"{call_count[0]}"
        )

    def test_max_iter_zero_runs_no_iterations(self, monkeypatch, noisy_terrace):
        """max_iter=0 must not run any iterations."""
        from probeflow.processing import tv
        noisy, _ = noisy_terrace
        call_count = [0]
        original = tv._nabla_T_apply

        def counting(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(tv, "_nabla_T_apply", counting)
        tv_denoise(noisy, method="huber_rof", max_iter=0)
        assert call_count[0] == 0

    def test_tol_honoured_below_50_iterations(self, monkeypatch):
        """For max_iter < 50, the convergence check must still fire
        (every 10 iterations after a 5-iter warmup) so a converged
        denoise stops early instead of always running the full cap."""
        from probeflow.processing import tv
        # A constant image converges to itself immediately.
        flat = np.full((16, 16), 5.0, dtype=np.float64)

        call_count = [0]
        original = tv._nabla_T_apply

        def counting(*args, **kwargs):
            call_count[0] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(tv, "_nabla_T_apply", counting)
        tv_denoise(flat, method="huber_rof", max_iter=30,
                   tol=1e-3, lam=0.05)
        # Constant input → RMSE between iterates is essentially 0;
        # convergence must trigger by the first check (iter 10), so
        # the loop should NOT run all 30 iterations.
        assert call_count[0] < 30, (
            f"tol was not honoured for max_iter<50; loop ran "
            f"{call_count[0]} of 30 iterations"
        )


class TestNablaDirectional:
    def test_nabla_x_reduces_horizontal_streaks(self):
        """x-only gradient should preferentially smooth vertical scratches."""
        rng = np.random.default_rng(1)
        img = rng.normal(scale=0.02, size=(64, 64))
        # Add 4 vertical scratches
        for c in (10, 25, 40, 55):
            img[:, c] += 2.0
        denoised = tv_denoise(img, method="huber_rof", lam=0.3,
                              nabla_comp="x", max_iter=200)
        assert denoised.shape == img.shape
