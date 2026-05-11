"""Physics-informed first-pass tests for ProbeFlow processing routines.

Synthetic arrays only.  No GUI.  No external files.  Deterministic.

Each test documents the physical motivation for its assertions so that a
future reader understands *why* the tolerance is what it is, not just
*what* is being checked.
"""
from __future__ import annotations

import numpy as np

from probeflow.processing import (
    edge_detect,
    gaussian_smooth,
    measure_periodicity,
    subtract_background,
)


# ── synthetic image helpers ───────────────────────────────────────────────────

def _tilted_plane(
    Ny: int, Nx: int,
    *,
    slope_x: float = 0.1,
    slope_y: float = 0.05,
    offset: float = 5.0,
) -> np.ndarray:
    """z = slope_x * col + slope_y * row + offset  (origin at top-left)."""
    Y, X = np.mgrid[:Ny, :Nx]
    return (slope_x * X + slope_y * Y + offset).astype(np.float64)


def _step_terrace(
    Ny: int, Nx: int,
    *,
    step_col: int,
    step_height: float = 2.0,
) -> np.ndarray:
    """Two flat terraces; columns >= step_col are raised by step_height."""
    arr = np.zeros((Ny, Nx), dtype=np.float64)
    arr[:, step_col:] = step_height
    return arr


def _sine_wave(
    Ny: int, Nx: int,
    *,
    fx: float = 0.0,
    fy: float = 0.0,
    amp: float = 1.0,
) -> np.ndarray:
    """z = amp * sin(2π*(fx*col + fy*row)), frequencies in cycles/pixel."""
    Y, X = np.mgrid[:Ny, :Nx]
    return (amp * np.sin(2.0 * np.pi * (fx * X + fy * Y))).astype(np.float64)


# ── polynomial background subtraction ────────────────────────────────────────

class TestPlaneSubtractionPhysics:
    """subtract_background must be exact on its own polynomial basis."""

    def test_removes_linear_tilt_to_machine_precision(self):
        """A pure tilted plane (z = ax + by + c) is removed exactly.

        An STM image of a flat terrace on a tilted sample looks like a linear
        plane.  subtract_background(order=1) spans that basis exactly, so the
        least-squares residual must be zero to floating-point precision.
        Residual > 1e-8 points to a bug in _poly_terms or in the coordinate
        normalisation.
        """
        arr = _tilted_plane(64, 64, slope_x=0.13, slope_y=0.07, offset=4.2)
        out = subtract_background(arr, order=1)
        assert float(np.ptp(out)) < 1e-8, (
            f"Peak-to-peak residual {np.ptp(out):.2e} after order-1 removal; "
            "expected < 1e-8."
        )

    def test_step_height_preserved_when_background_fitted_on_one_terrace(self):
        """Fitting the background on the lower terrace must not alter step height.

        Physical context: STM images of stepped metal surfaces have atomic steps
        of known height (~2 Å for Cu(111)).  Subtracting a plane fitted only on
        the lower terrace extrapolates that plane under the upper terrace; the
        step height recovered from the residual must equal the true step height
        exactly.
        """
        Ny, Nx = 64, 64
        step_col = Nx // 2
        step_h = 2.0

        arr = (
            _step_terrace(Ny, Nx, step_col=step_col, step_height=step_h)
            + _tilted_plane(Ny, Nx, slope_x=0.08, slope_y=0.03, offset=0.0)
        )

        lower_mask = np.zeros((Ny, Nx), dtype=bool)
        lower_mask[:, :step_col] = True
        out = subtract_background(arr, order=1, fit_mask=lower_mask)

        lower_median = float(np.median(out[:, :step_col]))
        upper_median = float(np.median(out[:, step_col:]))
        recovered_step = upper_median - lower_median

        assert abs(recovered_step - step_h) < 1e-8, (
            f"Recovered step height {recovered_step:.10f}, "
            f"true step height {step_h}; difference {abs(recovered_step - step_h):.2e}."
        )


# ── Gaussian smoothing ────────────────────────────────────────────────────────

class TestGaussianSmoothPhysics:
    """Gaussian smooth must be isotropic, intensity-conserving, and Fourier-correct."""

    def test_impulse_peak_stays_at_centre_and_intensity_conserved(self):
        """Impulse response must peak at the delta site and conserve total weight.

        The Gaussian kernel is normalised to unit sum, so it redistributes
        intensity without creating or destroying it.  A shifted peak indicates
        a non-centred or asymmetric kernel; a changed sum indicates wrong
        normalisation.
        """
        Ny, Nx = 64, 64
        cy, cx = Ny // 2, Nx // 2
        arr = np.zeros((Ny, Nx))
        arr[cy, cx] = 1.0

        out = gaussian_smooth(arr, sigma_px=4.0)

        peak = np.unravel_index(np.argmax(out), out.shape)
        assert peak == (cy, cx), (
            f"Peak moved to {peak}; expected ({cy}, {cx}).  "
            "The kernel may not be centred."
        )
        assert abs(out.sum() - 1.0) < 1e-10, (
            f"Total intensity changed from 1.0 to {out.sum():.12f}.  "
            "Kernel normalisation is broken."
        )

    def test_sinusoidal_corrugation_attenuated_by_fourier_prediction(self):
        """Amplitude of a sine wave must be suppressed by exp(−2π²σ²f²).

        Physical context: atomic corrugation is a periodic signal.  Gaussian
        smoothing with σ px attenuates spatial frequency f (cycles/px) by
        exp(−2π²σ²f²).  A σ = 3 px filter with f = 1/20 cy/px predicts an
        attenuation factor of ≈ 0.640.

        std of a pure sine equals amp/√2, so amplitude can be measured from
        the standard deviation without knowing the phase.  Tolerance 5 %
        accounts for boundary effects with the reflect padding.

        A factor-of-2 error in sigma (e.g. sigma used as diameter) changes
        the predicted attenuation by a factor of 4, far outside this tolerance.
        """
        Ny, Nx = 256, 256
        period_px = 20.0
        sigma_px = 3.0
        fx = 1.0 / period_px

        arr = _sine_wave(Ny, Nx, fx=fx, fy=0.0, amp=1.0)
        out = gaussian_smooth(arr, sigma_px=sigma_px)

        amp_in  = float(np.std(arr)) * np.sqrt(2.0)
        amp_out = float(np.std(out)) * np.sqrt(2.0)
        expected = np.exp(-2.0 * np.pi**2 * sigma_px**2 * fx**2)
        actual   = amp_out / amp_in

        assert abs(actual - expected) < 0.05, (
            f"Measured attenuation {actual:.4f}, "
            f"Fourier prediction {expected:.4f} (diff {abs(actual - expected):.4f}).  "
            "sigma may be applied as FWHM or diameter instead of std-dev."
        )


# ── LoG / edge detection ──────────────────────────────────────────────────────

class TestLoGPhysics:
    """LoG must give zero response on featureless inputs and localise steps correctly."""

    def test_log_on_constant_image_is_zero(self):
        """LoG of a spatially uniform image must be identically zero.

        Physical context: a featureless surface has no edges.  The Laplacian of
        a (Gaussian-smoothed) constant is zero by construction; any non-zero
        response indicates a normalisation or boundary artefact.
        """
        arr = np.full((48, 48), 3.7)
        out = edge_detect(arr, method='log', sigma=2.0)
        assert np.allclose(out, 0.0, atol=1e-9), (
            f"LoG of constant image: max |response| = {np.abs(out).max():.2e}; "
            "expected 0."
        )

    def test_dog_on_constant_image_is_zero(self):
        """DoG of a constant must be zero.

        DoG = G(σ₁) ∗ f − G(σ₂) ∗ f.  For constant f both terms equal f, so
        the difference is zero.
        """
        arr = np.full((48, 48), 3.7)
        out = edge_detect(arr, method='dog', sigma=2.0, sigma2=4.0)
        assert np.allclose(out, 0.0, atol=1e-9), (
            f"DoG of constant image: max |response| = {np.abs(out).max():.2e}; "
            "expected 0."
        )

    def test_log_on_tilted_plane_is_zero_in_interior(self):
        """LoG of a linear tilt must vanish away from the image boundary.

        Physical context: scanner tilt is the most common STM artefact.  The
        Laplacian of a linear function is analytically zero, and Gaussian
        smoothing preserves linearity for interior pixels, so LoG must not
        generate spurious edges from a featureless tilt.

        A margin of 5σ is used so that reflected-boundary contamination of the
        Gaussian pre-smoothing step is negligible (< 1e-6).
        """
        Ny, Nx = 64, 64
        sigma = 2.0
        arr = _tilted_plane(Ny, Nx, slope_x=0.2, slope_y=0.1, offset=10.0)
        out = edge_detect(arr, method='log', sigma=sigma)

        margin = int(np.ceil(5.0 * sigma))
        interior = out[margin:-margin, margin:-margin]
        assert np.allclose(interior, 0.0, atol=1e-6), (
            f"LoG of tilted plane: max interior |response| = "
            f"{np.abs(interior).max():.2e}; expected < 1e-6."
        )

    def test_log_detects_step_edge_within_two_sigma_of_true_position(self):
        """LoG must produce its peak response within 2σ of a synthetic step edge.

        Physical context: atomic steps appear as sharp height transitions.  The
        LoG response is the second derivative of the Gaussian-smoothed step:
        it has a positive lobe on the low side (at approximately col − σ) and
        a negative lobe on the high side (at approximately col + σ).  Both
        lobes must be located within 2σ + 2 px of the true step column.
        """
        Ny, Nx = 64, 64
        step_col = Nx // 2
        sigma = 3.0

        arr = _step_terrace(Ny, Nx, step_col=step_col, step_height=1.0)
        out = edge_detect(arr, method='log', sigma=sigma)

        profile    = out.mean(axis=0)
        peak_col   = int(np.argmax(profile))
        trough_col = int(np.argmin(profile))
        tol        = int(2 * sigma) + 2

        assert abs(peak_col - step_col) <= tol, (
            f"LoG positive lobe at col {peak_col}; step at {step_col}; "
            f"tolerance ±{tol}."
        )
        assert abs(trough_col - step_col) <= tol, (
            f"LoG negative lobe at col {trough_col}; step at {step_col}; "
            f"tolerance ±{tol}."
        )


# ── FFT / spatial frequency detection ────────────────────────────────────────

class TestMeasurePeriodicityPhysics:
    """measure_periodicity must recover spatial period and grating orientation."""

    def test_recovers_period_of_y_direction_grating_within_two_percent(self):
        """The dominant FFT peak must fall within 2 % of the true period.

        Physical context: surface reconstructions and moiré patterns produce
        periodic corrugations.  measure_periodicity searches the upper
        half-plane of the (Hanning-windowed) power spectrum; a y-direction
        grating is used so its peaks lie unambiguously in that half-plane.

        With Ny = 128 and period_px = 16, the peak falls exactly at integer
        FFT bin 8 from the centre, so frequency discretisation error is zero
        and the 2 % tolerance is achievable.
        """
        Ny, Nx = 128, 128
        period_px = 16
        px_m = 1e-9
        arr = _sine_wave(Ny, Nx, fx=0.0, fy=1.0 / period_px, amp=1.0)

        peaks = measure_periodicity(
            arr,
            pixel_size_x_m=px_m,
            pixel_size_y_m=px_m,
            n_peaks=3,
        )
        assert peaks, "No FFT peaks found for a pure sinusoidal grating."

        found    = peaks[0]["period_m"]
        expected = period_px * px_m
        rel_err  = abs(found - expected) / expected
        assert rel_err < 0.02, (
            f"Period {found * 1e9:.3f} nm, expected {expected * 1e9:.3f} nm "
            f"({rel_err * 100:.1f} % error).  "
            "Period–frequency conversion or fftshift coordinate may be wrong."
        )

    def test_recovers_orientation_of_diagonal_grating_within_five_degrees(self):
        """The physical orientation of a 45° diagonal grating must be within 5°.

        measure_periodicity searches the upper half-plane, so for a 45° grating
        with frequency vector (fx, fy) = (1/P, 1/P) it finds the conjugate
        peak at (−fx, −fy), giving an angle of −135°.  Folding mod 180° maps
        this to 45°, which is the physical orientation.

        Tolerance 5° is set conservatively; a transposed atan2(fx, fy) instead
        of atan2(fy, fx) would give −45° (mod 180° = 135°), a 90° error.
        """
        Ny, Nx = 128, 128
        period_px = 16
        px_m = 1e-9
        arr = _sine_wave(Ny, Nx, fx=1.0 / period_px, fy=1.0 / period_px, amp=1.0)

        peaks = measure_periodicity(
            arr,
            pixel_size_x_m=px_m,
            pixel_size_y_m=px_m,
            n_peaks=3,
        )
        assert peaks, "No FFT peaks found for a diagonal grating."

        angle     = peaks[0]["angle_deg"]
        angle_mod = angle % 180.0
        assert abs(angle_mod - 45.0) < 5.0, (
            f"Angle {angle:.1f}° (mod 180° = {angle_mod:.1f}°), "
            f"expected 45° ± 5°.  "
            "atan2 argument order or pixel_size_x/y may be swapped."
        )
