"""Physics-informed first-pass tests for ProbeFlow processing routines.

Synthetic arrays only.  No GUI.  No external files.  Deterministic.

Each test documents the physical motivation for its assertions so that a
future reader understands *why* the tolerance is what it is, not just
*what* is being checked.
"""
from __future__ import annotations

import numpy as np
import pytest

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


def _gaussian_bump(
    Ny: int, Nx: int,
    *,
    cy: int,
    cx: int,
    sigma: float,
    amp: float = 1.0,
) -> np.ndarray:
    """z = amp * exp(-((row−cy)² + (col−cx)²) / (2σ²))."""
    Y, X = np.mgrid[:Ny, :Nx]
    return (amp * np.exp(-((Y - cy) ** 2 + (X - cx) ** 2) / (2.0 * sigma ** 2))).astype(np.float64)


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


# ── second-pass background subtraction ───────────────────────────────────────

class TestBackgroundSubtractionSecondPass:
    """Harder subtract_background scenarios: steps, adsorbates, ROIs, outliers."""

    # ── A: step-position robustness ──────────────────────────────────────────

    @pytest.mark.parametrize("step_frac", [0.25, 0.50, 0.75])
    def test_step_height_robust_to_step_position(self, step_frac: float):
        """Step height must be exact regardless of step position in the image.

        Physical context: a real STM image may show a step anywhere; the
        lower-terrace fit mask covers 25 %–75 % of the image depending on
        where the step falls.  Least-squares on a linear plane is always
        rank-complete for any flat region with ≥ 3 pixels, so the recovered
        step height must be exact to floating-point precision regardless of
        terrace area asymmetry.

        A failure here (error > 1e-8) indicates that coordinate normalisation
        breaks when the fit domain covers only a sub-region of the image, or
        that the mask indexing is wrong.
        """
        Ny, Nx = 64, 64
        step_col = max(4, min(int(round(step_frac * Nx)), Nx - 4))
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
        recovered = upper_median - lower_median

        assert abs(recovered - step_h) < 1e-8, (
            f"step_frac={step_frac}: recovered step {recovered:.10f}, "
            f"true {step_h}; diff {abs(recovered - step_h):.2e}."
        )

    def test_each_terrace_is_flat_after_lower_terrace_fit(self):
        """Both terraces must have zero residual slope after subtracting
        the plane fitted on the lower terrace only.

        Physical context: subtracting the exact linear background leaves both
        terraces flat (their true surface is horizontal).  Non-zero ptp within
        a terrace reveals a residual tilt, which indicates the fit is
        numerically unstable or the normalisation is computed on the wrong
        subset of coordinates.

        Tolerance 1e-8: for a pure tilted plane the fit is rank-complete and
        the residual is limited by floating-point arithmetic only.
        """
        Ny, Nx = 64, 64
        step_col = Nx // 2
        arr = (
            _step_terrace(Ny, Nx, step_col=step_col, step_height=2.0)
            + _tilted_plane(Ny, Nx, slope_x=0.12, slope_y=0.06, offset=1.0)
        )

        lower_mask = np.zeros((Ny, Nx), dtype=bool)
        lower_mask[:, :step_col] = True
        out = subtract_background(arr, order=1, fit_mask=lower_mask)

        lower_ptp = float(np.ptp(out[:, :step_col]))
        upper_ptp = float(np.ptp(out[:, step_col:]))

        assert lower_ptp < 1e-8, (
            f"Lower terrace ptp = {lower_ptp:.2e}; expected < 1e-8."
        )
        assert upper_ptp < 1e-8, (
            f"Upper terrace ptp = {upper_ptp:.2e}; expected < 1e-8."
        )

    # ── B: Gaussian adsorbates ───────────────────────────────────────────────

    def test_gaussian_adsorbate_peak_amplitudes_preserved_with_substrate_mask(self):
        """Adsorbate peak heights must equal their true amplitude when the fit
        mask confines the polynomial fit to clean substrate pixels.

        Physical context: adsorbed molecules appear as Gaussian protrusions.
        Fitting the background on substrate pixels (outside the adsorbates)
        gives an unbiased plane; after subtraction the adsorbate height equals
        the true amplitude to floating-point precision.

        Tolerance 1 % accounts for the Gaussian tail outside the exclusion
        radius contributing a small positive offset to the substrate fit.

        Contrast: a global fit (including adsorbate pixels) absorbs part of
        the adsorbate signal into the background, underestimating peak heights
        by the fraction bump_area / image_area × amp (≈ 7 % here).
        """
        Ny, Nx = 64, 64
        bump_amp   = 3.0
        bump_sigma = 4.0
        positions  = [(16, 16), (16, 48), (48, 32)]

        arr = _tilted_plane(Ny, Nx, slope_x=0.05, slope_y=0.03, offset=0.0)
        for cy, cx in positions:
            arr += _gaussian_bump(Ny, Nx, cy=cy, cx=cx, sigma=bump_sigma, amp=bump_amp)

        # Exclude a disc of radius 3σ around each adsorbate from the fit.
        fit_mask = np.ones((Ny, Nx), dtype=bool)
        Y, X = np.mgrid[:Ny, :Nx]
        for cy, cx in positions:
            fit_mask[(Y - cy) ** 2 + (X - cx) ** 2 <= (3.0 * bump_sigma) ** 2] = False

        out = subtract_background(arr, order=1, fit_mask=fit_mask)

        for cy, cx in positions:
            recovered = float(out[cy, cx])
            assert abs(recovered - bump_amp) < 0.01 * bump_amp, (
                f"Bump at ({cy},{cx}): recovered {recovered:.6f}, "
                f"true {bump_amp}; diff {abs(recovered - bump_amp):.4f} "
                f"({abs(recovered - bump_amp) / bump_amp * 100:.2f} %)."
            )

    def test_substrate_background_flat_after_adsorbate_masked_fit(self):
        """Substrate residual after masked fit must be bounded by Gaussian
        tail leakage through the exclusion boundary.

        With a 3σ exclusion disc, the Gaussian tail amplitude at the boundary
        is bump_amp × exp(−4.5) ≈ 0.011 × bump_amp.  Substrate pixels near
        the boundary carry these tails and prevent a machine-precision result.
        The ptp across the substrate must nevertheless be well below
        bump_amp (< 5 %), confirming the dominant background tilt is removed.

        Contrast: a global fit (no mask) incorporates the full bump integrals
        (≈ 2πσ² × bump_amp per bump), biasing the fitted plane and leaving a
        tilted residual across the substrate that can exceed 10 % of bump_amp.
        The masked fit must be at least 3× flatter than the global fit.

        A failure here indicates that the fit_mask does not actually exclude
        the adsorbate pixels from the least-squares system.
        """
        Ny, Nx = 64, 64
        bump_amp   = 3.0
        bump_sigma = 4.0
        exclude_nsigma = 3.0
        positions  = [(16, 16), (16, 48), (48, 32)]

        arr = _tilted_plane(Ny, Nx, slope_x=0.05, slope_y=0.03, offset=0.0)
        for cy, cx in positions:
            arr += _gaussian_bump(Ny, Nx, cy=cy, cx=cx, sigma=bump_sigma, amp=bump_amp)

        fit_mask = np.ones((Ny, Nx), dtype=bool)
        Y, X = np.mgrid[:Ny, :Nx]
        for cy, cx in positions:
            fit_mask[(Y - cy) ** 2 + (X - cx) ** 2 <= (exclude_nsigma * bump_sigma) ** 2] = False

        masked_out = subtract_background(arr, order=1, fit_mask=fit_mask)
        global_out = subtract_background(arr, order=1)

        masked_ptp = float(np.ptp(masked_out[fit_mask]))
        global_ptp = float(np.ptp(global_out[fit_mask]))

        # Substrate residual is bounded by Gaussian tail at the exclusion boundary.
        tail_at_boundary = bump_amp * float(np.exp(-0.5 * exclude_nsigma ** 2))
        assert masked_ptp < 5.0 * tail_at_boundary, (
            f"Masked substrate ptp = {masked_ptp:.4f}; "
            f"Gaussian tail at {exclude_nsigma}σ = {tail_at_boundary:.4f}; "
            f"tolerance = 5× tail = {5.0 * tail_at_boundary:.4f}."
        )
        # Masked fit must be substantially flatter than the global fit.
        assert masked_ptp < global_ptp / 3.0, (
            f"Masked substrate ptp ({masked_ptp:.4f}) not 3× better than "
            f"global substrate ptp ({global_ptp:.4f}); fit_mask may be ignored."
        )

    # ── C: ROI mask changes the result ──────────────────────────────────────

    def test_substrate_mask_gives_accurate_island_height(self):
        """Background-only fit_mask must recover island height to machine precision.

        Physical context: a 2D material island (graphene, MoS₂, etc.) covers
        half the image.  The global fit absorbs the island into the polynomial,
        biasing both slope and offset; a substrate-only mask gives an exact fit.

        Two assertions:
        1. Masked fit recovers step height to 1e-8 (exact for linear input).
        2. Global fit gives a measurably different step height (> 1 % error),
           confirming the fit_mask is not a no-op.
        """
        Ny, Nx = 64, 64
        step_col = Nx // 2
        island_h = 5.0

        arr = (
            _step_terrace(Ny, Nx, step_col=step_col, step_height=island_h)
            + _tilted_plane(Ny, Nx, slope_x=0.10, slope_y=0.04, offset=0.0)
        )

        substrate_mask = np.zeros((Ny, Nx), dtype=bool)
        substrate_mask[:, :step_col] = True

        masked_out = subtract_background(arr, order=1, fit_mask=substrate_mask)
        global_out = subtract_background(arr, order=1)

        masked_step = (
            float(np.median(masked_out[:, step_col:]))
            - float(np.median(masked_out[:, :step_col]))
        )
        global_step = (
            float(np.median(global_out[:, step_col:]))
            - float(np.median(global_out[:, :step_col]))
        )

        assert abs(masked_step - island_h) < 1e-8, (
            f"Masked fit: recovered step {masked_step:.10f}, true {island_h}; "
            f"diff {abs(masked_step - island_h):.2e}."
        )
        assert abs(global_step - island_h) > 0.01 * island_h, (
            f"Global fit recovered step {global_step:.4f}, same as masked "
            f"({masked_step:.4f}) within 1 %; expected a visible bias from "
            "the island pixels being included in the global fit."
        )

    # ── D: outlier / hot-pixel robustness ────────────────────────────────────

    def test_single_hot_pixel_inflates_background_residual(self):
        """One extreme outlier must measurably inflate the background residual.

        Physical context: tip crashes and detector glitches produce isolated
        pixels with values hundreds of times larger than the surface.
        subtract_background() uses unconstrained least squares, which is not
        outlier-robust: the fitted plane tilts toward the spike, leaving the
        remaining background slightly non-flat.

        This is a *documentation test*: it records a known limitation, not a
        bug.  Failure would mean the implementation accidentally became robust
        (e.g., via median-based fitting), which would be fine but unexpected.

        The clean-plane ptp is < 1e-8 (machine precision).  The hot-pixel
        ptp on the good pixels is ≳ 0.1 (computed from leverage theory for a
        64×64 grid with one point at value 1000).  The 1e-4 threshold is a
        conservative lower bound that proves the residual is inflated above
        floating-point noise.
        """
        Ny, Nx = 64, 64
        hot_val = 1000.0
        hy, hx = Ny // 4, Nx // 4

        arr_clean = _tilted_plane(Ny, Nx, slope_x=0.08, slope_y=0.03, offset=0.0)
        arr_hot   = arr_clean.copy()
        arr_hot[hy, hx] = hot_val

        out_clean = subtract_background(arr_clean, order=1)
        out_hot   = subtract_background(arr_hot,   order=1)

        good = np.ones((Ny, Nx), dtype=bool)
        good[hy, hx] = False

        clean_ptp = float(np.ptp(out_clean[good]))
        hot_ptp   = float(np.ptp(out_hot[good]))

        assert hot_ptp > 1e-4, (
            f"Background ptp with hot pixel = {hot_ptp:.2e}; "
            "expected > 1e-4 (global fit should be biased by the outlier)."
        )
        assert hot_ptp > 1000 * clean_ptp, (
            f"Hot-pixel background ptp ({hot_ptp:.2e}) is less than 1000× "
            f"the clean-plane ptp ({clean_ptp:.2e}); "
            "the bias is smaller than expected from least-squares theory."
        )

    def test_fit_mask_excluding_hot_pixel_recovers_plane_exactly(self):
        """Masking the hot pixel must restore machine-precision flatness.

        Physical context: the standard STM workflow is to identify bad pixels
        visually and exclude them before background subtraction.  With the
        outlier masked, the remaining fit domain is a pure tilted plane, so
        the residual must be zero to floating-point precision.

        Tolerance 1e-8 (same as the clean-plane first-pass test).  A failure
        here indicates the fit_mask logic breaks when a single pixel is
        excluded — e.g., if the mask indexing uses the wrong boolean sense or
        the coordinate normalisation is computed over the unmasked full grid
        rather than the masked sub-grid.
        """
        Ny, Nx = 64, 64
        hot_val = 1000.0
        hy, hx = Ny // 4, Nx // 4

        arr = _tilted_plane(Ny, Nx, slope_x=0.08, slope_y=0.03, offset=0.0)
        arr[hy, hx] = hot_val

        good_mask = np.ones((Ny, Nx), dtype=bool)
        good_mask[hy, hx] = False

        out = subtract_background(arr, order=1, fit_mask=good_mask)

        residual_ptp = float(np.ptp(out[good_mask]))
        assert residual_ptp < 1e-8, (
            f"Residual ptp with hot pixel masked = {residual_ptp:.2e}; "
            "expected < 1e-8.  fit_mask may not correctly exclude the pixel "
            "or the coordinate normalisation is computed over all pixels "
            "rather than only the masked subset."
        )


# ── second-pass FFT / periodicity ─────────────────────────────────────────────

class TestMeasurePeriodicitySecondPass:
    """Physical-unit and lattice-orientation second-pass checks for measure_periodicity."""

    # ── A: physical units ─────────────────────────────────────────────────────

    def test_y_grating_period_recovered_in_physical_metres(self):
        """The physical period must match wavelength × pixel_size, not just
        the pixel count.

        Synthetic: 128×128 image, 16-px period, pixel_size = 0.1 nm/px.
        Expected period = 16 × 0.1 nm = 1.6 nm = 1.6 × 10⁻⁹ m.

        Passing pixel_size = 1 (no unit) while keeping the same grating would
        give period = 16 m — a 10⁸ × error caught by this test.  Swapping
        pixel_size_x_m and pixel_size_y_m for an isotropic image makes no
        difference here; test B covers that distinction.
        """
        Ny, Nx = 128, 128
        period_px    = 16
        pixel_size_m = 0.1e-9   # 0.1 nm/px

        arr = _sine_wave(Ny, Nx, fx=0.0, fy=1.0 / period_px, amp=1.0)
        peaks = measure_periodicity(
            arr,
            pixel_size_x_m=pixel_size_m,
            pixel_size_y_m=pixel_size_m,
            n_peaks=3,
        )
        assert peaks, "No peaks found."

        found    = peaks[0]["period_m"]
        expected = period_px * pixel_size_m
        rel_err  = abs(found - expected) / expected
        assert rel_err < 0.02, (
            f"Period {found * 1e9:.4f} nm, expected {expected * 1e9:.4f} nm "
            f"({rel_err * 100:.1f} % error).  "
            "pixel_size_m may not be applied to the frequency axis."
        )

    @pytest.mark.parametrize("pixel_size_nm,expected_period_nm", [
        (0.10, 1.6),
        (0.20, 3.2),
    ])
    def test_physical_period_scales_linearly_with_pixel_size(
        self, pixel_size_nm: float, expected_period_nm: float
    ):
        """Changing pixel size while keeping the grating in pixels must scale
        the recovered physical period proportionally.

        Same 16-px grating, two pixel sizes: 0.1 nm → 1.6 nm, 0.2 nm → 3.2 nm.

        A fixed-scale bug (e.g., pixel_size hard-coded to 1 nm) would give
        16 nm for both cases instead of the correct values, failing at least
        one branch.  A factor-of-2 error in pixel_size handling would swap
        the two expected results.
        """
        Ny, Nx = 128, 128
        period_px    = 16
        pixel_size_m = pixel_size_nm * 1e-9
        expected_m   = expected_period_nm * 1e-9

        arr = _sine_wave(Ny, Nx, fx=0.0, fy=1.0 / period_px, amp=1.0)
        peaks = measure_periodicity(
            arr,
            pixel_size_x_m=pixel_size_m,
            pixel_size_y_m=pixel_size_m,
            n_peaks=3,
        )
        assert peaks, f"No peaks found for pixel_size = {pixel_size_nm} nm."

        found   = peaks[0]["period_m"]
        rel_err = abs(found - expected_m) / expected_m
        assert rel_err < 0.02, (
            f"pixel_size = {pixel_size_nm} nm: period {found * 1e9:.4f} nm, "
            f"expected {expected_m * 1e9:.4f} nm ({rel_err * 100:.1f} % error)."
        )

    def test_physical_period_preserved_when_pixel_count_changes_at_fixed_scan_size(self):
        """The same physical wavelength must be recovered from grids of
        different pixel count when the scan size is held constant.

        Scan size = 12.8 nm.  Two resolutions:
        - 64 px  → pixel_size = 0.200 nm → grating period = 8 px → 1.6 nm
        - 128 px → pixel_size = 0.100 nm → grating period = 16 px → 1.6 nm

        Period bins land exactly at integer FFT bins in both cases so there
        is no discretisation error; a wrong frequency-axis indexing (e.g.,
        treating the bin index as cycles/image rather than cycles/px) would
        give different periods for the two resolutions.
        """
        scan_m           = 12.8e-9   # fixed scan size
        true_period_m    = 1.6e-9    # fixed physical wavelength

        for Ny, Nx in [(64, 64), (128, 128)]:
            pixel_size_m = scan_m / Ny
            period_px    = round(true_period_m / pixel_size_m)   # integer by construction

            arr = _sine_wave(Ny, Nx, fx=0.0, fy=1.0 / period_px, amp=1.0)
            peaks = measure_periodicity(
                arr,
                pixel_size_x_m=pixel_size_m,
                pixel_size_y_m=pixel_size_m,
                n_peaks=3,
            )
            assert peaks, f"No peaks found for {Ny}×{Nx} grid."

            found   = peaks[0]["period_m"]
            rel_err = abs(found - true_period_m) / true_period_m
            assert rel_err < 0.02, (
                f"{Ny}×{Nx} (pixel_size = {pixel_size_m * 1e9:.3f} nm): "
                f"period {found * 1e9:.4f} nm, expected {true_period_m * 1e9:.4f} nm "
                f"({rel_err * 100:.1f} % error)."
            )

    # ── B: lattice orientation ────────────────────────────────────────────────

    def test_grating_at_non_trivial_angle_recovered_within_five_degrees(self):
        """A grating whose wave-vector is not 45° must be recovered correctly.

        Grating vector (kx_bin, ky_bin) = (4, 3) on a 128×128 grid.
        These are integer FFT bins so there is no frequency discretisation
        error.  The physical angle is atan2(3, 4) ≈ 36.87°.

        The function searches the upper half-plane (negative fy) and finds
        the conjugate peak at (kx = −4, ky = −3) → atan2(−3, −4) ≈ −143°
        → mod 180° ≈ 37°.

        Error signatures:
        - Transposed array: bins (3, 4) found → atan2(4, 3) ≈ 53° (wrong).
        - Mirrored array: angle → 180° − 37° = 143° (wrong mod 180°).
        Both errors exceed the 5° tolerance.
        """
        Ny, Nx = 128, 128
        kx_bin, ky_bin = 4, 3
        px_m = 1e-9
        expected_angle = float(np.degrees(np.arctan2(ky_bin, kx_bin)))   # ≈ 36.87°

        arr = _sine_wave(Ny, Nx, fx=kx_bin / Nx, fy=ky_bin / Ny, amp=1.0)
        peaks = measure_periodicity(
            arr,
            pixel_size_x_m=px_m,
            pixel_size_y_m=px_m,
            n_peaks=3,
        )
        assert peaks, "No peaks found for 37° grating."

        angle_mod = peaks[0]["angle_deg"] % 180.0
        assert abs(angle_mod - expected_angle) < 5.0, (
            f"Angle {peaks[0]['angle_deg']:.1f}° (mod 180° = {angle_mod:.1f}°), "
            f"expected {expected_angle:.1f}° ± 5°.  "
            "Array may be transposed (gives ≈53°) or mirrored (gives ≈143°)."
        )

    def test_square_lattice_gives_two_perpendicular_peaks(self):
        """A square lattice must produce two 90°-separated FFT peaks.

        Lattice: sum of two equal-amplitude sines with grating vectors
        (kx, ky) = (4, 3) and (−3, 4) in FFT bins — perpendicular by
        construction (dot product = 0).  Both conjugates land at distinct
        positions in the upper-half search mask, separated by
        sqrt(50) ≈ 7.1 px > suppress_r = 6 px, so both are returned.

        The expected angles are atan2(3, 4) ≈ 36.87° and atan2(4, −3)
        ≈ 126.87°, differing by exactly 90°.

        This test catches axis swaps that affect only one direction (a pure
        90° offset shifts both angles equally and the separation remains 90°,
        but individual angles would each fail their expected-value check).
        """
        Ny, Nx = 128, 128
        px_m = 1e-9
        kx1, ky1 =  4,  3   # angle ≈  36.87°
        kx2, ky2 = -3,  4   # angle ≈ 126.87°; conjugate at (3, −4) in upper half

        arr = (
            _sine_wave(Ny, Nx, fx=kx1 / Nx, fy=ky1 / Ny, amp=1.0)
            + _sine_wave(Ny, Nx, fx=kx2 / Nx, fy=ky2 / Ny, amp=1.0)
        )
        peaks = measure_periodicity(
            arr,
            pixel_size_x_m=px_m,
            pixel_size_y_m=px_m,
            n_peaks=4,
        )
        assert len(peaks) >= 2, (
            f"Expected ≥ 2 peaks for a square lattice, got {len(peaks)}."
        )

        angles_mod = sorted(p["angle_deg"] % 180.0 for p in peaks[:2])
        raw_sep    = abs(angles_mod[1] - angles_mod[0])
        # Map to [0°, 90°]: angles mod 180° live on a semicircle.
        angle_sep  = min(raw_sep, 180.0 - raw_sep)

        assert abs(angle_sep - 90.0) < 5.0, (
            f"Peak separation {angle_sep:.1f}° (from angles mod 180°: "
            f"{angles_mod[0]:.1f}°, {angles_mod[1]:.1f}°); expected 90° ± 5°."
        )

        expected1 = float(np.degrees(np.arctan2(ky1, kx1)))   # ≈  36.87°
        expected2 = float(np.degrees(np.arctan2(ky2, kx2))) % 180.0   # ≈ 126.87°

        for ang in angles_mod:
            nearest_error = min(abs(ang - expected1), abs(ang - expected2))
            assert nearest_error < 5.0, (
                f"Peak at {ang:.1f}° does not match expected {expected1:.1f}° "
                f"or {expected2:.1f}° within 5°.  "
                "One lattice direction may be misidentified."
            )
