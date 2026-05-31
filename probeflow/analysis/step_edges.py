"""ProbeFlow — algorithmic step-edge detection for reproducible particle exclusion.

Molecules preferentially decorate substrate step edges, but a particle sitting on
a step is usually a topographic artefact (the step's own height jump) rather than
a clean adsorbate, so it should be dropped from counts/statistics.  The UniMR
Features tool lets the user *paint* over the step by hand; that is blunt and not
reproducible.  This module computes the step-edge region algorithmically from
physical parameters (a step-slope angle and an optional minimum step height) so
the exclusion is deterministic and re-runnable across a whole folder of scans.

Core idea
---------
A step edge is a *locally steep* region of the *substrate*.  The catch is that a
molecule's own perimeter is also steep, so a naive slope threshold flags every
molecule — including ones on flat terraces.  We therefore:

1. **Suppress molecules** with an edge-preserving morphological opening (closing
   for dark features): compact bumps narrower than the structuring element vanish
   while the laterally-extended terrace step is preserved.
2. Compute the **physical slope magnitude** (the same finite-difference pattern
   as :func:`probeflow.processing.alignment.facet_level`).
3. Threshold by a **step angle**; optionally keep only components whose
   terrace-to-terrace **height jump** exceeds ``min_step_height_m``.
4. **Dilate** to a margin so molecules sitting *adjacent* to (not only exactly on)
   the step are caught when the mask is later used for particle rejection.

Placement note for future maintainers / AI agents
--------------------------------------------------
This is a GUI-free numerical kernel.  It depends only on numpy + scipy.ndimage
(no OpenCV, no Qt).  Import it lazily from ``probeflow.gui.features`` or the CLI,
like the rest of :mod:`probeflow.analysis`, so the optional/heavier dependencies
stay out of the browse/convert path.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _finite_median(arr: np.ndarray) -> float:
    """Median of the finite values, or 0.0 if there are none."""
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if finite.size else 0.0


def _component_height_jump(
    base: np.ndarray, comp: np.ndarray, sample_px: int
) -> float:
    """Terrace-to-terrace height jump across one band component.

    Dilates the component by ``sample_px`` so the sampled region reaches the flat
    terrace on *both* sides of the step, then returns the robust spread
    (90th − 10th percentile) of the molecule-suppressed heights there.  This is
    orientation-free — it works for vertical, horizontal, diagonal or curved
    steps — and robust to noise.
    """
    from scipy.ndimage import binary_dilation

    region = binary_dilation(comp, iterations=max(1, int(sample_px)))
    vals = base[region]
    vals = vals[np.isfinite(vals)]
    if vals.size < 4:
        return 0.0
    return float(np.percentile(vals, 90) - np.percentile(vals, 10))


def step_edge_mask(
    arr: np.ndarray,
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    molecule_diameter_m: float,
    threshold_deg: float = 20.0,
    dilate_m: float = 0.3e-9,
    min_step_height_m: Optional[float] = None,
    suppress_dark: bool = False,
) -> np.ndarray:
    """Return a boolean mask of pixels lying on a substrate step edge.

    Parameters
    ----------
    arr
        2-D float scan plane (height data).  NaN/inf are tolerated.
    pixel_size_x_m, pixel_size_y_m
        Physical pixel width / height in metres.  Used to convert the
        finite-difference gradient into a dimensionless slope so the
        ``threshold_deg`` comparison is physically meaningful.
    molecule_diameter_m
        Approximate diameter of the molecules to suppress before detecting the
        step, in metres.  Sets the morphological structuring-element size: it
        must be at least one molecule wide so a lone molecule is fully removed,
        but smaller than a terrace so the step survives.  Must be > 0.
    threshold_deg
        Local surface-slope angle (degrees) above which a (molecule-suppressed)
        pixel is considered to be on a step.  A monatomic step is far steeper
        than terrace tilt / curvature, so a generous value such as the default
        20° separates them with wide margin on both sides.
    dilate_m
        Physical margin (metres) by which the detected band is grown, so a
        molecule sitting *next to* the step — not only squarely on it — overlaps
        the band when this mask is used to reject particles.  0 disables.
    min_step_height_m
        If given, keep only connected band components whose terrace-to-terrace
        height jump is at least this many metres.  Distinguishes a genuine step
        from a shallow undulation that happens to be locally steep.  ``None``
        (default) uses slope alone.
    suppress_dark
        Use a morphological *closing* instead of *opening*, to suppress dark
        depressions instead of bright protrusions.  Set this when segmenting
        dark features (``invert=True`` in :func:`segment_particles`).

    Returns
    -------
    np.ndarray of bool, same shape as *arr*.  True = on a step edge.  Fully
    deterministic (no RNG): identical inputs give an identical mask.
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("step_edge_mask expects a 2-D array")
    if molecule_diameter_m <= 0:
        raise ValueError("molecule_diameter_m must be > 0")

    Ny, Nx = a.shape
    empty = np.zeros((Ny, Nx), dtype=bool)
    if Ny < 3 or Nx < 3:
        return empty
    finite = np.isfinite(a)
    if not finite.any():
        return empty

    # Clamp pixel sizes (guard against 0 / NaN metadata) exactly as facet_level.
    dx = max(float(pixel_size_x_m), 1e-30)
    dy = max(float(pixel_size_y_m), 1e-30)
    geom_px_m = math.sqrt(dx * dy)

    from scipy.ndimage import grey_closing, grey_opening, label

    # ── 1. Molecule suppression ──────────────────────────────────────────────
    # Per-axis structuring element, odd and at least one molecule across, so a
    # lone molecule is fully removed.  Per-axis keeps it correct on rectangular
    # pixels.
    wx = max(3, 2 * math.ceil(molecule_diameter_m / dx) + 1)
    wy = max(3, 2 * math.ceil(molecule_diameter_m / dy) + 1)
    filled = np.where(finite, a, _finite_median(a))
    morph = grey_closing if suppress_dark else grey_opening
    base = morph(filled, size=(wy, wx))

    # ── 2. Physical slope magnitude ──────────────────────────────────────────
    gy, gx = np.gradient(base)
    slope_mag = np.sqrt((gx / dx) ** 2 + (gy / dy) ** 2)

    # ── 3. Slope threshold ───────────────────────────────────────────────────
    band = slope_mag > math.tan(math.radians(threshold_deg))

    # ── 4. Optional height gate ──────────────────────────────────────────────
    if min_step_height_m is not None and band.any():
        labeled, n = label(band)
        # Sample one structuring-element away so the collar reaches flat terrace.
        sample_px = max(wx, wy)
        kept = np.zeros_like(band)
        for i in range(1, n + 1):
            comp = labeled == i
            if _component_height_jump(base, comp, sample_px) >= float(min_step_height_m):
                kept |= comp
        band = kept

    # ── 5. Dilate to an exclusion margin ─────────────────────────────────────
    if dilate_m and dilate_m > 0 and band.any():
        from scipy.ndimage import binary_dilation

        iters = max(1, int(round(float(dilate_m) / geom_px_m)))
        band = binary_dilation(band, iterations=iters)

    return band & finite
