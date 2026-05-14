"""
Pure factory functions returning adversarial numpy arrays designed to expose
edge-cases and break processing assumptions.

Each factory returns a dict with at least:
    arr              : float64 ndarray (or 'fwd'/'bwd' for fwd_bwd_asymmetric)
    description      : str
    pixel_size_x_m   : float
    pixel_size_y_m   : float

Additional keys are documented per fixture.
No pytest decorators; these are plain callables usable from tests or scripts.
"""

from __future__ import annotations

import numpy as np


def flat_with_outlier() -> dict:
    """64x64 zeros with a single hot pixel at (32,32) = 1e-7 m.

    pixel_size 1 nm. Tests that outlier-pixel is NOT flagged as a bad scanline
    segment (it is too short), and that per-row alignment preserves the value.
    """
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[32, 32] = 1e-7
    return {
        "arr": arr,
        "description": "64x64 zeros with a single hot pixel = 1e-7 m at (32,32)",
        "pixel_size_x_m": 1e-9,
        "pixel_size_y_m": 1e-9,
    }


def tilted_plane_with_step() -> dict:
    """64x64 plane tilted along x plus a hard step at y=32.

    arr[y, x] = 0.1e-9 * x/63 + step_height * (y >= 32)
    step_height = 3e-10 m. pixel_size 1 nm.

    Extra keys: step_height_m, step_row.
    """
    step_height = 3e-10
    step_row = 32
    y_idx, x_idx = np.mgrid[:64, :64]
    arr = (0.1e-9 * x_idx / 63.0 + step_height * (y_idx >= step_row)).astype(np.float64)
    return {
        "arr": arr,
        "description": "64x64 tilted plane + step at row 32 (3e-10 m)",
        "pixel_size_x_m": 1e-9,
        "pixel_size_y_m": 1e-9,
        "step_height_m": step_height,
        "step_row": step_row,
    }


def lattice_with_scanline_glitch() -> dict:
    """64x64 atomic lattice (cos * cos) with a bright segment injected in row 20.

    Lattice: cos(2*pi*x/8) * cos(2*pi*y/8) * 1e-10 m.
    Glitch: row 20, cols 16..48 += 5e-10 m.
    pixel_size 0.5 nm.

    Extra keys: glitch_row, glitch_start, glitch_end, glitch_height, lattice_amplitude.
    """
    lattice_amplitude = 1e-10
    glitch_height = 5e-10
    glitch_row = 20
    glitch_start = 16
    glitch_end = 48  # exclusive

    y_idx, x_idx = np.mgrid[:64, :64]
    arr = (
        np.cos(2 * np.pi * x_idx / 8.0)
        * np.cos(2 * np.pi * y_idx / 8.0)
        * lattice_amplitude
    ).astype(np.float64)
    arr[glitch_row, glitch_start:glitch_end] += glitch_height

    return {
        "arr": arr,
        "description": "64x64 atomic lattice with 5e-10 m glitch segment in row 20",
        "pixel_size_x_m": 0.5e-9,
        "pixel_size_y_m": 0.5e-9,
        "glitch_row": glitch_row,
        "glitch_start": glitch_start,
        "glitch_end": glitch_end,
        "glitch_height": glitch_height,
        "lattice_amplitude": lattice_amplitude,
    }


def nan_horizontal_stripe() -> dict:
    """64x64 linear x-ramp with NaN stripe in rows 22..24.

    arr[:, x] = x / 63 * 1e-9.  Rows 22, 23, 24 set to NaN.
    pixel_size 1 nm.

    Extra keys: stripe_start=22, stripe_width=3.
    """
    stripe_start = 22
    stripe_width = 3
    x_idx = np.arange(64, dtype=np.float64)
    arr = np.broadcast_to(x_idx / 63.0 * 1e-9, (64, 64)).copy()
    arr[stripe_start : stripe_start + stripe_width, :] = np.nan

    return {
        "arr": arr,
        "description": "64x64 x-ramp with NaN stripe in rows 22..24",
        "pixel_size_x_m": 1e-9,
        "pixel_size_y_m": 1e-9,
        "stripe_start": stripe_start,
        "stripe_width": stripe_width,
    }


def real_islands_mimic_artefact() -> dict:
    """64x64 with 3 molecular islands and one scanline artefact at the same height.

    Islands (5x5 blobs) at (16,16), (40,40), (16,50) = 3e-10 m.
    Scanline artefact: row 32, cols 25..35 = 3e-10 m (same height, but contiguous
    horizontal segment — typical scanner artefact pattern).
    pixel_size 1 nm.

    Extra keys: island_height, island_centres, glitch_row, glitch_cols.
    """
    island_height = 3e-10
    glitch_row = 32
    glitch_cols = (25, 35)  # half-open [25, 35)
    island_centres = [(16, 16), (40, 40), (16, 50)]  # (row, col)

    arr = np.zeros((64, 64), dtype=np.float64)
    for row, col in island_centres:
        arr[row : row + 5, col : col + 5] = island_height
    arr[glitch_row, glitch_cols[0] : glitch_cols[1]] = island_height

    return {
        "arr": arr,
        "description": "64x64 islands + scanline artefact at same height (3e-10 m)",
        "pixel_size_x_m": 1e-9,
        "pixel_size_y_m": 1e-9,
        "island_height": island_height,
        "island_centres": island_centres,
        "glitch_row": glitch_row,
        "glitch_cols": glitch_cols,
    }


def anisotropic_pixels() -> dict:
    """64x64 slope with anisotropic physical pixel sizes.

    arr[y, x] = (x * 1e-9 + y * 3e-9) * 0.1
    pixel_size_x_m = 1e-9, pixel_size_y_m = 3e-9.
    Tests that facet_level uses physical pixel size.
    """
    y_idx, x_idx = np.mgrid[:64, :64]
    arr = ((x_idx * 1e-9 + y_idx * 3e-9) * 0.1).astype(np.float64)
    return {
        "arr": arr,
        "description": "64x64 anisotropic-pixel slope (px=1nm, py=3nm)",
        "pixel_size_x_m": 1e-9,
        "pixel_size_y_m": 3e-9,
    }


def non_square_image() -> dict:
    """48 rows x 96 cols. Left half = 0, right half = 5e-10 m (vertical step).

    pixel_size 1 nm.
    """
    arr = np.zeros((48, 96), dtype=np.float64)
    arr[:, 48:] = 5e-10
    return {
        "arr": arr,
        "description": "48x96 non-square: left half=0, right half=5e-10 m",
        "pixel_size_x_m": 1e-9,
        "pixel_size_y_m": 1e-9,
    }


def tiny_3x3() -> dict:
    """3x3 arange * 1e-10 m. pixel_size 1 nm. Edge case for small images."""
    arr = (np.arange(9, dtype=np.float64).reshape(3, 3)) * 1e-10
    return {
        "arr": arr,
        "description": "3x3 tiny image: arange * 1e-10 m",
        "pixel_size_x_m": 1e-9,
        "pixel_size_y_m": 1e-9,
    }


def tiny_2x2() -> dict:
    """2x2 [[1,2],[3,4]] * 1e-10 m. pixel_size 1 nm. Extreme edge case."""
    arr = np.array([[1, 2], [3, 4]], dtype=np.float64) * 1e-10
    return {
        "arr": arr,
        "description": "2x2 tiny image: [[1,2],[3,4]] * 1e-10 m",
        "pixel_size_x_m": 1e-9,
        "pixel_size_y_m": 1e-9,
    }


def constant_nonzero() -> dict:
    """32x32 all = 5e-10 m. Tests that high-pass filters preserve the mean.

    Extra key: constant_val.
    """
    constant_val = 5e-10
    arr = np.full((32, 32), constant_val, dtype=np.float64)
    return {
        "arr": arr,
        "description": "32x32 constant image at 5e-10 m",
        "pixel_size_x_m": 1e-9,
        "pixel_size_y_m": 1e-9,
        "constant_val": constant_val,
    }


def negative_heights() -> dict:
    """64x64 zeros with a depression and an adatom.

    arr[20:40, 20:40] = -2e-10 (vacancy / depression).
    arr[10:15, 10:15] = +1e-10 (adatom).
    pixel_size 1 nm.

    Extra keys: depression_height, adatom_height.
    """
    depression_height = -2e-10
    adatom_height = 1e-10
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[20:40, 20:40] = depression_height
    arr[10:15, 10:15] = adatom_height
    return {
        "arr": arr,
        "description": "64x64 with vacancy (-2e-10) and adatom (+1e-10)",
        "pixel_size_x_m": 1e-9,
        "pixel_size_y_m": 1e-9,
        "depression_height": depression_height,
        "adatom_height": adatom_height,
    }


def fwd_bwd_asymmetric() -> dict:
    """32x32 forward/backward scan pair with 3-pixel drift.

    fwd[y, x] = sin(2*pi*x/8) * 1e-10
    bwd[y, x] = sin(2*pi*(x+3)/8) * 1e-10  (3 px lateral drift)

    Returns dict with 'fwd' and 'bwd' keys (no 'arr' key).
    Extra keys: drift_px=3, period_px=8.
    """
    rng = np.random.default_rng(42)
    _ = rng  # keep for reproducibility anchor; not used here

    x_idx = np.arange(32, dtype=np.float64)
    row = np.sin(2 * np.pi * x_idx / 8.0) * 1e-10
    fwd = np.broadcast_to(row, (32, 32)).copy().astype(np.float64)

    row_bwd = np.sin(2 * np.pi * (x_idx + 3.0) / 8.0) * 1e-10
    bwd = np.broadcast_to(row_bwd, (32, 32)).copy().astype(np.float64)

    return {
        "fwd": fwd,
        "bwd": bwd,
        "description": "32x32 fwd/bwd pair with 3-px lateral drift, period 8 px",
        "pixel_size_x_m": 1e-9,
        "pixel_size_y_m": 1e-9,
        "drift_px": 3,
        "period_px": 8,
    }
