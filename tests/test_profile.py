"""Tests for probeflow.processing.line_profile."""

from __future__ import annotations

import numpy as np

from probeflow.processing import line_profile


def test_constant_image_constant_profile():
    arr = np.full((64, 64), 3.5, dtype=np.float64)
    s, z = line_profile(arr, (0, 0), (63, 0),
                        pixel_size_x_m=1e-10, pixel_size_y_m=1e-10)
    assert s.shape == z.shape
    assert np.allclose(z, 3.5)


def test_horizontal_line_recovers_row():
    rng = np.random.default_rng(42)
    arr = rng.normal(size=(32, 32))
    # Profile along row 10 from x=0 to x=31.
    s, z = line_profile(arr, (0, 10), (31, 10),
                        pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
                        n_samples=32)
    np.testing.assert_allclose(z, arr[10, :])
    assert np.isclose(s[-1], 31 * 1e-10)


def test_diagonal_distance_uses_both_pixel_sizes():
    arr = np.zeros((10, 10), dtype=np.float64)
    s, _ = line_profile(arr, (0, 0), (9, 9),
                        pixel_size_x_m=2e-10, pixel_size_y_m=1e-10)
    expected_len = float(np.hypot(9 * 2e-10, 9 * 1e-10))
    assert np.isclose(s[-1], expected_len)


def test_swath_averaging_is_smoother():
    # Build a noisy field with a clean horizontal stripe through it.
    rng = np.random.default_rng(0)
    base = rng.normal(scale=0.5, size=(64, 64))
    # Profile across the noisy field; swath width=1 vs width=11.
    _, z1 = line_profile(base, (0, 32), (63, 32),
                         pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
                         width_px=1.0, n_samples=64)
    _, z11 = line_profile(base, (0, 32), (63, 32),
                          pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
                          width_px=11.0, n_samples=64)
    # Wider swath averages more pixels → smaller variance.
    assert z11.std() < z1.std()


def test_rejects_zero_length():
    arr = np.zeros((8, 8))
    try:
        line_profile(arr, (3, 3), (3, 3),
                     pixel_size_x_m=1e-10, pixel_size_y_m=1e-10)
    except ValueError:
        return
    raise AssertionError("expected ValueError for zero-length segment")


def test_border_samples_outside_image_as_nan():
    arr = np.arange(9, dtype=np.float64).reshape(3, 3)

    _s, z = line_profile(
        arr,
        (-1, 1),
        (2, 1),
        pixel_size_x_m=1e-10,
        pixel_size_y_m=1e-10,
        n_samples=4,
        interp="nearest",
    )

    assert np.isnan(z[0])
    np.testing.assert_allclose(z[1:], arr[1, :])


def test_swath_width_uses_stable_ceil_sampling_at_border():
    arr = np.tile(np.arange(4, dtype=np.float64)[:, None] * 10.0, (1, 5))

    _s, z = line_profile(
        arr,
        (0, 0),
        (4, 0),
        pixel_size_x_m=1e-10,
        pixel_size_y_m=1e-10,
        width_px=2.5,
        n_samples=5,
        interp="nearest",
    )

    np.testing.assert_allclose(z, np.full(5, 5.0))
