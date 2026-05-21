"""Contract tests for shared display-rendering helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from probeflow.processing.display import (
    array_to_uint8,
    clip_range_from_array,
    finite_values,
    histogram_from_array,
    normalise_array,
)


def test_finite_values_contract():
    result = finite_values(np.array([[1.0, np.nan], [np.inf, 4.0]]))
    np.testing.assert_array_equal(result, [1.0, 4.0])
    assert result.ndim == 1
    assert finite_values(np.full((3, 3), np.nan)).size == 0
    assert finite_values(np.array([[1, 2], [3, 4]], dtype=np.int32)).dtype == np.float64


def test_clip_range_contract():
    arr = np.linspace(0.0, 100.0, 1000)
    arr[:2] = [np.nan, np.inf]
    vmin, vmax = clip_range_from_array(arr, 1.0, 99.0)
    assert np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax

    rng = np.random.default_rng(42)
    noisy = rng.normal(loc=1e-10, scale=1e-11, size=(100, 100))
    noisy[0, 0] = 1e6
    _, outlier_vmax = clip_range_from_array(noisy, 1.0, 99.0)
    assert outlier_vmax < 1e3

    for bad in (np.full((4, 4), np.nan), np.full((4, 4), np.inf)):
        with pytest.raises(ValueError, match="no finite values"):
            clip_range_from_array(bad)

    cmin, cmax = clip_range_from_array(np.full((10, 10), 5.0))
    assert np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin
    assert clip_range_from_array(np.arange(100.0), 5.0, 95.0)[0] >= clip_range_from_array(
        np.arange(100.0), 1.0, 99.0
    )[0]


def test_normalise_array_contract():
    arr = np.array([[0.0, 5.0, 10.0], [np.nan, np.inf, 2.5]])
    out = normalise_array(arr, 0.0, 10.0)
    assert out.dtype == np.float32
    assert out.shape == arr.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert out[1, 0] == pytest.approx(0.0)
    assert out[1, 1] == pytest.approx(0.0)
    assert (normalise_array(np.full((4, 4), 5.0), 5.0, 5.0) == 0.0).all()


def test_array_to_uint8_contract():
    explicit = array_to_uint8(np.array([[0.0, 0.5], [0.75, 1.0]]), vmin=0.0, vmax=1.0)
    assert explicit.dtype == np.uint8
    assert explicit.shape == (2, 2)
    assert explicit[0, 0] == 0
    assert explicit[1, 1] == 255

    with_nan = array_to_uint8(np.array([[1.0, np.nan], [3.0, 4.0]]))
    assert with_nan.dtype == np.uint8

    rng = np.random.default_rng(0)
    outlier = rng.normal(loc=5.0, scale=0.5, size=(50, 50))
    outlier[0, 0] = 1e9
    out = array_to_uint8(outlier)
    assert 0 <= int(out.min()) <= int(out.max()) <= 255

    with pytest.raises(ValueError, match="no finite values"):
        array_to_uint8(np.full((4, 4), np.nan))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        constant = array_to_uint8(np.full((8, 8), 3.14))
    assert constant.dtype == np.uint8

    assert array_to_uint8(np.linspace(0.0, 100.0, 1000), clip_percentiles=(10, 90)).dtype == np.uint8


def test_histogram_contract():
    arr = np.linspace(0.0, 10.0, 200)
    counts, edges = histogram_from_array(arr, bins=64)
    assert len(counts) == 64
    assert len(edges) == 65
    assert np.issubdtype(counts.dtype, np.integer)

    arr[:2] = [np.nan, np.inf]
    counts, _ = histogram_from_array(arr)
    assert 0 < counts.sum() <= 198

    with pytest.raises(ValueError, match="no finite values"):
        histogram_from_array(np.full((4, 4), np.nan))

    rng = np.random.default_rng(1)
    outlier = rng.normal(loc=0.0, scale=1.0, size=(100, 100))
    outlier[0, 0] = 1e9
    _, edges = histogram_from_array(outlier, clip_percentiles=(1.0, 99.0))
    vmin, vmax = clip_range_from_array(outlier, 1.0, 99.0)
    assert abs(edges[0] - vmin) < 1e-10
    assert abs(edges[-1] - vmax) < 1e-10
