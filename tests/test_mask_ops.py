"""Tests for morphological mask cleanup operations."""
from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing import mask_ops


def _square(n: int = 20) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    m[5:15, 5:15] = True
    return m


def test_invert():
    m = _square()
    assert np.array_equal(mask_ops.invert(m), ~m)


def test_dilate_grows_and_erode_shrinks():
    m = _square()
    assert mask_ops.dilate(m, 1).sum() > m.sum()
    assert mask_ops.erode(m, 1).sum() < m.sum()


def test_open_removes_speck():
    m = _square()
    m[0, 0] = True  # isolated speck
    opened = mask_ops.binary_open(m, 1)
    assert not opened[0, 0]
    # Bulk of the square survives opening (a disk element rounds the corners).
    assert opened[8:12, 8:12].all()


def test_close_fills_small_gap():
    m = _square()
    m[10, 10] = False  # 1-px hole
    closed = mask_ops.binary_close(m, 1)
    assert closed[10, 10]


def test_fill_holes():
    m = _square()
    m[8:12, 8:12] = False  # interior hole
    filled = mask_ops.fill_holes(m)
    assert filled[8:12, 8:12].all()


def test_remove_small_objects():
    m = _square()
    m[0, 0] = True
    cleaned = mask_ops.remove_small_objects(m, min_size=5)
    assert not cleaned[0, 0]
    assert cleaned[5:15, 5:15].all()


def test_remove_small_holes():
    m = _square()
    m[9, 9] = False
    filled = mask_ops.remove_small_holes(m, area_threshold=5)
    assert filled[9, 9]


def test_skeletonize_thins_to_line():
    m = _square()
    skel = mask_ops.skeletonize(m)
    assert skel.sum() < m.sum()
    assert skel.shape == m.shape


def test_remove_border_objects():
    m = np.zeros((20, 20), dtype=bool)
    m[0:5, 0:5] = True    # touches border → removed
    m[10:15, 10:15] = True  # interior → kept
    cleaned = mask_ops.remove_border_objects(m)
    assert not cleaned[0:5, 0:5].any()
    assert cleaned[10:15, 10:15].all()


@pytest.mark.parametrize("op", [
    mask_ops.dilate, mask_ops.erode, mask_ops.binary_open, mask_ops.binary_close,
])
def test_empty_mask_safe(op):
    empty = np.zeros((10, 10), dtype=bool)
    out = op(empty)
    assert out.shape == (10, 10)
    assert not out.any()


def test_shape_preserved_and_bool_dtype():
    m = _square()
    for fn in (mask_ops.invert, mask_ops.fill_holes, mask_ops.skeletonize,
               mask_ops.remove_small_objects, mask_ops.remove_border_objects):
        out = fn(m)
        assert out.shape == m.shape
        assert out.dtype == bool
