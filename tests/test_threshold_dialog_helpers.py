"""Qt-free checks for threshold-dialog display calculations."""

from __future__ import annotations

import numpy as np

from probeflow.gui.dialogs.threshold_dialog import (
    _display_range_from_finite,
    _threshold_bounds_valid,
)


def test_display_range_uses_percentiles_not_raw_extrema():
    finite = np.arange(100, dtype=np.float64)
    finite[-1] = 1_000_000.0

    display_min, display_max = _display_range_from_finite(finite)

    assert display_min > float(finite.min())
    assert display_max < float(finite.max())


def test_equal_threshold_bounds_select_a_constant_value():
    assert _threshold_bounds_valid(1.25, 1.25)
    assert not _threshold_bounds_valid(2.0, 1.0)
