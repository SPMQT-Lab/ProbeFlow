"""Contract tests for probeflow.processing.display_state.DisplayRangeState."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.display import array_to_uint8, clip_range_from_array
from probeflow.processing.display_state import DisplayRangeState


def test_percentile_mode_contract():
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=5.0, scale=1.0, size=(50, 50))
    drs = DisplayRangeState(low_pct=1.0, high_pct=99.0)

    assert drs.mode == "percentile"
    assert drs.resolve(arr) == pytest.approx(clip_range_from_array(arr, 1.0, 99.0))
    assert DisplayRangeState(low_pct=5.0, high_pct=95.0).resolve(
        np.linspace(0.0, 100.0, 1000)
    ) == pytest.approx(clip_range_from_array(np.linspace(0.0, 100.0, 1000), 5.0, 95.0))
    assert DisplayRangeState().resolve(np.full((4, 4), np.nan)) == (None, None)

    drs.set_manual(1.0, 5.0)
    drs.set_percentile(2.0, 98.0)
    assert drs.mode == "percentile"
    assert (drs.low_pct, drs.high_pct) == (2.0, 98.0)


def test_manual_mode_contract():
    arr = np.linspace(0.0, 100.0, 1000)
    drs = DisplayRangeState(low_pct=1.0, high_pct=99.0)
    drs.set_manual(30.0, 70.0)

    assert drs.mode == "manual"
    assert drs.resolve(arr) == pytest.approx((30.0, 70.0))
    assert abs(drs.resolve(arr)[0] - clip_range_from_array(arr, 1.0, 99.0)[0]) > 1.0

    drs.set_manual(-1e6, 1e6)
    assert drs.resolve(np.linspace(0.0, 1.0, 100)) == pytest.approx((-1e6, 1e6))
    assert drs.resolve(np.full((4, 4), np.nan)) == pytest.approx((-1e6, 1e6))


def test_mode_transition_reset_and_serialisation_contract():
    drs = DisplayRangeState(low_pct=5.0, high_pct=95.0)
    drs.set_manual(1.0, 2.0)
    assert (drs.low_pct, drs.high_pct) == (5.0, 95.0)

    drs.set_percentile(1.0, 99.0)
    arr = np.linspace(0.0, 100.0, 1000)
    assert drs.resolve(arr) == pytest.approx(clip_range_from_array(arr, 1.0, 99.0))

    drs.set_manual(1.0, 5.0)
    manual_dict = drs.to_dict()
    assert manual_dict["mode"] == "manual"
    assert manual_dict["vmin"] == pytest.approx(1.0)
    assert manual_dict["vmax"] == pytest.approx(5.0)

    drs.reset(low_pct=5.0, high_pct=95.0)
    assert drs.to_dict() == {
        "mode": "percentile",
        "low_pct": 5.0,
        "high_pct": 95.0,
        "vmin": None,
        "vmax": None,
    }


def test_manual_limits_affect_rendering_contract():
    arr = np.linspace(0.0, 1.0, 100).reshape(10, 10)
    drs1 = DisplayRangeState()
    drs2 = DisplayRangeState()
    drs1.set_manual(0.0, 0.5)
    drs2.set_manual(0.0, 1.0)

    assert not np.array_equal(
        array_to_uint8(arr, vmin=drs1.resolve(arr)[0], vmax=drs1.resolve(arr)[1]),
        array_to_uint8(arr, vmin=drs2.resolve(arr)[0], vmax=drs2.resolve(arr)[1]),
    )

    narrow = DisplayRangeState()
    narrow.set_manual(4.9, 5.1)
    vmin, vmax = narrow.resolve(arr)
    u8 = array_to_uint8(np.linspace(0.0, 10.0, 100).reshape(10, 10), vmin=vmin, vmax=vmax)
    assert np.sum((u8 == 0) | (u8 == 255)) > u8.size * 0.8


def test_manual_limits_are_separated_contract():
    for vmin, vmax in ((5.0, 5.0), (10.0, 5.0), (1.0, 1.0 + 1e-31)):
        drs = DisplayRangeState()
        drs.set_manual(vmin, vmax)
        assert drs.vmax > drs.vmin
