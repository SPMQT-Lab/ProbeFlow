"""Tests for spectroscopy plot helpers that do not need Qt."""

from __future__ import annotations

import numpy as np

from probeflow.analysis.spec_plot import choose_display_unit, spec_position_to_pixel


def test_choose_display_unit_scales_common_spectroscopy_domains():
    exact_cases = [
        ("m", np.array([5e-12, 10e-12, 15e-12]), "pm", None),
        ("A", np.array([10e-12, 20e-12, 30e-12]), "pA", None),
        ("A", np.array([1e-9, 2e-9, 3e-9]), "nA", None),
        ("V", np.array([0.01, 0.02, 0.03]), "mV", None),
        ("V", np.array([1.0, 2.0, 3.0]), "V", 1.0),
        ("Hz", np.array([1e-9, 2e-9]), "Hz", 1.0),
        ("", np.array([1.0, 2.0]), "", 1.0),
    ]

    for unit, values, expected_unit, expected_scale in exact_cases:
        scale, chosen_unit = choose_display_unit(unit, values)
        assert chosen_unit == expected_unit
        if expected_scale is not None:
            assert scale == expected_scale

    for values in (np.array([1e-9, 2e-9, 3e-9]), np.array([30e-9, 50e-9, 70e-9])):
        scale, chosen_unit = choose_display_unit("m", values)
        assert chosen_unit in {"Å", "nm"}
        assert 0.1 <= np.median(np.abs(values)) * scale < 1000


def test_choose_display_unit_handles_zero_empty_and_none_without_misleading_scale():
    assert choose_display_unit("A", np.zeros(10)) == (1e12, "pA")
    assert choose_display_unit("m", np.zeros(10)) == (1e9, "nm")
    assert choose_display_unit("m", np.array([])) == (1.0, "m")
    assert choose_display_unit("m", None) == (1.0, "m")


def test_spec_position_to_pixel_maps_scan_edges_offsets_and_rotation():
    width = height = 1e-7
    cases = [
        ((0.0, 0.0, (64, 64), (width, height), (0.0, 0.0), 0.0), (0.5, 0.5)),
        ((-width / 2, height / 2, (64, 64), (width, height), (0.0, 0.0), 0.0), (0.0, 0.0)),
        ((width / 2, -height / 2, (64, 64), (width, height), (0.0, 0.0), 0.0), (1.0, 1.0)),
        ((100e-9, 50e-9, (64, 64), (width, height), (100e-9, 50e-9), 0.0), (0.5, 0.5)),
        ((width / 4, 0.0, (64, 64), (width, height), (0.0, 0.0), 90.0), (0.5, 0.75)),
        ((0.0, 0.0, (64, 64), (width, height)), (0.5, 0.5)),
    ]

    for args, expected in cases:
        result = spec_position_to_pixel(*args)
        assert result is not None
        np.testing.assert_allclose(result, expected, atol=1e-9)


def test_spec_position_to_pixel_rejects_positions_outside_scan_frame():
    width = height = 1e-7
    assert spec_position_to_pixel(1e-3, 0.0, (64, 64), (width, height), (0.0, 0.0), 0.0) is None
    assert spec_position_to_pixel(0.0, 0.0, (64, 64), (width, height), (100e-9, 50e-9), 0.0) is None
