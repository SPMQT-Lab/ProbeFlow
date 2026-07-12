"""Tests for probeflow.measurements.formatting.

Consolidates the six parallel metre-to-display formatters that the
2026-05-27 deep review (arch-backend #5) flagged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from probeflow.measurements.formatting import (
    choose_display_unit,
    choose_length_unit,
    format_height_m,
    format_length_m,
    format_period_m,
    lookup_unit_scale,
    scale_length_m,
)


class TestChooseLengthUnit:
    def test_picoscale_returns_pm(self):
        scale, unit = choose_length_unit(1e-12)
        assert unit == "pm"
        assert scale == 1e12

    def test_angstrom_scale_returns_aa(self):
        scale, unit = choose_length_unit(2e-10)  # 2 Å
        assert unit == "Å"
        assert scale == 1e10

    def test_nano_scale_returns_nm(self):
        scale, unit = choose_length_unit(5e-9)  # 5 nm
        assert unit == "nm"
        assert scale == 1e9

    def test_micron_scale_returns_um(self):
        scale, unit = choose_length_unit(2e-6)
        assert unit == "µm"
        assert scale == 1e6

    def test_nan_returns_nm_default(self):
        scale, unit = choose_length_unit(float("nan"))
        assert unit == "nm"


class TestScaleLengthM:
    def test_returns_scaled_value(self):
        v, u = scale_length_m(2.5e-9)
        assert u == "nm"
        assert v == pytest.approx(2.5)

    def test_nan_input(self):
        v, u = scale_length_m(float("nan"))
        assert math.isnan(v)


class TestFormatLengthM:
    def test_default_precision_nm(self):
        # 2.5 nm → "2.5 nm" at default precision=3.
        assert format_length_m(2.5e-9) == "2.5 nm"

    def test_sub_nm_returns_angstroms(self):
        # 0.246 nm = 2.46 Å, and abs(0.246e-9) < 1e-9 → Å.
        assert format_length_m(0.246e-9) == "2.46 Å"

    def test_explicit_precision(self):
        # 0.24576 nm = 2.4576 Å → 4 sig figs → "2.458 Å"
        assert format_length_m(0.24576e-9, precision=4) == "2.458 Å"

    def test_nan_default_repr(self):
        assert format_length_m(float("nan")) == "—"

    def test_custom_nan_repr(self):
        assert format_length_m(float("nan"), nan_repr="N/A") == "N/A"

    def test_negative_value_preserves_sign(self):
        # Choose-unit uses abs(), but the formatted scalar is signed.
        s = format_length_m(-2.5e-9)
        assert s.startswith("-")


class TestFormatPeriodM:
    def test_split_value_unit_sub_nm(self):
        # Sub-nm → Å
        val, unit = format_period_m(0.246e-9)
        assert val == "2.46"
        assert unit == "Å"

    def test_split_value_unit_nm(self):
        val, unit = format_period_m(2.5e-9)
        assert val == "2.5"
        assert unit == "nm"

    def test_picoscale(self):
        val, unit = format_period_m(3e-12)
        assert unit == "pm"

    def test_nan(self):
        val, unit = format_period_m(float("nan"))
        assert val == "—"
        assert unit == ""


class TestFormatHeightM:
    def test_height_formatter_is_length_alias(self):
        # format_height_m is an alias for length formatting today.
        assert format_height_m(1.5e-9) == format_length_m(1.5e-9)


class TestChooseDisplayUnit:
    def test_chooses_nm_for_nm_scale_data(self):
        # Median of [10, 50, 100] nm = 50 nm; Å scale would give 500 (still
        # < 1000), but nm scale is more readable.  The algorithm picks
        # the SMALLEST prefix whose scaled magnitude is in [0.1, 1000],
        # so Å actually wins here too — adjust the test to use values
        # large enough that Å overflows.
        scale, unit = choose_display_unit("m", np.array([100e-9, 500e-9, 800e-9]))
        # Median 500 nm; Å scale gives 5000 → out of range; nm gives 500 ✓
        assert unit == "nm"
        assert scale == 1e9

    def test_empty_array_returns_si_unit(self):
        scale, unit = choose_display_unit("m", np.array([]))
        assert unit == "m"
        assert scale == 1.0

    def test_all_zero_returns_zero_default(self):
        scale, unit = choose_display_unit("m", np.zeros(10))
        assert unit == "nm"

    def test_nonfinite_samples_do_not_hide_finite_scale(self):
        values = np.array([200e-9, 500e-9, np.nan, np.inf, -np.inf])

        scale, unit = choose_display_unit("m", values)

        assert unit == "nm"
        assert scale == 1e9

    def test_only_nonfinite_samples_fall_back_to_si_unit(self):
        scale, unit = choose_display_unit("A", np.array([np.nan, np.inf]))

        assert unit == "A"
        assert scale == 1.0

    def test_unknown_si_unit_pass_through(self):
        scale, unit = choose_display_unit("Hz", np.array([1.0]))
        assert unit == "Hz"
        assert scale == 1.0


class TestLookupUnitScale:
    def test_known_label(self):
        result = lookup_unit_scale("m", "nm")
        assert result == (1e9, "nm")

    def test_unknown_label_returns_none(self):
        assert lookup_unit_scale("m", "bogus") is None

    def test_unknown_si_unit_returns_none(self):
        assert lookup_unit_scale("Hz", "kHz") is None


class TestLegacyWrappersDelegate:
    """Verify the legacy ``_fmt_m``/``_fmt_z``/``_choose_unit`` shims
    are thin re-exports of the consolidated helpers (review
    arch-backend #5)."""

    def test_simple_measurements_fmt_m(self):
        from probeflow.analysis.simple_measurements import _fmt_m
        assert _fmt_m(2.5e-9) == scale_length_m(2.5e-9)

    def test_line_periodicity_format_period(self):
        from probeflow.analysis.line_periodicity import format_period
        assert format_period(0.246e-9) == format_period_m(0.246e-9)

    def test_roi_statistics_fmt_z_metres(self):
        from probeflow.analysis.roi_statistics import _fmt_z
        assert _fmt_z(1.5e-9, "m") == scale_length_m(1.5e-9)

    def test_lattice_grid_choose_unit(self):
        from probeflow.analysis.lattice_grid import _choose_unit
        assert _choose_unit(2.5e-9) == choose_length_unit(2.5e-9)

    def test_spec_plot_reexports_choose_display_unit(self):
        # spec_plot.choose_display_unit must still work; it now
        # re-imports from measurements.formatting.
        from probeflow.analysis.spec_plot import choose_display_unit as sp_choose
        from probeflow.measurements.formatting import choose_display_unit as mf_choose
        # Same object — re-exported, not duplicated
        assert sp_choose is mf_choose
