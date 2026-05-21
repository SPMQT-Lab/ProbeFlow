"""Tests for pure utility functions in the FFT viewer dialog.

These tests target the zoom-span clamping helper ``_interval_with_min_span``
(a ``@staticmethod`` on ``FFTViewerDialog``) and the logic of
``_minimum_fft_spans`` — which is normally an instance method that depends on
heavy Qt state.  For the latter we replicate the formula as a pure helper so
the maths can be tested without constructing a real dialog.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from probeflow.gui.dialogs.fft_viewer import FFTViewerDialog

_interval_with_min_span = FFTViewerDialog._interval_with_min_span
_scroll_has_zoom_modifier = FFTViewerDialog._scroll_has_zoom_modifier


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")

    app = QApplication.instance()
    if app is not None:
        return app
    try:
        return QApplication([])
    except Exception as exc:
        pytest.skip(f"QApplication unavailable: {exc}")


class _MockScrollEvent:
    def __init__(self, key=""):
        self.key = key


# ── _interval_with_min_span ──────────────────────────────────────────────────

class TestIntervalWithMinSpan:
    def test_normal_case_unchanged(self):
        # span = 10 >= min = 5, returned as-is
        assert _interval_with_min_span(0.0, 10.0, 5.0) == (0.0, 10.0)

    def test_interval_too_small_is_centred_and_expanded(self):
        a, b = _interval_with_min_span(5.0, 5.01, 2.0)
        # centre preserved, total width = 2.0
        assert (a + b) * 0.5 == pytest.approx(5.005)
        assert (b - a) == pytest.approx(2.0)

    def test_zero_width_interval(self):
        # (3.0, 3.0) with min 1.0 -> centred at 3.0, width 1.0
        a, b = _interval_with_min_span(3.0, 3.0, 1.0)
        assert a == pytest.approx(2.5)
        assert b == pytest.approx(3.5)

    def test_negative_span_above_minimum_unchanged(self):
        # span = -10, |span| = 10 >= 2, so returned untouched (descending order)
        assert _interval_with_min_span(10.0, 0.0, 2.0) == (10.0, 0.0)

    def test_min_span_exactly_equal(self):
        # |span| == minimum → boundary case, returned unchanged
        assert _interval_with_min_span(0.0, 1.0, 1.0) == (0.0, 1.0)

    def test_very_small_positive_interval_enforced(self):
        a, b = _interval_with_min_span(4.99, 5.01, 1.0)
        assert a == pytest.approx(4.5)
        assert b == pytest.approx(5.5)

    @pytest.mark.parametrize(
        "a,b,minimum",
        [
            (0.0, 0.0, 1.0),
            (1.0, 1.0001, 0.5),
            (-2.0, -1.99, 0.25),
            (0.0, 10.0, 5.0),
            (10.0, 0.0, 2.0),
            (0.0, 1.0, 1.0),
        ],
    )
    def test_result_span_at_least_minimum(self, a, b, minimum):
        ra, rb = _interval_with_min_span(a, b, minimum)
        assert abs(rb - ra) >= minimum - 1e-12

    @pytest.mark.parametrize(
        "a,b,minimum",
        [
            (5.0, 5.01, 2.0),
            (3.0, 3.0, 1.0),
            (4.99, 5.01, 1.0),
            (-1.0, -0.999, 0.5),
        ],
    )
    def test_centre_preserved_when_expanded(self, a, b, minimum):
        ra, rb = _interval_with_min_span(a, b, minimum)
        assert (ra + rb) * 0.5 == pytest.approx((a + b) * 0.5)


# ── _minimum_fft_spans formula (pure-function port) ──────────────────────────

def _compute_minimum_spans(arr_shape, qx, qy):
    """Pure-function replica of FFTViewerDialog._minimum_fft_spans for testing."""
    Ny, Nx = arr_shape[:2]
    full_x = abs(float(qx[-1]) - float(qx[0]))
    full_y = abs(float(qy[-1]) - float(qy[0]))
    return (
        max(full_x * 1e-4, full_x * 4.0 / max(1, Nx)),
        max(full_y * 1e-4, full_y * 4.0 / max(1, Ny)),
    )


class TestMinimumFFTSpans:
    def test_256x256_qx_0_to_50(self):
        # min_x = max(50 * 1e-4, 50 * 4 / 256) = max(0.005, 0.78125) = 0.78125
        qx = [0.0, 50.0]
        qy = [0.0, 50.0]
        mx, my = _compute_minimum_spans((256, 256), qx, qy)
        assert mx == pytest.approx(50.0 * 4.0 / 256.0)
        assert mx == pytest.approx(0.78125)
        assert my == pytest.approx(0.78125)

    def test_1x1_edge_case_does_not_divide_by_zero(self):
        # Nx = Ny = 1 → max(1, 1) = 1, no zero division
        qx = [0.0, 10.0]
        qy = [0.0, 10.0]
        mx, my = _compute_minimum_spans((1, 1), qx, qy)
        # 4 px / 1 px = 4 → floor = 4 * full_span
        assert mx == pytest.approx(40.0)
        assert my == pytest.approx(40.0)

    def test_1024_pixel_image(self):
        # min_x = max(50 * 1e-4, 50 * 4 / 1024) = max(0.005, 0.1953125) = 0.1953125
        qx = [0.0, 50.0]
        qy = [0.0, 50.0]
        mx, my = _compute_minimum_spans((1024, 1024), qx, qy)
        assert mx == pytest.approx(50.0 * 4.0 / 1024.0)
        assert mx == pytest.approx(0.1953125)

    @pytest.mark.parametrize(
        "shape,qx,qy",
        [
            ((256, 256), [0.0, 50.0], [0.0, 50.0]),
            ((1024, 1024), [0.0, 50.0], [0.0, 50.0]),
            ((128, 64), [-10.0, 10.0], [-5.0, 5.0]),
            ((1, 1), [0.0, 1.0], [0.0, 1.0]),
        ],
    )
    def test_minimum_span_is_positive(self, shape, qx, qy):
        mx, my = _compute_minimum_spans(shape, qx, qy)
        assert mx > 0
        assert my > 0

    @pytest.mark.parametrize(
        "shape,qx,qy",
        [
            ((256, 256), [0.0, 50.0], [0.0, 50.0]),
            ((1024, 1024), [0.0, 50.0], [0.0, 50.0]),
            ((128, 64), [-10.0, 10.0], [-5.0, 5.0]),
        ],
    )
    def test_floor_is_at_least_four_pixels(self, shape, qx, qy):
        Ny, Nx = shape
        full_x = abs(qx[-1] - qx[0])
        full_y = abs(qy[-1] - qy[0])
        mx, my = _compute_minimum_spans(shape, qx, qy)
        # The floor should be ≥ (4 / N) × full_span in each axis.
        assert mx >= full_x * 4.0 / max(1, Nx) - 1e-12
        assert my >= full_y * 4.0 / max(1, Ny) - 1e-12


# ── _scroll_has_zoom_modifier ────────────────────────────────────────────────

class TestScrollHasZoomModifier:
    def test_empty_key_returns_false(self, qapp):
        # In headless test context, QApplication.keyboardModifiers() returns NoModifier,
        # so the result depends only on the key string (which is empty → False).
        assert _scroll_has_zoom_modifier(_MockScrollEvent(key="")) is False

    def test_ctrl_token(self):
        assert _scroll_has_zoom_modifier(_MockScrollEvent(key="ctrl")) is True

    def test_control_token(self):
        assert _scroll_has_zoom_modifier(_MockScrollEvent(key="control")) is True

    def test_cmd_token(self):
        assert _scroll_has_zoom_modifier(_MockScrollEvent(key="cmd")) is True

    def test_command_token(self):
        assert _scroll_has_zoom_modifier(_MockScrollEvent(key="command")) is True

    def test_meta_token(self):
        assert _scroll_has_zoom_modifier(_MockScrollEvent(key="meta")) is True

    def test_super_token(self):
        assert _scroll_has_zoom_modifier(_MockScrollEvent(key="super")) is True

    def test_case_insensitive_uppercase(self):
        assert _scroll_has_zoom_modifier(_MockScrollEvent(key="CTRL")) is True

    def test_shift_is_not_a_zoom_modifier(self, qapp):
        assert _scroll_has_zoom_modifier(_MockScrollEvent(key="shift")) is False

    def test_alt_is_not_a_zoom_modifier(self, qapp):
        assert _scroll_has_zoom_modifier(_MockScrollEvent(key="alt")) is False

    def test_compound_key_string_with_ctrl(self):
        assert _scroll_has_zoom_modifier(_MockScrollEvent(key="ctrl+shift")) is True

    def test_none_key_returns_false(self, qapp):
        # None is normalised to "" by `getattr(event, "key", "") or ""`.
        assert _scroll_has_zoom_modifier(_MockScrollEvent(key=None)) is False
