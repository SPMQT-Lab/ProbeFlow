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

from probeflow.gui.dialogs.fft_viewer import FFTViewerDialog, crop_to_bounds

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


class _MockFftClick:
    def __init__(self, ax, xdata=0.1, ydata=0.0):
        self.inaxes = ax
        self.button = 1
        self.xdata = xdata
        self.ydata = ydata


class _FakeWheelEvent:
    def __init__(self, delta_y=-120):
        from PySide6.QtCore import QEvent

        self._type = QEvent.Wheel
        self._delta_y = delta_y
        self.accepted = False

    def type(self):
        return self._type

    def angleDelta(self):
        class _Delta:
            def __init__(self, y):
                self._y = y

            def y(self):
                return self._y

        return _Delta(self._delta_y)

    def accept(self):
        self.accepted = True


def test_fft_viewer_gives_fft_canvas_priority(qapp):
    import numpy as np

    dlg = FFTViewerDialog(np.ones((16, 16)), (1e-9, 1e-9))
    try:
        dlg.resize(1180, 820)
        dlg.show()
        qapp.processEvents()

        assert dlg._canvas_real.width() <= 260
        assert dlg._canvas_fft.width() > dlg._canvas_real.width()
        assert dlg._canvas_fft.height() > dlg._canvas_real.height()
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_fft_splitter_can_give_more_height_to_fft(qapp):
    import numpy as np

    dlg = FFTViewerDialog(np.ones((32, 32)), (1e-9, 1e-9))
    try:
        dlg.resize(1180, 820)
        dlg.show()
        qapp.processEvents()

        dlg._fft_splitter.setSizes([260, 420])
        qapp.processEvents()
        compact_h = dlg._canvas_fft.height()

        dlg._fft_splitter.setSizes([640, 80])
        qapp.processEvents()
        expanded_h = dlg._canvas_fft.height()

        assert expanded_h > compact_h
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_focus_fft_hides_reference_and_tools_then_restores(qapp):
    import numpy as np

    dlg = FFTViewerDialog(np.ones((16, 16)), (1e-9, 1e-9))
    try:
        dlg.show()
        qapp.processEvents()

        dlg._focus_fft_btn.setChecked(True)
        qapp.processEvents()
        assert not dlg._left_panel.isVisible()
        assert not dlg._tab_widget.isVisible()

        dlg._focus_fft_btn.setChecked(False)
        qapp.processEvents()
        assert dlg._left_panel.isVisible()
        assert dlg._tab_widget.isVisible()
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_fft_lattice_panel_embeds_in_grid_tab_and_clear_removes_overlay(qapp):
    import numpy as np
    from PySide6.QtWidgets import QApplication

    dlg = FFTViewerDialog(np.ones((16, 16)), (1e-9, 1e-9))
    try:
        dlg.show()
        qapp.processEvents()
        dlg._on_open_fft_lattice()
        qapp.processEvents()

        overlay = dlg._fft_lattice_overlay
        panel = dlg._fft_lattice_panel
        assert overlay is not None
        assert panel is not None
        assert dlg._fft_lattice_dock is None
        assert dlg._clear_grid_btn.isEnabled()
        # Grid tab is at index 1; opening a grid switches to it
        assert dlg._tab_widget.currentIndex() == 1
        assert dlg._grid_tab_index == 1

        app = QApplication.instance()
        assert app is not None
        visible_tool_titles = {
            widget.windowTitle()
            for widget in app.topLevelWidgets()
            if widget.isVisible() and widget is not dlg
        }
        assert "Reciprocal Grid" not in visible_tool_titles

        dlg._on_clear_fft_lattice()
        qapp.processEvents()

        assert dlg._fft_lattice_overlay is None
        assert dlg._fft_lattice_panel is None
        assert not dlg._clear_grid_btn.isEnabled()
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_tab_layout_and_content(qapp):
    """Verify tab order and that key controls live in the expected tabs."""
    import numpy as np
    from PySide6.QtWidgets import QCheckBox, QGroupBox, QPushButton

    dlg = FFTViewerDialog(np.ones((16, 16)), (1e-9, 1e-9))
    try:
        dlg.show()
        qapp.processEvents()

        # Tab order: Inspect(0), Grid(1), Correction(2), Expert(3)
        assert dlg._tab_widget.tabText(0) == "Inspect"
        assert dlg._tab_widget.tabText(1) == "Grid"
        assert dlg._tab_widget.tabText(2) == "Correction"
        assert dlg._tab_widget.tabText(3) == "⚙ Expert"

        # Inspect tab is active on open
        assert dlg._tab_widget.currentIndex() == 0

        # Grid tab contains the Draw Grid button and Known structure group
        grid_tab = dlg._tab_widget.widget(1)
        grid_btn_texts = {btn.text() for btn in grid_tab.findChildren(QPushButton)}
        grid_group_titles = {grp.title() for grp in grid_tab.findChildren(QGroupBox)}
        assert "Draw Grid" in grid_btn_texts
        assert "Clear Grid" in grid_btn_texts
        assert "Known structure" in grid_group_titles
        assert "Compare with known structure" in grid_group_titles

        # Correction tab: correction label + preview + apply buttons
        corr_tab = dlg._tab_widget.widget(2)
        corr_btn_texts = {btn.text() for btn in corr_tab.findChildren(QPushButton)}
        assert "Preview corrected image" in corr_btn_texts
        assert "Apply correction" in corr_btn_texts

        # Expert tab: no grid panel section, has scanner calibration
        expert_tab = dlg._tab_widget.widget(3)
        expert_btn_texts = {btn.text() for btn in expert_tab.findChildren(QPushButton)}
        expert_cb_texts = {cb.text() for cb in expert_tab.findChildren(QCheckBox)}
        assert "Detect peaks" in expert_btn_texts
        assert "Pick peaks" in expert_cb_texts
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_fft_known_structure_updates_shell_and_target_controls(qapp):
    import numpy as np
    from probeflow.gui.lattice_correction_ui import KnownStructure

    dlg = FFTViewerDialog(np.ones((16, 16)), (1e-9, 1e-9))
    try:
        dlg.show()
        qapp.processEvents()

        structure = KnownStructure("Square 0.5 nm", "square", 0.5, 0.5, 90.0, "nm")
        dlg._known_structures = [structure]
        dlg._refresh_structure_combo(structure.name)
        dlg._on_structure_selected(0)
        qapp.processEvents()

        assert dlg._active_known_structure == structure
        assert dlg._bragg_sym_combo.currentText() == "Square"
        assert dlg._bragg_unit_combo.currentText() == "nm"
        assert dlg._bragg_a_spin.value() == pytest.approx(0.5)
        assert dlg._fft_ideal_combo.currentText() == "Square"
        assert dlg._fft_ideal_a_spin.value() == pytest.approx(0.5)
        assert dlg._fft_ideal_b_spin.value() == pytest.approx(0.5)
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_wheel_over_fft_grid_spinbox_scrolls_panel_without_changing_value(qapp):
    import numpy as np
    from probeflow.gui.no_wheel import _no_wheel_filter

    dlg = FFTViewerDialog(np.ones((16, 16)), (1e-9, 1e-9))
    try:
        dlg.show()
        qapp.processEvents()
        dlg._on_open_fft_lattice()
        qapp.processEvents()

        panel = dlg._fft_lattice_panel
        spin = panel._g1_spin
        old_value = spin.value()
        event = _FakeWheelEvent(delta_y=-120)

        assert _no_wheel_filter().eventFilter(spin, event)
        assert event.accepted
        assert spin.value() == old_value
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_fft_grid_correction_preview_and_apply_hooks(qapp):
    import numpy as np

    arr = np.arange(64, dtype=float).reshape(8, 8)
    previewed = []
    applied = []

    dlg = FFTViewerDialog(
        arr,
        (1e-9, 1e-9),
        get_image_fn=lambda: arr,
        preview_image_fn=previewed.append,
        clear_preview_fn=lambda: previewed.append(None),
        apply_correction_fn=lambda op, params: applied.append((op, params)),
    )
    try:
        dlg.show()
        qapp.processEvents()
        dlg._on_open_fft_lattice()
        qapp.processEvents()

        assert dlg._fft_correction is not None
        dlg._on_fft_preview_correction()
        assert previewed == []
        assert dlg._fft_preview_frame.isVisible()
        assert dlg._fft_preview_active

        dlg._on_fft_apply_correction()
        assert applied
        assert applied[0][0] == "affine_lattice_correction"
        assert applied[0][1]["source"] == "fft_reciprocal_grid"
        assert applied[0][1]["known_structure"]["name"] == "Hexagonal 2.46 Å"
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


def test_predicted_lattice_clicks_do_not_pick_unless_pick_mode_enabled(qapp):
    import numpy as np

    dlg = FFTViewerDialog(np.ones((16, 16)), (1e-9, 1e-9))
    try:
        dlg.show()
        qapp.processEvents()
        dlg._bragg_enable_cb.setChecked(True)
        dlg._bragg_snap_cb.setChecked(False)
        qapp.processEvents()

        event = _MockFftClick(dlg._ax_fft, xdata=0.1, ydata=0.0)
        dlg._on_press(event)
        assert dlg._calib_picks == []

        dlg._bragg_pick_cb.setChecked(True)
        dlg._on_press(event)
        assert len(dlg._calib_picks) == 1
    finally:
        dlg.close()
        dlg.deleteLater()
        qapp.processEvents()


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


# ── crop_to_bounds ───────────────────────────────────────────────────────────
class TestCropToBounds:
    def test_crop_shape_is_inclusive(self):
        import numpy as np
        a = np.arange(100.0).reshape(10, 10)
        cropped, _ = crop_to_bounds(a, (2, 6, 1, 4), (10e-9, 10e-9))
        # rows 2..6 inclusive = 5, cols 1..4 inclusive = 4
        assert cropped.shape == (5, 4)
        np.testing.assert_array_equal(cropped, a[2:7, 1:5])

    def test_scan_range_scales_preserving_pixel_size(self):
        import numpy as np
        a = np.zeros((10, 10))
        # 10 nm over 10 px = 1 nm/px on both axes.
        _, (w, h) = crop_to_bounds(a, (2, 6, 1, 4), (10e-9, 10e-9))
        # crop is 4 cols wide, 5 rows tall → 4 nm × 5 nm, still 1 nm/px.
        assert w == pytest.approx(4e-9)
        assert h == pytest.approx(5e-9)
        crop_nx, crop_ny = 4, 5
        assert (w / crop_nx) == pytest.approx(10e-9 / 10)  # px size x preserved
        assert (h / crop_ny) == pytest.approx(10e-9 / 10)  # px size y preserved

    def test_non_square_image_uses_correct_axes(self):
        import numpy as np
        # 20 rows (Ny), 10 cols (Nx); range = (width_x=5nm, height_y=20nm).
        a = np.zeros((20, 10))
        _, (w, h) = crop_to_bounds(a, (0, 9, 0, 4), (5e-9, 20e-9))
        # cols 0..4 = 5 of 10 → width 2.5 nm; rows 0..9 = 10 of 20 → height 10 nm
        assert w == pytest.approx(2.5e-9)
        assert h == pytest.approx(10e-9)

    def test_full_bounds_returns_full_range(self):
        import numpy as np
        a = np.zeros((8, 12))
        cropped, (w, h) = crop_to_bounds(a, (0, 7, 0, 11), (12e-9, 8e-9))
        assert cropped.shape == (8, 12)
        assert w == pytest.approx(12e-9)
        assert h == pytest.approx(8e-9)

    def test_bounds_clipped_to_array(self):
        import numpy as np
        a = np.zeros((10, 10))
        # Over-range bounds get clipped to the array extent.
        cropped, _ = crop_to_bounds(a, (-5, 99, -5, 99), (10e-9, 10e-9))
        assert cropped.shape == (10, 10)

    def test_degenerate_bounds_fall_back_to_full(self):
        import numpy as np
        a = np.arange(100.0).reshape(10, 10)
        # r1 < r0 after the swap is impossible here, but c1<c0 triggers fallback.
        cropped, (w, h) = crop_to_bounds(a, (5, 5, 8, 2), (10e-9, 10e-9))
        np.testing.assert_array_equal(cropped, a)
        assert (w, h) == (10e-9, 10e-9)

    def test_rejects_non_2d(self):
        import numpy as np
        with pytest.raises(ValueError):
            crop_to_bounds(np.zeros((4, 4, 3)), (0, 1, 0, 1), (1e-9, 1e-9))


# ── FFT source selector (whole image vs active ROI) ──────────────────────────
class TestFftSourceSelector:
    def test_no_roi_disables_roi_entry_and_defaults_whole_image(self, qapp):
        import numpy as np
        dlg = FFTViewerDialog(np.ones((32, 32)), (8e-9, 8e-9))
        try:
            assert dlg._fft_source == "whole_image"
            assert dlg._arr.shape == (32, 32)
            # ROI entry disabled when no ROI bounds were supplied.
            assert not dlg._fft_source_combo.model().item(1).isEnabled()
        finally:
            dlg.close()
            dlg.deleteLater()
            qapp.processEvents()

    def test_roi_passed_defaults_whole_image_but_enables_roi(self, qapp):
        import numpy as np
        dlg = FFTViewerDialog(
            np.ones((32, 32)), (8e-9, 8e-9),
            roi_bounds_px=(4, 19, 8, 23), roi_id="r1", roi_name="ROI 1",
        )
        try:
            # Default is still whole image (opt-in ROI).
            assert dlg._fft_source == "whole_image"
            assert dlg._arr.shape == (32, 32)
            assert dlg._fft_source_combo.model().item(1).isEnabled()
        finally:
            dlg.close()
            dlg.deleteLater()
            qapp.processEvents()

    def test_switching_to_roi_crops_array_and_scales_range(self, qapp):
        import numpy as np
        # 32x32 over 8 nm → 0.25 nm/px.
        dlg = FFTViewerDialog(
            np.random.default_rng(0).normal(size=(32, 32)), (8e-9, 8e-9),
            roi_bounds_px=(4, 19, 8, 23), roi_id="r1", roi_name="ROI 1",
        )
        try:
            dlg._fft_source_combo.setCurrentIndex(1)  # Active ROI
            qapp.processEvents()
            assert dlg._fft_source == "active_roi"
            # rows 4..19 = 16, cols 8..23 = 16
            assert dlg._arr.shape == (16, 16)
            # pixel size preserved: 16 * 0.25 nm = 4 nm on each axis
            assert dlg._scan_range_m[0] == pytest.approx(4e-9)
            assert dlg._scan_range_m[1] == pytest.approx(4e-9)
            # q-grid reflects the crop (Nyquist unchanged, finer resolution).
            assert dlg._qx is not None and dlg._qx.size == 16

            # Switching back restores the full source.
            dlg._fft_source_combo.setCurrentIndex(0)
            qapp.processEvents()
            assert dlg._fft_source == "whole_image"
            assert dlg._arr.shape == (32, 32)
            assert dlg._scan_range_m[0] == pytest.approx(8e-9)
        finally:
            dlg.close()
            dlg.deleteLater()
            qapp.processEvents()

    def test_pixel_spacing_preserved_between_sources(self, qapp):
        import numpy as np
        dlg = FFTViewerDialog(
            np.random.default_rng(1).normal(size=(32, 32)), (8e-9, 8e-9),
            roi_bounds_px=(4, 19, 8, 23),
        )
        try:
            # The q-grid is fftfreq(N, d) with d = scan_range/N (in nm). The
            # physical invariant across sources is the pixel spacing d (hence the
            # true Nyquist 1/2d) — not the discrete max bin, which differs for
            # even N because the Nyquist sample sits at the negative end.
            def _d_nm(dlg):
                nx = dlg._arr.shape[1]
                return float(dlg._scan_range_m[0]) * 1e9 / nx
            whole_d = _d_nm(dlg)
            dlg._fft_source_combo.setCurrentIndex(1)
            qapp.processEvents()
            roi_d = _d_nm(dlg)
            assert roi_d == pytest.approx(whole_d)
            # q-resolution (grid step) is finer for the smaller ROI extent.
            whole_dq = 1.0 / (32 * whole_d)
            roi_dq = 1.0 / (16 * roi_d)
            assert roi_dq > whole_dq
        finally:
            dlg.close()
            dlg.deleteLater()
            qapp.processEvents()
