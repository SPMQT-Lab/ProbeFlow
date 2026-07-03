from __future__ import annotations

import os

import numpy as np
import pytest

from probeflow.gui.dataset_builder.display import flatten_display_array
from probeflow.gui.dataset_builder.tab import DatasetBuilderPanel, DatasetBuilderSidebar
from probeflow.gui.dataset_builder.view_tray import (
    DatasetBuilderCurrentViewTray,
    DatasetBuilderViewTray,
)
from probeflow.gui.styling import THEMES

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - Qt optional in CI fallback
        pytest.skip(f"QApplication unavailable: {exc}")
    app = QApplication.instance()
    if app is not None:
        return app
    return QApplication([])


def test_global_view_tray_expands_and_toggles_flatten(qapp):
    tray = DatasetBuilderViewTray(THEMES["dark"])
    tray.show()
    qapp.processEvents()

    seen: list[bool] = []
    tray.flatten_toggled.connect(seen.append)

    assert tray.is_expanded() is False
    assert tray._body.isVisible() is False

    tray._toggle_btn.click()
    qapp.processEvents()
    assert tray.is_expanded() is True
    assert tray._body.isVisible() is True
    assert tray._toggle_btn.text() == "v Global image view settings"

    tray._flatten_btn.click()
    qapp.processEvents()
    assert tray.is_flatten_enabled() is True
    assert seen == [True]
    assert tray._pct_min.value() == 1.0
    assert tray._pct_max.value() == 99.0
    assert tray._hist_panel._min_w.isVisible() is False
    assert tray._hist_panel._max_w.isVisible() is False
    assert tray._hist_panel._brightness_w.isVisible() is False
    assert tray._hist_panel._contrast_w.isVisible() is False


def test_current_view_tray_arms_and_clears_flatten(qapp):
    tray = DatasetBuilderCurrentViewTray(THEMES["dark"])
    tray.show()
    qapp.processEvents()

    armed: list[bool] = []
    cleared: list[bool] = []
    tray.flatten_requested.connect(armed.append)
    tray.clear_requested.connect(lambda: cleared.append(True))

    assert tray._toggle_btn.text() == "> Current image view settings"

    tray._toggle_btn.click()
    qapp.processEvents()
    assert tray.is_expanded() is True
    assert tray._body.isVisible() is True

    tray._flatten_btn.click()
    qapp.processEvents()
    assert tray.is_flatten_armed() is True
    assert armed == [True]

    tray._clear_btn.click()
    qapp.processEvents()
    assert cleared == [True]


def test_flatten_display_array_is_display_only_and_removes_plane():
    arr = np.add.outer(np.linspace(0.0, 1.0, 48), np.linspace(0.0, 2.0, 48))
    raw = arr.copy()

    flat = flatten_display_array(arr)

    assert np.array_equal(arr, raw)
    assert flat.shape == arr.shape
    assert np.ptp(flat) < np.ptp(arr) * 0.1


def test_dataset_builder_panel_refresh_uses_display_only_flatten(qapp):
    panel = DatasetBuilderPanel(THEMES["dark"], {})
    panel.show()
    qapp.processEvents()
    arr = np.add.outer(np.linspace(0.0, 1.0, 64), np.linspace(0.0, 3.0, 64))
    panel._arr = arr.copy()
    panel._global_view_tray.set_flatten_enabled(True)
    panel._global_view_tray.set_percentile_bounds(5.0, 95.0)

    panel._refresh_display_preview(reset_zoom=True)
    qapp.processEvents()

    assert np.array_equal(panel._arr, arr)
    assert panel._display_arr is not None
    assert panel._display_arr.shape == arr.shape
    assert np.ptp(panel._display_arr) > 0.0
    assert panel._canvas._raw_arr is not None
    assert panel._canvas._raw_arr.shape == arr.shape


def test_dataset_builder_sidebar_places_view_tray_above_counts(qapp):
    sidebar = DatasetBuilderSidebar(THEMES["dark"])
    global_tray = DatasetBuilderViewTray(THEMES["dark"])
    current_tray = DatasetBuilderCurrentViewTray(THEMES["dark"])
    sidebar.set_global_view_tray(global_tray)
    sidebar.set_current_view_tray(current_tray)

    assert sidebar._global_view_host_lay.itemAt(0).widget() is global_tray
    assert sidebar._current_view_host_lay.itemAt(0).widget() is current_tray
    assert sidebar._counts.text() == "Queue not loaded"
    assert global_tray.is_expanded() is False
    assert current_tray.is_expanded() is False


def test_dataset_builder_panel_three_point_flatten_is_display_only(qapp):
    panel = DatasetBuilderPanel(THEMES["dark"], {})
    panel.show()
    qapp.processEvents()

    arr = np.add.outer(np.linspace(0.0, 1.0, 64), np.linspace(0.0, 3.0, 64))
    raw = arr.copy()
    panel._arr = arr.copy()
    panel._base_display_arr = None
    panel._display_arr = None

    panel._refresh_display_preview(reset_zoom=True)
    qapp.processEvents()
    base = panel._display_arr.copy()

    panel._current_view_tray._flatten_btn.click()
    qapp.processEvents()
    panel._on_current_view_point_clicked(0.10, 0.15)
    panel._on_current_view_point_clicked(0.50, 0.20)
    panel._on_current_view_point_clicked(0.80, 0.90)
    qapp.processEvents()

    assert np.array_equal(panel._arr, raw)
    assert panel._current_view_tray.is_flatten_armed() is False
    assert len(panel._canvas._zero_marker_items) == 0
    assert len(panel._current_view_points) == 3
    assert panel._display_arr is not None
    assert not np.array_equal(panel._display_arr, base)
    assert np.ptp(panel._display_arr) < np.ptp(base) or np.ptp(panel._display_arr) < np.ptp(raw)
