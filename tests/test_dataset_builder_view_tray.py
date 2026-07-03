from __future__ import annotations

import os

import numpy as np
import pytest

from probeflow.gui.dataset_builder.display import flatten_display_array
from probeflow.gui.dataset_builder.tab import DatasetBuilderPanel, DatasetBuilderSidebar
from probeflow.gui.dataset_builder.view_tray import DatasetBuilderViewTray
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


def test_view_tray_expands_and_toggles_flatten(qapp):
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
    assert tray._toggle_btn.text() == "v View"

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
    panel._view_tray.set_flatten_enabled(True)
    panel._view_tray.set_percentile_bounds(5.0, 95.0)

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
    tray = DatasetBuilderViewTray(THEMES["dark"])
    sidebar.set_view_tray(tray)

    assert sidebar._view_host_lay.itemAt(0).widget() is tray
    assert sidebar._counts.text() == "Queue not loaded"
    assert tray.is_expanded() is False
