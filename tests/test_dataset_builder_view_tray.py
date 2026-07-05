from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeySequence, QMouseEvent, QShortcut, QWheelEvent, QPixmap, QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea, QSplitter

from probeflow.core.mask import ImageMask
from probeflow.dataset_builder.quickseg import QuickSegPrepared
from probeflow.gui.dataset_builder.display import flatten_display_array
from probeflow.gui.dataset_builder.canvas import DatasetBuilderCanvas
from probeflow.gui.dataset_builder.tab import DatasetBuilderPanel, DatasetBuilderSidebar
from probeflow.gui.dataset_builder.view_tray import (
    DatasetBuilderCurrentViewTray,
    DatasetBuilderViewTray,
)
from probeflow.gui.dataset_builder.quickseg_controls import QuickSegControlsWidget
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


def test_dataset_builder_status_shortcut_helper_advances_only_on_success(qapp):
    panel = DatasetBuilderPanel(THEMES["dark"], {})
    panel.show()
    qapp.processEvents()

    calls: list[str] = []

    def ok_set_status(status: str) -> bool:
        calls.append(status)
        return True

    panel._set_status = ok_set_status  # type: ignore[method-assign]
    panel.next_item = lambda: calls.append("next")  # type: ignore[method-assign]

    panel._save_status_and_next("accepted")
    assert calls == ["accepted", "next"]

    calls.clear()

    def fail_set_status(status: str) -> bool:
        calls.append(status)
        return False

    panel._set_status = fail_set_status  # type: ignore[method-assign]
    panel._save_status_and_next("uncertain")
    assert calls == ["uncertain"]


def test_dataset_builder_shortcuts_cover_clear_undo_and_brush_size(qapp):
    panel = DatasetBuilderPanel(THEMES["dark"], {})
    panel.show()
    qapp.processEvents()

    keys = {sc.key().toString() for sc in panel.findChildren(QShortcut)}
    assert "Z" in keys
    assert "V" in keys
    assert "C" in keys
    assert "E" in keys
    assert "R" in keys
    assert QKeySequence(QKeySequence.Undo).toString() in keys
    assert "W" not in keys


def test_quickseg_controls_expose_basic_state_and_buttons(qapp):
    controls = QuickSegControlsWidget(THEMES["dark"])
    controls.show()
    qapp.processEvents()

    seen: list[str] = []
    controls.apply_requested.connect(lambda: seen.append("apply"))
    controls.save_next_requested.connect(lambda: seen.append("save_next"))

    controls.set_current_label(7)
    controls.set_seed_mode_status("Add seed mode")
    controls.set_result_status("Watershed ready")

    controls._apply_btn.click()
    controls._save_next_btn.click()
    qapp.processEvents()

    assert controls.current_label() == 7
    assert controls._seed_mode_lbl.text() == "Add seed mode"
    assert controls._result_lbl.text() == "Watershed ready"
    assert hasattr(controls, "_denoise_strength")
    assert hasattr(controls, "_smooth_along_scan")
    assert hasattr(controls, "_barrier_strength")
    assert not hasattr(controls, "_tv_iters")
    assert not hasattr(controls, "_gaussian_order")
    assert seen == ["apply", "save_next"]


def test_dataset_builder_right_pane_is_scrollable_and_advanced_expands(qapp):
    panel = DatasetBuilderPanel(THEMES["dark"], {})
    panel.show()
    qapp.processEvents()

    splitter = panel.findChild(QSplitter)
    assert splitter is not None
    right_scroll = splitter.widget(2)
    assert isinstance(right_scroll, QScrollArea)

    idx = panel._task_combo.findData("terrace_segmentation")
    assert idx >= 0
    panel._task_combo.setCurrentIndex(idx)
    qapp.processEvents()

    controls = panel._quickseg_controls
    assert controls is not None
    controls._advanced._toggle.click()
    qapp.processEvents()

    assert controls._advanced._body.isVisible() is True


def test_dataset_builder_quickseg_params_persist_and_reset(qapp, monkeypatch):
    monkeypatch.setattr("probeflow.gui.dataset_builder.tab.save_config", lambda cfg: None)
    cfg: dict = {}
    panel = DatasetBuilderPanel(THEMES["dark"], cfg)
    idx = panel._task_combo.findData("terrace_segmentation")
    assert idx >= 0
    panel._task_combo.setCurrentIndex(idx)
    qapp.processEvents()

    controls = panel._quickseg_controls
    assert controls is not None
    controls._smooth_along_scan.setValue(2.5)
    controls._denoise_strength.setValue(0.08)
    panel._persist_quickseg_params()

    assert cfg["dataset_builder_quickseg_params"]["smooth_along_scan"] == 2.5
    assert cfg["dataset_builder_quickseg_params"]["denoise_strength"] == 0.08

    panel2 = DatasetBuilderPanel(THEMES["dark"], cfg)
    idx2 = panel2._task_combo.findData("terrace_segmentation")
    assert idx2 >= 0
    panel2._task_combo.setCurrentIndex(idx2)
    qapp.processEvents()

    controls2 = panel2._quickseg_controls
    assert controls2 is not None
    assert controls2._smooth_along_scan.value() == 2.5
    assert controls2._denoise_strength.value() == 0.08

    controls2._reset_btn.click()
    qapp.processEvents()

    assert controls2._smooth_along_scan.value() == 1.2
    assert controls2._denoise_strength.value() == 0.04


def test_dataset_builder_load_current_does_not_return_early(qapp):
    panel = DatasetBuilderPanel(THEMES["dark"], {})
    panel.show()
    qapp.processEvents()

    class DummyItem:
        source_path = os.path.join(os.path.dirname(__file__), "..", "test_data", "createc_scan_close_100nm.dat")
        plane_index = 0
        display_id = "dummy_plane0"
        status = "blank"

    panel._queue = [DummyItem()]
    panel._current_index = 0
    panel._load_current()
    qapp.processEvents()

    assert panel._arr is not None or panel._canvas_status.text() != "No scan loaded"


def test_dataset_builder_plane_change_reuses_cached_indexed_items(qapp):
    panel = DatasetBuilderPanel(THEMES["dark"], {})
    panel.show()
    qapp.processEvents()

    calls: list[str] = []
    panel._hydrate_queue_from_indexed_items = lambda: calls.append("hydrate")  # type: ignore[method-assign]
    panel._source_entry.setText("C:/tmp/dataset")
    panel._indexed_items = [SimpleNamespace(path="dummy", item_type="scan")]

    panel._on_plane_changed()

    assert calls == ["hydrate"]


def test_dataset_builder_ctrl_z_undos_one_brush_stroke(qapp):
    panel = DatasetBuilderPanel(THEMES["dark"], {})
    panel.show()
    qapp.processEvents()

    arr = np.zeros((24, 24), dtype=float)
    panel._arr = arr.copy()
    panel._queue = [
        SimpleNamespace(
            display_id="sample_plane0",
            status="blank",
            source_path=__file__,
            plane_index=0,
        )
    ]
    panel._current_index = 0
    panel._current_mask = ImageMask.new(
        np.zeros(arr.shape, dtype=bool),
        method="manual",
        parameters={},
        name="step_edge",
    )

    panel._begin_paint_stroke()
    panel._paint_at(4, 4)
    panel._paint_at(8, 8)
    panel._end_paint_stroke()

    painted = panel._current_mask.data.copy()
    assert painted.sum() > 0
    assert len(panel._undo_stack) == 1

    panel.undo_paint()

    assert panel._current_mask is not None
    assert panel._current_mask.data.sum() == 0


def test_dataset_builder_canvas_emits_quickseg_click_and_zooms_on_wheel(qapp):
    canvas = DatasetBuilderCanvas()
    pixmap = QPixmap(128, 128)
    pixmap.fill(QColor("white"))
    canvas.set_source(pixmap, reset_zoom=True)
    canvas.set_raw_array(np.zeros((128, 128), dtype=float))
    canvas.set_quickseg_enabled(True)
    canvas.show()
    qapp.processEvents()

    clicks: list[tuple[int, int, int]] = []
    canvas.quickseg_click_requested.connect(lambda x, y, mods: clicks.append((x, y, mods)))

    QTest.mouseClick(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(20, 30))
    qapp.processEvents()
    assert clicks == [(20, 30, 0)]

    zoom_before = canvas.zoom()
    wheel = QWheelEvent(
        QPointF(20, 30),
        QPointF(20, 30),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(canvas.viewport(), wheel)
    qapp.processEvents()
    assert canvas.zoom() > zoom_before


def test_dataset_builder_canvas_right_drag_pans(qapp):
    canvas = DatasetBuilderCanvas()
    pixmap = QPixmap(900, 900)
    pixmap.fill(QColor("white"))
    canvas.set_source(pixmap, reset_zoom=True)
    canvas.set_raw_array(np.zeros((900, 900), dtype=float))
    canvas.show()

    scroll = QScrollArea()
    scroll.setWidget(canvas)
    scroll.resize(250, 250)
    scroll.show()
    qapp.processEvents()

    h0 = scroll.horizontalScrollBar().value()
    v0 = scroll.verticalScrollBar().value()

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(100, 100),
        QPointF(100, 100),
        QPointF(100, 100),
        Qt.RightButton,
        Qt.RightButton,
        Qt.NoModifier,
    )
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(40, 40),
        QPointF(40, 40),
        QPointF(40, 40),
        Qt.NoButton,
        Qt.RightButton,
        Qt.NoModifier,
    )
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(40, 40),
        QPointF(40, 40),
        QPointF(40, 40),
        Qt.RightButton,
        Qt.NoButton,
        Qt.NoModifier,
    )
    QApplication.sendEvent(canvas.viewport(), press)
    QApplication.sendEvent(canvas.viewport(), move)
    QApplication.sendEvent(canvas.viewport(), release)
    qapp.processEvents()

    assert (
        scroll.horizontalScrollBar().value() != h0
        or scroll.verticalScrollBar().value() != v0
    )


def test_dataset_builder_canvas_small_image_still_has_pan_room(qapp):
    canvas = DatasetBuilderCanvas()
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor("white"))
    canvas.setMinimumSize(420, 360)
    canvas.set_source(pixmap, reset_zoom=True)
    canvas.set_raw_array(np.zeros((64, 64), dtype=float))
    canvas.show()

    scroll = QScrollArea()
    scroll.setWidget(canvas)
    scroll.resize(250, 250)
    scroll.show()
    qapp.processEvents()

    h0 = scroll.horizontalScrollBar().value()
    v0 = scroll.verticalScrollBar().value()

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(120, 120),
        QPointF(120, 120),
        QPointF(120, 120),
        Qt.RightButton,
        Qt.RightButton,
        Qt.NoModifier,
    )
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(50, 50),
        QPointF(50, 50),
        QPointF(50, 50),
        Qt.NoButton,
        Qt.RightButton,
        Qt.NoModifier,
    )
    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(50, 50),
        QPointF(50, 50),
        QPointF(50, 50),
        Qt.RightButton,
        Qt.NoButton,
        Qt.NoModifier,
    )
    QApplication.sendEvent(canvas.viewport(), press)
    QApplication.sendEvent(canvas.viewport(), move)
    QApplication.sendEvent(canvas.viewport(), release)
    qapp.processEvents()

    assert (
        scroll.horizontalScrollBar().value() != h0
        or scroll.verticalScrollBar().value() != v0
    )


def test_dataset_builder_quickseg_modifier_enums_do_not_crash(qapp):
    panel = DatasetBuilderPanel(THEMES["dark"], {})
    idx = panel._task_combo.findData("terrace_segmentation")
    assert idx >= 0
    panel._task_combo.setCurrentIndex(idx)
    panel._arr = np.zeros((16, 16), dtype=float)

    calls: list[tuple[str, int, int]] = []
    panel._quickseg_add_seed = lambda x, y, *, new_terrace=False: calls.append(("add", x, y, int(new_terrace)))  # type: ignore[method-assign]
    panel._quickseg_delete_seed_at = lambda x, y: calls.append(("delete", x, y))  # type: ignore[method-assign]

    panel._quickseg_canvas_clicked(3, 4, Qt.ControlModifier)
    panel._quickseg_canvas_clicked(5, 6, Qt.AltModifier)
    panel._quickseg_canvas_clicked(7, 8, Qt.NoModifier)

    assert calls == [
        ("add", 3, 4, 1),
        ("delete", 5, 6),
        ("add", 7, 8, 0),
    ]


def test_dataset_builder_quickseg_preview_ignores_right_side_clicks(qapp):
    panel = DatasetBuilderPanel(THEMES["dark"], {})
    idx = panel._task_combo.findData("terrace_segmentation")
    assert idx >= 0
    panel._task_combo.setCurrentIndex(idx)
    arr = np.arange(256, dtype=float).reshape(16, 16)
    panel._arr = arr
    panel._base_display_arr = arr
    panel._display_arr = arr
    panel._queue = [
        SimpleNamespace(
            display_id="sample_plane0",
            status="blank",
            source_path=__file__,
            plane_index=0,
        )
    ]
    panel._current_index = 0
    panel._quickseg_cache = QuickSegPrepared(
        raw=arr,
        corrected=arr,
        equalized=arr,
        denoised=arr / arr.max(),
        gaussian=arr / arr.max(),
        gradient=arr / arr.max(),
        watershed_elevation=arr / arr.max(),
    )
    assert panel._quickseg_controls is not None
    panel._quickseg_controls._show_preprocessing_preview.setChecked(True)
    panel._refresh_display_preview()
    qapp.processEvents()

    assert panel._canvas.sceneRect().width() > 16

    calls: list[tuple[str, int, int]] = []
    panel._quickseg_add_seed = lambda x, y, *, new_terrace=False: calls.append(("add", x, y))  # type: ignore[method-assign]
    panel._quickseg_delete_seed_at = lambda x, y: calls.append(("delete", x, y))  # type: ignore[method-assign]

    panel._quickseg_canvas_clicked(20, 4, Qt.NoModifier)
    panel._quickseg_canvas_clicked(5, 4, Qt.NoModifier)

    assert calls == [("add", 5, 4)]
