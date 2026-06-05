"""Offscreen-Qt tests for the Masks manager panel."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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


def _mask_set():
    from probeflow.core.mask import ImageMask, MaskSet
    ms = MaskSet(image_id="img")
    data = np.zeros((20, 20), dtype=bool)
    data[5:15, 5:15] = True
    data[0, 0] = True  # speck for cleanup
    ms.add(ImageMask.new(data, name="m1"))
    return ms


def _panel(qapp, ms, callbacks=None):
    from probeflow.gui.mask_manager import MaskManagerPanel
    return MaskManagerPanel(lambda: ms, callbacks or {})


def test_panel_lists_masks_and_marks_active(qapp):
    ms = _mask_set()
    ms.set_active(ms.masks[0].id)
    panel = _panel(qapp, ms)
    assert panel._list.count() == 1
    assert panel._list.item(0).text().startswith("●")
    panel.deleteLater()


def test_set_active_via_panel(qapp):
    ms = _mask_set()
    changed = []
    panel = _panel(qapp, ms, {"on_mask_set_changed": lambda: changed.append(True)})
    panel._list.setCurrentRow(0)
    panel._on_set_active()
    assert ms.active_mask_id == ms.masks[0].id
    assert changed
    panel.deleteLater()


def test_cleanup_removes_small_objects(qapp):
    ms = _mask_set()
    panel = _panel(qapp, ms)
    panel._list.setCurrentRow(0)
    before = ms.masks[0].count()
    panel._cleanup_combo.setCurrentText("Remove small objects")
    panel._on_cleanup()
    after = ms.masks[0].count()
    assert after < before  # speck removed
    assert not ms.masks[0].data[0, 0]
    panel.deleteLater()


def test_invert_via_panel(qapp):
    ms = _mask_set()
    panel = _panel(qapp, ms)
    panel._list.setCurrentRow(0)
    before = ms.masks[0].count()
    panel._on_invert()
    assert ms.masks[0].count() == 20 * 20 - before
    panel.deleteLater()


def test_convert_to_roi_callback_fires(qapp):
    ms = _mask_set()
    captured = []
    panel = _panel(qapp, ms, {"convert_to_roi": captured.append})
    panel._list.setCurrentRow(0)
    panel._on_convert_roi()
    assert captured == [ms.masks[0].id]
    panel.deleteLater()


def test_delete_via_panel(qapp):
    ms = _mask_set()
    panel = _panel(qapp, ms)
    panel._list.setCurrentRow(0)
    panel._on_delete()
    assert len(ms.masks) == 0
    panel.deleteLater()
