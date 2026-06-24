"""Tests for FeatureCountingController lifecycle (review gui-arch #1).

The controller owns all FeatureCounting orchestration slots, shared between
ProbeFlowWindow (in-tab) and FeatureCountingWindow (floating).

These tests exercise the module-level helpers and the signal-wiring contract
without requiring a running Qt event loop.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from probeflow.gui.features.controller import (
    FeatureCountingController,
    _classify_summary,
    _pct_to_nm2,
)


# ── _pct_to_nm2 ──────────────────────────────────────────────────────────────

def test_pct_to_nm2_basic():
    arr = np.zeros((100, 200))     # 100 × 200 pixels
    px_x_m = 1e-9                  # 1 nm/pixel
    px_y_m = 1e-9
    # 1 % of 20 000 px² × (1 nm)² = 200 nm²
    result = _pct_to_nm2(1.0, arr, px_x_m, px_y_m)
    assert result == pytest.approx(200.0)


def test_pct_to_nm2_zero_percent():
    arr = np.zeros((50, 50))
    assert _pct_to_nm2(0.0, arr, 1e-9, 1e-9) == pytest.approx(0.0)


# ── _classify_summary ─────────────────────────────────────────────────────────

class _FakeClassified:
    def __init__(self, class_name, angle=0.0):
        self.class_name = class_name
        self.particle_orientation_deg = angle


def test_classify_summary_other_class():
    result = [_FakeClassified("other"), _FakeClassified("other")]
    summary = _classify_summary(result)
    assert "other: 2" in summary
    assert "100%" in summary


def test_classify_summary_multiple_classes():
    result = [
        _FakeClassified("A", 10.0),
        _FakeClassified("A", 10.0),
        _FakeClassified("B", 90.0),
    ]
    summary = _classify_summary(result)
    assert "A" in summary
    assert "B" in summary
    assert "  |  " in summary


def test_classify_summary_empty_angles():
    result = [_FakeClassified("X")]
    result[0].particle_orientation_deg = float("nan")
    summary = _classify_summary(result)
    assert "X: 1" in summary


# ── FeatureCountingController signal wiring ────────────────────────────────────
#
# We cannot instantiate QObject without a QApplication, so we test the
# module-level helpers and verify the controller's constructor wires signals
# by calling connect() on the sidebar mock.

def _make_mocks():
    """Build mock panel, sidebar, pool, status_cb triple."""
    panel = MagicMock()
    sidebar = MagicMock()
    pool = MagicMock()
    status_cb = MagicMock()
    preview_pool = MagicMock()
    return panel, sidebar, pool, status_cb, preview_pool


def test_controller_connects_direct_pass_throughs(monkeypatch):
    """Constructor wires crop_template, undo_label, mask_color directly to panel."""
    panel, sidebar, pool, status_cb, preview_pool = _make_mocks()

    # Since we can't easily skip QObject.__init__ without Qt, just verify
    # the sidebar mock receives the expected connect() calls after construction.
    try:
        FeatureCountingController(
            panel, sidebar, pool, status_cb,
            preview_pool=preview_pool,
        )
    except RuntimeError:
        pytest.skip("Qt not available in this environment")

    # crop_template_requested.connect must have been called with panel.begin_template_crop
    assert any(
        c.args[0] is panel.begin_template_crop
        for c in sidebar.crop_template_requested.connect.call_args_list
    )
    assert any(
        c.args[0] is panel.undo_last_label
        for c in sidebar.undo_label_requested.connect.call_args_list
    )
    assert any(
        c.args[0] is panel.set_mask_color
        for c in sidebar.mask_color_changed.connect.call_args_list
    )


def test_pct_to_nm2_rectangle():
    """Non-square array with anisotropic pixels."""
    arr = np.zeros((200, 100))   # 200 × 100 px
    px_x_m = 0.5e-9              # 0.5 nm x
    px_y_m = 2.0e-9              # 2.0 nm y
    # total area = 200 * 100 * 0.5nm * 2.0nm = 20000 nm²
    # 0.5% of 20000 = 100 nm²
    assert _pct_to_nm2(0.5, arr, px_x_m, px_y_m) == pytest.approx(100.0)


def test_send_to_particle_statistics_builds_set_and_emits():
    """Send-to-stats builds a FeatureSet, adds it to the shared store, emits a request."""
    from types import SimpleNamespace

    from probeflow.measurements.feature_sets import FeatureSetStore

    panel, sidebar, pool, status_cb, preview_pool = _make_mocks()
    store = FeatureSetStore()
    try:
        ctrl = FeatureCountingController(
            panel, sidebar, pool, status_cb,
            preview_pool=preview_pool,
            feature_set_store=store,
        )
    except RuntimeError:
        pytest.skip("Qt not available in this environment")

    panel.get_particles.return_value = [
        SimpleNamespace(to_dict=lambda: {"centroid_x_m": 10e-9, "centroid_y_m": 20e-9}),
        SimpleNamespace(to_dict=lambda: {"centroid_x_m": 30e-9, "centroid_y_m": 40e-9}),
    ]
    panel.current_scan.return_value = SimpleNamespace(
        scan_range_m=(100e-9, 80e-9), dims=(200, 160)
    )
    panel.current_entry.return_value = SimpleNamespace(stem="imgA")

    captured: dict = {}
    ctrl.open_particle_statistics_requested.connect(
        lambda ctx, sid: captured.update(ctx=ctx, sid=sid)
    )
    ctrl._on_send_to_particle_statistics("particles")

    assert len(store) == 1
    assert captured.get("sid")
    assert store.all()[0].point_count == 2


def test_send_to_particle_statistics_without_store_is_noop():
    panel, sidebar, pool, status_cb, preview_pool = _make_mocks()
    try:
        ctrl = FeatureCountingController(
            panel, sidebar, pool, status_cb, preview_pool=preview_pool
        )
    except RuntimeError:
        pytest.skip("Qt not available in this environment")
    ctrl._on_send_to_particle_statistics("particles")
    panel.get_particles.assert_not_called()
