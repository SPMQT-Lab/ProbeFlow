from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

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


def _disk(shape, cx, cy, r=4, height=1.0):
    yy, xx = np.indices(shape)
    return np.where((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2, height, 0.0)


def _sample_arr():
    arr = np.zeros((96, 96), dtype=float)
    arr = np.maximum(arr, _disk(arr.shape, 24, 24, r=5, height=2.0))
    arr = np.maximum(arr, _disk(arr.shape, 72, 72, r=5, height=2.0))
    return arr


def _sample_entry():
    return SimpleNamespace(stem="example", path=Path("/tmp/example.sxm"))


def _stepped_arr(n=256, step_col=None):
    """Two-terrace surface with a vertical 0.235 nm step."""
    step_col = step_col if step_col is not None else n // 2
    a = np.zeros((n, n), dtype=float)
    a[:, step_col:] = 0.235e-9
    return a


def _step_widgets(qapp):
    from PySide6.QtCore import QThreadPool
    from probeflow.gui.features import FeaturesPanel, FeaturesSidebar
    from probeflow.gui.features.controller import FeatureCountingController

    panel = FeaturesPanel({})
    sidebar = FeaturesSidebar({})
    ctrl = FeatureCountingController(
        panel, sidebar, QThreadPool.globalInstance(), status_cb=lambda *_: None,
    )
    return panel, sidebar, ctrl


class TestStepEdgeExclusionGUI:
    PX = 1.5e-10

    def test_off_by_default_builds_no_mask(self, qapp):
        panel, sidebar, ctrl = _step_widgets(qapp)
        assert sidebar.step_exclude_params()["enabled"] is False
        panel.load_entry(_sample_entry(), 0, _stepped_arr(), self.PX, self.PX, self.PX)
        assert ctrl._build_exclude_mask() is None
        assert panel.step_mask() is None
        panel.deleteLater(); sidebar.deleteLater()

    def test_enabled_builds_localized_band_and_overlay(self, qapp):
        panel, sidebar, ctrl = _step_widgets(qapp)
        panel.load_entry(_sample_entry(), 0, _stepped_arr(), self.PX, self.PX, self.PX)
        sidebar._step_exclude_cb.setChecked(True)
        sidebar._step_molsize_spin.setValue(1.0)
        sidebar._step_angle_spin.setValue(20.0)

        mask = ctrl._build_exclude_mask()
        assert mask is not None and mask.any(), "step band should be computed"
        cols = np.where(mask.any(axis=0))[0]
        assert abs(cols.mean() - 128) < 12, "band should hug the step column"
        assert panel._step_overlay_item is not None, "amber overlay should be shown"
        panel.deleteLater(); sidebar.deleteLater()

    def test_segment_injects_exclude_mask(self, qapp, monkeypatch):
        panel, sidebar, ctrl = _step_widgets(qapp)
        panel.load_entry(_sample_entry(), 0, _stepped_arr(), self.PX, self.PX, self.PX)
        sidebar._step_exclude_cb.setChecked(True)
        sidebar._step_molsize_spin.setValue(1.0)

        captured = {}
        monkeypatch.setattr(ctrl._pool, "start", lambda w: captured.setdefault("worker", w))
        ctrl._on_segment()

        params = captured["worker"]._params
        assert params["exclude_mask"] is not None
        assert params["exclude_mask"].shape == captured["worker"]._arr.shape
        panel.deleteLater(); sidebar.deleteLater()

    def test_disabled_segment_passes_no_mask(self, qapp, monkeypatch):
        panel, sidebar, ctrl = _step_widgets(qapp)
        panel.load_entry(_sample_entry(), 0, _stepped_arr(), self.PX, self.PX, self.PX)
        captured = {}
        monkeypatch.setattr(ctrl._pool, "start", lambda w: captured.setdefault("worker", w))
        ctrl._on_segment()
        assert captured["worker"]._params["exclude_mask"] is None
        panel.deleteLater(); sidebar.deleteLater()

    def test_preview_mask_sliced_to_match_array(self, qapp, monkeypatch):
        """Regression: the preview step-slices the array, so the mask must be
        sliced the same way or segment_particles raises on a shape mismatch."""
        panel, sidebar, ctrl = _step_widgets(qapp)
        big = _stepped_arr(1040)              # ≥ 1024 → preview step-slices by 2
        panel.load_entry(_sample_entry(), 0, big, self.PX, self.PX, self.PX)
        sidebar._step_exclude_cb.setChecked(True)
        sidebar._step_molsize_spin.setValue(1.0)

        captured = {}
        monkeypatch.setattr(ctrl._preview_pool, "start",
                            lambda w: captured.setdefault("worker", w))
        ctrl._on_preview()

        w = captured["worker"]
        assert w._arr.shape[0] < 1040, "preview should have downsampled"
        assert w._params["exclude_mask"] is not None
        assert w._params["exclude_mask"].shape == w._arr.shape
        panel.deleteLater(); sidebar.deleteLater()

    def test_painted_mask_behaviour_unchanged(self, qapp):
        """The painted brush still zeroes pixels in get_analysis_array (untouched)."""
        panel, _sidebar, _ctrl = _step_widgets(qapp)
        arr = _sample_arr()
        panel.load_entry(_sample_entry(), 0, arr, self.PX, self.PX, self.PX)
        panel._on_mask_painted(24.0, 24.0)    # paint over the first disk
        out = panel.get_analysis_array()
        assert out is not arr, "painted mask should still produce a modified array"
        assert panel.has_exclusion_mask()
        _sidebar.deleteLater()


def test_features_panel_sample_label_flow(qapp, monkeypatch):
    pytest.importorskip("cv2")
    from probeflow.analysis.features import segment_particles
    from probeflow.gui.features import FeaturesPanel

    arr = _sample_arr()
    parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.5, size_sigma_clip=None)

    panel = FeaturesPanel({})
    panel.load_entry(_sample_entry(), 0, arr, 1e-9)
    panel.set_mode("classify")
    panel.set_particles(parts, params_signature=(("threshold", "otsu"),), params_meta={"threshold": "otsu"})
    panel.set_sample_selection_armed(True)
    monkeypatch.setattr(
        panel,
        "_prompt_sample_label",
        lambda current_name="", current_color=(255, 255, 255): {
            "name": "target",
            "color": (12, 34, 56),
        },
    )

    panel._edit_sample_label(parts[0])

    assert panel.has_sample_labels() is True
    rows = panel.sample_label_rows()
    assert rows[0]["class_name"] == "target"
    assert rows[0]["color_rgb"] == [12, 34, 56]
    panel.deleteLater()


def test_features_window_classify_requires_labels(qapp):
    pytest.importorskip("cv2")
    from probeflow.analysis.features import segment_particles
    from probeflow.gui.features.window import FeatureCountingWindow

    arr = _sample_arr()
    parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.5, size_sigma_clip=None)

    win = FeatureCountingWindow(theme={})
    win._sidebar._select_mode("classify")
    win._panel.load_entry(_sample_entry(), 0, arr, 1e-9)
    win._panel.set_mode("classify")
    seg_params = win._sidebar.classify_segmentation_params()
    win._panel.set_particles(
        parts,
        params_signature=tuple(sorted(seg_params.items())),
        params_meta=seg_params,
    )

    win._ctrl._on_run("classify")

    assert "Click particles on the image" in win._sidebar._status_lbl.text()
    win.close()
    win.deleteLater()


def test_features_window_segmentation_change_clears_labels(qapp, monkeypatch):
    pytest.importorskip("cv2")
    from probeflow.analysis.features import segment_particles
    from probeflow.gui.features.window import FeatureCountingWindow

    arr = _sample_arr()
    parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.5, size_sigma_clip=None)

    win = FeatureCountingWindow(theme={})
    win._sidebar._select_mode("classify")
    win._panel.load_entry(_sample_entry(), 0, arr, 1e-9)
    seg_params = win._sidebar.classify_segmentation_params()
    win._panel.set_particles(
        parts,
        params_signature=tuple(sorted(seg_params.items())),
        params_meta=seg_params,
    )
    monkeypatch.setattr(
        win._panel,
        "_prompt_sample_label",
        lambda current_name="", current_color=(255, 255, 255): {
            "name": "target",
            "color": (255, 0, 0),
        },
    )
    win._panel._edit_sample_label(parts[0])

    assert win._panel.has_sample_labels() is True
    # The classify-segmentation sidebar shares its widgets with Particles
    # mode after the UniMR-style refactor, so the "min area" knob is now the
    # integer-valued `_min_area_slider` (units of 0.001% of image area), not
    # the legacy `_cls_min_area_spin` nm² double-spin.  Bump from the default
    # (1) to 50 so `classify_params_changed` fires with a clearly different
    # value and the controller clears sample labels.
    win._sidebar._min_area_slider.setValue(50)

    assert win._panel.has_sample_labels() is False
    assert "cleared" in win._sidebar._status_lbl.text().lower()
    win.close()
    win.deleteLater()


def test_features_window_classify_export_includes_samples(qapp, monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    from probeflow.analysis.features import Classification, segment_particles
    from probeflow.gui.features.window import FeatureCountingWindow

    arr = _sample_arr()
    parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.5, size_sigma_clip=None)

    captured = {}

    win = FeatureCountingWindow(theme={})
    win._sidebar._select_mode("classify")
    win._panel.load_entry(_sample_entry(), 0, arr, 1e-9)
    seg_params = win._sidebar.classify_segmentation_params()
    win._panel.set_particles(
        parts,
        params_signature=tuple(sorted(seg_params.items())),
        params_meta=seg_params,
    )
    monkeypatch.setattr(
        win._panel,
        "_prompt_sample_label",
        lambda current_name="", current_color=(255, 255, 255): {
            "name": "target",
            "color": (0, 255, 0),
        },
    )
    win._panel._edit_sample_label(parts[0])
    win._panel.set_classifications(
        [Classification(particle_index=parts[1].index, class_name="other", similarity=0.25)],
        meta={"params": {"encoder": "raw"}, "segmentation": seg_params},
    )

    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "classify.json"), "JSON (*.json)"),
    )

    def _capture_write_json(out_path, items, *, kind, extra_meta=None, **kwargs):
        captured["out_path"] = out_path
        captured["items"] = list(items)
        captured["kind"] = kind
        captured["extra_meta"] = dict(extra_meta or {})

    monkeypatch.setattr("probeflow.io.writers.json.write_json", _capture_write_json)

    win._ctrl._on_export("classify")

    assert captured["kind"] == "classifications"
    assert captured["extra_meta"]["samples"][0]["class_name"] == "target"
    assert captured["extra_meta"]["classification"]["params"]["encoder"] == "raw"
    win.close()
    win.deleteLater()


def _make_scan(arr):
    """Build a minimal real Scan around a single plane for export-provenance tests."""
    from pathlib import Path

    from probeflow.core.scan_model import Scan

    Ny, Nx = arr.shape
    return Scan(
        planes=[arr],
        plane_names=["Z forward"],
        plane_units=["m"],
        plane_synthetic=[False],
        header={},
        scan_range_m=(Nx * 1e-9, Ny * 1e-9),
        source_path=Path("/tmp/example.sxm"),
        source_format="sxm",
    )


def test_features_export_passes_scan_for_provenance(qapp, monkeypatch):
    """When loaded with a Scan, the GUI export forwards it to write_json.

    The CLI export records full provenance (scan range, pixel sizes, plane
    names/units) by passing ``scan=``; the GUI used to drop it and pass only a
    source path.  This pins the fix: a Scan threaded through ``load_entry``
    reaches ``write_json`` so GUI and CLI exports carry the same metadata.
    """
    pytest.importorskip("cv2")
    from probeflow.analysis.features import segment_particles
    from probeflow.gui.features.window import FeatureCountingWindow

    arr = _sample_arr()
    scan = _make_scan(arr)
    parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.5, size_sigma_clip=None)

    captured = {}

    def _capture_write_json(out_path, items, *, kind, scan=None, extra_meta=None, **kwargs):
        captured["kind"] = kind
        captured["scan"] = scan
        captured["extra_meta"] = dict(extra_meta or {})

    monkeypatch.setattr("probeflow.io.writers.json.write_json", _capture_write_json)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: ("/tmp/example_particles.json", "JSON (*.json)"),
    )

    win = FeatureCountingWindow(theme={})
    win._sidebar._select_mode("particles")
    win._panel.load_entry(_sample_entry(), 0, arr, 1e-9, 1e-9, 1e-9, scan=scan)
    win._panel.set_particles(parts)

    win._ctrl._on_export("particles")

    assert captured["kind"] == "particles"
    assert captured["scan"] is scan, "the loaded Scan must reach write_json"
    # With a Scan present, write_json supplies source_path itself, so we must not
    # also inject a bare/duplicate source key into extra_meta.
    assert "source" not in captured["extra_meta"]
    win.close()
    win.deleteLater()


def test_features_export_without_scan_falls_back_to_source(qapp, monkeypatch):
    """Loaded without a Scan (e.g. an array from the viewer), export still
    records a source pointer and passes scan=None — no provenance is invented."""
    pytest.importorskip("cv2")
    from probeflow.analysis.features import segment_particles
    from probeflow.gui.features.window import FeatureCountingWindow

    arr = _sample_arr()
    parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.5, size_sigma_clip=None)

    captured = {}

    def _capture_write_json(out_path, items, *, kind, scan=None, extra_meta=None, **kwargs):
        captured["scan"] = scan
        captured["extra_meta"] = dict(extra_meta or {})

    monkeypatch.setattr("probeflow.io.writers.json.write_json", _capture_write_json)
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: ("/tmp/example_particles.json", "JSON (*.json)"),
    )

    win = FeatureCountingWindow(theme={})
    win._sidebar._select_mode("particles")
    win._panel.load_entry(_sample_entry(), 0, arr, 1e-9)   # no scan
    win._panel.set_particles(parts)

    win._ctrl._on_export("particles")

    assert captured["scan"] is None
    assert captured["extra_meta"].get("source") == "/tmp/example.sxm"
    win.close()
    win.deleteLater()


def test_card_context_features_loads_floating_window(qapp, monkeypatch):
    """Regression: the Browse card 'Send to Feature Counting' context action
    must load the scan into the floating FeatureCountingWindow — the only
    Features workspace.  It used to load into a hidden duplicate panel in the
    main window's content stack, so the floating window opened empty."""
    from PySide6.QtCore import QThreadPool

    from probeflow.gui.app import ProbeFlowWindow

    arr = _sample_arr()
    scan = _make_scan(arr)
    monkeypatch.setattr("probeflow.gui.app.load_scan", lambda _path: scan)

    win = ProbeFlowWindow()
    try:
        win._on_card_context_action(_sample_entry(), "features")
        assert win._fc_window is not None, "floating FC window should open"
        assert QThreadPool.globalInstance().waitForDone(5000)
        qapp.processEvents()   # deliver the queued finished() signal
        assert win._fc_window._panel.current_array() is not None, \
            "scan must arrive in the floating window, not a hidden panel"
        assert win._fc_window._panel.current_entry().stem == "example"
    finally:
        if win._fc_window is not None:
            win._fc_window.close()
        win.close()
        win.deleteLater()


def test_load_scan_plane_for_analysis_scan_matches_processed_plane(qapp, monkeypatch):
    """Feature-analysis provenance must describe the processed plane, not raw scan 0."""
    from probeflow.core.scan_model import Scan
    from probeflow.gui.app import ProbeFlowWindow

    raw_forward = np.zeros((4, 5), dtype=float)
    raw_backward = np.arange(35, dtype=float).reshape(5, 7)
    loaded_scan = Scan(
        planes=[raw_forward, raw_backward],
        plane_names=["Z forward", "Z backward"],
        plane_units=["m", "m"],
        plane_synthetic=[False, False],
        header={"SCAN_PIXELS": "5 4"},
        scan_range_m=(14e-9, 10e-9),
        source_path=Path("/tmp/example.sxm"),
        source_format="sxm",
    )

    monkeypatch.setattr("probeflow.gui.app.load_scan", lambda _path: loaded_scan)

    saved_processing = {
        "geometric_ops": [
            {
                "op": "scale_image",
                "params": {"new_height": 6, "new_width": 8, "order": 1},
            }
        ]
    }
    win = ProbeFlowWindow.__new__(ProbeFlowWindow)
    win._saved_processing_get = lambda _entry: saved_processing

    arr, px_m, px_x_m, px_y_m, plane_idx, analysis_scan = (
        ProbeFlowWindow._load_scan_plane_for_analysis(win, _sample_entry(), 1)
    )

    assert plane_idx == 1
    assert arr.shape == (6, 8)
    assert analysis_scan.planes[0].shape == arr.shape
    assert analysis_scan.dims == (8, 6)
    assert analysis_scan.plane_names == ["Z backward"]
    assert analysis_scan.scan_range_m == pytest.approx((14e-9, 10e-9))
    assert px_x_m == pytest.approx(14e-9 / 8)
    assert px_y_m == pytest.approx(10e-9 / 6)
    assert px_m == pytest.approx(np.sqrt(px_x_m * px_y_m))
    assert [step.op for step in analysis_scan.processing_state.steps] == ["scale_image"]
