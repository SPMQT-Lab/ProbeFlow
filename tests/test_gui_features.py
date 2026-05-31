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
    from probeflow.gui.compat import ProbeFlowWindow

    arr = _sample_arr()
    parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.5, size_sigma_clip=None)

    win = ProbeFlowWindow()
    win._features_sidebar._select_mode("classify")
    win._features_panel.load_entry(_sample_entry(), 0, arr, 1e-9)
    win._features_panel.set_mode("classify")
    seg_params = win._features_sidebar.classify_segmentation_params()
    win._features_panel.set_particles(
        parts,
        params_signature=tuple(sorted(seg_params.items())),
        params_meta=seg_params,
    )

    win._features_ctrl._on_run("classify")

    assert "Click particles on the image" in win._features_sidebar._status_lbl.text()
    win.close()
    win.deleteLater()


def test_features_window_segmentation_change_clears_labels(qapp, monkeypatch):
    pytest.importorskip("cv2")
    from probeflow.analysis.features import segment_particles
    from probeflow.gui.compat import ProbeFlowWindow

    arr = _sample_arr()
    parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.5, size_sigma_clip=None)

    win = ProbeFlowWindow()
    win._features_sidebar._select_mode("classify")
    win._features_panel.load_entry(_sample_entry(), 0, arr, 1e-9)
    seg_params = win._features_sidebar.classify_segmentation_params()
    win._features_panel.set_particles(
        parts,
        params_signature=tuple(sorted(seg_params.items())),
        params_meta=seg_params,
    )
    monkeypatch.setattr(
        win._features_panel,
        "_prompt_sample_label",
        lambda current_name="", current_color=(255, 255, 255): {
            "name": "target",
            "color": (255, 0, 0),
        },
    )
    win._features_panel._edit_sample_label(parts[0])

    assert win._features_panel.has_sample_labels() is True
    # The classify-segmentation sidebar shares its widgets with Particles
    # mode after the UniMR-style refactor, so the "min area" knob is now the
    # integer-valued `_min_area_slider` (units of 0.001% of image area), not
    # the legacy `_cls_min_area_spin` nm² double-spin.  Bump from the default
    # (1) to 50 so `classify_params_changed` fires with a clearly different
    # value and the controller clears sample labels.
    win._features_sidebar._min_area_slider.setValue(50)

    assert win._features_panel.has_sample_labels() is False
    assert "cleared" in win._features_sidebar._status_lbl.text().lower()
    win.close()
    win.deleteLater()


def test_features_window_classify_export_includes_samples(qapp, monkeypatch, tmp_path):
    pytest.importorskip("cv2")
    from probeflow.analysis.features import Classification, segment_particles
    from probeflow.gui.compat import ProbeFlowWindow

    arr = _sample_arr()
    parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.5, size_sigma_clip=None)

    captured = {}

    win = ProbeFlowWindow()
    win._features_sidebar._select_mode("classify")
    win._features_panel.load_entry(_sample_entry(), 0, arr, 1e-9)
    seg_params = win._features_sidebar.classify_segmentation_params()
    win._features_panel.set_particles(
        parts,
        params_signature=tuple(sorted(seg_params.items())),
        params_meta=seg_params,
    )
    monkeypatch.setattr(
        win._features_panel,
        "_prompt_sample_label",
        lambda current_name="", current_color=(255, 255, 255): {
            "name": "target",
            "color": (0, 255, 0),
        },
    )
    win._features_panel._edit_sample_label(parts[0])
    win._features_panel.set_classifications(
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

    win._features_ctrl._on_export("classify")

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
    from probeflow.gui.compat import ProbeFlowWindow

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

    win = ProbeFlowWindow()
    win._features_sidebar._select_mode("particles")
    win._features_panel.load_entry(_sample_entry(), 0, arr, 1e-9, 1e-9, 1e-9, scan=scan)
    win._features_panel.set_particles(parts)

    win._features_ctrl._on_export("particles")

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
    from probeflow.gui.compat import ProbeFlowWindow

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

    win = ProbeFlowWindow()
    win._features_sidebar._select_mode("particles")
    win._features_panel.load_entry(_sample_entry(), 0, arr, 1e-9)   # no scan
    win._features_panel.set_particles(parts)

    win._features_ctrl._on_export("particles")

    assert captured["scan"] is None
    assert captured["extra_meta"].get("source") == "/tmp/example.sxm"
    win.close()
    win.deleteLater()
