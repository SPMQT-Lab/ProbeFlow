"""Multi-step workflow replay harness (2026-06-12).

Drives the real ImageViewerDialog through workflows the way a user performs
them, and enforces the single invariant every seam review so far has been
protecting piecewise:

    what you see  ==  what you export  ==  what the provenance replays

After each workflow step:

* the export array must equal the displayed array (structural — the export
  injects the display array — but pinned so that can never silently change);
* the canonical ProcessingState recorded on the export, after a JSON round
  trip (exactly what a .probeflow.json sidecar stores), replayed against the
  raw plane with the live ROI/mask sets, must reproduce the display
  bit-for-bit;
* the physical extent recorded on the export must match the replayed
  calibration (scale bars / FFT k-axes stay honest through shape-changing
  steps).

The sample scan is copied into tmp_path so sidecar writes (ROI/mask saves
are automatic) never pollute test_data.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SAMPLE_SXM = (Path(__file__).resolve().parent.parent
              / "test_data" / "sample_input" / "A250320.191933.sxm")

pytestmark = pytest.mark.skipif(
    not SAMPLE_SXM.exists(), reason="sample SXM not present")


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PySide6 unavailable: {exc}")
    app = QApplication.instance()
    if app is not None:
        return app
    try:
        return QApplication([])
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"QApplication unavailable: {exc}")


@pytest.fixture
def viewer(qapp, tmp_path):
    """A real ImageViewerDialog on a tmp copy of the sample scan."""
    from probeflow.gui import ImageViewerDialog, SxmFile, THEMES

    paths = []
    for i in range(2):  # two entries so navigation workflows can run
        p = tmp_path / f"scan_{i:02d}.sxm"
        shutil.copy(SAMPLE_SXM, p)
        paths.append(p)
    entries = [SxmFile(path=p, stem=p.stem) for p in paths]
    dlg = ImageViewerDialog(entries[0], entries, "gray", THEMES["dark"])
    yield dlg
    dlg.close()
    dlg.deleteLater()


def assert_wysiwyg_replays(dlg) -> None:
    """display == export == JSON-round-tripped canonical replay from raw."""
    from probeflow.core.processing_state import ProcessingState
    from probeflow.processing.state import (
        apply_processing_state_with_calibration,
    )

    display = dlg._display_arr if dlg._display_arr is not None else dlg._raw_arr
    assert display is not None, "viewer has no data"
    assert not getattr(dlg, "_processing_error", ""), dlg._processing_error
    assert not getattr(dlg, "_processing_roi_error", ""), dlg._processing_roi_error

    scan, plane = dlg._processed_scan_for_export()
    export_arr = np.asarray(scan.planes[plane])
    np.testing.assert_array_equal(
        export_arr, np.asarray(display),
        err_msg="export array differs from the displayed array")

    # Replay the canonical state recorded on the export after a JSON round
    # trip — exactly what a .probeflow.json provenance sidecar stores.
    state_dict = json.loads(json.dumps(scan.processing_state.to_dict()))
    state = ProcessingState.from_dict(state_dict)
    replayed, replay_range = apply_processing_state_with_calibration(
        dlg._raw_arr, state, dlg._image_roi_set,
        mask_set=getattr(dlg, "_image_mask_set", None),
        scan_range_m=getattr(dlg, "_scan_range_m", None),
    )
    np.testing.assert_array_equal(
        replayed, np.asarray(display, dtype=np.float64),
        err_msg="provenance replay does not reproduce the displayed array")

    if replay_range is not None:
        np.testing.assert_allclose(
            tuple(scan.scan_range_m), replay_range, rtol=1e-12,
            err_msg="exported physical extent differs from replayed calibration")


# ── Step drivers (the same handlers the GUI wires) ────────────────────────────

def apply_panel(dlg, **trigger_keys) -> None:
    """Set panel filter keys and press Apply — the real commit path."""
    state = dlg._processing_panel.state()
    state.update(trigger_keys)
    dlg._processing_panel.set_state(state)
    dlg._on_apply_processing()


def apply_mains(dlg, **params) -> None:
    """Mirror of the viewer's _apply_fft_correction closure for the Mains tab."""
    base = {
        "scan_speed_m_per_s": None,
        "scan_range_m": [float(v) for v in dlg._scan_range_m],
        "notch_shape": "streak",
        "snap_window_px": 0,
    }
    base.update(params)
    ops = list(dlg._processing.get("geometric_ops") or [])
    ops.append({"op": "mains_pickup_suppression", "params": base})
    dlg._processing["geometric_ops"] = ops
    dlg._refresh_processing_display()


def make_roi(dlg, x, y, w, h, name=None):
    from probeflow.core.roi import ROI

    roi = ROI.new("rectangle",
                  {"x": x, "y": y, "width": w, "height": h}, name=name)
    dlg._on_canvas_roi_created(roi)
    return roi


# ── Workflows ─────────────────────────────────────────────────────────────────

class TestWholeImageWorkflows:
    def test_background_then_filters(self, viewer):
        viewer._on_stm_background_applied({
            "fit_region": "whole_image",
            "line_statistic": "median",
            "model": "linear",
            "linear_x_first": False,
            "blur_length": None,
            "jump_threshold": None,
            "preserve_level": "median",
        })
        assert_wysiwyg_replays(viewer)

        apply_panel(viewer, smooth_sigma=1.5)
        assert_wysiwyg_replays(viewer)

        apply_panel(viewer, highpass_sigma=12.0)
        assert_wysiwyg_replays(viewer)

    def test_threshold_scale_quantize_chain(self, viewer):
        viewer._on_threshold_applied({"mode": "clip", "lower": None,
                                      "upper": 1e-9})
        assert_wysiwyg_replays(viewer)

        h, w = viewer._display_arr.shape
        viewer._on_scale_image_applied(
            {"new_width": int(w * 1.5), "new_height": int(h // 2)})
        assert_wysiwyg_replays(viewer)

        viewer._on_convert_bit_depth(8)
        assert_wysiwyg_replays(viewer)


class TestSelectionCommitWorkflows:
    def test_region_filter_after_flip_replays_where_drawn(self, viewer):
        """The scope-ordering fix, end-to-end through the real GUI commit:
        flip first, then commit a region filter — replay must land it where
        it was drawn on the flipped display."""
        viewer._on_geometric_op("flip_horizontal")
        viewer._zoom_lbl.set_selection(
            "rectangle", {"x": 4, "y": 4, "width": 24, "height": 24})
        apply_panel(viewer, smooth_sigma=3.0)
        assert viewer._processing["roi_filter_ops"][0]["after_geometric_ops"] == 1
        assert_wysiwyg_replays(viewer)

        # Second selection, second commit, then another flip on top.
        viewer._zoom_lbl.set_selection(
            "rectangle", {"x": 40, "y": 40, "width": 20, "height": 20})
        apply_panel(viewer, highpass_sigma=8.0)
        viewer._on_geometric_op("flip_vertical")
        assert_wysiwyg_replays(viewer)

    def test_promoted_roi_scope_freezes_against_moves(self, viewer):
        roi = make_roi(viewer, 8, 8, 30, 30, name="scope")
        viewer._roi_filter_scope_id = roi.id
        apply_panel(viewer, smooth_sigma=2.0)
        frozen = dict(viewer._processing["roi_filter_ops"][0]["frozen_geometry"])
        assert_wysiwyg_replays(viewer)

        # Move the live ROI afterwards: the committed step must not follow.
        roi.geometry["x"] = 120.0
        viewer._on_image_roi_set_changed()
        assert viewer._processing["roi_filter_ops"][0]["frozen_geometry"] == frozen
        assert_wysiwyg_replays(viewer)


class TestMainsWorkflows:
    def test_streak_pair_then_flip(self, viewer):
        apply_mains(viewer, extra_streaks_px=[18], harmonics=0,
                    notch_radius_px=3.0, notch_fill="background")
        assert_wysiwyg_replays(viewer)

        viewer._on_geometric_op("flip_horizontal")
        assert_wysiwyg_replays(viewer)


class TestGeometryWorkflows:
    def test_rotate_arbitrary_grows_canvas_consistently(self, viewer, monkeypatch):
        from PySide6.QtWidgets import QInputDialog

        monkeypatch.setattr(QInputDialog, "getDouble",
                            staticmethod(lambda *a, **k: (17.0, True)))
        viewer._on_rotate_arbitrary()
        assert viewer._display_arr.shape != viewer._raw_arr.shape
        assert_wysiwyg_replays(viewer)

    def test_shear_then_threshold(self, viewer):
        viewer._on_shear_applied({"shear_x": 0.08, "shear_y": 0.0})
        assert_wysiwyg_replays(viewer)
        viewer._on_threshold_applied({"mode": "clip", "lower": None,
                                      "upper": 5e-10})
        assert_wysiwyg_replays(viewer)


class TestArithmeticWorkflows:
    def test_constant_and_frozen_roi_arithmetic(self, viewer):
        # Whole-image constant offset (the dialog's whole-image spec).
        ops = list(viewer._processing.get("arithmetic_ops") or [])
        ops.append({"op": "arithmetic",
                    "params": {"operation": "add", "operand_type": "constant",
                               "value_si": 2e-10}})
        viewer._processing["arithmetic_ops"] = ops
        viewer._refresh_processing_display()
        assert_wysiwyg_replays(viewer)

        # ROI-scoped, frozen at commit (the producer's spec shape).
        roi = make_roi(viewer, 10, 10, 25, 25, name="arith")
        ops = list(viewer._processing.get("arithmetic_ops") or [])
        ops.append({
            "op": "arithmetic",
            "params": {"operation": "multiply", "operand_type": "constant",
                       "factor": 2.0},
            "roi_id": roi.id,
            "frozen_geometry": {"kind": roi.kind,
                                "geometry": dict(roi.geometry),
                                "coord_system": roi.coord_system},
            "after_geometric_ops": len(
                viewer._processing.get("geometric_ops") or []),
        })
        viewer._processing["arithmetic_ops"] = ops
        viewer._refresh_processing_display()
        assert_wysiwyg_replays(viewer)


class TestNavigationWorkflows:
    def test_pipeline_carries_to_next_image_and_back(self, viewer):
        apply_panel(viewer, smooth_sigma=2.0)
        viewer._on_geometric_op("flip_horizontal")
        assert_wysiwyg_replays(viewer)

        viewer._go_next()
        assert viewer._idx == 1
        assert_wysiwyg_replays(viewer)

        viewer._go_prev()
        assert viewer._idx == 0
        assert_wysiwyg_replays(viewer)

    def test_reset_returns_to_raw(self, viewer):
        apply_panel(viewer, smooth_sigma=2.0)
        viewer._on_reset_processing()
        assert viewer._processing == {}
        assert_wysiwyg_replays(viewer)


class TestHardCombinations:
    def test_frozen_roi_scope_survives_rotate_that_removes_live_roi(
            self, viewer, monkeypatch):
        """Commit a ROI-scoped filter, then rotate arbitrarily: the live ROI
        is invalidated and REMOVED from the set, but the frozen step must
        keep replaying and export must not be blocked."""
        from PySide6.QtWidgets import QInputDialog

        roi = make_roi(viewer, 12, 12, 28, 28, name="doomed")
        viewer._roi_filter_scope_id = roi.id
        apply_panel(viewer, smooth_sigma=2.5)

        monkeypatch.setattr(QInputDialog, "getDouble",
                            staticmethod(lambda *a, **k: (23.0, True)))
        viewer._on_rotate_arbitrary()
        assert viewer._image_roi_set.get(roi.id) is None, (
            "rotate_arbitrary should have removed the live area ROI")
        assert viewer._assert_exportable_processing() is True
        assert_wysiwyg_replays(viewer)

    def test_inverse_fft_selections_survive_json_round_trip(self, viewer):
        """Reconstruct-tab removals store selection dicts (tuples → lists in
        JSON); the replay must rebuild and apply them identically."""
        ops = list(viewer._processing.get("geometric_ops") or [])
        ops.append({"op": "inverse_fft_filter", "params": {
            "mode": "remove_selected",
            "conjugate_symmetric": True,
            "soft_px": 1.5,
            "selections": [
                {"kind": "ellipse", "dx": 14.0, "dy": 0.0, "rx": 3.0,
                 "ry": 5.0, "angle_deg": 20.0},
                {"kind": "rect", "dx": 0.0, "dy": 22.0, "half_w": 2.0,
                 "half_h": 4.0},
                {"kind": "paint", "radius": 2.0,
                 "stamps": [(8.0, 8.0), (9.0, 9.0), (10.0, 10.0)]},
            ],
        }})
        viewer._processing["geometric_ops"] = ops
        viewer._refresh_processing_display()
        assert_wysiwyg_replays(viewer)

    def test_periodic_notches_with_region_commit_on_top(self, viewer):
        viewer._processing["periodic_notches"] = [(12, 0), (0, 9)]
        viewer._processing["periodic_notch_radius"] = 2.5
        viewer._refresh_processing_display()
        assert_wysiwyg_replays(viewer)

        viewer._zoom_lbl.set_selection(
            "rectangle", {"x": 6, "y": 6, "width": 20, "height": 20})
        apply_panel(viewer, smooth_sigma=2.0)
        assert_wysiwyg_replays(viewer)

    def test_set_zero_point_and_plane_replay(self, viewer):
        viewer._processing["set_zero_xy"] = (10, 12)
        viewer._processing["set_zero_patch"] = 3
        viewer._refresh_processing_display()
        assert_wysiwyg_replays(viewer)

        viewer._processing.pop("set_zero_xy")
        viewer._processing["set_zero_plane_points"] = [(5, 5), (50, 8), (12, 48)]
        viewer._refresh_processing_display()
        assert_wysiwyg_replays(viewer)

    def test_undo_redo_keep_invariant(self, viewer):
        apply_panel(viewer, smooth_sigma=2.0)
        viewer._on_geometric_op("flip_horizontal")
        assert_wysiwyg_replays(viewer)

        viewer._on_undo_processing()
        assert viewer._processing.get("smooth_sigma") == pytest.approx(2.0)
        assert not viewer._processing.get("geometric_ops")
        assert_wysiwyg_replays(viewer)
        viewer._on_redo_processing()
        assert [op["op"] for op in viewer._processing["geometric_ops"]] == [
            "flip_horizontal",
        ]
        assert_wysiwyg_replays(viewer)

    def test_channel_switch_reapplies_pipeline(self, viewer):
        apply_panel(viewer, smooth_sigma=2.0)
        if viewer._ch_cb.count() < 2:
            pytest.skip("sample scan has a single channel")
        viewer._ch_cb.setCurrentIndex(1)
        assert_wysiwyg_replays(viewer)
        viewer._ch_cb.setCurrentIndex(0)
        assert_wysiwyg_replays(viewer)

    def test_kitchen_sink_chain(self, viewer):
        """Background → region commit → flip → mains streak → threshold →
        scale → quantize, asserting the invariant at the end of the chain."""
        viewer._on_stm_background_applied({
            "fit_region": "whole_image", "line_statistic": "median",
            "model": "linear", "linear_x_first": False, "blur_length": None,
            "jump_threshold": None, "preserve_level": "median",
        })
        viewer._zoom_lbl.set_selection(
            "rectangle", {"x": 8, "y": 8, "width": 22, "height": 22})
        apply_panel(viewer, smooth_sigma=2.0)
        viewer._on_geometric_op("flip_horizontal")
        apply_mains(viewer, extra_streaks_px=[15], harmonics=0,
                    notch_radius_px=2.0, notch_fill="background")
        viewer._on_threshold_applied({"mode": "clip", "lower": None,
                                      "upper": 2e-9})
        h, w = viewer._display_arr.shape
        viewer._on_scale_image_applied({"new_width": w // 2,
                                        "new_height": h // 2})
        viewer._on_convert_bit_depth(12)
        assert_wysiwyg_replays(viewer)


class TestSetZeroIntent:
    """Intent-level checks the wysiwyg invariant cannot see: the zero plane
    must anchor on the features the user clicked, not on their mirrored raw
    coordinates (2026-06-12 workflow review — picks mapped click fractions
    onto the raw frame while markers showed the clicked spots)."""

    @staticmethod
    def _pick(viewer, frac_x, frac_y):
        viewer._on_set_zero_pick(frac_x, frac_y)

    def _assert_clicked_pixels_zeroed(self, viewer, fracs):
        # patch=1 -> 3x3 sampling window: the fitted plane passes exactly
        # through the patch means at the picked points, so those means are
        # zero after subtraction (single pixels retain local noise).
        disp = viewer._display_arr
        Ny, Nx = disp.shape
        scale = float(np.nanstd(disp)) or 1.0
        for fx, fy in fracs:
            x = int(round(fx * (Nx - 1)))
            y = int(round(fy * (Ny - 1)))
            val = float(np.nanmean(disp[max(0, y - 1):y + 2,
                                        max(0, x - 1):x + 2]))
            assert abs(val) < 1e-9 * scale + 1e-20, (
                f"zero plane not anchored at clicked display patch "
                f"({x},{y}): {val!r}"
            )

    def test_zero_plane_picked_after_flip_anchors_clicked_features(self, viewer):
        viewer._on_geometric_op("flip_horizontal")
        viewer._set_zero_plane_btn.setChecked(True)
        fracs = [(0.2, 0.2), (0.8, 0.25), (0.3, 0.8)]
        for fx, fy in fracs:
            self._pick(viewer, fx, fy)

        assert viewer._processing["set_zero_after_geometric_ops"] == 1
        self._assert_clicked_pixels_zeroed(viewer, fracs)
        assert_wysiwyg_replays(viewer)

    def test_zero_plane_picked_after_rotation_uses_rotated_frame(
            self, viewer, monkeypatch):
        from PySide6.QtWidgets import QInputDialog

        monkeypatch.setattr(QInputDialog, "getDouble",
                            staticmethod(lambda *a, **k: (15.0, True)))
        viewer._on_rotate_arbitrary()
        viewer._set_zero_plane_btn.setChecked(True)
        fracs = [(0.45, 0.45), (0.55, 0.48), (0.5, 0.58)]  # central, valid
        for fx, fy in fracs:
            self._pick(viewer, fx, fy)

        self._assert_clicked_pixels_zeroed(viewer, fracs)
        assert_wysiwyg_replays(viewer)

    def test_zero_plane_without_geometric_ops_unchanged(self, viewer):
        viewer._set_zero_plane_btn.setChecked(True)
        fracs = [(0.2, 0.2), (0.7, 0.3), (0.4, 0.7)]
        for fx, fy in fracs:
            self._pick(viewer, fx, fy)
        assert viewer._processing.get("set_zero_after_geometric_ops", 0) == 0
        self._assert_clicked_pixels_zeroed(viewer, fracs)
        assert_wysiwyg_replays(viewer)
