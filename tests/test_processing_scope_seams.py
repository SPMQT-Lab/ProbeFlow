"""Adversarial seam tests for ROI/mask/region-scoped processing replay.

Targets the GUI-state → ProcessingState → apply/export path (review focus #3):
frozen-geometry durability, replay ordering against geometric ops, operand
resolver injection, and export-validation honesty.

Tests marked ``xfail(strict=True)`` document confirmed bugs found in the
2026-06-11 adversarial review; they flip to XPASS (and fail the suite) when
the underlying bug is fixed, at which point the marker should be removed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from probeflow.core.mask import ImageMask, MaskSet, _pack_bool
from probeflow.core.roi import ROI, ROISet
from probeflow.processing.gui_adapter import processing_state_from_gui
from probeflow.processing.state import (
    ProcessingState,
    ProcessingStep,
    apply_processing_state,
    assert_roi_references_resolved,
    missing_roi_references,
)

RNG_SEED = 0


def _noise(shape=(40, 60)) -> np.ndarray:
    return np.random.default_rng(RNG_SEED).normal(size=shape)


def _rect_frozen(x, y, w, h) -> dict:
    return {
        "kind": "rectangle",
        "geometry": {"x": x, "y": y, "width": w, "height": h},
        "coord_system": "pixel",
    }


def _region_smooth_spec(x, y, w, h, sigma=3.0, after_geometric_ops=None) -> dict:
    spec = {
        "op": "smooth",
        "params": {"sigma_px": sigma},
        "scope_kind": "region",
        "frozen_geometry": _rect_frozen(x, y, w, h),
    }
    if after_geometric_ops is not None:
        spec["after_geometric_ops"] = after_geometric_ops
    return spec


def _std(arr, x0, x1, y0, y1) -> float:
    return float(arr[y0:y1, x0:x1].std())


SMOOTHED = 0.3   # local std well below this ⇒ smoothing landed here
UNTOUCHED = 0.7  # local std above this ⇒ raw noise


# ── Frozen-scope durability (must keep passing) ───────────────────────────────

class TestFrozenScopeDurability:
    def test_frozen_roi_geometry_ignores_live_roi_move(self):
        """A committed ROI-scoped filter must use the geometry frozen at
        commit time, even when the live ROI has since moved."""
        arr = _noise()
        roi = ROI.new("rectangle", {"x": 40, "y": 20, "width": 16, "height": 16},
                      name="A")  # live ROI: already moved away
        roi_set = ROISet(image_id="t")
        roi_set.add(roi)

        gui = {"roi_filter_ops": [{
            "op": "smooth", "params": {"sigma_px": 3.0},
            "roi_id": roi.id,
            "frozen_geometry": _rect_frozen(2, 2, 16, 16),  # commit-time pos
        }]}
        out = apply_processing_state(arr, processing_state_from_gui(gui), roi_set)

        assert _std(out, 4, 16, 4, 16) < SMOOTHED, "frozen location not filtered"
        assert _std(out, 44, 54, 24, 34) > UNTOUCHED, (
            "filter followed the moved live ROI instead of the frozen geometry"
        )

    def test_frozen_roi_step_survives_roi_deletion(self):
        """Deleting the live ROI must not invalidate the already frozen step:
        it still replays, and validation does not report it missing."""
        arr = _noise()
        gui = {"roi_filter_ops": [{
            "op": "smooth", "params": {"sigma_px": 3.0},
            "roi_id": "deleted-roi-id",
            "frozen_geometry": _rect_frozen(2, 2, 16, 16),
        }]}
        state = processing_state_from_gui(gui)

        assert missing_roi_references(state, ROISet(image_id="t")) == []
        assert_roi_references_resolved(state, ROISet(image_id="t"))  # must not raise

        out = apply_processing_state(arr, state, ROISet(image_id="t"))
        assert _std(out, 4, 16, 4, 16) < SMOOTHED

    def test_two_roi_scoped_filters_coexist(self):
        """Filters committed to ROI A and ROI B must both survive with their
        own frozen geometry, not overwrite each other."""
        arr = _noise()
        gui = {"roi_filter_ops": [
            {"op": "smooth", "params": {"sigma_px": 3.0},
             "roi_id": "a", "frozen_geometry": _rect_frozen(2, 2, 14, 14)},
            {"op": "smooth", "params": {"sigma_px": 3.0},
             "roi_id": "b", "frozen_geometry": _rect_frozen(40, 22, 14, 14)},
        ]}
        out = apply_processing_state(arr, processing_state_from_gui(gui))

        assert _std(out, 4, 14, 4, 14) < SMOOTHED, "ROI A filter lost"
        assert _std(out, 42, 52, 24, 34) < SMOOTHED, "ROI B filter lost"
        # A region neither filter covers stays raw.
        assert _std(out, 22, 36, 4, 16) > UNTOUCHED

    def test_frozen_mask_ignores_live_mask_replacement(self):
        """A committed mask-scoped filter replays the raster snapshot taken at
        commit time, not the (since replaced) live mask."""
        arr = _noise()
        commit_raster = np.zeros(arr.shape, dtype=bool)
        commit_raster[2:18, 2:18] = True

        live = np.zeros(arr.shape, dtype=bool)
        live[22:38, 40:56] = True
        mask = ImageMask.new(live, name="m1")
        mask_set = MaskSet(image_id="t")
        mask_set.add(mask)

        gui = {"mask_filter_ops": [{
            "op": "smooth", "params": {"sigma_px": 3.0},
            "mask_id": mask.id,
            "frozen_mask": {"data": _pack_bool(commit_raster),
                            "shape": list(arr.shape)},
        }]}
        out = apply_processing_state(
            arr, processing_state_from_gui(gui), mask_set=mask_set)

        assert _std(out, 4, 16, 4, 16) < SMOOTHED, "frozen raster not used"
        assert _std(out, 42, 54, 24, 36) > UNTOUCHED, (
            "filter followed the replaced live mask instead of the snapshot"
        )

    def test_live_mask_shape_mismatch_warns_and_skips(self):
        """A live-mask step whose raster no longer matches the image shape is
        skipped with a warning — never applied at the wrong offset."""
        arr = _noise()
        stale = ImageMask.new(np.ones((10, 10), dtype=bool), name="stale")
        mask_set = MaskSet(image_id="t")
        mask_set.add(stale)
        state = ProcessingState(steps=[ProcessingStep("mask", {
            "mask_id": stale.id,
            "step": {"op": "smooth", "params": {"sigma_px": 3.0}},
        })])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = apply_processing_state(arr, state, mask_set=mask_set)

        assert any("shape" in str(w.message) for w in caught)
        np.testing.assert_array_equal(out, arr.astype(np.float64))

    def test_missing_live_mask_reference_blocks_export(self):
        """Export validation must raise for a live mask id that is gone."""
        state = ProcessingState(steps=[ProcessingStep("mask", {
            "mask_id": "gone",
            "step": {"op": "smooth", "params": {"sigma_px": 2.0}},
        })])
        with pytest.raises(ValueError, match="Export aborted"):
            assert_roi_references_resolved(
                state, ROISet(image_id="t"), MaskSet(image_id="t"))


# ── Replay ordering vs geometric ops (confirmed bug) ──────────────────────────

class TestScopeReplayOrdering:
    """Committed scope filters must replay at the pipeline position they were
    committed at (``after_geometric_ops``), so frozen geometry is always
    interpreted in the frame it was captured in.

    The 2026-06-11 review found that scope steps always replayed before
    geometric_ops: a region committed after a flip landed at the mirrored
    location. Fixed by recording the geometric-op count at commit time and
    interleaving scope steps at that position in processing_state_from_gui.
    """

    def test_region_committed_after_flip_lands_where_drawn(self):
        arr = _noise()
        # GUI history: flip applied first, THEN the user drew a selection at
        # display coords (2,2)-(18,18) and committed a smooth inside it.
        gui = {
            "geometric_ops": [{"op": "flip_horizontal"}],
            "roi_filter_ops": [_region_smooth_spec(2, 2, 16, 16,
                                                   after_geometric_ops=1)],
        }
        state = processing_state_from_gui(gui)
        assert [s.op for s in state.steps] == ["flip_horizontal", "roi"]
        out = apply_processing_state(arr, state)
        # The displayed/exported result must be smoothed where the user drew.
        assert _std(out, 4, 16, 4, 16) < SMOOTHED
        # ... and must NOT be smoothed at the mirrored location.
        w = arr.shape[1]
        assert _std(out, w - 16, w - 4, 4, 16) > UNTOUCHED

    def test_region_committed_before_flip_follows_the_flip(self):
        """A region committed BEFORE a flip replays in the raw frame and is
        carried along by the flip — the frozen-geometry model's intent."""
        arr = _noise()
        gui = {
            "roi_filter_ops": [_region_smooth_spec(2, 2, 16, 16,
                                                   after_geometric_ops=0)],
            "geometric_ops": [{"op": "flip_horizontal"}],
        }
        out = apply_processing_state(arr, processing_state_from_gui(gui))
        w = arr.shape[1]
        assert _std(out, w - 16, w - 4, 4, 16) < SMOOTHED
        assert _std(out, 4, 16, 4, 16) > UNTOUCHED

    def test_legacy_spec_without_position_keeps_historical_order(self):
        """Entries without after_geometric_ops (pre-fix sidecars/provenance)
        must keep their historical position before all geometric ops, so old
        saved states replay byte-identically."""
        gui = {
            "geometric_ops": [{"op": "flip_horizontal"}],
            "roi_filter_ops": [_region_smooth_spec(2, 2, 16, 16)],
        }
        state = processing_state_from_gui(gui)
        assert [s.op for s in state.steps] == ["roi", "flip_horizontal"]

    def test_region_committed_after_rot90_lands_where_drawn(self):
        """rotate_90_cw transposes the frame (40x60 → 60x40): the frozen
        geometry only makes sense interpreted on the rotated array."""
        arr = _noise()  # (40, 60)
        gui = {
            "geometric_ops": [{"op": "rotate_90_cw"}],
            "roi_filter_ops": [_region_smooth_spec(2, 2, 12, 12,
                                                   after_geometric_ops=1)],
        }
        out = apply_processing_state(arr, processing_state_from_gui(gui))
        assert out.shape == (60, 40)
        assert _std(out, 4, 12, 4, 12) < SMOOTHED
        assert _std(out, 20, 36, 30, 50) > UNTOUCHED

    def test_commits_at_different_positions_both_land_where_drawn(self):
        """Region A committed before the flip, region B after it: each must
        replay in its own frame, in commit order."""
        arr = _noise()
        w = arr.shape[1]
        gui = {
            "geometric_ops": [{"op": "flip_horizontal"}],
            "roi_filter_ops": [
                # A: drawn pre-flip at the left edge → displayed mirrored.
                _region_smooth_spec(2, 2, 14, 14, after_geometric_ops=0),
                # B: drawn post-flip at the bottom-left of the display.
                _region_smooth_spec(2, 24, 14, 14, after_geometric_ops=1),
            ],
        }
        state = processing_state_from_gui(gui)
        assert [s.op for s in state.steps] == ["roi", "flip_horizontal", "roi"]
        out = apply_processing_state(arr, state)
        assert _std(out, w - 14, w - 4, 4, 14) < SMOOTHED, "A not carried by flip"
        assert _std(out, 4, 14, 26, 36) < SMOOTHED, "B not at drawn location"
        # A's pre-flip location and B's mirrored location stay raw.
        assert _std(out, 4, 14, 4, 14) > UNTOUCHED
        assert _std(out, w - 14, w - 4, 26, 36) > UNTOUCHED

    def test_frozen_mask_committed_after_rot90_applies_in_rotated_frame(self):
        """A frozen mask raster snapshotted after a rot90 has the transposed
        display shape; replayed at its commit position it matches and applies
        (before the fix it was skipped with a shape-mismatch warning)."""
        arr = _noise()  # (40, 60)
        raster = np.zeros((60, 40), dtype=bool)  # post-rot90 display frame
        raster[4:16, 4:16] = True
        gui = {
            "geometric_ops": [{"op": "rotate_90_cw"}],
            "mask_filter_ops": [{
                "op": "smooth", "params": {"sigma_px": 3.0},
                "mask_id": "m1",
                "frozen_mask": {"data": _pack_bool(raster), "shape": [60, 40]},
                "after_geometric_ops": 1,
            }],
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = apply_processing_state(arr, processing_state_from_gui(gui))
        assert not any("shape" in str(w.message) for w in caught)
        assert out.shape == (60, 40)
        assert _std(out, 4, 16, 4, 16) < SMOOTHED

    def test_position_beyond_geometric_ops_is_not_dropped(self):
        """A position past the end of geometric_ops (partially restored or
        hand-edited state) emits after all geometric ops instead of being
        silently dropped."""
        gui = {
            "geometric_ops": [{"op": "flip_horizontal"}],
            "roi_filter_ops": [_region_smooth_spec(2, 2, 16, 16,
                                                   after_geometric_ops=99)],
        }
        state = processing_state_from_gui(gui)
        assert [s.op for s in state.steps] == ["flip_horizontal", "roi"]

    def test_commit_paths_record_current_geometric_op_count(self):
        """The viewer commit helpers must stamp after_geometric_ops with the
        number of geometric ops in the history at commit time."""
        from unittest.mock import MagicMock

        from probeflow.gui.viewer.image_viewer_processing_export_mixin import (
            ImageViewerProcessingExportMixin,
        )

        class Host(ImageViewerProcessingExportMixin):
            def __init__(self, processing, roi_set=None):
                self._processing = processing
                self._image_roi_set = roi_set
                self._roi_status_lbl = MagicMock()
                self._processing_panel = MagicMock()

            def _set_advanced_processing_state(self, state):
                pass

        geometric = [{"op": "flip_horizontal"}, {"op": "rotate_90_cw"}]

        host = Host({"smooth_sigma": 2.0, "geometric_ops": list(geometric)})
        host._commit_region_scoped_filters(
            {"kind": "rectangle",
             "geometry": {"x": 1, "y": 1, "width": 5, "height": 5}})
        committed = host._processing["roi_filter_ops"]
        assert [c["after_geometric_ops"] for c in committed] == [2]
        assert "smooth_sigma" not in host._processing

        roi = ROI.new("rectangle", {"x": 1, "y": 1, "width": 5, "height": 5})
        roi_set = ROISet(image_id="t")
        roi_set.add(roi)
        host2 = Host({"smooth_sigma": 2.0, "geometric_ops": list(geometric)},
                     roi_set)
        host2._commit_roi_scoped_filters(roi.id)
        committed2 = host2._processing["roi_filter_ops"]
        assert [c["after_geometric_ops"] for c in committed2] == [2]


# ── Operand resolver injection (confirmed bug) ────────────────────────────────

class TestOperandResolverInjection:
    @staticmethod
    def _resolver_recorder(shape):
        calls: list[dict] = []

        def resolver(params: dict) -> np.ndarray:
            calls.append(dict(params))
            return np.full(shape, 5.0)

        return resolver, calls

    def test_top_level_arithmetic_uses_injected_resolver(self):
        arr = np.ones((20, 20))
        resolver, calls = self._resolver_recorder(arr.shape)
        state = ProcessingState(steps=[ProcessingStep("arithmetic", {
            "operation": "add", "operand_type": "image",
            "source_path": "/nonexistent/operand.sxm", "plane_idx": 0,
        })])
        out = apply_processing_state(arr, state, operand_resolver=resolver)
        assert len(calls) == 1
        np.testing.assert_allclose(out, 6.0)

    @pytest.mark.parametrize("scope", ["roi", "mask"])
    def test_scoped_arithmetic_uses_injected_resolver(self, scope):
        """operand_resolver must be forwarded into the nested recursion for
        'roi'/'mask' scope steps — scoped image arithmetic previously fell
        back to disk I/O even when the caller injected a resolver."""
        arr = np.ones((20, 20))
        resolver, calls = self._resolver_recorder(arr.shape)
        nested = {"op": "arithmetic",
                  "params": {"operation": "add", "operand_type": "image",
                             "source_path": "/nonexistent/operand.sxm",
                             "plane_idx": 0}}
        if scope == "roi":
            params = {"roi_id": "r1",
                      "frozen_geometry": _rect_frozen(0, 0, 10, 10),
                      "step": nested}
        else:
            raster = np.zeros(arr.shape, dtype=bool)
            raster[:10, :10] = True
            params = {"mask_id": "m1",
                      "frozen_mask": {"data": _pack_bool(raster),
                                      "shape": list(arr.shape)},
                      "step": nested}
        state = ProcessingState(steps=[ProcessingStep(scope, params)])

        out = apply_processing_state(arr, state, operand_resolver=resolver)

        assert len(calls) == 1, "injected resolver was bypassed"
        assert out[5, 5] == pytest.approx(6.0)
        assert out[15, 15] == pytest.approx(1.0)


# ── Export validation honesty ─────────────────────────────────────────────────

class TestExportValidationHonesty:
    @pytest.mark.parametrize("scope", ["roi", "mask"])
    def test_malformed_frozen_snapshot_blocks_export(self, scope):
        """A frozen snapshot that cannot be rebuilt is silently skipped at
        replay, so validation must flag it — otherwise an export would
        silently omit the committed filter. The skip itself must warn."""
        arr = _noise((20, 20))
        if scope == "roi":
            params = {"roi_id": "r1",
                      # invalid kind ⇒ snapshot cannot be rebuilt
                      "frozen_geometry": {"kind": "rect",
                                          "geometry": {"x": 0, "y": 0,
                                                       "width": 10, "height": 10},
                                          "coord_system": "pixel"},
                      "step": {"op": "smooth", "params": {"sigma_px": 3.0}}}
        else:
            params = {"mask_id": "m1",
                      "frozen_mask": {"data": "!!!not-base64!!!", "shape": [20, 20]},
                      "step": {"op": "smooth", "params": {"sigma_px": 3.0}}}
        state = ProcessingState(steps=[ProcessingStep(scope, params)])

        flagged = missing_roi_references(
            state, ROISet(image_id="t"), MaskSet(image_id="t"))
        assert flagged and "malformed" in str(flagged[0]["value"])
        with pytest.raises(ValueError, match="Export aborted"):
            assert_roi_references_resolved(
                state, ROISet(image_id="t"), MaskSet(image_id="t"))

        # The interactive path still degrades gracefully — but loudly.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = apply_processing_state(arr, state)
        assert any("malformed" in str(w.message) for w in caught)
        np.testing.assert_array_equal(out, arr.astype(np.float64))

    @pytest.mark.parametrize("scope", ["roi", "mask"])
    def test_valid_frozen_snapshot_still_passes_validation(self, scope):
        """The malformed-snapshot check must not flag healthy frozen steps."""
        if scope == "roi":
            params = {"roi_id": "r1",
                      "frozen_geometry": _rect_frozen(0, 0, 10, 10),
                      "step": {"op": "smooth", "params": {"sigma_px": 3.0}}}
        else:
            raster = np.zeros((20, 20), dtype=bool)
            raster[:10, :10] = True
            params = {"mask_id": "m1",
                      "frozen_mask": {"data": _pack_bool(raster),
                                      "shape": [20, 20]},
                      "step": {"op": "smooth", "params": {"sigma_px": 3.0}}}
        state = ProcessingState(steps=[ProcessingStep(scope, params)])
        assert missing_roi_references(
            state, ROISet(image_id="t"), MaskSet(image_id="t")) == []

    def test_frozen_roi_steps_around_rotate_do_not_warn(self):
        """Frozen-geometry roi steps carry their own frame and are valid on
        either side of a rotation — no warning (was: a misleading 'ROI steps
        have been skipped' fired for any roi step anywhere in the state)."""
        arr = _noise((50, 50))
        frozen_step = ProcessingStep("roi", {
            "roi_id": "x",
            "frozen_geometry": _rect_frozen(20, 20, 12, 12),
            "step": {"op": "smooth", "params": {"sigma_px": 3.0}},
        })
        rotate = ProcessingStep("rotate_arbitrary", {"angle_degrees": 5.0})
        for order in ([frozen_step, rotate], [rotate, frozen_step]):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                apply_processing_state(arr, ProcessingState(steps=list(order)))
            assert not [w for w in caught
                        if "skipped" in str(w.message)
                        or "mislocated" in str(w.message)]

    def test_live_roi_step_after_rotate_warns_mislocated(self):
        """The genuinely ambiguous case — a live-resolving roi step after the
        rotation — warns about what actually happens (possible mislocation),
        not about a skip that never occurs."""
        arr = _noise((30, 30))
        roi = ROI.new("rectangle", {"x": 2, "y": 2, "width": 8, "height": 8})
        roi_set = ROISet(image_id="t")
        roi_set.add(roi)
        state = ProcessingState(steps=[
            ProcessingStep("rotate_arbitrary", {"angle_degrees": 5.0}),
            ProcessingStep("roi", {
                "roi_id": roi.id,
                "step": {"op": "smooth", "params": {"sigma_px": 2.0}},
            }),
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            apply_processing_state(arr, state, roi_set)
        assert any("mislocated" in str(w.message) for w in caught)
        assert not any("skipped" in str(w.message) for w in caught)

    def test_roi_scoped_arithmetic_preserves_frozen_geometry(self):
        """Nested-roi arithmetic_ops specs must keep their frozen_geometry
        (was: rebuilt with roi_id only, so scoped arithmetic silently
        followed later ROI moves while every other scoped filter froze)."""
        gui = {"arithmetic_ops": [{
            "op": "roi",
            "params": {
                "roi_id": "a",
                "frozen_geometry": _rect_frozen(2, 2, 10, 10),
                "step": {"op": "arithmetic",
                         "params": {"operation": "add",
                                    "operand_type": "constant",
                                    "value_si": 1.0}},
            },
        }]}
        state = processing_state_from_gui(gui)
        roi_steps = [s for s in state.steps if s.op == "roi"]
        assert roi_steps, "roi-scoped arithmetic step missing entirely"
        assert roi_steps[0].params.get("frozen_geometry") is not None

    def test_frozen_arithmetic_committed_after_flip_lands_where_drawn(self):
        """Flat arithmetic_ops spec with frozen_geometry + after_geometric_ops
        (the new producer format) replays at its commit position: the value
        offset lands inside the region the user drew on the flipped display."""
        arr = _noise()
        w = arr.shape[1]
        gui = {
            "geometric_ops": [{"op": "flip_horizontal"}],
            "arithmetic_ops": [{
                "op": "arithmetic",
                "params": {"operation": "add", "operand_type": "constant",
                           "value_si": 100.0},
                "roi_id": "a",
                "frozen_geometry": _rect_frozen(2, 2, 10, 10),
                "after_geometric_ops": 1,
            }],
        }
        state = processing_state_from_gui(gui)
        assert [s.op for s in state.steps] == ["flip_horizontal", "roi"]
        out = apply_processing_state(arr, state)
        assert float(out[4:10, 4:10].mean()) > 50.0, "offset not where drawn"
        assert float(out[4:10, w - 10:w - 4].mean()) < 50.0, "offset mirrored"

    def test_legacy_live_roi_arithmetic_keeps_position_and_live_resolution(self):
        """roi_id-only arithmetic entries (pre-fix saves) must keep the
        historical behaviour: emitted after geometric_ops, resolving the
        live ROI set."""
        roi = ROI.new("rectangle", {"x": 2, "y": 2, "width": 10, "height": 10})
        roi_set = ROISet(image_id="t")
        roi_set.add(roi)
        gui = {
            "geometric_ops": [{"op": "flip_horizontal"}],
            "arithmetic_ops": [{
                "op": "arithmetic",
                "params": {"operation": "add", "operand_type": "constant",
                           "value_si": 100.0},
                "roi_id": roi.id,
            }],
        }
        state = processing_state_from_gui(gui)
        assert [s.op for s in state.steps] == ["flip_horizontal", "roi"]
        assert state.steps[1].params.get("frozen_geometry") is None
        out = apply_processing_state(_noise(), state, roi_set)
        assert float(out[4:10, 4:10].mean()) > 50.0


class TestScanReplayEndToEnd:
    def test_positioned_scope_state_replays_through_scan_export_path(self):
        """A saved GUI dict carrying after_geometric_ops replays correctly
        through apply_processing_state_to_scan — the function the CLI and
        export paths use — landing the committed filter where it was drawn
        on the flipped display."""
        from types import SimpleNamespace

        from probeflow.processing.gui_adapter import apply_processing_state_to_scan

        arr = _noise()
        w = arr.shape[1]
        scan = SimpleNamespace(
            planes=[arr.copy()],
            scan_range_m=(50e-9, 40e-9),
            processing_state=None,
        )
        scan.record_processing_state = lambda state: setattr(
            scan, "processing_state", state)

        gui = {
            "geometric_ops": [{"op": "flip_horizontal"}],
            "roi_filter_ops": [_region_smooth_spec(2, 2, 16, 16,
                                                   after_geometric_ops=1)],
        }
        apply_processing_state_to_scan(scan, gui, plane_idx=0)

        out = scan.planes[0]
        assert _std(out, 4, 16, 4, 16) < SMOOTHED, "filter not where drawn"
        assert _std(out, w - 16, w - 4, 4, 16) > UNTOUCHED, "filter mirrored"
        assert scan.processing_state is not None


class TestSetZeroPositioning:
    def test_stamped_zero_plane_replays_after_the_flip(self):
        """A zero plane picked on the flipped display (stamped position 1)
        must replay after the flip, anchoring the clicked coordinates."""
        gui = {
            "set_zero_plane_points": [(2, 2), (30, 4), (10, 28)],
            "set_zero_patch": 1,
            "set_zero_after_geometric_ops": 1,
            "geometric_ops": [{"op": "flip_horizontal"}],
        }
        state = processing_state_from_gui(gui)
        assert [s.op for s in state.steps] == ["flip_horizontal", "set_zero_plane"]

        arr = _noise()
        out = apply_processing_state(arr, state)
        # patch=1 means a 3x3 sampling window: the plane passes exactly
        # through the patch means, so those (not the single pixels) are 0.
        for x, y in gui["set_zero_plane_points"]:
            patch = out[y - 1:y + 2, x - 1:x + 2]
            assert float(np.mean(patch)) == pytest.approx(0.0, abs=1e-12)

    def test_legacy_unstamped_zero_keeps_historical_order(self):
        gui = {
            "set_zero_plane_points": [(2, 2), (30, 4), (10, 28)],
            "set_zero_patch": 1,
            "geometric_ops": [{"op": "flip_horizontal"}],
        }
        state = processing_state_from_gui(gui)
        assert [s.op for s in state.steps] == ["set_zero_plane", "flip_horizontal"]

    def test_stamped_zero_point_orders_after_geometric_ops(self):
        gui = {
            "set_zero_xy": (5, 6),
            "set_zero_after_geometric_ops": 1,
            "geometric_ops": [{"op": "rotate_90_cw"}],
        }
        state = processing_state_from_gui(gui)
        assert [s.op for s in state.steps] == ["rotate_90_cw", "set_zero_point"]
