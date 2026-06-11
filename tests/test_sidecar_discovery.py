"""Adversarial tests for sidecar discovery precedence (review focus #3).

The contract: the canonical sidecar (``.rois.json`` / ``.masks.json``) wins;
provenance sidecars (``.probeflow.json`` / ``.provenance.json``) still allow
replay when the canonical one is absent. The 2026-06-11 review found the
chain stopped at the first *existing* candidate: a processing-only provenance
export (``"rois": null``) raised "contains no ROISet data" — violating
``missing_ok`` and hiding ROIs/masks present in a later candidate (call sites
swallow the exception, so the data silently never loaded).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from probeflow.core.mask import ImageMask, MaskSet
from probeflow.core.roi import ROI, ROISet
from probeflow.io.mask_sidecar import load_mask_set_sidecar
from probeflow.io.roi_sidecar import load_roi_set_sidecar


def _roi_set(name: str) -> ROISet:
    rs = ROISet(image_id="img")
    rs.add(ROI.new("rectangle", {"x": 1, "y": 1, "width": 5, "height": 5},
                   name=name))
    return rs


def _mask_set(name: str) -> MaskSet:
    ms = MaskSet(image_id="img")
    raster = np.zeros((8, 8), dtype=bool)
    raster[:4, :4] = True
    ms.add(ImageMask.new(raster, name=name))
    return ms


def _write(path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def scan(tmp_path):
    p = tmp_path / "A250320.191933.dat"
    p.write_bytes(b"fake scan")
    return p


class TestRoiDiscovery:
    def test_processing_only_provenance_is_not_an_error(self, scan):
        """A provenance export with ``"rois": null`` must read as "no ROIs
        here", honouring missing_ok — not raise (which call sites swallow,
        silently producing an empty set even when other candidates exist)."""
        _write(scan.parent / f"{scan.stem}.probeflow.json",
               {"rois": None, "masks": None, "processing_state": {}})
        roi_set, used = load_roi_set_sidecar(scan, missing_ok=True)
        assert roi_set is None
        assert used == scan.parent / f"{scan.stem}.rois.json"

    def test_chain_continues_past_payload_less_candidate(self, scan):
        """ROIs in a later candidate must still be found when an earlier
        candidate exists but holds no ROI data."""
        _write(scan.parent / f"{scan.stem}.probeflow.json",
               {"rois": None, "processing_state": {}})
        _write(scan.parent / f"{scan.stem}.provenance.json",
               {"rois": _roi_set("from-provenance").to_dict()})
        roi_set, used = load_roi_set_sidecar(scan)
        assert roi_set is not None
        assert roi_set.rois[0].name == "from-provenance"
        assert used.name.endswith(".provenance.json")

    def test_canonical_wins_over_provenance(self, scan):
        _write(scan.parent / f"{scan.stem}.rois.json",
               _roi_set("canonical").to_dict())
        _write(scan.parent / f"{scan.stem}.probeflow.json",
               {"rois": _roi_set("stale-export").to_dict()})
        roi_set, used = load_roi_set_sidecar(scan)
        assert roi_set.rois[0].name == "canonical"
        assert used.name.endswith(".rois.json")

    def test_empty_canonical_set_still_wins(self, scan):
        """An empty canonical sidecar means "the user deleted all ROIs" — it
        must not be skipped in favour of a stale provenance export."""
        _write(scan.parent / f"{scan.stem}.rois.json",
               ROISet(image_id="img").to_dict())
        _write(scan.parent / f"{scan.stem}.probeflow.json",
               {"rois": _roi_set("stale-export").to_dict()})
        roi_set, used = load_roi_set_sidecar(scan)
        assert roi_set is not None and roi_set.rois == []
        assert used.name.endswith(".rois.json")

    def test_corrupt_candidate_raises_not_silently_skipped(self, scan):
        """A damaged sidecar is data loss — surface it rather than silently
        substituting a stale fallback."""
        (scan.parent / f"{scan.stem}.rois.json").write_text(
            "{truncated", encoding="utf-8")
        _write(scan.parent / f"{scan.stem}.probeflow.json",
               {"rois": _roi_set("stale-export").to_dict()})
        with pytest.raises(ValueError, match="Could not read"):
            load_roi_set_sidecar(scan)

    def test_only_payload_less_candidates_without_missing_ok_raises_not_found(self, scan):
        _write(scan.parent / f"{scan.stem}.probeflow.json", {"rois": None})
        with pytest.raises(FileNotFoundError, match="tried"):
            load_roi_set_sidecar(scan)

    def test_explicit_sidecar_without_payload_still_raises(self, scan):
        """Pointing at a specific file keeps strict behaviour: no payload in
        the named file is an error, not a search miss."""
        side = scan.parent / "explicit.json"
        _write(side, {"rois": None})
        with pytest.raises(ValueError, match="no ROISet data"):
            load_roi_set_sidecar(scan, sidecar=side)


class TestMaskDiscovery:
    def test_processing_only_provenance_is_not_an_error(self, scan):
        _write(scan.parent / f"{scan.stem}.probeflow.json",
               {"rois": None, "masks": None, "processing_state": {}})
        mask_set, used = load_mask_set_sidecar(scan, missing_ok=True)
        assert mask_set is None
        assert used == scan.parent / f"{scan.stem}.masks.json"

    def test_chain_continues_past_payload_less_candidate(self, scan):
        _write(scan.parent / f"{scan.stem}.probeflow.json",
               {"masks": None, "processing_state": {}})
        _write(scan.parent / f"{scan.stem}.provenance.json",
               {"masks": _mask_set("from-provenance").to_dict()})
        mask_set, used = load_mask_set_sidecar(scan)
        assert mask_set is not None
        assert mask_set.masks[0].name == "from-provenance"
        assert used.name.endswith(".provenance.json")

    def test_rois_only_provenance_does_not_satisfy_mask_search(self, scan):
        """A provenance export carrying ROIs but no masks must not be
        mistaken for a mask source."""
        _write(scan.parent / f"{scan.stem}.probeflow.json",
               {"rois": _roi_set("r").to_dict(), "masks": None})
        mask_set, _used = load_mask_set_sidecar(scan, missing_ok=True)
        assert mask_set is None

    def test_corrupt_candidate_raises(self, scan):
        (scan.parent / f"{scan.stem}.masks.json").write_text(
            "[1,", encoding="utf-8")
        with pytest.raises(ValueError, match="Could not read"):
            load_mask_set_sidecar(scan)


class TestWriterLoaderFormatContract:
    def test_export_record_embeds_rois_and_masks_where_loaders_look(self, scan):
        """Pin the provenance writer ↔ sidecar loader format agreement: a
        ProvenanceRecord-style dict written to .probeflow.json must be
        readable by BOTH loaders, so replay-from-provenance keeps working if
        either side's format drifts."""
        record_dict = {
            "source_file": str(scan),
            "processing_state": {"steps": []},
            "rois": _roi_set("embedded").to_dict(),
            "masks": _mask_set("embedded").to_dict(),
        }
        _write(scan.parent / f"{scan.stem}.probeflow.json", record_dict)

        roi_set, _ = load_roi_set_sidecar(scan)
        mask_set, _ = load_mask_set_sidecar(scan)
        assert roi_set.rois[0].name == "embedded"
        assert mask_set.masks[0].name == "embedded"
        assert mask_set.masks[0].count() == 16


# ── GUI wrapper: corrupt sidecars must be visible, not read as "empty" ─────────

class TestGuiWrapperErrorSurface:
    def test_corrupt_roi_sidecar_returns_error_message(self, scan):
        from probeflow.gui.viewer.roi_sidecar import load_roi_set

        (scan.parent / f"{scan.stem}.rois.json").write_text(
            "{broken", encoding="utf-8")
        roi_set, err = load_roi_set(scan)
        assert roi_set.rois == []
        assert err is not None and "Could not load ROI sidecar" in err

    def test_corrupt_mask_sidecar_returns_error_message(self, scan):
        from probeflow.gui.viewer.mask_sidecar import load_mask_set

        (scan.parent / f"{scan.stem}.masks.json").write_text(
            "[1,", encoding="utf-8")
        mask_set, err = load_mask_set(scan)
        assert mask_set.masks == []
        assert err is not None and "Could not load mask sidecar" in err

    def test_missing_sidecar_is_not_an_error(self, scan):
        from probeflow.gui.viewer.mask_sidecar import load_mask_set
        from probeflow.gui.viewer.roi_sidecar import load_roi_set

        roi_set, roi_err = load_roi_set(scan)
        mask_set, mask_err = load_mask_set(scan)
        assert roi_err is None and mask_err is None
        assert roi_set.rois == [] and mask_set.masks == []

    def test_valid_sidecar_loads_without_error(self, scan):
        from probeflow.gui.viewer.roi_sidecar import load_roi_set

        _write(scan.parent / f"{scan.stem}.rois.json",
               _roi_set("kept").to_dict())
        roi_set, err = load_roi_set(scan)
        assert err is None
        assert roi_set.rois[0].name == "kept"

    def test_viewer_loaders_surface_corrupt_sidecar_on_status(self, scan):
        """The mixin load paths must show the error, not silently present an
        empty set the user might overwrite on the next save."""
        from types import SimpleNamespace

        from probeflow.gui.viewer.image_viewer_mask_mixin import (
            ImageViewerMaskMixin,
        )
        from probeflow.gui.viewer.image_viewer_roi_mixin import (
            ImageViewerRoiMixin,
        )

        (scan.parent / f"{scan.stem}.rois.json").write_text(
            "{broken", encoding="utf-8")
        (scan.parent / f"{scan.stem}.masks.json").write_text(
            "[1,", encoding="utf-8")
        entry = SimpleNamespace(path=scan, stem=scan.stem)

        class Host(ImageViewerRoiMixin, ImageViewerMaskMixin):
            def __init__(self):
                self.statuses: list[str] = []
                self._status_lbl = SimpleNamespace(setText=self.statuses.append)
                self._zoom_lbl = SimpleNamespace(
                    set_roi_set=lambda *_: None,
                    set_mask_overlay=lambda *a, **k: None,
                    clear_mask_overlay=lambda: None,
                )
                self._display_arr = None
                self._raw_arr = None

            def _sync_viewer_menu_actions(self):
                pass

            def _channel_unit(self):
                return 1.0, "m", "Z"

        host = Host()
        host._load_image_roi_set(entry)
        host._load_image_mask_set(entry)

        assert any("ROI sidecar" in s for s in host.statuses)
        assert any("mask sidecar" in s for s in host.statuses)
