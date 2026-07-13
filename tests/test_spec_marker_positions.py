"""Spec-marker positioning seams (Qt-free).

Covers the 2026-07-06 spectroscopy display review findings:

* ``SpecOverlayController.load`` must apply the Createc scan-frame offset for
  ``.dat`` images (spec positions and scan offsets share the OffsetX/OffsetY
  DAC coordinate system) instead of assuming a frame centred at the origin.
* Out-of-frame spec positions must NOT be drawn as fabricated centre markers;
  they are skipped and reported via ``out_of_frame``.
* SCAN_ANGLE units: Nanonis stores degrees.  ``_parse_sxm_angle_rad`` converts
  to radians, and the Createc→SXM converter writes degrees (both previously
  treated the value as radians).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from probeflow.analysis.spec_plot import _parse_sxm_angle_rad
from probeflow.gui.viewer.spec_overlay import SpecOverlayController

class _ZoomLblStub:
    def __init__(self):
        self.calls: list[list] = []

    def set_markers(self, markers):
        self.calls.append(list(markers))


def _fake_spec(position):
    return SimpleNamespace(position=position, metadata={})


def _controller_with_folder(tmp_path, spec_stems, image_stem="scan1"):
    """Generate anonymous VERT files under each stem so sniffing passes."""
    from tests.synthetic_files import write_createc_vert

    for stem in spec_stems:
        write_createc_vert(tmp_path / f"{stem}.VERT", sweep="time")
    zoom = _ZoomLblStub()
    spec_image_map = {stem: image_stem for stem in spec_stems}
    ctrl = SpecOverlayController(zoom, spec_image_map)
    entry = SimpleNamespace(path=tmp_path / f"{image_stem}.dat", stem=image_stem)
    return ctrl, zoom, entry


# 40 nm × 40 nm Createc frame centred at (10, -20) nm.
_DAT_HEADER = {"OffsetX": "10", "OffsetY": "-20", "Dacto[A]xy": "1.0", "Rotation": "0.00"}
_RANGE_M = (40e-9, 40e-9)
_SHAPE = (64, 64)


def _load(ctrl, entry, positions, monkeypatch, scan_format="dat", header=None):
    it = iter(positions)
    monkeypatch.setattr(
        "probeflow.io.spectroscopy.read_spec_file",
        lambda path: _fake_spec(next(it)),
    )
    ctrl.load(
        entry,
        _RANGE_M,
        _SHAPE,
        scan_format,
        header if header is not None else dict(_DAT_HEADER),
        show=True,
    )


class TestCreatecOffsetApplied:
    def test_dat_scan_offset_places_marker_at_expected_fraction(self, tmp_path, monkeypatch):
        ctrl, zoom, entry = _controller_with_folder(tmp_path, ["spec1"])
        # Tip at (15, -15) nm → (dx, dy) = (5, 5) nm from the frame centre:
        # frac_x = (5+20)/40 = 0.625, frac_y = 1 - (5+20)/40 = 0.375.
        _load(ctrl, entry, [(15e-9, -15e-9)], monkeypatch)
        assert len(ctrl.markers) == 1
        m = ctrl.markers[0]
        assert m["frac_x"] == pytest.approx(0.625)
        assert m["frac_y"] == pytest.approx(0.375)
        assert ctrl.out_of_frame == []
        assert zoom.calls[-1] == ctrl.markers

    def test_origin_centred_assumption_mislocates(self):
        # The same position under the pre-fix (0,0) offset lands at a
        # different spot — pin that the header offset changes the result.
        from probeflow.analysis.spec_plot import spec_position_to_pixel

        wrong = spec_position_to_pixel(15e-9, -15e-9, _SHAPE, _RANGE_M, (0.0, 0.0), 0.0)
        assert wrong == pytest.approx((0.875, 0.875))


class TestOutOfFramePolicy:
    def test_out_of_frame_specs_are_skipped_not_centred(self, tmp_path, monkeypatch):
        ctrl, zoom, entry = _controller_with_folder(tmp_path, ["far_away", "inside"])
        # candidates iterate in sorted order: far_away.VERT, inside.VERT
        _load(ctrl, entry, [(500e-9, 500e-9), (10e-9, -20e-9)], monkeypatch)
        assert [m["entry"].stem for m in ctrl.markers] == ["inside"]
        # The in-frame marker sits at the true centre because we placed it
        # there — but no marker was fabricated for the out-of-frame spec.
        assert ctrl.out_of_frame == ["far_away"]
        assert all(
            not (m["entry"].stem == "far_away") for m in zoom.calls[-1]
        )

    def test_roi_set_matches_markers(self, tmp_path, monkeypatch):
        ctrl, _zoom, entry = _controller_with_folder(tmp_path, ["far_away", "inside"])
        _load(ctrl, entry, [(500e-9, 500e-9), (10e-9, -20e-9)], monkeypatch)
        assert ctrl.roi_set is not None
        assert len(list(ctrl.roi_set.rois)) == 1

    def test_reload_clears_previous_out_of_frame(self, tmp_path, monkeypatch):
        ctrl, _zoom, entry = _controller_with_folder(tmp_path, ["spec1"])
        _load(ctrl, entry, [(500e-9, 500e-9)], monkeypatch)
        assert ctrl.out_of_frame == ["spec1"]
        _load(ctrl, entry, [(10e-9, -20e-9)], monkeypatch)
        assert ctrl.out_of_frame == []
        assert len(ctrl.markers) == 1


class TestScanAngleUnits:
    def test_parse_sxm_angle_reads_degrees(self):
        assert _parse_sxm_angle_rad({"SCAN_ANGLE": "90"}) == pytest.approx(np.pi / 2)
        assert _parse_sxm_angle_rad({"SCAN_ANGLE": "0.000E+0"}) == 0.0
        assert _parse_sxm_angle_rad({}) == 0.0

    def test_converter_writes_scan_angle_in_degrees(self, tmp_path):
        from probeflow.io.converters.createc_dat_to_sxm import construct_hdr

        dat_path = tmp_path / "synthetic.dat"
        dat_path.write_bytes(b"")
        hdr = construct_hdr({"Rotation": "45.0"}, dat_path, num_chan=2)
        assert float(hdr["SCAN_ANGLE"]) == pytest.approx(45.0)

    def test_controller_sxm_angle_treated_as_degrees(self, tmp_path, monkeypatch):
        # 90° frame rotation: tip displaced (0, +10) nm from the frame centre
        # maps to dx_rot = +10 nm → frac_x = 0.75, frac_y = 0.5.
        ctrl, _zoom, entry = _controller_with_folder(tmp_path, ["spec1"])
        header = {"SCAN_OFFSET": "1.0E-8   -2.0E-8", "SCAN_ANGLE": "90"}
        _load(ctrl, entry, [(10e-9, -10e-9)], monkeypatch,
              scan_format="sxm", header=header)
        assert len(ctrl.markers) == 1
        assert ctrl.markers[0]["frac_x"] == pytest.approx(0.75)
        assert ctrl.markers[0]["frac_y"] == pytest.approx(0.5)
