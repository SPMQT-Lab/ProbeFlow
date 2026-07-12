"""Tests for CLI ROI argument helpers and processing commands."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytest

from probeflow.cli import main as cli_main
from probeflow.cli.roi_args import load_named_roi, resolve_inline_roi
from probeflow.core.roi import ROI, ROISet


TEST_DATA = Path(__file__).parent.parent / "test_data"
SAMPLE_DAT = TEST_DATA / "createc_scan_close_100nm.dat"


@pytest.fixture
def require_sample_dat():
    if not SAMPLE_DAT.exists():
        pytest.skip("test_data sample .dat not available")
    return SAMPLE_DAT


# ─── load_named_roi ──────────────────────────────────────────────────────────

class TestLoadNamedRoi:
    def test_missing_input_file_returns_none_and_logs(self, tmp_path, caplog):
        missing = tmp_path / "does_not_exist.sxm"
        with caplog.at_level(logging.ERROR):
            result = load_named_roi(missing, "terrace")
        assert result is None
        # An error message should have been emitted (FileNotFoundError caught)
        assert any(record.levelno == logging.ERROR for record in caplog.records)

    def test_roi_name_not_in_sidecar_returns_none(self, tmp_path, caplog):
        scan_path = tmp_path / "scan.sxm"
        scan_path.write_bytes(b"")
        # Write a sidecar that exists but contains no ROIs.
        roi_set = ROISet(image_id=str(scan_path))
        (tmp_path / "scan.rois.json").write_text(
            json.dumps(roi_set.to_dict()),
            encoding="utf-8",
        )

        with caplog.at_level(logging.ERROR):
            result = load_named_roi(scan_path, "no_such_roi")

        assert result is None
        # Should log "ROI ... not found"
        assert any("not found" in record.getMessage() for record in caplog.records)

    def test_returns_roi_object_when_found(self, tmp_path):
        scan_path = tmp_path / "scan.sxm"
        scan_path.write_bytes(b"")
        roi = ROI.new(
            "rectangle",
            {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0},
            name="terrace",
        )
        roi_set = ROISet(image_id=str(scan_path))
        roi_set.add(roi)
        (tmp_path / "scan.rois.json").write_text(
            json.dumps(roi_set.to_dict()),
            encoding="utf-8",
        )

        loaded = load_named_roi(scan_path, "terrace")

        assert loaded is not None
        assert loaded.id == roi.id
        assert loaded.name == "terrace"

    def test_returns_roi_when_looked_up_by_id(self, tmp_path):
        scan_path = tmp_path / "scan.sxm"
        scan_path.write_bytes(b"")
        roi = ROI.new(
            "rectangle",
            {"x": 0.0, "y": 0.0, "width": 5.0, "height": 5.0},
            name="alpha",
        )
        roi_set = ROISet(image_id=str(scan_path))
        roi_set.add(roi)
        (tmp_path / "scan.rois.json").write_text(
            json.dumps(roi_set.to_dict()),
            encoding="utf-8",
        )

        loaded = load_named_roi(scan_path, roi.id)

        assert loaded is not None
        assert loaded.id == roi.id


# ─── resolve_inline_roi ──────────────────────────────────────────────────────

def _ns(**kwargs):
    """Build a minimal argparse Namespace with defaulted ROI flags."""
    defaults = {
        "input": Path("/dev/null"),
        "roi_rect": None,
        "roi_polygon": None,
        "roi_line": None,
        "roi": None,
        "sidecar": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestResolveInlineRoi:
    def test_no_args_returns_none_and_no_error(self):
        args = _ns()
        roi, error = resolve_inline_roi(args)
        assert roi is None
        assert error is False

    def test_rect_args_returns_rectangle_roi(self):
        args = _ns(roi_rect=[10.0, 20.0, 50.0, 80.0])
        roi, error = resolve_inline_roi(args)
        assert error is False
        assert roi is not None
        assert roi.kind == "rectangle"
        # Geometry should encode min/abs (any rect orientation).
        assert roi.geometry["x"] == 10.0
        assert roi.geometry["y"] == 20.0
        assert roi.geometry["width"] == 40.0
        assert roi.geometry["height"] == 60.0

    def test_rect_args_normalises_reversed_corners(self):
        args = _ns(roi_rect=[50.0, 80.0, 10.0, 20.0])
        roi, error = resolve_inline_roi(args)
        assert error is False
        assert roi is not None
        assert roi.geometry["x"] == 10.0
        assert roi.geometry["y"] == 20.0
        assert roi.geometry["width"] == 40.0
        assert roi.geometry["height"] == 60.0

    def test_polygon_args_returns_polygon_roi(self):
        args = _ns(roi_polygon=[0.0, 0.0, 5.0, 0.0, 5.0, 5.0])
        roi, error = resolve_inline_roi(args)
        assert error is False
        assert roi is not None
        assert roi.kind == "polygon"
        assert roi.geometry["vertices"] == [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]

    def test_multiple_flags_returns_error(self, caplog):
        args = _ns(
            roi_rect=[0.0, 0.0, 1.0, 1.0],
            roi_polygon=[0.0, 0.0, 1.0, 0.0, 0.5, 1.0],
        )
        with caplog.at_level(logging.ERROR):
            roi, error = resolve_inline_roi(args)
        assert roi is None
        assert error is True
        assert any("at most one" in r.getMessage() for r in caplog.records)

    def test_line_disallowed_unless_explicitly_allowed(self):
        args = _ns(roi_line=[0.0, 0.0, 5.0, 5.0])
        # allow_line=False (default): the roi_line flag should be ignored,
        # so resolve_inline_roi behaves like the no-args case.
        roi, error = resolve_inline_roi(args, allow_line=False)
        assert roi is None
        assert error is False

    def test_line_allowed_returns_line_roi(self):
        args = _ns(roi_line=[0.0, 0.0, 5.0, 5.0])
        roi, error = resolve_inline_roi(args, allow_line=True)
        assert error is False
        assert roi is not None
        assert roi.kind == "line"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"roi_rect": [0.0, float("nan"), 5.0, 5.0]},
            {"roi_polygon": [0.0, 0.0, 5.0, 0.0, float("inf"), 5.0]},
            {"roi_line": [0.0, 0.0, float("-inf"), 5.0]},
        ],
    )
    def test_nonfinite_inline_coordinates_return_logged_error(
        self, kwargs, caplog
    ):
        with caplog.at_level(logging.ERROR):
            roi, error = resolve_inline_roi(
                _ns(**kwargs),
                allow_line="roi_line" in kwargs,
            )

        assert roi is None
        assert error is True
        assert any("finite" in record.getMessage() for record in caplog.records)


# ─── processing commands (smoke tests) ───────────────────────────────────────

class TestCmdAlignRows:
    def test_smoke_png_exit_zero(self, require_sample_dat, tmp_path):
        # Use --png because per-plane SXM processing provenance refuses to
        # write a modified plane back into an all-plane SXM.
        out = tmp_path / "aligned.png"
        rc = cli_main([
            "align-rows", str(require_sample_dat),
            "--method", "median",
            "--png",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()

    def test_missing_input_raises(self, tmp_path):
        # _cmd_single_op does not wrap load_scan; a missing file surfaces
        # as a FileNotFoundError from the loader.
        missing = tmp_path / "nope.dat"
        with pytest.raises(FileNotFoundError):
            cli_main([
                "align-rows", str(missing),
                "--method", "median",
                "--png",
                "-o", str(tmp_path / "out.png"),
            ])


class TestCmdSmooth:
    def test_smoke_png_exit_zero(self, require_sample_dat, tmp_path):
        out = tmp_path / "smoothed.png"
        rc = cli_main([
            "smooth", str(require_sample_dat),
            "--sigma", "1.0",
            "--png",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()

    def test_missing_input_raises(self, tmp_path):
        missing = tmp_path / "nope.dat"
        with pytest.raises(FileNotFoundError):
            cli_main([
                "smooth", str(missing),
                "--sigma", "1.0",
                "--png",
                "-o", str(tmp_path / "out.png"),
            ])


class TestCmdPlaneBg:
    def test_smoke_png_exit_zero(self, require_sample_dat, tmp_path):
        # plane-bg also writes per-plane provenance, so use --png for the
        # smoke test to avoid the all-plane SXM refusal.
        out = tmp_path / "bg.png"
        rc = cli_main([
            "plane-bg", str(require_sample_dat),
            "--order", "1",
            "--png",
            "-o", str(out),
        ])
        assert rc == 0
        assert out.exists()

    def test_missing_input_returns_one(self, tmp_path):
        # _cmd_plane_bg wraps load_scan in try/except → exits 1 instead of raising.
        missing = tmp_path / "nope.dat"
        out = tmp_path / "bg.png"
        rc = cli_main([
            "plane-bg", str(missing),
            "--order", "1",
            "--png",
            "-o", str(out),
        ])
        assert rc == 1
        assert not out.exists()
