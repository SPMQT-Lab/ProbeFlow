"""Tests for Qt-free viewer helper modules.

These tests run without a QApplication and verify that scan_load.py and
processed_export.py work correctly on their own.
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── ViewerScanData ────────────────────────────────────────────────────────────

SAMPLE_SXM = Path(__file__).parent.parent / "test_data" / "sample_input" / "A250320.191933.sxm"


@pytest.mark.skipif(not SAMPLE_SXM.exists(), reason="sample SXM not present")
def test_load_scan_for_viewer_returns_data():
    from probeflow.gui.viewer.scan_load import load_scan_for_viewer

    result = load_scan_for_viewer(SAMPLE_SXM, channel_idx=0)

    assert result.raw_arr is not None
    assert result.n_planes > 0
    assert len(result.plane_names) == result.n_planes
    assert result.scan_range_m is not None
    assert result.source_format != ""


@pytest.mark.skipif(not SAMPLE_SXM.exists(), reason="sample SXM not present")
def test_load_scan_for_viewer_clamps_channel():
    from probeflow.gui.viewer.scan_load import load_scan_for_viewer

    # Requesting a very large channel index should not raise
    result = load_scan_for_viewer(SAMPLE_SXM, channel_idx=9999)
    assert result.raw_arr is not None


def test_load_scan_for_viewer_missing_file_returns_fallback():
    from probeflow.gui.viewer.scan_load import load_scan_for_viewer

    result = load_scan_for_viewer(Path("/nonexistent/file.sxm"), channel_idx=0)

    assert result.raw_arr is None
    assert result.n_planes == 0
    assert result.processing_history is None
    assert len(result.plane_names) > 0  # fallback names still provided


# ── build_processed_scan_for_export ──────────────────────────────────────────

@pytest.mark.skipif(not SAMPLE_SXM.exists(), reason="sample SXM not present")
def test_build_processed_scan_for_export_uses_display_arr():
    from probeflow.gui.viewer.processed_export import build_processed_scan_for_export

    fake_arr = np.ones((16, 16), dtype=np.float64) * 42.0
    scan, idx = build_processed_scan_for_export(
        SAMPLE_SXM, channel_idx=0, display_arr=fake_arr, processing_gui_state={},
    )

    np.testing.assert_array_equal(scan.planes[idx], fake_arr)


@pytest.mark.skipif(not SAMPLE_SXM.exists(), reason="sample SXM not present")
def test_build_processed_scan_for_export_no_display_arr_uses_raw():
    from probeflow.gui.viewer.processed_export import build_processed_scan_for_export

    scan, idx = build_processed_scan_for_export(
        SAMPLE_SXM, channel_idx=0, display_arr=None, processing_gui_state={},
    )

    assert scan.planes[idx] is not None
    assert scan.planes[idx].ndim == 2


# ── save_processed_image ──────────────────────────────────────────────────────

def _make_fake_scan(arr=None):
    """Return a minimal Scan-like object for export tests."""
    if arr is None:
        arr = np.random.default_rng(0).random((32, 32))
    scan = MagicMock()
    scan.planes = [arr]
    scan.plane_names = ["Z (fwd)"]
    scan.scan_range_m = (100e-9, 100e-9)
    scan.processing_state = MagicMock()
    scan.processing_state.steps = []
    scan.n_planes = 1
    return scan


def test_save_processed_image_png(tmp_path):
    from probeflow.gui.viewer.processed_export import save_processed_image

    scan = _make_fake_scan()
    out = tmp_path / "result.png"

    with patch(
        "probeflow.provenance.export.build_scan_export_provenance",
        return_value="prov",
    ):
        msg = save_processed_image(
            scan, 0, out, display_settings={"colormap": "gray"}
        )

    scan.save_png.assert_called_once_with(
        out, plane_idx=0,
        colormap="gray", clip_low=1.0, clip_high=99.0,
        add_scalebar=True, provenance="prov",
    )
    assert "Saved processed image" in msg


def test_save_processed_image_pdf_can_skip_provenance(tmp_path):
    from probeflow.gui.viewer.processed_export import save_processed_image

    scan = _make_fake_scan()
    out = tmp_path / "result.pdf"

    with patch(
        "probeflow.provenance.export.build_scan_export_provenance",
        return_value="prov",
    ) as build_provenance:
        msg = save_processed_image(
            scan, 0, out,
            display_settings={"colormap": "gray"},
            include_provenance=False,
        )

    build_provenance.assert_not_called()
    scan.save_pdf.assert_called_once_with(
        out, plane_idx=0,
        colormap="gray", clip_low=1.0, clip_high=99.0,
        show_scalebar=True,
        provenance=None,
    )
    assert "Saved processed image" in msg


def test_save_processed_image_png_can_skip_scalebar(tmp_path):
    from probeflow.gui.viewer.processed_export import save_processed_image

    scan = _make_fake_scan()
    out = tmp_path / "result.png"

    msg = save_processed_image(scan, 0, out, add_scalebar=False)

    scan.save_png.assert_called_once_with(
        out, plane_idx=0,
        colormap="gray", clip_low=1.0, clip_high=99.0,
        add_scalebar=False, provenance=None,
    )
    assert "Saved processed image" in msg


def test_save_processed_image_gwy_can_skip_provenance(tmp_path):
    from probeflow.gui.viewer.processed_export import save_processed_image

    scan = _make_fake_scan()
    out = tmp_path / "result.gwy"

    msg = save_processed_image(
        scan, 0, out,
        display_settings={"colormap": "gray"},
        include_provenance=False,
    )

    scan.save_gwy.assert_called_once_with(
        out, plane_idx=0,
        include_provenance=False,
        include_meta=False,
        provenance=None,
    )
    assert "Saved processed image" in msg


def test_save_processed_image_csv(tmp_path):
    from probeflow.gui.viewer.processed_export import save_processed_image

    scan = _make_fake_scan()
    out = tmp_path / "result.csv"

    with patch(
        "probeflow.provenance.export.build_scan_export_provenance",
        return_value="prov",
    ), patch(
        "probeflow.provenance.export.check_provenance_sidecar_collisions"
    ):
        msg = save_processed_image(
            scan, 0, out, display_settings={"colormap": "gray"}
        )

    scan.save_csv.assert_called_once_with(out, plane_idx=0, provenance="prov")
    assert "Saved processed image" in msg


def test_save_processed_image_unsupported_suffix(tmp_path):
    from probeflow.gui.viewer.processed_export import save_processed_image

    scan = _make_fake_scan()
    out = tmp_path / "result.xyz"

    msg = save_processed_image(scan, 0, out)

    assert "Unsupported" in msg


def test_save_processed_image_sxm_blocked_multiplane(tmp_path):
    from probeflow.gui.viewer.processed_export import save_processed_image

    scan = _make_fake_scan()
    scan.n_planes = 3
    scan.processing_state.steps = [{"id": "plane_bg"}]
    out = tmp_path / "result.sxm"

    msg = save_processed_image(scan, 0, out)

    assert "blocked" in msg
    scan.save_sxm.assert_not_called()


def test_save_processed_image_exception_returns_error_string(tmp_path):
    from probeflow.gui.viewer.processed_export import save_processed_image

    scan = _make_fake_scan()
    scan.save_png.side_effect = RuntimeError("disk full")
    out = tmp_path / "result.png"

    msg = save_processed_image(scan, 0, out)

    assert "Save processed image error" in msg
    assert "disk full" in msg


# ── save_viewer_png ──────────────────────────────────────────────────────────

def test_save_viewer_png_can_skip_provenance(tmp_path):
    from probeflow.gui.viewer.png_export import save_viewer_png

    arr = np.zeros((8, 12), dtype=float)
    out = tmp_path / "viewer.png"
    fake_scan = MagicMock()
    fake_scan.scan_range_m = (12e-9, 8e-9)
    drs = MagicMock()
    drs.resolve.return_value = (0.0, 1.0)

    with patch(
        "probeflow.core.scan_loader.load_scan",
        return_value=fake_scan,
    ), patch(
        "probeflow.provenance.export.build_scan_export_provenance",
        return_value="prov",
    ) as build_provenance, patch(
        "probeflow.processing.export_png",
    ) as export_png:
        msg = save_viewer_png(
            arr, str(out), tmp_path / "source.sxm",
            "gray", 1.0, 99.0, drs, {}, None, 0, "Z",
            include_provenance=False,
        )

    build_provenance.assert_not_called()
    assert export_png.call_args.kwargs["provenance"] is None
    assert "Saved" in msg


def test_save_viewer_png_uses_explicit_processed_scan_range(tmp_path):
    from probeflow.gui.viewer.png_export import save_viewer_png

    arr = np.zeros((4, 6), dtype=float)
    fake_scan = MagicMock()
    fake_scan.scan_range_m = (60e-9, 40e-9)
    drs = MagicMock()
    drs.resolve.return_value = (0.0, 1.0)
    processed_range = (12e-9, 8e-9)

    with patch(
        "probeflow.core.scan_loader.load_scan", return_value=fake_scan,
    ), patch(
        "probeflow.provenance.export.build_scan_export_provenance",
        return_value="prov",
    ) as build_provenance, patch(
        "probeflow.processing.export_png",
    ) as export_png:
        msg = save_viewer_png(
            arr, str(tmp_path / "viewer.png"), tmp_path / "source.sxm",
            "gray", 1.0, 99.0, drs, {}, None, 0, "Z",
            scan_range_m=processed_range,
        )

    assert "Saved" in msg
    assert export_png.call_args.kwargs["scan_range_m"] == processed_range
    provenance_scan = build_provenance.call_args.args[0]
    assert provenance_scan.scan_range_m == processed_range
    assert fake_scan.scan_range_m == (60e-9, 40e-9)


def test_save_viewer_png_fails_closed_when_requested_provenance_source_is_missing(tmp_path):
    from probeflow.gui.viewer.png_export import save_viewer_png

    drs = MagicMock()
    drs.resolve.return_value = (0.0, 1.0)
    with patch(
        "probeflow.core.scan_loader.load_scan",
        side_effect=FileNotFoundError("gone"),
    ), patch("probeflow.processing.export_png") as export_png:
        msg = save_viewer_png(
            np.zeros((4, 4)), str(tmp_path / "viewer.png"),
            tmp_path / "missing.sxm", "gray", 1.0, 99.0,
            drs, {}, None, 0, "Z", include_provenance=True,
            scan_range_m=(4e-9, 4e-9),
        )

    assert msg.startswith("Export error: source scan could not be loaded")
    export_png.assert_not_called()


# ── save_provenance_json ──────────────────────────────────────────────────────

def test_save_provenance_json_writes_file(tmp_path):
    from probeflow.gui.viewer.processed_export import save_provenance_json
    from probeflow.provenance.records import ProcessingHistory

    history = MagicMock(spec=ProcessingHistory)
    fake_record = MagicMock()
    fake_record.processing_history = history
    fake_record.to_json.return_value = '{"test": true}'

    out = tmp_path / "prov.probeflow.json"

    with patch(
        "probeflow.provenance.records.build_export_record",
        return_value=fake_record,
    ):
        msg, record = save_provenance_json(history, out, display_settings={})

    assert out.exists()
    assert "Saved provenance" in msg
    assert record is fake_record
