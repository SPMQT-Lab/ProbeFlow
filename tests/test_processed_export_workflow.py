"""Contract tests for the shared processed-image export workflow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from probeflow.workflows import ProcessedExportRequest, write_processed_export


def _scan() -> MagicMock:
    scan = MagicMock()
    scan.n_planes = 1
    scan.planes = [np.zeros((3, 4), dtype=np.float64)]
    scan.plane_names = ["Z"]
    scan.processing_state.steps = []
    return scan


def test_request_copies_mutable_interface_options(tmp_path):
    display = {"colormap": "gray"}
    options = {"clip_low": 1.0}
    request = ProcessedExportRequest(
        _scan(),
        tmp_path / "image.png",
        display_state=display,
        writer_options=options,
    )
    display["colormap"] = "viridis"
    options["clip_low"] = 4.0

    assert request.display_state == {"colormap": "gray"}
    assert request.writer_options == {"clip_low": 1.0}


@pytest.mark.parametrize(
    ("suffix", "method"),
    [
        (".png", "save_png"),
        (".pdf", "save_pdf"),
        (".csv", "save_csv"),
        (".gwy", "save_gwy"),
        (".sxm", "save_sxm"),
    ],
)
def test_workflow_routes_supported_formats(tmp_path, suffix, method):
    scan = _scan()
    request = ProcessedExportRequest(
        scan,
        tmp_path / f"image{suffix}",
        include_provenance=False,
    )

    result = write_processed_export(request)

    getattr(scan, method).assert_called_once()
    assert result.destination == request.destination
    assert result.export_format == suffix.lstrip(".")


def test_workflow_builds_one_provenance_record_for_the_writer(tmp_path):
    scan = _scan()
    request = ProcessedExportRequest(
        scan,
        tmp_path / "image.png",
        display_state={"colormap": "gray"},
        export_kind="viewer_png",
    )

    with patch(
        "probeflow.provenance.export.build_scan_export_provenance",
        return_value="provenance",
    ) as build:
        result = write_processed_export(request)

    assert build.call_count == 1
    assert scan.save_png.call_args.kwargs["provenance"] == "provenance"
    assert result.provenance == "provenance"


def test_workflow_uses_a_prebuilt_provenance_record(tmp_path):
    scan = _scan()
    request = ProcessedExportRequest(
        scan,
        tmp_path / "image.png",
        provenance="existing",
    )

    with patch(
        "probeflow.provenance.export.build_scan_export_provenance",
    ) as build:
        result = write_processed_export(request)

    build.assert_not_called()
    assert scan.save_png.call_args.kwargs["provenance"] == "existing"
    assert result.provenance == "existing"


def test_workflow_rejects_unsupported_formats_before_writing(tmp_path):
    scan = _scan()

    with pytest.raises(ValueError, match="Unsupported processed image format"):
        write_processed_export(
            ProcessedExportRequest(scan, tmp_path / "image.xyz")
        )

    scan.save.assert_not_called()
