"""Contract tests for provenance payloads and export paths."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from probeflow.io.writers.json import write_json
from probeflow.processing.display_state import DisplayRangeState
from probeflow.processing.state import ProcessingState, ProcessingStep
from probeflow.provenance.export import (
    ExportProvenance,
    background_processing_warnings,
    build_scan_export_provenance,
    png_display_state,
    processing_state_hash,
)


def _make_scan(
    shape=(32, 32),
    n_planes=2,
    source_format="dat",
    scan_range_m=(1e-7, 1e-7),
    source_path=None,
):
    from probeflow.core.scan_model import Scan

    rng = np.random.default_rng(42)
    return Scan(
        planes=[rng.standard_normal(shape) for _ in range(n_planes)],
        plane_names=["Z fwd", "Z bwd"][:n_planes],
        plane_units=["m", "m"][:n_planes],
        plane_synthetic=[False] * n_planes,
        header={},
        scan_range_m=scan_range_m,
        source_path=Path(source_path or "/fake/scan.dat"),
        source_format=source_format,
    )


def _minimal_provenance(**overrides) -> ExportProvenance:
    fields = dict(
        source_file="/data/scan.dat",
        source_format="dat",
        item_type="scan",
        channel_name="Z fwd",
        channel_index=0,
        array_shape=(128, 128),
        scan_range_m=(1.09e-7, 1.09e-7),
        units="m",
        processing_state={"steps": []},
        display_state={
            "mode": "percentile",
            "low_pct": 1.0,
            "high_pct": 99.0,
            "vmin": None,
            "vmax": None,
        },
        probeflow_version="0.0.0b0",
        export_timestamp="2026-04-25T07:15:00Z",
    )
    fields.update(overrides)
    return ExportProvenance(**fields)


def _gray_lut():
    return np.stack([np.arange(256, dtype=np.uint8)] * 3, axis=1)


def test_export_provenance_dict_contract():
    data = _minimal_provenance().to_dict()
    json.dumps(data)

    expected_keys = {
        "source_file",
        "source_format",
        "item_type",
        "channel_name",
        "channel_index",
        "array_shape",
        "scan_range_m",
        "units",
        "processing_state",
        "display_state",
        "probeflow_version",
        "export_timestamp",
        "export_kind",
        "source_id",
        "channel_id",
        "processing_state_hash",
        "artifact_id",
        "warnings",
    }
    assert expected_keys <= set(data)
    assert data["array_shape"] == [128, 128]
    assert isinstance(data["scan_range_m"], list)
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", data["export_timestamp"])

    none_data = _minimal_provenance(
        source_file=None,
        source_format=None,
        channel_name=None,
        channel_index=None,
        array_shape=None,
        scan_range_m=None,
        units=None,
        probeflow_version=None,
    ).to_dict()
    assert none_data["source_file"] is None
    assert none_data["array_shape"] is None
    assert none_data["scan_range_m"] is None


def test_scan_export_provenance_contract(tmp_path):
    scan = _make_scan(shape=(64, 48), source_format="dat", scan_range_m=(2e-7, 3e-7), source_path="/data/test.dat")
    ps = ProcessingState(steps=[ProcessingStep("align_rows", {"method": "median"})])
    drs = DisplayRangeState()
    drs.set_manual(1.0, 5.0)

    prov = ExportProvenance.from_scan_export(
        scan,
        channel_index=0,
        channel_name="Custom",
        processing_state=ps,
        display_state=drs,
        item_type="thumbnail",
    )
    data = prov.to_dict()

    assert prov.source_file == "test.dat"
    assert prov.source_format == "dat"
    assert prov.channel_index == 0
    assert prov.channel_name == "Custom"
    assert prov.array_shape == (64, 48)
    assert prov.scan_range_m == pytest.approx((2e-7, 3e-7))
    assert prov.units == "m"
    assert prov.item_type == "thumbnail"
    assert prov.processing_state == ps.to_dict()
    assert prov.display_state["mode"] == "manual"
    assert prov.probeflow_version is None or isinstance(prov.probeflow_version, str)
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", prov.export_timestamp)
    json.dumps(data)

    no_path_scan = scan.__class__(
        planes=scan.planes,
        plane_names=scan.plane_names,
        plane_units=scan.plane_units,
        plane_synthetic=scan.plane_synthetic,
        header={},
        scan_range_m=scan.scan_range_m,
        source_path=None,
        source_format=scan.source_format,
    )
    assert ExportProvenance.from_scan_export(no_path_scan, channel_index=0).source_file is None

    out = tmp_path / "prepared.png"
    built = build_scan_export_provenance(
        scan,
        channel_index=0,
        processing_state=ps,
        display_state=drs,
        export_kind="prepared_png",
        output_path=out,
    ).to_dict()
    assert built["export_kind"] == "prepared_png"
    assert built["output_path"] == out.name
    assert built["source_id"]
    assert built["channel_id"]
    assert built["artifact_id"]
    assert built["processing_state_hash"] == processing_state_hash(ps.to_dict())


def test_build_scan_export_provenance_embeds_masks(tmp_path):
    from probeflow.core.mask import ImageMask, MaskSet

    scan = _make_scan()
    ms = MaskSet(image_id="img")
    ms.add(ImageMask.new(np.ones((8, 8), bool), name="m1"))
    built = build_scan_export_provenance(
        scan,
        channel_index=0,
        export_kind="viewer_png",
        output_path=tmp_path / "out.png",
        mask_set=ms,
    ).to_dict()
    assert built["masks"]["image_id"] == "img"
    assert built["export_record"]["masks"]["image_id"] == "img"


def test_provenance_warning_hash_and_png_display_contract():
    state_a = {"steps": [{"op": "plane_bg", "params": {"order": 1, "fit_rect": (1, 2, 3, 4)}}]}
    state_b = {"steps": [{"params": {"fit_rect": (1, 2, 3, 4), "order": 1}, "op": "plane_bg"}]}
    assert processing_state_hash(state_a) == processing_state_hash(state_b)

    assert background_processing_warnings({"steps": [{"op": "stm_line_bg", "params": {}}]}) == ()
    assert background_processing_warnings({"steps": [{"op": "stm_background", "params": {}}]}) == ()
    warnings = background_processing_warnings({"steps": [{"op": "smooth", "params": {"sigma_px": 1.0}}]})
    assert len(warnings) == 1
    assert "background" in warnings[0]

    display = png_display_state(
        clip_low=2.0,
        clip_high=98.0,
        colormap="plasma",
        add_scalebar=False,
        scalebar_unit="nm",
        scalebar_pos="bottom-left",
    )
    assert display["mode"] == "percentile"
    assert display["low_pct"] == 2.0
    assert display["high_pct"] == 98.0
    assert display["colormap"] == "plasma"
    assert display["add_scalebar"] is False
    assert display["scalebar_pos"] == "bottom-left"


def test_json_export_provenance_contract(tmp_path):
    scan = _make_scan()
    ps = ProcessingState(steps=[ProcessingStep("smooth", {"sigma_px": 2.0})])
    drs = DisplayRangeState()
    drs.set_manual(-1e-9, 2e-9)
    prov = ExportProvenance.from_scan_export(scan, channel_index=0, processing_state=ps, display_state=drs)

    out = tmp_path / "out.json"
    write_json(out, [], kind="particles", provenance=prov)
    data = json.loads(out.read_text())

    assert "export_provenance" in data
    assert data["processing_state"] == ps.to_dict()
    assert data["display_state"]["mode"] == "manual"
    assert data["display_state"]["vmin"] == pytest.approx(-1e-9)
    assert "items" in data

    no_prov = tmp_path / "no_prov.json"
    write_json(no_prov, [], kind="particles")
    no_prov_data = json.loads(no_prov.read_text())
    assert "meta" in no_prov_data
    assert "export_provenance" not in no_prov_data

    existing = tmp_path / "existing.json"
    existing.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Output path already exists"):
        write_json(existing, [], kind="particles")
    assert existing.read_text(encoding="utf-8") == "sentinel"

    write_json(existing, [], kind="particles", overwrite=True)
    assert json.loads(existing.read_text(encoding="utf-8"))["meta"]["kind"] == "particles"


def test_png_sidecar_contract(tmp_path):
    from probeflow.processing import export_png

    arr = np.random.default_rng(0).standard_normal((32, 32))
    prov = ExportProvenance.from_scan_export(_make_scan(), channel_index=0, display_state=DisplayRangeState())

    no_prov = tmp_path / "no_prov.png"
    export_png(
        arr,
        no_prov,
        "gray",
        1.0,
        99.0,
        lut_fn=lambda _: _gray_lut(),
        scan_range_m=(0.0, 0.0),
        add_scalebar=False,
    )
    assert no_prov.exists()
    assert not no_prov.with_suffix("").with_suffix(".provenance.json").exists()

    with_prov = tmp_path / "with_prov.png"
    export_png(
        arr,
        with_prov,
        "gray",
        1.0,
        99.0,
        lut_fn=lambda _: _gray_lut(),
        scan_range_m=(0.0, 0.0),
        add_scalebar=False,
        provenance=prov,
    )
    sidecar = with_prov.with_suffix("").with_suffix(".provenance.json")
    data = json.loads(sidecar.read_text())
    assert with_prov.exists() and with_prov.stat().st_size > 0
    assert {"source_file", "source_format", "item_type", "processing_state", "display_state", "export_timestamp"} <= set(data)

    blocked = tmp_path / "blocked.png"
    blocked.write_bytes(b"artifact sentinel")
    blocked.with_suffix(".probeflow.json").write_text("sidecar sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Provenance sidecar"):
        export_png(
            arr,
            blocked,
            "gray",
            1.0,
            99.0,
            lut_fn=lambda _: _gray_lut(),
            scan_range_m=(0.0, 0.0),
            add_scalebar=False,
            provenance=prov,
        )
    assert blocked.read_bytes() == b"artifact sentinel"


def test_png_pixels_unchanged_by_provenance(tmp_path):
    from PIL import Image
    from probeflow.processing import export_png

    arr = np.random.default_rng(999).standard_normal((64, 64))
    prov = _minimal_provenance(array_shape=(64, 64), export_timestamp="2026-04-25T00:00:00Z")
    kwargs = dict(lut_fn=lambda _: _gray_lut(), scan_range_m=(0.0, 0.0), add_scalebar=False)
    out_no_prov = tmp_path / "no_prov.png"
    out_with_prov = tmp_path / "with_prov.png"

    export_png(arr, out_no_prov, "gray", 1.0, 99.0, **kwargs)
    export_png(arr, out_with_prov, "gray", 1.0, 99.0, **kwargs, provenance=prov)

    np.testing.assert_array_equal(np.asarray(Image.open(out_no_prov)), np.asarray(Image.open(out_with_prov)))


def test_write_png_standard_provenance_contract(tmp_path):
    from probeflow.io.writers.png import write_png

    scan = _make_scan()
    explicit = tmp_path / "explicit.png"
    write_png(scan, explicit, plane_idx=0, provenance=ExportProvenance.from_scan_export(scan, channel_index=0))
    assert explicit.exists()
    assert explicit.with_suffix("").with_suffix(".provenance.json").exists()

    scan.processing_history = [{"op": "align_rows", "params": {"method": "median"}, "timestamp": "T"}]
    auto = tmp_path / "auto.png"
    write_png(scan, auto, plane_idx=0, colormap="plasma", clip_low=2.0, clip_high=98.0, add_scalebar=False)

    data = json.loads(auto.with_suffix("").with_suffix(".provenance.json").read_text())
    assert data["export_kind"] == "png"
    assert data["processing_state"]["steps"][0]["op"] == "align_rows"
    assert data["display_state"]["colormap"] == "plasma"
    assert data["display_state"]["add_scalebar"] is False
    assert data["artifact_id"]


def test_prepared_png_warning_contract(tmp_path):
    from probeflow.provenance.prepared_export import write_prepared_png

    scan = _make_scan()
    raw_out = tmp_path / "aisurf_raw.png"
    write_prepared_png(
        scan,
        raw_out,
        processing_state=ProcessingState(steps=[ProcessingStep("smooth", {"sigma_px": 1.0})]),
        add_scalebar=False,
    )
    raw_data = json.loads(raw_out.with_suffix("").with_suffix(".provenance.json").read_text())
    assert raw_data["export_kind"] == "prepared_png"
    assert raw_data["warnings"]
    assert raw_data["processing_state"]["steps"][0]["op"] == "smooth"

    bg_out = tmp_path / "aisurf_bg.png"
    write_prepared_png(
        scan,
        bg_out,
        processing_state=ProcessingState(steps=[ProcessingStep("plane_bg", {"order": 1})]),
        add_scalebar=False,
    )
    bg_data = json.loads(bg_out.with_suffix("").with_suffix(".provenance.json").read_text())
    assert bg_data["warnings"] == []
    assert bg_data["processing_state_hash"]


def test_cli_png_sidecar_contract(tmp_path):
    from probeflow.cli import main as cli_main

    src = Path(__file__).resolve().parents[1] / "test_data" / "nanonis.sxm"
    out = tmp_path / "moire.png"
    assert cli_main(["sxm2png", str(src), "-o", str(out), "--no-scalebar"]) == 0

    data = json.loads(out.with_suffix("").with_suffix(".provenance.json").read_text())
    assert data["export_kind"] == "cli_sxm2png"
    assert data["processing_state"]["steps"] == []
    assert data["display_state"]["mode"] == "percentile"
    assert data["display_state"]["colormap"] == "gray"
    assert data["display_state"]["add_scalebar"] is False
    assert data["source_id"]


def test_cli_pipeline_and_prepare_png_provenance_contract(tmp_path):
    from probeflow.cli import main as cli_main

    src = Path(__file__).resolve().parents[1] / "test_data" / "nanonis.sxm"
    pipeline_out = tmp_path / "moire_processed.png"
    rc = cli_main([
        "pipeline",
        str(src),
        "--steps",
        "align-rows:median",
        "plane-bg:1",
        "--png",
        "-o",
        str(pipeline_out),
        "--no-scalebar",
    ])
    assert rc == 0
    pipeline_data = json.loads(pipeline_out.with_suffix("").with_suffix(".provenance.json").read_text())
    assert pipeline_data["export_kind"] == "cli_png"
    assert [step["op"] for step in pipeline_data["processing_state"]["steps"]] == ["align_rows", "plane_bg"]
    assert pipeline_data["processing_state_hash"]

    raw_prepare = tmp_path / "aisurf_raw.png"
    assert cli_main(["prepare-png", str(src), str(raw_prepare)]) == 0
    raw_data = json.loads(raw_prepare.with_suffix("").with_suffix(".provenance.json").read_text())
    assert raw_data["export_kind"] == "prepared_png"
    assert raw_data["processing_state"]["steps"] == []
    assert raw_data["warnings"]
    assert raw_data["display_state"]["add_scalebar"] is False

    bg_prepare = tmp_path / "aisurf_prepared.png"
    assert cli_main([
        "prepare-png",
        str(src),
        str(bg_prepare),
        "--steps",
        "align-rows:median",
        "plane-bg:1",
        "--colormap",
        "plasma",
    ]) == 0
    bg_data = json.loads(bg_prepare.with_suffix("").with_suffix(".provenance.json").read_text())
    assert [step["op"] for step in bg_data["processing_state"]["steps"]] == ["align_rows", "plane_bg"]
    assert bg_data["warnings"] == []
    assert bg_data["display_state"]["colormap"] == "plasma"
