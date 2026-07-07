from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from probeflow.cli.parser import _build_parser
from probeflow.core.mask import ImageMask
from probeflow.dataset_builder.annotations import save_mask_annotation, save_review_annotation
from probeflow.dataset_builder.export import export_dataset
from probeflow.dataset_builder.loading import load_scan_plane
from probeflow.dataset_builder.models import DatasetExportSpec, DatasetTaskConfig
from probeflow.dataset_builder.cache import (
    DatasetBuilderCache,
    QuickSegPreprocKey,
    QuickSegWatershedKey,
    quickseg_params_fingerprint,
    quickseg_seed_fingerprint,
    sample_cache_key,
)
from probeflow.dataset_builder.quickseg import (
    ADVANCED_PARAMETERS,
    QuickSegParams,
    QuickSegSeed,
    QuickSegPrepared,
    QuickSegState,
    detect_horizontal_artifact_mask,
    quickseg_gradient_sigmas,
    load_quickseg_state,
    prepare_quickseg_inputs,
    save_quickseg_state,
    watershed_labels,
)
from probeflow.dataset_builder.painting import paint_mask
from probeflow.dataset_builder.proposals import generate_proposal
from probeflow.dataset_builder.queue import build_queue, build_queue_from_indexed_items
from probeflow.dataset_builder.sidecar_state import load_review_record
from probeflow.core.indexing import ProbeFlowItem


def _sample_scan(tmp_path: Path) -> Path:
    src = Path(__file__).resolve().parents[1] / "test_data" / "sample_input" / "sxm" / "A250320.191933.sxm"
    dst = tmp_path / src.name
    shutil.copyfile(src, dst)
    return dst


def test_dataset_builder_review_state_preserves_existing_probeflow_sidecar(tmp_path):
    scan_path = _sample_scan(tmp_path)
    sidecar = scan_path.parent / f"{scan_path.stem}.probeflow.json"
    sidecar.write_text(
        json.dumps({"record_type": "existing_probe_flow_payload", "keep": True}),
        encoding="utf-8",
    )
    _scan, arr, _px_x, _px_y = load_scan_plane(scan_path, 0)
    mask_data = np.zeros(arr.shape, dtype=bool)
    mask_data[0, 0] = True
    config = DatasetTaskConfig(plane_index=0)

    save_mask_annotation(
        scan_path,
        ImageMask.new(mask_data, method="manual", name=config.label_name),
        config=config,
        status="accepted",
    )

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["keep"] is True
    assert payload["dataset_builder"]["records"]["step_edge_mask:plane0:step_edge"]["status"] == "accepted"

    queue = build_queue(tmp_path, plane_index=0)
    assert len(queue) == 1
    assert queue[0].status == "accepted"
    assert queue[0].has_mask_sidecar is True


def test_dataset_builder_review_status_can_be_saved_without_mask(tmp_path):
    scan_path = _sample_scan(tmp_path)
    config = DatasetTaskConfig(plane_index=0)

    save_review_annotation(scan_path, config=config, status="rejected")

    queue = build_queue(tmp_path, plane_index=0)
    assert len(queue) == 1
    assert queue[0].status == "rejected"
    assert queue[0].has_mask_sidecar is False


def test_dataset_builder_task_defaults_include_quickseg_labels():
    config = DatasetTaskConfig(task="terrace_segmentation")

    assert config.label_name == "quickseg_terraces"
    assert config.label_type == "instances"
    assert config.proposal_method == "quickseg"


def test_dataset_builder_quickseg_state_round_trip(tmp_path):
    scan_path = _sample_scan(tmp_path)
    _scan, arr, px_x, px_y = load_scan_plane(scan_path, 0)
    params = QuickSegParams()
    prepared = prepare_quickseg_inputs(arr, params, pixel_size_x_m=px_x, pixel_size_y_m=px_y)
    seeds = [
        QuickSegSeed(x=10, y=10, terrace_label_id=1, order=1),
        QuickSegSeed(x=30, y=30, terrace_label_id=2, order=2),
    ]
    result = watershed_labels(prepared, seeds, params)
    state = QuickSegState(seeds=seeds, current_label=2, next_order=3, params=params, result=result)
    config = DatasetTaskConfig(task="terrace_segmentation", plane_index=0)

    result_path, state_path = save_quickseg_state(
        scan_path,
        state,
        config=config,
        status="accepted",
    )

    assert state_path.exists()
    assert result_path is not None and result_path.exists()
    record = load_review_record(scan_path, task=config.task, plane_index=0, label_name=config.label_name)
    assert record is not None
    assert record.status == "accepted"
    assert record.task_data["quickseg"]["seeds"][0]["terrace_label_id"] == 1

    loaded_state, loaded_record, loaded_result_path = load_quickseg_state(scan_path, config=config)
    assert loaded_record is not None
    assert loaded_result_path.exists()
    assert loaded_state.result is not None
    assert loaded_state.result.shape == arr.shape
    assert loaded_state.seeds[1].terrace_label_id == 2


def test_dataset_builder_quickseg_watershed_uses_prepared_gradient(monkeypatch):
    raw = np.zeros((8, 8), dtype=float)
    gradient = np.zeros((8, 8), dtype=float)
    elevation = np.arange(64, dtype=float).reshape(8, 8)
    prepared = QuickSegPrepared(
        raw=raw,
        corrected=raw + 1.0,
        equalized=raw + 2.0,
        denoised=raw + 3.0,
        gaussian=raw + 4.0,
        gradient=gradient,
        watershed_elevation=elevation,
    )
    params = QuickSegParams()
    seeds = [
        QuickSegSeed(x=1, y=1, terrace_label_id=1, order=1),
        QuickSegSeed(x=6, y=6, terrace_label_id=2, order=2),
    ]
    seen = {}

    def fake_watershed(image, *, markers, connectivity, compactness, watershed_line):
        seen["image"] = image.copy()
        seen["markers"] = markers.copy()
        seen["connectivity"] = connectivity
        seen["compactness"] = compactness
        seen["watershed_line"] = watershed_line
        return markers.copy()

    monkeypatch.setattr("probeflow.dataset_builder.quickseg.watershed", fake_watershed)

    labels = watershed_labels(prepared, seeds, params)

    assert np.array_equal(seen["image"], elevation)
    assert np.array_equal(labels, seen["markers"])
    assert seen["markers"][1, 1] == 1
    assert seen["markers"][6, 6] == 2


def test_dataset_builder_quickseg_pipeline_returns_display_stages():
    y, x = np.mgrid[:48, :64]
    arr = 0.02 * x + 0.01 * y
    arr[:, 30:] += 1.0
    params = QuickSegParams()

    prepared = prepare_quickseg_inputs(arr, params)

    for stage in (
        prepared.flat_display,
        prepared.denoised,
        prepared.anisotropic_blur,
        prepared.gradient_contrast,
        prepared.connected_edge_mask,
        prepared.watershed_elevation,
    ):
        assert stage is not None
        assert stage.shape == arr.shape
    assert np.nanmax(prepared.watershed_elevation) <= 1.0
    assert np.nanmin(prepared.watershed_elevation) >= 0.0


def test_dataset_builder_quickseg_eight_knobs_affect_fingerprint():
    base = quickseg_params_fingerprint(QuickSegParams())
    variants = [
        QuickSegParams(denoise_strength=0.08),
        QuickSegParams(smooth_along_scan=2.0),
        QuickSegParams(smooth_across_scan=1.0),
        QuickSegParams(edge_scale="fine"),
        QuickSegParams(edge_sensitivity=90.0),
        QuickSegParams(min_edge_size=80),
        QuickSegParams(edge_connect_strength=0.7),
        QuickSegParams(barrier_strength=0.35),
        QuickSegParams(horizontal_defect_suppression=0.8),
    ]

    assert all(quickseg_params_fingerprint(params) != base for params in variants)
    assert ("advanced_parameters_version", ADVANCED_PARAMETERS["version"]) in base


def test_dataset_builder_quickseg_edge_scale_presets():
    assert quickseg_gradient_sigmas(QuickSegParams(edge_scale="fine")) == (0.4, 0.8, 1.4)
    assert quickseg_gradient_sigmas(QuickSegParams(edge_scale="balanced")) == (0.6, 1.2, 2.4)
    assert quickseg_gradient_sigmas(QuickSegParams(edge_scale="broad")) == (1.0, 2.0, 3.5)


def test_dataset_builder_quickseg_detects_full_width_horizontal_artifact():
    mask = np.zeros((100, 100), dtype=bool)
    mask[48:50, 4:96] = True

    artifact = detect_horizontal_artifact_mask(mask)

    assert artifact[48:50, 4:96].mean() > 0.9


def test_dataset_builder_quickseg_detects_horizontal_artifact_attached_to_other_edges():
    mask = np.zeros((100, 100), dtype=bool)
    mask[48:50, 2:98] = True
    mask[20:80, 50:52] = True
    mask[25:55, 25:55] |= np.eye(30, dtype=bool)

    artifact = detect_horizontal_artifact_mask(mask)

    assert artifact[48:50, 2:98].mean() > 0.9


def test_dataset_builder_quickseg_does_not_mark_short_horizontal_edge():
    mask = np.zeros((100, 100), dtype=bool)
    mask[48:50, 10:40] = True

    artifact = detect_horizontal_artifact_mask(mask)

    assert artifact.sum() == 0


def test_dataset_builder_quickseg_horizontal_suppression_lowers_elevation(monkeypatch):
    arr = np.zeros((40, 40), dtype=float)
    arr[:, 20:] = 1.0
    artifact = np.zeros_like(arr, dtype=bool)
    artifact[18:20, :] = True

    monkeypatch.setattr(
        "probeflow.dataset_builder.quickseg.detect_horizontal_artifact_mask",
        lambda _edge_mask: artifact,
    )

    off = prepare_quickseg_inputs(arr, QuickSegParams(horizontal_defect_suppression=0.0))
    on = prepare_quickseg_inputs(arr, QuickSegParams(horizontal_defect_suppression=1.0))

    assert np.allclose(off.watershed_elevation, off.watershed_elevation_unsuppressed)
    assert on.horizontal_artifact_mask[18:20, :].mean() > 0.9
    assert float(np.mean(on.watershed_elevation[18:20, :])) < float(
        np.mean(on.watershed_elevation_unsuppressed[18:20, :])
    )
    assert float(np.max(on.watershed_elevation[18:20, :])) < 0.5


def test_dataset_builder_cache_separates_preproc_and_watershed_layers(tmp_path):
    scan_path = _sample_scan(tmp_path)
    key = sample_cache_key(scan_path, 0)
    params = QuickSegParams()
    preproc_key = QuickSegPreprocKey(key, quickseg_params_fingerprint(params))
    seeds_a = [
        QuickSegSeed(x=1, y=2, terrace_label_id=1, order=1),
        QuickSegSeed(x=3, y=4, terrace_label_id=2, order=2),
    ]
    seeds_b = [
        QuickSegSeed(x=5, y=6, terrace_label_id=1, order=1),
    ]
    ws_key_a = QuickSegWatershedKey(preproc_key, quickseg_seed_fingerprint(seeds_a))
    ws_key_b = QuickSegWatershedKey(preproc_key, quickseg_seed_fingerprint(seeds_b))
    cache = DatasetBuilderCache()
    prepared = object()
    labels_a = np.ones((4, 4), dtype=np.int32)

    cache.put_preproc(preproc_key, prepared)  # type: ignore[arg-type]
    cache.put_watershed(ws_key_a, labels_a)

    assert cache.get_preproc(preproc_key) is prepared
    assert np.array_equal(cache.get_watershed(ws_key_a), labels_a)
    assert cache.get_watershed(ws_key_b) is None


def test_dataset_builder_build_queue_from_indexed_items_skips_spectra(tmp_path):
    scan_path = _sample_scan(tmp_path)
    spectrum_path = tmp_path / "spec.vert"
    spectrum_path.write_text("dummy", encoding="utf-8")
    items = [
        ProbeFlowItem(
            path=scan_path,
            display_name=scan_path.name,
            source_format="createc_dat",
            item_type="scan",
        ),
        ProbeFlowItem(
            path=spectrum_path,
            display_name=spectrum_path.name,
            source_format="createc_vert",
            item_type="spectrum",
        ),
    ]

    queue = build_queue_from_indexed_items(items, plane_index=0)

    assert len(queue) == 1
    assert queue[0].source_path == scan_path.resolve()
    assert queue[0].display_id.endswith("_plane0")


def test_dataset_builder_paint_mask_brush_and_eraser():
    mask = np.zeros((9, 9), dtype=bool)

    brushed, changed = paint_mask(mask, x=4, y=4, radius=2, value=True)
    assert changed is True
    assert bool(brushed[4, 4]) is True
    assert brushed.sum() > 1
    assert mask.sum() == 0

    erased, changed = paint_mask(brushed, x=4, y=4, radius=1, value=False)
    assert changed is True
    assert bool(erased[4, 4]) is False
    assert erased.sum() < brushed.sum()


def test_dataset_builder_step_edge_proposal_matches_plane_shape(tmp_path):
    scan_path = _sample_scan(tmp_path)
    scan, arr, px_x, px_y = load_scan_plane(scan_path, 0)
    config = DatasetTaskConfig(
        plane_index=0,
        proposal_params={"molecule_size_nm": 1.0, "margin_nm": 0.1},
    )

    proposal = generate_proposal(
        arr,
        px_x_m=px_x,
        px_y_m=px_y,
        config=config,
        source_channel=scan.plane_names[0],
    )

    assert proposal.label_type == "mask"
    assert proposal.mask is not None
    assert proposal.mask.shape == arr.shape
    assert proposal.parameters["source_channel"] == scan.plane_names[0]


def test_dataset_builder_export_writes_manifest_and_artifacts(tmp_path):
    scan_path = _sample_scan(tmp_path)
    _scan, arr, _px_x, _px_y = load_scan_plane(scan_path, 0)
    mask_data = np.zeros(arr.shape, dtype=bool)
    mask_data[:2, :3] = True
    config = DatasetTaskConfig(plane_index=0, annotator="tester")
    save_mask_annotation(
        scan_path,
        ImageMask.new(mask_data, method="manual", name=config.label_name),
        config=config,
        status="accepted",
        notes="unit test",
    )

    out = tmp_path / "dataset"
    summary = export_dataset(
        DatasetExportSpec(
            source=tmp_path,
            output_dir=out,
            plane_index=0,
            overwrite=False,
        )
    )

    assert summary["n_exported"] == 1
    rows = list(csv.DictReader((out / "manifest.csv").open("r", encoding="utf-8", newline="")))
    assert len(rows) == 1
    row = rows[0]
    assert row["review_status"] == "accepted"
    assert row["annotator"] == "tester"
    assert (out / row["array_path"]).exists()
    assert (out / row["preview_path"]).exists()
    assert (out / row["provenance_path"]).exists()
    mask_paths = json.loads(row["mask_paths_json"])
    assert (out / mask_paths["step_edge"]).exists()
    assert (out / mask_paths["step_edge_png"]).exists()


def test_dataset_cli_parser_exposes_dataset_subcommands():
    parser = _build_parser()

    args = parser.parse_args([
        "dataset",
        "export",
        "scans",
        "out",
        "--status",
        "accepted",
        "uncertain",
    ])

    assert args.command == "dataset"
    assert args.dataset_command == "export"
    assert args.status == ["accepted", "uncertain"]
