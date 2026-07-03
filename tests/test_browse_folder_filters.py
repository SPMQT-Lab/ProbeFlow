from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from probeflow.core.browse_filters import (
    FolderFilterState,
    completion_pct_from_visible_range,
    createc_visible_height_m,
    scan_matches_folder_filters,
)
from probeflow.core.indexing import ProbeFlowItem, subfolder_matches_filters
from probeflow.gui.models import SxmFile


def test_completion_pct_square_full_scan():
    assert completion_pct_from_visible_range(100e-9, 100e-9) == pytest.approx(100.0)


def test_completion_pct_rectangular_full_scan():
    assert completion_pct_from_visible_range(100e-9, 50e-9) == pytest.approx(50.0)


def test_createc_partial_height_completion():
    visible_h = createc_visible_height_m(100e-9, 512, 256)
    assert visible_h == pytest.approx(50e-9)
    assert completion_pct_from_visible_range(100e-9, visible_h) == pytest.approx(50.0)


def test_createc_first_column_strip_does_not_reduce_completion():
    # Width stays at the physical header range even though one stored column was removed.
    assert completion_pct_from_visible_range(100e-9, 100e-9) == pytest.approx(100.0)


def test_missing_metadata_excludes_scan_when_filter_active():
    state = FolderFilterState(size_enabled=True, max_width_nm=100.0, max_height_nm=100.0)
    assert not scan_matches_folder_filters(
        width_nm=None,
        height_nm=80.0,
        completion_pct=80.0,
        bias_mv=50.0,
        state=state,
    )


def test_size_and_bias_boundaries_are_inclusive():
    state = FolderFilterState(
        size_enabled=True,
        min_width_nm=25.0,
        max_width_nm=100.0,
        min_height_nm=20.0,
        max_height_nm=80.0,
        bias_enabled=True,
        min_bias_mv=-250.0,
        max_bias_mv=250.0,
    )
    assert scan_matches_folder_filters(
        width_nm=100.0,
        height_nm=80.0,
        completion_pct=None,
        bias_mv=-250.0,
        state=state,
    )
    assert scan_matches_folder_filters(
        width_nm=100.0,
        height_nm=80.0,
        completion_pct=None,
        bias_mv=250.0,
        state=state,
    )
    assert not scan_matches_folder_filters(
        width_nm=24.9,
        height_nm=80.0,
        completion_pct=None,
        bias_mv=0.0,
        state=state,
    )
    assert not scan_matches_folder_filters(
        width_nm=100.0,
        height_nm=19.9,
        completion_pct=None,
        bias_mv=0.0,
        state=state,
    )


def test_sxmfile_from_index_item_preserves_browse_filter_fields():
    item = ProbeFlowItem(
        path=Path("/tmp/scan.sxm"),
        display_name="scan",
        source_format="nanonis_sxm",
        item_type="scan",
        shape=(128, 256),
        scan_range=(120e-9, 120e-9),
        visible_scan_range=(120e-9, 80e-9),
        completion_pct=66.6666667,
        bias=0.125,
        metadata={},
    )
    sxm = SxmFile.from_index_item(item)
    assert sxm.scan_nm == pytest.approx(120.0)
    assert sxm.scan_width_nm == pytest.approx(120.0)
    assert sxm.scan_height_nm == pytest.approx(80.0)
    assert sxm.completion_pct == pytest.approx(66.6666667)
    assert sxm.bias_mv == pytest.approx(125.0)


def test_subfolder_matches_filters_uses_indexed_scan_metadata(tmp_path):
    src_root = Path(__file__).resolve().parents[1] / "test_data"
    src = next(src_root.glob("*.sxm"), None) or next(src_root.glob("*.dat"))
    sub = tmp_path / "sample_input"
    sub.mkdir()
    shutil.copy(src, sub / src.name)

    match_state = FolderFilterState(size_enabled=True, max_width_nm=1000.0, max_height_nm=1000.0)
    miss_state = FolderFilterState(size_enabled=True, max_width_nm=0.1, max_height_nm=0.1)

    assert subfolder_matches_filters(sub, match_state) is True
    assert subfolder_matches_filters(sub, miss_state) is False
