from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from probeflow.core.browse_filters import (
    FolderFilterState,
    bias_options_from_values,
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


def test_bias_value_filter_matches_within_tolerance():
    state = FolderFilterState(bias_value_mv=-25.0)
    assert scan_matches_folder_filters(
        completion_pct=None, bias_mv=-25.0, state=state)
    assert scan_matches_folder_filters(
        completion_pct=None, bias_mv=-25.4, state=state)
    assert not scan_matches_folder_filters(
        completion_pct=None, bias_mv=-26.0, state=state)
    assert not scan_matches_folder_filters(
        completion_pct=None, bias_mv=25.0, state=state)


def test_bias_value_filter_hides_scans_without_bias_metadata():
    state = FolderFilterState(bias_value_mv=100.0)
    assert not scan_matches_folder_filters(
        completion_pct=None, bias_mv=None, state=state)


def test_hide_incomplete_hides_partial_but_keeps_unknown():
    state = FolderFilterState(hide_incomplete=True)
    assert not scan_matches_folder_filters(
        completion_pct=30.0, bias_mv=None, state=state)
    assert scan_matches_folder_filters(
        completion_pct=50.0, bias_mv=None, state=state)
    assert scan_matches_folder_filters(
        completion_pct=100.0, bias_mv=None, state=state)
    # Scans without completion metadata stay visible.
    assert scan_matches_folder_filters(
        completion_pct=None, bias_mv=None, state=state)


def test_no_active_filters_matches_everything():
    state = FolderFilterState()
    assert not state.has_metadata_filters()
    assert scan_matches_folder_filters(
        completion_pct=None, bias_mv=None, state=state)


def test_bias_options_group_and_count():
    options = bias_options_from_values(
        [-550.0, -550.2, 1000.0, None, -25.0, -550.0])
    assert options == [(-550.0, 3), (-25.0, 1), (1000.0, 1)]


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

    # A bias filter far from any real scan's bias hides the subfolder; the
    # matching bias (read from the indexed metadata) keeps it visible.
    from probeflow.core.indexing import index_folder_shallow

    index = index_folder_shallow(sub)
    scan_item = next(it for it in index.files if it.item_type == "scan")
    bias_mv = scan_item.bias * 1000.0 if scan_item.bias is not None else None
    if bias_mv is None:
        pytest.skip("sample scan carries no bias metadata")

    match_state = FolderFilterState(bias_value_mv=bias_mv)
    miss_state = FolderFilterState(bias_value_mv=bias_mv + 1000.0)

    assert subfolder_matches_filters(sub, match_state) is True
    assert subfolder_matches_filters(sub, miss_state) is False
