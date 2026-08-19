from pathlib import Path

import pytest

from probeflow.gui.metadata_display import metadata_rows

REAL_SM4 = Path(__file__).resolve().parents[1] / "test_data" / "rhk.sm4"


def test_nested_sm4_pages_are_displayed_as_individual_rows():
    rows = metadata_rows({
        "RHK_SM4": True,
        "page_count": 2,
        "pages": [
            {"angle": 79.0, "label": "Topography"},
            {"angle": 80.0, "label": "Current"},
        ],
    })

    assert ("RHK_SM4", "True") in rows
    assert ("page_count", "2") in rows
    assert ("Page 1 — Topography [Left]", "") in rows
    assert ("angle", "79.0") in rows
    assert ("Page 2 — Current", "") in rows
    assert ("label", "Current") in rows


@pytest.mark.skipif(
    not REAL_SM4.exists(),
    reason="real SM4 fixture not present",
)
def test_real_sm4_metadata_does_not_collapse_pages_into_one_value():
    from probeflow.io.readers.rhk_sm4 import read_sm4
    from probeflow.gui.browse.panels import BrowseInfoPanel

    panel = BrowseInfoPanel.__new__(BrowseInfoPanel)
    panel._filter_meta = lambda: None
    panel._populate_metadata_rows(None, read_sm4(REAL_SM4).header)
    rows = panel._meta_rows

    assert ("page_count", "4") in rows
    assert ("Page 1 — Topography [Left]", "") in rows
    assert ("Page 3 — Current [Left]", "") in rows


def test_browse_panel_uses_structured_rows_for_sm4_headers():
    from probeflow.gui.browse.panels import BrowseInfoPanel

    panel = BrowseInfoPanel.__new__(BrowseInfoPanel)
    panel._filter_meta = lambda: None
    panel._populate_metadata_rows(None, {
        "RHK_SM4": True,
        "page_count": 4,
        "pages": [{"label": "Topography", "angle": 73.0}],
    })

    assert ("Page 1 — Topography [Left]", "") in panel._meta_rows
    assert ("angle", "73.0") in panel._meta_rows
