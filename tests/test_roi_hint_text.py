"""Tests for the shared ROI interaction-hint helpers."""

from __future__ import annotations

import pytest

from probeflow.gui.roi_items import roi_hint_text, roi_tooltip_html


@pytest.mark.parametrize(
    "kind, title",
    [
        ("line", "Line ROI"),
        ("rectangle", "Area ROI"),
        ("ellipse", "Area ROI"),
        ("polygon", "Area ROI"),
        ("freehand", "Area ROI"),
        ("multipolygon", "Area ROI"),
        ("point", "Point ROI"),
        ("something_else", "ROI"),
    ],
)
def test_hint_text_titles_by_kind(kind, title):
    assert roi_hint_text(kind).startswith(f"{title}:")


def test_status_hint_is_single_line():
    # Status-bar hint must stay on one line (no rich-text markup / breaks).
    txt = roi_hint_text("line")
    assert "<" not in txt and "\n" not in txt
    assert "select" in txt.lower()


def test_tooltip_is_multirow_rich_text():
    html = roi_tooltip_html("line")
    # Rich text enables Qt tooltip wrapping; <br> forces several short rows.
    assert html.startswith("<qt>") and html.endswith("</qt>")
    assert html.count("<br>") >= 2
    assert "Line ROI" in html


def test_tooltip_wording_reflects_select_then_edit():
    html = roi_tooltip_html("line").lower()
    assert "click to select" in html
    assert "endpoint" in html
