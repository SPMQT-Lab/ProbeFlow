"""Display-chain tests for STM vs qPlus AFM metadata in browse (review focus #5).

Extraction (metadata.py) and indexing (ProbeFlowItem) are covered elsewhere;
these tests pin the last hop the original constant-Δf bug lived in: item →
SxmFile (unit conversions) → the card / info-panel display strings. A qPlus
scan must read as a frequency shift, never as a tunnel current; missing
values must read as explicitly unknown, not silently vanish.

Qt-free: probeflow.gui.models and probeflow.gui.browse.panels._setp_display
do not need a QApplication.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from probeflow.core.indexing import ProbeFlowItem
from probeflow.gui.models import SxmFile, _card_meta_str
from probeflow.gui.browse.panels import _setp_display


def _item(**kw) -> ProbeFlowItem:
    base = dict(
        path=Path("/data/scan_0001.dat"),
        display_name="scan_0001",
        source_format="createc_dat",
        item_type="scan",
        shape=(256, 256),
        scan_range=(50e-9, 50e-9),
        bias=-0.25,            # V
        mtime_ns=1,
        size_bytes=2,
    )
    base.update(kw)
    return ProbeFlowItem(**base)


class TestStmDisplay:
    def test_current_setpoint_converts_and_displays_as_pa(self):
        entry = SxmFile.from_index_item(_item(setpoint=100e-12))  # 100 pA in A
        assert entry.current_pa == pytest.approx(100.0)
        assert entry.bias_mv == pytest.approx(-250.0)
        assert "I: 100 pA" in _card_meta_str(entry)
        assert "V: -250 mV" in _card_meta_str(entry)
        assert _setp_display(entry) == "100.0 pA"

    def test_missing_setpoint_reads_as_explicit_unknown(self):
        entry = SxmFile.from_index_item(_item(setpoint=None))
        assert "I: ?" in _card_meta_str(entry)
        assert _setp_display(entry) == "—"


class TestQPlusDisplay:
    def test_delta_f_displays_as_frequency_never_as_current(self):
        """The original bug: a constant-Δf setpoint shown as an absurd pA
        current. The display chain must label it Δf in Hz."""
        entry = SxmFile.from_index_item(_item(
            setpoint=None,
            feedback_setpoint=-2.5,           # negative Δf is typical
            feedback_setpoint_unit="Hz",
            feedback_setpoint_label="Δf setpoint",
        ))
        assert entry.current_pa is None
        card = _card_meta_str(entry)
        assert "Δf: -2.5 Hz" in card
        assert "pA" not in card
        assert _setp_display(entry) == "-2.5 Hz"

    def test_generic_non_current_feedback_uses_its_label(self):
        entry = SxmFile.from_index_item(_item(
            setpoint=None,
            feedback_setpoint=0.7,
            feedback_setpoint_unit=None,
            feedback_setpoint_label="Feedback setpoint",
        ))
        card = _card_meta_str(entry)
        assert "Feedback setpoint: 0.7" in card
        assert "Δf" not in card and "pA" not in card

    def test_qplus_header_end_to_end_never_yields_current(self):
        """From a raw Createc qPlus header (PLL on, stale FBLogIset from the
        old current loop) all the way to the display strings: the stale 700
        must not surface as 7e+14 pA anywhere."""
        from probeflow.core.metadata import (
            _extract_createc_feedback_setpoint,
            _extract_createc_fields,
        )

        hdr = {
            "BiasVolt.[mV]": "-250",
            "FBChannel": "4",
            "PLLOn": "1",
            "SetPoint": "-3.2",      # Δf in Hz
            "FBLogIset": "700",      # stale leftover from the current loop
        }
        bias, setpoint, _comment, _dt = _extract_createc_fields(hdr)
        fb, fb_unit, fb_label = _extract_createc_feedback_setpoint(hdr)
        assert setpoint is None
        entry = SxmFile.from_index_item(_item(
            bias=bias, setpoint=setpoint,
            feedback_setpoint=fb,
            feedback_setpoint_unit=fb_unit,
            feedback_setpoint_label=fb_label,
        ))
        card = _card_meta_str(entry)
        assert "pA" not in card
        assert "Δf: -3.2 Hz" in card
        assert _setp_display(entry) == "-3.2 Hz"


class TestErrorItemDisplay:
    def test_load_error_item_degrades_to_explicit_unknowns(self):
        entry = SxmFile.from_index_item(_item(
            shape=None, bias=None, setpoint=None,
            load_error="truncated header",
        ))
        card = _card_meta_str(entry)
        assert "V: ?" in card and "I: ?" in card
        assert _setp_display(entry) == "—"
