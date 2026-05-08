"""Tests for PNG export and provenance sidecar.

test_export_provenance.py covers ExportProvenance serialisation and provenance
builder functions.  This file focuses on the gaps:
  - export_png writes a valid PNG file to disk
  - sidecar .provenance.json is written alongside when provenance is provided
  - sidecar is not written when provenance=None
  - png_display_state includes colormap when specified
  - png_display_state excludes colormap when not specified
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from probeflow.provenance.export import png_display_state


# ── Shared helpers ────────────────────────────────────────────────────────────

def _lut_fn(key: str) -> np.ndarray:
    """Minimal grey LUT: all values map to the same grey level."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, :] = np.arange(256, dtype=np.uint8)[:, None]
    return lut


def _make_arr(shape=(32, 32)):
    rng = np.random.default_rng(42)
    return rng.standard_normal(shape).astype(np.float64)


# ── export_png writes a PNG ───────────────────────────────────────────────────

class TestExportPng:
    def test_writes_png_file(self, tmp_path):
        from probeflow.processing.image import export_png
        out = tmp_path / "out.png"
        export_png(
            _make_arr(),
            out,
            colormap_key="gray",
            clip_low=1.0,
            clip_high=99.0,
            lut_fn=_lut_fn,
            scan_range_m=(1e-7, 1e-7),
        )
        assert out.exists()
        assert out.stat().st_size > 0

    def test_output_is_valid_png(self, tmp_path):
        from PIL import Image
        from probeflow.processing.image import export_png
        out = tmp_path / "out.png"
        export_png(
            _make_arr(),
            out,
            colormap_key="gray",
            clip_low=1.0,
            clip_high=99.0,
            lut_fn=_lut_fn,
            scan_range_m=(1e-7, 1e-7),
        )
        img = Image.open(out)
        assert img.format == "PNG"

    def test_output_dimensions_match_input(self, tmp_path):
        from PIL import Image
        from probeflow.processing.image import export_png
        arr = _make_arr((48, 64))
        out = tmp_path / "out.png"
        export_png(
            arr,
            out,
            colormap_key="gray",
            clip_low=1.0,
            clip_high=99.0,
            lut_fn=_lut_fn,
            scan_range_m=(1e-7, 1e-7),
        )
        img = Image.open(out)
        assert img.size == (64, 48)  # PIL: (width, height) = (cols, rows)

    def test_no_sidecar_when_provenance_is_none(self, tmp_path):
        from probeflow.processing.image import export_png
        out = tmp_path / "out.png"
        export_png(
            _make_arr(),
            out,
            colormap_key="gray",
            clip_low=1.0,
            clip_high=99.0,
            lut_fn=_lut_fn,
            scan_range_m=(1e-7, 1e-7),
            provenance=None,
        )
        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        assert not sidecar.exists()

    def test_sidecar_written_when_provenance_provided(self, tmp_path):
        from probeflow.processing.image import export_png
        from probeflow.provenance.export import ExportProvenance
        out = tmp_path / "out.png"
        prov = ExportProvenance(
            source_file=str(tmp_path / "scan.dat"),
            source_format="dat",
            item_type="scan",
            channel_name="Z fwd",
            channel_index=0,
            array_shape=(32, 32),
            scan_range_m=(1e-7, 1e-7),
            units="m",
            processing_state={"steps": []},
            display_state={"mode": "percentile", "low_pct": 1.0, "high_pct": 99.0,
                           "vmin": None, "vmax": None},
            probeflow_version="0.0.0",
            export_timestamp="2026-01-01T00:00:00Z",
        )
        export_png(
            _make_arr(),
            out,
            colormap_key="gray",
            clip_low=1.0,
            clip_high=99.0,
            lut_fn=_lut_fn,
            scan_range_m=(1e-7, 1e-7),
            provenance=prov,
        )
        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        assert sidecar.exists()

    def test_sidecar_is_valid_json(self, tmp_path):
        from probeflow.processing.image import export_png
        from probeflow.provenance.export import ExportProvenance
        out = tmp_path / "out.png"
        prov = ExportProvenance(
            source_file=str(tmp_path / "scan.dat"),
            source_format="dat",
            item_type="scan",
            channel_name="Z fwd",
            channel_index=0,
            array_shape=(32, 32),
            scan_range_m=(1e-7, 1e-7),
            units="m",
            processing_state={"steps": []},
            display_state={"mode": "percentile", "low_pct": 1.0, "high_pct": 99.0,
                           "vmin": None, "vmax": None},
            probeflow_version="0.0.0",
            export_timestamp="2026-01-01T00:00:00Z",
        )
        export_png(
            _make_arr(),
            out,
            colormap_key="gray",
            clip_low=1.0,
            clip_high=99.0,
            lut_fn=_lut_fn,
            scan_range_m=(1e-7, 1e-7),
            provenance=prov,
        )
        sidecar = out.with_suffix("").with_suffix(".provenance.json")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_scalebar_disabled_does_not_crash(self, tmp_path):
        from probeflow.processing.image import export_png
        out = tmp_path / "no_bar.png"
        export_png(
            _make_arr(),
            out,
            colormap_key="gray",
            clip_low=1.0,
            clip_high=99.0,
            lut_fn=_lut_fn,
            scan_range_m=(0.0, 0.0),  # zero range suppresses scale bar
        )
        assert out.exists()


# ── png_display_state ─────────────────────────────────────────────────────────

class TestPngDisplayState:
    def test_returns_dict_with_mode_key(self):
        d = png_display_state()
        assert "mode" in d

    def test_colormap_included_when_specified(self):
        d = png_display_state(colormap="viridis")
        assert d.get("colormap") == "viridis"

    def test_colormap_absent_when_not_specified(self):
        d = png_display_state()
        assert "colormap" not in d

    def test_scalebar_fields_included_when_specified(self):
        d = png_display_state(add_scalebar=True, scalebar_unit="nm", scalebar_pos="bottom-left")
        assert d["add_scalebar"] is True
        assert d["scalebar_unit"] == "nm"
        assert d["scalebar_pos"] == "bottom-left"

    def test_percentile_mode_preserved_from_display_range_state(self):
        from probeflow.processing.display_state import DisplayRangeState
        drs = DisplayRangeState(mode="percentile", low_pct=2.0, high_pct=98.0)
        d = png_display_state(drs)
        assert d["mode"] == "percentile"
        assert d["low_pct"] == pytest.approx(2.0)
        assert d["high_pct"] == pytest.approx(98.0)

    def test_manual_mode_preserved_from_display_range_state(self):
        from probeflow.processing.display_state import DisplayRangeState
        drs = DisplayRangeState()
        drs.set_manual(1.0, 5.0)
        d = png_display_state(drs)
        assert d["mode"] == "manual"
        assert d["vmin"] == pytest.approx(1.0)
        assert d["vmax"] == pytest.approx(5.0)

    def test_colormap_added_to_display_range_state_dict(self):
        from probeflow.processing.display_state import DisplayRangeState
        drs = DisplayRangeState()
        d = png_display_state(drs, colormap="afm_hot")
        assert d["colormap"] == "afm_hot"
        assert "mode" in d
