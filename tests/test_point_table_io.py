"""Tests for the point-table importer (CSV / JSON) for point statistics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from probeflow.measurements.feature_sets import FeatureSet, FeatureSetStore
from probeflow.measurements.point_table_io import (
    default_image_shape,
    default_scan_range_m,
    load_point_table,
    sniff_point_table,
)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def test_feature_finder_csv_roundtrip(tmp_path):
    csv_text = (
        "index,x_px,y_px,x_nm,y_nm,value\n"
        "0,10.0,20.0,5.0,10.0,1.23\n"
        "1,30.0,40.0,15.0,20.0,4.56\n"
    )
    p = _write(tmp_path / "ff.csv", csv_text)
    preview = sniff_point_table(p)
    assert preview.kind == "probeflow_csv"
    assert preview.units == "nm"
    assert preview.has_id_column is True
    assert preview.n_points == 2

    (fs,) = load_point_table(p, scan_range_m=(100e-9, 100e-9), image_shape=(100, 100))
    assert fs.point_count == 2
    # nm columns -> metres
    np.testing.assert_allclose(fs.points_m[0], [5e-9, 10e-9])
    np.testing.assert_allclose(fs.points_m[1], [15e-9, 20e-9])


def test_measurements_csv_units_from_unit_column(tmp_path):
    csv_text = (
        "point_id,x_px,y_px,x_phys,y_phys,z_value,x_unit,y_unit,z_unit,channel,source_label,roi_id\n"
        "p0,1,2,3.0,4.0,0.1,nm,nm,m,Z,feat,\n"
        "p1,5,6,7.0,8.0,0.2,nm,nm,m,Z,feat,\n"
    )
    p = _write(tmp_path / "meas.csv", csv_text)
    preview = sniff_point_table(p)
    assert preview.units == "nm"
    assert preview.has_id_column is True
    (fs,) = load_point_table(p, scan_range_m=(50e-9, 50e-9), image_shape=(50, 50))
    np.testing.assert_allclose(fs.points_m[0], [3e-9, 4e-9])


def test_generic_csv_bare_xy_needs_units(tmp_path):
    p = _write(tmp_path / "bare.csv", "x,y\n1.0,2.0\n3.0,4.0\n")
    preview = sniff_point_table(p)
    assert preview.kind == "generic_csv"
    assert preview.units == "unknown"
    assert preview.has_id_column is False
    with pytest.raises(ValueError):
        load_point_table(p)  # units cannot be inferred
    (fs,) = load_point_table(p, units="nm", scan_range_m=(10e-9, 10e-9), image_shape=(10, 10))
    np.testing.assert_allclose(fs.points_m[1], [3e-9, 4e-9])


def test_generic_csv_headerless_with_implied_id(tmp_path):
    # leading 0-based integer sequence => id column, x/y are the next two columns
    p = _write(tmp_path / "noheader.csv", "0,12,34\n1,56,78\n2,90,11\n")
    preview = sniff_point_table(p)
    assert preview.has_header is False
    assert preview.has_id_column is True
    assert preview.n_points == 3
    (fs,) = load_point_table(p, units="px", scan_range_m=(100e-9, 100e-9), image_shape=(100, 100))
    np.testing.assert_allclose(fs.points_px[0], [12.0, 34.0])


def test_generic_csv_headerless_no_id(tmp_path):
    p = _write(tmp_path / "xy.csv", "12,34\n56,78\n")
    preview = sniff_point_table(p)
    assert preview.has_id_column is False
    (fs,) = load_point_table(p, units="px", scan_range_m=(100e-9, 100e-9), image_shape=(100, 100))
    np.testing.assert_allclose(fs.points_px[0], [12.0, 34.0])
    np.testing.assert_allclose(fs.points_px[1], [56.0, 78.0])


def test_px_units_pixel_columns_detected(tmp_path):
    p = _write(tmp_path / "pxcols.csv", "x_px,y_px\n4,5\n6,7\n")
    preview = sniff_point_table(p)
    assert preview.units == "px"


def test_semicolon_delimiter_sniffed(tmp_path):
    p = _write(tmp_path / "semi.csv", "x_nm;y_nm\n1.0;2.0\n3.0;4.0\n")
    preview = sniff_point_table(p)
    assert preview.delimiter == ";"
    assert preview.units == "nm"
    (fs,) = load_point_table(p, scan_range_m=(10e-9, 10e-9), image_shape=(10, 10))
    assert fs.point_count == 2


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def test_feature_set_store_json_roundtrip(tmp_path):
    fs1 = FeatureSet.from_points(
        name="img A",
        points_px=[[1, 2], [3, 4]],
        points_m=[[1e-9, 2e-9], [3e-9, 4e-9]],
        scan_range_m=(10e-9, 10e-9),
        image_shape=(10, 10),
    )
    fs2 = FeatureSet.from_points(
        name="img B",
        points_px=[[5, 6]],
        points_m=[[5e-9, 6e-9]],
        scan_range_m=(10e-9, 10e-9),
        image_shape=(10, 10),
    )
    store = FeatureSetStore([fs1, fs2])
    p = tmp_path / "sets.json"
    store.save(p)

    preview = sniff_point_table(p)
    assert preview.kind == "feature_set_store_json"
    assert preview.n_sets == 2
    assert preview.needs_calibration is False

    sets = load_point_table(p)
    assert len(sets) == 2
    assert {s.name for s in sets} == {"img A", "img B"}


# --------------------------------------------------------------------------- #
# Calibration defaults
# --------------------------------------------------------------------------- #
def test_default_image_shape_keeps_aspect():
    assert default_image_shape((100e-9, 50e-9)) == (512, 1024)
    assert default_image_shape((50e-9, 100e-9)) == (1024, 512)


def test_default_scan_range_contains_points():
    bbox = (0.0, 0.0, 100.0, 80.0)  # nm
    w, h = default_scan_range_m(bbox, "nm")
    assert w >= 100e-9
    assert h >= 80e-9


def test_unrecognised_json_raises(tmp_path):
    p = _write(tmp_path / "weird.json", json.dumps({"foo": "bar"}))
    with pytest.raises(ValueError):
        sniff_point_table(p)


def test_default_scan_range_sized_to_extent_for_offset_origin():
    bbox = (100.0, 200.0, 110.0, 208.0)  # nm, offset origin
    w, h = default_scan_range_m(bbox, "nm")
    np.testing.assert_allclose(w, 10e-9 * 1.05)
    np.testing.assert_allclose(h, 8e-9 * 1.05)


def test_offset_origin_points_recentred_and_distances_preserved(tmp_path):
    raw_nm = np.array([[100.0, 200.0], [110.0, 208.0], [104.0, 203.0]])
    csv_text = "x_nm,y_nm\n" + "\n".join(f"{x},{y}" for x, y in raw_nm) + "\n"
    p = _write(tmp_path / "offset.csv", csv_text)

    preview = sniff_point_table(p)
    assert any("offset origin" in note for note in preview.notes)

    (fs,) = load_point_table(p)
    w, h = fs.scan_range_m
    assert (fs.points_m >= 0.0).all()
    assert (fs.points_m[:, 0] <= w).all()
    assert (fs.points_m[:, 1] <= h).all()
    assert "import_recentered_offset_m" in fs.metadata
    # Re-centring is a rigid translation: inter-point distances are unchanged.
    raw_m = raw_nm * 1e-9
    np.testing.assert_allclose(
        np.linalg.norm(fs.points_m[1] - fs.points_m[0]),
        np.linalg.norm(raw_m[1] - raw_m[0]),
    )


def test_negative_coordinates_recentred_into_field(tmp_path):
    csv_text = "x_nm,y_nm\n-50.0,-40.0\n-45.0,-38.0\n-42.0,-33.0\n"
    p = _write(tmp_path / "negative.csv", csv_text)
    (fs,) = load_point_table(p)
    w, h = fs.scan_range_m
    assert (fs.points_m >= 0.0).all()
    assert (fs.points_m[:, 0] <= w).all()
    assert (fs.points_m[:, 1] <= h).all()


def test_points_that_fit_an_explicit_field_are_not_shifted(tmp_path):
    # A corner cluster inside a user-specified field must stay put, so
    # multi-file imports sharing one coordinate frame remain aligned.
    csv_text = "x_nm,y_nm\n60.0,70.0\n90.0,85.0\n"
    p = _write(tmp_path / "corner.csv", csv_text)
    (fs,) = load_point_table(p, scan_range_m=(100e-9, 100e-9), image_shape=(100, 100))
    np.testing.assert_allclose(fs.points_m[0], [60e-9, 70e-9])
    np.testing.assert_allclose(fs.points_m[1], [90e-9, 85e-9])
    assert "import_recentered_offset_m" not in fs.metadata


# --------------------------------------------------------------------------- #
# Format-robustness battery: "slightly different" real-world CSV shapes
# --------------------------------------------------------------------------- #
def test_excel_bom_does_not_hide_units(tmp_path):
    p = _write(tmp_path / "bom.csv", "﻿x_nm,y_nm\n1.0,2.0\n3.0,4.0\n")
    preview = sniff_point_table(p)
    assert preview.units == "nm"


def test_parenthesised_and_bracketed_unit_headers(tmp_path):
    for name, text in (
        ("paren.csv", "x (nm),y (nm)\n1.0,2.0\n3.0,4.0\n"),
        ("bracket.csv", "X [nm],Y [nm]\n1.0,2.0\n3.0,4.0\n"),
        ("dotted.csv", "x.nm,y.nm\n1.0,2.0\n3.0,4.0\n"),
    ):
        preview = sniff_point_table(_write(tmp_path / name, text))
        assert preview.units == "nm", name
        assert preview.n_points == 2, name


def test_decimal_comma_values_parse(tmp_path):
    p = _write(tmp_path / "euro.csv", "x_nm;y_nm\n1,25;2,50\n3,75;4,10\n")
    preview = sniff_point_table(p)
    assert preview.units == "nm"
    assert preview.n_points == 2
    assert any("Decimal commas" in note for note in preview.notes)
    (fs,) = load_point_table(p, scan_range_m=(10e-9, 10e-9), image_shape=(10, 10))
    np.testing.assert_allclose(fs.points_m[0], [1.25e-9, 2.50e-9])


def test_comment_header_and_space_delimited(tmp_path):
    p = _write(tmp_path / "commented.csv", "# x_nm y_nm\n1.0 2.0\n3.0 4.0\n")
    preview = sniff_point_table(p)
    assert preview.units == "nm"
    assert preview.n_points == 2
    (fs,) = load_point_table(p, scan_range_m=(10e-9, 10e-9), image_shape=(10, 10))
    np.testing.assert_allclose(fs.points_m[1], [3e-9, 4e-9])


def test_gwyddion_style_bare_comment_header(tmp_path):
    p = _write(tmp_path / "gwy.csv", "# point list\n# x y\n1.0e-9 2.0e-9\n3.0e-9 4.0e-9\n")
    preview = sniff_point_table(p)
    assert preview.n_points == 2
    assert preview.units == "unknown"  # bare x/y: user chooses (here: m)
    (fs,) = load_point_table(p, units="m", scan_range_m=(10e-9, 10e-9), image_shape=(10, 10))
    np.testing.assert_allclose(fs.points_m[0], [1e-9, 2e-9])


def test_imagej_xm_ym_is_ambiguous_not_metres(tmp_path):
    p = _write(tmp_path / "ij.csv", " ,Area,XM,YM\n1,5,1.0,2.0\n2,6,3.0,4.0\n")
    preview = sniff_point_table(p)
    assert preview.units == "unknown"
    assert any("ImageJ" in note for note in preview.notes)
    # The XM/YM columns are still the chosen coordinate columns.
    assert preview.x_col == 2 and preview.y_col == 3


def test_angstrom_and_pm_units(tmp_path):
    p = _write(tmp_path / "ang.csv", "x_A,y_A\n10.0,20.0\n30.0,40.0\n")
    preview = sniff_point_table(p)
    assert preview.units == "angstrom"
    (fs,) = load_point_table(p, scan_range_m=(10e-9, 10e-9), image_shape=(10, 10))
    np.testing.assert_allclose(fs.points_m[0], [1e-9, 2e-9])  # 10 A = 1 nm

    p2 = _write(tmp_path / "pico.csv", "x_pm,y_pm\n1000,2000\n3000,4000\n")
    preview2 = sniff_point_table(p2)
    assert preview2.units == "pm"
    (fs2,) = load_point_table(p2, scan_range_m=(10e-9, 10e-9), image_shape=(10, 10))
    np.testing.assert_allclose(fs2.points_m[0], [1e-9, 2e-9])  # 1000 pm = 1 nm


def test_unparseable_rows_are_counted_and_explained(tmp_path):
    p = _write(tmp_path / "mixed.csv", "x_nm,y_nm\n1.0,2.0\nbad,row\n3.0,4.0\n")
    preview = sniff_point_table(p)
    assert preview.n_points == 2
    assert preview.n_dropped_rows == 1
    assert any("skipped" in note for note in preview.notes)


def test_all_rows_unparseable_gives_actionable_error(tmp_path):
    # Decimal commas with a comma delimiter cannot be auto-fixed; the error
    # should say rows failed to parse rather than "no point rows found".
    p = _write(tmp_path / "broken.csv", "x_nm,y_nm\na,b\nc,d\n")
    with pytest.raises(ValueError, match="failed to\\s+parse"):
        load_point_table(p, scan_range_m=(10e-9, 10e-9), image_shape=(10, 10))


def test_sample_rows_and_columns_exposed_for_preview(tmp_path):
    p = _write(tmp_path / "prev.csv", "id,x_nm,y_nm\n0,1.0,2.0\n1,3.0,4.0\n")
    preview = sniff_point_table(p)
    assert preview.x_col == 1 and preview.y_col == 2
    assert preview.sample_rows[0] == ("0", "1.0", "2.0")


# --------------------------------------------------------------------------- #
# Multi-image (frame) splitting
# --------------------------------------------------------------------------- #
def test_frame_column_splits_into_aligned_sets(tmp_path):
    p = _write(
        tmp_path / "frames.csv",
        "particle,frame,x_nm,y_nm\n"
        "0,0,101.0,102.0\n1,0,103.0,104.0\n"
        "2,1,101.5,102.5\n3,1,103.5,104.5\n",
    )
    preview = sniff_point_table(p)
    assert preview.frame_column == "frame"
    assert preview.n_sets == 2

    sets = load_point_table(p)
    assert [fs.point_count for fs in sets] == [2, 2]
    a, b = sets
    assert a.scan_range_m == b.scan_range_m
    # One global re-centring shift: cross-set geometry is preserved.
    np.testing.assert_allclose((b.points_m[0] - a.points_m[0]) * 1e9, [0.5, 0.5])
    assert a.metadata["import_frame"] == "0"
    assert b.metadata["import_frame"] == "1"


def test_single_valued_frame_column_does_not_split(tmp_path):
    p = _write(
        tmp_path / "oneframe.csv",
        "particle,frame,x_nm,y_nm\n0,0,1.0,2.0\n1,0,3.0,4.0\n",
    )
    preview = sniff_point_table(p)
    assert preview.frame_column is None
    assert preview.n_sets == 1
    sets = load_point_table(p, scan_range_m=(10e-9, 10e-9), image_shape=(10, 10))
    assert len(sets) == 1 and sets[0].point_count == 2


# --------------------------------------------------------------------------- #
# Pixel-size derivation (ProbeFlow CSV round-trip)
# --------------------------------------------------------------------------- #
def test_pixel_size_derived_from_px_and_nm_columns(tmp_path):
    p = _write(
        tmp_path / "ff.csv",
        "index,x_px,y_px,x_nm,y_nm,value\n"
        "0,10.0,20.0,5.0,15.0,1.2\n"
        "1,30.0,40.0,15.0,30.0,3.4\n",
    )
    preview = sniff_point_table(p)
    assert preview.kind == "probeflow_csv"
    assert preview.pixel_size_m is not None
    np.testing.assert_allclose(preview.pixel_size_m, [0.5e-9, 0.75e-9])
    assert any("Pixel size" in note for note in preview.notes)
