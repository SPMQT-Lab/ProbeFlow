"""Tests for the point-table importer (CSV / JSON) for Particle Statistics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from probeflow.measurements.feature_sets import FeatureSet, FeatureSetStore
from probeflow.measurements.point_table_io import (
    default_image_shape,
    default_scan_range_m,
    feature_items_to_feature_set,
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
def test_feature_counting_particles_json(tmp_path):
    payload = {
        "meta": {
            "kind": "particles",
            "scan_range_m": [100e-9, 80e-9],
            "pixels": [200, 160],
        },
        "items": [
            {"index": 0, "centroid_x_m": 10e-9, "centroid_y_m": 20e-9, "area_nm2": 5.0},
            {"index": 1, "centroid_x_m": 30e-9, "centroid_y_m": 40e-9, "area_nm2": 6.0},
        ],
    }
    p = _write(tmp_path / "particles.json", json.dumps(payload))
    preview = sniff_point_table(p)
    assert preview.kind == "probeflow_json"
    assert preview.scan_range_m == (100e-9, 80e-9)
    assert preview.image_shape == (160, 200)  # (ny, nx)
    assert preview.needs_calibration is False

    (fs,) = load_point_table(p)
    assert fs.point_count == 2
    np.testing.assert_allclose(fs.points_m[0], [10e-9, 20e-9], rtol=1e-6)
    np.testing.assert_allclose(fs.points_m[1], [30e-9, 40e-9], rtol=1e-6)


def test_feature_counting_detections_json(tmp_path):
    payload = {
        "meta": {"kind": "detections", "scan_range_m": [50e-9, 50e-9], "pixels": [100, 100]},
        "items": [
            {"index": 0, "x_m": 5e-9, "y_m": 6e-9, "x_px": 10, "y_px": 12, "correlation": 0.9},
        ],
    }
    p = _write(tmp_path / "det.json", json.dumps(payload))
    (fs,) = load_point_table(p)
    assert fs.point_count == 1
    np.testing.assert_allclose(fs.points_m[0], [5e-9, 6e-9], rtol=1e-6)


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


# --------------------------------------------------------------------------- #
# Feature Counting -> FeatureSet (the live FC send path; exercises the adapter)
# --------------------------------------------------------------------------- #
def test_feature_items_to_feature_set_from_dicts():
    items = [
        {"index": 0, "centroid_x_m": 10e-9, "centroid_y_m": 20e-9, "area_nm2": 5.0},
        {"index": 1, "centroid_x_m": 30e-9, "centroid_y_m": 40e-9, "area_nm2": 6.0},
    ]
    fs = feature_items_to_feature_set(
        items,
        scan_range_m=(100e-9, 80e-9),
        image_shape=(160, 200),
        name="fc particles",
        source_type="feature_counting_particles",
    )
    assert fs.point_count == 2
    np.testing.assert_allclose(fs.points_m[0], [10e-9, 20e-9], rtol=1e-6)
    np.testing.assert_allclose(fs.points_m[1], [30e-9, 40e-9], rtol=1e-6)


def test_feature_items_to_feature_set_accepts_objects():
    class _P:
        def __init__(self, x, y):
            self.x_m, self.y_m, self.x_px, self.y_px = x, y, 0, 0

        def to_dict(self):
            return {"x_m": self.x_m, "y_m": self.y_m, "x_px": self.x_px, "y_px": self.y_px}

    fs = feature_items_to_feature_set(
        [_P(5e-9, 6e-9)],
        scan_range_m=(50e-9, 50e-9),
        image_shape=(100, 100),
        name="fc detections",
    )
    assert fs.point_count == 1
    np.testing.assert_allclose(fs.points_m[0], [5e-9, 6e-9], rtol=1e-6)
