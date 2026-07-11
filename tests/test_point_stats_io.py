"""Tests for the point-statistics CSV/JSON serialisers."""

from __future__ import annotations

import json

import numpy as np

from probeflow.measurements.point_stats_io import (
    point_stats_csv_text,
    point_stats_json_text,
)


def _scalars():
    return [
        ("Points", 42, ""),
        ("Region area", 100.0, "nm^2"),
        ("Density", 0.42, "nm^-2"),
        ("NN mean", 1.234567, "nm"),
        ("NN median", None, "nm"),  # missing value -> empty cell
    ]


def test_csv_has_header_rows_and_formats_values():
    text = point_stats_csv_text(_scalars())
    lines = text.strip().splitlines()
    assert lines[0] == "quantity,value,unit"
    assert lines[1] == "Points,42,"
    assert lines[2] == "Region area,100,nm^2"
    assert lines[4] == "NN mean,1.23457,nm"       # rounded to 6 sig figs
    assert lines[5] == "NN median,,nm"            # None -> empty


def test_csv_escapes_commas_in_labels():
    text = point_stats_csv_text([("a, b", 1, "")])
    assert "a; b,1," in text


def test_json_roundtrips_scalars_and_curves():
    curves = {
        "g_r": {"r_nm": np.array([0.5, 1.5]), "g": np.array([0.9, 1.1])},
    }
    text = point_stats_json_text(_scalars(), curves)
    obj = json.loads(text)
    assert obj["statistics"][0] == {"quantity": "Points", "value": 42, "unit": ""}
    assert obj["statistics"][4]["value"] is None
    # numpy arrays serialised as plain lists
    assert obj["curves"]["g_r"]["r_nm"] == [0.5, 1.5]
    assert obj["curves"]["g_r"]["g"] == [0.9, 1.1]


def test_json_without_curves_omits_the_key():
    obj = json.loads(point_stats_json_text(_scalars()))
    assert "curves" not in obj
