from __future__ import annotations

import math

import pytest

from probeflow.analysis.lattice_distortion import IdealLattice, MeasuredLattice, compute_correction
from probeflow.gui.lattice_correction_ui import (
    CONFIG_KEY,
    DEFAULT_STRUCTURE,
    KnownStructure,
    correction_display_values,
    correction_main_lines,
    delete_structure,
    ideal_lattice_from_structure,
    piezo_constant_recommendation,
    structure_from_dict,
    structure_from_period,
    structures_from_config,
    structures_to_config,
    upsert_structure,
)


def test_empty_config_returns_default_structure():
    structures = structures_from_config({})

    assert structures == [DEFAULT_STRUCTURE]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_structure_config_rejects_nonfinite_dimensions(bad_value):
    assert structure_from_dict({
        "name": "invalid",
        "a_nm": bad_value,
        "b_nm": 0.246,
        "angle_deg": 60.0,
    }) is None


def test_structure_round_trip_preserves_unrelated_config():
    original = {"dark_mode": False}
    structures = [
        KnownStructure("Square test", "square", 0.5, 0.5, 90.0, "nm"),
        KnownStructure("Hex test", "hexagonal", 0.246, 0.246, 60.0, "Å"),
    ]

    cfg = structures_to_config(original, structures)
    loaded = structures_from_config(cfg)

    assert cfg["dark_mode"] is False
    assert cfg[CONFIG_KEY][0]["name"] == "Square test"
    assert loaded == structures


def test_upsert_and_delete_structure_by_name():
    initial = [KnownStructure("A", "square", 1.0, 1.0, 90.0, "nm")]
    updated = upsert_structure(
        initial,
        KnownStructure("A", "hexagonal", 0.246, 0.246, 60.0, "Å"),
    )

    assert len(updated) == 1
    assert updated[0].symmetry == "hexagonal"
    assert delete_structure(updated, "A") == [DEFAULT_STRUCTURE]


def test_hexagonal_structure_uses_equivalent_angle_closest_to_measurement():
    structure = KnownStructure("hex", "hexagonal", 0.246, 0.246, 60.0, "Å")

    assert ideal_lattice_from_structure(structure, measured_angle_deg=58.0).angle_deg == 60.0
    assert ideal_lattice_from_structure(structure, measured_angle_deg=121.0).angle_deg == 120.0


def test_structure_from_period_sets_symmetry_target():
    structure = structure_from_period("line", 2.46e-10, "hexagonal")

    assert structure.symmetry == "hexagonal"
    assert structure.a_nm == pytest.approx(0.246)
    assert structure.b_nm == pytest.approx(0.246)
    assert structure.angle_deg == pytest.approx(60.0)


def test_correction_display_values_include_imagej_style_shear_angle():
    correction = compute_correction(
        MeasuredLattice(a_nm=(2.0, 0.0), b_nm=(0.5, 1.0)),
        IdealLattice(a_nm=1.0, b_nm=1.0, angle_deg=90.0),
    )
    assert not isinstance(correction, str)

    values = correction_display_values(correction, preserve_orientation=True)
    lines = correction_main_lines(correction, preserve_orientation=True)

    assert values["y_over_x"] == pytest.approx(correction.y_over_x)
    assert values["y_scale"] == pytest.approx(correction.x_scale * correction.y_over_x)
    assert values["shear_angle_deg"] == pytest.approx(math.degrees(math.atan(correction.shear)))
    assert "Undistort: y/x=" in lines[0]
    assert "shear=" in lines[0]


def test_piezo_recommendation_identity_keeps_constants():
    correction = compute_correction(
        MeasuredLattice(a_nm=(1.0, 0.0), b_nm=(0.0, 1.0)),
        IdealLattice(a_nm=1.0, b_nm=1.0, angle_deg=90.0),
    )
    assert not isinstance(correction, str)

    rec = piezo_constant_recommendation(correction, x_current=96.52, y_current=88.3)

    assert rec.x_multiplier == pytest.approx(1.0)
    assert rec.y_multiplier == pytest.approx(1.0)
    assert rec.x_new == pytest.approx(96.52)
    assert rec.y_new == pytest.approx(88.3)
    assert rec.warning == ""


def test_piezo_recommendation_uses_stretch_diagonal():
    correction = compute_correction(
        MeasuredLattice(a_nm=(2.0, 0.0), b_nm=(0.0, 0.5)),
        IdealLattice(a_nm=1.0, b_nm=1.0, angle_deg=90.0),
    )
    assert not isinstance(correction, str)

    rec = piezo_constant_recommendation(correction, x_current=100.0, y_current=100.0)

    assert rec.x_multiplier == pytest.approx(0.5)
    assert rec.y_multiplier == pytest.approx(2.0)
    assert rec.x_new == pytest.approx(50.0)
    assert rec.y_new == pytest.approx(200.0)
    assert rec.x_percent_change == pytest.approx(-50.0)
    assert rec.y_percent_change == pytest.approx(100.0)
    assert ">10% change" in rec.warning


def test_piezo_recommendation_reports_shear_as_residual():
    correction = compute_correction(
        MeasuredLattice(a_nm=(1.0, 0.0), b_nm=(0.05, 1.0)),
        IdealLattice(a_nm=1.0, b_nm=1.0, angle_deg=90.0),
    )
    assert not isinstance(correction, str)

    rec = piezo_constant_recommendation(correction, x_current=100.0, y_current=100.0)

    assert rec.x_new == pytest.approx(100.0, rel=2e-3)
    assert rec.y_new == pytest.approx(100.0, rel=2e-3)
    assert abs(rec.residual_shear_angle_deg) > 0.1
    assert "Residual shear" in rec.residual_text


def test_piezo_recommendation_rejects_invalid_constants():
    correction = compute_correction(
        MeasuredLattice(a_nm=(1.0, 0.0), b_nm=(0.0, 1.0)),
        IdealLattice(a_nm=1.0, b_nm=1.0, angle_deg=90.0),
    )
    assert not isinstance(correction, str)

    with pytest.raises(ValueError, match="x_current"):
        piezo_constant_recommendation(correction, x_current=0.0, y_current=1.0)
    with pytest.raises(ValueError, match="y_current"):
        piezo_constant_recommendation(correction, x_current=1.0, y_current=-1.0)
