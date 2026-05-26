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
    structure_from_period,
    structures_from_config,
    structures_to_config,
    upsert_structure,
)


def test_empty_config_returns_default_structure():
    structures = structures_from_config({})

    assert structures == [DEFAULT_STRUCTURE]


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
