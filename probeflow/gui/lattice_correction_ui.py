"""Shared GUI helpers for lattice correction workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from probeflow.analysis.lattice_distortion import IdealLattice, LatticeCorrection

CONFIG_KEY = "lattice_structures_v1"


@dataclass(frozen=True)
class KnownStructure:
    """User-visible known surface lattice."""

    name: str
    symmetry: str
    a_nm: float
    b_nm: float
    angle_deg: float
    unit: str = "Å"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "symmetry": self.symmetry,
            "a_nm": float(self.a_nm),
            "b_nm": float(self.b_nm),
            "angle_deg": float(self.angle_deg),
            "unit": self.unit,
        }


DEFAULT_STRUCTURE = KnownStructure(
    name="Hexagonal 2.46 Å",
    symmetry="hexagonal",
    a_nm=0.246,
    b_nm=0.246,
    angle_deg=60.0,
    unit="Å",
)

VALID_SYMMETRIES = ("square", "rectangular", "hexagonal", "custom")


def normalise_symmetry(value: str | None) -> str:
    text = (value or "").strip().lower()
    return text if text in VALID_SYMMETRIES else "custom"


def structure_from_dict(data: object) -> KnownStructure | None:
    if not isinstance(data, dict):
        return None
    try:
        name = str(data.get("name", "")).strip()
        symmetry = normalise_symmetry(str(data.get("symmetry", "")))
        a_nm = float(data.get("a_nm", 0.0))
        b_nm = float(data.get("b_nm", a_nm))
        angle_deg = float(data.get("angle_deg", 90.0))
        unit = str(data.get("unit", "Å")).strip() or "Å"
    except Exception:
        return None
    if not name or a_nm <= 0.0 or b_nm <= 0.0 or not (1.0 <= angle_deg <= 179.0):
        return None
    return KnownStructure(name, symmetry, a_nm, b_nm, angle_deg, unit)


def structures_from_config(cfg: dict | None) -> list[KnownStructure]:
    raw = (cfg or {}).get(CONFIG_KEY, [])
    items = raw if isinstance(raw, list) else []
    structures = [s for s in (structure_from_dict(item) for item in items) if s is not None]
    return structures or [DEFAULT_STRUCTURE]


def structures_to_config(
    cfg: dict | None,
    structures: Iterable[KnownStructure],
) -> dict:
    out = dict(cfg or {})
    cleaned = []
    seen: set[str] = set()
    for structure in structures:
        name_key = structure.name.strip().casefold()
        if name_key and name_key not in seen:
            cleaned.append(structure.as_dict())
            seen.add(name_key)
    out[CONFIG_KEY] = cleaned or [DEFAULT_STRUCTURE.as_dict()]
    return out


def load_known_structures() -> list[KnownStructure]:
    from probeflow.gui.config import load_config

    return structures_from_config(load_config())


def save_known_structures(structures: Iterable[KnownStructure]) -> None:
    from probeflow.gui.config import load_config, save_config

    save_config(structures_to_config(load_config(), structures))


def upsert_structure(
    structures: Iterable[KnownStructure],
    structure: KnownStructure,
) -> list[KnownStructure]:
    out = []
    replaced = False
    target = structure.name.strip().casefold()
    for item in structures:
        if item.name.strip().casefold() == target:
            out.append(structure)
            replaced = True
        else:
            out.append(item)
    if not replaced:
        out.append(structure)
    return out


def delete_structure(
    structures: Iterable[KnownStructure],
    name: str,
) -> list[KnownStructure]:
    target = name.strip().casefold()
    out = [item for item in structures if item.name.strip().casefold() != target]
    return out or [DEFAULT_STRUCTURE]


def structure_display_value_nm(structure: KnownStructure) -> float:
    return structure.a_nm * 10.0 if structure.unit == "Å" else structure.a_nm


def structure_angle_for_measurement(
    structure: KnownStructure,
    measured_angle_deg: float | None,
) -> float:
    angle = float(structure.angle_deg)
    if structure.symmetry != "hexagonal" or measured_angle_deg is None:
        return angle
    candidates = (angle, 180.0 - angle)
    return min(candidates, key=lambda candidate: abs(candidate - measured_angle_deg))


def ideal_lattice_from_structure(
    structure: KnownStructure,
    *,
    measured_angle_deg: float | None = None,
) -> IdealLattice:
    return IdealLattice(
        a_nm=float(structure.a_nm),
        b_nm=float(structure.b_nm),
        angle_deg=structure_angle_for_measurement(structure, measured_angle_deg),
    )


def structure_from_period(
    name: str,
    period_m: float,
    symmetry: str,
) -> KnownStructure:
    a_nm = float(period_m) * 1e9
    sym = normalise_symmetry(symmetry)
    if sym == "hexagonal":
        return KnownStructure(name, sym, a_nm, a_nm, 60.0, "Å")
    if sym == "square":
        return KnownStructure(name, sym, a_nm, a_nm, 90.0, "Å")
    if sym == "rectangular":
        return KnownStructure(name, sym, a_nm, a_nm, 90.0, "Å")
    return KnownStructure(name, "custom", a_nm, a_nm, 90.0, "Å")


def correction_display_values(
    correction: LatticeCorrection,
    *,
    preserve_orientation: bool,
) -> dict[str, float | str]:
    y_scale = correction.x_scale * correction.y_over_x
    shear_angle_deg = math.degrees(math.atan(correction.shear))
    return {
        "x_scale": float(correction.x_scale),
        "y_scale": float(y_scale),
        "y_over_x": float(correction.y_over_x),
        "shear": float(correction.shear),
        "shear_angle_deg": float(shear_angle_deg),
        "rotation_deg": float(correction.polar_rotation_deg),
        "rotation_state": "removed" if preserve_orientation else "applied",
    }


def correction_main_lines(
    correction: LatticeCorrection,
    *,
    preserve_orientation: bool,
) -> list[str]:
    values = correction_display_values(correction, preserve_orientation=preserve_orientation)
    return [
        (
            f"Undistort: y/x={values['y_over_x']:.5f}   "
            f"shear={values['shear_angle_deg']:+.3f}°"
        ),
        (
            f"X× {values['x_scale']:.5f}   Y× {values['y_scale']:.5f}   "
            f"rotation {values['rotation_deg']:.3f}° {values['rotation_state']}"
        ),
    ]
