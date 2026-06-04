"""Shared GUI helpers for lattice correction workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class PiezoConstantRecommendation:
    """X/Y-only piezo constant update derived from a lattice correction."""

    x_multiplier: float
    y_multiplier: float
    x_new: float
    y_new: float
    x_percent_change: float
    y_percent_change: float
    residual_shear: float
    residual_shear_angle_deg: float
    residual_rotation_deg: float
    residual_text: str
    warning: str


DEFAULT_STRUCTURE = KnownStructure(
    name="Hexagonal",
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


# Legacy default-structure name that baked the lattice constant into the string
# (it went stale when the user edited Lattice a). Migrate it to the plain name.
_LEGACY_DEFAULT_NAMES = {"hexagonal 2.46 å"}


def _migrate_legacy_name(structure: KnownStructure) -> KnownStructure:
    if structure.name.strip().casefold() in _LEGACY_DEFAULT_NAMES:
        return replace(structure, name=DEFAULT_STRUCTURE.name)
    return structure


def structures_from_config(cfg: dict | None) -> list[KnownStructure]:
    raw = (cfg or {}).get(CONFIG_KEY, [])
    items = raw if isinstance(raw, list) else []
    structures = [
        _migrate_legacy_name(s)
        for s in (structure_from_dict(item) for item in items)
        if s is not None
    ]
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


def piezo_constant_recommendation(
    correction: LatticeCorrection,
    *,
    x_current: float,
    y_current: float,
) -> PiezoConstantRecommendation:
    """Recommend X/Y piezo constants from the orientation-preserving stretch.

    Piezo constants only expose independent X and Y scale terms.  The full
    lattice correction can also contain shear and rotation, so this helper uses
    the diagonal of ``stretch_matrix`` as the closest X/Y-only update and reports
    the off-diagonal part as residual distortion.
    """
    for name, value in (("x_current", x_current), ("y_current", y_current)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a positive finite value")

    stretch = correction.stretch_matrix
    x_multiplier = float(stretch[0, 0])
    y_multiplier = float(stretch[1, 1])
    for name, value in (("x_multiplier", x_multiplier), ("y_multiplier", y_multiplier)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value")

    residual_shear = float(stretch[0, 1] / x_multiplier)
    residual_shear_angle_deg = math.degrees(math.atan(residual_shear))
    residual_rotation_deg = float(correction.polar_rotation_deg)
    if math.isfinite(residual_rotation_deg):
        rotation_text = f"rotation {residual_rotation_deg:+.3f}° ignored"
    else:
        rotation_text = "rotation/reflection ignored"
    residual_text = (
        f"Residual shear {residual_shear_angle_deg:+.3f}°; "
        f"{rotation_text}"
    )

    x_new = float(x_current) * x_multiplier
    y_new = float(y_current) * y_multiplier
    x_percent_change = (x_multiplier - 1.0) * 100.0
    y_percent_change = (y_multiplier - 1.0) * 100.0
    warning = ""
    if max(abs(x_percent_change), abs(y_percent_change)) > 10.0:
        warning = "Warning: >10% change; verify grid alignment and scan calibration."

    return PiezoConstantRecommendation(
        x_multiplier=x_multiplier,
        y_multiplier=y_multiplier,
        x_new=x_new,
        y_new=y_new,
        x_percent_change=x_percent_change,
        y_percent_change=y_percent_change,
        residual_shear=residual_shear,
        residual_shear_angle_deg=residual_shear_angle_deg,
        residual_rotation_deg=residual_rotation_deg,
        residual_text=residual_text,
        warning=warning,
    )
