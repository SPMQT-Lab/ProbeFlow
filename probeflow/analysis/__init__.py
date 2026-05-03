"""Measurement algorithms and analysis helpers.

Architectural role
------------------
``analysis`` contains operations that calculate information from scan images or
spectroscopy data: particles, lattice parameters, profiles, unit-cell tools,
classification helpers, and analysis export helpers. In the intended graph
architecture, these algorithms produce values that provenance-aware adapters can
wrap into ``MeasurementNode`` records.

Boundary rules
--------------
Keep measurement algorithms here, not graph models. Do not define
``MeasurementNode`` or other provenance node dataclasses in this package. Do not
place parser/writer boundaries, CLI routing, GUI widgets, or raw ``Scan`` model
ownership here.
"""

from probeflow.analysis.features import (
    Classification,
    Detection,
    Particle,
    classify_particles,
    count_features,
    segment_particles,
)
from probeflow.analysis.lattice import (
    LatticeParams,
    LatticeResult,
    UnitCellResult,
    average_unit_cell,
    extract_lattice,
    write_lattice_pdf,
)
from probeflow.analysis.xmgrace_export import Curve, build_agr, export_bundle

__all__ = [
    "Classification",
    "Curve",
    "Detection",
    "LatticeParams",
    "LatticeResult",
    "Particle",
    "UnitCellResult",
    "average_unit_cell",
    "build_agr",
    "classify_particles",
    "count_features",
    "export_bundle",
    "extract_lattice",
    "segment_particles",
    "write_lattice_pdf",
]
