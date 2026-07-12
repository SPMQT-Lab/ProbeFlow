"""Measurement algorithms and analysis helpers.

Architectural role
------------------
``analysis`` contains operations that calculate information from scan images
or spectroscopy data: particles, lattice parameters, profiles, unit-cell
tools, classification helpers, and analysis export helpers.  Canonical
result dataclasses live in :mod:`probeflow.measurements`; this package
calls into them and returns either those dataclasses directly or
domain-specific results (e.g. :class:`LatticeResult`,
:class:`PeriodicityResult`).

Boundary rules
--------------
Keep measurement algorithms here.  Provenance dataclasses (linear history
records) belong in ``probeflow.provenance``.  Do not place parser/writer
boundaries, CLI routing, GUI widgets, or raw ``Scan`` model ownership here.
"""

from probeflow.analysis.lattice import (
    LatticeParams,
    LatticeResult,
    UnitCellResult,
    average_unit_cell,
    extract_lattice,
    write_lattice_pdf,
)
from probeflow.analysis.line_periodicity import (
    PeriodicityDiagnostic,
    PeriodicityResult,
    estimate_line_periodicity,
    format_period,
    format_result_text,
)
__all__ = [
    "PeriodicityDiagnostic",
    "PeriodicityResult",
    "estimate_line_periodicity",
    "format_period",
    "format_result_text",
    "LatticeParams",
    "LatticeResult",
    "UnitCellResult",
    "average_unit_cell",
    "extract_lattice",
    "write_lattice_pdf",
]
