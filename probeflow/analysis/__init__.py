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
from probeflow.analysis.preview import (
    FeatureMode,
    apply_preview_background,
    detect_preview_features,
    PreviewAnalysisParams,
    PreviewFeatureRow,
    PreviewResult,
    preview_record,
    run_preview,
)
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
from probeflow.analysis.xmgrace_export import Curve, build_agr, export_bundle

__all__ = [
    "Classification",
    "Curve",
    "PeriodicityDiagnostic",
    "PeriodicityResult",
    "estimate_line_periodicity",
    "format_period",
    "format_result_text",
    "Detection",
    "FeatureMode",
    "apply_preview_background",
    "detect_preview_features",
    "LatticeParams",
    "LatticeResult",
    "Particle",
    "PreviewAnalysisParams",
    "PreviewFeatureRow",
    "PreviewResult",
    "UnitCellResult",
    "average_unit_cell",
    "build_agr",
    "classify_particles",
    "count_features",
    "export_bundle",
    "extract_lattice",
    "preview_record",
    "run_preview",
    "segment_particles",
    "write_lattice_pdf",
]
