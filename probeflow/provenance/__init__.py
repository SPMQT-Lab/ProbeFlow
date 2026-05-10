"""Provenance records and the future probe-owned graph model.

Architectural role
------------------
``provenance`` is the only intended home for ``ImageNode``,
``MeasurementNode``, ``OperationNode``, ``ArtifactNode``, and ``ScanGraph``.
Each future ``Probe`` should own its graph. Parsers will create root image
nodes, processing operations will create virtual image recipes, analysis
operations will create measurement nodes, and writers will create artifact
nodes.

Boundary rules
--------------
The current export provenance code is export-level history, not the final graph
model. Keep graph node dataclasses here when they are added. Do not define them
in ``processing`` or ``analysis``; those packages should expose algorithms that
can be wrapped by provenance-aware adapters.
"""

from probeflow.provenance import export as _impl
from probeflow.provenance.graph import (
    ImageNode,
    MeasurementNode,
    Node,
    OpRegistry,
    ScanGraph,
    graph_from_dict,
    graph_to_dict,
    materialize_image,
)
from probeflow.provenance.records import (
    DAT_TO_SXM_CONVERSION_WARNING,
    PROCESSED_EXPORT_WARNING,
    ExportRecord,
    ProcessingHistory,
    ProcessingStep,
    SourceRecord,
    append_processing_state,
    build_export_record,
    display_lines,
    processing_history_from_scan,
    source_record_from_scan,
)

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})

__all__ = [
    "ImageNode",
    "MeasurementNode",
    "Node",
    "OpRegistry",
    "ScanGraph",
    "graph_from_dict",
    "graph_to_dict",
    "materialize_image",
    "DAT_TO_SXM_CONVERSION_WARNING",
    "PROCESSED_EXPORT_WARNING",
    "ExportRecord",
    "ProcessingHistory",
    "ProcessingStep",
    "SourceRecord",
    "append_processing_state",
    "build_export_record",
    "display_lines",
    "processing_history_from_scan",
    "source_record_from_scan",
]
