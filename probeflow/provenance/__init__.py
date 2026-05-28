"""Provenance records — production-canonical history of every export.

Architectural role
------------------
``provenance`` owns the *production* history model:

* :class:`ProcessingHistory` — the ordered list of canonical processing steps
  associated with a scan, the structure every export sidecar writes and every
  reader parses.
* :class:`ProvenanceStep` — one step in that history, with op name, params,
  and a kernel version stamp.
* :class:`ExportRecord` / :class:`SourceRecord` — the JSON-friendly records
  that PNG / SXM / CSV / PDF / GWY writers emit alongside the artifact.

This is what every export, GUI History panel, and CLI provenance flag actually
uses today (see :func:`processing_history_from_scan`,
:func:`append_processing_state`, :func:`build_export_record`).

Experimental graph model (`probeflow.provenance.graph`)
-------------------------------------------------------
``probeflow.provenance.graph`` contains a richer DAG-style alternate
representation (:class:`ScanGraph` with :class:`ImageNode` /
:class:`MeasurementNode` vertices, virtual-image recipes, ``materialize_image``
re-execution).  It is **not wired to any production code path** — the linear
``ProcessingHistory`` model above is the source of truth.  The graph module
is kept and exercised by its own tests so that a future migration can build
on a working baseline, but it is intentionally *not* re-exported from this
package's top-level namespace; import directly from
``probeflow.provenance.graph`` if you need it (review arch-backend #6).

Boundary rules
--------------
Keep history records and any future graph node dataclasses inside this
package.  Do not define them in ``processing`` or ``analysis``; those
packages should expose algorithms that can be wrapped by provenance-aware
adapters.
"""

from probeflow.provenance import export as _impl
from probeflow.provenance.records import (
    DAT_TO_SXM_CONVERSION_WARNING,
    PROCESSED_EXPORT_WARNING,
    ExportRecord,
    ProcessingHistory,
    ProcessingStep,  # deprecated alias — prefer ProvenanceStep
    ProvenanceStep,
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
    "DAT_TO_SXM_CONVERSION_WARNING",
    "PROCESSED_EXPORT_WARNING",
    "ExportRecord",
    "ProcessingHistory",
    "ProcessingStep",      # deprecated alias for ProvenanceStep
    "ProvenanceStep",
    "SourceRecord",
    "append_processing_state",
    "build_export_record",
    "display_lines",
    "processing_history_from_scan",
    "source_record_from_scan",
]
