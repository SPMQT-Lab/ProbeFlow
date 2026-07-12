"""GUI-free image transformation operations.

Architectural role
------------------
``processing`` contains array-in/array-out numerical transformations for scan
images: flattening, row alignment, smoothing, FFT filters, edge detection,
and similar kernels.  The canonical entry point is
:func:`probeflow.processing.state.apply_processing_state` (or the
calibration-aware variant
:func:`probeflow.processing.state.apply_processing_state_with_calibration`),
which walks a :class:`probeflow.processing.state.ProcessingState` of
:class:`ProcessingStep` entries and dispatches each to its kernel here.
The resulting steps are appended to the scan's
:class:`probeflow.provenance.records.ProcessingHistory` for export.

Boundary rules
--------------
Keep this package focused on operation functions, state adapters, and thin
wrappers around existing kernels.  Provenance history records belong in
``probeflow.provenance``.  Do not add GUI widgets, vendor parsers, or writer
implementations here.
"""

from probeflow.processing import image as _impl

globals().update({
    name: value
    for name, value in vars(_impl).items()
    if not (name.startswith("__") and name.endswith("__"))
})

__all__ = [
    name for name in globals()
    if not name.startswith("_")
]
