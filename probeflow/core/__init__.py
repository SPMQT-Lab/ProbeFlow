"""Core ProbeFlow domain objects and scan identity services.

Architectural role
------------------
This package is the intended home for the stable Session -> Probe ->
Scan/Spectrum model. Today it owns ``Scan``, scan loading, metadata, validation,
indexing, and source identity. Future ``Session``, abstract ``Probe``, and
``Spectrum`` domain models should live here.

Boundary rules
--------------
``core`` may attach a provenance graph to a probe object, but it must not define
``ImageNode``, ``MeasurementNode``, ``OperationNode``, ``ArtifactNode``, or
``ScanGraph``. Those graph dataclasses belong in ``probeflow.provenance``.
Keep parser/writer implementations in ``probeflow.io`` and array algorithms in
``probeflow.processing`` / ``probeflow.analysis``.
"""

from probeflow.core.scan_model import PLANE_CANON_NAMES, PLANE_CANON_UNITS, Scan
from probeflow.core.scan_loader import SUPPORTED_SUFFIXES, load_scan
from probeflow.core.metadata import ScanMetadata, metadata_from_scan, read_scan_metadata
from probeflow.core.indexing import ProbeFlowItem, index_folder
from probeflow.core.loaders import LoadSignature, identify_scan_file, identify_spectrum_file
from probeflow.core.roi import AREA_ROI_KINDS, ROI, ROISet

__all__ = [
    "PLANE_CANON_NAMES",
    "PLANE_CANON_UNITS",
    "SUPPORTED_SUFFIXES",
    "Scan",
    "load_scan",
    "ScanMetadata",
    "metadata_from_scan",
    "read_scan_metadata",
    "ProbeFlowItem",
    "index_folder",
    "LoadSignature",
    "identify_scan_file",
    "identify_spectrum_file",
    "ROI",
    "ROISet",
    "AREA_ROI_KINDS",
]
