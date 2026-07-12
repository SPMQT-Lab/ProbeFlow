"""Core ProbeFlow domain objects and scan identity services.

Architectural role
------------------
``core`` owns the production domain model: :class:`Scan` (with its planes,
header, scan_range_m, and attached :class:`ProcessingState`), scan loading,
metadata, validation, indexing, source identity, and ROIs.  An eventual
``Session`` / abstract ``Probe`` / ``Spectrum`` triple would also live here.

Provenance attached to a ``Scan`` is the linear
:class:`probeflow.provenance.records.ProcessingHistory` model — that's the
representation every export sidecar writes and every reader parses.

Boundary rules
--------------
Keep parser/writer implementations in ``probeflow.io`` and array algorithms
in ``probeflow.processing`` / ``probeflow.analysis``.  Provenance records
live in ``probeflow.provenance``; ``core`` does not define them.
"""

from probeflow.core.scan_model import PLANE_CANON_NAMES, PLANE_CANON_UNITS, Scan
from probeflow.core.scan_loader import SUPPORTED_SUFFIXES, load_scan
from probeflow.core.metadata import ScanMetadata, metadata_from_scan, read_scan_metadata
from probeflow.core.indexing import ProbeFlowItem, index_folder
from probeflow.core.loaders import LoadSignature, identify_scan_file, identify_spectrum_file
from probeflow.core.roi import (
    AREA_ROI_KINDS,
    ROI,
    ROISet,
    ResizeHandle,
    resize_handles,
    resize_roi,
    roi_from_mask,
)
from probeflow.core.mask import ImageMask, MaskSet, mask_name

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
    "ResizeHandle",
    "resize_handles",
    "resize_roi",
    "roi_from_mask",
    "ImageMask",
    "MaskSet",
    "mask_name",
]
