"""ProbeFlow — STM scan browser, processor, and Createc↔Nanonis toolkit.

This package provides:
  * A Qt desktop GUI (``probeflow.gui``)
  * Createc ``.dat`` → Nanonis ``.sxm`` and PNG conversion pipelines
    (``probeflow.io.converters.createc_dat_to_sxm``, ``probeflow.io.converters.createc_dat_to_png``)
  * A GUI-free image processing library for STM data
    (``probeflow.processing``)
  * A unified command-line interface (``probeflow.cli``)

The library is importable without PySide6:

    from probeflow import processing
    from probeflow.io.converters.createc_dat_to_sxm import process_dat, convert_dat_to_sxm
    from probeflow.io.converters.createc_dat_to_png import dat_to_hdr_imgs

Launch the GUI via ``probeflow gui`` (see ``pyproject.toml`` for the
console-script wiring) or programmatically via ``probeflow.gui.main()``.
"""

__version__ = "beta"

# Public API: the vendor-agnostic Scan abstraction + dispatcher.
# Importing these does not pull in PySide6 / matplotlib.
from probeflow.core.scan_model import Scan
from probeflow.core.scan_loader import load_scan
from probeflow.core.metadata import ScanMetadata, read_scan_metadata, metadata_from_scan
from probeflow.core.indexing import ProbeFlowItem, index_folder
from probeflow.core.loaders import LoadSignature, identify_scan_file, identify_spectrum_file

__all__ = [
    "Scan", "load_scan",
    "ScanMetadata", "read_scan_metadata", "metadata_from_scan",
    "ProbeFlowItem", "index_folder",
    "LoadSignature", "identify_scan_file", "identify_spectrum_file",
    "__version__",
]
