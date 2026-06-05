"""Backward-compatibility re-exports for ``probeflow.gui``.

This module used to be ``probeflow.gui._legacy`` and held the main-window
implementation; classes have since been transplanted into their proper
subpackages (``app``, ``browse``, ``convert``, ``dialogs``, ``features``,
``navbar``, ``processing``, ``terminal``).  The remaining purpose of this
file is to keep the historical ``from probeflow.gui import X`` import surface
stable while ``gui/__init__.py`` re-exports each name through its
``_LEGACY_EXPORTS`` set.

Boundary rules: do not add parsers, writers, numerical kernels, analysis
algorithms, model definitions, or graph-node dataclasses here — those belong
in ``io/``, ``processing/``, ``analysis/``, ``core/``, or ``provenance/``.
"""

from __future__ import annotations

# Re-exports below are consumed via ``gui/__init__.py:_LEGACY_EXPORTS`` and
# (in two test cases) via ``import probeflow.gui.compat as gui_mod``.

# QFileDialog is patched at module-level by
# tests/test_gui_processing_panel.py (``gui_mod.QFileDialog``); keep it here
# even though no code in this file references it directly.
from PySide6.QtWidgets import QFileDialog  # noqa: F401

from probeflow.gui.app import ProbeFlowWindow, main  # noqa: F401
from probeflow.gui.navbar import Navbar  # noqa: F401
from probeflow.gui.processing import ProcessingControlPanel  # noqa: F401
from probeflow.gui.terminal import DeveloperTerminalWidget, _DevSidebar  # noqa: F401
from probeflow.gui.dialogs import (  # noqa: F401
    AboutDialog,
    EdgeDetectionDialog,
    FFTViewerDialog,
    PeriodicFilterDialog,
    SpecMappingDialog,
    SpecOverlayDialog,
    SpecViewerDialog,
    STMBackgroundDialog,
)
from probeflow.gui.dialogs.image_viewer import ImageViewerDialog  # noqa: F401
from probeflow.gui.dialogs.definitions import _DefinitionsDialog  # noqa: F401
from probeflow.gui.browse import (  # noqa: F401
    BrowseInfoPanel,
    BrowseToolPanel,
    ThumbnailGrid,
)
from probeflow.gui.convert import ConvertPanel, ConvertSidebar  # noqa: F401
