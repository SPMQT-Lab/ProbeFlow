"""ProbeFlow — PySide6 GUI for STM scan browsing, processing, and Createc→Nanonis conversion.

NOTE — this is the working GUI implementation, not deprecated code.
The `_legacy` suffix reflects an in-progress refactor: classes will be
moved out of this file into dedicated submodules (`gui/dialogs/`,
`gui/viewer/`, `gui/browse/`, `gui/convert/`, `gui/features/`,
`gui/terminal/`) opportunistically as features touch them. Until that
work completes, the bulk of the GUI lives here. New widgets / dialogs
should still go in their proper subpackage; only edits to existing
classes belong in this file.

Boundary rules: do not add parsers, writers, numerical kernels,
analysis algorithms, model definitions, or graph-node dataclasses
here — those belong in `io/`, `processing/`, `analysis/`, `core/`, or
`provenance/` respectively.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import os as _os
_os.environ.setdefault("QT_API", "pyside6")
import matplotlib
matplotlib.use("QtAgg")

from PySide6.QtCore import (
    Qt, QThreadPool,
    Signal, Slot,
)
from PySide6.QtGui import (
    QAction, QActionGroup, QCursor, QFont, QKeySequence,
    QPixmap, QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
    QDialog, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton,
    QDockWidget, QScrollArea, QSizePolicy, QSplitter, QStackedWidget,
    QStatusBar, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QVBoxLayout, QWidget,
)
from probeflow.gui.utils import _open_url, _format_scan_conditions
from probeflow.gui.navbar import Navbar

from probeflow.gui.viewer.display_range import DisplayRangeController
from probeflow.gui.viewer.histogram import HistogramPanel
from probeflow.processing.gui_adapter import (
    processing_state_from_gui,
)
from probeflow.processing.state import (
    assert_roi_references_resolved,
    missing_roi_references,
)
from probeflow.provenance import (
    ProcessingHistory,
    append_processing_state,
    build_export_record,
    display_lines,
    processing_history_from_scan,
)
from probeflow.gui.features import (
    FeaturesPanel,
    FeaturesSidebar,
    _FeaturesWorker,
    _FeaturesWorkerSignals,
)
from probeflow.gui.features.tv import (
    TVPanel,
    TVSidebar,
    _TVWorker,
    _TVWorkerSignals,
)
from probeflow.gui.processing import ProcessingControlPanel
from probeflow.gui.terminal import DeveloperTerminalWidget
from probeflow.gui.dialogs import (
    AboutDialog,
    FFTViewerDialog,
    PeriodicFilterDialog,
    SpecMappingDialog,
    SpecOverlayDialog,
    SpecViewerDialog,
    STMBackgroundDialog,
)
from probeflow.core.scan_loader import load_scan
from probeflow.gui.viewer import (
    BadLinePreviewController,
    DeferredPlaneAction,
    DisplaySliderController,
    ProcessingUndoController,
    SetZeroPlaneController,
    SpecOverlayController,
    activate_roi,
    active_roi,
    active_roi_id,
    delete_active_roi,
    delete_roi,
    export_histogram,
    export_line_profile,
    has_roi_aware_local_filter,
    invert_active_roi,
    invert_roi,
    load_roi_set,
    plot_roi_line_profile,
    rename_roi,
    resolve_channel_unit,
    roi_canvas_created,
    roi_canvas_moved,
    roi_line_endpoint_changed,
    roi_line_set_width,
    save_roi_set,
    save_viewer_png,
    select_nth_roi,
    selected_or_active_roi_id,
    show_roi_fft,
    show_roi_histogram,
    transform_roi_set_for_display_op,
)
from probeflow.gui.config import (
    CONFIG_PATH,
    DEFAULT_CUSHION,
    LOGO_PATH,
    LOGO_GIF_PATH,
    LOGO_NAV_PATH,
    GITHUB_URL,
    GUI_FONT_SIZES,
    GUI_FONT_DEFAULT,
    normalise_gui_font_size,
    load_config,
    save_config,
)
from probeflow.gui.styling import (
    NAVBAR_DARK_BG,
    NAVBAR_LIGHT_BG,
    NAVBAR_H,
    THEMES,
    _sep,
    _build_qss,
)

# ── Extracted GUI helpers (re-exported for compatibility) ─────────────────────
from probeflow.gui.models import (
    PLANE_NAMES,
    FolderEntry,
    SxmFile,
    VertFile,
    scan_image_folder,
)
from probeflow.gui.rendering import (
    CMAP_KEY,
    CMAP_NAMES,
    DEFAULT_CMAP_KEY,
    DEFAULT_CMAP_LABEL,
    STM_COLORMAPS,
    THUMBNAIL_CHANNEL_DEFAULT,
    THUMBNAIL_CHANNEL_OPTIONS,
    _apply_processing,
)
from probeflow.gui.workers import (
    ConversionWorker,
    ViewerLoader,
)
from probeflow.gui.browse import ThumbnailGrid
from probeflow.gui.viewer.widgets import (
    LineProfilePanel,
    RulerWidget,
    ScaleBarWidget,
)
from probeflow.gui.widgets import ImageMeasurementsPanel
from probeflow.gui.image_canvas import ImageCanvas
from probeflow.gui.roi_manager_dock import ROIManagerDock
from probeflow.gui.viewer import ImageMeasurementController


# ── Viewer and browse support lives in extracted GUI modules. ───────────────
from probeflow.gui.dialogs.image_viewer import ImageViewerDialog

# ── Browse panels ─────────────────────────────────────────────────────────────
from probeflow.gui.browse import BrowseInfoPanel, BrowseToolPanel


# ── Features tab integration ────────────────────────────────────────────────
# Specialized add-on workflows live in probeflow.gui.features. Keep this main
# GUI file focused on Browse/Viewer/Convert orchestration; Features owns tools
# like particle counting, template counting, lattice extraction, and future
# TV-denoise/background-removal panels so optional analysis dependencies do not
# leak into routine browsing or image manipulation.


# ── Spec viewer dialog ───────────────────────────────────────────────────────
# ── Convert panel/sidebar ─────────────────────────────────────────────────────
from probeflow.gui.convert import ConvertPanel, ConvertSidebar


# ── Processing definitions panel ──────────────────────────────────────────────
from probeflow.gui.dialogs.definitions import _DefinitionsDialog

# ── Developer terminal sidebar ────────────────────────────────────────────────
from probeflow.gui.terminal import _DevSidebar


# ── Developer terminal ────────────────────────────────────────────────────────
# ── Main window and entry point (extracted to app.py) ─────────────────────────
from probeflow.gui.app import ProbeFlowWindow, main
