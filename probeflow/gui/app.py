"""ProbeFlow main window and application entry point.

This module contains ``ProbeFlowWindow`` (the top-level QMainWindow) and the
``main()`` entry point.  It was extracted from the historical ``_legacy.py``
as Phase 5 of the ongoing refactor; ``probeflow.gui.compat`` now re-imports
from here for backward compatibility.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

import numpy as np

import os as _os
_os.environ.setdefault("QT_API", "pyside6")
import matplotlib
matplotlib.use("QtAgg")

from probeflow.gui.typography import ui_font
from PySide6.QtCore import (
    Qt, QObject, QRunnable, QThreadPool,
    Signal, Slot,
)
from PySide6.QtGui import (
    QAction, QActionGroup, QFont, QKeySequence, QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QFileDialog,
    QHeaderView, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QSplitter, QStackedWidget,
    QStatusBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from probeflow.gui.utils import _open_url
from probeflow.gui.features import (
    FeaturesPanel,
    FeaturesSidebar,
)
from probeflow.gui.features.controller import FeatureCountingController
from probeflow.gui.features.tv import (
    TVPanel,
    TVSidebar,
    _TVWorker,
    _TVWorkerSignals,
)
from probeflow.gui.terminal import DeveloperTerminalWidget
from probeflow.gui.dialogs import (
    AboutDialog,
    SpecMappingDialog,
    SpecOverlayDialog,
)
from probeflow.core.scan_loader import load_scan
from probeflow.gui.config import (
    GITHUB_URL,
    GUI_FONT_SIZES,
    normalise_gui_font_size,
    load_config,
    save_config,
)
from probeflow.gui.desktop_layout import (
    apply_screen_fraction_geometry,
    b64_to_qbytearray,
    qbytearray_to_b64,
    restore_geometry_or_default,
)
from probeflow.gui.styling import (
    THEMES,
    THEME_PRESETS,
    theme_is_dark,
    _build_qss,
    _build_palette,
)
from probeflow.gui.typography import ui_family
from probeflow.gui.models import (
    FolderEntry,
    SxmFile,
    VertFile,
    scan_image_folder,
)
from probeflow.gui.rendering import (
    CMAP_KEY,
    DEFAULT_CMAP_KEY,
    DEFAULT_CMAP_LABEL,
    THUMBNAIL_CHANNEL_DEFAULT,
    THUMBNAIL_CHANNEL_OPTIONS,
)
from probeflow.gui.workers import (
    ConversionWorker,
    FilteredFolderExportWorker,
)
from probeflow.gui.browse import ThumbnailGrid, BrowseInfoPanel, BrowseToolPanel
from probeflow.gui.convert import ConvertPanel, ConvertSidebar
from probeflow.gui.dataset_builder import DatasetBuilderPanel, DatasetBuilderSidebar
from probeflow.gui.dialogs.definitions import _DefinitionsDialog
from probeflow.gui.terminal import _DevSidebar
from probeflow.gui.dialogs.image_viewer import ImageViewerDialog
from probeflow.gui.dialogs import SpecViewerDialog


# ── Scan-load background worker ───────────────────────────────────────────────

class _ScanLoadSignals(QObject):
    """Signals for the off-thread scan-load worker."""
    finished = Signal(object, str)   # (result_tuple_or_None, error_str)


class _ScanLoadWorker(QRunnable):
    """Load a scan plane off the main thread so the FC window stays responsive.

    ``load_fn`` must be callable as ``load_fn(entry, plane_idx) → tuple`` where
    the returned tuple is ``(arr, px_m, px_x_m, px_y_m, plane_idx, scan)``
    (i.e. the same signature as ``_load_scan_plane_for_analysis``).
    """

    def __init__(self, load_fn, entry, plane_idx: int) -> None:
        super().__init__()
        # Parent the signals to the QApplication (main thread). QThreadPool
        # auto-deletes this QRunnable on the *worker* thread; without a
        # main-thread parent the sole-owned signals QObject would be destroyed
        # off-thread, corrupting Qt internals (a SIGSEGV seen in an unrelated
        # app event filter). run() deleteLater()s it to avoid accumulation.
        self.signals   = _ScanLoadSignals(QApplication.instance())
        self._load_fn  = load_fn
        self._entry    = entry
        self._plane_idx = plane_idx

    def run(self) -> None:
        try:
            result = self._load_fn(self._entry, self._plane_idx)
            self.signals.finished.emit(result, "")
        except Exception as exc:  # noqa: BLE001
            self.signals.finished.emit(None, str(exc))
        finally:
            self.signals.deleteLater()


# ── Main window ───────────────────────────────────────────────────────────────
class ProbeFlowWindow(QMainWindow):
    LEFT_SIDEBAR_DEFAULT_W = 280
    LEFT_SIDEBAR_MIN_W = 240
    RIGHT_INSPECTOR_DEFAULT_W = 340
    RIGHT_INSPECTOR_MIN_W = 300
    CENTRAL_BROWSER_MIN_W = 500

    def __init__(self, *, open_survey: Optional[Path] = None,
                 browse_folder: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("ProbeFlow")
        self.setMinimumSize(1100, 720)
        self.resize(1480, 800)
        self._show_maximized_on_start = False

        self._cfg      = load_config()
        # Theme is now a named preset; migrate from the legacy dark_mode boolean.
        self._theme_name = self._cfg.get("theme_name") or (
            "dark" if self._cfg.get("dark_mode", True) else "light"
        )
        if self._theme_name not in THEMES:
            self._theme_name = "dark"
        self._dark     = theme_is_dark(self._theme_name)
        self._gui_font_size = normalise_gui_font_size(self._cfg.get("gui_font_size"))
        self._mode     = "browse"
        self._running  = False
        self._n_loaded = 0
        self._pending_survey = Path(open_survey) if open_survey else None
        self._pending_browse = Path(browse_folder) if browse_folder else None
        # Spec → image mapping (populated by user via "Map spectra…" dialogs;
        # kept empty by default so we never auto-attach spectra to the wrong
        # image based on coordinate guesses alone). Keys are spec stems,
        # values are image stems within the currently loaded folder.
        self._spec_image_map: dict[str, str] = {}
        # Non-modal viewer dialogs are shown with show() rather than exec() so
        # that the browse window remains interactive while scans are open.  We
        # keep explicit Python references here so the dialogs are not
        # garbage-collected while open (show() does not block like exec()).
        self._open_viewers: set = set()

        # Per-scan processing memory: path_str → (mtime_at_save, processing dict).
        # Populated when a viewer closes; restored when the same scan is reopened
        # *and* its mtime is unchanged (review gui-arch #10).  This prevents
        # cached processing from replaying on top of stale data after ScanFlow
        # or an external editor rewrites the file.
        self._saved_processing: dict[str, tuple[float | None, dict]] = {}

        # Probe whether the optional Survey/ScanFlow integration is importable
        # (review gui-arch #17).  We resolve this before ``_build_ui`` so the
        # Tools menu and the deferred SurveyPanel construction agree.
        self._survey_available = self._probe_survey_available()

        self._build_ui()
        self._apply_theme()
        self._restore_desktop_layout()

        # If launched with --browse (e.g. via the Restart action), open that
        # folder immediately so the user lands back where they were.
        if self._pending_browse is not None:
            try:
                self._grid.set_root(self._pending_browse)
                self._update_browse_status()
            except Exception:
                pass   # non-fatal: bad path just opens to an empty Browse

        # If launched with --open-survey, jump straight into Survey mode with
        # the manifest pre-loaded. Wire the panel's log_message into status bar.
        if self._pending_survey is not None:
            if not getattr(self, "_survey_available", False):
                self._status_bar.showMessage(
                    "Survey mode is unavailable — install `scanflow` to enable it."
                )
            else:
                try:
                    self._survey_panel.log_message.connect(self._status_bar.showMessage)
                    if self._survey_panel.load_manifest(self._pending_survey):
                        self._switch_mode("survey")
                except Exception as e:
                    self._status_bar.showMessage(f"Could not open survey: {e}")

    # ── Per-scan processing cache (mtime-aware) ───────────────────────────────

    @staticmethod
    def _entry_mtime(entry) -> float | None:
        """Return current mtime for an entry's path, or ``None`` on error."""
        try:
            import os as _os
            return float(_os.path.getmtime(str(entry.path)))
        except Exception:
            return None

    def _saved_processing_get(self, entry) -> dict | None:
        """Return cached processing for *entry* if and only if its mtime matches.

        On mismatch (file rewritten on disk by ScanFlow / external editor /
        re-save), drop the cache entry and emit a status-bar notice so the
        user understands why their saved processing no longer applies.
        """
        key = str(entry.path)
        cached = self._saved_processing.get(key)
        if cached is None:
            return None
        cached_mtime, proc = cached
        current_mtime = self._entry_mtime(entry)
        if (cached_mtime is None or current_mtime is None
                or abs(current_mtime - cached_mtime) > 1e-3):
            self._saved_processing.pop(key, None)
            if hasattr(self, "_status_bar"):
                self._status_bar.showMessage(
                    f"Saved processing for {entry.path.name} dropped — file "
                    "changed on disk."
                )
            return None
        return proc

    def _saved_processing_set(self, entry, processing: dict) -> None:
        """Cache *processing* for *entry* together with the current mtime."""
        key = str(entry.path)
        if processing:
            self._saved_processing[key] = (self._entry_mtime(entry), dict(processing))
        else:
            self._saved_processing.pop(key, None)

    # ── Optional Survey integration probe ─────────────────────────────────────
    @staticmethod
    def _probe_survey_available() -> bool:
        """Return True iff :mod:`probeflow.gui.survey` is importable.

        Used so the GUI degrades gracefully when the optional ``scanflow``
        dependency is missing (review gui-arch #17): the Tools-menu Survey
        action becomes disabled, the content-stack slot shows a help label,
        and any ``--open-survey`` startup argument logs a status-bar notice
        instead of failing.
        """
        try:
            import probeflow.gui.survey  # noqa: F401
            return True
        except Exception:
            return False

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        v_lay = QVBoxLayout(central)
        v_lay.setContentsMargins(0, 0, 0, 0)
        v_lay.setSpacing(0)

        # Body splitter
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setHandleWidth(5)
        v_lay.addWidget(self._splitter, 1)

        t = THEMES[self._theme_name]

        # ── Content stack (center+left area) ──────────────────────────────────
        self._content_stack = QStackedWidget()

        # Browse mode: inner splitter [BrowseToolPanel | ThumbnailGrid]
        self._browse_tools = BrowseToolPanel(t, self._cfg)
        self._browse_tools.setMinimumWidth(self.LEFT_SIDEBAR_MIN_W)
        self._browse_tools.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._grid         = ThumbnailGrid(t)
        self._grid.setMinimumWidth(self.CENTRAL_BROWSER_MIN_W)
        self._browse_splitter = QSplitter(Qt.Horizontal)
        self._browse_splitter.setHandleWidth(3)
        self._browse_splitter.setChildrenCollapsible(False)
        self._browse_splitter.addWidget(self._browse_tools)
        self._browse_splitter.addWidget(self._grid)
        self._browse_splitter.setStretchFactor(0, 0)
        self._browse_splitter.setStretchFactor(1, 1)

        self._conv_panel    = ConvertPanel(t, self._cfg)
        self._features_panel = FeaturesPanel(t)
        self._tv_panel       = TVPanel(t)
        self._dev_terminal   = DeveloperTerminalWidget(t)
        self._dataset_builder_panel = DatasetBuilderPanel(t, self._cfg)
        # Survey integration is optional — when ``probeflow.gui.survey``
        # cannot be imported (e.g. scanflow not installed in this venv) we
        # show a placeholder label at the Survey content stack index instead
        # of crashing the GUI launch (review gui-arch #17).
        if self._survey_available:
            from probeflow.gui.survey import SurveyPanel
            self._survey_panel = SurveyPanel()
        else:
            self._survey_panel = QLabel(
                "Survey mode is unavailable in this install.\n\n"
                "Install the optional `scanflow` dependency to enable it."
            )
            self._survey_panel.setWordWrap(True)
            self._survey_panel.setAlignment(Qt.AlignCenter)
        # NOTE: Feature Counting is not a stack page — it opens as a floating
        # FeatureCountingWindow (see _open_fc_window) so Browse stays visible.
        self._content_stack.addWidget(self._browse_splitter)        # idx 0 browse
        self._content_stack.addWidget(self._conv_panel)             # idx 1 convert
        self._content_stack.addWidget(self._features_panel)         # idx 2 features
        self._content_stack.addWidget(self._tv_panel)               # idx 3 tv
        self._content_stack.addWidget(self._dev_terminal)           # idx 4 dev
        self._content_stack.addWidget(self._survey_panel)           # idx 5 survey
        self._content_stack.addWidget(self._dataset_builder_panel)  # idx 6 dataset_builder
        self._splitter.addWidget(self._content_stack)

        # ── Right: sidebar stack ───────────────────────────────────────────────
        self._sidebar_stack    = QStackedWidget()
        self._sidebar_stack.setMinimumWidth(self.RIGHT_INSPECTOR_MIN_W)
        self._sidebar_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._browse_info      = BrowseInfoPanel(t, self._cfg)
        self._convert_sidebar  = ConvertSidebar(t, self._cfg)
        self._features_sidebar = FeaturesSidebar(t)
        self._tv_sidebar       = TVSidebar(t)
        self._dev_sidebar      = _DevSidebar(t)
        self._dataset_builder_sidebar = DatasetBuilderSidebar(t)
        self._sidebar_stack.addWidget(self._browse_info)        # idx 0
        self._sidebar_stack.addWidget(self._convert_sidebar)    # idx 1
        self._sidebar_stack.addWidget(self._features_sidebar)   # idx 2
        self._sidebar_stack.addWidget(self._tv_sidebar)         # idx 3
        self._sidebar_stack.addWidget(self._dev_sidebar)        # idx 4
        # Placeholder sidebar for Survey mode — metadata is shown in the main
        # panel itself, so the right column is just a small hint label.
        _survey_sidebar_placeholder = QWidget()
        _ssp_lay = QVBoxLayout(_survey_sidebar_placeholder)
        _ssp_lay.setContentsMargins(8, 8, 8, 8)
        _ssp_lay.addWidget(QLabel(
            "<b>Survey mode</b><br><br>"
            "Click a feature in the list to view its details. "
            "Use the Process button to open the .dat in the image viewer, "
            "then Save polished PNG to upgrade its slide image."
        ))
        _ssp_lay.itemAt(0).widget().setWordWrap(True)
        _ssp_lay.addStretch(1)
        self._sidebar_stack.addWidget(_survey_sidebar_placeholder)  # idx 5
        self._sidebar_stack.addWidget(self._dataset_builder_sidebar)  # idx 6
        self._splitter.addWidget(self._sidebar_stack)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._apply_default_splitter_sizes()

        # Features tab plumbing — worker dispatch is owned by the controller.
        self._features_pool = QThreadPool.globalInstance()
        self._features_preview_pool = QThreadPool()
        self._features_preview_pool.setMaxThreadCount(1)

        # Floating Feature Counting window (lazy-created on first open)
        self._fc_window = None

        # One application-level feature-set store shared across image viewers,
        # Feature Counting, and file imports, so points from any source can be
        # pooled together in Particle Statistics.
        from probeflow.measurements.feature_sets import FeatureSetStore
        self._feature_set_store = FeatureSetStore()

        # TV-denoise tab plumbing
        self._tv_pool    = QThreadPool.globalInstance()
        self._tv_signals = _TVWorkerSignals()
        self._tv_signals.finished.connect(self._on_tv_finished)
        self._tv_sidebar.load_from_browse_requested.connect(
            self._on_tv_load_from_browse)
        self._tv_sidebar.run_requested.connect(self._on_tv_run)
        self._tv_sidebar.revert_requested.connect(self._on_tv_revert)
        self._tv_sidebar.save_png_requested.connect(self._on_tv_save_png)

        # Wire signals
        self._browse_tools.open_folder_requested.connect(self._open_browse_folder)
        self._grid.entry_selected.connect(self._on_entry_select)
        self._grid.selection_changed.connect(self._on_selection_changed)
        self._grid.view_requested.connect(self._open_viewer)
        self._grid.card_context_action.connect(self._on_card_context_action)
        self._grid.folder_changed.connect(self._on_grid_folder_changed)
        self._grid.folder_filter_started.connect(self._on_folder_filter_started)
        self._grid.folder_filter_finished.connect(self._on_folder_filter_finished)
        self._browse_tools.colormap_changed.connect(self._on_thumbnail_colormap_changed)
        self._browse_tools.thumbnail_align_changed.connect(self._on_thumbnail_align_changed)
        self._browse_tools.map_spectra_requested.connect(self._on_map_spectra)
        self._browse_tools.overlay_spectra_requested.connect(self._on_overlay_selected_spectra)
        self._browse_tools.filter_changed.connect(self._on_filter_changed)
        self._browse_tools.folder_filter_changed.connect(self._on_folder_filter_changed)
        self._browse_tools.export_filtered_requested.connect(self._on_export_filtered_folder)
        self._browse_tools.thumbnail_channel_changed.connect(self._on_thumbnail_channel_changed)
        self._browse_tools.thumbnail_size_changed.connect(self._on_thumbnail_size_changed)
        self._browse_tools.open_fc_window_requested.connect(self._open_fc_window)
        # Apply saved thumbnail size preference.
        saved_size = self._cfg.get("thumbnail_size", "large")
        if saved_size != "large":
            self._grid.set_thumbnail_size(saved_size)
        # Sync initial filter state from the toolbar into the grid so the
        # two agree even before the first folder is opened.
        self._grid.apply_filter(self._browse_tools.get_filter_mode())
        self._grid.set_folder_filter_state(self._browse_tools.get_folder_filter_state())
        self._convert_sidebar.run_btn.clicked.connect(self._run)
        self._conv_panel.input_entry.textChanged.connect(self._update_count)

        self._features_panel.go_to_browse_requested.connect(
            lambda: self._switch_mode("browse"))
        self._features_sidebar.load_from_browse_requested.connect(
            self._on_features_load_from_browse)
        # All other sidebar signals are wired by the controller.
        self._features_ctrl = FeatureCountingController(
            self._features_panel, self._features_sidebar,
            self._features_pool,
            status_cb=lambda msg: self._status_bar.showMessage(msg),
            preview_pool=self._features_preview_pool,
            parent_widget=self,
            feature_set_store=self._feature_set_store,
        )
        self._features_ctrl.open_particle_statistics_requested.connect(
            self._on_open_particle_statistics_from_features)
        self._dataset_builder_panel.counts_changed.connect(
            self._dataset_builder_sidebar.set_counts)
        # Status bar
        self._status_bar = QStatusBar()
        self._status_bar.setFont(ui_font(10))
        self.setStatusBar(self._status_bar)
        self._dataset_builder_panel.status_message.connect(self._status_bar.showMessage)
        self._status_bar.showMessage("Open a folder to browse scans")

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        self._mode_actions: dict[str, QAction] = {}
        self._font_size_actions: dict[str, QAction] = {}
        self._theme_actions: dict[str, QAction] = {}
        self._thumbnail_cmap_actions: dict[str, QAction] = {}
        self._thumbnail_channel_actions: dict[str, QAction] = {}
        self._thumbnail_align_actions: dict[str, QAction] = {}

        self._mode_action_group = QActionGroup(self)
        self._mode_action_group.setExclusive(True)
        self._font_size_action_group = QActionGroup(self)
        self._font_size_action_group.setExclusive(True)
        self._theme_action_group = QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        self._thumbnail_cmap_action_group = QActionGroup(self)
        self._thumbnail_cmap_action_group.setExclusive(True)
        self._thumbnail_channel_action_group = QActionGroup(self)
        self._thumbnail_channel_action_group.setExclusive(True)
        self._thumbnail_align_action_group = QActionGroup(self)
        self._thumbnail_align_action_group.setExclusive(True)

        def _mode_action(menu, text: str, mode: str, shortcut: str | None = None):
            action = QAction(text, self)
            action.setCheckable(True)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(
                lambda _checked=False, value=mode: self._switch_mode(value)
            )
            self._mode_action_group.addAction(action)
            self._mode_actions[mode] = action
            menu.addAction(action)
            return action

        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open folder...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._menu_open_folder)
        file_menu.addAction(open_action)
        refresh_action = QAction("Refresh folder", self)
        refresh_action.setShortcut(QKeySequence("F5"))
        refresh_action.setToolTip(
            "Rescan the current Browse folder for new files.\n"
            "Use this when the STM has written new scans while ProbeFlow is open.")
        refresh_action.triggered.connect(lambda: self._grid.refresh())
        file_menu.addAction(refresh_action)
        file_menu.addSeparator()
        restart_action = QAction("Restart ProbeFlow", self)
        restart_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        restart_action.setToolTip(
            "Relaunch ProbeFlow — picks up any code changes immediately "
            "(useful during development with an editable pip install)")
        restart_action.triggered.connect(self._restart_app)
        file_menu.addAction(restart_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # All main-window workspaces live in one menu so the app's structure is
        # visible at a glance; Ctrl+1..5 step through them in menu order.
        # Feature Counting is deliberately NOT here — it opens as a floating
        # tool window (Tools menu), not a workspace page.
        workspace_menu = menu_bar.addMenu("Workspace")
        _mode_action(workspace_menu, "Browse", "browse", "Ctrl+1")
        _mode_action(workspace_menu, "Convert Createc .dat to .sxm", "convert", "Ctrl+2")
        _mode_action(workspace_menu, "Feature Counting", "features")
        _mode_action(workspace_menu, "TV denoise", "tv", "Ctrl+4")
        survey_action = _mode_action(
            workspace_menu, "Survey (ScanFlow campaign)", "survey", "Ctrl+Shift+S"
        )
        if not self._survey_available:
            survey_action.setEnabled(False)
            survey_action.setToolTip(
                "Survey mode requires the optional `scanflow` dependency."
            )
        _mode_action(workspace_menu, "Developer tools", "dev", "Ctrl+5")

        view_menu = menu_bar.addMenu("View")
        reset_layout_action = QAction("Reset window layout", self)
        reset_layout_action.triggered.connect(self._reset_window_layout)
        view_menu.addAction(reset_layout_action)
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("Theme")
        for key, label, _is_dark in THEME_PRESETS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, name=key: self._set_theme(name)
            )
            self._theme_action_group.addAction(action)
            self._theme_actions[key] = action
            theme_menu.addAction(action)
        # Ctrl+Shift+T flips between the base dark and light themes.
        self._theme_toggle_shortcut = QShortcut(QKeySequence("Ctrl+Shift+T"), self)
        self._theme_toggle_shortcut.activated.connect(self._toggle_theme)
        text_menu = view_menu.addMenu("Text size")
        for label in GUI_FONT_SIZES:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=label: self._on_gui_font_size_changed(value)
            )
            self._font_size_action_group.addAction(action)
            self._font_size_actions[label] = action
            text_menu.addAction(action)
        cmap_menu = view_menu.addMenu("Thumbnail colourmap")
        for label in ("Gray", "Viridis", "Inferno", "Magma", "Plasma", "Cividis"):
            if label not in CMAP_KEY:
                continue
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=label: self._set_thumbnail_colormap(value)
            )
            self._thumbnail_cmap_action_group.addAction(action)
            self._thumbnail_cmap_actions[label] = action
            cmap_menu.addAction(action)
        channel_menu = view_menu.addMenu("Thumbnail channel")
        for label in THUMBNAIL_CHANNEL_OPTIONS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=label: self._set_thumbnail_channel(value)
            )
            self._thumbnail_channel_action_group.addAction(action)
            self._thumbnail_channel_actions[label] = action
            channel_menu.addAction(action)
        # Row alignment is a thumbnail *display* setting like colourmap and
        # channel, so it lives with them here (it used to be a one-item
        # top-level "Processing" menu, which wrongly implied image processing).
        align_menu = view_menu.addMenu("Thumbnail row alignment")
        for label in ("None", "Mean", "Median"):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=label: self._set_thumbnail_align(value)
            )
            self._thumbnail_align_action_group.addAction(action)
            self._thumbnail_align_actions[label] = action
            align_menu.addAction(action)

        tools_menu = menu_bar.addMenu("Tools")
        # Feature Counting opens a floating window (not a workspace page), so
        # it is a plain action here rather than a checkable mode entry.
        fc_action = QAction("Feature Counting…", self)
        fc_action.setShortcut(QKeySequence("Ctrl+3"))
        fc_action.setToolTip(
            "Open the floating Feature Counting window — segmentation, "
            "template matching, classification and lattice extraction."
        )
        fc_action.triggered.connect(self._open_fc_window)
        tools_menu.addAction(fc_action)
        map_action = QAction("Map Spectra to Images...", self)
        map_action.triggered.connect(self._on_map_spectra)
        tools_menu.addAction(map_action)
        tools_menu.addSeparator()
        _mode_action(tools_menu, "Feature Counting…", "features")
        _mode_action(tools_menu, "TV denoise", "tv", "Ctrl+4")
        _mode_action(tools_menu, "Dataset Builder", "dataset_builder", "Ctrl+Shift+D")
        tools_menu.addSeparator()
        survey_action = _mode_action(
            tools_menu, "Survey (ScanFlow campaign)", "survey", "Ctrl+Shift+S"
        )
        if not self._survey_available:
            survey_action.setEnabled(False)
            survey_action.setToolTip(
                "Survey mode requires the optional `scanflow` dependency."
            )
        tools_menu.addSeparator()
        _mode_action(tools_menu, "Developer tools", "dev", "Ctrl+5")
        prefs_action = QAction("Preferences...", self)
        prefs_action.setEnabled(False)
        tools_menu.addAction(prefs_action)

        help_menu = menu_bar.addMenu("Help")
        definitions_action = QAction("Definitions", self)
        definitions_action.setShortcut(QKeySequence("Ctrl+6"))
        definitions_action.triggered.connect(self._show_definitions)
        help_menu.addAction(definitions_action)
        help_menu.addSeparator()
        github_action = QAction("GitHub", self)
        github_action.triggered.connect(lambda: _open_url(GITHUB_URL))
        help_menu.addAction(github_action)
        report_action = QAction("Report issue", self)
        report_action.triggered.connect(lambda: _open_url(f"{GITHUB_URL}/issues"))
        help_menu.addAction(report_action)
        help_menu.addSeparator()
        about_action = QAction("About ProbeFlow", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        self._sync_menu_actions()
        self._install_quit_drain()

    def _apply_default_splitter_sizes(self) -> None:
        center_default = max(
            self.CENTRAL_BROWSER_MIN_W,
            self.width()
            - self.LEFT_SIDEBAR_DEFAULT_W
            - self.RIGHT_INSPECTOR_DEFAULT_W,
        )
        self._browse_splitter.setSizes([
            self.LEFT_SIDEBAR_DEFAULT_W,
            center_default,
        ])
        self._splitter.setSizes([
            self.LEFT_SIDEBAR_DEFAULT_W + center_default,
            self.RIGHT_INSPECTOR_DEFAULT_W,
        ])

    def _apply_default_metadata_table_columns(self) -> None:
        table = getattr(getattr(self, "_browse_info", None), "meta_table", None)
        if table is None:
            return
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(72)
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        table.setColumnWidth(0, 132)

    def _restore_metadata_table_columns(self, widths: list[int] | None) -> None:
        if not widths:
            self._apply_default_metadata_table_columns()
            return
        table = getattr(getattr(self, "_browse_info", None), "meta_table", None)
        if table is None:
            return
        self._apply_default_metadata_table_columns()
        for col, width in enumerate(widths):
            if col < table.columnCount():
                table.setColumnWidth(col, max(72, int(width)))

    # Bump this when the main-window pane structure changes incompatibly
    # (e.g. a splitter pane is added or removed).  Saved layouts with a
    # mismatching version are discarded with a status-bar notice so users
    # are not silently stuck on a stale layout after an upgrade
    # (review gui-arch #18).
    LAYOUT_VERSION = 2

    def _restore_desktop_layout(self) -> None:
        layout = self._cfg.get("layout", {}).get("main_window", {})
        stored_version = layout.get("version")
        version_mismatch = (
            stored_version is not None and stored_version != self.LAYOUT_VERSION
        )

        restore_geometry_or_default(self, layout.get("geometry"), 0.88)
        state = layout.get("state")
        if state and not version_mismatch:
            try:
                self.restoreState(b64_to_qbytearray(state))
            except Exception:
                _log.warning("Could not restore saved window layout; using "
                             "defaults", exc_info=True)

        main_sizes = layout.get("splitter_sizes")
        if (not version_mismatch
                and main_sizes and len(main_sizes) == self._splitter.count()):
            self._splitter.setSizes([max(1, int(x)) for x in main_sizes])
        else:
            self._apply_default_splitter_sizes()

        browse_sizes = layout.get("browse_splitter_sizes")
        if (not version_mismatch
                and browse_sizes
                and len(browse_sizes) == self._browse_splitter.count()):
            self._browse_splitter.setSizes([max(1, int(x)) for x in browse_sizes])

        self._restore_metadata_table_columns(
            None if version_mismatch
            else layout.get("metadata_table_column_widths")
        )
        self._show_maximized_on_start = (
            False if version_mismatch else bool(layout.get("was_maximized"))
        )

        if version_mismatch:
            self._status_bar.showMessage(
                f"Window layout reset because pane structure changed "
                f"(saved v{stored_version} -> current v{self.LAYOUT_VERSION})."
            )

    def _save_desktop_layout_into(self, cfg: dict) -> None:
        layout_root = cfg.setdefault("layout", {})
        layout = layout_root.setdefault("main_window", {})
        layout["version"] = self.LAYOUT_VERSION
        layout["geometry"] = qbytearray_to_b64(self.saveGeometry())
        layout["state"] = qbytearray_to_b64(self.saveState())
        layout["was_maximized"] = self.isMaximized()
        layout["splitter_sizes"] = self._splitter.sizes()
        layout["browse_splitter_sizes"] = self._browse_splitter.sizes()

        table = getattr(getattr(self, "_browse_info", None), "meta_table", None)
        if table is not None:
            layout["metadata_table_column_widths"] = [
                table.columnWidth(i) for i in range(table.columnCount())
            ]

    def _reset_window_layout(self) -> None:
        cfg = load_config()
        if isinstance(cfg.get("layout"), dict):
            cfg["layout"].pop("main_window", None)
            cfg["layout"].pop("image_viewer", None)
        save_config(cfg)
        self._cfg = cfg
        apply_screen_fraction_geometry(self, 0.88)
        self._apply_default_splitter_sizes()
        self._apply_default_metadata_table_columns()
        self._show_maximized_on_start = False
        self._status_bar.showMessage("Window layout reset.")

    def _menu_open_folder(self) -> None:
        self._switch_mode("browse")
        self._open_browse_folder()

    def _set_theme(self, name: str) -> None:
        if name not in THEMES:
            return
        if name == self._theme_name:
            self._sync_menu_actions()
            return
        self._theme_name = name
        self._dark = theme_is_dark(name)
        self._apply_theme()
        self._sync_menu_actions()

    def _set_dark_mode(self, dark: bool) -> None:
        """Back-compat shim: pick the base dark or light preset."""
        self._set_theme("dark" if dark else "light")

    def _apply_thumbnail_setting(
        self, combo_attr: str, handler, value, transform=None
    ) -> None:
        """Sync a Browse-tools combo to ``value`` and invoke ``handler``.

        ``combo_attr`` is the attribute name on ``self._browse_tools``.
        If the combo's current text already matches ``value``, the menu
        triggered this update directly, so we run ``handler`` ourselves
        (optionally piping the value through ``transform`` first).  If
        the combo differs, ``setCurrentText`` triggers the combo's own
        signal — which already calls ``handler`` — so we skip the manual
        call here to avoid double-dispatch.
        """
        combo = getattr(self._browse_tools, combo_attr, None) \
            if hasattr(self, "_browse_tools") else None
        if combo is not None and combo.currentText() != value:
            combo.setCurrentText(value)
        else:
            handler(transform(value) if transform is not None else value)
        self._sync_menu_actions()

    def _set_thumbnail_colormap(self, label: str) -> None:
        self._apply_thumbnail_setting(
            "cmap_cb",
            self._on_thumbnail_colormap_changed,
            label,
            transform=lambda lbl: CMAP_KEY.get(lbl, DEFAULT_CMAP_KEY),
        )

    def _set_thumbnail_channel(self, channel: str) -> None:
        self._apply_thumbnail_setting(
            "thumbnail_channel_cb",
            self._on_thumbnail_channel_changed,
            channel,
        )

    def _set_thumbnail_align(self, mode: str) -> None:
        self._apply_thumbnail_setting(
            "align_rows_cb",
            self._on_thumbnail_align_changed,
            mode,
        )

    def _show_definitions(self) -> None:
        theme = THEMES[self._theme_name]
        dlg = getattr(self, "_definitions_dialog", None)
        if dlg is None:
            dlg = _DefinitionsDialog(theme, self)
            self._definitions_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    # ── Mode switching ─────────────────────────────────────────────────────────
    def _switch_mode(self, mode: str):
        self._mode = mode
        if mode == "browse":
            self._content_stack.setCurrentIndex(0)
            self._sidebar_stack.setCurrentIndex(0)
            n = len(self._grid.get_entries())
            self._status_bar.showMessage(
                f"{n} scan(s) loaded" if n else "Open a folder to browse scans")
        elif mode == "features":
            self._content_stack.setCurrentIndex(2)
            self._sidebar_stack.setCurrentIndex(2)
            self._status_bar.showMessage(
                "Feature Counting — open a scan from Browse or use the floating window")
        elif mode == "tv":
            self._content_stack.setCurrentIndex(3)
            self._sidebar_stack.setCurrentIndex(3)
            if self._tv_panel.current_array() is None:
                self._status_bar.showMessage(
                    "Pick a scan in Browse, then 'Load primary scan from Browse'")
            else:
                self._status_bar.showMessage("TV-denoise — adjust parameters and Run")
        elif mode == "dev":
            self._content_stack.setCurrentIndex(4)
            self._sidebar_stack.setCurrentIndex(4)
            self._status_bar.showMessage(
                "Developer terminal — run shell commands and Python scripts")
        elif mode == "survey":
            self._content_stack.setCurrentIndex(5)
            self._sidebar_stack.setCurrentIndex(5)
            self._status_bar.showMessage(
                "Survey mode — open a ScanFlow survey.json, polish each feature, then Export PPTX")
        elif mode == "dataset_builder":
            self._content_stack.setCurrentIndex(6)
            self._sidebar_stack.setCurrentIndex(6)
            self._status_bar.showMessage(
                "Dataset Builder - queue, propose, correct, save, export")
        else:
            self._content_stack.setCurrentIndex(1)
            self._sidebar_stack.setCurrentIndex(1)
            self._update_count(self._conv_panel.input_entry.text())
        self._update_tab_styles()

    def _update_tab_styles(self):
        self._sync_menu_actions()

    def _sync_menu_actions(self) -> None:
        if hasattr(self, "_mode_actions"):
            for mode, action in self._mode_actions.items():
                action.blockSignals(True)
                action.setChecked(self._mode == mode)
                action.blockSignals(False)
        if hasattr(self, "_theme_actions"):
            for key, action in self._theme_actions.items():
                action.blockSignals(True)
                action.setChecked(key == self._theme_name)
                action.blockSignals(False)
        if hasattr(self, "_font_size_actions"):
            for label, action in self._font_size_actions.items():
                action.blockSignals(True)
                action.setChecked(label == self._gui_font_size)
                action.blockSignals(False)
        if hasattr(self, "_thumbnail_cmap_actions"):
            cmap_label = (
                self._browse_tools.cmap_cb.currentText()
                if hasattr(self, "_browse_tools")
                else self._cfg.get("colormap", DEFAULT_CMAP_LABEL)
            )
            for label, action in self._thumbnail_cmap_actions.items():
                action.blockSignals(True)
                action.setChecked(label == cmap_label)
                action.blockSignals(False)
        if hasattr(self, "_thumbnail_channel_actions"):
            channel = (
                self._browse_tools.thumbnail_channel_cb.currentText()
                if hasattr(self, "_browse_tools")
                else THUMBNAIL_CHANNEL_DEFAULT
            )
            for label, action in self._thumbnail_channel_actions.items():
                action.blockSignals(True)
                action.setChecked(label == channel)
                action.blockSignals(False)
        if hasattr(self, "_thumbnail_align_actions"):
            align = (
                self._browse_tools.align_rows_cb.currentText()
                if hasattr(self, "_browse_tools")
                else "None"
            )
            for label, action in self._thumbnail_align_actions.items():
                action.blockSignals(True)
                action.setChecked(label == align)
                action.blockSignals(False)

    # ── Browse ─────────────────────────────────────────────────────────────────
    def _open_browse_folder(self):
        dialog = QFileDialog(self, "Open folder containing scan / .VERT files")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, False)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        if not dialog.exec():
            return
        selected = dialog.selectedFiles()
        d = selected[0] if selected else ""
        if not d:
            return
        self._switch_mode("browse")
        # Shallow navigation: the grid drives indexing. Set the new root and
        # let the grid render the immediate folder + subfolder cards.
        self._grid.set_root(Path(d))
        # New root → discard previous spec mapping; user can rebuild it.
        self._spec_image_map = {}
        self._browse_info.clear()
        self._update_browse_status()

    def _on_grid_folder_changed(self, path: Path):
        """Status-bar update + clear any selection-driven UI when navigating."""
        self._browse_info.clear()
        self._update_browse_status()

    def _format_browse_counts(self, counts) -> str:
        parts: list[str] = []
        if counts.visible_folders:
            parts.append(
                f"{counts.visible_folders} folder{'s' if counts.visible_folders != 1 else ''}"
            )
        if counts.visible_scans:
            parts.append(
                f"{counts.visible_scans} scan{'s' if counts.visible_scans != 1 else ''}"
            )
        if counts.visible_spectra:
            parts.append(
                f"{counts.visible_spectra} spec{'s' if counts.visible_spectra != 1 else ''}"
            )
        if not parts:
            parts.append("0 items")
        if counts.hidden_items:
            parts.append(f"{counts.hidden_items} hidden")
        return ", ".join(parts)

    def _update_browse_status(self):
        entries = self._grid.get_entries()
        n_folders = sum(1 for e in entries if isinstance(e, FolderEntry))
        n_sxm     = sum(1 for e in entries if isinstance(e, SxmFile))
        n_vert    = sum(1 for e in entries if isinstance(e, VertFile))
        self._n_loaded = n_sxm + n_vert
        cur = self._grid.current_dir()
        parts: list[str] = []
        if n_folders:
            parts.append(f"{n_folders} folder{'s' if n_folders != 1 else ''}")
        if n_sxm:
            parts.append(f"{n_sxm} scan{'s' if n_sxm != 1 else ''}")
        if n_vert:
            parts.append(f"{n_vert} spec{'s' if n_vert != 1 else ''}")
        desc = ", ".join(parts) if parts else "0 items"
        loc = cur.name if cur else "?"
        self._status_bar.showMessage(
            f"{loc}: {desc} — Double-click a folder to navigate, a scan to view")

    def _on_entry_select(self, entry):
        if isinstance(entry, VertFile):
            self._browse_info.show_vert_entry(entry)
            n_sel = len(self._grid.get_selected())
            sweep = entry.sweep_type.replace("_", " ")
            self._status_bar.showMessage(
                f"{entry.stem}  |  {sweep}  |  {entry.n_points} pts  |  "
                f"{n_sel} selected / {self._n_loaded} total  |  Double-click to view")
        else:
            cmap_key, _, proc = self._grid.get_card_state(entry)
            self._browse_info.show_entry(entry, cmap_key, proc)
            n_sel = len(self._grid.get_selected())
            self._status_bar.showMessage(
                f"{entry.stem}  |  {entry.Nx}×{entry.Ny} px  |  "
                f"{n_sel} selected / {self._n_loaded} total  |  Double-click to view full size")

    def _on_selection_changed(self, n_selected: int):
        self._browse_tools.update_selection_hint(n_selected)

    def _on_filter_changed(self, mode: str):
        self._grid.apply_filter(mode)
        entries = self._grid.get_entries()
        n_sxm  = sum(1 for e in entries if isinstance(e, SxmFile))
        n_vert = sum(1 for e in entries if isinstance(e, VertFile))
        img_word  = "image"    if n_sxm  == 1 else "images"
        spec_word = "spectrum" if n_vert == 1 else "spectra"
        if mode == "images":
            msg = f"{n_sxm} {img_word}  ({n_vert} {spec_word} hidden)"
        elif mode == "spectra":
            msg = f"{n_vert} {spec_word}  ({n_sxm} {img_word} hidden)"
        else:
            msg = f"{n_sxm} {img_word}, {n_vert} {spec_word}"
        self._status_bar.showMessage(msg)

    # Updated browse-filter/status handlers for metadata folder filtering.
    def _update_browse_status(self):
        counts = self._grid.get_filter_counts()
        self._n_loaded = counts.visible_scans + counts.visible_spectra
        cur = self._grid.current_dir()
        desc = self._format_browse_counts(counts)
        loc = cur.name if cur else "?"
        self._status_bar.showMessage(
            f"{loc}: {desc} â€” Double-click a folder to navigate, a scan to view")

    def _on_entry_select(self, entry):
        if entry is None:
            self._browse_info.clear()
            self._update_browse_status()
            return
        if isinstance(entry, VertFile):
            self._browse_info.show_vert_entry(entry)
            n_sel = len(self._grid.get_selected())
            sweep = entry.sweep_type.replace("_", " ")
            self._status_bar.showMessage(
                f"{entry.stem}  |  {sweep}  |  {entry.n_points} pts  |  "
                f"{n_sel} selected / {self._n_loaded} total  |  Double-click to view")
            return
        cmap_key, _, proc = self._grid.get_card_state(entry)
        self._browse_info.show_entry(entry, cmap_key, proc)
        n_sel = len(self._grid.get_selected())
        self._status_bar.showMessage(
            f"{entry.stem}  |  {entry.Nx}Ã—{entry.Ny} px  |  "
            f"{n_sel} selected / {self._n_loaded} total  |  Double-click to view full size")

    def _on_filter_changed(self, mode: str):
        self._grid.apply_filter(mode)
        self._update_browse_status()

    def _on_folder_filter_changed(self, state) -> None:
        self._grid.set_folder_filter_state(state)

    def _on_folder_filter_started(self, folder_name: str) -> None:
        self._status_bar.showMessage(f"Filtering {folder_name}...")

    def _on_folder_filter_finished(self, counts) -> None:
        cur = self._grid.current_dir()
        loc = cur.name if cur else "?"
        self._n_loaded = counts.visible_scans + counts.visible_spectra
        self._status_bar.showMessage(f"{loc}: {self._format_browse_counts(counts)}")

    def _on_export_filtered_folder(self) -> None:
        current_dir = self._grid.current_dir()
        if current_dir is None:
            self._status_bar.showMessage("Open a browse folder first.")
            return
        entries = self._grid.get_visible_scan_entries()
        if not entries:
            self._status_bar.showMessage("No matching scan files to export.")
            return
        dest = QFileDialog.getExistingDirectory(
            self,
            "Export filtered folder",
            str(Path.home()),
        )
        if not dest:
            return
        worker = FilteredFolderExportWorker(
            [entry.path for entry in entries],
            dest,
        )
        worker.signals.finished.connect(self._on_export_filtered_finished)
        worker.signals.failed.connect(self._on_export_filtered_failed)
        self._status_bar.showMessage(
            f"Exporting {len(entries)} filtered scan file{'s' if len(entries) != 1 else ''}..."
        )
        QThreadPool.globalInstance().start(worker)

    def _on_export_filtered_finished(self, result) -> None:
        self._status_bar.showMessage(
            f"Exported {result.copied} scan file{'s' if result.copied != 1 else ''} "
            f"to {result.destination.name} ({result.collisions} skipped, {result.errors} errors)"
        )

    def _on_export_filtered_failed(self, message: str) -> None:
        self._status_bar.showMessage(f"Filtered export failed: {message}")

    def _on_thumbnail_channel_changed(self, channel: str):
        n = self._grid.set_thumbnail_channel(channel)
        if n == 0:
            self._status_bar.showMessage(f"Thumbnail channel: {channel}")
        else:
            self._status_bar.showMessage(
                f"Thumbnail channel: {channel} — queued {n} image thumbnail"
                f"{'s' if n != 1 else ''}")
        self._sync_menu_actions()

    def _on_thumbnail_colormap_changed(self, cmap_key: str):
        n = self._grid.set_thumbnail_colormap(cmap_key)
        label = next((l for l, k in CMAP_KEY.items() if k == cmap_key), cmap_key)
        if n == 0:
            self._status_bar.showMessage(f"Thumbnail colormap: {label}")
        else:
            self._status_bar.showMessage(
                f"Thumbnail colormap: {label} — queued {n} image thumbnail"
                f"{'s' if n != 1 else ''}")
        self._refresh_primary_channel_previews()
        self._sync_menu_actions()

    def _on_thumbnail_align_changed(self, mode: str):
        n = self._grid.set_thumbnail_align_rows(mode)
        label = mode if mode in ("Median", "Mean") else "None"
        if n == 0:
            self._status_bar.showMessage(f"Thumbnail align rows: {label}")
        else:
            self._status_bar.showMessage(
                f"Thumbnail align rows: {label} — queued {n} image thumbnail"
                f"{'s' if n != 1 else ''}")
        self._sync_menu_actions()

    def _on_thumbnail_size_changed(self, name: str) -> None:
        self._grid.set_thumbnail_size(name)

    def _refresh_primary_channel_previews(self):
        entry = self._grid.get_primary_entry()
        if entry and isinstance(entry, SxmFile):
            self._browse_info.load_channels(
                entry, self._grid.thumbnail_colormap(), processing=None)

    def _on_map_spectra(self):
        """Open the folder-level spec→image mapping dialog."""
        entries = self._grid.get_entries()
        sxm_entries  = [e for e in entries if isinstance(e, SxmFile)]
        vert_entries = [e for e in entries if isinstance(e, VertFile)]
        if not vert_entries:
            self._status_bar.showMessage("No spectroscopy files in the current folder.")
            return
        if not sxm_entries:
            self._status_bar.showMessage("No images loaded — open a folder with .sxm files first.")
            return
        dlg = SpecMappingDialog(sxm_entries, vert_entries, self._spec_image_map, self)
        if dlg.exec() == QDialog.Accepted:
            new_map = dlg.get_mapping()
            self._spec_image_map.clear()
            self._spec_image_map.update(new_map)
            self._status_bar.showMessage(
                f"Spec mapping updated: {len(new_map)} of "
                f"{len(vert_entries)} spectra assigned.")

    def _on_overlay_selected_spectra(self):
        entries = [
            e for e in self._grid.get_selected_entries()
            if isinstance(e, VertFile)
        ]
        if len(entries) < 2:
            self._status_bar.showMessage(
                "Select two or more spectra with Ctrl-click before overlaying.")
            return
        t = THEMES[self._theme_name]
        dlg = SpecOverlayDialog(entries, t, self)
        dlg.exec()

    def _on_card_context_action(self, entry, action: str):
        """Dispatch ScanCard right-click actions (Send to Features, export, show metadata)."""
        if action == "features":
            # Open (or raise) the floating Feature Counting window and load
            # the scan into it off-thread — same path as its own "Load from
            # Browse" button, so saved viewer processing is applied too.
            self._open_fc_window()
            plane_idx = self._fc_window._sidebar.plane_index()
            self._fc_window._sidebar.set_status("Loading scan…")
            worker = _ScanLoadWorker(
                self._load_scan_plane_for_analysis, entry, plane_idx)
            worker.signals.finished.connect(
                lambda result, error, e=entry: self._on_fc_load_finished(e, result, error)
            )
            QThreadPool.globalInstance().start(worker)
            self._status_bar.showMessage(f"{entry.stem} → Feature Counting")

        elif action == "export_metadata_csv":
            try:
                _scan = load_scan(entry.path)
                header = dict(getattr(_scan, "header", {}) or {})
            except Exception as exc:
                self._status_bar.showMessage(f"Could not read scan: {exc}")
                return
            if not header:
                self._status_bar.showMessage("No metadata to export")
                return
            out_path, _ = QFileDialog.getSaveFileName(
                self, "Export metadata as CSV",
                str(Path.home() / f"{entry.stem}_metadata.csv"),
                "CSV files (*.csv)")
            if not out_path:
                return
            try:
                import csv
                with open(out_path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["key", "value"])
                    for k in sorted(header):
                        w.writerow([k, header[k]])
                self._status_bar.showMessage(f"Metadata → {out_path}")
            except Exception as exc:
                self._status_bar.showMessage(f"Export error: {exc}")

        elif action == "show_metadata":
            try:
                _scan = load_scan(entry.path)
                header = dict(getattr(_scan, "header", {}) or {})
            except Exception as exc:
                self._status_bar.showMessage(f"Could not read scan: {exc}")
                return
            dlg = QDialog(self)
            dlg.setWindowTitle(f"Metadata — {entry.stem}")
            dlg.resize(560, 600)
            v = QVBoxLayout(dlg)
            tbl = QTableWidget(len(header), 2, dlg)
            tbl.setHorizontalHeaderLabels(["Key", "Value"])
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tbl.verticalHeader().setVisible(False)
            tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
            tbl.setFont(ui_font(9))
            for row, k in enumerate(sorted(header)):
                tbl.setItem(row, 0, QTableWidgetItem(str(k)))
                tbl.setItem(row, 1, QTableWidgetItem(str(header[k])))
            v.addWidget(tbl)
            close_btn = QPushButton("Close", dlg)
            close_btn.clicked.connect(dlg.accept)
            v.addWidget(close_btn)
            dlg.exec()

    # ── Floating Feature Counting window ──────────────────────────────────────

    def _open_fc_window(self) -> None:
        """Open (or raise) the floating Feature Counting window."""
        if self._fc_window is None:
            from probeflow.gui.features.window import FeatureCountingWindow
            theme = THEMES[self._theme_name]
            self._fc_window = FeatureCountingWindow(
                parent=None, theme=theme, feature_set_store=self._feature_set_store
            )
            self._fc_window.load_from_browse_needed.connect(
                self._on_fc_load_from_browse)
            self._fc_window.open_particle_statistics_needed.connect(
                self._on_open_particle_statistics_from_features)
        self._fc_window.show()
        self._fc_window.raise_()
        self._fc_window.activateWindow()

    def _load_scan_plane_for_analysis(
        self, entry, plane_idx: int
    ) -> tuple:
        """Load a scan plane and apply any saved viewer processing.

        Returns ``(arr, px_m, px_x_m, px_y_m, actual_plane_idx, scan)`` or
        raises.  The returned array is the *processed* version — identical to
        what the user last saw in the image viewer — so Feature Counting and
        TV-denoise work on the same data the user inspected.  ``scan`` is the
        loaded :class:`Scan`, carried through so analysis exports can record the
        same provenance the CLI does.
        """
        _scan = load_scan(entry.path)
        if plane_idx >= _scan.n_planes:
            plane_idx = 0
        raw_arr = _scan.planes[plane_idx]
        w_m, h_m = _scan.scan_range_m
        if raw_arr is None:
            raise ValueError("Scan returned no array for that plane.")

        arr = np.asarray(raw_arr, dtype=np.float64)
        analysis_range = (float(w_m), float(h_m))
        processing_state = None

        # Apply saved processing (align rows, background, etc.) so Feature
        # Counting sees the same image the user processed in the viewer.  Use
        # the calibrated state walker, not the display thumbnail helper, so any
        # shape-changing step updates scan_range_m and therefore the pixel sizes
        # exported with downstream feature JSON.
        saved_proc = self._saved_processing_get(entry)
        if saved_proc:
            try:
                from probeflow.processing.gui_adapter import processing_state_from_gui
                from probeflow.processing.state import apply_processing_state_with_calibration

                processing_state = processing_state_from_gui(saved_proc)
                if processing_state.steps:
                    # Resolve roi / mask scope steps against persisted sidecars
                    # so scoped local filters replay here too (review:
                    # mask-scope replay not connected through feature counting).
                    roi_set = mask_set = None
                    try:
                        from probeflow.io.roi_sidecar import load_roi_set_sidecar
                        roi_set, _ = load_roi_set_sidecar(entry.path, missing_ok=True)
                    except Exception:
                        roi_set = None
                    try:
                        from probeflow.io.mask_sidecar import load_mask_set_sidecar
                        mask_set, _ = load_mask_set_sidecar(entry.path, missing_ok=True)
                    except Exception:
                        mask_set = None
                    arr, new_range = apply_processing_state_with_calibration(
                        arr,
                        processing_state,
                        roi_set,
                        mask_set=mask_set,
                        scan_range_m=analysis_range,
                    )
                    if new_range is not None:
                        analysis_range = (float(new_range[0]), float(new_range[1]))
            except Exception:
                processing_state = None
                arr = np.asarray(raw_arr, dtype=np.float64)
                analysis_range = (float(w_m), float(h_m))
                pass   # fall back to raw if processing fails

        Ny, Nx = arr.shape
        if Nx > 0 and Ny > 0 and analysis_range[0] > 0 and analysis_range[1] > 0:
            px_x_m = float(analysis_range[0] / Nx)
            px_y_m = float(analysis_range[1] / Ny)
            px_m = float(np.sqrt(px_x_m * px_y_m))
        else:
            px_x_m = px_y_m = px_m = 1e-10

        # Hand Features a scan that describes the exact plane it will analyze.
        # A one-plane wrapper avoids scan.dims reporting plane 0 when the user
        # analyzed another channel, and keeps exported pixels/range in lockstep
        # with the processed array.
        from probeflow.core.scan_model import Scan

        plane_name = (
            _scan.plane_names[plane_idx]
            if plane_idx < len(_scan.plane_names) else f"plane {plane_idx}"
        )
        plane_unit = (
            _scan.plane_units[plane_idx]
            if plane_idx < len(_scan.plane_units) else ""
        )
        plane_synth = (
            bool(_scan.plane_synthetic[plane_idx])
            if plane_idx < len(_scan.plane_synthetic) else False
        )
        analysis_scan = Scan(
            planes=[np.asarray(arr, dtype=np.float64).copy()],
            plane_names=[plane_name],
            plane_units=[plane_unit],
            plane_synthetic=[plane_synth],
            header=dict(_scan.header or {}),
            scan_range_m=analysis_range,
            source_path=_scan.source_path,
            source_format=_scan.source_format,
            processing_state=_scan.processing_state,
            experiment_metadata=dict(getattr(_scan, "experiment_metadata", {}) or {}),
            warnings=tuple(getattr(_scan, "warnings", ()) or ()),
        )
        if processing_state is not None and processing_state.steps:
            analysis_scan.record_processing_state(processing_state)

        return arr, px_m, px_x_m, px_y_m, plane_idx, analysis_scan

    def _on_open_particle_statistics_from_features(self, scan_context, set_id: str) -> None:
        """Open (or raise) Particle Statistics for a set sent from Feature Counting."""
        from probeflow.gui.dialogs.particle_statistics import ParticleStatisticsDialog

        theme = THEMES[self._theme_name]
        dlg = getattr(self, "_features_particle_statistics_dialog", None)
        try:
            if dlg is not None:
                dlg.isVisible()
        except RuntimeError:
            dlg = None
        if dlg is None:
            dlg = ParticleStatisticsDialog(
                scan=scan_context,
                feature_set_store=self._feature_set_store,
                theme=theme,
                initial_mode="real",
                parent=None,
            )
            self._features_particle_statistics_dialog = dlg
            try:
                dlg.destroyed.connect(
                    lambda _obj=None: setattr(self, "_features_particle_statistics_dialog", None)
                )
            except Exception:
                pass
        else:
            dlg.refresh_probe_context(
                scan=scan_context, feature_set_store=self._feature_set_store
            )
            dlg.set_current_mode("real")
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        if set_id:
            dlg.select_feature_set(set_id)

    def _on_fc_load_from_browse(self) -> None:
        """Bridge: read Browse selection → load into the floating FC window (off-thread)."""
        from probeflow.gui.models import VertFile
        if self._fc_window is None:
            return
        entry = self._grid.get_primary_entry()
        if not entry:
            self._fc_window._sidebar.set_status(
                "Select a scan in the Browse tab first.")
            return
        if isinstance(entry, VertFile):
            self._fc_window._sidebar.set_status(
                "Selected entry is not a topography scan.")
            return
        plane_idx = self._fc_window._sidebar.plane_index()
        self._fc_window._sidebar.set_status("Loading scan…")
        worker = _ScanLoadWorker(self._load_scan_plane_for_analysis, entry, plane_idx)
        worker.signals.finished.connect(
            lambda result, error, e=entry: self._on_fc_load_finished(e, result, error)
        )
        QThreadPool.globalInstance().start(worker)

    def _on_fc_load_finished(self, entry, result, error: str) -> None:
        """Receive the background-loaded scan and hand it to the FC window."""
        if self._fc_window is None:
            return
        if error:
            self._fc_window._sidebar.set_status(f"Could not read scan: {error}")
            return
        arr, px_m, px_x_m, px_y_m, plane_idx, scan = result
        self._fc_window.load_entry(entry, plane_idx, arr, px_m, px_x_m, px_y_m, scan=scan)

    # ── TV-denoise tab handlers ────────────────────────────────────────────────
    def _on_tv_load_from_browse(self):
        entry = self._grid.get_primary_entry()
        if not entry:
            self._tv_sidebar.set_status("Select a scan in the Browse tab first.")
            self._status_bar.showMessage("Pick a scan in Browse to load it into TV-denoise")
            return
        if not entry or isinstance(entry, VertFile):
            self._tv_sidebar.set_status("Selected entry is not a topography scan.")
            return
        plane_idx = self._tv_sidebar.plane_index()
        try:
            arr, px_m, _px_x_m, _px_y_m, plane_idx, _scan = \
                self._load_scan_plane_for_analysis(entry, plane_idx)
        except Exception as exc:
            self._tv_sidebar.set_status(f"Could not read scan: {exc}")
            return
        self._tv_panel.load_entry(entry, plane_idx, arr, px_m)
        self._tv_sidebar.set_status(
            f"Loaded {entry.stem} (plane {plane_idx}). Adjust parameters and Run.")

    def _on_tv_run(self):
        arr = self._tv_panel.current_array()
        if arr is None:
            self._tv_sidebar.set_status("Load a scan first.")
            return
        params = self._tv_sidebar.params()
        self._tv_sidebar.set_running(True)
        self._tv_sidebar.set_status(f"Running TV-denoise ({params['method']})…")
        worker = _TVWorker(arr, params, self._tv_signals)
        self._tv_pool.start(worker)

    def _on_tv_finished(self, result, error: str):
        self._tv_sidebar.set_running(False)
        if error:
            self._tv_sidebar.set_status(f"TV-denoise failed: {error}")
            self._status_bar.showMessage(f"TV-denoise failed: {error}")
            return
        self._tv_panel.set_denoised(result)
        self._tv_sidebar.set_status("Done. Save the denoised PNG, or Run again.")

    def _on_tv_revert(self):
        self._tv_panel.set_denoised(None)
        self._tv_sidebar.set_status("Reverted to original.")

    def _on_tv_save_png(self):
        out = self._tv_panel.current_denoised()
        if out is None:
            self._tv_sidebar.set_status("Run TV-denoise first.")
            return
        entry = self._tv_panel.current_entry()
        plane_idx = self._tv_panel.current_plane_idx()
        suggested = (Path.home() /
                     f"{entry.stem if entry else 'tv'}_p{plane_idx}_tvdenoise.png")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save denoised PNG", str(suggested), "PNG (*.png)")
        if not out_path:
            return

        from PySide6.QtWidgets import QMessageBox as _QMB
        reply = _QMB.question(
            self, "Scale bar",
            "Include scale bar in the exported PNG?",
            _QMB.Yes | _QMB.No, _QMB.Yes,
        )
        add_scalebar = (reply == _QMB.Yes)

        try:
            from probeflow.processing import export_png
            from probeflow.io.writers.png import lut_from_matplotlib
            px_m = self._tv_panel.current_pixel_size()
            Ny, Nx = out.shape
            scan_range_m = (px_m * Nx, px_m * Ny)
            export_png(
                out, out_path, "gray", 1.0, 99.0,
                lut_fn=lut_from_matplotlib,
                scan_range_m=scan_range_m,
                add_scalebar=add_scalebar,
                scalebar_unit="nm",
                scalebar_pos="bottom-right",
            )
            self._tv_sidebar.set_status(f"Saved → {out_path}")
            self._status_bar.showMessage(f"Saved {out_path}")
        except Exception as exc:
            self._tv_sidebar.set_status(f"Save failed: {exc}")

    def _open_viewer(self, entry):
        t = THEMES[self._theme_name]
        is_spec = isinstance(entry, VertFile)
        if is_spec:
            dlg = SpecViewerDialog(entry, t, self)
        else:
            cmap_key, clip, proc = self._grid.get_card_state(entry)
            # Restore any processing the user applied last time this scan was open.
            saved = self._saved_processing_get(entry)
            if saved:
                proc = dict(saved)
            sxm_entries = [e for e in self._grid.get_entries() if isinstance(e, SxmFile)]
            initial_plane_idx = self._grid.thumbnail_plane_index_for_entry(entry)
            dlg = ImageViewerDialog(entry, sxm_entries, cmap_key, t, self,
                                    clip_low=clip[0], clip_high=clip[1],
                                    processing=proc,
                                    spec_image_map=self._spec_image_map,
                                    initial_plane_idx=initial_plane_idx)
            # Share the application-level feature-set store with this viewer.
            dlg._feature_set_store_obj = self._feature_set_store
        # Use show() instead of exec() so the dialog is non-modal: the browse
        # window stays interactive, and all child windows (FFT viewer, Reciprocal
        # Grid panel, etc.) get normal macOS window controls (minimize, resize).
        # Keep a Python reference so the dialog is not garbage-collected while open.
        self._track_open_viewer(dlg)
        def _on_closed(_result, d=dlg, spec=is_spec):
            self._untrack_open_viewer(d)
            # ── Save processing state so it's restored when this scan is reopened ──
            if not spec:
                try:
                    last_entry = d._entries[d._idx]
                    state = dict(getattr(d, "_processing", {}) or {})
                    # mtime-aware set: empty state evicts the cache entry,
                    # non-empty state stores (mtime, state) (review gui-arch #10).
                    self._saved_processing_set(last_entry, state)
                    # Also update the Browse thumbnail so it shows the processed view.
                    self._grid.set_entry_processing(str(last_entry.path), state)
                except Exception:
                    _log.warning("Could not save viewer processing state for "
                                 "reopen / Browse thumbnail", exc_info=True)
            # Handle "Send to …" actions that were NOT already handled immediately
            # (e.g. user closed the viewer without clicking Send after setting action).
            if not spec and d._deferred.is_pending():
                self._load_from_viewer(d, d._deferred.action)
        dlg.finished.connect(_on_closed)
        # immediate_action_requested fires when user clicks "→ Feature Counting" /
        # "→ TV Denoising" so the viewer stays open and the action runs right away.
        # Only the image viewer exposes this signal; spectroscopy viewers don't.
        if not is_spec:
            dlg.immediate_action_requested.connect(
                lambda action, _d=dlg: self._load_from_viewer_live(_d, action)
            )
        dlg.show()

    def _track_open_viewer(self, dlg) -> None:
        """Hold a strong ref to a modeless viewer and reap it on destruction."""
        if dlg is None:
            return
        try:
            self._open_viewers.add(dlg)
        except AttributeError:
            self._open_viewers = {dlg}
        try:
            dlg.destroyed.connect(
                lambda _obj=None, _dlg=dlg: self._untrack_open_viewer(_dlg)
            )
        except Exception:
            _log.warning("Could not connect destroyed() for viewer tracking",
                         exc_info=True)

    def _untrack_open_viewer(self, dlg) -> None:
        viewers = getattr(self, "_open_viewers", None)
        if not viewers:
            return
        try:
            viewers.discard(dlg)
        except AttributeError:
            try:
                viewers.remove(dlg)
            except ValueError:
                pass

    def _load_from_viewer(self, dlg, action: str):
        """Load the processed array from a closed ImageViewerDialog into Features or TV."""
        entry = dlg._entries[dlg._idx]
        plane_idx = dlg._deferred.plane_idx
        arr = dlg._display_arr if dlg._display_arr is not None else dlg._raw_arr
        if arr is None:
            self._status_bar.showMessage("Viewer had no image data to send.")
            return
        scan_range = dlg._scan_range_m
        shape = arr.shape
        if scan_range and shape and shape[0] > 0 and shape[1] > 0:
            w_m, h_m = float(scan_range[0]), float(scan_range[1])
            px_x_m = float(w_m / shape[1])
            px_y_m = float(h_m / shape[0])
            px_m = float(np.sqrt(px_x_m * px_y_m))
        else:
            px_x_m = px_y_m = px_m = 1e-10
        if action == "features":
            self._open_fc_window()
            self._fc_window.load_entry(entry, plane_idx, arr, px_m, px_x_m, px_y_m)
            self._status_bar.showMessage(f"{entry.stem} → Feature Counting")
        elif action == "tv":
            self._switch_mode("tv")
            self._tv_panel.load_entry(entry, plane_idx, arr, px_m)
            self._tv_sidebar.set_status(
                f"Loaded {entry.stem} (processed, plane {plane_idx}). Adjust parameters and Run.")
            self._status_bar.showMessage(f"{entry.stem} → TV Denoising")

    def _load_from_viewer_live(self, dlg, action: str) -> None:
        """Load processed data from an *open* ImageViewerDialog into FC/TV without closing it.

        Called via ``immediate_action_requested`` so the image viewer stays open.
        Clears ``_deferred`` so the ``_on_closed`` handler doesn't repeat the action.
        """
        entry = dlg._entries[dlg._idx]
        plane_idx = dlg._deferred.plane_idx
        arr = dlg._display_arr if dlg._display_arr is not None else dlg._raw_arr
        dlg._deferred.clear()   # consumed — prevent _on_closed from re-firing
        if arr is None:
            self._status_bar.showMessage("Viewer had no image data to send.")
            return
        scan_range = dlg._scan_range_m
        shape = arr.shape
        if scan_range and shape and shape[0] > 0 and shape[1] > 0:
            w_m, h_m = float(scan_range[0]), float(scan_range[1])
            px_x_m = float(w_m / shape[1])
            px_y_m = float(h_m / shape[0])
            px_m = float(np.sqrt(px_x_m * px_y_m))
        else:
            px_x_m = px_y_m = px_m = 1e-10
        if action == "features":
            self._open_fc_window()
            self._fc_window.load_entry(entry, plane_idx, arr, px_m, px_x_m, px_y_m)
            self._status_bar.showMessage(f"{entry.stem} → Feature Counting (viewer open)")
        elif action == "tv":
            self._switch_mode("tv")
            self._tv_panel.load_entry(entry, plane_idx, arr, px_m)
            self._tv_sidebar.set_status(
                f"Loaded {entry.stem} (processed, plane {plane_idx}). Adjust parameters and Run.")
            self._status_bar.showMessage(f"{entry.stem} → TV Denoising")

    # ── Convert ────────────────────────────────────────────────────────────────
    def _update_count(self, text: str = ""):
        d = (text or self._conv_panel.input_entry.text()).strip()
        if d and Path(d).is_dir():
            n = len(list(Path(d).glob("*.dat")))
            self._convert_sidebar.update_file_count(n)
        else:
            self._convert_sidebar.update_file_count(-1)

    def _run(self):
        if self._running:
            return
        in_dir  = self._conv_panel.input_entry.text().strip()
        out_dir = self._conv_panel.get_output_dir()
        do_png  = self._convert_sidebar.png_cb.isChecked()
        do_sxm  = self._convert_sidebar.sxm_cb.isChecked()
        do_npy_raw = self._convert_sidebar.npy_raw_cb.isChecked()
        do_npy_physical = self._convert_sidebar.npy_physical_cb.isChecked()
        clip_lo = self._convert_sidebar.clip_low_spin.value()
        clip_hi = self._convert_sidebar.clip_high_spin.value()

        if not in_dir:
            self._conv_panel.log("ERROR: Please select an input folder.", "err"); return
        if out_dir and not Path(out_dir).is_dir():
            self._conv_panel.log(f"ERROR: Output folder not found: {out_dir}", "err"); return
        if not do_png and not do_sxm and not do_npy_raw and not do_npy_physical:
            self._conv_panel.log("ERROR: Select at least one output format.", "err"); return
        if not Path(in_dir).is_dir():
            self._conv_panel.log(f"ERROR: Input folder not found: {in_dir}", "err"); return

        self._running = True
        self._convert_sidebar.run_btn.setText("  Running…  ")
        self._convert_sidebar.run_btn.setEnabled(False)
        self._status_bar.showMessage("Converting…")

        worker = ConversionWorker(
            in_dir, out_dir, do_png, do_sxm, clip_lo, clip_hi,
            do_npy_raw=do_npy_raw,
            do_npy_physical=do_npy_physical,
        )
        worker.signals.log_msg.connect(self._conv_panel.log)
        worker.signals.finished.connect(self._on_done)
        QThreadPool.globalInstance().start(worker)

    @Slot(str)
    def _on_done(self, out_dir: str):
        self._running = False
        self._convert_sidebar.run_btn.setText("  RUN  ")
        self._convert_sidebar.run_btn.setEnabled(True)
        sxm_dir = Path(out_dir) / "sxm"
        entries = scan_image_folder(sxm_dir) if sxm_dir.exists() else []
        if entries:
            self._grid.load(entries, folder_path=str(sxm_dir))
            self._n_loaded = len(entries)
            self._switch_mode("browse")
            self._status_bar.showMessage(
                f"Done — {self._n_loaded} scan(s) ready to browse")
        else:
            self._status_bar.showMessage("Done")

    # ── Theme ──────────────────────────────────────────────────────────────────
    def _toggle_theme(self):
        # Quick keyboard flip (Ctrl+Shift+T) between the base dark and light themes.
        self._set_theme("light" if self._dark else "dark")

    def _on_gui_font_size_changed(self, label: str):
        self._gui_font_size = normalise_gui_font_size(label)
        self._apply_theme()
        self._status_bar.showMessage(f"Text size: {self._gui_font_size}")

    def _apply_theme(self):
        t = THEMES[self._theme_name]
        app = QApplication.instance()
        app.setFont(QFont(ui_family(), GUI_FONT_SIZES[self._gui_font_size]))
        app.setPalette(_build_palette(t))
        app.setStyleSheet(_build_qss(t, GUI_FONT_SIZES[self._gui_font_size]))
        self._grid.apply_theme(t)
        self._browse_tools.apply_theme(t)
        self._browse_info.apply_theme(t)
        self._conv_panel.apply_theme(t)
        self._dataset_builder_panel.apply_theme(t)
        self._dataset_builder_sidebar.apply_theme(t)
        if self._fc_window is not None:
            self._fc_window.apply_theme(t)
        self._update_tab_styles()

    # ── About ──────────────────────────────────────────────────────────────────
    def _show_about(self):
        t   = THEMES[self._theme_name]
        dlg = AboutDialog(t, self)
        dlg.exec()

    # ── Restart ────────────────────────────────────────────────────────────────
    def _restart_app(self) -> None:
        """Relaunch ProbeFlow in a fresh process and close this window.

        Because ProbeFlow is installed with ``pip install -e .``, the new
        process picks up any source-file edits you made since the last launch —
        no reinstall step needed.  The new window appears before this one
        closes so there is no gap in the taskbar.

        The current folder is passed via ``--browse`` so you land back where
        you were automatically.
        """
        import subprocess

        # Always use ``python -m probeflow gui`` rather than sys.argv so this
        # works identically whether ProbeFlow was started as ``probeflow gui``,
        # ``python -m probeflow gui``, or from an IDE.
        args = [sys.executable, "-m", "probeflow", "gui"]

        cur = self._grid.current_dir()
        if cur:
            args += ["--browse", str(cur)]

        subprocess.Popen(args)
        # quit() bypasses closeEvent, so drain workers explicitly before the
        # event loop (and then the interpreter) tears down under them.
        self._drain_worker_pools()
        QApplication.instance().quit()

    # ── Close ──────────────────────────────────────────────────────────────────
    def _install_quit_drain(self) -> None:
        """Drain worker pools at application quit, whichever window quits it.

        ``closeEvent`` only covers quitting via this main window — but with
        ``quitOnLastWindowClosed``, closing a modeless viewer left open after
        the main window closed quits the app with no drain, racing teardown
        against in-flight pool renders the viewer started in the meantime
        (the same crash class the closeEvent drain was added for). Qt drops
        the connection automatically if this window is destroyed first.
        """
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._drain_worker_pools)

    def _drain_worker_pools(self, timeout_ms: int = 5000) -> None:
        """Drop queued work and wait for in-flight workers before teardown.

        Without this, QApplication / interpreter teardown races still-running
        pool workers (thumbnail renders, conversions, feature previews) whose
        signals objects are parented to the QApplication — observed as
        sporadic crashes on quit. A bounded wait keeps a stuck worker from
        turning quit into a hang.
        """
        pools = [QThreadPool.globalInstance()]
        fc = getattr(self, "_fc_window", None)
        if fc is not None:
            pools.append(getattr(fc, "_preview_pool", None))
        for pool in pools:
            if pool is None:
                continue
            try:
                pool.clear()
                if not pool.waitForDone(timeout_ms):
                    _log.warning(
                        "Worker pool did not finish within %d ms at shutdown",
                        timeout_ms,
                    )
            except Exception:
                _log.warning("Worker pool drain failed", exc_info=True)

    def closeEvent(self, event):
        self._drain_worker_pools()
        cfg = load_config()
        cfg.update({
            "theme_name":    self._theme_name,
            "dark_mode":     self._dark,
            "input_dir":     self._conv_panel.input_entry.text(),
            "output_dir":    self._conv_panel.output_entry.text(),
            "custom_output": self._conv_panel._custom_out_cb.isChecked(),
            "do_png":        self._convert_sidebar.png_cb.isChecked(),
            "do_sxm":        self._convert_sidebar.sxm_cb.isChecked(),
            "clip_low":      self._convert_sidebar.clip_low_spin.value(),
            "clip_high":     self._convert_sidebar.clip_high_spin.value(),
            "colormap":       self._browse_tools.cmap_cb.currentText(),
            "browse_filter":  self._browse_tools.get_filter_mode(),
            "gui_font_size":  self._gui_font_size,
            "thumbnail_size": self._browse_tools.size_cb.currentText().lower(),
        })
        self._save_desktop_layout_into(cfg)
        save_config(cfg)
        super().closeEvent(event)


# ── Entry point ────────────────────────────────────────────────────────────────
def main(*, open_survey: "Optional[Path]" = None,
         browse_folder: "Optional[Path]" = None) -> None:
    app    = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ProbeFlow")
    from probeflow.gui.tooltips import install_global_tooltips
    install_global_tooltips(app)
    window = ProbeFlowWindow(open_survey=open_survey, browse_folder=browse_folder)
    if getattr(window, "_show_maximized_on_start", False):
        window.showMaximized()
    else:
        window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
