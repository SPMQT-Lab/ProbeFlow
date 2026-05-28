"""ProbeFlow main window and application entry point.

This module contains ``ProbeFlowWindow`` (the top-level QMainWindow) and the
``main()`` entry point.  It was extracted from ``gui/_legacy.py`` as Phase 5
of the ongoing refactor; ``_legacy.py`` now re-imports from here for backward
compatibility.
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
    QAction, QActionGroup, QFont, QKeySequence,
)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QFileDialog,
    QHeaderView, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QSplitter, QStackedWidget,
    QStatusBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from probeflow.gui.utils import _open_url
from probeflow.gui.navbar import Navbar
from probeflow.gui.features import (
    FeaturesPanel,
    FeaturesSidebar,
    _FeaturesWorker,
)
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
    _build_qss,
)
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
)
from probeflow.gui.browse import ThumbnailGrid, BrowseInfoPanel, BrowseToolPanel
from probeflow.gui.convert import ConvertPanel, ConvertSidebar
from probeflow.gui.dialogs.definitions import _DefinitionsDialog
from probeflow.gui.terminal import _DevSidebar
from probeflow.gui.dialogs.image_viewer import ImageViewerDialog
from probeflow.gui.dialogs import SpecViewerDialog


# ── Main window ───────────────────────────────────────────────────────────────
class ProbeFlowWindow(QMainWindow):
    LEFT_SIDEBAR_DEFAULT_W = 280
    LEFT_SIDEBAR_MIN_W = 240
    RIGHT_INSPECTOR_DEFAULT_W = 340
    RIGHT_INSPECTOR_MIN_W = 300
    CENTRAL_BROWSER_MIN_W = 500

    def __init__(self, *, open_survey: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("ProbeFlow")
        self.setMinimumSize(1100, 720)
        self.resize(1480, 800)
        self._show_maximized_on_start = False

        self._cfg      = load_config()
        self._dark     = self._cfg.get("dark_mode", True)
        self._gui_font_size = normalise_gui_font_size(self._cfg.get("gui_font_size"))
        self._mode     = "browse"
        self._running  = False
        self._n_loaded = 0
        self._pending_survey = Path(open_survey) if open_survey else None
        # Spec → image mapping (populated by user via "Map spectra…" dialogs;
        # kept empty by default so we never auto-attach spectra to the wrong
        # image based on coordinate guesses alone). Keys are spec stems,
        # values are image stems within the currently loaded folder.
        self._spec_image_map: dict[str, str] = {}
        # Non-modal viewer dialogs are shown with show() rather than exec() so
        # that the browse window remains interactive while scans are open.  We
        # keep explicit Python references here so the dialogs are not
        # garbage-collected while open (show() does not block like exec()).
        self._open_viewers: list = []

        # Per-scan processing memory: path_str → processing dict.
        # Populated when a viewer closes; restored when the same scan is reopened.
        self._saved_processing: dict = {}

        self._build_ui()
        self._apply_theme()
        self._restore_desktop_layout()

        # If launched with --open-survey, jump straight into Survey mode with
        # the manifest pre-loaded. Wire the panel's log_message into status bar.
        if self._pending_survey is not None:
            try:
                self._survey_panel.log_message.connect(self._status_bar.showMessage)
                if self._survey_panel.load_manifest(self._pending_survey):
                    self._switch_mode("survey")
            except Exception as e:
                self._status_bar.showMessage(f"Could not open survey: {e}")

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        v_lay = QVBoxLayout(central)
        v_lay.setContentsMargins(0, 0, 0, 0)
        v_lay.setSpacing(0)

        self._navbar = Navbar(self._dark, self._gui_font_size)
        self._navbar.theme_toggle_clicked.connect(self._toggle_theme)
        self._navbar.font_size_changed.connect(self._on_gui_font_size_changed)
        self._navbar.about_clicked.connect(self._show_about)
        v_lay.addWidget(self._navbar)

        # Body splitter
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.setHandleWidth(5)
        v_lay.addWidget(self._splitter, 1)

        t = THEMES["dark" if self._dark else "light"]

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
        from probeflow.gui.survey import SurveyPanel
        self._survey_panel   = SurveyPanel()
        self._content_stack.addWidget(self._browse_splitter)        # idx 0 browse
        self._content_stack.addWidget(self._conv_panel)             # idx 1 convert
        self._content_stack.addWidget(self._features_panel)         # idx 2 features
        self._content_stack.addWidget(self._tv_panel)               # idx 3 tv
        self._content_stack.addWidget(self._dev_terminal)           # idx 4 dev
        self._content_stack.addWidget(self._survey_panel)           # idx 5 survey
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
        self._splitter.addWidget(self._sidebar_stack)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._apply_default_splitter_sizes()

        # Features tab plumbing.  Review gui-arch #2 (fixed 2026-05-28):
        # each spawned _FeaturesWorker now owns its own signals; we
        # connect ``worker.signals.finished`` at spawn time so concurrent
        # runs cannot cross-talk via a shared signal instance.
        self._features_pool    = QThreadPool.globalInstance()

        # Floating Feature Counting window (lazy-created on first open)
        self._fc_window = None

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
        self._browse_tools.colormap_changed.connect(self._on_thumbnail_colormap_changed)
        self._browse_tools.thumbnail_align_changed.connect(self._on_thumbnail_align_changed)
        self._browse_tools.map_spectra_requested.connect(self._on_map_spectra)
        self._browse_tools.overlay_spectra_requested.connect(self._on_overlay_selected_spectra)
        self._browse_tools.filter_changed.connect(self._on_filter_changed)
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
        self._convert_sidebar.run_btn.clicked.connect(self._run)
        self._conv_panel.input_entry.textChanged.connect(self._update_count)

        self._features_panel.go_to_browse_requested.connect(
            lambda: self._switch_mode("browse"))
        self._features_sidebar.load_from_browse_requested.connect(
            self._on_features_load_from_browse)
        self._features_sidebar.run_requested.connect(self._on_features_run)
        self._features_sidebar.export_requested.connect(self._on_features_export)
        self._features_sidebar.crop_template_requested.connect(
            self._features_panel.begin_template_crop)
        self._features_sidebar.classify_params_changed.connect(
            self._on_classify_params_changed)
        self._features_sidebar.segment_requested.connect(
            self._on_features_segment_requested)
        self._features_sidebar.undo_label_requested.connect(
            self._features_panel.undo_last_label)
        self._features_sidebar.mode_changed.connect(
            self._on_features_mode_changed)
        self._features_sidebar.mask_paint_toggled.connect(
            self._on_features_mask_paint_toggled)
        self._features_sidebar.mask_clear_requested.connect(
            self._features_panel.clear_exclusion_mask)
        self._features_sidebar.mask_color_changed.connect(
            self._features_panel.set_mask_color)

        # Status bar
        self._status_bar = QStatusBar()
        self._status_bar.setFont(QFont("Helvetica", 10))
        self.setStatusBar(self._status_bar)
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
        recent_action = QAction("Open recent", self)
        recent_action.setEnabled(False)
        file_menu.addAction(recent_action)
        file_menu.addSeparator()
        export_image_action = QAction("Export image...", self)
        export_image_action.setEnabled(False)
        file_menu.addAction(export_image_action)
        export_processed_action = QAction("Export processed image...", self)
        export_processed_action.setEnabled(False)
        file_menu.addAction(export_processed_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu_bar.addMenu("View")
        _mode_action(view_menu, "Browse", "browse", "Ctrl+1")
        view_menu.addSeparator()
        reset_layout_action = QAction("Reset window layout", self)
        reset_layout_action.triggered.connect(self._reset_window_layout)
        view_menu.addAction(reset_layout_action)
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("Theme")
        for label, dark in (("Dark mode", True), ("Light mode", False)):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=dark: self._set_dark_mode(value)
            )
            self._theme_action_group.addAction(action)
            self._theme_actions["dark" if dark else "light"] = action
            theme_menu.addAction(action)
        self._theme_actions["dark"].setShortcut(QKeySequence("Ctrl+Shift+T"))
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

        processing_menu = menu_bar.addMenu("Processing")
        align_menu = processing_menu.addMenu("Align rows")
        for label in ("None", "Mean", "Median"):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=label: self._set_thumbnail_align(value)
            )
            self._thumbnail_align_action_group.addAction(action)
            self._thumbnail_align_actions[label] = action
            align_menu.addAction(action)

        convert_menu = menu_bar.addMenu("Convert")
        _mode_action(convert_menu, "Convert Createc .dat to .sxm...", "convert", "Ctrl+2")
        batch_convert_action = QAction("Batch convert folder...", self)
        batch_convert_action.triggered.connect(lambda: self._switch_mode("convert"))
        convert_menu.addAction(batch_convert_action)

        tools_menu = menu_bar.addMenu("Tools")
        map_action = QAction("Map Spectra to Images...", self)
        map_action.triggered.connect(self._on_map_spectra)
        tools_menu.addAction(map_action)
        tools_menu.addSeparator()
        fc_window_action = QAction("Open Feature Counting window…", self)
        fc_window_action.setShortcut(QKeySequence("Ctrl+Shift+F"))
        fc_window_action.triggered.connect(self._open_fc_window)
        tools_menu.addAction(fc_window_action)
        _mode_action(tools_menu, "Feature counting (tab)", "features", "Ctrl+3")
        _mode_action(tools_menu, "TV denoise", "tv", "Ctrl+4")
        tools_menu.addSeparator()
        _mode_action(tools_menu, "Survey (ScanFlow campaign)", "survey", "Ctrl+Shift+S")
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

    def _restore_desktop_layout(self) -> None:
        layout = self._cfg.get("layout", {}).get("main_window", {})
        restore_geometry_or_default(self, layout.get("geometry"), 0.88)
        state = layout.get("state")
        if state:
            try:
                self.restoreState(b64_to_qbytearray(state))
            except Exception:
                pass

        main_sizes = layout.get("splitter_sizes")
        if main_sizes and len(main_sizes) == self._splitter.count():
            self._splitter.setSizes([max(1, int(x)) for x in main_sizes])
        else:
            self._apply_default_splitter_sizes()

        browse_sizes = layout.get("browse_splitter_sizes")
        if browse_sizes and len(browse_sizes) == self._browse_splitter.count():
            self._browse_splitter.setSizes([max(1, int(x)) for x in browse_sizes])

        self._restore_metadata_table_columns(layout.get("metadata_table_column_widths"))
        self._show_maximized_on_start = bool(layout.get("was_maximized"))

    def _save_desktop_layout_into(self, cfg: dict) -> None:
        layout_root = cfg.setdefault("layout", {})
        layout = layout_root.setdefault("main_window", {})
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

    def _set_dark_mode(self, dark: bool) -> None:
        dark = bool(dark)
        if self._dark == dark:
            self._sync_menu_actions()
            return
        self._dark = dark
        self._navbar.set_dark(self._dark)
        self._apply_theme()

    def _set_thumbnail_colormap(self, label: str) -> None:
        if hasattr(self, "_browse_tools") and self._browse_tools.cmap_cb.currentText() != label:
            self._browse_tools.cmap_cb.setCurrentText(label)
        else:
            self._on_thumbnail_colormap_changed(CMAP_KEY.get(label, DEFAULT_CMAP_KEY))
        self._sync_menu_actions()

    def _set_thumbnail_channel(self, channel: str) -> None:
        if hasattr(self, "_browse_tools") and self._browse_tools.thumbnail_channel_cb.currentText() != channel:
            self._browse_tools.thumbnail_channel_cb.setCurrentText(channel)
        else:
            self._on_thumbnail_channel_changed(channel)
        self._sync_menu_actions()

    def _set_thumbnail_align(self, mode: str) -> None:
        if hasattr(self, "_browse_tools") and self._browse_tools.align_rows_cb.currentText() != mode:
            self._browse_tools.align_rows_cb.setCurrentText(mode)
        else:
            self._on_thumbnail_align_changed(mode)
        self._sync_menu_actions()

    def _show_definitions(self) -> None:
        theme = THEMES["dark" if self._dark else "light"]
        dlg = getattr(self, "_definitions_dialog", None)
        if dlg is None:
            dlg = _DefinitionsDialog(theme, self)
            self._definitions_dialog = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    # ── Mode switching ─────────────────────────────────────────────────────────
    def _switch_mode(self, mode: str):
        if mode == "defs":
            self._show_definitions()
            self._status_bar.showMessage(
                "Processing algorithm reference opened in a separate window")
            self._sync_menu_actions()
            return
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
            if self._features_panel.current_array() is None:
                self._status_bar.showMessage(
                    "Pick a scan in Browse, then 'Load primary scan from Browse'")
            else:
                self._status_bar.showMessage("FeatureCounting — pick a mode and Run")
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
            dark_key = "dark" if self._dark else "light"
            for key, action in self._theme_actions.items():
                action.blockSignals(True)
                action.setChecked(key == dark_key)
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
            cmap_key, _, proc = self._grid.get_card_state(entry.stem)
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
        primary = self._grid.get_primary()
        if primary:
            entry = next((e for e in self._grid.get_entries()
                          if e.stem == primary), None)
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
        selected = self._grid.get_selected()
        entries = [
            e for e in self._grid.get_entries()
            if isinstance(e, VertFile) and e.stem in selected
        ]
        if len(entries) < 2:
            self._status_bar.showMessage(
                "Select two or more spectra with Ctrl-click before overlaying.")
            return
        t = THEMES["dark" if self._dark else "light"]
        dlg = SpecOverlayDialog(entries, t, self)
        dlg.exec()

    def _on_card_context_action(self, entry, action: str):
        """Dispatch ScanCard right-click actions (Send to Features, export, show metadata)."""
        if action == "features":
            self._switch_mode("features")
            try:
                _scan = load_scan(entry.path)
            except Exception as exc:
                self._status_bar.showMessage(f"Could not read scan: {exc}")
                return
            plane_idx = self._features_sidebar.plane_index()
            if plane_idx >= _scan.n_planes:
                plane_idx = 0
            arr = _scan.planes[plane_idx]
            if arr is None:
                self._status_bar.showMessage("Could not read scan plane.")
                return
            w_m, h_m = _scan.scan_range_m
            Ny, Nx = arr.shape
            if Nx <= 0 or Ny <= 0 or w_m <= 0 or h_m <= 0:
                px_x_m = px_y_m = px_m = 1e-10
            else:
                px_x_m = float(w_m / Nx)
                px_y_m = float(h_m / Ny)
                px_m = float(np.sqrt(px_x_m * px_y_m))
            self._features_panel.load_entry(entry, plane_idx, arr, px_m, px_x_m, px_y_m)
            self._features_sidebar.set_status(
                f"Loaded {entry.stem} (plane {plane_idx})")
            self._status_bar.showMessage(f"{entry.stem} sent to FeatureCounting")

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
            tbl.setFont(QFont("Helvetica", 9))
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
            self._fc_window = FeatureCountingWindow(parent=None)
            self._fc_window.load_from_browse_needed.connect(
                self._on_fc_load_from_browse)
        self._fc_window.show()
        self._fc_window.raise_()
        self._fc_window.activateWindow()

    def _load_scan_plane_for_analysis(
        self, entry, plane_idx: int
    ) -> tuple:
        """Load a scan plane and apply any saved viewer processing.

        Returns ``(arr, px_m, px_x_m, px_y_m, actual_plane_idx)`` or raises.
        The returned array is the *processed* version — identical to what the
        user last saw in the image viewer — so Feature Counting and TV-denoise
        work on the same data the user inspected.
        """
        _scan = load_scan(entry.path)
        if plane_idx >= _scan.n_planes:
            plane_idx = 0
        arr = _scan.planes[plane_idx]
        w_m, h_m = _scan.scan_range_m
        if arr is None:
            raise ValueError("Scan returned no array for that plane.")
        Ny, Nx = arr.shape
        if Nx > 0 and Ny > 0 and w_m > 0 and h_m > 0:
            px_x_m = float(w_m / Nx)
            px_y_m = float(h_m / Ny)
            px_m   = float(np.sqrt(px_x_m * px_y_m))
        else:
            px_x_m = px_y_m = px_m = 1e-10

        # Apply saved processing (align rows, background, etc.) so Feature
        # Counting sees the same image the user processed in the viewer.
        saved_proc = self._saved_processing.get(str(entry.path))
        if saved_proc:
            try:
                from probeflow.gui.rendering import _apply_processing
                arr = _apply_processing(arr, saved_proc)
            except Exception:
                pass   # fall back to raw if processing fails

        return arr, px_m, px_x_m, px_y_m, plane_idx

    def _on_fc_load_from_browse(self) -> None:
        """Bridge: read Browse selection → load into the floating FC window."""
        from probeflow.gui.models import VertFile
        if self._fc_window is None:
            return
        primary = self._grid.get_primary()
        if not primary:
            self._fc_window._sidebar.set_status(
                "Select a scan in the Browse tab first.")
            return
        entry = next((e for e in self._grid.get_entries() if e.stem == primary), None)
        if not entry or isinstance(entry, VertFile):
            self._fc_window._sidebar.set_status(
                "Selected entry is not a topography scan.")
            return
        plane_idx = self._fc_window._sidebar.plane_index()
        try:
            arr, px_m, px_x_m, px_y_m, plane_idx = \
                self._load_scan_plane_for_analysis(entry, plane_idx)
        except Exception as exc:
            self._fc_window._sidebar.set_status(f"Could not read scan: {exc}")
            return
        self._fc_window.load_entry(entry, plane_idx, arr, px_m, px_x_m, px_y_m)

    # ── Features tab handlers ──────────────────────────────────────────────────
    def _on_features_load_from_browse(self):
        primary = self._grid.get_primary()
        if not primary:
            self._features_sidebar.set_status("Select a scan in the Browse tab first.")
            self._status_bar.showMessage("Pick a scan in Browse to load it into FeatureCounting")
            return
        entry = next((e for e in self._grid.get_entries() if e.stem == primary), None)
        if not entry or isinstance(entry, VertFile):
            self._features_sidebar.set_status("Selected entry is not a topography scan.")
            return
        plane_idx = self._features_sidebar.plane_index()
        try:
            arr, px_m, px_x_m, px_y_m, plane_idx = \
                self._load_scan_plane_for_analysis(entry, plane_idx)
        except Exception as exc:
            self._features_sidebar.set_status(f"Could not read scan: {exc}")
            return
        self._features_panel.load_entry(entry, plane_idx, arr, px_m, px_x_m, px_y_m)
        self._features_sidebar.set_status(
            f"Loaded {entry.stem} (plane {plane_idx}, px = {px_m * 1e12:.1f} pm)")

    def _on_features_mask_paint_toggled(self, painting: bool) -> None:
        self._features_panel.set_mask_painting(
            painting, self._features_sidebar.brush_size())
        if painting:
            self._features_sidebar.set_status(
                "Mask mode — click or drag on the image to paint exclusion zones.")
        else:
            status = ("Mask active — excluded regions shown in colour."
                      if self._features_panel.has_exclusion_mask()
                      else "Mask drawing stopped.")
            self._features_sidebar.set_status(status)

    def _on_features_run(self, mode: str):
        arr = self._features_panel.get_analysis_array()   # applies exclusion mask
        if arr is None:
            self._features_sidebar.set_status("Load a scan first.")
            return
        px_m = self._features_panel.current_pixel_size()
        px_x_m, px_y_m = self._features_panel.current_pixel_sizes()
        if px_m <= 0:
            self._features_sidebar.set_status("Scan has no physical pixel size.")
            return

        if mode == "particles":
            params = self._features_sidebar.particles_params()
        elif mode == "template":
            tmpl = self._features_panel.get_template()
            if tmpl is None:
                self._features_sidebar.set_status(
                    "Crop a template first (Template mode → 'Crop template…').")
                return
            params = self._features_sidebar.template_params()
            params["template"] = tmpl
        elif mode == "lattice":
            params = {}
        elif mode == "classify":
            particles = self._features_panel.get_particles()
            if not particles:
                self._features_sidebar.set_status(
                    "Press '① Segment' first to find particles.")
                return
            if not self._features_panel.has_sample_labels():
                self._features_sidebar.set_status(
                    "Click particles on the image to label at least one example.")
                return
            idx_to_p = {p.index: p for p in particles}
            samples = [
                (v["name"], idx_to_p[k])
                for k, v in self._features_panel._sample_labels.items()
                if k in idx_to_p
            ]
            run_p = self._features_sidebar.classify_run_params()
            params = {"particles": particles, "samples": samples,
                      "use_sharpness": run_p.get("use_sharpness", False)}
        else:
            self._features_sidebar.set_status(f"Unknown mode {mode!r}")
            return

        self._features_sidebar.set_status(f"Running {mode}…")
        worker = _FeaturesWorker(
            mode, arr, px_m, px_x_m, px_y_m, params,
        )
        worker.signals.finished.connect(self._on_features_finished)
        self._features_pool.start(worker)

    def _on_features_finished(self, mode: str, result, error: str):
        if error:
            self._features_sidebar.set_status(f"{mode} failed: {error}")
            self._status_bar.showMessage(f"{mode} failed: {error}")
            return
        if mode == "particles":
            self._features_panel.set_particles(result)
            current_mode = self._features_sidebar.current_mode()
            if current_mode == "classify":
                self._features_panel.set_mode("classify")
                self._features_panel.set_sample_selection_armed(True)
                self._features_sidebar.set_status(
                    f"Found {len(result)} particle(s). "
                    "Click any particle to label it, then press ② Run.")
            else:
                self._features_sidebar.set_status(f"Found {len(result)} particle(s).")
            self._status_bar.showMessage(f"Segmentation: {len(result)} particle(s)")
        elif mode == "template":
            self._features_panel.set_detections(result)
            self._features_sidebar.set_status(
                f"Found {len(result)} match(es).")
        elif mode == "lattice":
            self._features_panel.set_lattice(result)
            self._features_sidebar.set_status(
                f"|a|={result.a_length_m * 1e9:.3f} nm  "
                f"|b|={result.b_length_m * 1e9:.3f} nm  "
                f"γ={result.gamma_deg:.1f}°")
        elif mode == "classify":
            _BIN = 15
            class_angles: dict = {}
            for c in result:
                class_angles.setdefault(c.class_name, []).append(
                    getattr(c, "particle_orientation_deg", 0.0))
            total = len(result)
            parts = []
            for cls_name in sorted(class_angles):
                angles = class_angles[cls_name]
                if cls_name == "other":
                    pct = 100.0 * len(angles) / total if total > 0 else 0.0
                    parts.append(f"other: {len(angles)} ({pct:.0f}%)")
                    continue
                valid = np.array([a for a in angles if a == a], dtype=float)
                if valid.size == 0:
                    parts.append(f"{cls_name}: {len(angles)}")
                    continue
                bins = np.floor(valid / _BIN).astype(int)
                for b in sorted(set(bins.tolist())):
                    n_b = int((bins == b).sum())
                    mean_a = float(valid[bins == b].mean())
                    pct = 100.0 * n_b / total if total > 0 else 0.0
                    parts.append(f"{cls_name}({mean_a:.0f}°): {n_b} ({pct:.0f}%)")
            summary = "  |  ".join(parts)
            self._features_panel.set_classifications(result)
            self._features_sidebar.set_status(
                f"Classified {total} particle(s) — {summary}")

    def _features_segmentation_signature(self, params: dict) -> tuple:
        return tuple(sorted(params.items()))

    def _on_classify_params_changed(self) -> None:
        self._features_panel.clear_sample_labels()
        self._features_sidebar.set_status(
            "Segmentation parameters changed — sample labels cleared.")

    def _on_features_mode_changed(self, mode: str) -> None:
        """Arm/disarm classify clicking when the analysis mode tab changes."""
        self._features_panel.set_mode(mode)
        if mode == "classify" and self._features_panel.get_particles():
            self._features_panel.set_sample_selection_armed(True)
            self._features_sidebar.set_status(
                "Click any particle on the image to label it, then press Run.")
        elif mode != "classify":
            self._features_panel.set_sample_selection_armed(False)

    def _on_features_segment_requested(self) -> None:
        """Step 1 — segment particles with current threshold + exclusion mask.

        Covers both the Particles and Classify analysis modes:
        * In Particles mode the contours are the final result.
        * In Classify mode sample-selection clicking is auto-armed after
          segmentation so the user can immediately start labeling particles.
        """
        arr = self._features_panel.get_analysis_array()   # applies exclusion mask
        if arr is None:
            self._features_sidebar.set_status("Load a scan first.")
            return
        px_m = self._features_panel.current_pixel_size()
        px_x_m, px_y_m = self._features_panel.current_pixel_sizes()
        if px_m <= 0:
            self._features_sidebar.set_status("Scan has no physical pixel size.")
            return
        params = self._features_sidebar.particles_params()
        self._features_sidebar.set_status("Segmenting…")
        worker = _FeaturesWorker(
            "particles", arr, px_m, px_x_m, px_y_m, params,
        )
        worker.signals.finished.connect(self._on_features_finished)
        self._features_pool.start(worker)

    def _on_features_export(self, mode: str):
        if mode == "particles":
            items = self._features_panel.get_particles()
            kind  = "particles"
            extra_meta = {"source": None}
        elif mode == "template":
            items = self._features_panel.get_detections()
            kind  = "detections"
            extra_meta = {"source": None}
        elif mode == "lattice":
            lat = self._features_panel.get_lattice()
            items = [lat] if lat is not None else []
            kind  = "lattice"
            extra_meta = {"source": None}
        elif mode == "classify":
            items = self._features_panel.get_classifications()
            kind  = "classifications"
            extra_meta = {
                "samples": self._features_panel.sample_label_rows(),
                "classification": self._features_panel._classification_meta,
            }
        else:
            return
        entry = self._features_panel.current_entry()
        if mode != "classify":
            extra_meta["source"] = str(entry.path) if entry else None
        if not items:
            self._features_sidebar.set_status("Nothing to export — run an analysis first.")
            return
        suggested = (Path.home() / f"{entry.stem if entry else 'features'}_{kind}.json")
        out_path, _ = QFileDialog.getSaveFileName(
            self, f"Export {kind} JSON", str(suggested), "JSON (*.json)")
        if not out_path:
            return
        try:
            from probeflow.io.writers.json import write_json
            write_json(out_path, items, kind=kind, extra_meta=extra_meta)
            self._features_sidebar.set_status(f"Exported → {out_path}")
            self._status_bar.showMessage(f"Exported {kind} → {out_path}")
        except Exception as exc:
            self._features_sidebar.set_status(f"Export failed: {exc}")

    # ── TV-denoise tab handlers ────────────────────────────────────────────────
    def _on_tv_load_from_browse(self):
        primary = self._grid.get_primary()
        if not primary:
            self._tv_sidebar.set_status("Select a scan in the Browse tab first.")
            self._status_bar.showMessage("Pick a scan in Browse to load it into TV-denoise")
            return
        entry = next((e for e in self._grid.get_entries() if e.stem == primary), None)
        if not entry or isinstance(entry, VertFile):
            self._tv_sidebar.set_status("Selected entry is not a topography scan.")
            return
        plane_idx = self._tv_sidebar.plane_index()
        try:
            _scan = load_scan(entry.path)
            if plane_idx >= _scan.n_planes:
                plane_idx = 0
            arr = _scan.planes[plane_idx]
            w_m, h_m = _scan.scan_range_m
        except Exception as exc:
            self._tv_sidebar.set_status(f"Could not read scan: {exc}")
            return
        if arr is None:
            self._tv_sidebar.set_status("Could not read scan plane.")
            return
        Ny, Nx = arr.shape
        if Nx <= 0 or Ny <= 0 or w_m <= 0 or h_m <= 0:
            px_m = 1e-10
        else:
            px_m = float(np.sqrt((w_m / Nx) * (h_m / Ny)))
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
                add_scalebar=True,
                scalebar_unit="nm",
                scalebar_pos="bottom-right",
            )
            self._tv_sidebar.set_status(f"Saved → {out_path}")
            self._status_bar.showMessage(f"Saved {out_path}")
        except Exception as exc:
            self._tv_sidebar.set_status(f"Save failed: {exc}")

    def _open_viewer(self, entry):
        t = THEMES["dark" if self._dark else "light"]
        is_spec = isinstance(entry, VertFile)
        if is_spec:
            dlg = SpecViewerDialog(entry, t, self)
        else:
            cmap_key, clip, proc = self._grid.get_card_state(entry.stem)
            # Restore any processing the user applied last time this scan was open.
            saved = self._saved_processing.get(str(entry.path))
            if saved:
                proc = dict(saved)
            sxm_entries = [e for e in self._grid.get_entries() if isinstance(e, SxmFile)]
            initial_plane_idx = self._grid.thumbnail_plane_index_for_entry(entry)
            dlg = ImageViewerDialog(entry, sxm_entries, cmap_key, t, self,
                                    clip_low=clip[0], clip_high=clip[1],
                                    processing=proc,
                                    spec_image_map=self._spec_image_map,
                                    initial_plane_idx=initial_plane_idx)
        # Use show() instead of exec() so the dialog is non-modal: the browse
        # window stays interactive, and all child windows (FFT viewer, Reciprocal
        # Grid panel, etc.) get normal macOS window controls (minimize, resize).
        # Keep a Python reference so the dialog is not garbage-collected while open.
        self._open_viewers.append(dlg)
        def _on_closed(_result, d=dlg, spec=is_spec):
            try:
                self._open_viewers.remove(d)
            except ValueError:
                pass
            # ── Save processing state so it's restored when this scan is reopened ──
            if not spec:
                try:
                    last_entry = d._entries[d._idx]
                    state = dict(getattr(d, "_processing", {}) or {})
                    key = str(last_entry.path)
                    if state:
                        self._saved_processing[key] = state
                    else:
                        # User reset to original — forget the saved state too.
                        self._saved_processing.pop(key, None)
                    # Also update the Browse thumbnail so it shows the processed view.
                    self._grid.set_entry_processing(key, state)
                except Exception:
                    pass
            # Handle "Send to …" actions requested from inside the image viewer.
            if not spec and d._deferred.is_pending():
                self._load_from_viewer(d, d._deferred.action)
        dlg.finished.connect(_on_closed)
        dlg.show()

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
            self._switch_mode("features")
            self._features_panel.load_entry(entry, plane_idx, arr, px_m, px_x_m, px_y_m)
            self._features_sidebar.set_status(
                f"Loaded {entry.stem} (processed, plane {plane_idx}, px = {px_m * 1e12:.1f} pm)")
            self._status_bar.showMessage(f"{entry.stem} → Feature Counting")
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
        clip_lo = self._convert_sidebar.clip_low_spin.value()
        clip_hi = self._convert_sidebar.clip_high_spin.value()

        if not in_dir:
            self._conv_panel.log("ERROR: Please select an input folder.", "err"); return
        if out_dir and not Path(out_dir).is_dir():
            self._conv_panel.log(f"ERROR: Output folder not found: {out_dir}", "err"); return
        if not do_png and not do_sxm:
            self._conv_panel.log("ERROR: Select at least one output format.", "err"); return
        if not Path(in_dir).is_dir():
            self._conv_panel.log(f"ERROR: Input folder not found: {in_dir}", "err"); return

        self._running = True
        self._convert_sidebar.run_btn.setText("  Running…  ")
        self._convert_sidebar.run_btn.setEnabled(False)
        self._status_bar.showMessage("Converting…")

        worker = ConversionWorker(in_dir, out_dir, do_png, do_sxm, clip_lo, clip_hi)
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
        self._set_dark_mode(not self._dark)

    def _on_gui_font_size_changed(self, label: str):
        self._gui_font_size = normalise_gui_font_size(label)
        self._navbar.blockSignals(True)
        self._navbar.set_font_size(self._gui_font_size)
        self._navbar.blockSignals(False)
        self._apply_theme()
        self._status_bar.showMessage(f"Text size: {self._gui_font_size}")

    def _apply_theme(self):
        t = THEMES["dark" if self._dark else "light"]
        app = QApplication.instance()
        app.setFont(QFont("Helvetica", GUI_FONT_SIZES[self._gui_font_size]))
        app.setStyleSheet(_build_qss(t, GUI_FONT_SIZES[self._gui_font_size]))
        self._grid.apply_theme(t)
        self._browse_tools.apply_theme(t)
        self._browse_info.apply_theme(t)
        self._conv_panel.apply_theme(t)
        self._update_tab_styles()

    # ── About ──────────────────────────────────────────────────────────────────
    def _show_about(self):
        t   = THEMES["dark" if self._dark else "light"]
        dlg = AboutDialog(t, self)
        dlg.exec()

    # ── Close ──────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        cfg = load_config()
        cfg.update({
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
def main(*, open_survey: "Optional[Path]" = None) -> None:
    app    = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ProbeFlow")
    window = ProbeFlowWindow(open_survey=open_survey)
    if getattr(window, "_show_maximized_on_start", False):
        window.showMaximized()
    else:
        window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
