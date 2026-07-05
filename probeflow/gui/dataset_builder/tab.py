"""Top-level Dataset Builder cockpit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal, QThreadPool
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from probeflow.core.mask import ImageMask
from probeflow.dataset_builder.annotations import (
    save_mask_annotation,
    save_review_annotation,
)
from probeflow.dataset_builder.loading import load_scan_plane
from probeflow.dataset_builder.models import DatasetExportSpec, DatasetQueueItem, DatasetTaskConfig
from probeflow.dataset_builder.cache import (
    DatasetBuilderCache,
    LoadedSampleRaw,
    QuickSegPreprocKey,
    QuickSegWatershedKey,
    quickseg_params_fingerprint,
    quickseg_seed_fingerprint,
    sample_cache_key,
)
from probeflow.dataset_builder.quickseg import (
    QuickSegParams,
    QuickSegSeed,
    QuickSegState,
    load_quickseg_state,
    prepare_quickseg_inputs,
    quickseg_overlay_rgba,
    save_quickseg_state,
    watershed_labels,
)
from probeflow.dataset_builder.painting import paint_mask
from probeflow.dataset_builder.queue import build_queue, queue_counts
from probeflow.gui.dataset_builder.display import (
    dataset_builder_display_array,
    current_image_view_array,
    percentile_value,
)
from probeflow.gui.dataset_builder.canvas import DatasetBuilderCanvas
from probeflow.gui.dataset_builder.view_tray import (
    DatasetBuilderCurrentViewTray,
    DatasetBuilderViewTray,
)
from probeflow.gui.dataset_builder.quickseg_controls import QuickSegControlsWidget
from probeflow.gui.workers import (
    DatasetBuilderExportWorker,
    DatasetBuilderFolderIndexLoader,
    DatasetBuilderQueueHydrateWorker,
    DatasetBuilderSampleLoadWorker,
    QuickSegPrepWorker,
    QuickSegWatershedWorker,
)
from probeflow.io.mask_sidecar import load_mask_set_sidecar
from probeflow.processing.display import array_to_uint8


class DatasetBuilderPanel(QWidget):
    """Queue-based labelling cockpit for Dataset Builder."""

    status_message = Signal(str)
    counts_changed = Signal(dict)

    def __init__(self, theme: dict, cfg: dict | None = None, parent=None):
        super().__init__(parent)
        self._theme = dict(theme)
        self._cfg = dict(cfg or {})
        self._queue: list[DatasetQueueItem] = []
        self._current_index = -1
        self._scan = None
        self._arr: np.ndarray | None = None
        self._px_x_m = 1e-10
        self._px_y_m = 1e-10
        self._current_mask: ImageMask | None = None
        self._overlay_visible = True
        self._paint_mode = "brush"
        self._undo_stack: list[np.ndarray] = []
        self._stroke_snapshot: np.ndarray | None = None
        self._stroke_dirty = False
        self._base_display_arr: np.ndarray | None = None
        self._display_arr: np.ndarray | None = None
        self._current_view_points: list[tuple[int, int]] = []
        self._current_view_capture: list[tuple[int, int]] = []
        self._task_widgets: dict[str, list[QWidget]] = {}
        self._quickseg_state = QuickSegState()
        self._quickseg_cache = None
        self._quickseg_controls: QuickSegControlsWidget | None = None
        self._quickseg_overlay_visible = True
        self._quickseg_overlay_dirty = False
        self._quickseg_seed_history: list[dict[str, object]] = []
        self._quickseg_seed_points: list[QuickSegSeed] = []
        self._quickseg_result: np.ndarray | None = None
        self._quickseg_result_path: Path | None = None
        self._quickseg_review_record = None
        self._quickseg_seed_mode = "add"
        self._indexed_items: list | None = None
        self._folder_source: Path | None = None
        self._sample_cache = DatasetBuilderCache()
        self._interactive_pool = QThreadPool(self)
        self._interactive_pool.setMaxThreadCount(2)
        self._export_pool = QThreadPool(self)
        self._export_pool.setMaxThreadCount(1)
        self._queue_index_token = object()
        self._queue_hydrate_token = object()
        self._sample_load_token = object()
        self._quickseg_prep_token = object()
        self._quickseg_ws_token = object()
        self._export_token = object()

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = QHBoxLayout()
        self._source_entry = QLineEdit()
        self._source_entry.setPlaceholderText("Source folder or scan")
        browse_btn = QPushButton("Browse")
        load_btn = QPushButton("Load Queue")
        export_btn = QPushButton("Export Dataset")
        export_btn.setObjectName("accentBtn")
        top.addWidget(QLabel("Dataset Builder"))
        top.addWidget(self._source_entry, 1)
        top.addWidget(browse_btn)
        top.addWidget(load_btn)
        top.addWidget(export_btn)
        root.addLayout(top)

        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        root.addWidget(body, 1)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(6)
        self._progress_lbl = QLabel("0 / 0 labelled")
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "Unlabelled", "Accepted", "Uncertain", "Rejected"])
        self._queue_list = QListWidget()
        left_lay.addWidget(self._progress_lbl)
        left_lay.addWidget(self._filter_combo)
        left_lay.addWidget(self._queue_list, 1)
        body.addWidget(left)

        canvas_host = QWidget()
        canvas_lay = QVBoxLayout(canvas_host)
        canvas_lay.setContentsMargins(0, 0, 0, 0)
        canvas_lay.setSpacing(4)
        self._canvas_status = QLabel("No scan loaded")
        self._canvas = DatasetBuilderCanvas()
        self._canvas.setMinimumSize(420, 360)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setWidget(self._canvas)
        canvas_lay.addWidget(self._canvas_status)
        canvas_lay.addWidget(self._scroll, 1)
        body.addWidget(canvas_host)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)

        self._global_view_tray = DatasetBuilderViewTray(theme, self)
        self._global_view_tray.set_expanded(True)
        self._global_view_tray.percentiles_changed.connect(self._on_view_percentiles_changed)
        self._global_view_tray.flatten_toggled.connect(self._on_flatten_toggled)

        self._current_view_tray = DatasetBuilderCurrentViewTray(theme, self)
        self._current_view_tray.set_expanded(True)
        self._current_view_tray.flatten_requested.connect(self._on_current_flatten_toggled)
        self._current_view_tray.clear_requested.connect(self.clear_current_image_flatten)

        self._task_combo = QComboBox()
        self._task_combo.addItem("Step-edge / ignore-mask", "step_edge_mask")
        self._task_combo.addItem("QuickSeg terrace segmentation", "terrace_segmentation")
        self._plane_spin = QSpinBox()
        self._plane_spin.setRange(0, 64)
        self._status_combo = QComboBox()
        self._status_combo.addItems(["draft", "accepted", "uncertain", "rejected"])
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(1, 128)
        self._brush_spin.setValue(8)

        shared_form = QFormLayout()
        shared_form.addRow("Preset", self._task_combo)
        shared_form.addRow("Plane", self._plane_spin)
        shared_form.addRow("Status", self._status_combo)
        right_lay.addLayout(shared_form)

        step_panel = QWidget()
        step_lay = QVBoxLayout(step_panel)
        step_lay.setContentsMargins(0, 0, 0, 0)
        step_lay.setSpacing(8)
        form = QFormLayout()
        form.addRow("Brush px", self._brush_spin)
        step_lay.addLayout(form)

        brush_btn = QPushButton("Brush")
        brush_btn.setCheckable(True)
        eraser_btn = QPushButton("Eraser")
        eraser_btn.setCheckable(True)
        save_btn = QPushButton("Save")
        save_next_btn = QPushButton("Save + Next")
        accept_btn = QPushButton("Accept")
        uncertain_btn = QPushButton("Uncertain")
        reject_btn = QPushButton("Reject")
        clear_btn = QPushButton("Clear Overlay")
        for btn in (
            brush_btn,
            eraser_btn,
            save_btn,
            save_next_btn,
            accept_btn,
            uncertain_btn,
            reject_btn,
            clear_btn,
        ):
            step_lay.addWidget(btn)
        step_lay.addStretch(1)
        self._step_panel = step_panel

        self._quickseg_controls = QuickSegControlsWidget(theme, self)
        self._quickseg_controls.apply_theme(theme)
        self._quickseg_controls.apply_requested.connect(self._apply_quickseg_params)
        self._quickseg_controls.new_label_requested.connect(self._quickseg_new_label)
        self._quickseg_controls.undo_seed_requested.connect(self._quickseg_undo_last_seed)
        self._quickseg_controls.clear_seeds_requested.connect(self._quickseg_clear_seeds)
        self._quickseg_controls.refresh_requested.connect(self._quickseg_refresh_watershed)
        self._quickseg_controls.clear_result_requested.connect(self._quickseg_clear_result)
        self._quickseg_controls.save_requested.connect(self.save_current)
        self._quickseg_controls.save_next_requested.connect(self.save_and_next)
        self._quickseg_controls.accept_requested.connect(lambda: self._save_status_and_next("accepted"))
        self._quickseg_controls.uncertain_requested.connect(lambda: self._save_status_and_next("uncertain"))
        self._quickseg_controls.reject_requested.connect(lambda: self._save_status_and_next("rejected"))
        self._quickseg_controls.parameters_changed.connect(self._quickseg_update_overlay)

        self._task_stack = QWidget()
        task_stack_lay = QVBoxLayout(self._task_stack)
        task_stack_lay.setContentsMargins(0, 0, 0, 0)
        task_stack_lay.setSpacing(0)
        task_stack_lay.addWidget(step_panel)
        task_stack_lay.addWidget(self._quickseg_controls)

        right_lay.addWidget(self._task_stack)
        right_lay.addStretch(1)
        right.setMinimumWidth(360)
        body.addWidget(right)
        body.setSizes([240, 760, 360])

        self._shortcut_lbl = QLabel(
            "A prev | F save+accept+next | D save+uncertain+next | S save+reject+next | "
            "Q overlay | E eraser | R brush | Z clear | Ctrl+Z undo | "
            "V brush+ | C brush-"
        )
        self._shortcut_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self._shortcut_lbl)

        browse_btn.clicked.connect(self._browse_source)
        load_btn.clicked.connect(self.load_queue)
        export_btn.clicked.connect(self._export_dataset)
        self._queue_list.currentRowChanged.connect(self._on_queue_row_changed)
        self._filter_combo.currentTextChanged.connect(lambda _text: self._refresh_queue_list())
        self._task_combo.currentIndexChanged.connect(self._on_task_changed)
        self._plane_spin.valueChanged.connect(self._on_plane_changed)
        brush_btn.clicked.connect(lambda: self.set_paint_mode("brush"))
        eraser_btn.clicked.connect(lambda: self.set_paint_mode("eraser"))
        save_btn.clicked.connect(self.save_current)
        save_next_btn.clicked.connect(self.save_and_next)
        clear_btn.clicked.connect(self.clear_overlay)
        self._brush_btn = brush_btn
        self._eraser_btn = eraser_btn
        self._canvas.paint_stroke_started.connect(self._begin_paint_stroke)
        self._canvas.paint_stroke_finished.connect(self._end_paint_stroke)
        self._canvas.mask_painted.connect(self._paint_at)
        self._canvas.pixel_clicked.connect(self._on_current_view_point_clicked)
        self._canvas.quickseg_click_requested.connect(self._quickseg_canvas_clicked)
        self.set_paint_mode("brush")

        self._install_shortcuts()
        self.apply_theme(theme)

    def apply_theme(self, theme: dict) -> None:
        self._theme = dict(theme)
        border = theme.get("border", "#3a414c")
        bg = theme.get("main_bg", "#16181d")
        self._canvas_status.setStyleSheet(f"padding: 4px; border-bottom: 1px solid {border};")
        self._shortcut_lbl.setStyleSheet(
            f"background: {theme.get('status_bg', bg)}; color: {theme.get('status_fg', '#9aa1ab')};"
            f"border: 1px solid {border}; border-radius: 4px; padding: 5px;"
        )
        self._global_view_tray.apply_theme(theme)
        self._current_view_tray.apply_theme(theme)
        if self._quickseg_controls is not None:
            self._quickseg_controls.apply_theme(theme)
        self._sync_task_ui()
        if self._arr is not None:
            self._refresh_display_preview()

    def set_source(self, source: str | Path) -> None:
        self._source_entry.setText(str(source))
        self.load_queue()

    def _install_shortcuts(self) -> None:
        self._shortcuts: dict[str, QShortcut] = {}

        def _make(key: str, slot) -> None:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
            self._shortcuts[key] = sc

        _make("A", self.previous_item)
        _make("F", lambda: self._shortcut_save_status("accepted"))
        _make("D", lambda: self._shortcut_save_status("uncertain"))
        _make("S", lambda: self._shortcut_save_status("rejected"))
        _make("Q", self._shortcut_toggle_overlay)
        _make("E", self._shortcut_eraser)
        _make("R", self._shortcut_r)
        _make("Z", self._shortcut_z)
        _make("V", self._shortcut_brush_plus)
        _make("C", self._shortcut_brush_minus)
        undo_key = QKeySequence(QKeySequence.Undo).toString()
        undo = QShortcut(QKeySequence.Undo, self)
        undo.setContext(Qt.WidgetWithChildrenShortcut)
        undo.activated.connect(self._shortcut_undo)
        self._shortcuts[undo_key] = undo
        self._sync_shortcuts_for_task()

    def _sync_shortcuts_for_task(self) -> None:
        is_step_edge = self._task() == "step_edge_mask"
        is_quickseg = self._task() == "terrace_segmentation"
        for key in ("E", "V", "C", "Z"):
            if key in self._shortcuts:
                self._shortcuts[key].setEnabled(is_step_edge)
        if "R" in self._shortcuts:
            self._shortcuts["R"].setEnabled(True)
        for key in ("A", "F", "D", "S", "Q"):
            if key in self._shortcuts:
                self._shortcuts[key].setEnabled(True)
        undo_key = QKeySequence(QKeySequence.Undo).toString()
        if undo_key in self._shortcuts:
            self._shortcuts[undo_key].setEnabled(True)

    def _shortcut_save_status(self, status: str) -> None:
        if self._task() == "terrace_segmentation":
            self._save_status_and_next(status)
            return
        self._save_status_and_next(status)

    def _shortcut_toggle_overlay(self) -> None:
        if self._task() == "terrace_segmentation":
            self.toggle_quickseg_overlay()
        else:
            self.toggle_overlay()

    def _shortcut_eraser(self) -> None:
        if self._task() == "step_edge_mask":
            self.set_paint_mode("eraser")

    def _shortcut_r(self) -> None:
        if self._task() == "terrace_segmentation":
            self._quickseg_refresh_watershed()
        else:
            self.set_paint_mode("brush")

    def _shortcut_z(self) -> None:
        if self._task() == "step_edge_mask":
            self.clear_overlay()

    def _shortcut_brush_plus(self) -> None:
        if self._task() == "step_edge_mask":
            self._brush_spin.setValue(self._brush_spin.value() + 1)

    def _shortcut_brush_minus(self) -> None:
        if self._task() == "step_edge_mask":
            self._brush_spin.setValue(max(1, self._brush_spin.value() - 1))

    def _shortcut_undo(self) -> None:
        if self._task() == "step_edge_mask":
            self.undo_paint()
        else:
            self._quickseg_undo_last_seed()

    def _browse_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Dataset Builder source folder")
        if path:
            self.set_source(path)

    def load_queue(self) -> None:
        source_text = self._source_entry.text().strip()
        if not source_text:
            return
        source = Path(source_text)
        self._folder_source = source
        self._indexed_items = None
        self._queue = []
        self._current_index = -1
        self._queue_list.clear()
        try:
            if source.is_file():
                config = self._config()
                self._queue = build_queue(
                    source,
                    task=config.task,
                    label_name=config.label_name,
                    plane_index=self._plane_spin.value(),
                )
                self._current_index = 0 if self._queue else -1
                self._refresh_queue_list()
                self._load_current()
                self.status_message.emit(f"Dataset Builder queue loaded: {len(self._queue)} scan(s)")
                return
        except Exception as exc:
            self.status_message.emit(f"Dataset Builder queue failed: {exc}")
            return
        self._start_folder_index(source)

    def _start_folder_index(self, source: Path) -> None:
        self._queue_index_token = object()
        self._canvas_status.setText(f"Indexing {source.name}...")
        self.status_message.emit(f"Indexing {source.name}...")
        worker = DatasetBuilderFolderIndexLoader(source, self._queue_index_token)
        worker.signals.indexed.connect(self._on_folder_indexed)
        worker.signals.failed.connect(self._on_folder_index_failed)
        self._interactive_pool.start(worker)

    def _on_folder_indexed(self, source, indexed_items, token) -> None:
        if token is not self._queue_index_token or Path(source) != self._folder_source:
            return
        self._indexed_items = list(indexed_items or [])
        self._hydrate_queue_from_indexed_items()

    def _on_folder_index_failed(self, source, message: str, token) -> None:
        if token is not self._queue_index_token or Path(source) != self._folder_source:
            return
        self.status_message.emit(f"Dataset Builder indexing failed: {message}")

    def _hydrate_queue_from_indexed_items(self) -> None:
        if self._folder_source is None:
            return
        if not self._indexed_items:
            self._queue = []
            self._current_index = -1
            self._refresh_queue_list()
            self._load_current()
            self.status_message.emit("Dataset Builder queue loaded: 0 scan(s)")
            return
        config = self._config()
        self._queue_hydrate_token = object()
        worker = DatasetBuilderQueueHydrateWorker(
            self._folder_source,
            self._indexed_items,
            config.task,
            config.label_name,
            config.plane_index,
            self._queue_hydrate_token,
        )
        worker.signals.loaded.connect(self._on_queue_hydrated)
        worker.signals.failed.connect(self._on_queue_hydrate_failed)
        self._interactive_pool.start(worker)

    def _on_queue_hydrate_failed(self, source, message: str, token) -> None:
        if token is not self._queue_hydrate_token or Path(source) != self._folder_source:
            return
        self.status_message.emit(f"Dataset Builder queue failed: {message}")

    def _on_queue_hydrated(self, source, queue_items, token) -> None:
        if token is not self._queue_hydrate_token or Path(source) != self._folder_source:
            return
        previous_path = self._queue[self._current_index].source_path if 0 <= self._current_index < len(self._queue) else None
        self._queue = list(queue_items or [])
        if previous_path is not None:
            self._current_index = next((i for i, item in enumerate(self._queue) if item.source_path == previous_path), 0 if self._queue else -1)
        else:
            self._current_index = 0 if self._queue else -1
        self._refresh_queue_list()
        self._load_current()
        self.status_message.emit(f"Dataset Builder queue loaded: {len(self._queue)} scan(s)")

    def _task(self) -> str:
        return str(self._task_combo.currentData() or "step_edge_mask")

    def _config(self, *, status: str | None = None) -> DatasetTaskConfig:
        task = self._task()
        return DatasetTaskConfig(
            task=task,
            label_name="step_edge" if task == "step_edge_mask" else "quickseg_terraces",
            label_type="mask" if task == "step_edge_mask" else "instances",
            proposal_method="step_edge" if task == "step_edge_mask" else "quickseg",
            plane_index=self._plane_spin.value(),
            proposal_params=self._quickseg_controls.parameters().to_dict() if task == "terrace_segmentation" and self._quickseg_controls else {},
        )

    def _on_task_changed(self, *_args) -> None:
        self._sync_task_ui()
        if self._source_entry.text().strip():
            if self._indexed_items is not None:
                self._hydrate_queue_from_indexed_items()
            else:
                self.load_queue()
        else:
            self._refresh_shortcut_help()

    def _on_plane_changed(self, *_args) -> None:
        source_text = self._source_entry.text().strip()
        if not source_text:
            return
        source = Path(source_text)
        if source.is_file():
            self.load_queue()
        elif self._indexed_items is not None:
            self._hydrate_queue_from_indexed_items()

    def _sync_task_ui(self) -> None:
        task = self._task()
        is_step = task == "step_edge_mask"
        is_quickseg = task == "terrace_segmentation"
        if hasattr(self, "_step_panel"):
            self._step_panel.setVisible(is_step)
        if self._quickseg_controls is not None:
            self._quickseg_controls.setVisible(is_quickseg)
            self._quickseg_controls.set_review_enabled(True)
        self._canvas.set_paint_enabled(is_step)
        self._canvas.set_quickseg_enabled(is_quickseg)
        if self._quickseg_controls is not None:
            self._quickseg_controls.set_seed_mode_status(
                "Add seed mode" if self._quickseg_seed_mode == "add" else "Delete seed mode"
            )
            self._quickseg_controls.set_current_label(self._quickseg_state.current_label)
            if is_quickseg:
                if self._quickseg_result is None:
                    result_text = "No watershed result"
                else:
                    result_text = f"Watershed ready ({self._quickseg_result.shape[0]}x{self._quickseg_result.shape[1]})"
                self._quickseg_controls.set_result_status(result_text)
        self._refresh_shortcut_help()

    def _refresh_shortcut_help(self) -> None:
        if self._task() == "terrace_segmentation":
            text = (
                "A prev | F save+accept+next | D save+uncertain+next | S save+reject+next | "
                "Q overlay | Ctrl/Cmd+LMB new terrace | LMB seed | Alt+LMB delete nearest | "
                "Ctrl+Z undo seed | R refresh watershed"
            )
        else:
            text = (
                "A prev | F save+accept+next | D save+uncertain+next | S save+reject+next | "
                "Q overlay | E eraser | R brush | Z clear | Ctrl+Z undo | "
                "V brush+ | C brush-"
            )
        self._shortcut_lbl.setText(text)

    def _refresh_queue_list(self) -> None:
        current_path = (
            self._queue[self._current_index].source_path
            if 0 <= self._current_index < len(self._queue)
            else None
        )
        self._queue_list.blockSignals(True)
        try:
            self._queue_list.clear()
            filter_mode = self._filter_combo.currentText()
            for index, item in enumerate(self._queue):
                if filter_mode == "Unlabelled" and item.status != "blank":
                    continue
                if filter_mode == "Accepted" and item.status != "accepted":
                    continue
                if filter_mode == "Uncertain" and item.status != "uncertain":
                    continue
                if filter_mode == "Rejected" and item.status != "rejected":
                    continue
                text = f"{_status_symbol(item.status)}  {item.display_id}"
                row = QListWidgetItem(text)
                row.setData(Qt.UserRole, index)
                self._queue_list.addItem(row)
                if item.source_path == current_path:
                    self._queue_list.setCurrentItem(row)
        finally:
            self._queue_list.blockSignals(False)
        counts = queue_counts(self._queue)
        labelled = sum(counts.get(k, 0) for k in ("accepted", "uncertain", "rejected", "exported"))
        self._progress_lbl.setText(f"{labelled} / {counts.get('total', 0)} labelled")
        self.counts_changed.emit(counts)

    def _on_queue_row_changed(self, row: int) -> None:
        item = self._queue_list.item(row)
        if item is None:
            return
        index = item.data(Qt.UserRole)
        if isinstance(index, int) and index != self._current_index:
            self._current_index = index
            self._load_current()

    def _current_item(self) -> DatasetQueueItem | None:
        if 0 <= self._current_index < len(self._queue):
            return self._queue[self._current_index]
        return None

    def _current_sample_key(self) -> SampleCacheKey | None:
        item = self._current_item()
        if item is None:
            return None
        return sample_cache_key(item.source_path, item.plane_index)

    def _load_current(self) -> None:
        self._current_mask = None
        self._overlay_visible = True
        self._undo_stack.clear()
        self._stroke_snapshot = None
        self._stroke_dirty = False
        self._canvas.clear_mask_overlay()
        self._canvas.clear_rgba_overlay()
        self._current_view_points = []
        self._cancel_current_image_flatten(refresh=False)
        self._quickseg_seed_history.clear()
        self._quickseg_seed_points = []
        self._quickseg_result = None
        self._quickseg_result_path = None
        self._quickseg_review_record = None
        self._quickseg_cache = None
        self._quickseg_state = QuickSegState()

        item = self._current_item()
        if item is None:
            self._arr = None
            self._scan = None
            self._base_display_arr = None
            self._display_arr = None
            self._global_view_tray.clear_histogram(self._theme)
            self._canvas.setText("No scan")
            self._canvas_status.setText("No scan loaded")
            return

        key = self._current_sample_key()
        flatten_enabled = self._global_view_tray.is_flatten_enabled()
        cached = self._sample_cache.get_sample(key) if key is not None else None
        if cached is not None:
            self._apply_loaded_sample(cached, reset_zoom=True)
            self._post_sample_load(item)
            return

        self._canvas_status.setText(f"Loading {item.display_id}...")
        self.status_message.emit(f"Loading {item.display_id}...")
        self._sample_load_token = object()
        worker = DatasetBuilderSampleLoadWorker(
            item.source_path,
            item.plane_index,
            flatten_enabled,
            self._sample_load_token,
        )
        worker.signals.loaded.connect(self._on_sample_loaded)
        worker.signals.failed.connect(self._on_sample_load_failed)
        self._interactive_pool.start(worker)

    def _on_sample_load_failed(self, key, message: str, token) -> None:
        if token is not self._sample_load_token:
            return
        self.status_message.emit(f"Could not load {Path(key.path).name}: {message}")

    def _on_sample_loaded(self, key, result: LoadedSampleRaw, token) -> None:
        if result is not None:
            self._sample_cache.put_sample(result)
            if result.display_arr is not None:
                self._sample_cache.put_display(result.key, bool(result.flatten_enabled), result.display_arr)
        if token is not self._sample_load_token or key != self._current_sample_key():
            return
        self._apply_loaded_sample(result, reset_zoom=True)
        self._post_sample_load(self._current_item())

    def _prefetch_next_sample(self) -> None:
        if self._current_index < 0 or self._current_index + 1 >= len(self._queue):
            return
        next_item = self._queue[self._current_index + 1]
        next_key = sample_cache_key(next_item.source_path, next_item.plane_index)
        if self._sample_cache.get_sample(next_key) is not None:
            return
        token = object()
        worker = DatasetBuilderSampleLoadWorker(
            next_item.source_path,
            next_item.plane_index,
            self._global_view_tray.is_flatten_enabled(),
            token,
        )
        worker.signals.loaded.connect(self._on_sample_loaded)
        worker.signals.failed.connect(self._on_sample_load_failed)
        self._interactive_pool.start(worker)

    def _apply_loaded_sample(self, sample: LoadedSampleRaw, *, reset_zoom: bool = False) -> None:
        self._scan = sample.scan
        self._arr = sample.arr
        self._px_x_m = float(sample.px_x_m)
        self._px_y_m = float(sample.px_y_m)
        self._base_display_arr = sample.display_arr if sample.display_arr is not None and bool(sample.flatten_enabled) == self._global_view_tray.is_flatten_enabled() else None
        self._display_arr = None
        self._sync_task_ui()
        self._refresh_display_preview(reset_zoom=reset_zoom)

    def _post_sample_load(self, item: DatasetQueueItem | None) -> None:
        if item is None:
            return
        if self._task() == "terrace_segmentation":
            self._load_existing_quickseg(item)
        else:
            self._load_existing_mask(item)
            coverage = 100.0 * float(self._current_mask.data.mean()) if self._current_mask else 0.0
            self._canvas_status.setText(
                f"{item.display_id} | task: {self._task()} | status: {item.status} | "
                f"mask coverage: {coverage:.2f}%"
            )
        self._prefetch_next_sample()

    def _base_display_array(self) -> np.ndarray | None:
        if self._arr is None:
            return None
        key = self._current_sample_key()
        if key is not None:
            cached = self._sample_cache.get_display(key, self._global_view_tray.is_flatten_enabled())
            if cached is not None:
                self._base_display_arr = cached
                return cached
        if self._base_display_arr is not None:
            return self._base_display_arr
        self._base_display_arr = dataset_builder_display_array(
            self._arr,
            flatten=self._global_view_tray.is_flatten_enabled(),
        )
        if key is not None:
            self._sample_cache.put_display(key, self._global_view_tray.is_flatten_enabled(), self._base_display_arr)
        return self._base_display_arr

    def _refresh_display_preview(self, *_, reset_zoom: bool = False) -> None:
        base_arr = self._base_display_array()
        if base_arr is None:
            self._global_view_tray.clear_histogram(self._theme)
            self._canvas.set_raw_array(None)
            self._canvas.set_source(QPixmap(), reset_zoom=reset_zoom)
            self._canvas.setText("No scan")
            return
        try:
            display_arr = current_image_view_array(base_arr, self._current_view_points)
        except Exception:
            display_arr = np.asarray(base_arr, dtype=np.float64)
        self._display_arr = display_arr
        self._global_view_tray.set_array(display_arr)
        lo_pct, hi_pct = self._global_view_tray.percentile_bounds()
        try:
            vmin = percentile_value(display_arr, lo_pct)
            vmax = percentile_value(display_arr, hi_pct)
        except Exception:
            self._global_view_tray.clear_histogram(self._theme)
            self._canvas.set_raw_array(display_arr)
            self._canvas.set_source(QPixmap(), reset_zoom=reset_zoom)
            return
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            center = float(vmin if np.isfinite(vmin) else 0.0)
            vmin = center - 1.0
            vmax = center + 1.0
        self._canvas.set_raw_array(display_arr)
        self._canvas.set_source(_array_pixmap(display_arr, vmin=vmin, vmax=vmax), reset_zoom=reset_zoom)

    def _on_flatten_toggled(self, checked: bool) -> None:
        self._base_display_arr = None
        self._display_arr = None
        if self._arr is not None:
            self._refresh_display_preview()

    def _on_view_percentiles_changed(self, lo: float, hi: float) -> None:
        if self._arr is None:
            return
        self._display_arr = None
        self._refresh_display_preview()

    def _on_current_flatten_toggled(self, checked: bool) -> None:
        if checked:
            self._start_current_image_flatten()
        else:
            self._cancel_current_image_flatten()

    def _start_current_image_flatten(self) -> None:
        if self._arr is None:
            self._current_view_tray.set_flatten_armed(False)
            return
        self._current_view_points = []
        self._current_view_capture = []
        self._canvas.set_zero_markers([])
        self._canvas.set_set_zero_mode(True)
        self._refresh_display_preview()
        self.status_message.emit("3 point flatten armed")

    def _cancel_current_image_flatten(self, *, refresh: bool = True) -> None:
        self._current_view_capture = []
        self._canvas.set_zero_markers([])
        self._canvas.set_set_zero_mode(False)
        self._current_view_tray.set_flatten_armed(False)
        if refresh and self._arr is not None:
            self._refresh_display_preview()

    def clear_current_image_flatten(self) -> None:
        self._current_view_points = []
        self._cancel_current_image_flatten(refresh=False)
        if self._arr is not None:
            self._refresh_display_preview()
        self.status_message.emit("Current image flatten cleared")

    def _on_current_view_point_clicked(self, frac_x: float, frac_y: float) -> None:
        if self._arr is None or not self._current_view_tray.is_flatten_armed():
            return
        base_arr = self._base_display_array()
        if base_arr is None:
            return
        width = int(base_arr.shape[1])
        height = int(base_arr.shape[0])
        x_px = max(0, min(width - 1, int(round(float(frac_x) * width))))
        y_px = max(0, min(height - 1, int(round(float(frac_y) * height))))
        self._current_view_capture.append((x_px, y_px))
        markers = [
            {"label": str(i + 1), "frac_x": px / max(width, 1), "frac_y": py / max(height, 1)}
            for i, (px, py) in enumerate(self._current_view_capture)
        ]
        self._canvas.set_zero_markers(markers)
        if len(self._current_view_capture) < 3:
            self.status_message.emit(f"3 point flatten: point {len(self._current_view_capture)} / 3")
            return
        candidate_points = self._current_view_capture[:3]
        try:
            corrected = current_image_view_array(base_arr, candidate_points)
        except Exception as exc:
            self._current_view_capture = []
            self._canvas.set_zero_markers([])
            self._canvas.set_set_zero_mode(False)
            self._current_view_tray.set_flatten_armed(False)
            self.status_message.emit(f"3 point flatten failed: {exc}")
            return
        self._current_view_points = candidate_points
        self._current_view_capture = []
        self._canvas.set_zero_markers([])
        self._canvas.set_set_zero_mode(False)
        self._current_view_tray.set_flatten_armed(False)
        self._display_arr = corrected
        self._refresh_display_preview()
        self.status_message.emit("Current image display flattened")

    def toggle_view_tray(self) -> None:
        self._global_view_tray.set_expanded(not self._global_view_tray.is_expanded())

    def view_tray_widget(self) -> DatasetBuilderViewTray:
        return self._global_view_tray

    def current_view_tray_widget(self) -> DatasetBuilderCurrentViewTray:
        return self._current_view_tray

    def _load_existing_mask(self, item: DatasetQueueItem) -> None:
        try:
            mask_set, _path = load_mask_set_sidecar(item.source_path, missing_ok=True)
        except Exception as exc:
            self.status_message.emit(f"Could not load mask sidecar: {exc}")
            return
        if mask_set is None:
            return
        mask = mask_set.get_by_name("step_edge") or mask_set.active()
        if mask is not None:
            self._current_mask = mask
            self._show_mask(mask.data)

    def _quickseg_preproc_key(self) -> QuickSegPreprocKey | None:
        if self._arr is None or self._quickseg_controls is None or self._current_sample_key() is None:
            return None
        params = self._quickseg_controls.parameters()
        self._quickseg_state.params = params
        return QuickSegPreprocKey(
            self._current_sample_key(),
            quickseg_params_fingerprint(params),
        )

    def _quickseg_prepare_cache(self) -> None:
        """Synchronously ensure the current QuickSeg preprocessing cache exists."""

        if self._arr is None or self._quickseg_controls is None:
            return
        key = self._quickseg_preproc_key()
        if key is None:
            return
        cached = self._sample_cache.get_preproc(key)
        if cached is not None:
            self._quickseg_cache = cached
            return
        self._quickseg_cache = prepare_quickseg_inputs(
            self._arr,
            self._quickseg_state.params,
            pixel_size_x_m=self._px_x_m,
            pixel_size_y_m=self._px_y_m,
        )
        self._sample_cache.put_preproc(key, self._quickseg_cache)

    def _quickseg_prepare_async(self) -> None:
        if self._task() != "terrace_segmentation" or self._arr is None or self._quickseg_controls is None:
            return
        key = self._quickseg_preproc_key()
        if key is None:
            return
        cached = self._sample_cache.get_preproc(key)
        if cached is not None:
            self._quickseg_cache = cached
            self._quickseg_update_status("QuickSeg preprocessing ready")
            if self._quickseg_state.seeds:
                self._quickseg_refresh_watershed(force_sync=True)
            return
        self._quickseg_prep_token = object()
        self._quickseg_cache = None
        self._quickseg_update_status("QuickSeg preprocessing...")
        worker = QuickSegPrepWorker(
            key,
            self._arr,
            self._quickseg_state.params,
            self._px_x_m,
            self._px_y_m,
            self._quickseg_prep_token,
        )
        worker.signals.finished.connect(self._on_quickseg_prepared)
        worker.signals.failed.connect(self._on_quickseg_prep_failed)
        self._interactive_pool.start(worker)

    def _on_quickseg_prepared(self, key, prepared, token) -> None:
        self._sample_cache.put_preproc(key, prepared)
        if token is not self._quickseg_prep_token or key != self._quickseg_preproc_key():
            return
        self._quickseg_cache = prepared
        self._quickseg_update_status("QuickSeg preprocessing ready")
        if self._quickseg_state.seeds:
            self._quickseg_refresh_watershed()

    def _on_quickseg_prep_failed(self, key, message: str, token) -> None:
        if token is not self._quickseg_prep_token or key != self._quickseg_preproc_key():
            return
        self.status_message.emit(f"QuickSeg preprocessing failed: {message}")

    def _quickseg_watershed_key(self) -> QuickSegWatershedKey | None:
        preproc_key = self._quickseg_preproc_key()
        if preproc_key is None:
            return None
        return QuickSegWatershedKey(preproc_key, quickseg_seed_fingerprint(self._quickseg_state.seeds))

    def _quickseg_mark_dirty(self) -> None:
        self._quickseg_result = None
        self._quickseg_state.result = None
        self._quickseg_overlay_dirty = True
        self._quickseg_update_overlay()
        self._quickseg_update_status("Watershed pending")

    def _quickseg_update_overlay(self) -> None:
        if self._task() != "terrace_segmentation" or self._arr is None:
            self._canvas.clear_rgba_overlay()
            return
        if not self._quickseg_overlay_visible:
            self._canvas.clear_rgba_overlay()
            return
        labels = self._quickseg_result
        if labels is None:
            labels = np.zeros(self._arr.shape, dtype=np.int32)
        rgba = quickseg_overlay_rgba(
            labels,
            self._quickseg_state.seeds,
            show_seeds=self._quickseg_controls.show_seeds() if self._quickseg_controls else True,
            show_boundaries=self._quickseg_controls.show_boundaries() if self._quickseg_controls else True,
            show_filled_regions=self._quickseg_controls.show_filled_regions() if self._quickseg_controls else True,
            opacity=self._quickseg_controls.overlay_opacity() if self._quickseg_controls else 0.55,
        )
        self._canvas.set_rgba_overlay(rgba)

    def _quickseg_update_status(self, prefix: str | None = None) -> None:
        if self._current_index < 0 or self._current_index >= len(self._queue):
            return
        item = self._queue[self._current_index]
        seed_count = len(self._quickseg_state.seeds)
        result_state = "ready" if self._quickseg_result is not None else "pending"
        text = prefix or "QuickSeg ready"
        self._canvas_status.setText(
            f"{item.display_id} | task: {self._task()} | status: {item.status} | "
            f"seeds: {seed_count} | result: {result_state}"
        )
        if self._quickseg_controls is not None:
            self._quickseg_controls.set_current_label(self._quickseg_state.current_label)
            self._quickseg_controls.set_seed_mode_status(
                f"Add seed mode (label {self._quickseg_state.current_label})"
            )
            if self._quickseg_result is None:
                self._quickseg_controls.set_result_status(text or "Watershed pending")
            else:
                self._quickseg_controls.set_result_status(
                    f"Watershed ready ({self._quickseg_result.shape[0]}x{self._quickseg_result.shape[1]})"
                )

    def _quickseg_new_label(self) -> None:
        self._quickseg_state.current_label = max(1, self._quickseg_state.current_label + 1)
        self._quickseg_update_status("New terrace label")

    def _quickseg_add_seed(self, x: int, y: int, *, new_terrace: bool = False) -> None:
        if self._arr is None:
            return
        if new_terrace:
            prev_label = self._quickseg_state.current_label
            self._quickseg_state.current_label = max(1, prev_label + 1)
            label = self._quickseg_state.current_label
        else:
            prev_label = self._quickseg_state.current_label
            label = prev_label
        seed = QuickSegSeed(
            x=int(x),
            y=int(y),
            terrace_label_id=int(label),
            order=int(self._quickseg_state.next_order),
        )
        self._quickseg_state.next_order += 1
        self._quickseg_state.seeds.append(seed)
        action = {"action": "new_seed" if new_terrace else "add_seed", "seed": seed, "previous_label": prev_label}
        self._quickseg_seed_history.append(action)
        self._quickseg_mark_dirty()
        self._quickseg_refresh_watershed()
        self._quickseg_update_status("Seed added")

    def _quickseg_delete_seed_at(self, x: int, y: int) -> None:
        if not self._quickseg_state.seeds:
            return
        best_idx = None
        best_dist = None
        for idx, seed in enumerate(self._quickseg_state.seeds):
            dist = (int(seed.x) - int(x)) ** 2 + (int(seed.y) - int(y)) ** 2
            if best_dist is None or dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is None:
            return
        removed = self._quickseg_state.seeds.pop(best_idx)
        self._quickseg_seed_history.append({"action": "delete_seed", "seed": removed, "index": best_idx})
        self._quickseg_state.current_label = max((s.terrace_label_id for s in self._quickseg_state.seeds), default=1)
        self._quickseg_mark_dirty()
        self._quickseg_refresh_watershed()
        self._quickseg_update_status("Seed deleted")

    def _quickseg_canvas_clicked(self, x: int, y: int, modifiers: int) -> None:
        if self._task() != "terrace_segmentation":
            return
        mods = int(getattr(modifiers, "value", modifiers))
        alt_mod = int(getattr(Qt.AltModifier, "value", Qt.AltModifier))
        ctrl_mod = int(getattr(Qt.ControlModifier, "value", Qt.ControlModifier))
        meta_mod = int(getattr(Qt.MetaModifier, "value", Qt.MetaModifier))
        if mods & alt_mod:
            self._quickseg_delete_seed_at(x, y)
            return
        if mods & (ctrl_mod | meta_mod):
            self._quickseg_add_seed(x, y, new_terrace=True)
            return
        self._quickseg_add_seed(x, y, new_terrace=False)

    def _quickseg_undo_last_seed(self) -> None:
        if not self._quickseg_seed_history:
            self.status_message.emit("QuickSeg undo stack is empty")
            return
        action = self._quickseg_seed_history.pop()
        kind = str(action.get("action"))
        if kind in {"add_seed", "new_seed"}:
            seed = action["seed"]
            self._quickseg_state.seeds = [
                s for s in self._quickseg_state.seeds
                if not (s.order == seed.order and s.terrace_label_id == seed.terrace_label_id and s.x == seed.x and s.y == seed.y)
            ]
            if kind == "new_seed":
                self._quickseg_state.current_label = int(action.get("previous_label", 1))
            else:
                self._quickseg_state.current_label = max((s.terrace_label_id for s in self._quickseg_state.seeds), default=1)
        elif kind == "delete_seed":
            seed = action["seed"]
            index = int(action.get("index", len(self._quickseg_state.seeds)))
            self._quickseg_state.seeds.insert(min(max(index, 0), len(self._quickseg_state.seeds)), seed)
            self._quickseg_state.current_label = max((s.terrace_label_id for s in self._quickseg_state.seeds), default=1)
        self._quickseg_mark_dirty()
        self._quickseg_refresh_watershed()
        self.status_message.emit("Undid QuickSeg seed action")

    def _quickseg_clear_seeds(self) -> None:
        self._quickseg_state.seeds = []
        self._quickseg_state.current_label = 1
        self._quickseg_state.next_order = 1
        self._quickseg_seed_history.clear()
        self._quickseg_mark_dirty()
        self._quickseg_refresh_watershed()
        self.status_message.emit("QuickSeg seeds cleared")

    def _quickseg_clear_result(self) -> None:
        self._quickseg_result = None
        self._quickseg_state.result = None
        self._quickseg_overlay_dirty = True
        self._quickseg_update_overlay()
        self._quickseg_update_status("QuickSeg result cleared")

    def _quickseg_refresh_watershed(self, *, force_sync: bool = False) -> None:
        if self._task() != "terrace_segmentation":
            return
        if self._arr is None:
            self.status_message.emit("No scan loaded")
            return
        if self._quickseg_controls is None:
            return
        if self._quickseg_cache is None:
            if force_sync:
                self._quickseg_prepare_cache()
            else:
                self._quickseg_prepare_async()
                return
        if self._quickseg_cache is None:
            self.status_message.emit("QuickSeg preprocessing unavailable")
            return
        w_key = self._quickseg_watershed_key()
        if w_key is None:
            self.status_message.emit("QuickSeg preprocessing unavailable")
            return
        cached = self._sample_cache.get_watershed(w_key)
        if cached is not None:
            self._quickseg_result = cached
            self._quickseg_state.result = cached
            self._quickseg_state.result_path = str(self._quickseg_result_path) if self._quickseg_result_path else None
            self._quickseg_overlay_dirty = False
            self._quickseg_update_overlay()
            self._quickseg_update_status("QuickSeg watershed refreshed")
            self.status_message.emit("QuickSeg watershed refreshed")
            return
        if not self._quickseg_state.seeds:
            self._quickseg_result = None
            self._quickseg_state.result = None
            self._quickseg_overlay_dirty = False
            self._quickseg_update_overlay()
            self._quickseg_update_status("QuickSeg seeds required")
            return
        if not force_sync:
            self._quickseg_ws_token = object()
            worker = QuickSegWatershedWorker(
                w_key,
                self._quickseg_cache,
                self._quickseg_state.seeds,
                self._quickseg_state.params,
                self._quickseg_ws_token,
            )
            worker.signals.finished.connect(self._on_quickseg_watershed_ready)
            worker.signals.failed.connect(self._on_quickseg_watershed_failed)
            self._quickseg_update_status("QuickSeg watershed running...")
            self._interactive_pool.start(worker)
            return
        self._quickseg_result = watershed_labels(
            self._quickseg_cache,
            self._quickseg_state.seeds,
            self._quickseg_state.params,
        )
        self._sample_cache.put_watershed(w_key, self._quickseg_result)
        self._quickseg_state.result = self._quickseg_result
        self._quickseg_state.result_path = str(self._quickseg_result_path) if self._quickseg_result_path else None
        self._quickseg_overlay_dirty = False
        self._quickseg_update_overlay()
        self._quickseg_update_status("QuickSeg watershed refreshed")
        self.status_message.emit("QuickSeg watershed refreshed")

    def _on_quickseg_watershed_ready(self, key, labels, token) -> None:
        self._sample_cache.put_watershed(key, labels)
        if token is not self._quickseg_ws_token or key != self._quickseg_watershed_key():
            return
        self._quickseg_result = labels
        self._quickseg_state.result = labels
        self._quickseg_state.result_path = str(self._quickseg_result_path) if self._quickseg_result_path else None
        self._quickseg_overlay_dirty = False
        self._quickseg_update_overlay()
        self._quickseg_update_status("QuickSeg watershed refreshed")
        self.status_message.emit("QuickSeg watershed refreshed")

    def _on_quickseg_watershed_failed(self, key, message: str, token) -> None:
        if token is not self._quickseg_ws_token or key != self._quickseg_watershed_key():
            return
        self.status_message.emit(f"QuickSeg watershed failed: {message}")

    def _apply_quickseg_params(self) -> None:
        if self._task() != "terrace_segmentation":
            return
        if self._quickseg_controls is None:
            return
        self._quickseg_state.params = self._quickseg_controls.parameters()
        self._quickseg_cache = None
        self._quickseg_mark_dirty()
        self._quickseg_prepare_async()
        if not self._quickseg_controls.auto_refresh_after_apply() or not self._quickseg_state.seeds:
            self.status_message.emit("QuickSeg parameters applied")

    def _quickseg_clear_result_only(self) -> None:
        self._quickseg_result = None
        self._quickseg_state.result = None
        self._quickseg_overlay_dirty = True
        self._quickseg_update_overlay()

    def _load_existing_quickseg(self, item: DatasetQueueItem) -> None:
        if self._quickseg_controls is None:
            return
        try:
            self._quickseg_state, self._quickseg_review_record, self._quickseg_result_path = load_quickseg_state(
                item.source_path,
                config=self._config(),
            )
        except Exception as exc:
            self.status_message.emit(f"Could not load QuickSeg sidecar: {exc}")
            self._quickseg_state = QuickSegState()
            self._quickseg_result = None
            self._quickseg_result_path = None
            self._quickseg_prepare_cache()
            self._quickseg_update_overlay()
            self._quickseg_update_status("QuickSeg ready")
            return
        self._quickseg_controls.set_parameters(self._quickseg_state.params)
        self._quickseg_controls.set_current_label(self._quickseg_state.current_label)
        self._quickseg_seed_points = list(self._quickseg_state.seeds)
        self._quickseg_seed_history.clear()
        self._quickseg_cache = None
        self._quickseg_result = self._quickseg_state.result
        self._quickseg_prepare_async()
        if self._quickseg_result is not None:
            self._quickseg_update_overlay()
            self._quickseg_update_status("QuickSeg ready")

    def set_paint_mode(self, mode: str) -> None:
        if mode not in {"brush", "eraser"}:
            return
        self._paint_mode = mode
        self._brush_btn.setChecked(mode == "brush")
        self._eraser_btn.setChecked(mode == "eraser")
        self._canvas.set_paint_enabled(True)
        self.status_message.emit(f"{mode.capitalize()} mode")

    def undo_paint(self) -> None:
        if self._current_mask is None or not self._undo_stack:
            self.status_message.emit("Undo stack is empty")
            return
        self._current_mask.data = self._undo_stack.pop()
        self._show_mask(self._current_mask.data)
        self._update_canvas_status()
        self.status_message.emit("Undid last mask edit")

    def _paint_at(self, x: int, y: int) -> None:
        if self._arr is None or not (0 <= self._current_index < len(self._queue)):
            return
        if self._current_mask is None:
            self._current_mask = ImageMask.new(
                np.zeros(self._arr.shape, dtype=bool),
                method="manual",
                parameters={
                    "dataset_builder_task": self._task(),
                    "label_name": "step_edge",
                    "data_basis": "scan_plane",
                },
                name="step_edge",
            )
        edited, changed = paint_mask(
            self._current_mask.data,
            x=x,
            y=y,
            radius=self._brush_spin.value(),
            value=self._paint_mode == "brush",
        )
        if not changed:
            return
        self._current_mask.data = edited
        self._stroke_dirty = True
        self._mark_mask_edited()
        self._overlay_visible = True
        self._show_mask(edited)
        self._update_canvas_status()

    def _begin_paint_stroke(self) -> None:
        if self._arr is None or not (0 <= self._current_index < len(self._queue)):
            self._stroke_snapshot = None
            self._stroke_dirty = False
            return
        if self._current_mask is None:
            self._stroke_snapshot = np.zeros(self._arr.shape, dtype=bool)
        else:
            self._stroke_snapshot = self._current_mask.data.copy()
        self._stroke_dirty = False

    def _end_paint_stroke(self) -> None:
        if self._stroke_snapshot is None:
            self._stroke_dirty = False
            return
        if self._stroke_dirty and self._current_mask is not None:
            if not np.array_equal(self._current_mask.data, self._stroke_snapshot):
                self._undo_stack.append(self._stroke_snapshot)
                if len(self._undo_stack) > 25:
                    self._undo_stack.pop(0)
        self._stroke_snapshot = None
        self._stroke_dirty = False

    def _mark_mask_edited(self) -> None:
        if self._current_mask is None:
            return
        if not str(self._current_mask.method).endswith("+manual"):
            self._current_mask.method = f"{self._current_mask.method}+manual"
        params = dict(self._current_mask.parameters)
        params["human_corrected"] = True
        params["edit_count"] = int(params.get("edit_count") or 0) + 1
        params["last_edit_mode"] = self._paint_mode
        self._current_mask.parameters = params

    def _update_canvas_status(self) -> None:
        if not (0 <= self._current_index < len(self._queue)):
            return
        item = self._queue[self._current_index]
        coverage = 100.0 * float(self._current_mask.data.mean()) if self._current_mask else 0.0
        self._canvas_status.setText(
            f"{item.display_id} | task: {self._task()} | status: {item.status} | "
            f"mask coverage: {coverage:.2f}%"
        )

    def _show_mask(self, data: np.ndarray) -> None:
        if self._overlay_visible:
            self._canvas.set_mask_overlay(data)

    def save_current(self) -> bool:
        if not (0 <= self._current_index < len(self._queue)):
            self.status_message.emit("No scan loaded")
            return False
        item = self._queue[self._current_index]
        status = self._status_combo.currentText()
        config = self._config()
        try:
            if self._task() == "terrace_segmentation":
                if self._quickseg_controls is None:
                    self.status_message.emit("QuickSeg controls unavailable")
                    return False
                if self._quickseg_cache is None and self._arr is not None:
                    self._quickseg_prepare_cache()
                if self._quickseg_result is None and self._quickseg_state.seeds:
                    self._quickseg_refresh_watershed(force_sync=True)
                self._quickseg_state.result = self._quickseg_result
                result_path, state_path = save_quickseg_state(
                    item.source_path,
                    self._quickseg_state,
                    config=config,
                    status=status,
                )
                self._quickseg_result_path = result_path
                self._quickseg_review_record = state_path
            else:
                if self._current_mask is None:
                    self.status_message.emit("Generate or load a mask before saving")
                    return False
                save_mask_annotation(
                    item.source_path,
                    self._current_mask,
                    config=config,
                    status=status,
                )
        except Exception as exc:
            self.status_message.emit(f"Save failed: {exc}")
            return False
        self._queue[self._current_index] = DatasetQueueItem(
            source_path=item.source_path,
            plane_index=item.plane_index,
            display_id=item.display_id,
            status=status,
            has_mask_sidecar=item.has_mask_sidecar or self._task() == "step_edge_mask",
            has_roi_sidecar=item.has_roi_sidecar,
            reviewed_at=item.reviewed_at,
            exported_at=item.exported_at,
            load_error=item.load_error,
        )
        self._refresh_queue_list()
        self.status_message.emit(f"Saved {item.display_id} as {status}")
        return True

    def save_and_next(self) -> None:
        if self.save_current():
            self.next_item()

    def _save_status_and_next(self, status: str) -> None:
        if self._set_status(status):
            self.next_item()

    def _set_status(self, status: str) -> bool:
        self._status_combo.setCurrentText(status)
        if self._task() == "terrace_segmentation":
            return self.save_current()
        if status == "rejected" and self._current_mask is None and 0 <= self._current_index < len(self._queue):
            item = self._queue[self._current_index]
            try:
                save_review_annotation(
                    item.source_path,
                    config=self._config(),
                    status=status,
                )
            except Exception as exc:
                self.status_message.emit(f"Save failed: {exc}")
                return
            self._queue[self._current_index] = DatasetQueueItem(
                source_path=item.source_path,
                plane_index=item.plane_index,
                display_id=item.display_id,
                status=status,
                has_mask_sidecar=item.has_mask_sidecar,
                has_roi_sidecar=item.has_roi_sidecar,
                reviewed_at=item.reviewed_at,
                exported_at=item.exported_at,
                load_error=item.load_error,
            )
            self._refresh_queue_list()
            self.status_message.emit(f"Marked {item.display_id} rejected")
            return True
        return self.save_current()

    def clear_overlay(self) -> None:
        if self._task() == "terrace_segmentation":
            self._quickseg_clear_result()
            return
        self._current_mask = None
        self._canvas.clear_mask_overlay()
        self.status_message.emit("Current proposal cleared")

    def toggle_overlay(self) -> None:
        if self._task() == "terrace_segmentation":
            self.toggle_quickseg_overlay()
            return
        self._overlay_visible = not self._overlay_visible
        if not self._overlay_visible:
            self._canvas.clear_mask_overlay()
        elif self._current_mask is not None:
            self._show_mask(self._current_mask.data)

    def toggle_quickseg_overlay(self) -> None:
        self._quickseg_overlay_visible = not self._quickseg_overlay_visible
        self._quickseg_update_overlay()
        self.status_message.emit(
            "QuickSeg overlay shown" if self._quickseg_overlay_visible else "QuickSeg overlay hidden"
        )

    def previous_item(self) -> None:
        if self._queue:
            self._current_index = max(0, self._current_index - 1)
            self._refresh_queue_list()
            self._load_current()

    def next_item(self) -> None:
        if self._queue:
            self._current_index = min(len(self._queue) - 1, self._current_index + 1)
            self._refresh_queue_list()
            self._load_current()

    def _export_dataset(self) -> None:
        source = self._source_entry.text().strip()
        if not source:
            return
        out = QFileDialog.getExistingDirectory(self, "Dataset export folder")
        if not out:
            return
        self._export_token = object()
        spec = DatasetExportSpec(
            source=Path(source),
            output_dir=Path(out),
            task=self._task(),
            label_name="step_edge" if self._task() == "step_edge_mask" else "quickseg_terraces",
            plane_index=self._plane_spin.value(),
            include_statuses=("accepted", "uncertain"),
            overwrite=False,
        )
        worker = DatasetBuilderExportWorker(spec)
        worker.signals.finished.connect(self._on_export_finished)
        worker.signals.failed.connect(self._on_export_failed)
        self.status_message.emit("Exporting dataset snapshot...")
        self._export_pool.start(worker)

    def _on_export_finished(self, summary) -> None:
        if summary is None:
            return
        self.status_message.emit(
            f"Exported {summary.get('n_exported', 0)} sample(s) to {summary.get('output_dir', '')}"
        )

    def _on_export_failed(self, message: str) -> None:
        self.status_message.emit(f"Dataset export failed: {message}")


class DatasetBuilderSidebar(QWidget):
    """Small outer sidebar for the top-level Dataset Builder mode."""

    def __init__(self, theme: dict, parent=None):
        super().__init__(parent)
        self._theme = dict(theme)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(8, 8, 8, 8)
        self._lay.setSpacing(6)
        self._title = QLabel("<b>Dataset Builder</b>")
        self._counts = QLabel("Queue not loaded")
        self._counts.setWordWrap(True)
        self._global_view_host = QWidget()
        self._global_view_host_lay = QVBoxLayout(self._global_view_host)
        self._global_view_host_lay.setContentsMargins(0, 0, 0, 0)
        self._global_view_host_lay.setSpacing(0)
        self._current_view_host = QWidget()
        self._current_view_host_lay = QVBoxLayout(self._current_view_host)
        self._current_view_host_lay.setContentsMargins(0, 0, 0, 0)
        self._current_view_host_lay.setSpacing(0)
        self._global_view_tray = None
        self._current_view_tray = None
        self._lay.addWidget(self._title)
        self._lay.addWidget(self._global_view_host)
        self._lay.addWidget(self._current_view_host)
        self._lay.addWidget(self._counts)
        self._lay.addStretch(1)
        self.apply_theme(theme)

    def apply_theme(self, theme: dict) -> None:
        self._theme = dict(theme)

    def set_global_view_tray(self, tray: QWidget) -> None:
        if self._global_view_tray is tray:
            return
        if self._global_view_tray is not None:
            self._global_view_host_lay.removeWidget(self._global_view_tray)
            self._global_view_tray.setParent(None)
        self._global_view_tray = tray
        if tray is not None:
            self._global_view_host_lay.addWidget(tray)

    def set_current_view_tray(self, tray: QWidget) -> None:
        if self._current_view_tray is tray:
            return
        if self._current_view_tray is not None:
            self._current_view_host_lay.removeWidget(self._current_view_tray)
            self._current_view_tray.setParent(None)
        self._current_view_tray = tray
        if tray is not None:
            self._current_view_host_lay.addWidget(tray)

    def set_view_tray(self, tray: QWidget) -> None:
        self.set_global_view_tray(tray)

    def set_counts(self, counts: dict) -> None:
        self._counts.setText(
            "\n".join(
                [
                    f"Total: {counts.get('total', 0)}",
                    f"Accepted: {counts.get('accepted', 0)}",
                    f"Uncertain: {counts.get('uncertain', 0)}",
                    f"Rejected: {counts.get('rejected', 0)}",
                    f"Blank: {counts.get('blank', 0)}",
                ]
            )
        )


def _array_pixmap(arr: np.ndarray, *, vmin: float | None = None, vmax: float | None = None) -> QPixmap:
    u8 = array_to_uint8(arr, vmin=vmin, vmax=vmax)
    h, w = u8.shape
    qimg = QImage(u8.data, w, h, u8.strides[0], QImage.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimg)


def _status_symbol(status: str) -> str:
    return {
        "blank": "--",
        "draft": "..",
        "accepted": "OK",
        "uncertain": "?",
        "rejected": "NO",
        "exported": "EX",
    }.get(status, "--")
