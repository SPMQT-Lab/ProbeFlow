"""Top-level Dataset Builder cockpit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
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
    proposal_to_mask,
    save_mask_annotation,
    save_review_annotation,
)
from probeflow.dataset_builder.export import export_dataset
from probeflow.dataset_builder.loading import load_scan_plane
from probeflow.dataset_builder.models import DatasetExportSpec, DatasetQueueItem, DatasetTaskConfig
from probeflow.dataset_builder.painting import paint_mask
from probeflow.dataset_builder.proposals import generate_proposal
from probeflow.dataset_builder.queue import build_queue, queue_counts
from probeflow.gui.dataset_builder.display import (
    dataset_builder_display_array,
    percentile_value,
)
from probeflow.gui.dataset_builder.canvas import DatasetBuilderCanvas
from probeflow.gui.dataset_builder.view_tray import DatasetBuilderViewTray
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
        self._display_arr: np.ndarray | None = None

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

        self._view_tray = DatasetBuilderViewTray(theme, self)
        self._view_tray.set_expanded(True)
        self._view_tray.percentiles_changed.connect(self._on_view_percentiles_changed)
        self._view_tray.flatten_toggled.connect(self._on_flatten_toggled)

        form = QFormLayout()
        self._task_combo = QComboBox()
        self._task_combo.addItem("Step-edge / ignore-mask", "step_edge_mask")
        self._method_combo = QComboBox()
        self._method_combo.addItems(["step_edge", "canny"])
        self._plane_spin = QSpinBox()
        self._plane_spin.setRange(0, 64)
        self._status_combo = QComboBox()
        self._status_combo.addItems(["draft", "accepted", "uncertain", "rejected"])
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(1, 128)
        self._brush_spin.setValue(8)
        form.addRow("Preset", self._task_combo)
        form.addRow("Proposal", self._method_combo)
        form.addRow("Plane", self._plane_spin)
        form.addRow("Status", self._status_combo)
        form.addRow("Brush px", self._brush_spin)
        right_lay.addLayout(form)

        propose_btn = QPushButton("Generate Proposal")
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
            propose_btn,
            brush_btn,
            eraser_btn,
            save_btn,
            save_next_btn,
            accept_btn,
            uncertain_btn,
            reject_btn,
            clear_btn,
        ):
            right_lay.addWidget(btn)
        right_lay.addStretch(1)
        body.addWidget(right)
        body.setSizes([240, 760, 260])

        self._shortcut_lbl = QLabel(
            "A prev | D next | F save+next | Q overlay | W proposal | "
            "E eraser | R brush | Z undo | X reject | C clear | V view | "
            "1 accept | 2 uncertain | 3 reject | [ / ] brush"
        )
        self._shortcut_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self._shortcut_lbl)

        browse_btn.clicked.connect(self._browse_source)
        load_btn.clicked.connect(self.load_queue)
        export_btn.clicked.connect(self._export_dataset)
        self._queue_list.currentRowChanged.connect(self._on_queue_row_changed)
        self._filter_combo.currentTextChanged.connect(lambda _text: self._refresh_queue_list())
        self._plane_spin.valueChanged.connect(lambda _v: self.load_queue() if self._source_entry.text() else None)
        propose_btn.clicked.connect(self.generate_proposal)
        brush_btn.clicked.connect(lambda: self.set_paint_mode("brush"))
        eraser_btn.clicked.connect(lambda: self.set_paint_mode("eraser"))
        save_btn.clicked.connect(self.save_current)
        save_next_btn.clicked.connect(self.save_and_next)
        accept_btn.clicked.connect(lambda: self._set_status("accepted"))
        uncertain_btn.clicked.connect(lambda: self._set_status("uncertain"))
        reject_btn.clicked.connect(lambda: self._set_status("rejected"))
        clear_btn.clicked.connect(self.clear_overlay)
        self._brush_btn = brush_btn
        self._eraser_btn = eraser_btn
        self._canvas.mask_painted.connect(self._paint_at)
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
        self._view_tray.apply_theme(theme)
        if self._arr is not None:
            self._refresh_display_preview()

    def set_source(self, source: str | Path) -> None:
        self._source_entry.setText(str(source))
        self.load_queue()

    def _install_shortcuts(self) -> None:
        bindings = {
            "A": self.previous_item,
            "D": self.next_item,
            "F": self.save_and_next,
            "Q": self.toggle_overlay,
            "W": self.generate_proposal,
            "C": self.clear_overlay,
            "X": lambda: self._set_status("rejected"),
            "1": lambda: self._set_status("accepted"),
            "2": lambda: self._set_status("uncertain"),
            "3": lambda: self._set_status("rejected"),
            "[": lambda: self._brush_spin.setValue(max(1, self._brush_spin.value() - 1)),
            "]": lambda: self._brush_spin.setValue(self._brush_spin.value() + 1),
        }
        for key, slot in bindings.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
        extra = {
            "E": lambda: self.set_paint_mode("eraser"),
            "R": lambda: self.set_paint_mode("brush"),
            "Z": self.undo_paint,
            "V": self.toggle_view_tray,
        }
        for key, slot in extra.items():
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)

    def _browse_source(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Dataset Builder source folder")
        if path:
            self.set_source(path)

    def load_queue(self) -> None:
        source = self._source_entry.text().strip()
        if not source:
            return
        try:
            self._queue = build_queue(
                source,
                task=self._task(),
                label_name="step_edge",
                plane_index=self._plane_spin.value(),
            )
        except Exception as exc:
            self.status_message.emit(f"Dataset Builder queue failed: {exc}")
            return
        self._current_index = 0 if self._queue else -1
        self._refresh_queue_list()
        self._load_current()
        self.status_message.emit(f"Dataset Builder queue loaded: {len(self._queue)} scan(s)")

    def _task(self) -> str:
        return str(self._task_combo.currentData() or "step_edge_mask")

    def _config(self, *, status: str | None = None) -> DatasetTaskConfig:
        return DatasetTaskConfig(
            task=self._task(),
            label_name="step_edge",
            label_type="mask",
            proposal_method=self._method_combo.currentText(),
            plane_index=self._plane_spin.value(),
        )

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

    def _load_current(self) -> None:
        self._current_mask = None
        self._overlay_visible = True
        self._undo_stack.clear()
        self._canvas.clear_mask_overlay()
        if not (0 <= self._current_index < len(self._queue)):
            self._arr = None
            self._display_arr = None
            self._view_tray.clear_histogram(self._theme)
            self._canvas.setText("No scan")
            self._canvas_status.setText("No scan loaded")
            return
        item = self._queue[self._current_index]
        try:
            self._scan, self._arr, self._px_x_m, self._px_y_m = load_scan_plane(
                item.source_path,
                item.plane_index,
            )
        except Exception as exc:
            self.status_message.emit(f"Could not load {item.source_path.name}: {exc}")
            return
        self._display_arr = None
        self._refresh_display_preview(reset_zoom=True)
        self._load_existing_mask(item)
        coverage = 100.0 * float(self._current_mask.data.mean()) if self._current_mask else 0.0
        self._canvas_status.setText(
            f"{item.display_id} | task: {self._task()} | status: {item.status} | "
            f"mask coverage: {coverage:.2f}%"
        )

    def _current_display_array(self) -> np.ndarray | None:
        if self._arr is None:
            return None
        if self._display_arr is not None:
            return self._display_arr
        self._display_arr = dataset_builder_display_array(
            self._arr,
            flatten=self._view_tray.is_flatten_enabled(),
        )
        return self._display_arr

    def _refresh_display_preview(self, *_, reset_zoom: bool = False) -> None:
        display_arr = self._current_display_array()
        if display_arr is None:
            self._view_tray.clear_histogram(self._theme)
            self._canvas.set_raw_array(None)
            self._canvas.set_source(QPixmap(), reset_zoom=reset_zoom)
            self._canvas.setText("No scan")
            return
        self._view_tray.set_array(display_arr)
        lo_pct, hi_pct = self._view_tray.percentile_bounds()
        try:
            vmin = percentile_value(display_arr, lo_pct)
            vmax = percentile_value(display_arr, hi_pct)
        except Exception:
            self._view_tray.clear_histogram(self._theme)
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
        self._display_arr = None
        if self._arr is not None:
            self._refresh_display_preview()

    def _on_view_percentiles_changed(self, lo: float, hi: float) -> None:
        if self._arr is None:
            return
        self._display_arr = None
        self._refresh_display_preview()

    def toggle_view_tray(self) -> None:
        self._view_tray.set_expanded(not self._view_tray.is_expanded())

    def view_tray_widget(self) -> DatasetBuilderViewTray:
        return self._view_tray

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
        before = self._current_mask.data.copy()
        edited, changed = paint_mask(
            self._current_mask.data,
            x=x,
            y=y,
            radius=self._brush_spin.value(),
            value=self._paint_mode == "brush",
        )
        if not changed:
            return
        self._undo_stack.append(before)
        if len(self._undo_stack) > 25:
            self._undo_stack.pop(0)
        self._current_mask.data = edited
        self._mark_mask_edited()
        self._overlay_visible = True
        self._show_mask(edited)
        self._update_canvas_status()

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

    def generate_proposal(self) -> None:
        if self._arr is None or not (0 <= self._current_index < len(self._queue)):
            return
        item = self._queue[self._current_index]
        channel = (
            self._scan.plane_names[item.plane_index]
            if self._scan is not None and item.plane_index < len(self._scan.plane_names)
            else f"plane {item.plane_index}"
        )
        try:
            config = self._config()
            proposal = generate_proposal(
                self._arr,
                px_x_m=self._px_x_m,
                px_y_m=self._px_y_m,
                config=config,
                source_channel=channel,
            )
            self._current_mask = proposal_to_mask(
                proposal,
                config=config,
                source_path=item.source_path,
                source_channel=channel,
            )
        except Exception as exc:
            self.status_message.emit(f"Proposal failed: {exc}")
            return
        self._overlay_visible = True
        self._show_mask(self._current_mask.data)
        self._canvas_status.setText(
            f"{item.display_id} | proposal: {self._current_mask.method} | "
            f"mask coverage: {100.0 * float(self._current_mask.data.mean()):.2f}%"
        )
        self.status_message.emit(f"Proposal generated for {item.display_id}")

    def save_current(self) -> bool:
        if self._current_mask is None or not (0 <= self._current_index < len(self._queue)):
            self.status_message.emit("Generate or load a mask before saving")
            return False
        item = self._queue[self._current_index]
        status = self._status_combo.currentText()
        try:
            save_mask_annotation(
                item.source_path,
                self._current_mask,
                config=self._config(),
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
            has_mask_sidecar=True,
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

    def _set_status(self, status: str) -> None:
        self._status_combo.setCurrentText(status)
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
            return
        self.save_current()

    def clear_overlay(self) -> None:
        self._current_mask = None
        self._canvas.clear_mask_overlay()
        self.status_message.emit("Current proposal cleared")

    def toggle_overlay(self) -> None:
        self._overlay_visible = not self._overlay_visible
        if not self._overlay_visible:
            self._canvas.clear_mask_overlay()
        elif self._current_mask is not None:
            self._show_mask(self._current_mask.data)

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
        try:
            summary = export_dataset(
                DatasetExportSpec(
                    source=Path(source),
                    output_dir=Path(out),
                    task=self._task(),
                    label_name="step_edge",
                    plane_index=self._plane_spin.value(),
                    include_statuses=("accepted", "uncertain"),
                    overwrite=False,
                )
            )
        except Exception as exc:
            self.status_message.emit(f"Dataset export failed: {exc}")
            return
        self.status_message.emit(
            f"Exported {summary['n_exported']} sample(s) to {summary['output_dir']}"
        )


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
        self._view_host = QWidget()
        self._view_host_lay = QVBoxLayout(self._view_host)
        self._view_host_lay.setContentsMargins(0, 0, 0, 0)
        self._view_host_lay.setSpacing(0)
        self._view_tray = None
        self._lay.addWidget(self._title)
        self._lay.addWidget(self._view_host)
        self._lay.addWidget(self._counts)
        self._lay.addStretch(1)
        self.apply_theme(theme)

    def apply_theme(self, theme: dict) -> None:
        self._theme = dict(theme)

    def set_view_tray(self, tray: QWidget) -> None:
        if self._view_tray is tray:
            return
        if self._view_tray is not None:
            self._view_host_lay.removeWidget(self._view_tray)
            self._view_tray.setParent(None)
        self._view_tray = tray
        if tray is not None:
            self._view_host_lay.addWidget(tray)

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
