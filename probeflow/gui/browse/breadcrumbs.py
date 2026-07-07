"""Breadcrumb navigation bar for the browse grid."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from probeflow.gui.typography import ui_font
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor, QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

# ── BreadcrumbBar ─────────────────────────────────────────────────────────────
class _BreadcrumbBar(QWidget):
    """Path strip with clickable segments + back/up buttons.

    Segments are clickable and emit ``segment_clicked(Path)``. Back/up buttons
    emit their own signals so the grid can decide whether they're enabled.
    """

    segment_clicked = Signal(object)  # Path
    back_requested  = Signal()
    up_requested    = Signal()

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self._t = t
        self._root: Optional[Path] = None
        self._current: Optional[Path] = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(4)

        self._back_btn = QPushButton("←")
        self._back_btn.setFixedSize(24, 24)
        self._back_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._back_btn.setEnabled(False)
        self._back_btn.setVisible(False)
        self._back_btn.clicked.connect(self.back_requested)
        lay.addWidget(self._back_btn)

        self._up_btn = QPushButton("↑")
        self._up_btn.setFixedSize(24, 24)
        self._up_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self._up_btn.setEnabled(False)
        self._up_btn.setVisible(False)
        self._up_btn.clicked.connect(self.up_requested)
        lay.addWidget(self._up_btn)

        self._segments_host = QWidget()
        self._segments_lay = QHBoxLayout(self._segments_host)
        self._segments_lay.setContentsMargins(6, 0, 0, 0)
        self._segments_lay.setSpacing(4)
        self._segments_lay.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._segments_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self._segments_host, 1)

        self.setFixedHeight(36)
        self.apply_theme(t)

    def apply_theme(self, t: dict):
        self._t = t
        self.setStyleSheet(f"background-color: {t['main_bg']};")
        for btn in (self._back_btn, self._up_btn):
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {t['card_bg']}; "
                f"color: {t['fg']}; border: 1px solid {t['sep']}; "
                f"border-radius: 3px; }}"
                f"QPushButton:hover:enabled {{ border: 1px solid {t['accent_bg']}; }}"
                f"QPushButton:disabled {{ color: {t['sub_fg']}; }}"
            )
        self._restyle_segments()

    def _restyle_segments(self):
        t = self._t
        for i in range(self._segments_lay.count()):
            w = self._segments_lay.itemAt(i).widget()
            if isinstance(w, QPushButton):
                w.setStyleSheet(
                    f"QPushButton {{ background: transparent; color: {t['fg']}; "
                    f"border: none; padding: 2px 6px; }}"
                    f"QPushButton:hover {{ color: {t['accent_bg']}; "
                    f"text-decoration: underline; }}"
                )
            elif isinstance(w, QLabel):
                w.setStyleSheet(f"color: {t['sub_fg']}; background: transparent;")

    def set_state(self, root: Optional[Path], current: Optional[Path],
                  *, can_go_back: bool):
        self._root = root
        self._current = current
        can_go_up = current is not None and root is not None and current != root
        # Hidden (not just greyed) when unusable: disabled 24px squares read as
        # inert decoration next to the folder title.
        self._back_btn.setEnabled(can_go_back)
        self._back_btn.setVisible(can_go_back)
        self._up_btn.setEnabled(can_go_up)
        self._up_btn.setVisible(can_go_up)
        self._rebuild_segments()

    def _clear_segments(self):
        while self._segments_lay.count():
            item = self._segments_lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _rebuild_segments(self):
        self._clear_segments()
        if self._root is None or self._current is None:
            placeholder = QLabel("No folder open")
            placeholder.setFont(ui_font(10))
            self._segments_lay.addWidget(placeholder)
            self._restyle_segments()
            return

        # Build relative segment chain: root, then each subdir down to current.
        try:
            rel = self._current.relative_to(self._root)
            tail = [] if str(rel) == "." else list(rel.parts)
        except ValueError:
            tail = []

        # Root segment uses the folder name, full chain uses parts.
        segments: list[tuple[str, Path]] = [(self._root.name or str(self._root), self._root)]
        cum = self._root
        for part in tail:
            cum = cum / part
            segments.append((part, cum))

        for i, (name, path) in enumerate(segments):
            if i:
                sep = QLabel("›")
                sep.setFont(ui_font(11))
                self._segments_lay.addWidget(sep)
            btn = QPushButton(name)
            btn.setFont(ui_font(10, weight=QFont.Bold if i == len(segments) - 1 else QFont.Normal))
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.setFlat(True)
            btn.clicked.connect(lambda _=False, p=path: self.segment_clicked.emit(p))
            self._segments_lay.addWidget(btn)
        self._segments_lay.addStretch(1)
        self._restyle_segments()
