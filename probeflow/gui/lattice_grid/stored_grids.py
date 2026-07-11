"""Shared "stored grids" model + list widget for the lattice-grid tools.

Both the real-space grid panel and the FFT reciprocal-grid panel host the same
widget, so storing/recalling extra grid layers looks and works identically in
both spaces. A stored grid is a static, coloured snapshot: align the editable
grid to one lattice, store it, then fit the next lattice — Edit swaps a stored
layer back into the editor (parking the current grid in its slot) so two
coexisting lattices can be compared and refined in turns.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from probeflow.analysis.lattice_grid import LatticeGrid
from probeflow.gui.typography import ui_font

from .constants import STORED_GRID_PALETTE


def next_stored_color(n_existing: int) -> str:
    """Palette colour for the next stored layer (cycles when exhausted)."""
    return STORED_GRID_PALETTE[n_existing % len(STORED_GRID_PALETTE)]


@dataclass
class StoredGrid:
    """A static snapshot of one grid layer."""

    grid: LatticeGrid
    cells: int
    line_width_px: float
    color: str      # hex colour of this layer's lines
    summary: str    # short human-readable lattice description
    detail: str = ""  # optional longer description for the row tooltip


class StoredGridList(QWidget):
    """"Store grid" button plus one row per stored layer (swatch · summary · Edit · ✕)."""

    store_requested = Signal()
    edit_requested = Signal(int)
    remove_requested = Signal(int)

    def __init__(self, parent=None, *, store_button_row: QHBoxLayout | None = None):
        """``store_button_row`` — optional existing row to place the Store
        button into (so hosts can pair it with a neighbour button); when None
        the button sits on its own row inside this widget."""
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)

        self.store_btn = QPushButton("Store grid")
        self.store_btn.setFont(ui_font(9))
        self.store_btn.setFixedHeight(24)
        self.store_btn.setToolTip(
            "Keep the current grid as a coloured static layer and continue "
            "editing — e.g. fit a second lattice to find a coincidence pattern."
        )
        self.store_btn.setDefault(False)
        self.store_btn.setAutoDefault(False)
        self.store_btn.clicked.connect(self.store_requested.emit)
        if store_button_row is not None:
            store_button_row.addWidget(self.store_btn)
        else:
            lay.addWidget(self.store_btn)

        self._rows_lay = QVBoxLayout()
        self._rows_lay.setContentsMargins(0, 0, 0, 0)
        self._rows_lay.setSpacing(2)
        lay.addLayout(self._rows_lay)
        self._row_widgets: list[QWidget] = []

    def set_entries(self, entries: list[StoredGrid]) -> None:
        """Rebuild the rows from the host's stored-grid list."""
        for row in self._row_widgets:
            self._rows_lay.removeWidget(row)
            row.deleteLater()
        self._row_widgets = []
        for i, entry in enumerate(entries):
            row = QWidget()
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(4)

            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background-color: {entry.color}; border-radius: 2px;"
            )
            swatch.setToolTip("Colour of this stored grid layer on the image.")
            row_lay.addWidget(swatch)

            # Elide long summaries so the Edit/remove buttons are never pushed
            # out of the row; the full text lives in the tooltip.
            lbl = QLabel()
            lbl.setFont(ui_font(8))
            metrics = lbl.fontMetrics()
            lbl.setText(metrics.elidedText(entry.summary, Qt.ElideRight, 190))
            lbl.setToolTip(entry.detail or entry.summary)
            lbl.setMinimumWidth(40)
            row_lay.addWidget(lbl, 1)

            edit_btn = QPushButton("Edit")
            edit_btn.setFixedSize(48, 22)
            edit_btn.setToolTip(
                "Swap this layer into the editor; the grid you are editing "
                "takes its place here."
            )
            edit_btn.setDefault(False)
            edit_btn.setAutoDefault(False)
            edit_btn.clicked.connect(lambda _c=False, idx=i: self.edit_requested.emit(idx))
            row_lay.addWidget(edit_btn)

            del_btn = QPushButton("✕")
            del_btn.setFixedSize(24, 22)
            del_btn.setToolTip("Remove this stored grid layer from the image.")
            del_btn.setDefault(False)
            del_btn.setAutoDefault(False)
            del_btn.clicked.connect(lambda _c=False, idx=i: self.remove_requested.emit(idx))
            row_lay.addWidget(del_btn)

            self._rows_lay.addWidget(row)
            self._row_widgets.append(row)
