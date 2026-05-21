"""Quick-access toolbar for the image viewer.

Three rows of buttons:
  Row 1 — interaction/drawing-tool modes (mutually exclusive toggles + Clear)
  Row 2 — common display, processing, and measurement actions
  Row 3 — ROI / selection utilities (Mask, Invert)

The toolbar emits ``mode_requested`` or ``action_requested`` signals and
contains no processing or analysis logic itself.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from probeflow.core.resources import asset_path


_MODE_BUTTONS: list[tuple[str, str, str, str]] = [
    ("pan",       "Pan",     "Navigate the image by dragging. Does not create ROIs.", "pan"),
    (
        "rectangle",
        "Rect",
        "Draw a rectangular ROI for processing or measurement.  [R]",
        "rectangle",
    ),
    (
        "ellipse",
        "Ellipse",
        "Draw an elliptical ROI for processing or measurement.  [E]",
        "ellipse",
    ),
    ("polygon",   "Poly",    "Draw a polygon ROI by clicking vertices.  [P]", "polygon"),
    ("freehand",  "Free",    "Draw a freehand ROI.  [F]", "freehand"),
    (
        "line",
        "Line",
        "Draw a line ROI for profiles, step heights, and periodicity.  [L]",
        "line",
    ),
    ("point",     "Point",   "Place a point ROI.  [T]", "point"),
]

_ACTION_BUTTONS: list[tuple[str, str, str, str]] = [
    (
        "auto_contrast",
        "Auto",
        "Autoscale image contrast using the current display settings.",
        "auto_contrast",
    ),
    ("plane_background", "Plane",     "Open simple plane/background subtraction.", "plane_background"),
    ("stm_background",   "STM BG",   "Open STM scan-line background subtraction.", "stm_background"),
    ("bad_lines",        "Bad lines", "Open bad scan-line correction.", "bad_lines"),
    ("open_fft",         "FFT",       "Open the FFT viewer for the current image.", "open_fft"),
    ("open_lattice_grid", "Grid",     "Open the lattice/grid measurement tool.", "open_lattice_grid"),
    (
        "line_periodicity",
        "Period",
        "Estimate periodicity from the selected line ROI.",
        "line_periodicity",
    ),
    ("line_profile",     "Profile",   "Show a line profile for the selected line ROI.", "line_profile"),
]

_ROI_ACTION_BUTTONS: list[tuple[str, str, str, str]] = [
    (
        "mask_selection",
        "Mask",
        "Create or apply a mask from the current ROI/selection.",
        "mask_selection",
    ),
    ("invert_selection", "Invert", "Invert the current area ROI or mask.", "invert_selection"),
]

# Future ROI utility actions may include Grow, Shrink, and Specify.
# These should be implemented in the ROI/selection backend first,
# then exposed here as toolbar actions. Do not implement ROI geometry
# modification logic directly in the toolbar.

_LINE_ACTIONS = {"line_periodicity", "line_profile"}
_AREA_ACTIONS = {"mask_selection", "invert_selection"}
_TOOLBAR_ICON_SIZE = QSize(18, 18)


class ImageQuickToolbar(QWidget):
    """Three-row quick-access toolbar for the image viewer."""

    mode_requested = Signal(str)
    action_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode_btns: dict[str, QPushButton] = {}
        self._action_btns: dict[str, QPushButton] = {}
        self._build()

    # ── public API ────────────────────────────────────────────────────────────

    def set_active_mode(self, key: str) -> None:
        """Check the toggle for *key* and uncheck all others."""
        for k, btn in self._mode_btns.items():
            btn.blockSignals(True)
            btn.setChecked(k == key)
            btn.blockSignals(False)

    def set_action_enabled(
        self,
        key: str,
        enabled: bool,
        *,
        enabled_tip: str | None = None,
        disabled_tip: str | None = None,
    ) -> None:
        """Enable or disable an action button by its key."""
        btn = self._action_btns.get(key)
        if btn is None:
            return

        btn.setEnabled(enabled)

        if enabled and enabled_tip:
            btn.setToolTip(enabled_tip)
        elif not enabled and disabled_tip:
            btn.setToolTip(disabled_tip)

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(3)

        root.addLayout(self._build_mode_row())

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep1)

        root.addLayout(self._build_action_row())

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep2)

        root.addLayout(self._build_roi_row())

    def _build_mode_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(3)
        row.setContentsMargins(0, 0, 0, 0)

        group = QButtonGroup(self)
        group.setExclusive(True)

        for key, label, tip, icon_name in _MODE_BUTTONS:
            btn = self._make_btn(label, tip, icon_name=icon_name, checkable=True)
            btn.setChecked(key == "pan")
            btn.clicked.connect(lambda _checked=False, k=key: self.mode_requested.emit(k))
            group.addButton(btn)
            self._mode_btns[key] = btn
            row.addWidget(btn)

        row.addSpacing(6)

        clear_btn = self._make_btn(
            "Clear",
            "Clear all ROIs and the angle overlay from the image.",
            icon_name="clear_selection",
        )
        clear_btn.clicked.connect(lambda: self.action_requested.emit("clear_selection"))
        self._action_btns["clear_selection"] = clear_btn
        row.addWidget(clear_btn)

        row.addStretch(1)
        return row

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(3)
        row.setContentsMargins(0, 0, 0, 0)

        for key, label, tip, icon_name in _ACTION_BUTTONS:
            btn = self._make_btn(label, tip, icon_name=icon_name)
            btn.clicked.connect(lambda _checked=False, k=key: self.action_requested.emit(k))
            if key in _LINE_ACTIONS:
                btn.setEnabled(False)
            self._action_btns[key] = btn
            row.addWidget(btn)

        row.addStretch(1)
        return row

    def _build_roi_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(3)
        row.setContentsMargins(0, 0, 0, 0)

        for key, label, tip, icon_name in _ROI_ACTION_BUTTONS:
            btn = self._make_btn(label, tip, icon_name=icon_name)
            btn.clicked.connect(lambda _checked=False, k=key: self.action_requested.emit(k))
            btn.setEnabled(key not in _AREA_ACTIONS)
            self._action_btns[key] = btn
            row.addWidget(btn)

        row.addStretch(1)
        return row

    @staticmethod
    def _make_btn(
        label: str,
        tip: str,
        *,
        icon_name: str | None = None,
        checkable: bool = False,
    ) -> QPushButton:
        btn = QPushButton(label)
        if icon_name:
            icon_path = asset_path(f"toolbar/{icon_name}.png")
            if icon_path.exists():
                btn.setIcon(QIcon(str(icon_path)))
                btn.setIconSize(_TOOLBAR_ICON_SIZE)
        btn.setCheckable(checkable)
        btn.setFixedHeight(28)
        btn.setMinimumWidth(46)
        btn.setMaximumWidth(96)
        btn.setToolTip(tip)
        btn.setDefault(False)
        btn.setAutoDefault(False)
        return btn
