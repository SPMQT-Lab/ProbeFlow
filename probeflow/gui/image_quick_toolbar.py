"""Quick-access toolbar for the image viewer.

Three rows of buttons:
  Row 1 — interaction/drawing-tool modes (mutually exclusive toggles + Clear)
  Row 2 — common display, processing, and measurement actions
  Row 3 — ROI / selection utilities (Mask, Invert)

The toolbar emits ``mode_requested`` or ``action_requested`` signals and
contains no processing or analysis logic itself.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QAction, QActionGroup, QIcon
from probeflow.gui.typography import ui_font
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from probeflow.core.resources import asset_path


# Common drawing tools shown directly on the mode row.  The default "pan" tool is
# shown as a plain cursor symbol (it selects ROIs and drags to pan — no drawing).
_MODE_BUTTONS: list[tuple[str, str, str, str]] = [
    ("pan",       "↖",       "Cursor — click an ROI to select it, or drag to pan the image.", ""),
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
    (
        "line",
        "Line",
        "Draw a line ROI for profiles, step heights, and periodicity.  [L]",
        "line",
    ),
    ("point",     "Point",   "Place a point ROI.  [P]", "point"),
]

# Less-common drawing tools tucked behind the "More" popup.
_MORE_MODE_BUTTONS: list[tuple[str, str, str, str]] = [
    ("polygon",   "Polygon",  "Draw a polygon ROI by clicking vertices.  [G]", "polygon"),
    ("freehand",  "Freehand", "Draw a freehand ROI.  [F]", "freehand"),
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

_AREA_ACTIONS = {"mask_selection", "invert_selection"}
_TOOLBAR_ICON_SIZE = QSize(18, 18)


class ImageQuickToolbar(QWidget):
    """Two-row quick-access toolbar for the image viewer.

    Row 1 — drawing-tool modes (common ones inline, rarer ones under "More") + Clear.
    Row 2 — ROI/selection utilities (Mask, Invert).

    Processing/measurement actions live in the sidebar tabs, not here.
    """

    mode_requested = Signal(str)
    action_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode_btns: dict[str, QPushButton] = {}
        self._action_btns: dict[str, QPushButton] = {}
        self._more_actions: dict[str, QAction] = {}
        self._more_btn: QToolButton | None = None
        self._build()

    # ── public API ────────────────────────────────────────────────────────────

    def set_active_mode(self, key: str) -> None:
        """Check the toggle for *key* and uncheck all others (incl. the More popup)."""
        for k, btn in self._mode_btns.items():
            btn.blockSignals(True)
            btn.setChecked(k == key)
            btn.blockSignals(False)
        in_more = key in self._more_actions
        if self._more_btn is not None:
            self._more_btn.setChecked(in_more)
        for k, act in self._more_actions.items():
            act.setChecked(k == key)

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

        root.addLayout(self._build_roi_row())

    def _build_mode_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(3)
        row.setContentsMargins(0, 0, 0, 0)

        group = QButtonGroup(self)
        group.setExclusive(True)

        for key, label, tip, icon_name in _MODE_BUTTONS:
            btn = self._make_btn(label, tip, icon_name=icon_name, checkable=True)
            btn.setObjectName("modeToolBtn")  # enables the :checked highlight
            btn.setChecked(key == "pan")
            if key == "pan":
                btn.setFont(ui_font(13))  # render the cursor glyph a touch larger
            btn.clicked.connect(lambda _checked=False, k=key: self.mode_requested.emit(k))
            group.addButton(btn)
            self._mode_btns[key] = btn
            row.addWidget(btn)

        # "More" popup for the less-common drawing tools (Polygon, Freehand).
        self._more_btn = QToolButton()
        self._more_btn.setObjectName("imageToolMore")
        self._more_btn.setText("More")
        self._more_btn.setCheckable(True)
        self._more_btn.setToolTip("More drawing tools")
        self._more_btn.setFixedHeight(28)
        self._more_btn.setMinimumWidth(56)
        self._more_btn.setPopupMode(QToolButton.InstantPopup)
        self._more_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        more_menu = QMenu(self._more_btn)
        more_group = QActionGroup(self)
        more_group.setExclusive(True)
        for key, label, tip, icon_name in _MORE_MODE_BUTTONS:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setToolTip(tip)
            icon_path = asset_path(f"toolbar/{icon_name}.png")
            if icon_path.exists():
                act.setIcon(QIcon(str(icon_path)))
            act.triggered.connect(lambda _checked=False, k=key: self.mode_requested.emit(k))
            more_group.addAction(act)
            more_menu.addAction(act)
            self._more_actions[key] = act
        self._more_btn.setMenu(more_menu)
        row.addWidget(self._more_btn)

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
