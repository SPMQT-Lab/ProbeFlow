"""Shared constants for the lattice-grid GUI tools."""

from __future__ import annotations

from PySide6.QtGui import QColor

# -- colours -----------------------------------------------------------------

_COL_GRID = QColor("#89b4fa")  # blue  - lattice lines
_COL_A = QColor("#a6e3a1")  # green - a vector / handle
_COL_B = QColor("#fab387")  # peach - b vector / handle
_COL_ORIGIN = QColor("#f38ba8")  # red   - origin handle
_COL_ROT = QColor("#cba6f7")  # purple - rotation handle
_COL_SCALE = QColor("#f9e2af")  # yellow - scale handle
_COL_LABEL = QColor("#cdd6f4")  # text labels

# -- handle IDs --------------------------------------------------------------

_HANDLE_NONE = 0
_HANDLE_ORIGIN = 1
_HANDLE_A = 2
_HANDLE_B = 3
_HANDLE_ROT = 4
_HANDLE_SCALE = 5

# Screen-space hit radius used by both controller and FFT overlay.
HIT_RADIUS_PX = 12

# Visual handle size in screen pixels (drawn via cosmetic pen / fixed size).
_HANDLE_SCREEN_R = 6.0
