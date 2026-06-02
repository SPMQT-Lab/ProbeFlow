"""GUI theme constants, QSS stylesheet builder, and separator helper."""

from __future__ import annotations

from probeflow.gui.config import GUI_FONT_SIZES, GUI_FONT_DEFAULT

NAVBAR_DARK_BG  = "#3273dc"
NAVBAR_LIGHT_BG = "#ffffff"
NAVBAR_H        = 58

# Modern-neutral palette: restrained neutral surfaces, one confident accent, soft
# selection tints and clear borders.  All keys consumed elsewhere (fg, sep, accent_bg,
# main_bg, entry_bg, card_bg, …) are preserved; ``surface``/``raised``/``sel_tint``/
# ``border``/``hover`` were added for the design-system QSS.
THEMES = {
    "dark": {
        "bg":         "#1e2128",
        "fg":         "#e6e8eb",
        "entry_bg":   "#1b1e24",
        "btn_bg":     "#2a2f38",
        "btn_fg":     "#e6e8eb",
        "log_bg":     "#14161a",
        "log_fg":     "#e6e8eb",
        "ok_fg":      "#5fd07f",
        "err_fg":     "#ff6b81",
        "warn_fg":    "#f0b072",
        "info_fg":    "#e6e8eb",
        "accent_bg":  "#4d8dff",
        "accent_fg":  "#0c0e12",
        "sep":        "#2c313a",
        "sub_fg":     "#9aa1ab",
        "sidebar_bg": "#14161a",
        "main_bg":    "#16181d",
        "status_bg":  "#1b1e24",
        "status_fg":  "#9aa1ab",
        "card_bg":    "#1e2128",
        "card_sel":   "#2b3a55",
        "card_fg":    "#e6e8eb",
        "tab_act":    "#242831",
        "tab_inact":  "#16181d",
        "tree_bg":    "#16181d",
        "tree_fg":    "#e6e8eb",
        "tree_sel":   "#2b3a55",
        "tree_head":  "#1e2128",
        "splitter":   "#2c313a",
        "surface":    "#1e2128",
        "raised":     "#242831",
        "sel_tint":   "#2b3a55",
        "border":     "#3a414c",
        "hover":      "#333a44",
    },
    "light": {
        "bg":         "#ffffff",
        "fg":         "#1c1e21",
        "entry_bg":   "#ffffff",
        "btn_bg":     "#eceef2",
        "btn_fg":     "#1c1e21",
        "log_bg":     "#f7f8fa",
        "log_fg":     "#1c1e21",
        "ok_fg":      "#1a7f37",
        "err_fg":     "#d11a2a",
        "warn_fg":    "#b9770b",
        "info_fg":    "#1c1e21",
        "accent_bg":  "#2f6feb",
        "accent_fg":  "#ffffff",
        "sep":        "#e4e7ec",
        "sub_fg":     "#6b7280",
        "sidebar_bg": "#f3f4f7",
        "main_bg":    "#f6f7f9",
        "status_bg":  "#f0f1f4",
        "status_fg":  "#6b7280",
        "card_bg":    "#ffffff",
        "card_sel":   "#e6efff",
        "card_fg":    "#1c1e21",
        "tab_act":    "#ffffff",
        "tab_inact":  "#eef0f3",
        "tree_bg":    "#ffffff",
        "tree_fg":    "#1c1e21",
        "tree_sel":   "#e6efff",
        "tree_head":  "#f3f4f7",
        "splitter":   "#e4e7ec",
        "surface":    "#ffffff",
        "raised":     "#ffffff",
        "sel_tint":   "#e6efff",
        "border":     "#d8dbe1",
        "hover":      "#e4e7ec",
    },
}


def _sep():
    from PySide6.QtWidgets import QFrame
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedHeight(1)
    return line


def _build_qss(t: dict, font_pt: int = GUI_FONT_SIZES[GUI_FONT_DEFAULT]) -> str:
    return f"""
QMainWindow, QWidget {{
    background-color: {t['main_bg']};
    color: {t['fg']};
    font-family: Helvetica, Arial, sans-serif;
    font-size: {font_pt}pt;
}}
/* Labels are transparent so they sit cleanly on any surface (cards, sidebars). */
QLabel {{ background: transparent; }}
QToolTip {{
    background-color: {t['raised']};
    color: {t['fg']};
    border: 1px solid {t['border']};
    border-radius: 6px;
    padding: 4px 8px;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background-color: {t['main_bg']};
    border: none;
}}
BrowseToolPanel, BrowseToolPanel QWidget,
BrowseInfoPanel, BrowseInfoPanel QWidget,
ConvertSidebar, ConvertSidebar QWidget,
ConvertPanel, ConvertPanel QWidget {{
    background-color: {t['sidebar_bg']};
}}
BrowseToolPanel QLabel, BrowseInfoPanel QLabel,
ConvertSidebar QLabel, ConvertPanel QLabel {{
    color: {t['fg']};
    background: transparent;
}}

/* ── Buttons ──────────────────────────────────────────────────────────── */
/* Secondary (default): subtle filled, bordered, rounded. */
QPushButton {{
    background-color: {t['btn_bg']};
    color: {t['btn_fg']};
    border: 1px solid {t['border']};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background-color: {t['hover']}; }}
QPushButton:pressed {{ background-color: {t['sel_tint']}; }}
QPushButton:disabled {{
    background-color: {t['main_bg']};
    color: {t['sub_fg']};
    border-color: {t['sep']};
}}
/* Primary (accent). */
QPushButton#accentBtn {{
    background-color: {t['accent_bg']};
    color: {t['accent_fg']};
    border: 1px solid {t['accent_bg']};
    font-weight: 600;
}}
QPushButton#accentBtn:hover {{ background-color: {t['accent_bg']}; }}
QPushButton#accentBtn:disabled {{
    background-color: {t['main_bg']};
    color: {t['sub_fg']};
    border-color: {t['sep']};
}}
/* Ghost (navigation / low-emphasis): transparent, accent text. */
QPushButton#ghostBtn {{
    background: transparent;
    color: {t['accent_bg']};
    border: none;
    border-radius: 6px;
    padding: 4px 10px;
    font-weight: 600;
}}
QPushButton#ghostBtn:hover {{ background-color: {t['sel_tint']}; }}
/* Segmented buttons (browse filters). */
QPushButton#segBtnLeft, QPushButton#segBtnMid, QPushButton#segBtnRight {{
    background-color: {t['btn_bg']};
    color: {t['btn_fg']};
    border: 1px solid {t['border']};
    padding: 4px 10px;
    margin: 0px;
    border-radius: 0px;
}}
QPushButton#segBtnLeft {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; }}
QPushButton#segBtnRight {{ border-top-right-radius: 6px; border-bottom-right-radius: 6px; }}
QPushButton#segBtnMid {{ border-left: none; border-right: none; }}
QPushButton#segBtnLeft:hover, QPushButton#segBtnMid:hover,
QPushButton#segBtnRight:hover {{ background-color: {t['hover']}; }}
QPushButton#segBtnLeft:checked, QPushButton#segBtnMid:checked,
QPushButton#segBtnRight:checked {{
    background-color: {t['accent_bg']};
    color: {t['accent_fg']};
    border-color: {t['accent_bg']};
    font-weight: 600;
}}
/* Navbar buttons (on the coloured navbar). */
QPushButton#navBtn {{
    color: #ffffff;
    background-color: transparent;
    border: 1px solid rgba(255,255,255,0.40);
    border-radius: 6px;
    padding: 4px 12px;
}}
QPushButton#navBtn:hover {{ background-color: rgba(255,255,255,0.18); }}

/* ── Inputs ───────────────────────────────────────────────────────────── */
QComboBox {{
    background-color: {t['entry_bg']};
    color: {t['fg']};
    border: 1px solid {t['border']};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {t['accent_bg']};
}}
QComboBox:hover {{ border-color: {t['sub_fg']}; }}
QComboBox:focus {{ border-color: {t['accent_bg']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {t['raised']};
    color: {t['fg']};
    selection-background-color: {t['accent_bg']};
    selection-color: {t['accent_fg']};
    border: 1px solid {t['border']};
    border-radius: 6px;
    outline: none;
    padding: 4px;
}}
QComboBox QAbstractItemView::item {{
    min-height: 24px;
    padding: 3px 8px;
    border-radius: 4px;
}}
QLineEdit, QDoubleSpinBox, QSpinBox {{
    background-color: {t['entry_bg']};
    color: {t['fg']};
    border: 1px solid {t['border']};
    border-radius: 6px;
    padding: 5px 8px;
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {t['accent_bg']};
}}
QTextEdit {{
    background-color: {t['log_bg']};
    color: {t['log_fg']};
    border: 1px solid {t['border']};
    border-radius: 8px;
}}
QCheckBox {{ color: {t['fg']}; spacing: 8px; background: transparent; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {t['border']};
    border-radius: 4px;
    background-color: {t['entry_bg']};
}}
QCheckBox::indicator:checked {{
    background-color: {t['accent_bg']};
    border-color: {t['accent_bg']};
}}

/* ── Sliders ──────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px; border-radius: 2px; background: {t['sep']};
}}
QSlider::sub-page:horizontal {{ background: {t['accent_bg']}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -6px 0;
    border-radius: 7px; background: {t['accent_bg']};
    border: 2px solid {t['surface']};
}}

/* ── Tables & lists ───────────────────────────────────────────────────── */
QTableWidget {{
    background-color: {t['tree_bg']};
    color: {t['tree_fg']};
    border: 1px solid {t['sep']};
    border-radius: 8px;
    gridline-color: transparent;
    alternate-background-color: {t['main_bg']};
}}
QTableWidget::item {{ padding: 4px 6px; border: none; }}
QTableWidget::item:selected {{
    background-color: {t['sel_tint']};
    color: {t['fg']};
}}
QHeaderView::section {{
    background-color: {t['tree_head']};
    color: {t['sub_fg']};
    border: none;
    padding: 5px 6px;
    font-weight: 600;
}}
QListWidget {{
    background-color: {t['tree_bg']};
    color: {t['tree_fg']};
    border: 1px solid {t['sep']};
    border-radius: 8px;
    outline: none;
    padding: 2px;
}}
QListWidget::item {{ padding: 5px 6px; border-radius: 6px; }}
QListWidget::item:hover {{ background-color: {t['hover']}; }}
QListWidget::item:selected {{
    background-color: {t['sel_tint']};
    color: {t['fg']};
}}

/* ── Scrollbars ───────────────────────────────────────────────────────── */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{
    background-color: {t['border']}; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background-color: {t['sub_fg']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{
    background-color: {t['border']}; border-radius: 5px; min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{ background-color: {t['sub_fg']}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── Tabs ─────────────────────────────────────────────────────────────── */
QTabWidget::pane {{ border: none; border-top: 1px solid {t['sep']}; }}
QTabBar {{ border: none; qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent;
    color: {t['sub_fg']};
    padding: 6px 12px;
    border: none;
    border-bottom: 2px solid transparent;
    min-width: 36px;
}}
QTabBar::tab:selected {{
    color: {t['fg']};
    font-weight: 600;
    border-bottom: 2px solid {t['accent_bg']};
}}
QTabBar::tab:hover:!selected {{ color: {t['fg']}; }}

/* ── Status bar / dialogs / separators ────────────────────────────────── */
QStatusBar {{
    background-color: {t['status_bg']};
    color: {t['status_fg']};
    border-top: 1px solid {t['sep']};
    font-size: 10pt;
}}
QDialog {{ background-color: {t['bg']}; color: {t['fg']}; }}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {t['sep']};
    background-color: {t['sep']};
    max-height: 1px;
}}

/* ── Floating tool panels (hover over the image canvas) ───────────────── */
#floatingPanel {{
    background-color: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: 10px;
}}
#floatingPanelHeader {{
    background-color: {t['raised']};
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    border-bottom: 1px solid {t['sep']};
}}
#floatingPanelTitle {{ color: {t['fg']}; font-weight: 600; background: transparent; }}
#floatingPanelBody {{
    background-color: {t['surface']};
    border-bottom-left-radius: 10px;
    border-bottom-right-radius: 10px;
}}
QToolButton#floatingPanelClose {{
    background: transparent; color: {t['sub_fg']};
    border: none; border-radius: 4px; padding: 1px 6px;
}}
QToolButton#floatingPanelClose:hover {{ background-color: {t['err_fg']}; color: #ffffff; }}

/* ── Modal tool overlay (dimmed scrim + centred card) ─────────────────── */
#modalOverlay {{ background-color: rgba(0, 0, 0, 0.45); }}
#overlayCard {{
    background-color: {t['surface']};
    border: 1px solid {t['border']};
    border-radius: 12px;
}}
#overlayCardBody, #advancedToolsPanel {{ background-color: {t['surface']}; }}
QToolButton#overlayCardClose {{
    background: transparent; color: {t['sub_fg']};
    border: none; border-radius: 6px; padding: 2px 8px; font-size: 13pt;
}}
QToolButton#overlayCardClose:hover {{ background-color: {t['hover']}; color: {t['fg']}; }}

/* ── Collapsible sidebar rail + chevrons ──────────────────────────────── */
#sidebarRail {{
    background-color: {t['sidebar_bg']};
    border-left: 1px solid {t['sep']};
}}
QToolButton#sidebarRailBtn, QToolButton#sidebarExpandBtn,
QToolButton#sidebarCollapseBtn {{
    background: transparent;
    color: {t['sub_fg']};
    border: none;
    border-radius: 6px;
    padding: 4px 2px;
    font-weight: 600;
}}
QToolButton#sidebarRailBtn:hover, QToolButton#sidebarExpandBtn:hover,
QToolButton#sidebarCollapseBtn:hover {{
    background-color: {t['hover']};
    color: {t['fg']};
}}
QToolButton#sidebarCollapseBtn {{ margin: 0 4px 0 8px; font-size: 13pt; }}
"""


__all__ = [
    "NAVBAR_DARK_BG", "NAVBAR_LIGHT_BG", "NAVBAR_H",
    "THEMES",
    "_sep", "_build_qss",
]
