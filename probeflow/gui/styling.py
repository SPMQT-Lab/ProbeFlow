"""GUI theme constants, QSS stylesheet builder, and separator helper."""

from __future__ import annotations

from probeflow.gui.config import GUI_FONT_SIZES, GUI_FONT_DEFAULT
from probeflow.gui.typography import ui_family

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
    # ── Midnight: near-pure-black dark (OLED / dark-room imaging) ──────────────
    "midnight": {
        "bg":         "#050608",
        "fg":         "#eef0f3",
        "entry_bg":   "#0f1115",
        "btn_bg":     "#1a1d22",
        "btn_fg":     "#eef0f3",
        "log_bg":     "#000000",
        "log_fg":     "#eef0f3",
        "ok_fg":      "#57d977",
        "err_fg":     "#ff6b81",
        "warn_fg":    "#f0b072",
        "info_fg":    "#eef0f3",
        "accent_bg":  "#5b9cff",
        "accent_fg":  "#04060a",
        "sep":        "#1c2026",
        "sub_fg":     "#8b929c",
        "sidebar_bg": "#000000",
        "main_bg":    "#050608",
        "status_bg":  "#0f1115",
        "status_fg":  "#8b929c",
        "card_bg":    "#0f1115",
        "card_sel":   "#1c2740",
        "card_fg":    "#eef0f3",
        "tab_act":    "#14171c",
        "tab_inact":  "#050608",
        "tree_bg":    "#050608",
        "tree_fg":    "#eef0f3",
        "tree_sel":   "#1c2740",
        "tree_head":  "#0f1115",
        "splitter":   "#1c2026",
        "surface":    "#0f1115",
        "raised":     "#161a20",
        "sel_tint":   "#1c2740",
        "border":     "#2a2f38",
        "hover":      "#1e222a",
    },
    # ── Slate: cool blue-grey dark ────────────────────────────────────────────
    "slate": {
        "bg":         "#1b212b",
        "fg":         "#e3e8ef",
        "entry_bg":   "#181e27",
        "btn_bg":     "#2a3440",
        "btn_fg":     "#e3e8ef",
        "log_bg":     "#141922",
        "log_fg":     "#e3e8ef",
        "ok_fg":      "#6bd58e",
        "err_fg":     "#ff7a90",
        "warn_fg":    "#f0b878",
        "info_fg":    "#e3e8ef",
        "accent_bg":  "#6aa3ff",
        "accent_fg":  "#0b1018",
        "sep":        "#2d3947",
        "sub_fg":     "#93a0b3",
        "sidebar_bg": "#161b23",
        "main_bg":    "#1b212b",
        "status_bg":  "#181e27",
        "status_fg":  "#93a0b3",
        "card_bg":    "#222b38",
        "card_sel":   "#2b3a55",
        "card_fg":    "#e3e8ef",
        "tab_act":    "#222b38",
        "tab_inact":  "#161b23",
        "tree_bg":    "#161b23",
        "tree_fg":    "#e3e8ef",
        "tree_sel":   "#2b3a55",
        "tree_head":  "#222b38",
        "splitter":   "#2d3947",
        "surface":    "#222b38",
        "raised":     "#283340",
        "sel_tint":   "#2b3a55",
        "border":     "#3a4658",
        "hover":      "#2f3b4a",
    },
    # ── Paper: warm, low-glare light ──────────────────────────────────────────
    "paper": {
        "bg":         "#fbf9f4",
        "fg":         "#2b2620",
        "entry_bg":   "#fffdf8",
        "btn_bg":     "#ece6da",
        "btn_fg":     "#2b2620",
        "log_bg":     "#fbf9f4",
        "log_fg":     "#2b2620",
        "ok_fg":      "#2f7d32",
        "err_fg":     "#b3261e",
        "warn_fg":    "#9a6a00",
        "info_fg":    "#2b2620",
        "accent_bg":  "#2f66d6",
        "accent_fg":  "#ffffff",
        "sep":        "#e6dfd2",
        "sub_fg":     "#6f665a",
        "sidebar_bg": "#f1ece1",
        "main_bg":    "#f5f1e8",
        "status_bg":  "#ece6da",
        "status_fg":  "#6f665a",
        "card_bg":    "#fffdf8",
        "card_sel":   "#e7edfb",
        "card_fg":    "#2b2620",
        "tab_act":    "#fffdf8",
        "tab_inact":  "#ece6da",
        "tree_bg":    "#fffdf8",
        "tree_fg":    "#2b2620",
        "tree_sel":   "#e7edfb",
        "tree_head":  "#f1ece1",
        "splitter":   "#e6dfd2",
        "surface":    "#fffdf8",
        "raised":     "#fffdf8",
        "sel_tint":   "#e7edfb",
        "border":     "#d9d2c4",
        "hover":      "#ece6da",
    },
    # ── High contrast: maximum legibility (black/white + vivid accent) ────────
    "high_contrast": {
        "bg":         "#000000",
        "fg":         "#ffffff",
        "entry_bg":   "#000000",
        "btn_bg":     "#1a1a1a",
        "btn_fg":     "#ffffff",
        "log_bg":     "#000000",
        "log_fg":     "#ffffff",
        "ok_fg":      "#00e676",
        "err_fg":     "#ff5252",
        "warn_fg":    "#ffd740",
        "info_fg":    "#ffffff",
        "accent_bg":  "#4da3ff",
        "accent_fg":  "#000000",
        "sep":        "#ffffff",
        "sub_fg":     "#d4d4d4",
        "sidebar_bg": "#000000",
        "main_bg":    "#000000",
        "status_bg":  "#000000",
        "status_fg":  "#d4d4d4",
        "card_bg":    "#0a0a0a",
        "card_sel":   "#00305f",
        "card_fg":    "#ffffff",
        "tab_act":    "#0a0a0a",
        "tab_inact":  "#000000",
        "tree_bg":    "#000000",
        "tree_fg":    "#ffffff",
        "tree_sel":   "#00305f",
        "tree_head":  "#0a0a0a",
        "splitter":   "#ffffff",
        "surface":    "#0a0a0a",
        "raised":     "#141414",
        "sel_tint":   "#00305f",
        "border":     "#ffffff",
        "hover":      "#1f1f1f",
    },
}

# Ordered theme presets shown in the picker: (key, label, is_dark).
THEME_PRESETS: tuple[tuple[str, str, bool], ...] = (
    ("dark",          "Dark",          True),
    ("light",         "Light",         False),
    ("midnight",      "Midnight",      True),
    ("slate",         "Slate",         True),
    ("paper",         "Paper",         False),
    ("high_contrast", "High contrast", True),
)

_THEME_IS_DARK = {key: is_dark for key, _label, is_dark in THEME_PRESETS}


def theme_is_dark(name: str) -> bool:
    """Return whether the named theme is a dark theme (defaults to dark)."""
    return _THEME_IS_DARK.get(name, True)


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
    font-family: "{ui_family()}", Helvetica, Arial;
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
/* Secondary (default): borderless filled, rounded.  Height is governed by
   min-height (not vertical padding) so widgets that set a small fixed height
   (setFixedHeight(22), common across the app) keep their text un-clipped. */
QPushButton {{
    background-color: {t['btn_bg']};
    color: {t['btn_fg']};
    border: none;
    border-radius: 6px;
    padding: 0 10px;
    min-height: 26px;
    /* One step under the body size: overrides the scattered tiny per-widget
       fonts so button text is never dwarfed by the button box. */
    font-size: {max(font_pt - 1, 9)}pt;
}}
QPushButton:hover {{ background-color: {t['hover']}; }}
QPushButton:pressed {{ background-color: {t['sel_tint']}; }}
QPushButton:disabled {{
    background-color: {t['main_bg']};
    color: {t['sub_fg']};
}}
/* Primary (accent). */
QPushButton#accentBtn {{
    background-color: {t['accent_bg']};
    color: {t['accent_fg']};
    font-weight: 600;
}}
QPushButton#accentBtn:hover {{ background-color: {t['accent_bg']}; }}
QPushButton#accentBtn:disabled {{
    background-color: {t['main_bg']};
    color: {t['sub_fg']};
}}
/* Back / up navigation chip: accent-tinted so it clearly reads as "go back". */
QPushButton#ghostBtn {{
    background-color: {t['sel_tint']};
    color: {t['accent_bg']};
    border: none;
    border-radius: 8px;
    padding: 6px 18px;
    font-weight: 700;
    font-size: 13pt;
}}
QPushButton#ghostBtn:hover {{
    background-color: {t['accent_bg']};
    color: {t['accent_fg']};
}}
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
/* Active drawing-tool highlight (toolbar mode toggles + the "More" popup). */
QPushButton#modeToolBtn:checked {{
    background-color: {t['accent_bg']};
    color: {t['accent_fg']};
    font-weight: 600;
}}
QToolButton#imageToolMore:checked {{
    background-color: {t['accent_bg']};
    color: {t['accent_fg']};
}}
/* ── Inputs ───────────────────────────────────────────────────────────── */
QComboBox {{
    background-color: {t['entry_bg']};
    color: {t['fg']};
    border: 1px solid {t['border']};
    border-radius: 6px;
    padding: 2px 8px;
    min-height: 24px;
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
    padding: 2px 8px;
    min-height: 24px;
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border: 1px solid {t['accent_bg']};
}}
/* Restyling the spin-box body drops Qt's native step buttons, so give the
   sub-controls an explicit, visible size and clear triangle arrows. */
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid {t['border']};
    border-top-right-radius: 6px;
    background: {t['btn_bg']};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid {t['border']};
    border-bottom-right-radius: 6px;
    background: {t['btn_bg']};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {t['hover']};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {t['fg']};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {t['fg']};
}}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{
    border-bottom-color: {t['border']};
}}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
    border-top-color: {t['border']};
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
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* ── Menus ────────────────────────────────────────────────────────────── */
QMenuBar {{
    background-color: {t['bg']};
    color: {t['fg']};
    border: none;
}}
QMenuBar::item {{
    background: transparent;
    color: {t['fg']};
    padding: 4px 8px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{ background-color: {t['hover']}; }}
QMenuBar::item:pressed {{ background-color: {t['sel_tint']}; }}
QMenu {{
    background-color: {t['card_bg']};
    color: {t['fg']};
    border: 1px solid {t['sep']};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{
    background: transparent;
    padding: 4px 24px 4px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{ background-color: {t['hover']}; }}
QMenu::item:disabled {{ color: {t['sub_fg']}; }}
QMenu::separator {{
    height: 1px;
    background-color: {t['sep']};
    margin: 4px 8px;
}}

/* ── Tabs ─────────────────────────────────────────────────────────────── */
QTabWidget::pane {{ border: none; border-top: 1px solid {t['sep']}; }}
QTabBar {{ border: none; qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent;
    color: {t['sub_fg']};
    padding: 6px 8px;
    border: none;
    border-bottom: 2px solid transparent;
    min-width: 30px;
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

/* Latest-result headline in the Measure detail view. */
QLabel#resultSummary {{
    background-color: {t['sel_tint']};
    color: {t['fg']};
    border-radius: 6px;
    padding: 6px 10px;
    font-weight: 600;
}}

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
/* Bigger, clearer collapse/expand chevrons. */
QToolButton#sidebarCollapseBtn {{ margin: 0 2px 0 4px; font-size: 19pt; }}
QToolButton#sidebarExpandBtn {{ font-size: 19pt; color: {t['accent_bg']}; }}
"""


def _build_palette(t: dict):
    """Build a QPalette from the theme so ``palette(...)`` roles and native widget
    bits (combo popups, disabled text, tooltips) resolve correctly in both modes.

    The app styles most widgets via QSS; without a matching palette the unstyled
    bits and any ``color: palette(mid)`` text fall back to Qt's default *light*
    palette, which is unreadable in dark mode.
    """
    from PySide6.QtGui import QColor, QPalette

    def c(key: str) -> "QColor":
        return QColor(t[key])

    p = QPalette()
    p.setColor(QPalette.Window, c("main_bg"))
    p.setColor(QPalette.WindowText, c("fg"))
    p.setColor(QPalette.Base, c("entry_bg"))
    p.setColor(QPalette.AlternateBase, c("surface"))
    p.setColor(QPalette.Text, c("fg"))
    p.setColor(QPalette.Button, c("btn_bg"))
    p.setColor(QPalette.ButtonText, c("btn_fg"))
    p.setColor(QPalette.ToolTipBase, c("raised"))
    p.setColor(QPalette.ToolTipText, c("fg"))
    p.setColor(QPalette.PlaceholderText, c("sub_fg"))
    p.setColor(QPalette.Mid, c("sub_fg"))
    p.setColor(QPalette.Midlight, c("sep"))
    p.setColor(QPalette.Dark, c("border"))
    p.setColor(QPalette.Highlight, c("accent_bg"))
    p.setColor(QPalette.HighlightedText, c("accent_fg"))
    p.setColor(QPalette.Link, c("accent_bg"))
    p.setColor(QPalette.BrightText, c("err_fg"))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, c("sub_fg"))
    return p


__all__ = [
    "THEMES", "THEME_PRESETS", "theme_is_dark",
    "_sep", "_build_qss", "_build_palette",
]
