"""Smoke tests for widgets extracted from the legacy GUI module."""

from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")

    app = QApplication.instance()
    if app is not None:
        return app
    try:
        return QApplication([])
    except Exception as exc:
        pytest.skip(f"QApplication unavailable: {exc}")


def _theme() -> dict[str, str]:
    return {
        "bg": "#1e1e2e",
        "fg": "#cdd6f4",
        "sidebar_bg": "#181825",
    }


def test_processing_panel_imports_from_new_module_and_gui_package(qapp):
    from probeflow.gui import ProcessingControlPanel as PublicPanel
    from probeflow.gui.processing import ProcessingControlPanel

    assert PublicPanel is ProcessingControlPanel

    quick = ProcessingControlPanel("browse_quick")
    full = ProcessingControlPanel("viewer_full")

    assert quick.state() == {"align_rows": None, "remove_bad_lines": None}
    assert full.state()["align_rows"] is None

    quick.close()
    full.close()


def test_terminal_widgets_import_from_new_module_and_gui_package(qapp):
    from probeflow.gui import DeveloperTerminalWidget as PublicTerminal
    from probeflow.gui import _DevSidebar as PublicDevSidebar
    from probeflow.gui import _TerminalPane as PublicPane
    from probeflow.gui.terminal import DeveloperTerminalWidget, _DevSidebar, _TerminalPane

    assert PublicTerminal is DeveloperTerminalWidget
    assert PublicDevSidebar is _DevSidebar
    assert PublicPane is _TerminalPane

    pane = _TerminalPane()
    pane.set_prompt("test $ ")
    pane.show_prompt()
    assert pane.toPlainText().startswith("test $ ")

    widget = DeveloperTerminalWidget(_theme())
    assert widget.findChild(_TerminalPane) is not None
    sidebar = _DevSidebar(_theme())
    assert sidebar.layout() is not None

    pane.close()
    widget.close()
    sidebar.close()


def test_main_window_browse_layout_uses_resizable_splitters(qapp):
    from probeflow.gui import ProbeFlowWindow

    window = ProbeFlowWindow()
    window.show()

    assert window._browse_tools.minimumWidth() == 240
    assert window._grid.minimumWidth() == 500
    assert window._sidebar_stack.minimumWidth() == 300
    assert window._browse_tools.maximumWidth() > 10000
    assert window._sidebar_stack.maximumWidth() > 10000
    assert not window._splitter.childrenCollapsible()
    assert not window._browse_splitter.childrenCollapsible()

    main_sizes = window._splitter.sizes()
    browse_sizes = window._browse_splitter.sizes()
    assert main_sizes[1] >= 300
    assert browse_sizes[0] >= 240
    assert browse_sizes[1] >= 500

    window.close()


def test_main_window_uses_standard_menus_for_secondary_views(qapp):
    from probeflow.gui import ProbeFlowWindow

    window = ProbeFlowWindow()
    top_menu_names = [action.text() for action in window.menuBar().actions()]

    assert top_menu_names == ["File", "View", "Processing", "Convert", "Tools", "Help"]
    assert not hasattr(window, "_tab_features")
    assert not hasattr(window, "_tab_tv")
    assert not hasattr(window, "_tab_dev")
    assert not hasattr(window, "_tab_defs")

    def action(menu_name: str, text: str):
        top_action = next(
            item for item in window.menuBar().actions() if item.text() == menu_name
        )
        menu = top_action.menu()
        for item in menu.actions():
            if item.text() == text:
                return item
            submenu = item.menu()
            if submenu is not None:
                for subitem in submenu.actions():
                    if subitem.text() == text:
                        return subitem
        raise AssertionError(f"Missing menu action: {menu_name} > {text}")

    assert action("View", "Dark mode" if window._dark else "Light mode").isChecked()
    assert action("View", window._gui_font_size).isChecked()
    assert action("View", window._browse_tools.cmap_cb.currentText()).isChecked()
    assert action("View", "Z").isChecked()
    assert action("Processing", "None").isChecked()

    action("View", "Light mode").trigger()
    assert window._dark is False
    assert action("View", "Light mode").isChecked()

    action("View", "Large").trigger()
    assert window._gui_font_size == "Large"
    assert action("View", "Large").isChecked()

    action("View", "Viridis").trigger()
    assert window._browse_tools.cmap_cb.currentText() == "Viridis"
    assert action("View", "Viridis").isChecked()

    action("View", "Current").trigger()
    assert window._browse_tools.thumbnail_channel_cb.currentText() == "Current"
    assert action("View", "Current").isChecked()

    action("Processing", "Mean").trigger()
    assert window._browse_tools.align_rows_cb.currentText() == "Mean"
    assert action("Processing", "Mean").isChecked()

    action("Tools", "Feature counting").trigger()
    assert window._mode == "features"
    action("Tools", "TV denoise").trigger()
    assert window._mode == "tv"
    action("Tools", "Developer tools").trigger()
    assert window._mode == "dev"
    action("Help", "Definitions").trigger()
    assert window._mode == "dev"
    assert window._definitions_dialog.isVisible()
    definitions_dialog = window._definitions_dialog
    action("Help", "Definitions").trigger()
    assert window._definitions_dialog is definitions_dialog
    definitions_dialog.close()
    action("Convert", "Convert Createc .dat to .sxm...").trigger()
    assert window._mode == "convert"
    action("View", "Browse").trigger()
    assert window._mode == "browse"

    window.close()


def test_definitions_content_includes_bad_scanline_terms():
    from probeflow.gui import _legacy as gui_mod

    definitions = gui_mod._DEFINITIONS_HTML

    for term in (
        "Bad scan-line segment",
        "Threshold",
        "Minimum segment length (px)",
        "Maximum adjacent bad lines",
        "Bright bad segment",
        "Dark bad segment",
        "Preview detection",
        "Apply correction",
        "STM Background",
        "Scan-line profile",
        "Line statistic",
        "Linear fit in x first",
        "Piezo creep",
        "Piezo creep + x^2 / x^3",
        "Sqrt creep",
        "Low-pass background",
        "Line-by-line background",
        "Fit region",
        "Preview background",
        "Preview corrected image",
    ):
        assert term in definitions
