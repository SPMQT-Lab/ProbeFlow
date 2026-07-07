"""Independent top-level window hosting one workspace (panel + sidebar).

Workspaces (STM File Converter, TV denoise, Dataset Builder, Survey,
Developer tools) used to live as pages in the main window's stacked
layout; they now open as separate windows so users can flick between a
workspace and Browse without losing either — the same arrangement the
floating Feature Counting window (:class:`FeatureCountingWindow`) has
always used.

Usage (from ProbeFlowWindow)
----------------------------
    win = WorkspaceWindow(key="tv", title="TV Denoise",
                          panel=tv_panel, sidebar=tv_sidebar, parent=self)
    win.show()

Closing the window merely hides it (Qt's default for a top-level widget
that is not ``WA_DeleteOnClose``), so all panel state survives a
close/reopen cycle.  ``closeEvent`` is deliberately not overridden:
an ``event.ignore()`` there would abort ``closeAllWindows()`` and break
macOS Cmd+Q.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QSplitter, QStatusBar, QWidget

from probeflow.gui.typography import ui_font


class WorkspaceWindow(QMainWindow):
    """A workspace's panel + optional sidebar in an independent window."""

    def __init__(self, *, key: str, title: str, panel: QWidget,
                 sidebar: QWidget | None = None,
                 parent: QWidget | None = None,
                 size: tuple[int, int] = (1200, 760),
                 splitter_sizes: tuple[int, int] = (860, 340),
                 sidebar_min_w: int = 300):
        # Qt.Window ensures an independent top-level window with its own
        # taskbar entry; the parent link means Qt tears these down with the
        # main window, so quitOnLastWindowClosed still fires correctly.
        super().__init__(parent, Qt.Window)
        self.key = key
        self.setWindowTitle(f"ProbeFlow — {title}")
        self.resize(*size)

        self._panel = panel
        self._sidebar = sidebar

        self._status_bar = QStatusBar()
        self._status_bar.setFont(ui_font(10))
        self.setStatusBar(self._status_bar)

        if sidebar is not None:
            sidebar.setMinimumWidth(sidebar_min_w)
            splitter = QSplitter(Qt.Horizontal)
            splitter.addWidget(panel)
            splitter.addWidget(sidebar)
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes(list(splitter_sizes))
            self.setCentralWidget(splitter)
        else:
            self.setCentralWidget(panel)

    def show_status(self, msg: str) -> None:
        self._status_bar.showMessage(msg)

    def apply_theme(self, t: dict) -> None:
        """Sync with the host main window's theme.

        Most visible styling comes from the QApplication-level QSS, which
        reaches every top-level window automatically; this forwards the
        theme dict to panels that restyle themselves.
        """
        for w in (self._panel, self._sidebar):
            if w is None:
                continue
            if hasattr(w, "apply_theme"):
                w.apply_theme(t)
            elif hasattr(w, "_t"):
                w._t = dict(t)
