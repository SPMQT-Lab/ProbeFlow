"""Developer-terminal widgets for the ProbeFlow GUI."""

from __future__ import annotations

import re as _re
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


class _TerminalPane(QTextEdit):
    """Single-pane terminal edit: prompt + input + output all in one area.

    Prompt text is protected from editing.  Up/Down navigate history.
    Ctrl+C sends SIGTERM to the running process.
    """

    command_entered = Signal(str)
    interrupt_requested = Signal()

    _MONO = "Cascadia Code, Consolas, Courier New" if sys.platform == "win32" else "Monospace"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setUndoRedoEnabled(False)
        self.setAcceptRichText(False)
        self._history: list[str] = []
        self._history_idx: int = -1
        self._input_start: int = 0   # document position right after prompt
        self._prompt_str: str = "$ "

    # ── Public API ─────────────────────────────────────────────────────────────
    def set_prompt(self, prompt: str):
        self._prompt_str = prompt

    def show_prompt(self):
        """Append a new prompt line and position cursor after it."""
        from PySide6.QtGui import QTextCharFormat, QTextCursor
        c = self.textCursor()
        c.movePosition(QTextCursor.End)

        # Coloured prompt: cyan cwd portion, white "$"
        fmt_prompt = QTextCharFormat()
        fmt_prompt.setForeground(QColor("#89dceb"))
        fmt_prompt.setFont(QFont(self._MONO, 10))
        c.insertText(self._prompt_str, fmt_prompt)

        # Reset to plain style for typed input
        fmt_plain = QTextCharFormat()
        fmt_plain.setForeground(QColor("#cdd6f4"))
        fmt_plain.setFont(QFont(self._MONO, 10))
        c.setCharFormat(fmt_plain)

        self.setTextCursor(c)
        self._input_start = c.position()
        self._history_idx = -1
        self.ensureCursorVisible()

    def append_output(self, text: str):
        """Insert text at end (before the current input line)."""
        from PySide6.QtGui import QTextCharFormat, QTextCursor
        c = self.textCursor()
        c.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#cdd6f4"))
        fmt.setFont(QFont(self._MONO, 10))
        c.insertText(text, fmt)
        self.setTextCursor(c)
        self.ensureCursorVisible()

    def append_error(self, text: str):
        from PySide6.QtGui import QTextCharFormat, QTextCursor
        c = self.textCursor()
        c.movePosition(QTextCursor.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#f38ba8"))
        fmt.setFont(QFont(self._MONO, 10))
        c.insertText(text, fmt)
        self.setTextCursor(c)
        self.ensureCursorVisible()

    # ── Input text helpers ─────────────────────────────────────────────────────
    def _current_input(self) -> str:
        from PySide6.QtGui import QTextCursor
        c = self.textCursor()
        c.setPosition(self._input_start)
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        return c.selectedText()

    def _replace_input(self, text: str):
        from PySide6.QtGui import QTextCursor
        c = self.textCursor()
        c.setPosition(self._input_start)
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        c.insertText(text)
        self.setTextCursor(c)

    def _clamp_cursor(self):
        from PySide6.QtGui import QTextCursor
        c = self.textCursor()
        if c.position() < self._input_start and not c.hasSelection():
            c.movePosition(QTextCursor.End)
            self.setTextCursor(c)

    # ── Key handling ───────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        from PySide6.QtGui import QTextCursor
        key = event.key()
        mods = event.modifiers()

        # Ctrl+C → interrupt
        if mods & Qt.ControlModifier and key == Qt.Key_C:
            self.interrupt_requested.emit()
            return

        # Ctrl+A → select all input (not entire document)
        if mods & Qt.ControlModifier and key == Qt.Key_A:
            c = self.textCursor()
            c.setPosition(self._input_start)
            c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            self.setTextCursor(c)
            return

        # Enter → submit
        if key in (Qt.Key_Return, Qt.Key_Enter):
            cmd = self._current_input().strip()
            c = self.textCursor()
            c.movePosition(QTextCursor.End)
            c.insertText("\n")
            self.setTextCursor(c)
            self._input_start = c.position()
            if cmd:
                self._history = [x for x in self._history if x != cmd]
                self._history.insert(0, cmd)
            self.command_entered.emit(cmd)
            return

        # Up → history back
        if key == Qt.Key_Up:
            if self._history:
                self._history_idx = min(self._history_idx + 1, len(self._history) - 1)
                self._replace_input(self._history[self._history_idx])
            return

        # Down → history forward
        if key == Qt.Key_Down:
            if self._history_idx > 0:
                self._history_idx -= 1
                self._replace_input(self._history[self._history_idx])
            elif self._history_idx == 0:
                self._history_idx = -1
                self._replace_input("")
            return

        # Home → jump to start of input
        if key == Qt.Key_Home and not (mods & Qt.ShiftModifier):
            c = self.textCursor()
            c.setPosition(self._input_start)
            self.setTextCursor(c)
            return

        # Don't allow editing before input start
        c = self.textCursor()
        if c.position() < self._input_start and not c.hasSelection():
            c.movePosition(QTextCursor.End)
            self.setTextCursor(c)

        # Backspace: never go before the prompt
        if key == Qt.Key_Backspace:
            c = self.textCursor()
            if c.position() <= self._input_start and not c.hasSelection():
                return

        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        # Allow click-to-position for selection/copy; but editing keys will
        # clamp back to the input zone automatically.

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        menu.exec(event.globalPos())


class DeveloperTerminalWidget(QWidget):
    """VS Code-style embedded terminal: prompt + input + output in one pane."""

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self._cwd = Path.cwd()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Top bar
        bar = QWidget()
        bar.setFixedHeight(32)
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(8, 0, 8, 0)
        bar_lay.setSpacing(6)
        title_lbl = QLabel("TERMINAL")
        title_lbl.setFont(QFont("Helvetica", 8, QFont.Bold))
        bar_lay.addWidget(title_lbl)
        bar_lay.addStretch()

        ext_btn = QPushButton("Open External Terminal")
        ext_btn.setFont(QFont("Helvetica", 8))
        ext_btn.setFixedHeight(22)
        ext_btn.setToolTip(
            "Open a real system terminal in this directory.\n"
            "Use this for interactive tools like claude, ipython, vim, etc.")
        ext_btn.clicked.connect(self._open_external)
        bar_lay.addWidget(ext_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFont(QFont("Helvetica", 8))
        clear_btn.setFixedHeight(22)
        clear_btn.clicked.connect(self._clear)
        bar_lay.addWidget(clear_btn)
        lay.addWidget(bar)

        # Terminal pane
        self._pane = _TerminalPane(self)
        self._pane.command_entered.connect(self._run_command)
        self._pane.interrupt_requested.connect(self._interrupt)
        lay.addWidget(self._pane, 1)

        # Process
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        self._process.setWorkingDirectory(str(self._cwd))

        self._apply_theme(t)
        self._set_cwd(self._cwd)

    def _set_cwd(self, cwd: Path):
        self._cwd = cwd
        self._process.setWorkingDirectory(str(cwd))
        self._pane.set_prompt(f"{cwd.name} $ ")
        self._pane.show_prompt()

    def _apply_theme(self, t: dict):
        bg = t.get("bg", "#1e1e2e")
        fg = t.get("fg", "#cdd6f4")
        self._pane.setStyleSheet(
            f"QTextEdit {{ background-color: {bg}; color: {fg}; "
            f"border: none; padding: 6px; }}")
        self.setStyleSheet(f"background-color: {bg};")
        # tint the top bar slightly darker
        bar_bg = t.get("sidebar_bg", "#181825")
        self.findChild(QWidget).setStyleSheet(
            f"background-color: {bar_bg}; color: {fg};")

    def _clear(self):
        self._pane.clear()
        self._set_cwd(self._cwd)

    # Tools that require a real PTY — intercept and open external terminal.
    _PTY_TOOLS = frozenset({
        "claude", "ipython", "ipython3",
        "vim", "vi", "nvim", "nano", "emacs",
        "htop", "top", "btop", "less", "more", "man",
        "ssh", "sftp", "ftp", "telnet",
    })

    def _run_command(self, cmd: str):
        if not cmd:
            self._pane.show_prompt()
            return
        if self._process.state() != QProcess.NotRunning:
            self._pane.append_error("[busy — press Ctrl+C to interrupt]\n")
            self._pane.show_prompt()
            return

        # Detect interactive tools that need a real PTY
        first = cmd.split()[0].lstrip("./") if cmd.split() else ""
        # plain `python` / `python3` with no args also needs a PTY
        is_bare_python = first in ("python", "python3", "python3.x") and len(cmd.split()) == 1
        if first in self._PTY_TOOLS or is_bare_python:
            self._pane.append_output(
                f"['{first}' needs an interactive terminal — opening external terminal…]\n")
            self._pane.show_prompt()
            self._open_external()
            return

        import shutil
        if sys.platform.startswith("win"):
            shell, args = "cmd.exe", ["/c", cmd]
        else:
            shell = shutil.which("bash") or "/bin/sh"
            args = ["-c", cmd]
        self._process.start(shell, args)

    def _interrupt(self):
        if self._process.state() != QProcess.NotRunning:
            self._process.kill()

    def _open_external(self):
        """Launch the real system terminal in the current working directory.

        Interactive tools (claude, ipython, vim, etc.) need a proper PTY —
        use this button instead of running them in the embedded pane.
        """
        cwd = str(self._cwd)
        try:
            if sys.platform.startswith("win"):
                # Try Windows Terminal first, fall back to wt, then cmd
                import shutil
                if shutil.which("wt"):
                    subprocess.Popen(["wt", "--startingDirectory", cwd])
                else:
                    subprocess.Popen(["cmd.exe"], cwd=cwd,
                                     creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                # On WSL: open wsl.exe so we stay in the Linux env
                # Check if we're inside WSL
                is_wsl = Path("/proc/version").exists() and \
                    "microsoft" in Path("/proc/version").read_text().lower()
                if is_wsl:
                    subprocess.Popen(
                        ["wsl.exe", "--cd", cwd],
                        start_new_session=True,
                    )
                else:
                    # Plain Linux: try common terminal emulators
                    import shutil as _sh
                    for term in ("gnome-terminal", "xterm", "konsole", "xfce4-terminal"):
                        if _sh.which(term):
                            subprocess.Popen([term], cwd=cwd, start_new_session=True)
                            break
            self._pane.append_output("[External terminal opened]\n")
        except Exception as exc:
            self._pane.append_error(f"[Could not open external terminal: {exc}]\n")
        self._pane.show_prompt()

    def _on_output(self):
        raw = bytes(self._process.readAllStandardOutput())
        text = raw.decode("utf-8", errors="replace")
        text = _re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
        text = _re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", text)
        if text:
            self._pane.append_output(text)

    def _on_finished(self, exit_code: int, exit_status):
        if exit_code != 0:
            self._pane.append_error(f"[exited {exit_code}]\n")
        self._pane.show_prompt()

    def _on_error(self, error):
        self._pane.append_error(f"[error: {error}]\n")
        self._pane.show_prompt()


# ── Developer terminal sidebar ────────────────────────────────────────────────
class _DevSidebar(QWidget):
    """Sidebar for the Dev tab: shows cwd, quick links, and info."""

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        title = QLabel("Developer Mode")
        title.setFont(QFont("Helvetica", 11, QFont.Bold))
        lay.addWidget(title)

        info = QLabel(
            "Run shell commands and Python scripts in the ProbeFlow environment.\n\n"
            "↑/↓ — command history   Ctrl+C — interrupt\n\n"
            "All ProbeFlow packages available: import probeflow, numpy, scipy…\n\n"
            "⚠  Interactive tools (claude, ipython, vim) need a real PTY.\n"
            "Use 'Open External Terminal' to launch WSL / Windows Terminal."
        )
        info.setFont(QFont("Helvetica", 9))
        info.setWordWrap(True)
        lay.addWidget(info)

        example_lbl = QLabel("Quick examples:")
        example_lbl.setFont(QFont("Helvetica", 9, QFont.Bold))
        lay.addWidget(example_lbl)

        examples = QLabel(
            "python3 -c \"import probeflow; print(probeflow.__file__)\"\n\n"
            "python3 scripts/my_analysis.py\n\n"
            "ls -lh *.dat\n\n"
            "python3 -c \"from probeflow.io.readers.createc_scan import read_dat; "
            "import numpy as np; s = read_dat('scan.dat'); "
            "print(np.nanmin(s.planes[0])*1e10, 'A')\""
        )
        examples.setFont(QFont("Courier New" if sys.platform == "win32" else "Monospace", 8))
        examples.setWordWrap(True)
        examples.setStyleSheet("color: #888;")
        lay.addWidget(examples)

        lay.addStretch()
