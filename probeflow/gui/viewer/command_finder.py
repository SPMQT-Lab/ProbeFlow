"""Image-viewer command finder dialog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from probeflow.gui.viewer.shortcuts import (
    VIEWER_COMMAND_BY_ID,
    ViewerCommand,
    display_shortcuts_for_all_platforms,
    viewer_finder_commands,
)


_COMMAND_ID_ROLE = Qt.UserRole
_ENABLED_ROLE = Qt.UserRole + 1


@dataclass(frozen=True)
class CommandMatch:
    command: ViewerCommand
    score: int


def _compact(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _is_subsequence(needle: str, haystack: str) -> bool:
    if not needle:
        return True
    pos = 0
    for ch in haystack:
        if ch == needle[pos]:
            pos += 1
            if pos == len(needle):
                return True
    return False


def command_search_text(command: ViewerCommand) -> str:
    return " ".join((
        command.label,
        command.menu_group,
        command.command_id.replace(".", " "),
        " ".join(command.aliases),
    )).casefold()


def command_match_score(command: ViewerCommand, query: str) -> int | None:
    query = " ".join(str(query).casefold().split())
    if not query:
        return 0

    label = command.label.casefold()
    haystack = command_search_text(command)
    if label == query:
        return 1000
    if label.startswith(query):
        return 900
    if query in label:
        return 800 - min(label.index(query), 100)
    if any(query in alias.casefold() for alias in command.aliases):
        return 760

    tokens = query.split()
    if all(token in haystack for token in tokens):
        return 650 - min(sum(haystack.index(token) for token in tokens), 100)

    compact_query = _compact(query)
    if not compact_query:
        return 0
    compact_label = _compact(command.label)
    compact_haystack = _compact(haystack)
    if compact_query in compact_label:
        return 560 - min(compact_label.index(compact_query), 100)
    if _is_subsequence(compact_query, compact_haystack):
        return 420 - min(len(compact_haystack) - len(compact_query), 100)
    return None


def filter_viewer_commands(
    commands: tuple[ViewerCommand, ...] | list[ViewerCommand],
    query: str,
) -> list[ViewerCommand]:
    matches: list[CommandMatch] = []
    for command in commands:
        score = command_match_score(command, query)
        if score is not None:
            matches.append(CommandMatch(command, score))
    matches.sort(
        key=lambda match: (
            -match.score,
            match.command.menu_group.casefold(),
            match.command.label.casefold(),
        )
    )
    return [match.command for match in matches]


class CommandFinderDialog(QDialog):
    """Searchable command launcher for image-viewer menu actions."""

    def __init__(
        self,
        actions_by_id: Mapping[str, QAction],
        *,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Command finder")
        self.setModal(False)
        self.resize(620, 520)

        self._actions_by_id: dict[str, QAction] = dict(actions_by_id)
        self._commands: tuple[ViewerCommand, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search image-viewer commands")
        layout.addWidget(self._search)

        self._list = QListWidget()
        layout.addWidget(self._list, 1)

        self._detail = QLabel("")
        self._detail.setWordWrap(True)
        layout.addWidget(self._detail)

        row = QHBoxLayout()
        self._close_after_run = QCheckBox("Close after running")
        self._close_after_run.setChecked(True)
        row.addWidget(self._close_after_run)
        row.addStretch()

        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._run_selected)
        row.addWidget(self._run_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        row.addWidget(close_btn)
        layout.addLayout(row)

        self._search.textChanged.connect(self._refresh_list)
        self._search.returnPressed.connect(self._run_selected)
        self._list.itemDoubleClicked.connect(self._run_item)
        self._list.currentItemChanged.connect(lambda _new, _old: self._update_detail())
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.focus_search)

        self.set_actions(actions_by_id)

    def set_actions(self, actions_by_id: Mapping[str, QAction]) -> None:
        self._actions_by_id = dict(actions_by_id)
        self._commands = tuple(
            command for command in viewer_finder_commands()
            if command.command_id in self._actions_by_id
        )
        self._refresh_list()

    def focus_search(self) -> None:
        self._search.setFocus(Qt.ShortcutFocusReason)
        self._search.selectAll()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() == Qt.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def _refresh_list(self) -> None:
        current_query = self._search.text()
        selected_command_id = self._current_command_id()
        commands = filter_viewer_commands(self._commands, current_query)

        self._list.clear()
        select_row = 0
        for row, command in enumerate(commands):
            action = self._actions_by_id.get(command.command_id)
            enabled = bool(action is not None and action.isEnabled())
            shortcut = display_shortcuts_for_all_platforms(command.shortcuts)
            suffix = f"    {shortcut}" if shortcut else ""
            disabled = "" if enabled else "    unavailable"
            item = QListWidgetItem(f"{command.label}{suffix}{disabled}")
            item.setData(_COMMAND_ID_ROLE, command.command_id)
            item.setData(_ENABLED_ROLE, enabled)
            item.setToolTip(command.status_tip)
            if not enabled:
                item.setForeground(QBrush(QColor("#7a7f87")))
            self._list.addItem(item)
            if command.command_id == selected_command_id:
                select_row = row

        if self._list.count():
            self._list.setCurrentRow(min(select_row, self._list.count() - 1))
        self._update_detail()

    def _current_command_id(self) -> str | None:
        item = self._list.currentItem()
        if item is None:
            return None
        value = item.data(_COMMAND_ID_ROLE)
        return str(value) if value else None

    def _update_detail(self) -> None:
        command_id = self._current_command_id()
        command = VIEWER_COMMAND_BY_ID.get(command_id or "")
        action = self._actions_by_id.get(command_id or "")
        enabled = bool(action is not None and action.isEnabled())
        self._run_btn.setEnabled(enabled)
        if command is None:
            self._detail.setText("No matching commands.")
            return
        shortcut = display_shortcuts_for_all_platforms(command.shortcuts)
        pieces = [command.menu_group]
        if shortcut:
            pieces.append(shortcut)
        if not enabled:
            pieces.append("Unavailable")
        detail = "  |  ".join(pieces)
        if command.status_tip:
            detail = f"{detail}\n{command.status_tip}"
        self._detail.setText(detail)

    def _run_item(self, item: QListWidgetItem) -> None:
        command_id = str(item.data(_COMMAND_ID_ROLE) or "")
        self._run_command(command_id)

    def _run_selected(self) -> None:
        command_id = self._current_command_id()
        if command_id:
            self._run_command(command_id)

    def _run_command(self, command_id: str) -> None:
        action = self._actions_by_id.get(command_id)
        if action is None or not action.isEnabled():
            self._update_detail()
            return
        if self._close_after_run.isChecked():
            self.accept()
        action.trigger()
