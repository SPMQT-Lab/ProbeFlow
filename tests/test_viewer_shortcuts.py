"""Tests for image-viewer shortcut metadata."""

from __future__ import annotations

from probeflow.gui.viewer.shortcuts import VIEWER_COMMANDS, VIEWER_COMMAND_BY_ID


def _norm(shortcut: str) -> str:
    return shortcut.replace(" ", "").lower()


def test_viewer_shortcut_registry_is_unique_and_avoids_roi_keys():
    command_ids = [command.command_id for command in VIEWER_COMMANDS]
    assert len(command_ids) == len(set(command_ids))
    assert set(command_ids) == set(VIEWER_COMMAND_BY_ID)

    shortcuts = [
        _norm(shortcut)
        for command in VIEWER_COMMANDS
        for shortcut in command.shortcuts
    ]
    assert len(shortcuts) == len(set(shortcuts))

    reserved_roi_keys = {
        "r",
        "e",
        "p",
        "f",
        "l",
        "t",
        "i",
        "delete",
        "backspace",
        "left",
        "right",
        *{str(i) for i in range(1, 10)},
    }
    assert not reserved_roi_keys.intersection(shortcuts)
