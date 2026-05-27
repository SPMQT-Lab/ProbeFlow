"""Tests for image-viewer shortcut metadata."""

from __future__ import annotations

from probeflow.gui.viewer.shortcuts import (
    VIEWER_COMMANDS,
    VIEWER_COMMAND_BY_ID,
    display_shortcuts_for_all_platforms,
    viewer_command,
    viewer_finder_commands,
)


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


def test_command_finder_shortcut_and_visible_commands_are_high_level():
    assert "Ctrl+K" in viewer_command("viewer.command_finder").shortcuts
    assert not viewer_command("viewer.command_finder").finder_visible

    finder_ids = {command.command_id for command in viewer_finder_commands()}
    assert "processing.stm_background" in finder_ids
    assert "processing.image_operations" in finder_ids
    assert "fft.periodic_filter" in finder_ids
    assert "measure.clear_lattice_grid" in finder_ids
    assert "help.definitions" in finder_ids
    assert not any(command_id.startswith("roi.tool.") for command_id in finder_ids)
    assert "image.threshold" in finder_ids


def test_shortcut_help_shows_mac_and_windows_forms():
    assert display_shortcuts_for_all_platforms(("Ctrl+K",)) == "⌘K / Ctrl+K"
    assert display_shortcuts_for_all_platforms(
        ("Ctrl+Y", "Ctrl+Shift+Z")
    ) == "⌘Y / Ctrl+Y; ⇧⌘Z / Ctrl+Shift+Z"
