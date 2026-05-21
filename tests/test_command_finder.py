"""Tests for image-viewer command finder helpers."""

from __future__ import annotations

import pytest


pytest.importorskip("PySide6.QtWidgets")


def test_command_finder_filters_and_ranks_high_level_commands():
    from probeflow.gui.viewer.command_finder import filter_viewer_commands
    from probeflow.gui.viewer.shortcuts import viewer_finder_commands

    commands = viewer_finder_commands()

    stm = filter_viewer_commands(commands, "stm")
    assert stm
    assert stm[0].command_id == "processing.stm_background"

    periodic = filter_viewer_commands(commands, "periodic")
    assert periodic
    assert periodic[0].command_id == "fft.periodic_filter"

    arithmetic = filter_viewer_commands(commands, "image math")
    assert arithmetic
    assert arithmetic[0].command_id == "processing.image_operations"

    assert filter_viewer_commands(commands, "threshold") == []
