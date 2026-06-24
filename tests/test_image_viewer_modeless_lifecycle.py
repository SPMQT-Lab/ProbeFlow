"""Tests for the ImageViewerDialog modeless-child registry (review gui-arch #22).

The registry collects modeless child dialogs created via ``dlg.show()`` so
that :meth:`closeEvent` can iterate-close them before the viewer tears down,
preventing queued signal handlers from firing on a partially-destroyed
viewer (the f6aac4a class of RuntimeError).

These tests poke the mixin methods directly with mock dialogs — no Qt
event loop is required, so they run on every platform.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from probeflow.gui.viewer.image_viewer_processing_export_mixin import (
    ImageViewerProcessingExportMixin,
)


class _Host(ImageViewerProcessingExportMixin):
    """Minimal host that exposes the mixin methods without needing a QDialog."""

    def __init__(self) -> None:
        self._modeless_children: list = []


def _fake_dialog(visible: bool = True, raise_close: bool = False):
    dlg = MagicMock()
    dlg.isVisible.return_value = visible
    if raise_close:
        dlg.close.side_effect = RuntimeError("forced close failure")
    # ``destroyed`` is a Qt signal; emulate connect() as a no-op so the
    # tracker does not blow up.
    dlg.destroyed = MagicMock()
    dlg.destroyed.connect = MagicMock()
    dlg.force_close = None
    return dlg


def test_track_modeless_child_appends_dialog():
    host = _Host()
    dlg = _fake_dialog()

    host._track_modeless_child(dlg)

    assert host._modeless_children == [dlg]
    # The tracker must subscribe to ``destroyed`` so the entry is reaped
    # when Qt deallocates the dialog — otherwise the list grows unbounded
    # over the viewer's lifetime.
    dlg.destroyed.connect.assert_called_once()


def test_track_modeless_child_handles_none():
    host = _Host()
    host._track_modeless_child(None)
    assert host._modeless_children == []


def test_track_modeless_child_handles_missing_attribute():
    host = _Host()
    del host._modeless_children
    dlg = _fake_dialog()
    host._track_modeless_child(dlg)
    assert host._modeless_children == [dlg]


def test_untrack_modeless_child_removes_entry():
    host = _Host()
    dlg = _fake_dialog()
    host._modeless_children.append(dlg)

    host._untrack_modeless_child(dlg)

    assert host._modeless_children == []


def test_untrack_modeless_child_tolerates_missing_entry():
    host = _Host()
    host._untrack_modeless_child(_fake_dialog())  # not in list
    assert host._modeless_children == []


def test_close_modeless_children_closes_visible_dialogs():
    host = _Host()
    visible = _fake_dialog(visible=True)
    hidden = _fake_dialog(visible=False)
    host._modeless_children.extend([visible, hidden])

    host._close_modeless_children()

    visible.close.assert_called_once()
    hidden.close.assert_not_called()


def test_close_modeless_children_uses_force_close_when_available():
    host = _Host()
    dialog = _fake_dialog(visible=True)
    dialog.force_close = MagicMock()
    host._modeless_children.append(dialog)

    host._close_modeless_children()

    dialog.force_close.assert_called_once()
    dialog.close.assert_not_called()


def test_close_modeless_children_tolerates_runtime_error_on_isvisible():
    """A child whose C++ object was already deleted raises RuntimeError on
    ``isVisible()``. Teardown must not cascade out of this."""

    host = _Host()
    dead = _fake_dialog()
    dead.isVisible.side_effect = RuntimeError("Internal C++ object deleted.")
    alive = _fake_dialog(visible=True)
    host._modeless_children.extend([dead, alive])

    host._close_modeless_children()

    # The live dialog still gets a close() call even though the dead one
    # raised first — the iteration continues.
    alive.close.assert_called_once()
    dead.close.assert_not_called()


def test_close_modeless_children_tolerates_runtime_error_on_close():
    host = _Host()
    flaky = _fake_dialog(visible=True, raise_close=True)
    healthy = _fake_dialog(visible=True)
    host._modeless_children.extend([flaky, healthy])

    host._close_modeless_children()

    flaky.close.assert_called_once()
    healthy.close.assert_called_once()


def test_close_modeless_children_handles_missing_attribute():
    host = _Host()
    del host._modeless_children
    # No exception even when the list was never set up.
    host._close_modeless_children()
