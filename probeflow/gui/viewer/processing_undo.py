"""Processing undo/redo controller extracted from ImageViewerDialog."""

from __future__ import annotations

import copy
from typing import Callable

_DEPTH = 50


class ProcessingUndoController:
    """Owns the undo/redo stacks for image processing state.

    Callers push a snapshot before mutating ``_processing`` (Apply / Reset),
    then pop it on Undo or Redo.  The controller drives button-enabled state
    and syncs the viewer menu after every stack change.
    """

    def __init__(
        self,
        undo_btn,
        redo_btn,
        sync_menu_fn: Callable[[], None],
    ) -> None:
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._undo_btn = undo_btn
        self._redo_btn = redo_btn
        self._sync_menu_fn = sync_menu_fn

    # ── Read-only state ───────────────────────────────────────────────────────

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    # ── Stack operations ──────────────────────────────────────────────────────

    def push(self, processing: dict) -> None:
        """Deep-copy *processing* onto the undo stack and clear redo."""
        self._undo_stack.append(copy.deepcopy(processing))
        if len(self._undo_stack) > _DEPTH:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self.update_buttons()

    def try_coalesce(self, base_state: dict) -> bool:
        """Coalesce if the last undo snapshot equals *base_state*.

        Returns ``True`` (clearing Redo and updating buttons) when coalesced —
        the caller must NOT also call :meth:`push`.  Returns ``False`` when the
        caller should call :meth:`push` as normal.
        """
        if self._undo_stack and self._undo_stack[-1] == base_state:
            self._redo_stack.clear()
            self.update_buttons()
            return True
        return False

    def discard_last_undo_if_eq(self, processing: dict) -> None:
        """Pop the most recent undo snapshot if it equals *processing*.

        Used by the coalesced-align-undo path to discard a no-op snapshot
        (when the user toggled align back to the same value).
        """
        if self._undo_stack and processing == self._undo_stack[-1]:
            self._undo_stack.pop()
            self.update_buttons()

    def undo(self, current: dict) -> dict | None:
        """Pop undo stack, push *current* to redo.

        Returns the state to restore, or ``None`` if the stack is empty.
        """
        if not self._undo_stack:
            return None
        self._redo_stack.append(copy.deepcopy(current))
        state = self._undo_stack.pop()
        self.update_buttons()
        return state

    def redo(self, current: dict) -> dict | None:
        """Pop redo stack, push *current* to undo.

        Returns the state to restore, or ``None`` if the stack is empty.
        """
        if not self._redo_stack:
            return None
        self._undo_stack.append(copy.deepcopy(current))
        state = self._redo_stack.pop()
        self.update_buttons()
        return state

    # ── Button sync ───────────────────────────────────────────────────────────

    def update_buttons(self) -> None:
        if self._undo_btn is not None:
            self._undo_btn.setEnabled(self.can_undo)
        if self._redo_btn is not None:
            self._redo_btn.setEnabled(self.can_redo)
        self._sync_menu_fn()
