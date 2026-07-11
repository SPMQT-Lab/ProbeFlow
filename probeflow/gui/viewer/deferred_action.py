"""Deferred plane action contract for ImageViewerDialog → ProbeFlowWindow handoff."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeferredPlaneAction:
    """Records a 'send to …' request made inside an ImageViewerDialog.

    The viewer sets ``action`` and ``plane_idx`` before calling
    ``dialog.accept()``.  The parent window reads these attributes after
    ``dialog.exec()`` returns to determine what to do with the image data.

    Attributes
    ----------
    action:
        One of ``"tv"`` or ``""`` (no action requested).
    plane_idx:
        Index of the channel/plane the user had selected when they triggered
        the send action.
    """

    action: str = ""
    plane_idx: int = 0

    def is_pending(self) -> bool:
        return self.action in ("tv",)

    def clear(self) -> None:
        self.action = ""
        self.plane_idx = 0
