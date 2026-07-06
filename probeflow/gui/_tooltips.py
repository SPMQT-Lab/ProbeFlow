"""Shared tooltip helpers for ProbeFlow GUI widgets.

Tooltip wrapping is owned by the application-wide event filter in
:mod:`probeflow.gui.tooltips` (``install_global_tooltips``), which reformats
every tooltip into a width-capped rich-text column at show time.  ``tip``
therefore passes text through unchanged; it survives only so the many
existing ``_tip("...")`` call sites keep working.  New code can call
``setToolTip`` with plain text directly.
"""

from __future__ import annotations


def tip(text: str, width: int = 50) -> str:
    """Return tooltip text unchanged (wrapping happens globally at show time).

    ``width`` is accepted for backward compatibility and ignored — hard-wrapping
    here used to fight the global rich-text wrapper: the pre-broken lines became
    ``<br>`` breaks, giving these tooltips a ragged, narrower column than every
    other tooltip in the app.
    """
    del width
    return text
