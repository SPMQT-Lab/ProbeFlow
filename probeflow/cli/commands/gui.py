"""GUI launch CLI command."""

from __future__ import annotations

from pathlib import Path


def _cmd_gui(args) -> int:
    from probeflow.gui import main as _gui_main
    survey = getattr(args, "open_survey", None)
    if survey is not None:
        _gui_main(open_survey=Path(survey))
    else:
        _gui_main()
    return 0


__all__ = ["_cmd_gui"]
