"""GUI launch CLI command."""

from __future__ import annotations

from pathlib import Path


def _cmd_gui(args) -> int:
    from probeflow.gui import main as _gui_main
    survey = getattr(args, "open_survey", None)
    browse = getattr(args, "browse", None)
    _gui_main(
        open_survey=Path(survey) if survey is not None else None,
        browse_folder=Path(browse) if browse is not None else None,
    )
    return 0


__all__ = ["_cmd_gui"]
