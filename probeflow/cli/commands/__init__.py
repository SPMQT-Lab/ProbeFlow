"""Extraction staging area for CLI command runners.

Each submodule re-exports the ``_cmd_*`` runner functions that currently live
in ``probeflow/cli/_legacy.py``.  The submodules are not yet wired into the
parser; the active entry point is still ``cli/_legacy.py:_build_parser``.

Migration path for each command
--------------------------------
1. Move the ``_cmd_*`` function (and any private helpers it calls) from
   ``cli/_legacy.py`` into the appropriate submodule here.
2. Update ``cli/parser.py`` to import the runner from the submodule instead
   of from ``_legacy``.
3. Remove the name from the submodule's ``__all__`` re-export once the
   original copy in ``_legacy`` is gone.

New commands should be added directly to the right submodule — not to
``_legacy``.
"""
