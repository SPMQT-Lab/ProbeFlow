"""CLI command runner submodules.

Each ``_cmd_*`` runner lives in the topic-specific submodule below
(``analysis``, ``conversion``, ``info``, ``processing``, ``roi``,
``scan``, ``spectroscopy``, ``gui``).  The runners are wired into
``cli/parser.py`` and called from there.

Add new CLI commands directly to the appropriate submodule.  The
legacy ``cli/_legacy.py`` module is now a backward-compatibility
re-export shim only — do not add new entries there.

(Review arch-backend #17, 2026-05-28: the previous docstring claimed
this was an in-progress extraction staging area pointing at
``_legacy._build_parser`` as the active entry point.  That migration
is complete; the description here now reflects the actual structure.)
"""
