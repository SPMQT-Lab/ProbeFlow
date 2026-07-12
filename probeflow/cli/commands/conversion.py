"""Legacy vendor conversion CLI commands."""

from __future__ import annotations

import sys


def _run_converter(main, program: str, rest: list[str]) -> int:
    """Run a legacy converter without leaking its argv into the caller."""
    forwarded = rest[1:] if rest and rest[0] == "--" else rest
    previous_argv = sys.argv
    try:
        sys.argv = [program, *forwarded]
        return main() or 0
    finally:
        sys.argv = previous_argv


def _cmd_dat2sxm(args) -> int:
    from probeflow.io.converters.createc_dat_to_sxm import main as _main
    return _run_converter(_main, "dat-sxm", args.rest)


def _cmd_dat2png(args) -> int:
    from probeflow.io.converters.createc_dat_to_png import main as _main
    return _run_converter(_main, "dat-png", args.rest)


def _cmd_dat2npy(args) -> int:
    from probeflow.io.converters.createc_dat_to_npy import main as _main
    return _run_converter(_main, "dat-npy", args.rest)


__all__ = ["_cmd_dat2png", "_cmd_dat2npy", "_cmd_dat2sxm"]
