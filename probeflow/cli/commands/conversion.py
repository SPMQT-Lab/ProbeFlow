"""Legacy vendor conversion CLI commands."""

from __future__ import annotations

import sys


def _cmd_dat2sxm(args) -> int:
    from probeflow.io.converters.createc_dat_to_sxm import main as _main
    forwarded = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    sys.argv = ["dat-sxm"] + forwarded
    _main()
    return 0


def _cmd_dat2png(args) -> int:
    from probeflow.io.converters.createc_dat_to_png import main as _main
    forwarded = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    sys.argv = ["dat-png"] + forwarded
    _main()
    return 0


__all__ = ["_cmd_dat2png", "_cmd_dat2sxm"]
