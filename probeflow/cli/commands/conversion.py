"""Legacy vendor conversion CLI commands."""

from __future__ import annotations

import sys


def _cmd_dat2sxm(args) -> int:
    from probeflow.io.converters.createc_dat_to_sxm import main as _main
    forwarded = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    sys.argv = ["dat-sxm"] + forwarded
    return _main() or 0


def _cmd_dat2png(args) -> int:
    from probeflow.io.converters.createc_dat_to_png import main as _main
    forwarded = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    sys.argv = ["dat-png"] + forwarded
    return _main() or 0


def _cmd_dat2npy(args) -> int:
    from probeflow.io.converters.createc_dat_to_npy import main as _main
    forwarded = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    sys.argv = ["dat-npy"] + forwarded
    return _main() or 0


__all__ = ["_cmd_dat2png", "_cmd_dat2npy", "_cmd_dat2sxm"]
