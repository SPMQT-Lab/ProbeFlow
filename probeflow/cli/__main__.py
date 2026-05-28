"""Module execution entry point for ``python -m probeflow.cli``."""

from __future__ import annotations

import sys

from probeflow.cli import main


if __name__ == "__main__":
    sys.exit(main())
