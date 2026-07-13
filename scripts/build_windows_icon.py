#!/usr/bin/env python
"""Generate the ProbeFlow Windows icon from the committed logo asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "probeflow" / "assets" / "logo.gif"
DEFAULT_OUTPUT = REPO_ROOT / "packaging" / "windows" / "ProbeFlow.ico"
ICON_SIZES = ((16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256))


def build_icon(source: Path, output: Path) -> Path:
    """Render a multi-resolution RGBA Windows ICO file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.seek(0)
        image.convert("RGBA").save(output, format="ICO", sizes=ICON_SIZES)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"Created {build_icon(args.source, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
