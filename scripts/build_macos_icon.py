#!/usr/bin/env python
"""Generate the ProbeFlow macOS icon from the committed logo asset."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import struct

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "probeflow" / "assets" / "logo.gif"
DEFAULT_OUTPUT = REPO_ROOT / "packaging" / "macos" / "ProbeFlow.icns"
ICNS_REPRESENTATIONS = (
    (b"icp4", 16),
    (b"icp5", 32),
    (b"icp6", 64),
    (b"ic07", 128),
    (b"ic08", 256),
    (b"ic09", 512),
    (b"ic10", 1024),
    (b"ic11", 32),
    (b"ic12", 64),
    (b"ic13", 256),
    (b"ic14", 512),
)


def build_icon(source: Path, output: Path) -> Path:
    """Render modern PNG-backed representations into an ICNS container."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.seek(0)
        master = image.convert("RGBA")
        chunks: list[bytes] = []
        rendered: dict[int, bytes] = {}
        for icon_type, size in ICNS_REPRESENTATIONS:
            if size not in rendered:
                buffer = BytesIO()
                master.resize((size, size), Image.Resampling.LANCZOS).save(
                    buffer,
                    format="PNG",
                )
                rendered[size] = buffer.getvalue()
            payload = rendered[size]
            chunks.append(icon_type + struct.pack(">I", len(payload) + 8) + payload)
    body = b"".join(chunks)
    output.write_bytes(b"icns" + struct.pack(">I", len(body) + 8) + body)
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
