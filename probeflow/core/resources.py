"""Runtime resource paths packaged with ProbeFlow."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PACKAGE_ROOT / "assets"
FILE_CUSHIONS_DIR = PACKAGE_ROOT / "data" / "file_cushions"


def asset_path(name: str) -> Path:
    """Return the path to a packaged GUI asset."""
    return ASSETS_DIR / name
