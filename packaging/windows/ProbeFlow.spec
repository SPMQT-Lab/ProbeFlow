# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the native Windows x64 ProbeFlow application."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import tomllib

from PyInstaller.utils.hooks import collect_submodules, copy_metadata


ROOT = Path(SPECPATH).resolve().parents[1]
WINDOWS_DIR = ROOT / "packaging" / "windows"
METADATA = tomllib.loads(
    (WINDOWS_DIR / "app_metadata.toml").read_text(encoding="utf-8")
)["application"]
LICENSE_DIR = Path(
    os.environ.get("PROBEFLOW_LICENSE_DIR", ROOT / "build" / "windows" / "licenses")
)
if not LICENSE_DIR.is_dir():
    raise FileNotFoundError(
        f"Release license bundle is missing: {LICENSE_DIR}; use scripts/build_windows_app.ps1"
    )


def literal_assignment(path: Path, name: str) -> str:
    """Read a top-level literal assignment without importing the package."""

    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return str(ast.literal_eval(node.value))
    raise ValueError(f"{name} is not a top-level literal assignment in {path}")


version_module, version_attribute = METADATA["version_source"].rsplit(".", 1)
module_path = ROOT.joinpath(*version_module.split("."))
version_path = (
    module_path / "__init__.py" if module_path.is_dir() else module_path.with_suffix(".py")
)
SOURCE_VERSION = literal_assignment(version_path, version_attribute)
expected_product_version = SOURCE_VERSION.replace("rc", " RC ")
if expected_product_version != METADATA["product_version"]:
    raise ValueError("The package version and Windows product version describe different releases")

hidden_imports = collect_submodules("probeflow.gui")
hidden_imports += [
    "cv2",
    "gwyfile.objects",
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_pdf",
    "matplotlib.backends.backend_qtagg",
    "probeflow.analysis.lattice",
    "probeflow.io.writers.gwy",
    "sklearn.cluster",
    "sklearn.metrics",
]

datas = [
    (str(ROOT / "probeflow" / "assets"), "probeflow/assets"),
    (str(ROOT / "probeflow" / "data" / "file_cushions"), "probeflow/data/file_cushions"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "packaging" / "THIRD_PARTY_NOTICES.md"), "."),
    (str(WINDOWS_DIR / "QT_LGPL_COMPLIANCE.md"), "."),
    (str(LICENSE_DIR), "THIRD_PARTY_LICENSES"),
]
datas += copy_metadata("gwyfile")

a = Analysis(
    [str(ROOT / "probeflow" / "gui" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[str(ROOT / "packaging" / "pyinstaller-hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2", "pytest", "tkinter"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=METADATA["executable_name"],
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(ROOT / METADATA["icon"]),
    version=str(WINDOWS_DIR / "version_info.txt"),
    uac_admin=False,
    uac_uiaccess=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=METADATA["executable_name"],
)
