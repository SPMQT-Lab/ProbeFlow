# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the native Apple Silicon ProbeFlow application."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib

from PyInstaller.utils.hooks import collect_submodules, copy_metadata


ROOT = Path(SPECPATH).resolve().parents[1]
MACOS_DIR = ROOT / "packaging" / "macos"
METADATA = tomllib.loads(
    (MACOS_DIR / "app_metadata.toml").read_text(encoding="utf-8")
)["application"]


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
if SOURCE_VERSION.split("rc", 1)[0] != METADATA["bundle_short_version"]:
    raise ValueError(
        "The package version and macOS bundle_short_version describe different releases"
    )

# ProbeFlow's GUI compatibility modules use importlib to preserve public import
# paths during the ongoing refactor, so static analysis cannot see every GUI
# module. Application, numerical and file-I/O imports reached by those modules
# are then followed normally by PyInstaller and its package hooks.
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
]
datas += copy_metadata("gwyfile")

a = Analysis(
    [str(ROOT / "probeflow" / "gui" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
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
    argv_emulation=False,
    target_arch=METADATA["primary_architecture"],
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=METADATA["executable_name"],
)

app = BUNDLE(
    coll,
    name=f"{METADATA['name']}.app",
    icon=str(ROOT / METADATA["icon"]),
    bundle_identifier=METADATA["bundle_identifier"],
    version=METADATA["bundle_short_version"],
    info_plist={
        "CFBundleName": METADATA["name"],
        "CFBundleDisplayName": METADATA["name"],
        "CFBundleShortVersionString": METADATA["bundle_short_version"],
        "CFBundleVersion": str(METADATA["bundle_build_number"]),
        "NSHumanReadableCopyright": METADATA["copyright"],
        "LSApplicationCategoryType": METADATA["category_type"],
        "LSMinimumSystemVersion": METADATA["minimum_macos_version"],
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": True,
        "NSAppleScriptEnabled": False,
    },
)
