"""Contracts for the standalone Windows x64 package and installer."""

from __future__ import annotations

from pathlib import Path
import struct
import tomllib

from scripts.validate_windows_app import (
    IMAGE_FILE_MACHINE_AMD64,
    IMAGE_FILE_MACHINE_I386,
    _machine_is_allowed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_DIR = REPO_ROOT / "packaging" / "windows"


def _metadata() -> dict:
    return tomllib.loads(
        (WINDOWS_DIR / "app_metadata.toml").read_text(encoding="utf-8")
    )["application"]


def test_windows_release_identity_and_target_are_explicit():
    metadata = _metadata()

    assert metadata["name"] == metadata["executable_name"] == "ProbeFlow"
    assert metadata["publisher"] == "SPMQT-Lab"
    assert metadata["version_source"] == "probeflow.__version__"
    assert metadata["product_version"] == "1.0.0 RC 1"
    assert metadata["file_version"] == [1, 0, 0, 1]
    assert metadata["minimum_windows_release"] == "Windows 10 version 1809"
    assert metadata["primary_architecture"] == "x86_64"
    assert metadata["installer_scope"] == "per-user"


def test_generated_windows_icon_is_a_multi_image_ico():
    data = (WINDOWS_DIR / "ProbeFlow.ico").read_bytes()
    reserved, image_type, count = struct.unpack("<HHH", data[:6])

    assert reserved == 0
    assert image_type == 1
    assert count >= 6


def test_windows_dependency_and_build_toolchain_are_pinned():
    constraints = (WINDOWS_DIR / "constraints-x64.txt").read_text(encoding="utf-8")
    requirements = (WINDOWS_DIR / "requirements-build.txt").read_text(encoding="utf-8")

    for dependency in (
        "numpy==2.1.3",
        "PySide6==6.11.0",
        "opencv-python==4.13.0.92",
        "scikit-learn==1.6.1",
        "gwyfile==0.3.0",
        "pefile==2024.8.26",
        "pywin32-ctypes==0.2.3",
    ):
        assert dependency in constraints
    assert "pyinstaller==6.21.0" in requirements
    assert "pyinstaller-hooks-contrib==2026.6" in requirements


def test_windows_pyinstaller_recipe_has_resources_metadata_and_exclusions():
    path = WINDOWS_DIR / "ProbeFlow.spec"
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    for expected in (
        "app_metadata.toml",
        'collect_submodules("probeflow.gui")',
        '"probeflow/assets"',
        '"probeflow/data/file_cushions"',
        '"THIRD_PARTY_NOTICES.md"',
        '"QT_LGPL_COMPLIANCE.md"',
        '"THIRD_PARTY_LICENSES"',
        'ROOT / "packaging" / "pyinstaller-hooks"',
        '"matplotlib.backends.backend_pdf"',
        'console=False',
        'uac_admin=False',
        'version=str(WINDOWS_DIR / "version_info.txt")',
    ):
        assert expected in source


def test_windows_builder_creates_clean_bundle_and_runs_smoke_test():
    source = (REPO_ROOT / "scripts" / "build_windows_app.ps1").read_text(
        encoding="utf-8"
    )

    for expected in (
        "3, 13, 14",
        "constraints-x64.txt",
        "runtime_licenses.toml",
        "collect_python_licenses.py",
        "prepare_qt_release_materials.py",
        "CPython-3.13.14",
        "ProbeFlow.spec",
        "validate_windows_app.py",
        '"--smoke-test"',
        'QT_QPA_PLATFORM = "offscreen"',
    ):
        assert expected in source


def test_windows_validator_requires_x64_and_rejects_unused_qt():
    path = REPO_ROOT / "scripts" / "validate_windows_app.py"
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")

    for expected in (
        "IMAGE_FILE_MACHINE_AMD64",
        "Qt6Core.dll",
        "Qt6Widgets.dll",
        "python313.dll",
        "qpdf.dll",
        "qt6qml.dll",
        "qt6quick.dll",
        "qtvirtualkeyboard",
        "ProductVersion",
    ):
        assert expected in source
    assert 'internal / "PySide6" / "Qt6Core.dll"' in source
    assert 'internal / "PySide6" / "Qt6Widgets.dll"' in source
    assert '"PySide6" / "Qt" / "bin"' not in source


def test_windows_validator_only_exempts_the_nsis_x86_uninstaller():
    assert _machine_is_allowed(Path("ProbeFlow.exe"), IMAGE_FILE_MACHINE_AMD64)
    assert _machine_is_allowed(
        Path("Uninstall ProbeFlow.exe"), IMAGE_FILE_MACHINE_I386
    )
    assert not _machine_is_allowed(Path("ProbeFlow.exe"), IMAGE_FILE_MACHINE_I386)
    assert not _machine_is_allowed(Path("_internal/foreign.dll"), IMAGE_FILE_MACHINE_I386)


def test_nsis_installer_is_per_user_and_has_clean_uninstall():
    source = (WINDOWS_DIR / "ProbeFlow.nsi").read_text(encoding="utf-8")

    for expected in (
        "RequestExecutionLevel user",
        '$LOCALAPPDATA\\Programs\\ProbeFlow',
        "SetShellVarContext current",
        'CreateShortcut "$SMPROGRAMS\\ProbeFlow.lnk"',
        "CurrentVersion\\Uninstall\\ProbeFlow",
        'WriteUninstaller "$INSTDIR\\Uninstall ProbeFlow.exe"',
        '"UninstallString" "$\\\"$INSTDIR\\Uninstall ProbeFlow.exe$\\\""',
        'RMDir /r "$INSTDIR"',
        "${RunningX64}",
    ):
        assert expected in source
    assert "RequestExecutionLevel admin" not in source


def test_installer_builder_installs_smoke_tests_and_uninstalls():
    source = (REPO_ROOT / "scripts" / "build_windows_installer.ps1").read_text(
        encoding="utf-8"
    )

    for expected in (
        "ProbeFlow-$ArtifactVersion-Windows-x64-Setup.exe",
        "ProbeFlow-$ArtifactVersion-Windows-x64-portable.zip",
        "makensis.exe",
        '"/WX"',
        "Get-FileHash -Algorithm SHA256",
        '"/S", "/D=$InstallRoot"',
        "validate_windows_app.py",
        '"--smoke-test"',
        "Uninstall ProbeFlow.exe",
    ):
        assert expected in source


def test_windows_workflow_runs_tests_builds_and_uploads_verified_artifacts():
    source = (REPO_ROOT / ".github" / "workflows" / "build-windows.yml").read_text(
        encoding="utf-8"
    )

    for expected in (
        "workflow_dispatch",
        "windows-2022",
        'python-version: "3.13.14"',
        "python -m pytest -q",
        "nsis --version=3.12.0",
        "build_windows_app.ps1",
        "build_windows_installer.ps1",
        "actions/upload-artifact@v4",
        "ProbeFlow-1.0.0-rc1-Windows-x64-Setup.exe",
    ):
        assert expected in source
