"""Contracts for the standalone macOS build recipe."""

from __future__ import annotations

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
MACOS_DIR = REPO_ROOT / "packaging" / "macos"


def test_desktop_extra_contains_features_but_not_test_tools():
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = config["project"]["optional-dependencies"]

    assert set(extras["desktop"]) == {"opencv-python", "scikit-learn", "gwyfile"}
    assert extras["all"] == extras["desktop"]
    assert "pytest" not in extras["desktop"]


def test_windowed_entry_point_uses_the_freeze_aware_launcher():
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert config["project"]["gui-scripts"] == {
        "probeflow-gui": "probeflow.gui.launcher:main",
    }

    launcher = (REPO_ROOT / "probeflow" / "gui" / "launcher.py").read_text(
        encoding="utf-8"
    )
    assert '"--smoke-test"' in launcher
    for optional_module in ("cv2", "gwyfile.objects", "sklearn"):
        assert optional_module in launcher


def test_build_toolchain_is_pinned():
    requirements = (MACOS_DIR / "requirements-build.txt").read_text(encoding="utf-8")
    constraints = (MACOS_DIR / "constraints-arm64.txt").read_text(encoding="utf-8")

    assert "pyinstaller==6.21.0" in requirements
    assert "pyinstaller-hooks-contrib==2026.6" in requirements
    for dependency in (
        "numpy==2.1.3",
        "PySide6==6.11.0",
        "opencv-python==4.13.0.92",
        "scikit-learn==1.6.1",
        "gwyfile==0.3.0",
    ):
        assert dependency in constraints


def test_pyinstaller_recipe_tracks_metadata_and_required_resources():
    spec_path = MACOS_DIR / "ProbeFlow.spec"
    source = spec_path.read_text(encoding="utf-8")
    compile(source, str(spec_path), "exec")

    for expected in (
        "app_metadata.toml",
        'METADATA["version_source"]',
        'module_path / "__init__.py"',
        'collect_submodules("probeflow.gui")',
        '"probeflow/assets"',
        '"probeflow/data/file_cushions"',
        '"THIRD_PARTY_NOTICES.md"',
        '"QT_LGPL_COMPLIANCE.md"',
        '"THIRD_PARTY_LICENSES"',
        'ROOT / "packaging" / "pyinstaller-hooks"',
        '"cv2"',
        '"gwyfile.objects"',
        '"matplotlib.backends.backend_pdf"',
        'copy_metadata("gwyfile")',
        '"sklearn.cluster"',
        'console=False',
        'argv_emulation=False',
        'os.environ.get("PROBEFLOW_CODESIGN_IDENTITY")',
    ):
        assert expected in source


def test_build_script_recreates_an_arm64_python_313_environment():
    source = (REPO_ROOT / "scripts" / "build_macos_app.sh").read_text(encoding="utf-8")

    assert 'PYTHON_VERSION="3.13.14"' in source
    assert "www.python.org/ftp/python" in source
    assert "8e58affb218c155a1dfdc27b291f817129669f8760e7a297adb2e4439ba5d2e8" in source
    assert "pkgutil --expand-full" in source
    assert "relocate_python_framework.py" in source
    assert "validate_macos_app.py" in source
    assert "collect_python_licenses.py" in source
    assert "prepare_qt_release_materials.py" in source
    assert "CPython-${PYTHON_VERSION}/LICENSE.txt" in source
    assert '"${APP}/Contents/MacOS/ProbeFlow" --smoke-test' in source
    assert "platform.machine()" in source
    assert '"${PYTHON}" -m venv --clear' in source
    assert '"${ROOT}[desktop]"' in source
    assert "constraints-arm64.txt" in source


def test_dmg_builder_creates_and_verifies_drag_install_artifact():
    builder = (REPO_ROOT / "scripts" / "build_macos_dmg.sh").read_text(
        encoding="utf-8"
    )
    validator = (REPO_ROOT / "scripts" / "validate_macos_dmg.sh").read_text(
        encoding="utf-8"
    )

    for expected in (
        "build_macos_app.sh",
        "ProbeFlow-${ARTIFACT_VERSION}-macOS-arm64.dmg",
        "ln -s /Applications",
        "hdiutil create",
        "-format UDZO",
        "validate_macos_dmg.sh",
        "shasum -a 256",
        "PROBEFLOW_CODESIGN_IDENTITY",
        "/usr/bin/codesign",
    ):
        assert expected in builder

    for expected in (
        "hdiutil verify",
        "hdiutil attach",
        "-readonly",
        "validate_macos_app.py",
        '"${MOUNTED_APP}/Contents/MacOS/ProbeFlow" --smoke-test',
        'readlink "${MOUNT_POINT}/Applications"',
    ):
        assert expected in validator


def test_release_excludes_unused_gpl_only_qt_plugin():
    hook = (
        REPO_ROOT / "packaging" / "pyinstaller-hooks" / "hook-PySide6.QtGui.py"
    ).read_text(encoding="utf-8")
    validator = (REPO_ROOT / "scripts" / "validate_macos_app.py").read_text(
        encoding="utf-8"
    )

    assert "pyside6_library_info.collect_module" in hook
    assert "libqtvirtualkeyboardplugin.dylib" in hook
    assert "libqpdf.dylib" in hook
    assert "QtVirtualKeyboard.framework" in validator
    assert "Nested binary has a different signing team" in validator
    assert "Nested binary lacks hardened runtime" in validator


def test_runtime_license_manifest_pins_corresponding_qt_sources():
    config = tomllib.loads(
        (REPO_ROOT / "packaging" / "runtime_licenses.toml").read_text(encoding="utf-8")
    )

    assert {item["component"] for item in config["qt_source_archives"]} == {
        "qtbase",
        "qtimageformats",
        "qtsvg",
        "qttranslations",
        "pyside-setup",
    }
    assert all(len(item["sha256"]) == 64 for item in config["qt_source_archives"])
    assert "PySide6" in config["licenses_from_source_archives"]
    assert "pyinstaller" in config["runtime_distributions"]
    assert "qtvirtualkeyboard" not in {
        item["component"] for item in config["qt_source_archives"]
    }


def test_notarization_script_keeps_credentials_out_of_arguments():
    source = (REPO_ROOT / "scripts" / "notarize_macos_dmg.sh").read_text(
        encoding="utf-8"
    )

    for expected in (
        "PROBEFLOW_CODESIGN_IDENTITY",
        "PROBEFLOW_NOTARY_PROFILE",
        "notarytool submit",
        "--keychain-profile",
        "stapler staple",
        "stapler validate",
        "spctl",
        "context:primary-signature",
    ):
        assert expected in source
    for secret_argument in ("--apple-id", "--password", "--issuer", "--key-id"):
        assert secret_argument not in source


def test_github_release_requires_explicit_unsigned_mode_and_release_checks():
    source = (REPO_ROOT / "scripts" / "publish_github_release.sh").read_text(
        encoding="utf-8"
    )

    for expected in (
        '"--unsigned"',
        "status --porcelain",
        "origin/main",
        "gh auth status",
        "stapler validate",
        "shasum -a 256 -c",
        "validate_macos_dmg.sh",
        "qt_source_archives",
        "QT_CORRESPONDING_SOURCE.txt",
        "gh release create",
        "--prerelease",
    ):
        assert expected in source
