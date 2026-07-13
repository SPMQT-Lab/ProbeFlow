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
        '"cv2"',
        '"gwyfile.objects"',
        '"matplotlib.backends.backend_pdf"',
        'copy_metadata("gwyfile")',
        '"sklearn.cluster"',
        'console=False',
        'argv_emulation=False',
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
