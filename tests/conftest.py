"""Shared pytest fixtures."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "test_data" / "sample_input"
CUSHION_DIR = REPO_ROOT / "probeflow" / "data" / "file_cushions"

GUI_TEST_MODULES = {
    "test_gui_index_integration.py",
    "test_gui_processing_panel.py",
    "test_gui_features.py",
    "test_viewer_window_menu.py",
    "test_roi_click_selection_canvas.py",
    "test_viewer_region_levels.py",
    "test_viewer_region_contrast_integration.py",
}

MIXED_QT_FIXTURE_MODULES = {
    "test_feature_lattice.py",
    "test_pair_correlation.py",
}

MIXED_QT_TESTS = {
    ("test_lattice_grid.py", "test_export_png_creates_file"),
}


def _qt_application_preflight() -> tuple[bool, str]:
    """Probe QApplication creation in a subprocess so Qt aborts do not kill pytest."""
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    code = (
        "from PySide6.QtWidgets import QApplication; "
        "app = QApplication([]); "
        "app.quit()"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)

    if result.returncode == 0:
        return True, ""

    detail = (result.stderr or result.stdout or "").strip().splitlines()
    tail = detail[-1] if detail else "no diagnostic output"
    return False, f"exit code {result.returncode}: {tail}"


def pytest_collection_modifyitems(config, items):
    _ = config
    gui_items = []
    for item in items:
        module_name = Path(str(getattr(item, "path", getattr(item, "fspath", "")))).name
        if module_name in GUI_TEST_MODULES:
            gui_items.append(item)
        elif module_name in MIXED_QT_FIXTURE_MODULES and "qapp" in getattr(item, "fixturenames", ()):
            gui_items.append(item)
        elif (module_name, item.name) in MIXED_QT_TESTS:
            gui_items.append(item)
    if not gui_items:
        return

    ok, reason = _qt_application_preflight()
    if ok:
        return

    marker = pytest.mark.skip(
        reason=f"Qt QApplication preflight failed in subprocess ({reason})"
    )
    for item in gui_items:
        item.add_marker(marker)


@pytest.fixture
def sample_dat_files():
    files = sorted(SAMPLE_DIR.glob("*.dat"))
    assert files, f"No .dat files found in {SAMPLE_DIR}"
    return files


@pytest.fixture
def first_sample_dat(sample_dat_files):
    return sample_dat_files[0]


@pytest.fixture
def cushion_dir():
    return CUSHION_DIR
