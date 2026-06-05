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
    "test_thumbnail_cache_pixmap.py",
    "test_line_periodicity_plot.py",
    "test_angle_update_and_dock_restore.py",
    "test_fft_phase_gui.py",
    "test_fft_selection_gui.py",
    "test_mains_pickup_gui.py",
    "test_roi_resize_handles_canvas.py",
    "test_worker_signals_lifetime.py",
}

MIXED_QT_FIXTURE_MODULES = {
    "test_feature_lattice.py",
    "test_pair_correlation.py",
    "test_fft_viewer_utils.py",
    "test_definitions_dialog.py",
    "test_lattice_grid.py",
}

# Tests that construct a QApplication inline rather than via a ``qapp`` fixture,
# so the fixturename-based gating in MIXED_QT_FIXTURE_MODULES cannot see them.
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
    if not ok:
        marker = pytest.mark.skip(
            reason=f"Qt QApplication preflight failed in subprocess ({reason})"
        )
        for item in gui_items:
            item.add_marker(marker)
        return

    # Qt works here: run each GUI test in its own forked subprocess. Offscreen
    # Qt is a single process-wide QApplication with a global QThreadPool and a
    # PySide wrapper cache keyed by C++ pointer address; objects leaked by one
    # test get their address (and stale Python wrapper) reused by a later test,
    # which surfaces as an intermittent SIGSEGV or a bogus AttributeError on a
    # recycled wrapper. A fresh address space per test removes the reuse entirely
    # and contains any residual crash to a single reported test instead of
    # aborting the whole run. Requires pytest-forked; without it the marker is a
    # harmless no-op and the tests run in-process (the prior behaviour).
    if config.pluginmanager.hasplugin("forked"):
        for item in gui_items:
            item.add_marker(pytest.mark.forked)


@pytest.fixture(autouse=True)
def _drain_qt_between_tests():
    """Retire Qt background workers and deferred deletions after each test.

    The GUI tests start real ``QThreadPool`` workers (e.g. ``ViewerLoader``) and
    rely on ``deleteLater()`` for widget cleanup, but they never run a Qt event
    loop, so neither finished-worker signals nor deferred deletions are flushed
    at the test boundary. They then leak into the *next* test and are processed
    at a nondeterministic moment — delivering a queued ``loaded`` signal to, or
    running ``~QObject`` on, a half-torn-down dialog. That is the intermittent
    offscreen-Qt SIGSEGV (it migrates between tests and passes on re-run).

    Draining here makes each GUI test start from a clean slate: wait for the
    global pool to finish (so no worker thread can still emit into a widget the
    next test tears down), deliver the now-posted cross-thread signals to their
    still-live receivers, then run the deferred deletions and settle.

    Costs nothing when Qt was never imported (pure non-GUI sessions) or when no
    ``QApplication`` exists yet.
    """
    yield

    qtwidgets = sys.modules.get("PySide6.QtWidgets")
    if qtwidgets is None:
        return
    app = qtwidgets.QApplication.instance()
    if app is None:
        return

    from PySide6.QtCore import QEvent, QThreadPool

    # 1. Let in-flight workers finish so their loaded()/failed() events are all
    #    posted to the main thread before we flush. Returns immediately when the
    #    pool is idle; the timeout is a safety bound that should never be hit.
    QThreadPool.globalInstance().waitForDone(5000)
    # 2. Deliver queued cross-thread signals to receivers that are still alive
    #    (token-guarded slots drop stale content harmlessly).
    app.processEvents()
    # 3. Now actually destroy everything deleteLater()'d; Qt drops each object's
    #    pending posted events as it runs the destructor.
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    # 4. Settle anything the deletions themselves posted.
    app.processEvents()


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
