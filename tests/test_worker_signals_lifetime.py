"""Regression: pooled-worker signals must be owned by a main-thread object.

QThreadPool auto-deletes a finished QRunnable on the *worker* thread. If the
runnable is the sole owner of a parentless ``*Signals`` QObject (which carries
cross-thread signal/slot connections), that QObject is destroyed off the main
thread, corrupting Qt internals — observed as a hard SIGSEGV inside the
app-level tooltip event filter when opening Feature Finder / Feature Counting.

The fix parents the worker-owned signals to the QApplication (created on the
main thread) so Shiboken never C++-destroys it from the worker thread; the
worker ``deleteLater()``s it to avoid accumulation. These tests lock the
invariant for the workers in the Feature-Counting flow.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest


@pytest.fixture
def qapp():
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PySide6 unavailable: {exc}")
    return QApplication.instance() or QApplication([])


def test_features_worker_signals_parented_to_app(qapp):
    from probeflow.gui.features import _FeaturesWorker

    w = _FeaturesWorker(
        "particles", np.zeros((8, 8), dtype=float), 1e-9, 1e-9, 1e-9,
        {"threshold": "otsu"},
    )
    # Auto-created signals must be owned by the main-thread QApplication so the
    # worker's off-thread auto-delete can't destroy them.
    assert w.signals.parent() is qapp


def test_features_worker_keeps_caller_signals_unparented(qapp):
    """When the caller supplies signals, the worker must not reparent them
    (the caller owns their lifetime)."""
    from probeflow.gui.features import _FeaturesWorker, _FeaturesWorkerSignals

    sig = _FeaturesWorkerSignals()
    w = _FeaturesWorker(
        "particles", np.zeros((8, 8), dtype=float), 1e-9, 1e-9, 1e-9,
        {"threshold": "otsu"}, signals=sig,
    )
    assert w.signals is sig


def test_scan_load_worker_signals_parented_to_app(qapp):
    from probeflow.gui.app import _ScanLoadWorker

    w = _ScanLoadWorker(lambda entry, plane: ("ok",), None, 0)
    assert w.signals.parent() is qapp


def test_pooled_worker_base_parents_signals_and_runs_work(qapp):
    """The shared _PooledWorker base parents its signals to the app and runs
    work() (not run()). Covers the migrated browse/viewer/conversion workers."""
    from PySide6.QtCore import QObject, Signal
    from probeflow.gui.workers import _PooledWorker

    class _Sig(QObject):
        done = Signal()

    ran = []

    class _W(_PooledWorker):
        def __init__(self):
            super().__init__(_Sig())

        def work(self):
            ran.append(True)

    w = _W()
    assert w.signals.parent() is qapp
    w.run()                      # base run() -> work() + deleteLater(signals)
    assert ran == [True]


def test_conversion_worker_signals_parented_to_app(qapp):
    from probeflow.gui.workers import ConversionWorker

    w = ConversionWorker("in", "out", False, False, 1.0, 99.0)
    assert w.signals.parent() is qapp


def test_scan_load_worker_runs_on_pool_without_crash(qapp):
    """Smoke: run a worker through a real QThreadPool and pump the event loop.
    Exercises the auto-delete-on-worker-thread teardown that the fix guards."""
    from PySide6.QtCore import QThreadPool
    from probeflow.gui.app import _ScanLoadWorker

    received = []
    w = _ScanLoadWorker(lambda entry, plane: ("payload",), None, 0)
    w.signals.finished.connect(lambda result, error: received.append((result, error)))
    pool = QThreadPool.globalInstance()
    pool.start(w)
    pool.waitForDone(5000)
    # Pump the event loop so queued finished + deleteLater are processed.
    for _ in range(5):
        qapp.processEvents()
    assert received and received[0][1] == ""
