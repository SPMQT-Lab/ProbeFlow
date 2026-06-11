"""Adversarial seam tests for browse workers and the thumbnail grid.

Review focus #1 (async browse loading / Qt worker lifetime): every worker must
emit *something* for every render it owes — a render exception that escapes
``work()`` leaves cards/panels silently stuck on their placeholder forever
(the QThreadPool swallows the exception). Also covers the timer-sliced card
build surviving mid-build appearance changes.

Tests marked ``xfail(strict=True)`` document confirmed bugs from the
2026-06-11 adversarial review.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp(monkeypatch):
    monkeypatch.setenv("PROBEFLOW_DISABLE_BROWSE_CACHE", "1")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:
        pytest.skip(f"PySide6 unavailable: {exc}")
    app = QApplication.instance()
    if app is not None:
        return app
    try:
        return QApplication([])
    except Exception as exc:
        pytest.skip(f"QApplication unavailable: {exc}")


def _scan_entry(i: int = 0):
    from probeflow.gui.models import SxmFile
    return SxmFile(path=Path(f"/tmp/probeflow_fake_{i:04d}.sxm"),
                   stem=f"probeflow_fake_{i:04d}")


def _raise_render(**_kw):
    raise RuntimeError("render blew up")


# ── Worker emit-on-failure contracts ──────────────────────────────────────────

class TestWorkerFailureEmission:
    def test_thumbnail_loader_emits_on_render_exception(self, qapp, monkeypatch):
        """A render exception must produce a null-image emit (failure
        placeholder), not escape work() with zero emits — the QThreadPool
        swallows the exception and the card would stay loading forever."""
        import probeflow.gui.workers as workers

        monkeypatch.setattr(workers, "render_scan_image", _raise_render)
        monkeypatch.setattr(workers, "load_thumbnail_plane",
                            lambda p, ch: (np.ones((4, 4)), ["z"]))
        got = []
        loader = workers.ThumbnailLoader(_scan_entry(), "gray", object(), 8, 8)
        loader.signals.loaded.connect(lambda *a: got.append(a))
        loader.work()  # must not raise
        assert len(got) == 1, "no loaded signal for a failed render"
        assert got[0][1].isNull(), "failure must be a null QImage placeholder"

    def test_spec_thumbnail_loader_emits_on_render_exception(self, qapp, monkeypatch):
        """Contrast case: the spec loader already guards its render call and
        emits a null image — the contract every loader should meet."""
        import probeflow.gui.workers as workers
        from probeflow.gui.models import VertFile

        monkeypatch.setattr(workers, "render_spec_thumbnail", _raise_render)
        got = []
        entry = VertFile(path=Path("/tmp/probeflow_fake.VERT"), stem="fake")
        loader = workers.SpecThumbnailLoader(entry, object(), 8, 8)
        loader.signals.loaded.connect(lambda *a: got.append(a))
        loader.work()
        assert len(got) == 1
        assert got[0][1].isNull()

    def test_folder_thumbnail_loader_emits_on_render_exception(self, qapp, monkeypatch):
        """One bad sample path must yield a None slot in the emitted list,
        not kill the worker before the folder card is updated."""
        import probeflow.gui.workers as workers

        monkeypatch.setattr(workers, "render_scan_image", _raise_render)
        monkeypatch.setattr(workers, "load_thumbnail_plane",
                            lambda p, ch: (np.ones((4, 4)), ["z"]))
        got = []
        loader = workers.FolderThumbnailLoader(
            "folder-key", [Path("/tmp/a.sxm"), Path("/tmp/b.sxm")],
            "gray", object(), 8, 8)
        loader.signals.loaded.connect(lambda *a: got.append(a))
        loader.work()
        assert len(got) == 1, "folder card never updated after render failure"
        assert got[0][1] == [None, None]

    def test_channel_preview_loader_load_failure_emits_failed(self, qapp, monkeypatch):
        import probeflow.core.scan_loader as scan_loader
        import probeflow.gui.workers as workers

        def boom_load(path):
            raise OSError("network path gone")

        monkeypatch.setattr(scan_loader, "load_scan", boom_load)
        failed = []
        loader = workers.ChannelPreviewLoader(_scan_entry(), "gray", object(), 8, 8)
        loader.signals.failed.connect(lambda *a: failed.append(a))
        loader.work()
        assert len(failed) == 1

    def test_channel_preview_loader_survives_one_bad_plane(self, qapp, monkeypatch):
        """Each plane render is guarded independently: one unusual plane
        emits a null preview for its slot and the remaining planes still
        arrive (was: the exception killed the worker mid-stream and the
        panel stayed partially populated with no failed signal)."""
        import probeflow.core.scan_loader as scan_loader
        import probeflow.gui.workers as workers

        scan = SimpleNamespace(
            n_planes=3,
            plane_names=["a", "b", "c"],
            header={},
            planes=[np.ones((4, 4))] * 3,
        )
        monkeypatch.setattr(scan_loader, "load_scan", lambda p: scan)

        calls = {"n": 0}

        def flaky_render(**_kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("plane 2 render blew up")
            return None  # graceful per-plane failure for the others

        monkeypatch.setattr(workers, "render_scan_image", flaky_render)

        meta, loaded, failed = [], [], []
        loader = workers.ChannelPreviewLoader(_scan_entry(), "gray", object(), 8, 8)
        loader.signals.meta_ready.connect(lambda *a: meta.append(a))
        loader.signals.loaded.connect(lambda *a: loaded.append(a))
        loader.signals.failed.connect(lambda *a: failed.append(a))
        loader.work()  # must not raise

        assert len(meta) == 1
        # Every plane slot must be resolved with a loaded emit (null QImage
        # for the broken one); no terminal failure for a per-plane problem.
        assert len(loaded) == 3, (
            f"panel left partially populated: {len(loaded)} of 3 planes "
            f"delivered, failed={len(failed)}"
        )
        assert not failed
        assert all(img.isNull() for _idx, img, _tok in loaded)


# ── Thumbnail grid: timer-sliced card build ───────────────────────────────────

def _make_grid(n_entries: int):
    from probeflow.gui import THEMES
    from probeflow.gui.browse import ThumbnailGrid

    grid = ThumbnailGrid(THEMES["dark"])
    # No real workers: thumbnails are irrelevant to card construction.
    grid._pool = SimpleNamespace(
        start=lambda loader, priority=0: None,
        activeThreadCount=lambda: 0,
        maxThreadCount=lambda: 8,
    )
    entries = [_scan_entry(i) for i in range(n_entries)]
    return grid, entries


def _drain_events(app, n: int = 80) -> None:
    for _ in range(n):
        app.processEvents()


class TestTimerSlicedCardBuild:
    def test_card_build_completes_undisturbed(self, qapp):
        n = 300  # > _CARD_BUILD_BATCH so the timer-sliced path is exercised
        grid, entries = _make_grid(n)
        grid.load(entries, "/tmp")
        assert len(grid._cards) < n, "expected timer-sliced build"
        _drain_events(qapp)
        assert len(grid._cards) == n
        assert not grid._card_build_queue

    def test_appearance_change_mid_build_still_builds_all_cards(self, qapp):
        """Colormap/channel/align-rows changes replace _load_token to drop
        in-flight renders; that must not cancel the timer-sliced card build
        (2026-06-11 review: the shared token orphaned the tail 180 of 300
        cards), and entries whose cards are not built yet must stay in the
        pending queue so they get thumbnails with the new appearance."""
        from probeflow.gui.models import browse_entry_key

        grid, entries = _make_grid(300)
        grid.load(entries, "/tmp")
        assert grid._card_build_queue, "expected the build to still be draining"
        tail_keys = {browse_entry_key(e) for e in grid._card_build_queue}

        grid.set_thumbnail_colormap("viridis")  # user changes appearance now

        # Tail entries (cards not built yet) must remain pending so their
        # thumbnails render once their cards exist.
        assert tail_keys <= set(grid._thumbnail_pending), (
            "re-render dropped not-yet-built entries from the pending queue"
        )

        _drain_events(qapp)
        assert len(grid._cards) == len(entries), (
            f"only {len(grid._cards)} of {len(entries)} cards built; "
            f"{len(grid._card_build_queue)} entries orphaned in the queue"
        )
        assert not grid._card_build_queue

    def test_appearance_change_mid_build_invalidates_inflight_renders(self, qapp):
        """The reason _rerender_scan_thumbnails replaces _load_token at all:
        a thumbnail rendered with the old appearance that lands after the
        change must be dropped, while results carrying the new token land."""
        from PySide6.QtGui import QImage, QPixmap

        from probeflow.gui.models import browse_entry_key

        grid, entries = _make_grid(10)
        grid.load(entries, "/tmp")
        _drain_events(qapp)
        old_token = grid._load_token

        grid.set_thumbnail_colormap("viridis")

        key = browse_entry_key(entries[0])
        img = QImage(4, 4, QImage.Format_RGB32)
        img.fill(0xFF0000)
        grid._on_thumb(key, img, old_token)  # stale delivery
        card = grid._cards[key]
        assert card.img_lbl.pixmap() is None or card.img_lbl.pixmap().isNull(), (
            "stale (old-appearance) thumbnail was applied after the change"
        )
        grid._on_thumb(key, img, grid._load_token)  # fresh delivery
        assert isinstance(card.img_lbl.pixmap(), QPixmap)
        assert not card.img_lbl.pixmap().isNull()

    def test_filter_change_mid_build_still_builds_all_cards(self, qapp):
        """apply_filter mid-build must not lose queued cards, and the grid
        index must stay consistent so late cards append after early ones."""
        grid, entries = _make_grid(300)
        grid.load(entries, "/tmp")
        assert grid._card_build_queue

        grid.apply_filter("images")
        _drain_events(qapp)

        assert len(grid._cards) == len(entries)
        assert not grid._card_build_queue
        assert grid._next_grid_index == len(entries), (
            "grid placement index out of sync after mid-build relayout"
        )

    def test_navigation_mid_build_discards_stale_slices(self, qapp):
        """Loading a new entry list mid-build must drop the old queue and
        build exactly the new folder's cards (no duplicates, no leftovers)."""
        grid, entries = _make_grid(300)
        grid.load(entries, "/tmp")
        assert grid._card_build_queue

        new_entries = [_scan_entry(i) for i in range(1000, 1010)]
        grid.load(new_entries, "/tmp/other")
        _drain_events(qapp)

        from probeflow.gui.models import browse_entry_key
        assert set(grid._cards) == {browse_entry_key(e) for e in new_entries}
        assert not grid._card_build_queue


# ── Navigation state consistency ──────────────────────────────────────────────

class TestNavigationStateConsistency:
    """The grid's navigation state must always describe what is on screen.

    _current_dir is set optimistically at navigation intent (breadcrumb
    feedback) while the old grid stays interactive until the off-thread index
    lands — these tests pin the seams of that window (2026-06-11 review,
    focus #1: old-grid interaction while a new folder is indexing).
    """

    def test_index_failure_restores_displayed_folder(self, qapp):
        """When the index for a navigation target fails, the grid keeps
        showing the old folder — current_dir must be restored to match, or
        refresh() retargets the failed path and Back history corrupts."""
        grid, entries = _make_grid(3)
        root = Path("/tmp/probeflow_navtest")
        grid.load(entries, str(root))

        grid.navigate_to(root / "sub")
        assert grid.current_dir() == root / "sub"  # optimistic intent

        grid._on_folder_index_failed(root / "sub", "boom", grid._nav_token)

        assert grid.current_dir() == root, (
            "current_dir still points at the failed folder while the old "
            "folder's entries are displayed"
        )
        assert grid._history == [], (
            "failed navigation left its push on the Back history"
        )
        # The displayed entries were never replaced.
        assert grid.get_entries() == entries
        # Next navigation pushes the folder the user was actually in.
        grid.navigate_to(root / "other")
        assert grid._history == [root]

    def test_stale_index_failure_is_ignored(self, qapp):
        """A failure arriving for a superseded navigation (token mismatch)
        must not touch current navigation state."""
        grid, entries = _make_grid(3)
        root = Path("/tmp/probeflow_navtest")
        grid.load(entries, str(root))

        grid.navigate_to(root / "a")
        stale_token = grid._nav_token
        grid.navigate_to(root / "b")
        grid._on_folder_index_failed(root / "a", "boom", stale_token)

        assert grid.current_dir() == root / "b"

    def test_render_announces_selection_reset(self, qapp):
        """Re-rendering (navigation landing / refresh) clears the selection;
        listeners driving selection-dependent UI must be told, not left with
        a stale count."""
        grid, entries = _make_grid(4)
        grid.load(entries, "/tmp/probeflow_navtest")
        grid._on_card_click(entries[0], ctrl=False)
        assert grid.get_selected()

        seen: list[int] = []
        grid.selection_changed.connect(seen.append)
        grid.load(list(entries), "/tmp/probeflow_navtest/other")

        assert seen and seen[-1] == 0
        assert grid.get_selected() == set()
        assert grid.get_primary_entry() is None

    def test_old_grid_selection_still_works_while_indexing(self, qapp):
        """During the indexing window the old grid stays interactive by
        design: clicking an old card must still select it and report a
        primary entry consistent with the displayed entries."""
        grid, entries = _make_grid(4)
        root = Path("/tmp/probeflow_navtest")
        grid.load(entries, str(root))

        grid.navigate_to(root / "slow")  # index never lands (stubbed pool)
        grid._on_card_click(entries[1], ctrl=False)

        assert grid.get_primary_entry() is entries[1]
        assert grid.get_entries() == entries


# ── Shutdown drain ────────────────────────────────────────────────────────────

class TestShutdownDrain:
    """Worker pools must be drained before teardown no matter which window
    quits the app (2026-06-11 review, focus #1 Test 1 item 9)."""

    def test_drain_waits_for_inflight_global_pool_workers(self, qapp):
        from PySide6.QtCore import QRunnable, QThreadPool

        from probeflow.gui.app import ProbeFlowWindow

        done = []

        class Slow(QRunnable):
            def run(self):
                import time
                time.sleep(0.2)
                done.append(True)

        QThreadPool.globalInstance().start(Slow())
        win = ProbeFlowWindow.__new__(ProbeFlowWindow)  # no heavy __init__
        ProbeFlowWindow._drain_worker_pools(win, timeout_ms=5000)

        assert done == [True], "drain returned before in-flight worker finished"
        assert QThreadPool.globalInstance().activeThreadCount() == 0

    def test_drain_tolerates_stuck_pool_and_missing_optional_pools(self, qapp, monkeypatch):
        """A pool that never finishes must produce a bounded wait and a log
        warning — not a hang or an exception; absent optional pools
        (_features_preview_pool, _fc_window) are skipped."""
        from PySide6.QtCore import QThreadPool

        from probeflow.gui.app import ProbeFlowWindow

        calls = []

        class StuckPool:
            def clear(self):
                calls.append("clear")

            def waitForDone(self, _ms):
                calls.append("wait")
                return False  # simulated stuck worker

        monkeypatch.setattr(QThreadPool, "globalInstance",
                            staticmethod(lambda: StuckPool()))
        win = ProbeFlowWindow.__new__(ProbeFlowWindow)
        ProbeFlowWindow._drain_worker_pools(win, timeout_ms=10)  # must not raise

        assert calls == ["clear", "wait"]

    def test_quit_drain_fires_on_about_to_quit(self, qapp):
        """The drain must also be connected to QApplication.aboutToQuit:
        closing a modeless viewer last (after the main window already
        closed) quits the app without running the main window's closeEvent."""
        from PySide6.QtWidgets import QMainWindow

        from probeflow.gui.app import ProbeFlowWindow

        win = ProbeFlowWindow.__new__(ProbeFlowWindow)
        QMainWindow.__init__(win)  # minimal QObject init, no ProbeFlow UI
        drained = []
        win._drain_worker_pools = lambda *a, **k: drained.append(True)
        try:
            ProbeFlowWindow._install_quit_drain(win)
            qapp.aboutToQuit.emit()
            assert drained == [True]
        finally:
            try:
                qapp.aboutToQuit.disconnect(win._drain_worker_pools)
            except (RuntimeError, TypeError):
                pass
            win.deleteLater()
