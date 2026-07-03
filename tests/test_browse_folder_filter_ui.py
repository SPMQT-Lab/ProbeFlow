from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from probeflow.core.browse_filters import FolderFilterState


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
    return QApplication([])


def _scan_entry(name: str, width_nm: float, height_nm: float, bias_mv: float = 0.0):
    from probeflow.gui.models import SxmFile

    return SxmFile(
        path=Path(f"/tmp/{name}.sxm"),
        stem=name,
        Nx=64,
        Ny=64,
        scan_nm=width_nm,
        scan_width_nm=width_nm,
        scan_height_nm=height_nm,
        completion_pct=(100.0 * width_nm * height_nm) / (max(width_nm, height_nm) ** 2),
        bias_mv=bias_mv,
    )


def test_filter_tray_expands_and_emits_state_immediately(qapp):
    from probeflow.gui.browse.panels import BrowseToolPanel
    from probeflow.gui import THEMES

    panel = BrowseToolPanel(THEMES["dark"], {})
    seen = []
    panel.folder_filter_changed.connect(lambda state: seen.append(state))

    assert panel._folder_filter_box.isHidden() is True
    panel._toggle_folder_filters()
    assert panel._folder_filter_box.isHidden() is False

    panel._size_filter_btn.click()
    assert seen
    assert isinstance(seen[-1], FolderFilterState)
    assert seen[-1].size_enabled is True


def test_grid_metadata_filter_hides_non_matching_scans_and_clears_selection(qapp):
    from probeflow.gui import THEMES
    from probeflow.gui.browse import ThumbnailGrid

    grid = ThumbnailGrid(THEMES["dark"])
    grid._pool = SimpleNamespace(
        start=lambda loader, priority=0: None,
        activeThreadCount=lambda: 0,
        maxThreadCount=lambda: 8,
    )
    keep = _scan_entry("keep", 40.0, 40.0)
    hide = _scan_entry("hide", 120.0, 120.0)
    grid.load([keep, hide], "/tmp")
    qapp.processEvents()
    qapp.processEvents()

    grid._on_card_click(hide, False)
    assert grid.get_primary_entry() == hide

    grid.set_folder_filter_state(
        FolderFilterState(size_enabled=True, max_width_nm=100.0, max_height_nm=100.0)
    )

    assert grid.get_primary_entry() is None
    assert grid.get_visible_scan_entries() == [keep]


def test_grid_shows_filtering_status_during_async_folder_filter(qapp):
    from probeflow.gui import THEMES
    from probeflow.gui.browse import ThumbnailGrid
    from probeflow.gui.models import FolderEntry

    started = []
    launches = []
    grid = ThumbnailGrid(THEMES["dark"])
    grid.folder_filter_started.connect(lambda name: started.append(name))
    grid._pool = SimpleNamespace(
        start=lambda loader, priority=0: launches.append(loader),
        activeThreadCount=lambda: 0,
        maxThreadCount=lambda: 8,
    )
    grid.load([FolderEntry(path=Path("/tmp/sub"), n_scans=1)], "/tmp")
    grid.set_folder_filter_state(
        FolderFilterState(size_enabled=True, max_width_nm=100.0, max_height_nm=100.0)
    )

    assert launches
    assert started == ["tmp"]
    assert "Filtering tmp..." == grid._path_lbl.text()


def test_filtered_export_worker_copies_only_available_files(tmp_path, qapp):
    from probeflow.gui.workers import FilteredFolderExportWorker

    src_a = tmp_path / "a.sxm"
    src_b = tmp_path / "b.sxm"
    src_a.write_text("a", encoding="utf-8")
    src_b.write_text("b", encoding="utf-8")
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "b.sxm").write_text("existing", encoding="utf-8")

    got = []
    worker = FilteredFolderExportWorker([src_a, src_b], dest)
    worker.signals.finished.connect(lambda result: got.append(result))
    worker.run()

    assert got
    result = got[0]
    assert result.copied == 1
    assert result.collisions == 1
    assert (dest / "a.sxm").read_text(encoding="utf-8") == "a"
    assert (dest / "b.sxm").read_text(encoding="utf-8") == "existing"


def test_app_export_uses_global_thread_pool(monkeypatch, tmp_path, qapp):
    from probeflow.gui.app import ProbeFlowWindow

    entry = SimpleNamespace(path=tmp_path / "scan.sxm")
    started = {}

    class _Signals:
        def __init__(self):
            self.finished = SimpleNamespace(connect=lambda fn: None)
            self.failed = SimpleNamespace(connect=lambda fn: None)

    class _Worker:
        def __init__(self, paths, destination):
            started["paths"] = list(paths)
            started["destination"] = destination
            self.signals = _Signals()

    class _Pool:
        def start(self, worker):
            started["worker"] = worker

    fake = SimpleNamespace(
        _grid=SimpleNamespace(
            current_dir=lambda: tmp_path,
            get_visible_scan_entries=lambda: [entry],
        ),
        _status_bar=SimpleNamespace(showMessage=lambda msg: started.setdefault("msg", msg)),
        _on_export_filtered_finished=lambda result: None,
        _on_export_filtered_failed=lambda message: None,
    )

    monkeypatch.setattr("probeflow.gui.app.FilteredFolderExportWorker", _Worker)
    monkeypatch.setattr(
        "probeflow.gui.app.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(tmp_path / "out"),
    )
    monkeypatch.setattr(
        "probeflow.gui.app.QThreadPool.globalInstance",
        lambda: _Pool(),
    )

    ProbeFlowWindow._on_export_filtered_folder(fake)

    assert started["paths"] == [entry.path]
    assert started["destination"] == str(tmp_path / "out")
    assert "worker" in started
