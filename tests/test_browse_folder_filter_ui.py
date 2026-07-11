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


def _spec_entry(name: str):
    from probeflow.gui.models import VertFile

    return VertFile(path=Path(f"/tmp/{name}.VERT"), stem=name)


def _quiet_grid(qapp):
    from probeflow.gui import THEMES
    from probeflow.gui.browse import ThumbnailGrid

    grid = ThumbnailGrid(THEMES["dark"])
    grid._pool = SimpleNamespace(
        start=lambda loader, priority=0: None,
        activeThreadCount=lambda: 0,
        maxThreadCount=lambda: 8,
    )
    return grid


def test_panel_emits_state_for_bias_pick_and_hide_incomplete(qapp):
    from probeflow.gui import THEMES
    from probeflow.gui.browse.panels import BrowseToolPanel

    panel = BrowseToolPanel(THEMES["dark"], {})
    seen: list[FolderFilterState] = []
    panel.folder_filter_changed.connect(lambda state: seen.append(state))

    panel.set_bias_options([(-25.0, 3), (1000.0, 2)])
    assert [panel.bias_cb.itemText(i) for i in range(panel.bias_cb.count())] == [
        "All biases", "-25 mV (3)", "1000 mV (2)",
    ]

    panel.bias_cb.setCurrentIndex(1)
    assert seen and seen[-1].bias_value_mv == pytest.approx(-25.0)

    panel._hide_incomplete_cb.setChecked(True)
    assert seen[-1].hide_incomplete is True
    assert seen[-1].bias_value_mv == pytest.approx(-25.0)


def test_panel_bias_options_rebuild_preserves_selection(qapp):
    from probeflow.gui import THEMES
    from probeflow.gui.browse.panels import BrowseToolPanel

    panel = BrowseToolPanel(THEMES["dark"], {})
    panel.set_bias_options([(-25.0, 3), (1000.0, 2)])
    panel.bias_cb.setCurrentIndex(2)  # 1000 mV

    # Same bias still present after a rebuild → selection survives.
    panel.set_bias_options([(1000.0, 5)])
    assert panel.bias_cb.currentData() == pytest.approx(1000.0)

    # Bias gone → falls back to All and announces the change.
    seen: list[FolderFilterState] = []
    panel.folder_filter_changed.connect(lambda state: seen.append(state))
    panel.set_bias_options([(-550.0, 1)])
    assert panel.bias_cb.currentData() is None
    assert seen and seen[-1].bias_value_mv is None


def test_grid_bias_filter_hides_non_matching_scans_and_clears_selection(qapp):
    grid = _quiet_grid(qapp)
    keep = _scan_entry("keep", 40.0, 40.0, bias_mv=-25.0)
    hide = _scan_entry("hide", 120.0, 120.0, bias_mv=1000.0)
    grid.load([keep, hide], "/tmp")
    qapp.processEvents()
    qapp.processEvents()

    grid._on_card_click(hide, False)
    assert grid.get_primary_entry() == hide

    grid.set_folder_filter_state(FolderFilterState(bias_value_mv=-25.0))

    assert grid.get_primary_entry() is None
    assert grid.get_visible_scan_entries() == [keep]


def test_grid_hide_incomplete_filter(qapp):
    grid = _quiet_grid(qapp)
    full = _scan_entry("full", 100.0, 100.0)      # completion 100 %
    partial = _scan_entry("partial", 100.0, 30.0)  # completion 30 %
    grid.load([full, partial], "/tmp")
    qapp.processEvents()
    qapp.processEvents()

    grid.set_folder_filter_state(FolderFilterState(hide_incomplete=True))
    assert grid.get_visible_scan_entries() == [full]


def test_grid_bias_options_lists_distinct_biases(qapp):
    grid = _quiet_grid(qapp)
    grid.load(
        [
            _scan_entry("a", 10.0, 10.0, bias_mv=-25.0),
            _scan_entry("b", 10.0, 10.0, bias_mv=-25.3),
            _scan_entry("c", 10.0, 10.0, bias_mv=1000.0),
            _spec_entry("s"),
        ],
        "/tmp",
    )
    qapp.processEvents()
    assert grid.bias_options() == [(-25.0, 2), (1000.0, 1)]


def test_folders_hidden_in_images_and_spectra_modes(qapp):
    from probeflow.gui.models import FolderEntry

    grid = _quiet_grid(qapp)
    folder = FolderEntry(path=Path("/tmp/sub"), n_scans=1)
    scan = _scan_entry("scan", 10.0, 10.0)
    grid.load([folder, scan], "/tmp")
    qapp.processEvents()

    assert grid._is_entry_visible(folder) is True
    grid.apply_filter("images")
    assert grid._is_entry_visible(folder) is False
    assert grid._is_entry_visible(scan) is True
    grid.apply_filter("spectra")
    assert grid._is_entry_visible(folder) is False
    grid.apply_filter("all")
    assert grid._is_entry_visible(folder) is True


def test_ctrl_click_multi_selects_spectra_but_not_images(qapp):
    grid = _quiet_grid(qapp)
    img_a = _scan_entry("img_a", 10.0, 10.0)
    img_b = _scan_entry("img_b", 10.0, 10.0)
    spec_a = _spec_entry("spec_a")
    spec_b = _spec_entry("spec_b")
    grid.load([img_a, img_b, spec_a, spec_b], "/tmp")
    qapp.processEvents()
    qapp.processEvents()

    # Ctrl+click across two spectra accumulates a multi-selection.
    grid._on_card_click(spec_a, True)
    grid._on_card_click(spec_b, True)
    assert len(grid.get_selected()) == 2

    # Ctrl+click on an image behaves like a plain click: single selection.
    grid._on_card_click(img_a, True)
    assert len(grid.get_selected()) == 1
    assert grid.get_primary_entry() == img_a
    grid._on_card_click(img_b, True)
    assert len(grid.get_selected()) == 1
    assert grid.get_primary_entry() == img_b


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
    grid.set_folder_filter_state(FolderFilterState(hide_incomplete=True))

    assert launches
    assert started == ["tmp"]
    assert "Filtering tmp..." == grid._path_lbl.text()


def test_sort_by_size_orders_largest_first_with_spectra_last(qapp):
    grid = _quiet_grid(qapp)
    small = _scan_entry("a_small", 10.0, 10.0)
    big = _scan_entry("z_big", 200.0, 200.0)
    spec = _spec_entry("m_spec")
    grid.load([small, big, spec], "/tmp")
    qapp.processEvents()

    grid.set_sort_mode("size")
    files = [e for e in grid.get_entries()]
    assert files == [big, small, spec]

    grid.set_sort_mode("name")
    assert grid.get_entries() == [small, spec, big]


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
