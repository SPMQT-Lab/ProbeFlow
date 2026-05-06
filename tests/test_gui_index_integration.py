"""Tests for GUI folder discovery via index_folder().

These tests cover the pure filtering helpers and the SxmFile/VertFile
conversion layer.  They do not require Qt or a running GUI.
"""

from __future__ import annotations

import os
import importlib
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from probeflow.core.indexing import (
    ProbeFlowItem,
    image_browser_items,
    split_indexed_items,
)
from probeflow.gui import (
    _card_meta_str,
    _scan_items_to_sxm,
    _spec_items_to_vert,
    BrowseInfoPanel,
    BrowseToolPanel,
    GUI_FONT_DEFAULT,
    GUI_FONT_SIZES,
    load_config,
    Navbar,
    normalise_gui_font_size,
    render_scan_image,
    render_scan_thumbnail,
    render_with_processing,
    resolve_thumbnail_plane_index,
    save_config,
    THUMBNAIL_CHANNEL_DEFAULT,
    SxmFile,
    ThumbnailGrid,
    THEMES,
    VertFile,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_item(
    name: str = "test.dat",
    item_type: str = "scan",
    source_format: str = "createc_dat",
    shape=None,
    load_error: str | None = None,
    bias: float | None = None,
    setpoint: float | None = None,
    scan_range=None,
    metadata: dict | None = None,
) -> ProbeFlowItem:
    return ProbeFlowItem(
        path=Path(name),
        display_name=Path(name).stem,
        source_format=source_format,
        item_type=item_type,
        shape=shape,
        scan_range=scan_range,
        bias=bias,
        setpoint=setpoint,
        load_error=load_error,
        metadata=metadata or {},
    )


SAMPLE_ITEMS = [
    _make_item("step.dat",    item_type="scan",     source_format="createc_dat", shape=(330, 511)),
    _make_item("moire.sxm",   item_type="scan",     source_format="nanonis_sxm", shape=(160, 160)),
    _make_item("spec.VERT",   item_type="spectrum", source_format="createc_vert"),
    _make_item("spec.dat",    item_type="spectrum", source_format="nanonis_dat_spectrum"),
    _make_item("broken.dat",  item_type="scan",     source_format="createc_dat", load_error="bad zlib"),
    _make_item("unknown.txt", item_type="unknown",  source_format="unknown"),
]


def test_gui_extracted_modules_import_without_main_window():
    for module_name in (
        "probeflow.gui.models",
        "probeflow.gui.rendering",
        "probeflow.gui.workers",
        "probeflow.gui.browse",
        "probeflow.gui.viewer.widgets",
    ):
        importlib.import_module(module_name)


def test_gui_compatibility_reexports_remain_available():
    from probeflow.gui import ImageViewerDialog, SxmFile, ThumbnailGrid, render_scan_image

    assert SxmFile.__name__ == "SxmFile"
    assert ThumbnailGrid.__name__ == "ThumbnailGrid"
    assert ImageViewerDialog.__name__ == "ImageViewerDialog"
    assert callable(render_scan_image)


class TestGuiWorkers:
    def test_thumbnail_loader_selects_requested_plane_and_emits(self, qapp, monkeypatch):
        from PIL import Image
        import probeflow.gui.workers as worker_mod

        token = object()
        calls = {}
        emitted = []

        class FakeScan:
            plane_names = ["Z forward", "Current forward"]
            n_planes = 2
            planes = [np.zeros((3, 3)), np.ones((3, 3))]

        def fake_render(**kwargs):
            calls.update(kwargs)
            return Image.new("RGB", (2, 2))

        monkeypatch.setattr(worker_mod, "load_scan", lambda _path: FakeScan())
        monkeypatch.setattr(worker_mod, "render_scan_image", fake_render)

        loader = worker_mod.ThumbnailLoader(
            SxmFile(path=Path("scan.dat"), stem="scan"),
            "gray",
            token,
            148,
            116,
            processing={"align_rows": "median"},
            thumbnail_channel="Current",
        )
        loader.signals.loaded.connect(lambda *args: emitted.append(args))
        loader.run()

        assert calls["arr"] is FakeScan.planes[1]
        assert "scan_path" not in calls
        assert calls["size"] == (148, 116)
        assert calls["processing"] == {"align_rows": "median"}
        assert len(emitted) == 1
        assert emitted[0][0] == "scan"
        assert emitted[0][2] is token

    def test_thumbnail_loader_suppresses_emit_when_render_fails(self, qapp, monkeypatch):
        import probeflow.gui.workers as worker_mod

        emitted = []

        def fail_load_scan(_path):
            raise ValueError("bad scan")

        monkeypatch.setattr(worker_mod, "load_scan", fail_load_scan)
        monkeypatch.setattr(worker_mod, "render_scan_image", lambda **_kwargs: None)

        loader = worker_mod.ThumbnailLoader(
            SxmFile(path=Path("broken.dat"), stem="broken"),
            "gray",
            object(),
            148,
            116,
        )
        loader.signals.loaded.connect(lambda *args: emitted.append(args))
        loader.run()

        assert emitted == []

    def test_channel_and_viewer_loaders_preserve_arr_vs_file_semantics(
        self, qapp, monkeypatch
    ):
        from PIL import Image
        import probeflow.gui.workers as worker_mod

        calls = []

        def fake_render(**kwargs):
            calls.append(kwargs)
            return Image.new("RGB", (2, 2))

        monkeypatch.setattr(worker_mod, "render_scan_image", fake_render)

        entry = SxmFile(path=Path("scan.sxm"), stem="scan")
        arr = np.ones((4, 4))
        ch_emitted = []
        ch_signals = worker_mod.ChannelSignals()
        ch_signals.loaded.connect(lambda *args: ch_emitted.append(args))
        worker_mod.ChannelLoader(
            entry,
            2,
            "plasma",
            "channel-token",
            124,
            98,
            ch_signals,
            processing={"align_rows": "mean"},
            arr=arr,
        ).run()

        viewer_arr_emitted = []
        viewer_arr = worker_mod.ViewerLoader(
            entry,
            "gray",
            "viewer-arr-token",
            None,
            plane_idx=1,
            processing={"align_rows": "median"},
            arr=arr,
        )
        viewer_arr.signals.loaded.connect(lambda *args: viewer_arr_emitted.append(args))
        viewer_arr.run()

        viewer_file_emitted = []
        viewer_file = worker_mod.ViewerLoader(
            entry,
            "gray",
            "viewer-file-token",
            None,
            plane_idx=1,
            processing={"align_rows": "median"},
        )
        viewer_file.signals.loaded.connect(lambda *args: viewer_file_emitted.append(args))
        viewer_file.run()

        assert calls[0]["scan_path"] is None
        assert calls[0]["arr"] is arr
        assert calls[0]["processing"] == {"align_rows": "mean"}
        assert calls[1]["scan_path"] is None
        assert calls[1]["arr"] is arr
        assert calls[1]["processing"] is None
        assert calls[2]["scan_path"] == entry.path
        assert calls[2]["arr"] is None
        assert calls[2]["processing"] == {"align_rows": "median"}
        assert ch_emitted[0][0] == 2
        assert ch_emitted[0][2] == "channel-token"
        assert viewer_arr_emitted[0][1] == "viewer-arr-token"
        assert viewer_file_emitted[0][1] == "viewer-file-token"

    def test_spec_thumbnail_loader_emits_only_when_render_succeeds(
        self, qapp, monkeypatch
    ):
        from PIL import Image
        import probeflow.gui.workers as worker_mod

        calls = []
        monkeypatch.setattr(
            worker_mod,
            "render_spec_thumbnail",
            lambda *args, **kwargs: calls.append((args, kwargs)) or Image.new("RGB", (2, 2)),
        )

        emitted = []
        entry = VertFile(path=Path("spec.VERT"), stem="spec")
        loader = worker_mod.SpecThumbnailLoader(entry, "token", 120, 80, dark=False)
        loader.signals.loaded.connect(lambda *args: emitted.append(args))
        loader.run()

        assert calls[0][0] == (entry.path,)
        assert calls[0][1] == {"size": (120, 80), "dark": False}
        assert emitted[0][0] == "spec"
        assert emitted[0][2] == "token"

        monkeypatch.setattr(worker_mod, "render_spec_thumbnail", lambda *_a, **_k: None)
        emitted.clear()
        loader.run()

        assert emitted == []

    def test_conversion_worker_reports_empty_sxm_directory(self, qapp, tmp_path):
        import probeflow.gui.workers as worker_mod

        logs = []
        finished = []
        worker = worker_mod.ConversionWorker(
            str(tmp_path / "input"),
            str(tmp_path / "output"),
            do_png=False,
            do_sxm=True,
            clip_low=1.0,
            clip_high=99.0,
        )
        Path(worker.in_dir).mkdir()
        worker.signals.log_msg.connect(lambda *args: logs.append(args))
        worker.signals.finished.connect(finished.append)

        worker.run()

        assert any(tag == "warn" and "No .dat files found" in msg for msg, tag in logs)
        assert finished == [worker.out_dir]

    def test_conversion_worker_records_per_file_sxm_failures(
        self, qapp, tmp_path, monkeypatch
    ):
        import json
        import probeflow.io.converters.createc_dat_to_sxm as dat_sxm_mod
        import probeflow.gui.workers as worker_mod

        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()
        good = in_dir / "good.dat"
        bad = in_dir / "bad.dat"
        good.write_bytes(b"good")
        bad.write_bytes(b"bad")

        def fake_convert(dat, sxm_out, cushion_dir, clip_low, clip_high):
            assert cushion_dir == worker_mod.DEFAULT_CUSHION
            assert clip_low == 2.0
            assert clip_high == 98.0
            if dat.name == "bad.dat":
                raise ValueError("decode failed")
            (sxm_out / f"{dat.stem}.sxm").write_bytes(b"sxm")

        monkeypatch.setattr(dat_sxm_mod, "convert_dat_to_sxm", fake_convert)

        logs = []
        finished = []
        worker = worker_mod.ConversionWorker(
            str(in_dir),
            str(out_dir),
            do_png=False,
            do_sxm=True,
            clip_low=2.0,
            clip_high=98.0,
        )
        worker.signals.log_msg.connect(lambda *args: logs.append(args))
        worker.signals.finished.connect(finished.append)

        worker.run()

        errors_path = out_dir / "sxm" / "errors.json"
        assert (out_dir / "sxm" / "good.sxm").exists()
        assert json.loads(errors_path.read_text()) == {"bad.dat": "decode failed"}
        assert any(tag == "err" and "FAILED bad.dat" in msg for msg, tag in logs)
        assert any(tag == "warn" and "1 file(s) failed" in msg for msg, tag in logs)
        assert finished == [str(out_dir)]

TESTDATA = Path(__file__).resolve().parents[1] / "test_data"


@pytest.fixture
def qapp():
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


# ── Test A: image_browser_items returns only non-errored scans ────────────────

class TestImageBrowserItems:
    def test_returns_only_scans(self):
        result = image_browser_items(SAMPLE_ITEMS)
        assert all(it.item_type == "scan" for it in result)

    def test_excludes_spectra(self):
        result = image_browser_items(SAMPLE_ITEMS)
        assert all(it.source_format not in {"createc_vert", "nanonis_dat_spectrum"}
                   for it in result)

    def test_excludes_errored(self):
        result = image_browser_items(SAMPLE_ITEMS)
        assert all(it.load_error is None for it in result)

    def test_count(self):
        result = image_browser_items(SAMPLE_ITEMS)
        assert len(result) == 2  # step.dat + moire.sxm

    def test_empty_input(self):
        assert image_browser_items([]) == []


# ── Test A2: large Viewer rendering preserves measured pixels ────────────────

class TestViewerRenderSizing:
    def test_scan_render_helper_preserves_native_createc_preview_size(self):
        img = render_scan_image(
            scan_path=TESTDATA / "createc_scan_preview_120nm.dat",
            size=None,
        )

        assert img is not None
        assert img.size == (63, 64)

    def test_scan_thumbnail_does_not_upscale_by_default(self):
        img = render_scan_thumbnail(
            TESTDATA / "sxm_moire_10nm.sxm",
            size=(900, 800),
            allow_upscale=False,
        )

        assert img is not None
        assert img.size == (160, 160)

    def test_viewer_render_defaults_to_native_pixel_size(self):
        img = render_scan_thumbnail(
            TESTDATA / "sxm_moire_10nm.sxm",
            size=None,
        )

        assert img is not None
        assert img.size == (160, 160)

    def test_viewer_ruler_layout_keeps_image_label_full_size(self, qapp, monkeypatch):
        from PySide6.QtGui import QPixmap
        from probeflow.gui import ImageViewerDialog, THEMES

        monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
        entry = SxmFile(path=TESTDATA / "sxm_moire_10nm.sxm", stem="sxm_moire_10nm")
        dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
        dlg._scan_range_m = (10e-9, 10e-9)
        token = dlg._token

        dlg._on_loaded(QPixmap(160, 160), token)
        qapp.processEvents()

        assert dlg._zoom_lbl.size().width() == 160
        assert dlg._zoom_lbl.size().height() == 160
        assert dlg._ruler_container.sizeHint().width() >= 186
        assert dlg._ruler_container.sizeHint().height() >= 186
        assert dlg._ruler_container.size().width() >= 186
        assert dlg._ruler_container.size().height() >= 186

        dlg.close()
        dlg.deleteLater()

    def test_small_viewer_image_opens_at_native_size(self, qapp, monkeypatch):
        from PySide6.QtGui import QPixmap
        from probeflow.gui import ImageViewerDialog, THEMES

        monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
        entry = SxmFile(
            path=TESTDATA / "createc_scan_preview_120nm.dat",
            stem="createc_scan_preview_120nm",
        )
        dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
        dlg._scan_range_m = (120e-9, 120e-9)
        dlg.show()
        qapp.processEvents()

        dlg._on_loaded(QPixmap(63, 64), dlg._token)
        qapp.processEvents()

        assert dlg._zoom_lbl.zoom() == 1.0
        assert dlg._zoom_lbl.size().width() == 63
        assert dlg._zoom_lbl.size().height() == 64

        dlg._zoom_lbl.zoom_by(2.0)
        qapp.processEvents()

        assert dlg._zoom_lbl.size().width() == 126
        assert dlg._zoom_lbl.size().height() == 128

        dlg._zoom_lbl.reset_zoom()
        qapp.processEvents()

        assert dlg._zoom_lbl.size().width() == 63
        assert dlg._zoom_lbl.size().height() == 64

        dlg.close()
        dlg.deleteLater()

    def test_large_viewer_image_opens_at_native_size(self, qapp, monkeypatch):
        from PySide6.QtGui import QPixmap
        from probeflow.gui import ImageViewerDialog, THEMES

        monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
        entry = SxmFile(path=TESTDATA / "createc_scan_hires_survey_99nm.dat",
                        stem="createc_scan_hires_survey_99nm")
        dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
        dlg._scan_range_m = (99e-9, 99e-9)

        dlg._on_loaded(QPixmap(1023, 1024), dlg._token)
        qapp.processEvents()

        assert dlg._zoom_lbl.zoom() == 1.0
        assert dlg._zoom_lbl.size().width() == 1023
        assert dlg._zoom_lbl.size().height() == 1024

        dlg.close()
        dlg.deleteLater()

    def test_processed_viewer_render_defaults_to_native_pixel_size(self):
        from probeflow.core.scan_loader import load_scan

        scan = load_scan(TESTDATA / "sxm_moire_10nm.sxm")
        img = render_scan_image(
            arr=scan.planes[0],
            colormap="gray",
            clip_low=1.0,
            clip_high=99.0,
            processing={"align_rows": "median"},
            size=None,
        )
        compat = render_with_processing(
            scan.planes[0],
            "gray",
            1.0,
            99.0,
            {"align_rows": "median"},
            size=None,
        )

        assert img is not None
        assert img.size == (160, 160)
        assert compat is not None
        assert compat.size == img.size

    def test_scan_workers_share_render_helper(self, monkeypatch):
        from PIL import Image
        import probeflow.gui as gui_mod
        import probeflow.gui.workers as worker_mod

        calls = []

        class FakeScan:
            plane_names = ["Z forward", "Current forward"]
            n_planes = 2
            planes = [np.zeros((4, 4)), np.ones((4, 4))]

        def fake_render(**kwargs):
            calls.append(kwargs)
            return Image.new("RGB", (2, 2))

        monkeypatch.setattr(worker_mod, "load_scan", lambda _path: FakeScan())
        monkeypatch.setattr(worker_mod, "render_scan_image", fake_render)

        entry = SxmFile(path=Path("scan.dat"), stem="scan")
        gui_mod.ThumbnailLoader(
            entry, "gray", object(), 148, 116,
            thumbnail_channel="Current",
        ).run()
        gui_mod.ChannelLoader(
            entry, 1, "gray", object(), 124, 98, gui_mod.ChannelSignals(),
        ).run()
        gui_mod.ViewerLoader(
            entry, "gray", object(), None, plane_idx=1,
        ).run()

        assert len(calls) == 3
        assert calls[0]["arr"] is FakeScan.planes[1]
        assert calls[0]["size"] == (148, 116)
        assert calls[1]["arr"] is None
        assert calls[1]["scan_path"] == entry.path
        assert calls[1]["size"] == (124, 98)
        assert calls[2]["arr"] is None
        assert calls[2]["scan_path"] == entry.path
        assert calls[2]["size"] is None

    def test_viewer_display_range_refresh_preserves_zoom_and_reuses_display_array(self, qapp, monkeypatch):
        from probeflow.gui import ImageViewerDialog, THEMES
        import probeflow.gui as gui_mod

        monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
        monkeypatch.setattr(gui_mod, "load_scan", lambda _path: pytest.fail("display refresh reloaded source"))

        class SyncPool:
            def start(self, loader):
                loader.run()

        entry = SxmFile(path=Path("scan.sxm"), stem="scan")
        dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
        dlg._pool = SyncPool()
        dlg._raw_arr = np.arange(100, dtype=float).reshape(10, 10)
        dlg._display_arr = dlg._raw_arr
        dlg._scan_range_m = (10e-9, 10e-9)
        dlg._scan_plane_names = ["Z forward"]
        dlg._scan_plane_units = ["m"]

        dlg._refresh_viewer_pixmap(reset_zoom=True)
        qapp.processEvents()
        dlg._zoom_lbl.zoom_by(0.5)
        qapp.processEvents()

        dlg._drs.set_percentile(2.0, 98.0)
        dlg._refresh_display_range()
        qapp.processEvents()

        assert dlg._zoom_lbl.zoom() == 0.5

        dlg.close()
        dlg.deleteLater()

    def test_viewer_histogram_bounds_stay_visible_at_full_range(self, qapp, monkeypatch):
        from probeflow.gui import ImageViewerDialog, THEMES

        monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)

        entry = SxmFile(path=Path("scan.sxm"), stem="scan")
        dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
        dlg._raw_arr = np.arange(100, dtype=float).reshape(10, 10)
        dlg._display_arr = dlg._raw_arr
        dlg._scan_plane_names = ["Z forward"]
        dlg._scan_plane_units = ["m"]
        dlg._drs.set_percentile(0.0, 100.0)

        dlg._update_histogram()

        x0, x1 = dlg._ax.get_xlim()
        low_x = dlg._low_line.get_xdata()[0]
        high_x = dlg._high_line.get_xdata()[0]

        assert x0 < low_x < x1
        assert x0 < high_x < x1

        dlg.close()
        dlg.deleteLater()

    def test_viewer_histogram_manual_range_preserves_zoom(self, qapp, monkeypatch):
        from probeflow.gui import ImageViewerDialog, THEMES
        import probeflow.gui as gui_mod

        monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
        monkeypatch.setattr(gui_mod, "load_scan", lambda _path: pytest.fail("hist refresh reloaded source"))

        class SyncPool:
            def start(self, loader):
                loader.run()

        entry = SxmFile(path=Path("scan.sxm"), stem="scan")
        dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
        dlg._pool = SyncPool()
        dlg._raw_arr = np.arange(100, dtype=float).reshape(10, 10)
        dlg._display_arr = dlg._raw_arr
        dlg._scan_range_m = (10e-9, 10e-9)
        dlg._scan_plane_names = ["Z forward"]
        dlg._scan_plane_units = ["m"]
        dlg._update_histogram()
        dlg._refresh_viewer_pixmap(reset_zoom=True)
        qapp.processEvents()
        dlg._zoom_lbl.zoom_by(0.5)
        qapp.processEvents()

        dlg._dragging = "low"
        dlg._low_line.set_xdata([10.0, 10.0])
        dlg._high_line.set_xdata([90.0, 90.0])
        dlg._on_hist_release(type("Event", (), {})())
        qapp.processEvents()

        assert dlg._zoom_lbl.zoom() == 0.5

        dlg.close()
        dlg.deleteLater()

    def test_viewer_navigation_resets_zoom(self, qapp, monkeypatch):
        from probeflow.gui import ImageViewerDialog, THEMES
        import probeflow.gui as gui_mod

        class FakeScan:
            plane_names = ["Z forward"]
            plane_units = ["m"]
            n_planes = 1
            planes = [np.arange(100, dtype=float).reshape(10, 10)]
            header = {}
            scan_range_m = (10e-9, 10e-9)

        class SyncPool:
            def start(self, loader):
                loader.run()

        load_current = ImageViewerDialog._load_current
        monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
        monkeypatch.setattr(gui_mod, "load_scan", lambda _path: FakeScan())

        entries = [SxmFile(path=Path("a.sxm"), stem="a"), SxmFile(path=Path("b.sxm"), stem="b")]
        dlg = ImageViewerDialog(entries[0], entries, "gray", THEMES["dark"])
        dlg._pool = SyncPool()
        monkeypatch.setattr(ImageViewerDialog, "_load_current", load_current)
        load_current(dlg, reset_zoom=True)
        qapp.processEvents()
        dlg._zoom_lbl.zoom_by(0.5)
        qapp.processEvents()

        dlg._go_next()
        qapp.processEvents()

        assert dlg._zoom_lbl.zoom() == 1.0

        dlg.close()
        dlg.deleteLater()

    def test_viewer_processing_refresh_applies_processing_once(self, qapp, monkeypatch):
        from probeflow.gui import ImageViewerDialog, THEMES
        import probeflow.gui as gui_mod

        calls = []

        def fake_apply(arr, processing):
            calls.append(processing)
            return arr + 1

        class SyncPool:
            def start(self, loader):
                loader.run()

        monkeypatch.setattr(ImageViewerDialog, "_load_current", lambda self: None)
        monkeypatch.setattr(gui_mod, "_apply_processing", fake_apply)

        entry = SxmFile(path=Path("scan.sxm"), stem="scan")
        dlg = ImageViewerDialog(entry, [entry], "gray", THEMES["dark"])
        dlg._pool = SyncPool()
        dlg._raw_arr = np.arange(100, dtype=float).reshape(10, 10)
        dlg._display_arr = dlg._raw_arr
        dlg._processing = {"align_rows": "median"}
        dlg._scan_range_m = (10e-9, 10e-9)
        dlg._scan_plane_names = ["Z forward"]
        dlg._scan_plane_units = ["m"]

        dlg._refresh_processing_display()
        qapp.processEvents()

        assert calls == [{"align_rows": "median"}]

        dlg.close()
        dlg.deleteLater()


class TestThumbnailChannelResolution:
    def test_frequency_shift_forward_is_selected(self):
        names = [
            "Z forward",
            "Z backward",
            "Current forward",
            "Current backward",
            "OC M1 Freq. Shift forward",
            "OC M1 Freq. Shift backward",
        ]

        assert resolve_thumbnail_plane_index(names, "Frequency shift") == 4

    def test_backward_only_afm_channel_falls_back_to_z(self):
        names = [
            "Z forward",
            "Z backward",
            "OC M1 Freq. Shift backward",
        ]

        assert resolve_thumbnail_plane_index(names, "Frequency shift") == 0

    def test_missing_channel_falls_back_to_plane_zero(self):
        names = ["Z forward", "Z backward", "Current forward", "Current backward"]

        assert resolve_thumbnail_plane_index(names, "Amplitude") == 0

    def test_current_forward_is_selected(self):
        names = ["Z forward", "Z backward", "Current forward", "Current backward"]

        assert resolve_thumbnail_plane_index(names, "Current") == 2

    def test_default_is_z(self):
        names = ["Z forward", "Current forward"]

        assert resolve_thumbnail_plane_index(names, THUMBNAIL_CHANNEL_DEFAULT) == 0


class TestThumbnailGridChannelSelection:
    @staticmethod
    def _patch_thumbnail_loader(monkeypatch):
        import probeflow.gui.browse as browse_mod

        captured = []

        class _Signal:
            def connect(self, _slot):
                pass

        class _Signals:
            loaded = _Signal()

        class FakeThumbnailLoader:
            def __init__(self, entry, colormap, token, w, h, *args, **kwargs):
                captured.append({
                    "stem": entry.stem,
                    "colormap": colormap,
                    "thumbnail_channel": kwargs.get("thumbnail_channel"),
                    "processing": kwargs.get("processing"),
                })
                self.signals = _Signals()

        class FakePool:
            def start(self, _loader):
                pass

        monkeypatch.setattr(browse_mod, "ThumbnailLoader", FakeThumbnailLoader)
        return captured, FakePool

    def test_load_defaults_to_z_channel_for_scan_cards(self, qapp, monkeypatch):
        import probeflow.gui as gui_mod

        captured, FakePool = self._patch_thumbnail_loader(monkeypatch)
        grid = ThumbnailGrid(gui_mod.THEMES["dark"])
        grid._pool = FakePool()
        entries = [
            SxmFile(path=Path("a.sxm"), stem="a"),
            VertFile(path=Path("spec.VERT"), stem="spec"),
        ]

        grid.load(entries)

        assert [(c["stem"], c["thumbnail_channel"]) for c in captured] == [("a", "Z")]

        grid.close()
        grid.deleteLater()

    def test_changing_thumbnail_channel_rerenders_scan_cards(self, qapp, monkeypatch):
        import probeflow.gui as gui_mod

        captured, FakePool = self._patch_thumbnail_loader(monkeypatch)
        grid = ThumbnailGrid(gui_mod.THEMES["dark"])
        grid._pool = FakePool()
        entries = [
            SxmFile(path=Path("a.sxm"), stem="a"),
            SxmFile(path=Path("b.sxm"), stem="b"),
            VertFile(path=Path("spec.VERT"), stem="spec"),
        ]
        grid.load(entries)
        captured.clear()

        n = grid.set_thumbnail_channel("Frequency shift")

        assert n == 2
        assert [(c["stem"], c["thumbnail_channel"]) for c in captured] == [
            ("a", "Frequency shift"),
            ("b", "Frequency shift"),
        ]

        grid.close()
        grid.deleteLater()

    def test_changing_thumbnail_colormap_rerenders_all_scan_cards(self, qapp, monkeypatch):
        import probeflow.gui as gui_mod

        captured, FakePool = self._patch_thumbnail_loader(monkeypatch)
        grid = ThumbnailGrid(gui_mod.THEMES["dark"])
        grid._pool = FakePool()
        entries = [
            SxmFile(path=Path("a.sxm"), stem="a"),
            SxmFile(path=Path("b.sxm"), stem="b"),
            VertFile(path=Path("spec.VERT"), stem="spec"),
        ]
        grid.load(entries)
        captured.clear()

        n = grid.set_thumbnail_colormap("plasma")

        assert n == 2
        assert [(c["stem"], c["colormap"]) for c in captured] == [
            ("a", "plasma"),
            ("b", "plasma"),
        ]

        grid.close()
        grid.deleteLater()

    def test_changing_thumbnail_align_rows_rerenders_preview_only(self, qapp, monkeypatch):
        import probeflow.gui as gui_mod

        captured, FakePool = self._patch_thumbnail_loader(monkeypatch)
        grid = ThumbnailGrid(gui_mod.THEMES["dark"])
        grid._pool = FakePool()
        entries = [SxmFile(path=Path("a.sxm"), stem="a")]
        grid.load(entries)
        captured.clear()

        n = grid.set_thumbnail_align_rows("Median")

        assert n == 1
        assert captured[0]["processing"] == {"align_rows": "median"}
        assert grid.get_card_state("a")[2] == {}

        captured.clear()
        grid.set_thumbnail_align_rows("None")
        assert captured[0]["processing"] is None

        grid.close()
        grid.deleteLater()


class TestBrowseLayoutCleanup:
    def test_font_size_helper_and_config_round_trip(self, tmp_path, monkeypatch):
        import probeflow.gui as gui_mod

        cfg_path = tmp_path / "config.json"
        monkeypatch.setattr(gui_mod, "CONFIG_PATH", cfg_path)

        assert normalise_gui_font_size("bogus") == GUI_FONT_DEFAULT
        assert GUI_FONT_SIZES == {"Small": 9, "Medium": 12, "Large": 14}

        save_config({"gui_font_size": "Large", "dark_mode": False})
        cfg = load_config()

        assert cfg["gui_font_size"] == "Large"
        assert cfg["dark_mode"] is False

    def test_browse_tool_panel_has_live_thumbnail_controls_only(self, qapp):
        from PySide6.QtWidgets import QPushButton

        panel = BrowseToolPanel(THEMES["dark"], {})
        button_texts = {btn.text() for btn in panel.findChildren(QPushButton)}

        assert panel.align_rows_cb.currentText() == "None"
        assert "Map spectra to images…" in button_texts
        assert "Apply to selection" not in button_texts
        assert "Auto clip (GMM)" not in button_texts
        assert "Apply to selected thumbnails" not in button_texts
        assert "Apply to all thumbnails" not in button_texts
        assert "↩ Undo last thumbnail change" not in button_texts
        assert "⟲ Reset to original (clear all filters)" not in button_texts
        assert "⬇ Export PNG…" not in button_texts

        panel.close()
        panel.deleteLater()

    def test_navbar_font_size_menu_defaults_and_emits(self, qapp):
        navbar = Navbar(True, "bogus")
        seen = []
        navbar.font_size_changed.connect(seen.append)

        assert navbar._font_size_label == "Medium"

        navbar.set_font_size("Large")

        assert seen == ["Large"]
        assert navbar._font_size_label == "Large"

        navbar.close()
        navbar.deleteLater()

    def test_browse_info_panel_keeps_key_values_and_channel_slots(self, qapp, monkeypatch):
        import probeflow.gui.browse as browse_mod

        class FakePool:
            def start(self, _loader):
                pass

        panel = BrowseInfoPanel(THEMES["dark"], {})
        panel._pool = FakePool()
        entry = SxmFile(
            path=TESTDATA / "sxm_moire_10nm.sxm",
            stem="sxm_moire_10nm",
            Nx=160,
            Ny=160,
            scan_nm=10.0,
            bias_mv=100.0,
            current_pa=50.0,
        )
        monkeypatch.setattr(browse_mod, "load_scan", lambda _path: type(
            "ScanLike",
            (),
            {"plane_names": ["Z forward", "Current forward"], "n_planes": 2, "header": {}},
        )())

        panel.show_entry(entry, "gray", {})

        assert panel._qi["pixels"].text() == "160 × 160"
        assert panel._qi["size"].text() == "10.0 nm"
        assert panel._qi["bias"].text() == "100 mV"
        assert panel._qi["setp"].text() == "50.0 pA"
        assert [lbl.text() for lbl in panel._ch_name_lbls] == [
            "Z forward",
            "Current forward",
        ]
        assert panel.name_lbl.sizePolicy().verticalPolicy().name == "Maximum"
        assert not hasattr(panel, "font_size_cb")
        assert panel._meta_widget.isVisible() is False
        assert panel.layout().stretch(panel.layout().indexOf(panel._meta_widget)) == 0
        assert panel.layout().stretch(panel.layout().indexOf(panel._bottom_spacer)) == 1

        panel.close()
        panel.deleteLater()

    def test_browse_info_show_entry_uses_raw_channel_previews(self, qapp, monkeypatch):
        panel = BrowseInfoPanel(THEMES["dark"], {})
        captured = {}

        def fake_load_channels(entry, colormap_key, processing=None):
            captured["processing"] = processing

        monkeypatch.setattr(panel, "load_channels", fake_load_channels)
        monkeypatch.setattr(panel, "_load_metadata", lambda _entry: None)

        entry = SxmFile(path=Path("scan.sxm"), stem="scan", Nx=4, Ny=4)
        panel.show_entry(entry, "gray", {"align_rows": "median"})

        assert captured["processing"] is None

        panel.close()
        panel.deleteLater()

    def test_browse_info_load_channels_decodes_once_and_renders_raw_planes(self, qapp, monkeypatch):
        import probeflow.gui.browse as browse_mod
        import probeflow.gui.workers as worker_mod
        from PIL import Image

        render_calls = []
        load_calls = []

        class SyncPool:
            def start(self, loader):
                loader.run()

        class FakeScan:
            plane_names = ["Z forward", "Current forward", "Aux"]
            plane_units = ["m", "A", "V"]
            n_planes = 3
            planes = [
                np.zeros((4, 4)),
                np.ones((4, 4)),
                np.full((4, 4), 2.0),
            ]
            header = {}

        def fake_load_scan(path):
            load_calls.append(path)
            return FakeScan()

        def fake_render(**kwargs):
            render_calls.append(kwargs)
            return Image.new("RGB", (2, 2))

        monkeypatch.setattr(browse_mod, "load_scan", fake_load_scan)
        monkeypatch.setattr(worker_mod, "render_scan_image", fake_render)

        panel = BrowseInfoPanel(THEMES["dark"], {})
        panel._pool = SyncPool()
        entry = SxmFile(path=Path("scan.sxm"), stem="scan", Nx=4, Ny=4)

        panel.load_channels(entry, "viridis", processing=None)

        assert load_calls == [entry.path]
        assert all(
            call["arr"] is plane
            for call, plane in zip(render_calls, FakeScan.planes)
        )
        assert all(call["scan_path"] is None for call in render_calls)
        assert all(call["processing"] is None for call in render_calls)

        panel.close()
        panel.deleteLater()

    def test_metadata_table_wraps_and_uses_resizable_columns(self, qapp):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QHeaderView

        panel = BrowseInfoPanel(THEMES["dark"], {})
        panel._meta_rows = [
            ("COMMENT", "long metadata value " * 20),
            ("SCAN_RANGE", "1 2"),
        ]
        panel._filter_meta()

        assert panel.meta_table.wordWrap() is True
        assert panel.meta_table.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
        assert panel.meta_table.horizontalHeader().sectionResizeMode(0) == QHeaderView.Interactive
        assert panel.meta_table.horizontalHeader().sectionResizeMode(1) == QHeaderView.Stretch
        assert panel.meta_table.rowHeight(0) >= panel.meta_table.verticalHeader().defaultSectionSize()

        panel._toggle_meta()

        assert panel._meta_widget.isHidden() is False
        assert panel.layout().stretch(panel.layout().indexOf(panel._meta_widget)) == 1
        assert panel.layout().stretch(panel.layout().indexOf(panel._bottom_spacer)) == 0

        panel.close()
        panel.deleteLater()


class TestSpecViewerRawData:
    def test_raw_data_table_shows_all_rows_with_display_units(self, qapp, monkeypatch):
        from probeflow.gui import SpecViewerDialog, THEMES
        from probeflow.io.spectroscopy import SpecChannel, SpecData

        monkeypatch.setattr(SpecViewerDialog, "_load", lambda self: None)
        entry = VertFile(path=TESTDATA / "spectrum_time_trace_5k.VERT", stem="spec")
        dlg = SpecViewerDialog(entry, THEMES["dark"])
        x = np.arange(25, dtype=float) / 1000.0
        dlg._spec = SpecData(
            header={},
            channels={
                "I": np.full(25, -2.5e-10),
                "Z": np.zeros(25),
                "V": np.full(25, -0.3),
            },
            x_array=x,
            x_label="Time (s)",
            x_unit="s",
            y_units={"I": "A", "Z": "m", "V": "V"},
            position=(0.0, 0.0),
            metadata={"n_points": 25, "sweep_type": "time_trace"},
            channel_order=["I", "Z", "V"],
            default_channels=["I"],
            channel_info={
                "I": SpecChannel(
                    key="I",
                    source_name="I",
                    source_label="I",
                    unit="A",
                    roles=("current",),
                    display_label="Current channel",
                ),
                "Z": SpecChannel(
                    key="Z",
                    source_name="Raw column 9",
                    source_label="Raw column 9",
                    unit="m",
                    roles=("z_feedback",),
                    display_label="Raw column 9 - Z feedback",
                ),
                "V": SpecChannel(
                    key="V",
                    source_name="V",
                    source_label="V",
                    unit="V",
                    roles=("bias_axis",),
                    display_label="Bias",
                ),
            },
        )

        table = dlg._raw_data_table()

        assert table.rowCount() == 25
        assert table.horizontalHeaderItem(1).text() == "Current channel (pA)"
        assert table.horizontalHeaderItem(2).text() == "Raw column 9 - Z feedback (nm)"
        assert table.horizontalHeaderItem(3).text() == "Bias (mV)"
        assert dlg._channel_display_label("Z") == "Raw column 9 - Z feedback  (nm)"
        assert table.item(24, 1).text() == "-250"
        assert table.item(24, 2).text() == "0"
        assert table.item(24, 3).text() == "-300"

        dlg.close()
        dlg.deleteLater()


# ── Test B: split_indexed_items separates scans, spectra, errors ──────────────

class TestSplitIndexedItems:
    def test_scans_contain_only_scans(self):
        scans, _, _ = split_indexed_items(SAMPLE_ITEMS)
        assert all(it.item_type == "scan" for it in scans)

    def test_spectra_contain_only_spectra(self):
        _, spectra, _ = split_indexed_items(SAMPLE_ITEMS)
        assert all(it.item_type == "spectrum" for it in spectra)

    def test_errors_contain_errored_items(self):
        _, _, errors = split_indexed_items(SAMPLE_ITEMS)
        assert all(it.load_error for it in errors)

    def test_spectra_not_in_scan_list(self):
        scans, spectra, _ = split_indexed_items(SAMPLE_ITEMS)
        scan_paths = {it.path for it in scans}
        spec_paths = {it.path for it in spectra}
        assert scan_paths.isdisjoint(spec_paths)

    def test_counts(self):
        scans, spectra, errors = split_indexed_items(SAMPLE_ITEMS)
        assert len(scans)   == 2  # step.dat + moire.sxm
        assert len(spectra) == 2  # spec.VERT + spec.dat
        assert len(errors)  == 1  # broken.dat


# ── Test C: errored items excluded from scan list ─────────────────────────────

class TestErrorItemsExcluded:
    def test_errored_scan_not_in_browser(self):
        result = image_browser_items(SAMPLE_ITEMS)
        names = {it.path.name for it in result}
        assert "broken.dat" not in names

    def test_only_errored_items_have_load_error(self):
        _, _, errors = split_indexed_items(SAMPLE_ITEMS)
        assert all(e.load_error is not None for e in errors)
        scans, spectra, _ = split_indexed_items(SAMPLE_ITEMS)
        assert all(e.load_error is None for e in scans + spectra)


# ── Test D: _scan_items_to_sxm conversion ────────────────────────────────────

class TestScanItemsToSxm:
    def test_produces_sxm_file_objects(self):
        result = _scan_items_to_sxm(SAMPLE_ITEMS)
        assert all(isinstance(e, SxmFile) for e in result)

    def test_shape_mapped_correctly(self):
        item = _make_item("a.dat", item_type="scan", source_format="createc_dat",
                          shape=(330, 511))
        result = _scan_items_to_sxm([item])
        assert result[0].Nx == 511
        assert result[0].Ny == 330

    def test_bias_converted_to_mv(self):
        item = _make_item("a.dat", item_type="scan", source_format="createc_dat",
                          shape=(4, 4), bias=0.05)   # 50 mV in V
        result = _scan_items_to_sxm([item])
        assert abs(result[0].bias_mv - 50.0) < 1e-9

    def test_setpoint_converted_to_pa(self):
        item = _make_item("a.dat", item_type="scan", source_format="createc_dat",
                          shape=(4, 4), setpoint=4.4e-10)  # 440 pA
        result = _scan_items_to_sxm([item])
        assert abs(result[0].current_pa - 440.0) < 1e-6

    def test_unknown_setpoint_formats_as_unknown_current(self):
        entry = SxmFile(
            path=TESTDATA / "createc_scan_island_60nm.dat",
            stem="createc_scan_island_60nm",
            Nx=511,
            Ny=512,
            scan_nm=60.0,
            bias_mv=1213.0,
            current_pa=None,
        )

        assert "I: ?" in _card_meta_str(entry)

    def test_afm_experiment_label_reaches_scan_card_metadata(self):
        item = _make_item(
            "afm.dat",
            item_type="scan",
            source_format="createc_dat",
            shape=(4, 4),
            metadata={
                "experiment_metadata": {
                    "acquisition_mode": "afm",
                    "topography_role": "afm_topography",
                }
            },
        )

        entry = _scan_items_to_sxm([item])[0]

        assert entry.acquisition_label == "AFM df topography"
        assert "AFM df topography" in _card_meta_str(entry)

    def test_scan_range_converted_to_nm(self):
        item = _make_item("a.sxm", item_type="scan", source_format="nanonis_sxm",
                          shape=(4, 4), scan_range=(10e-9, 10e-9))
        result = _scan_items_to_sxm([item])
        assert abs(result[0].scan_nm - 10.0) < 1e-9

    def test_source_format_mapped(self):
        dat = _make_item("a.dat", item_type="scan", source_format="createc_dat", shape=(4, 4))
        sxm = _make_item("b.sxm", item_type="scan", source_format="nanonis_sxm", shape=(4, 4))
        results = _scan_items_to_sxm([dat, sxm])
        assert results[0].source_format == "dat"
        assert results[1].source_format == "sxm"

    def test_errored_item_gets_stub_entry(self):
        item = _make_item("bad.dat", item_type="scan", source_format="createc_dat",
                          load_error="zlib error")
        result = _scan_items_to_sxm([item])
        assert len(result) == 1
        assert result[0].Nx == 512  # default

    def test_spectra_excluded(self):
        result = _scan_items_to_sxm(SAMPLE_ITEMS)
        names = {e.path.name for e in result}
        assert "spec.VERT" not in names
        assert "spec.dat"  not in names

    def test_stem_deduplication(self):
        dup1 = _make_item("scan.sxm", item_type="scan", source_format="nanonis_sxm", shape=(4, 4))
        dup2 = _make_item("scan.dat", item_type="scan", source_format="createc_dat", shape=(4, 4))
        # Same stem "scan" — second should be dropped.
        result = _scan_items_to_sxm([dup1, dup2])
        assert len(result) == 1


# ── Test E: _spec_items_to_vert conversion ────────────────────────────────────

class TestSpecItemsToVert:
    def test_produces_vert_file_objects(self):
        result = _spec_items_to_vert(SAMPLE_ITEMS)
        assert all(isinstance(e, VertFile) for e in result)

    def test_sweep_type_populated(self):
        item = _make_item("s.VERT", item_type="spectrum", source_format="createc_vert",
                          metadata={"sweep_type": "bias_sweep", "n_points": 1000})
        result = _spec_items_to_vert([item])
        assert result[0].sweep_type == "bias_sweep"

    def test_measurement_label_populated(self):
        item = _make_item(
            "s.VERT",
            item_type="spectrum",
            source_format="createc_vert",
            metadata={
                "sweep_type": "bias_sweep",
                "n_points": 1000,
                "measurement_family": "iz",
                "derivative_label": "dI/dz",
            },
        )
        result = _spec_items_to_vert([item])
        assert result[0].measurement_family == "iz"
        assert result[0].measurement_label == "I(z) / dI/dz"

    def test_n_points_populated(self):
        item = _make_item("s.VERT", item_type="spectrum", source_format="createc_vert",
                          metadata={"sweep_type": "bias_sweep", "n_points": 1024})
        result = _spec_items_to_vert([item])
        assert result[0].n_points == 1024

    def test_scan_items_excluded(self):
        result = _spec_items_to_vert(SAMPLE_ITEMS)
        names = {e.path.name for e in result}
        assert "step.dat" not in names
        assert "moire.sxm" not in names

    def test_errored_spec_gets_stub(self):
        item = _make_item("bad.VERT", item_type="spectrum", source_format="createc_vert",
                          load_error="parse error")
        result = _spec_items_to_vert([item])
        assert len(result) == 1
        assert result[0].sweep_type == "unknown"
        assert result[0].n_points == 0


# ── Test F: round-trip through real fixtures ──────────────────────────────────

class TestRealFixtureRoundTrip:
    TESTDATA = Path(__file__).resolve().parents[1] / "test_data"

    def test_createc_scans_appear_in_sxm_list(self):
        from probeflow.core.indexing import index_folder
        items = index_folder(self.TESTDATA, recursive=False, include_errors=True)
        sxm_list = _scan_items_to_sxm(items)
        names = {e.path.name for e in sxm_list}
        assert "createc_scan_step_20nm.dat"    in names
        assert "createc_scan_terrace_109nm.dat" in names

    def test_nanonis_sxm_appears_in_sxm_list(self):
        from probeflow.core.indexing import index_folder
        items = index_folder(self.TESTDATA, recursive=False, include_errors=True)
        sxm_list = _scan_items_to_sxm(items)
        names = {e.path.name for e in sxm_list}
        assert "sxm_moire_10nm.sxm" in names

    def test_spectra_appear_in_vert_list(self):
        from probeflow.core.indexing import index_folder
        items = index_folder(self.TESTDATA, recursive=False, include_errors=True)
        vert_list = _spec_items_to_vert(items)
        assert len(vert_list) >= 3

    def test_no_errors_on_real_fixtures(self):
        from probeflow.core.indexing import index_folder
        items = index_folder(self.TESTDATA, recursive=False, include_errors=True)
        errors = [it for it in items if it.load_error]
        assert errors == []


class TestSpecViewerLifetime:
    TESTDATA = Path(__file__).resolve().parents[1] / "test_data"

    def test_static_unit_controls_survive_load_cleanup(self):
        try:
            import shiboken6
            from PySide6.QtWidgets import QApplication
            from probeflow.gui import SpecViewerDialog, THEMES
        except Exception as exc:
            pytest.skip(f"Qt unavailable: {exc}")

        app = QApplication.instance()
        if app is None:
            try:
                app = QApplication([])
            except Exception as exc:
                pytest.skip(f"QApplication unavailable: {exc}")

        spec_path = self.TESTDATA / "createc_ivt_telegraph_300mv_a.VERT"
        entry = VertFile(path=spec_path, stem=spec_path.stem)
        dlg = SpecViewerDialog(entry, THEMES["light"])

        # Process any queued deleteLater calls from _load(). The static unit
        # group contains QComboBoxes; deleting it during channel refresh caused
        # a native Qt crash when closing the dialog.
        app.processEvents()
        assert shiboken6.isValid(dlg._z_unit_cb)
        assert shiboken6.isValid(dlg._i_unit_cb)
        assert shiboken6.isValid(dlg._v_unit_cb)

        dlg.close()
        dlg.deleteLater()
        app.processEvents()
