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
    resolve_thumbnail_plane_index,
    save_config,
    THUMBNAIL_CHANNEL_DEFAULT,
    SxmFile,
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


class TestThumbnailChannelResolution:
    def test_scientific_thumbnail_channels_select_forward_plane_or_fallback_to_z(self):
        names = [
            "Z forward",
            "Z backward",
            "Current forward",
            "Current backward",
            "OC M1 Freq. Shift forward",
            "OC M1 Freq. Shift backward",
        ]
        assert resolve_thumbnail_plane_index(names, "Frequency shift") == 4
        assert resolve_thumbnail_plane_index(names, "Current") == 2
        assert resolve_thumbnail_plane_index(names, "Amplitude") == 0
        names_without_forward_freq = ["Z forward", "OC M1 Freq. Shift backward"]
        assert resolve_thumbnail_plane_index(names_without_forward_freq, "Frequency shift") == 0
        assert (
            resolve_thumbnail_plane_index(
                ["Z forward", "Current forward"], THUMBNAIL_CHANNEL_DEFAULT
            )
            == 0
        )


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

    def test_displayed_csv_uses_processing_controls(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QCheckBox
        from probeflow.gui import SpecViewerDialog, THEMES
        from probeflow.io.spectroscopy import SpecChannel, SpecData

        monkeypatch.setattr(SpecViewerDialog, "_load", lambda self: None)
        entry = VertFile(path=TESTDATA / "spectrum_time_trace_5k.VERT", stem="spec")
        dlg = SpecViewerDialog(entry, THEMES["dark"])
        x = np.linspace(-1.0, 1.0, 11)
        dlg._spec = SpecData(
            header={},
            channels={"I": 2e-9 * x + 1e-9},
            x_array=x,
            x_label="Bias (V)",
            x_unit="V",
            y_units={"I": "A"},
            position=(0.0, 0.0),
            metadata={"n_points": 11, "sweep_type": "bias_sweep"},
            channel_order=["I"],
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
            },
        )
        cb = QCheckBox("I")
        cb.setChecked(True)
        dlg._checkboxes = {"I": cb}
        dlg._derivative_cb.setCurrentText("Numerical dy/dx")

        text = dlg._current_csv_text()

        assert "numerical dI/dV" in text
        assert "nA/V" in text
        assert ",2" in text

        dlg.close()
        dlg.deleteLater()

    def test_viewer_rejects_invalid_savgol_without_replacing_valid_plot(
        self, qapp, monkeypatch
    ):
        from PySide6.QtWidgets import QCheckBox
        from probeflow.gui import SpecViewerDialog, THEMES
        from probeflow.io.spectroscopy import SpecData

        monkeypatch.setattr(SpecViewerDialog, "_load", lambda self: None)
        entry = VertFile(path=TESTDATA / "spectrum_time_trace_5k.VERT", stem="spec")
        dlg = SpecViewerDialog(entry, THEMES["dark"])
        x = np.linspace(-1.0, 1.0, 9)
        dlg._spec = SpecData(
            header={},
            channels={"I": x * x},
            x_array=x,
            x_label="Bias (V)",
            x_unit="V",
            y_units={"I": "A"},
            position=(0.0, 0.0),
            metadata={"n_points": 9, "sweep_type": "bias_sweep", "setpoint_a": 2.0},
            channel_order=["I"],
            default_channels=["I"],
        )
        cb = QCheckBox("I")
        cb.setChecked(True)
        dlg._checkboxes = {"I": cb}
        dlg._smoothing_cb.setCurrentText("Savitzky-Golay")
        dlg._smooth_points_spin.setValue(3)
        dlg._savgol_order_spin.setValue(2)
        dlg._redraw()
        assert dlg._displayed_traces
        canvas = dlg._canvas
        valid_y = dlg._displayed_traces[0].y_display.copy()

        dlg._smooth_points_spin.setValue(4)

        assert dlg._canvas is canvas
        np.testing.assert_allclose(dlg._displayed_traces[0].y_display, valid_y)
        assert "window must be odd" in dlg._status.text()

        dlg._smooth_points_spin.setValue(5)
        dlg._savgol_order_spin.setValue(5)
        assert "polynomial order must be smaller" in dlg._status.text()

        dlg._savgol_order_spin.setValue(2)
        dlg._smooth_points_spin.setValue(99)
        assert "must not exceed available points" in dlg._status.text()

        dlg.close()
        dlg.deleteLater()

    def test_viewer_crosshair_measures_displayed_trace(self, qapp, monkeypatch):
        from PySide6.QtWidgets import QCheckBox
        from probeflow.gui import SpecViewerDialog, THEMES
        from probeflow.io.spectroscopy import SpecData

        monkeypatch.setattr(SpecViewerDialog, "_load", lambda self: None)
        entry = VertFile(path=TESTDATA / "spectrum_time_trace_5k.VERT", stem="spec")
        dlg = SpecViewerDialog(entry, THEMES["dark"])
        x = np.array([0.0, 1.0, 2.0, 3.0])
        dlg._spec = SpecData(
            header={},
            channels={"I": np.array([0.0, 2.0, 4.0, 6.0])},
            x_array=x,
            x_label="Bias (V)",
            x_unit="V",
            y_units={"I": "A"},
            position=(0.0, 0.0),
            metadata={"n_points": 4, "sweep_type": "bias_sweep", "setpoint_a": 2.0},
            channel_order=["I"],
            default_channels=["I"],
        )
        cb = QCheckBox("I")
        cb.setChecked(True)
        dlg._checkboxes = {"I": cb}
        dlg._normalize_cb.setCurrentText("Constant")
        dlg._norm_constant_spin.setValue(2.0)
        dlg._redraw()
        dlg._set_measure_mode(True)
        axis = dlg._fig.axes[0]

        class Event:
            inaxes = axis
            button = 1

            def __init__(self, xdata, ydata):
                self.xdata = xdata
                self.ydata = ydata

        dlg._on_canvas_click(Event(1.0, 1.0))
        dlg._on_canvas_click(Event(3.0, 3.0))

        assert dlg._measurement is not None
        assert dlg._measurement.dx == pytest.approx(2.0)
        assert dlg._measurement.dy == pytest.approx(2.0)
        assert dlg._measurement.slope == pytest.approx(1.0)
        lbl = dlg._measure_lbl.text()
        assert "slope" in lbl
        assert "=" in lbl
        dlg._copy_measurement()
        assert "trace\tx1\ty1" in qapp.clipboard().text()
        dlg._add_measurement_result()
        measurements = dlg._measurement_table.results()
        assert len(measurements) == 1
        assert measurements[0].kind == "spectrum_delta"
        assert measurements[0].context["data_basis"] == "displayed_trace"
        assert measurements[0].context["normalization"] == "constant"

        dlg.close()
        dlg.deleteLater()

    def test_overlay_dialog_exports_long_csv(self, qapp):
        from probeflow.gui import SpecOverlayDialog, THEMES

        entries = [
            VertFile(path=TESTDATA / "createc_ivt_telegraph_300mv_a.VERT", stem="a"),
            VertFile(path=TESTDATA / "createc_ivt_telegraph_300mv_b.VERT", stem="b"),
        ]
        dlg = SpecOverlayDialog(entries, THEMES["dark"])

        text = dlg._current_csv_text()

        assert "probeflow_displayed_spectra" in text
        assert "source_file,spectrum_id,trace_label" in text
        assert ",a,a I," in text
        assert ",b,b I," in text

        dlg.close()
        dlg.deleteLater()

    def test_overlay_dialog_skips_incompatible_x_axes(self, qapp):
        from probeflow.gui import SpecOverlayDialog, THEMES

        entries = [
            VertFile(path=TESTDATA / "createc_ivt_telegraph_300mv_a.VERT", stem="time"),
            VertFile(path=TESTDATA / "createc_vert_didz_image_state.VERT", stem="bias"),
        ]
        dlg = SpecOverlayDialog(entries, THEMES["dark"])

        displayed = dlg._current_displayed_spectra()
        text = dlg._current_csv_text()

        assert len(displayed) == 1
        assert displayed[0].spectrum_id == "time"
        assert "bias" not in text
        assert "Skipped 1 spectra" in dlg._status.text()

        dlg.close()
        dlg.deleteLater()

    def test_overlay_dialog_skips_matching_labels_with_different_x_values(
        self, qapp, monkeypatch
    ):
        from probeflow.gui import SpecOverlayDialog, THEMES
        from probeflow.io.spectroscopy import SpecData
        import probeflow.io.spectroscopy as spec_io

        def fake_read_spec_file(path):
            x = np.array([0.0, 0.5, 1.0])
            if Path(path).stem == "shifted":
                x = np.array([0.0, 0.6, 1.0])
            return SpecData(
                header={},
                channels={"I": np.array([1.0, 2.0, 3.0])},
                x_array=x,
                x_label="Bias (V)",
                x_unit="V",
                y_units={"I": "A"},
                position=(0.0, 0.0),
                metadata={"n_points": 3, "sweep_type": "bias_sweep"},
                channel_order=["I"],
                default_channels=["I"],
            )

        monkeypatch.setattr(spec_io, "read_spec_file", fake_read_spec_file)
        entries = [
            VertFile(path=Path("reference.VERT"), stem="reference"),
            VertFile(path=Path("shifted.VERT"), stem="shifted"),
        ]
        dlg = SpecOverlayDialog(entries, THEMES["dark"])

        displayed = dlg._current_displayed_spectra()
        text = dlg._current_csv_text()

        assert len(displayed) == 1
        assert displayed[0].spectrum_id == "reference"
        assert "shifted" not in text
        assert "x-axis values differ from reference" in dlg._status.text()

        dlg.close()
        dlg.deleteLater()


class TestBrowserIndexContracts:
    def test_split_keeps_good_scans_spectra_and_load_errors_disjoint(self):
        scans, spectra, errors = split_indexed_items(SAMPLE_ITEMS)

        assert {item.path.name for item in scans} == {"step.dat", "moire.sxm"}
        assert {item.path.name for item in spectra} == {"spec.VERT", "spec.dat"}
        assert {item.path.name for item in errors} == {"broken.dat"}
        assert {item.path for item in scans}.isdisjoint(item.path for item in spectra)
        assert all(item.load_error is None for item in scans + spectra)
        assert all(item.load_error is not None for item in errors)
        assert "broken.dat" not in {item.path.name for item in image_browser_items(SAMPLE_ITEMS)}

    def test_scan_items_preserve_physical_metadata_and_filter_spectra(self):
        items = [
            _make_item(
                "a.dat",
                item_type="scan",
                source_format="createc_dat",
                shape=(330, 511),
                bias=0.05,
                setpoint=4.4e-10,
                metadata={
                    "experiment_metadata": {
                        "acquisition_mode": "afm",
                        "topography_role": "afm_topography",
                    }
                },
            ),
            _make_item(
                "b.sxm",
                item_type="scan",
                source_format="nanonis_sxm",
                shape=(4, 4),
                scan_range=(10e-9, 10e-9),
            ),
            _make_item("spec.VERT", item_type="spectrum", source_format="createc_vert"),
        ]

        dat, sxm = _scan_items_to_sxm(items)

        assert isinstance(dat, SxmFile)
        assert dat.Nx == 511
        assert dat.Ny == 330
        assert dat.bias_mv == pytest.approx(50.0)
        assert dat.current_pa == pytest.approx(440.0)
        assert dat.source_format == "dat"
        assert dat.acquisition_label == "AFM df topography"
        assert "AFM df topography" in _card_meta_str(dat)
        assert sxm.scan_nm == pytest.approx(10.0)
        assert sxm.source_format == "sxm"

    def test_scan_items_keep_error_stub_unknown_current_and_dedupe_stems(self):
        entries = _scan_items_to_sxm(
            [
                _make_item("bad.dat", item_type="scan", load_error="zlib error"),
                _make_item("scan.sxm", item_type="scan", source_format="nanonis_sxm", shape=(4, 4)),
                _make_item("scan.dat", item_type="scan", source_format="createc_dat", shape=(4, 4)),
            ]
        )

        assert entries[0].Nx == 512
        assert "I: ?" in _card_meta_str(
            SxmFile(
                path=TESTDATA / "createc_scan_island_60nm.dat",
                stem="createc_scan_island_60nm",
                Nx=511,
                Ny=512,
                scan_nm=60.0,
                bias_mv=1213.0,
                current_pa=None,
            )
        )
        assert [entry.path.name for entry in entries[1:]] == ["scan.sxm"]

    def test_spec_items_preserve_trace_metadata_and_filter_scans(self):
        items = [
            _make_item("scan.dat", item_type="scan", source_format="createc_dat", shape=(4, 4)),
            _make_item(
                "s.VERT",
                item_type="spectrum",
                source_format="createc_vert",
                metadata={
                    "sweep_type": "bias_sweep",
                    "n_points": 1000,
                    "measurement_family": "iz",
                    "derivative_label": "dI/dz",
                },
            ),
            _make_item(
                "bad.VERT",
                item_type="spectrum",
                source_format="createc_vert",
                load_error="parse error",
            ),
        ]

        good, errored = _spec_items_to_vert(items)

        assert isinstance(good, VertFile)
        assert good.sweep_type == "bias_sweep"
        assert good.n_points == 1000
        assert good.measurement_family == "iz"
        assert good.measurement_label == "I(z) / dI/dz"
        assert errored.sweep_type == "unknown"
        assert errored.n_points == 0

    TESTDATA = Path(__file__).resolve().parents[1] / "test_data"

    def test_real_fixtures_index_without_errors_and_map_to_scan_or_spectrum_cards(self):
        from probeflow.core.indexing import index_folder

        items = index_folder(self.TESTDATA, recursive=False, include_errors=True)
        sxm_list = _scan_items_to_sxm(items)
        vert_list = _spec_items_to_vert(items)
        errors = [item for item in items if item.load_error]
        scan_names = {entry.path.name for entry in sxm_list}

        assert "createc_scan_step_20nm.dat" in scan_names
        assert "createc_scan_terrace_109nm.dat" in scan_names
        assert "sxm_moire_10nm.sxm" in scan_names
        assert len(vert_list) >= 3
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
