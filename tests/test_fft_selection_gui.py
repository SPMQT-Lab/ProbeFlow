"""GUI tests for the FFT viewer's Reconstruct tab + Fourier selection overlay."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

N = 128


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


def _scene():
    yy, xx = np.mgrid[:N, :N]
    return np.exp(-(((xx - 40) ** 2 + (yy - 70) ** 2) / 400.0)) + 0.5 * np.sin(2 * np.pi * 8 * xx / N)


def _dialog(qapp, captured=None, roi=False, new_image_fn=None):
    from probeflow.gui.dialogs.fft_viewer import FFTViewerDialog
    img = _scene()
    apply_fn = (lambda op, p: captured.update(op=op, params=p)) if captured is not None else None
    kw = {}
    if roi:
        kw = dict(roi_bounds_px=(10, 90, 10, 90), roi_id="roi1", roi_name="r")
    dlg = FFTViewerDialog(img, (10e-9, 10e-9), apply_correction_fn=apply_fn,
                          get_image_fn=lambda: img, new_image_fn=new_image_fn, **kw)
    dlg._tab_widget.setCurrentIndex(dlg._reconstruct_tab_index)
    return dlg, img


def _ev(ax, xq, yq, shift=False):
    """A minimal stand-in for a matplotlib mouse event in the FFT axes."""
    disp = ax.transData.transform((xq, yq))

    class _E:
        inaxes = ax
        xdata = xq
        ydata = yq
        x = float(disp[0])
        y = float(disp[1])
        button = 1
        guiEvent = None
        key = "shift" if shift else None
    return _E()


def _draw(ov, ax, kind, p0, p1, shift=False):
    """Drag a shape from corner ``p0`` to ``p1`` (q-space)."""
    ov.set_tool(kind)
    ov.on_press(_ev(ax, *p0))
    ov.on_motion(_ev(ax, *p1, shift=shift))
    ov.on_release(_ev(ax, *p1, shift=shift))
    ov.set_tool(None)


def _paint(ov, ax, pts):
    ov.set_tool("paint")
    ov.on_press(_ev(ax, *pts[0]))
    for p in pts[1:]:
        ov.on_motion(_ev(ax, *p))
    ov.on_release(_ev(ax, *pts[-1]))
    ov.set_tool(None)


def _seed(dlg, kind="ellipse"):
    """Add one selection without driving the canvas — for reconstruct-tab tests."""
    from probeflow.gui.dialogs.fft_selection import FourierSelection
    ov = dlg._ensure_selection_overlay()
    ov._sels.append(FourierSelection(kind, cx_q=1.5, cy_q=0.0, rx_q=0.5, ry_q=0.5))
    ov._selected = len(ov._sels) - 1
    dlg._on_selection_changed()
    return ov


class TestSelectionOverlay:
    def test_draw_move_resize_delete(self, qapp):
        dlg, _img = _dialog(qapp)
        ov = dlg._ensure_selection_overlay()
        _draw(ov, dlg._ax_fft, "ellipse", (1.0, 0.5), (2.0, 1.5))
        assert ov.count() == 1 and len(ov._artists) > 0
        e = ov.to_regions()[0]
        assert e["kind"] == "ellipse" and e["dx"] > 0 and e["rx"] >= 1.0
        # resize: enlarging rx_q grows the px radius
        ov._sels[0].rx_q *= 2
        assert ov.to_regions()[0]["rx"] > e["rx"]
        ov.delete_selected()
        assert ov.count() == 0

    def test_tiny_click_is_discarded(self, qapp):
        dlg, _ = _dialog(qapp)
        ov = dlg._ensure_selection_overlay()
        _draw(ov, dlg._ax_fft, "ellipse", (1.0, 0.5), (1.0, 0.5))
        assert ov.count() == 0   # a click without a drag makes no selection

    def test_ellipse_independent_axes(self, qapp):
        dlg, _ = _dialog(qapp)
        ov = dlg._ensure_selection_overlay()
        _draw(ov, dlg._ax_fft, "ellipse", (1.0, 0.2), (3.0, 0.6))
        e = ov.to_regions()[0]
        assert e["rx"] != e["ry"]   # free drag → unequal semi-axes

    def test_shift_makes_regular(self, qapp):
        dlg, _ = _dialog(qapp)
        ov = dlg._ensure_selection_overlay()
        _draw(ov, dlg._ax_fft, "ellipse", (1.0, 0.2), (3.0, 0.6), shift=True)
        e = ov.to_regions()[0]
        assert e["rx"] == pytest.approx(e["ry"])   # Shift → circle

    def test_rectangle_kind_and_patches(self, qapp):
        from matplotlib.patches import Rectangle
        dlg, _ = _dialog(qapp)
        ov = dlg._ensure_selection_overlay()
        _draw(ov, dlg._ax_fft, "rect", (-2.0, -1.0), (-1.0, -0.2))
        r = ov.to_regions()[0]
        assert r["kind"] == "rect" and r["half_w"] >= 1.0 and r["half_h"] >= 1.0
        # rectangle patch + its conjugate
        assert sum(isinstance(a, Rectangle) for a in ov._artists) == 2

    def test_paint_region(self, qapp):
        from matplotlib.image import AxesImage
        dlg, _ = _dialog(qapp)
        ov = dlg._ensure_selection_overlay()
        ov.set_brush_radius_px(6)
        _paint(ov, dlg._ax_fft, [(0.5, -2.0), (1.0, -1.8), (1.5, -1.5), (2.0, -1.0)])
        assert ov.count() == 1
        p = ov.to_regions()[0]
        assert p["kind"] == "paint" and len(p["stamps"]) >= 1 and p["radius"] > 0
        # the painted overlay is drawn as an image
        assert any(isinstance(a, AxesImage) for a in ov._artists)

    def test_conjugate_drawn(self, qapp):
        dlg, _ = _dialog(qapp)
        ov = dlg._ensure_selection_overlay()
        _draw(ov, dlg._ax_fft, "ellipse", (1.0, 0.5), (2.0, 1.5))
        # two ellipse patches per selection (feature + conjugate)
        from matplotlib.patches import Ellipse
        assert sum(isinstance(a, Ellipse) for a in ov._artists) == 2

    def test_clear(self, qapp):
        dlg, _ = _dialog(qapp)
        ov = dlg._ensure_selection_overlay()
        _draw(ov, dlg._ax_fft, "ellipse", (1.0, 0.5), (2.0, 1.5))
        _draw(ov, dlg._ax_fft, "rect", (-2.0, -1.0), (-1.0, -0.2))
        assert ov.count() == 2
        dlg._on_clear_selections()
        assert ov.count() == 0


class TestReconstructTab:
    def test_tab_present(self, qapp):
        dlg, _ = _dialog(qapp)
        labels = [dlg._tab_widget.tabText(i) for i in range(dlg._tab_widget.count())]
        assert "Inverse FFT" in labels
        assert dlg._reconstruct_active()

    def test_preview_does_not_mutate_and_sets_active(self, qapp):
        dlg, img = _dialog(qapp)
        _seed(dlg)
        before = img.copy()
        dlg._on_reconstruct_preview()
        assert dlg._reconstruct_preview_active is True
        assert np.array_equal(img, before)   # source untouched
        dlg._on_reconstruct_clear()
        assert dlg._reconstruct_preview_active is False

    def test_whole_image_apply_routes_op(self, qapp):
        captured: dict = {}
        dlg, _ = _dialog(qapp, captured=captured)
        _seed(dlg)
        dlg._recon_mode_combo.setCurrentIndex(1)        # Keep selected
        dlg._recon_soft_spin.setValue(2)
        dlg._on_reconstruct_apply()
        assert captured["op"] == "inverse_fft_filter"
        p = captured["params"]
        assert p["mode"] == "keep_selected"
        assert p["conjugate_symmetric"] is True
        assert p["soft_px"] == 2.0
        assert len(p["selections"]) == 1
        assert p["fft_source"] == "whole_image"

    def test_apply_without_selection_warns(self, qapp):
        captured: dict = {}
        dlg, _ = _dialog(qapp, captured=captured)
        dlg._on_reconstruct_apply()
        assert "op" not in captured
        assert "selection" in dlg._recon_status_lbl.text().lower()

    def test_roi_apply_degrades_without_new_image_host(self, qapp):
        captured: dict = {}
        dlg, _ = _dialog(qapp, captured=captured, roi=True)
        # switch FFT source to the ROI
        dlg._fft_source = "active_roi"
        dlg._arr, dlg._scan_range_m = dlg._resolve_source_array()
        _seed(dlg)
        dlg._on_reconstruct_apply()
        assert "op" not in captured                     # no whole-image op routed
        assert "export" in dlg._recon_status_lbl.text().lower()

    def test_roi_apply_uses_new_image_host_when_available(self, qapp):
        opened = {}

        def _open_new_image(arr, scan_range_m, provenance):
            opened["arr"] = arr
            opened["scan_range_m"] = scan_range_m
            opened["provenance"] = provenance

        captured: dict = {}
        dlg, _ = _dialog(qapp, captured=captured, roi=True, new_image_fn=_open_new_image)
        dlg._fft_source = "active_roi"
        dlg._arr, dlg._scan_range_m = dlg._resolve_source_array()
        dlg._recompute_fft()
        _seed(dlg)
        dlg._on_reconstruct_apply()

        assert "op" not in captured
        assert opened["arr"].shape == dlg._arr.shape
        assert opened["scan_range_m"] == pytest.approx(tuple(dlg._scan_range_m))
        assert opened["provenance"]["op"] == "inverse_fft_filter"
        assert opened["provenance"]["params"]["fft_source"] == "active_roi"
        assert opened["provenance"]["params"]["fft_roi_id"] == "roi1"

    def test_export_writes_file(self, qapp, monkeypatch, tmp_path):
        dlg, _ = _dialog(qapp)
        _seed(dlg)
        out = tmp_path / "result.csv"
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(out), "CSV (*.csv)"))
        dlg._on_reconstruct_export("result")
        assert out.exists() and out.stat().st_size > 0

    def test_tooltips_wrapped(self, qapp):
        dlg, _ = _dialog(qapp)
        for w in (dlg._recon_mode_combo, dlg._recon_conj_cb, dlg._recon_soft_spin,
                  dlg._recon_preview_btn, dlg._recon_apply_btn):
            tt = w.toolTip()
            assert tt and max(len(line) for line in tt.split("\n")) <= 52

    def test_fields_are_content_width_not_stretched(self, qapp):
        """Spin boxes / combos must not stretch across the row (arrows would be
        pushed to the far edge, away from the value)."""
        from PySide6.QtWidgets import QSizePolicy
        dlg, _ = _dialog(qapp)
        for w in (dlg._recon_mode_combo, dlg._recon_soft_spin, dlg._recon_view_combo,
                  dlg._mains_harm_spin, dlg._mains_speed_spin, dlg._mains_freq_combo,
                  dlg._mains_radius_spin):
            assert w.sizePolicy().horizontalPolicy() == QSizePolicy.Maximum
