"""Generate the GUI screenshots used by README.md and docs/gui.md.

Drives the real widgets offscreen against copies of test_data fixtures and
saves PNGs to docs/images/.  Rerun after UI changes to refresh the docs:

    QT_QPA_PLATFORM=offscreen python scripts/generate_gui_screenshots.py

Fixtures are copied to a temp folder first because viewers write ROI/mask
sidecars next to the scans they open.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTDATA = REPO / "test_data"
OUT = REPO / "docs" / "images"

BROWSE_FIXTURES = [
    "createc_scan_terrace_109nm.dat",
    "createc_scan_atomic_11nm.dat",
    "createc_scan_hires_atomic_9nm.dat",
    "createc_scan_island_60nm.dat",
    "createc_scan_molecular_30nm_pos.dat",
    "createc_scan_step_20nm.dat",
    "createc_scan_overview_240nm_pos.dat",
    "createc_scan_qplus_10ch_afm.dat",
    "sxm_moire_10nm.sxm",
]

VIEWER_FIXTURE = "createc_scan_terrace_109nm.dat"
FFT_FIXTURE = "sxm_moire_10nm.sxm"
FEATURE_FIXTURE = "sxm_moire_10nm.sxm"


def _settle(app, seconds: float = 2.0) -> None:
    """Pump the event loop until async loaders/thumbnails have painted."""
    from PySide6.QtCore import QThreadPool

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    QThreadPool.globalInstance().waitForDone(10_000)
    for _ in range(50):
        app.processEvents()
        time.sleep(0.01)


def _grab(widget, name: str) -> None:
    path = OUT / name
    widget.grab().save(str(path))
    print(f"wrote {path.relative_to(REPO)}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="probeflow_shots_"))
    scans = tmp / "scans"
    scans.mkdir()
    for name in BROWSE_FIXTURES:
        shutil.copy2(TESTDATA / name, scans / name)

    # Keep the user's real GUI config untouched.
    import probeflow.gui.config as gui_config

    gui_config.CONFIG_PATH = tmp / "config.json"

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)

    from probeflow.core.scan_loader import load_scan
    from probeflow.gui.models import SxmFile
    from probeflow.gui.styling import THEMES, _build_palette, _build_qss

    theme = THEMES["dark"]
    app.setPalette(_build_palette(theme))
    app.setStyleSheet(_build_qss(theme))

    # ── 1. Main window: Browse grid with a folder loaded ──────────────────────
    from probeflow.gui.app import ProbeFlowWindow

    win = ProbeFlowWindow(browse_folder=scans)
    win.resize(1480, 860)
    win.show()
    _settle(app, 4.0)
    _grab(win, "gui_browse.png")
    win.close()
    _settle(app, 0.5)

    # ── 2. Image viewer on a terrace scan ──────────────────────────────────────
    from probeflow.gui.dialogs.image_viewer import ImageViewerDialog

    entry = SxmFile(path=scans / VIEWER_FIXTURE, stem=Path(VIEWER_FIXTURE).stem)
    viewer = ImageViewerDialog(entry, [entry], "gray", theme)
    viewer.resize(1380, 860)
    viewer.show()
    _settle(app, 3.0)
    _grab(viewer, "gui_viewer.png")

    # ── 3. STM background dialog opened from that viewer ──────────────────────
    viewer._on_open_stm_background()
    dlg = viewer._stm_background_dialog
    if dlg is not None:
        dlg.resize(1180, 760)
        _settle(app, 1.0)
        dlg._preview("corrected")
        _settle(app, 2.0)
        _grab(dlg, "gui_stm_background.png")
        dlg.close()
    viewer.close()
    _settle(app, 0.5)

    # ── 4. FFT viewer on an atomic-resolution scan ─────────────────────────────
    from probeflow.gui.dialogs.fft_viewer import FFTViewerDialog

    from probeflow.processing.alignment import align_rows
    from probeflow.processing.background import subtract_background

    scan = load_scan(scans / FFT_FIXTURE)
    # The FFT viewer is normally opened on the processed image; mirror that
    # by levelling the raw plane first so the Bragg peaks are visible.
    arr = subtract_background(align_rows(scan.planes[0], "median"), order=1)
    fft = FFTViewerDialog(arr, scan.scan_range_m, "gray", theme)
    fft.resize(1280, 820)
    fft.show()
    _settle(app, 2.0)
    # Zoom in on the spectral content, as a user inspecting Bragg peaks would.
    fft._zoom_by(0.25)
    _settle(app, 2.0)
    _grab(fft, "gui_fft.png")
    fft.close()
    _settle(app, 0.5)

    # ── 5. Feature finder with a detection run ─────────────────────────────────
    from probeflow.gui.dialogs.feature_finder import FeatureFinderDialog

    fscan = load_scan(scans / FEATURE_FIXTURE)
    farr = subtract_background(align_rows(fscan.planes[0], "median"), order=1)
    w_m, h_m = fscan.scan_range_m
    px_x = w_m / farr.shape[1]
    px_y = h_m / farr.shape[0]
    finder = FeatureFinderDialog(
        farr, pixel_size_x_m=px_x, pixel_size_y_m=px_y, theme=theme)
    finder.resize(1100, 700)
    finder.show()
    _settle(app, 1.0)
    # Detect the dark moiré sites: minima, smoothed a little, with a minimum
    # spacing so each site is found once — the settings a user would dial in.
    finder._minima_btn.click()
    finder._below_btn.click()
    finder._smooth_spin.setValue(2.0)
    finder._dist_spin.setValue(8.0)
    finder._run_detection()
    _settle(app, 2.0)
    _grab(finder, "gui_feature_finder.png")
    finder.close()
    _settle(app, 0.5)

    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
