"""Windowed entry point used by installed and frozen ProbeFlow applications."""

from __future__ import annotations

import argparse
from multiprocessing import freeze_support
from pathlib import Path
import tempfile
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ProbeFlow")
    parser.add_argument(
        "--browse",
        type=Path,
        metavar="FOLDER",
        help="open FOLDER in the Browse workspace at startup",
    )
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def _run_smoke_test() -> None:
    """Exercise the frozen GUI, optional features and packaged resources."""

    import cv2
    from gwyfile.objects import GwyContainer
    import numpy as np
    import sklearn
    from PySide6.QtWidgets import QApplication

    from probeflow import display_version
    from probeflow.core.resources import ASSETS_DIR, FILE_CUSHIONS_DIR
    from probeflow.gui.app import ProbeFlowWindow, _configure_application_metadata
    from probeflow.processing.pdf_export import export_image_pdf

    required_resources = (
        ASSETS_DIR / "logo.png",
        ASSETS_DIR / "toolbar" / "open_fft.png",
        FILE_CUSHIONS_DIR / "header_format.json",
        FILE_CUSHIONS_DIR / "pre_payload_bytes.bin",
    )
    missing = [str(path) for path in required_resources if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing packaged resources: {', '.join(missing)}")

    # Use a private argv so Qt never attempts to interpret --smoke-test.
    app = QApplication.instance() or QApplication(["ProbeFlow-smoke-test"])
    app.setQuitOnLastWindowClosed(False)
    _configure_application_metadata(app)
    window = ProbeFlowWindow()
    window.show()
    app.processEvents()
    if not window.isVisible():
        raise RuntimeError("ProbeFlow main window did not become visible")
    window.close()
    app.processEvents()

    # Exercise the dynamically selected vector backend after Qt has already
    # initialised Matplotlib. This catches a frozen app that can display images
    # but omitted ``matplotlib.backends.backend_pdf`` from its bundle.
    with tempfile.TemporaryDirectory(prefix="probeflow-smoke-") as tmp_dir:
        pdf_path = Path(tmp_dir) / "packaged-smoke.pdf"
        gray_lut = np.stack(
            [np.arange(256, dtype=np.uint8)] * 3,
            axis=1,
        )
        export_image_pdf(
            np.arange(64, dtype=float).reshape(8, 8),
            pdf_path,
            "gray",
            1.0,
            99.0,
            lut_fn=lambda _name: gray_lut,
            scan_range_m=(8e-9, 8e-9),
            add_scalebar=False,
        )
        if not pdf_path.is_file() or pdf_path.read_bytes()[:4] != b"%PDF":
            raise RuntimeError("Packaged PDF export smoke test failed")

    # Referencing the imported symbols makes these checks explicit and keeps
    # static analysis from treating them as incidental imports.
    print(
        f"ProbeFlow {display_version()} packaged GUI smoke test passed; "
        f"OpenCV {cv2.__version__}, scikit-learn {sklearn.__version__}, "
        f"gwyfile {GwyContainer.__module__.split('.', 1)[0]}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse desktop-launch arguments and start the Qt application."""

    freeze_support()
    args = _build_parser().parse_args(argv)

    if args.smoke_test:
        _run_smoke_test()
        return

    # Keep the expensive GUI import below argument parsing. This allows entry
    # point checks such as ``probeflow-gui --help`` to remain lightweight and
    # ensures multiprocessing support is initialized first in frozen builds.
    from probeflow.gui import main as run_gui

    run_gui(browse_folder=args.browse)


if __name__ == "__main__":
    main()
