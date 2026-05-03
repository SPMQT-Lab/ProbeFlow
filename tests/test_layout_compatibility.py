"""Layout checks for the canonical ProbeFlow package tree."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import pytest


PF = "probe" + "flow"

OLD_WRAPPER_MODULES = tuple(
    f"{PF}.{name}"
    for name in (
        "_analysis_helpers",
        "common",
        "createc_interpretation",
        "dat_png",
        "dat_sxm",
        "display",
        "display_state",
        "export_provenance",
        "features",
        "file_type",
        "gui_browse",
        "gui_features",
        "gui_models",
        "gui_processing",
        "gui_rendering",
        "gui_tv",
        "gui_viewer_widgets",
        "gui_workers",
        "indexing",
        "lattice",
        "loaders",
        "metadata",
        "prepared_export",
        "processing_state",
        "readers",
        "scan",
        "scan_model",
        "source_identity",
        "spec_io",
        "spec_plot",
        "spec_processing",
        "sxm_io",
        "validation",
        "writers",
        "xmgrace_export",
    )
)


def test_backend_imports_do_not_require_qt():
    import subprocess
    script = ";".join([
        "import sys",
        "import probeflow",
        "from probeflow import processing",
        "from probeflow.core.scan_loader import load_scan",
        "from probeflow.core.scan_model import Scan",
        "from probeflow.processing import align_rows",
        "from probeflow.processing.state import ProcessingState",
        "assert probeflow.Scan is Scan",
        "assert callable(load_scan)",
        "assert callable(align_rows)",
        "assert ProcessingState.__name__ == 'ProcessingState'",
        "assert processing.align_rows is align_rows",
        "assert 'PySide6' not in sys.modules, 'unexpected PySide6 import'",
    ])
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_canonical_io_and_analysis_imports_are_available():
    from probeflow.analysis.features import segment_particles
    from probeflow.analysis.lattice import LatticeParams
    from probeflow.io.converters.createc_dat_to_png import dat_to_hdr_imgs
    from probeflow.io.converters.createc_dat_to_sxm import convert_dat_to_sxm
    from probeflow.io.readers.createc_scan import read_dat
    from probeflow.io.readers.nanonis_sxm import read_sxm
    from probeflow.io.writers.json import write_json

    assert callable(dat_to_hdr_imgs)
    assert callable(convert_dat_to_sxm)
    assert callable(segment_particles)
    assert LatticeParams.__name__ == "LatticeParams"
    assert callable(read_dat)
    assert callable(read_sxm)
    assert callable(write_json)


def test_pure_gui_helpers_import_without_qt():
    import subprocess
    script = ";".join([
        "import sys",
        "from probeflow.gui import models, rendering",
        "from probeflow.processing import gui_adapter",
        "assert callable(gui_adapter.processing_state_from_gui)",
        "assert models.SxmFile.__name__ == 'SxmFile'",
        "assert callable(rendering.resolve_thumbnail_plane_index)",
        "assert 'PySide6' not in sys.modules, 'unexpected PySide6 import'",
    ])
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_gui_entrypoint_import_when_qt_available():
    pytest.importorskip("PySide6")

    from probeflow.gui import ImageViewerDialog, SxmFile, main

    assert ImageViewerDialog.__name__ == "ImageViewerDialog"
    assert SxmFile.__name__ == "SxmFile"
    assert callable(main)


def test_cli_import_path_remains_available():
    from probeflow.cli import main
    from probeflow.cli.processing_ops import _op_plane_bg

    assert callable(main)
    assert _op_plane_bg(1).name == "plane_bg"


def test_plugin_foundation_imports():
    from probeflow.plugins import PluginRegistry

    registry = PluginRegistry()
    assert registry.operations() == []


def test_spec_plot_private_helpers_import_from_canonical_module():
    from probeflow.analysis.spec_plot import _parse_sxm_offset, spec_position_to_pixel

    assert callable(_parse_sxm_offset)
    assert callable(spec_position_to_pixel)


def test_root_contains_no_relocation_wrapper_modules():
    root = Path(__file__).resolve().parents[1] / "probeflow"
    assert {path.name for path in root.glob("*.py")} == {"__init__.py"}


def test_readers_and_writers_wrapper_packages_are_removed():
    root = Path(__file__).resolve().parents[1] / "probeflow"
    assert not (root / "readers").exists()
    assert not (root / "writers").exists()


def test_repo_imports_use_canonical_package_paths():
    repo = Path(__file__).resolve().parents[1]
    search_roots = [repo / "probeflow", repo / "tests", repo / "README.md", repo / "pyproject.toml"]
    patterns = [
        re.compile(rf"\b(?:from|import)\s+{re.escape(module)}(?:\b|\.| )")
        for module in OLD_WRAPPER_MODULES
    ]
    offenders: list[tuple[str, str]] = []

    for root in search_roots:
        paths = [root] if root.is_file() else list(root.rglob("*.py"))
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                if pattern.search(text):
                    offenders.append((str(path.relative_to(repo)), pattern.pattern))

    assert offenders == []


def test_graph_node_types_are_reserved_for_provenance():
    root = Path(__file__).resolve().parents[1] / "probeflow"
    forbidden_defs = (
        "class ImageNode",
        "class MeasurementNode",
        "class OperationNode",
        "class ArtifactNode",
        "class ScanGraph",
    )
    offenders = []
    for package in ("processing", "analysis"):
        for path in (root / package).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in forbidden_defs:
                if marker in text:
                    offenders.append((str(path.relative_to(root)), marker))

    assert offenders == []
