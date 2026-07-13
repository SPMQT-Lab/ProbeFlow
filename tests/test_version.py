"""Release-version consistency tests."""

from __future__ import annotations

from pathlib import Path
import tomllib

from probeflow import __version__, display_version
from probeflow.measurements.export import _get_version as measurement_export_version
from probeflow.provenance.export import _get_version as provenance_export_version


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_candidate_version():
    assert __version__ == "1.0.0rc1"
    assert display_version() == "1.0.0 RC 1"


def test_pyproject_reads_the_package_version_source():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "version" in config["project"]["dynamic"]
    assert config["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "probeflow.__version__",
    }


def test_export_surfaces_use_the_release_version():
    assert measurement_export_version() == __version__
    assert provenance_export_version() == __version__
