"""Smoke tests for CLI entry points and converter failure statuses."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

import probeflow.cli as cli
from probeflow.cli import _legacy
from probeflow.cli.commands.conversion import (
    _cmd_dat2npy,
    _cmd_dat2png,
    _cmd_dat2sxm,
    _run_converter,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dat2npy_is_available_through_legacy_cli_exports():
    assert _legacy._cmd_dat2npy is _cmd_dat2npy
    assert cli._cmd_dat2npy is _cmd_dat2npy
    assert "_cmd_dat2npy" in _legacy.__all__


def test_converter_restores_process_argv_after_success():
    original = sys.argv
    seen = []

    def converter():
        seen.append(list(sys.argv))
        return 0

    assert _run_converter(converter, "dat-npy", ["--", "--input", "scan.dat"]) == 0
    assert seen == [["dat-npy", "--input", "scan.dat"]]
    assert sys.argv is original


def test_converter_restores_process_argv_after_failure():
    original = sys.argv

    def converter():
        raise RuntimeError("conversion failed")

    with pytest.raises(RuntimeError, match="conversion failed"):
        _run_converter(converter, "dat-sxm", ["scan.dat"])
    assert sys.argv is original


def test_dat2sxm_batch_returns_nonzero_when_file_fails(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    (src / "broken.dat").write_text("not a createc file", encoding="utf-8")

    rc = _cmd_dat2sxm(argparse.Namespace(rest=[
        "--input-dir", str(src),
        "--output-dir", str(out),
    ]))

    assert rc == 1
    assert (out / "errors.json").exists()


def test_dat2png_batch_returns_nonzero_when_file_fails(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    (src / "broken.dat").write_text("not a createc file", encoding="utf-8")

    rc = _cmd_dat2png(argparse.Namespace(rest=[
        "--input-dir", str(src),
        "--output-dir", str(out),
    ]))

    assert rc == 1
    assert (out / "errors.json").exists()


def test_python_m_probeflow_cli_help_works():
    result = subprocess.run(
        [sys.executable, "-m", "probeflow.cli", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "ProbeFlow" in result.stdout


def test_index_folder_script_help_imports_canonical_indexer():
    result = subprocess.run(
        [sys.executable, "scripts/index_folder.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "List recognised SPM files" in result.stdout
