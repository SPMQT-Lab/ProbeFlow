"""Tests for the startup dependency check and crash banner."""

from __future__ import annotations

import io
import sys

from probeflow.core import env_check
from probeflow.core.env_check import (
    VERIFIED,
    _parse,
    check_environment,
    format_exception_with_environment,
    report_environment,
)


def test_parse_is_lenient():
    assert _parse("2.1.3") == (2, 1, 3)
    assert _parse("6.11.0") == (6, 11, 0)
    assert _parse("2.1.0rc1") == (2, 1, 0)
    assert _parse("weird") == ()


def test_too_old_dependency_gets_clear_upgrade_message(monkeypatch):
    monkeypatch.setattr(
        env_check, "_installed_version",
        lambda dist: "1.24.3" if dist == "numpy" else None,
    )
    warnings = check_environment()
    assert len(warnings) == 1
    assert "ProbeFlow needs numpy >= 1.26; you have 1.24.3" in warnings[0]
    assert "pip install 'numpy>=1.26'" in warnings[0]


def test_newer_than_verified_gets_first_suspect_note(monkeypatch):
    monkeypatch.setattr(
        env_check, "_installed_version",
        lambda dist: "7.2.0" if dist == "PySide6" else None,
    )
    warnings = check_environment()
    assert len(warnings) == 1
    assert "PySide6 7.2.0 is newer than the last verified 6.11.0" in warnings[0]
    assert "constraints.txt" in warnings[0]


def test_verified_patch_release_is_silent(monkeypatch):
    # Same major.minor as the verified pin (a patch release) must not warn.
    monkeypatch.setattr(
        env_check, "_installed_version",
        lambda dist: "2.1.9" if dist == "numpy" else None,
    )
    assert check_environment() == []


def test_missing_distribution_is_skipped(monkeypatch):
    monkeypatch.setattr(env_check, "_installed_version", lambda dist: None)
    assert check_environment() == []


def test_report_prints_only_when_warnings_exist(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(env_check, "_installed_version", lambda dist: None)
    assert report_environment(stream) == []
    assert stream.getvalue() == ""

    monkeypatch.setattr(
        env_check, "_installed_version",
        lambda dist: "1.0" if dist == "numpy" else None,
    )
    warnings = report_environment(stream)
    assert warnings
    text = stream.getvalue()
    assert text.startswith("ProbeFlow environment check:")
    assert "you have 1.0" in text


def test_current_environment_is_inside_verified_range():
    """The dev/CI environment itself must never trip the too-old check."""
    for warning in check_environment():
        assert "ProbeFlow needs" not in warning, warning


def test_crash_banner_chains_previous_hook_and_names_environment(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(env_check, "_previous_excepthook", None)
    monkeypatch.setattr(sys, "excepthook", lambda t, e, tb: calls.append(e))
    env_check.install_crash_banner()
    try:
        # Idempotent: a second install keeps the first chain.
        hook = sys.excepthook
        env_check.install_crash_banner()
        assert sys.excepthook is hook

        err = RuntimeError("boom")
        sys.excepthook(RuntimeError, err, None)
        assert calls == [err]  # original hook still ran
        banner = capsys.readouterr().err
        assert "ProbeFlow crashed. Environment:" in banner
        assert "Python" in banner
        assert "constraints.txt" in banner
        assert "github.com/SPMQT-Lab/ProbeFlow/issues" in banner
    finally:
        monkeypatch.setattr(env_check, "_previous_excepthook", None)


def test_format_exception_with_environment():
    try:
        raise ValueError("bad input")
    except ValueError as exc:
        text = format_exception_with_environment(exc)
    assert "ValueError: bad input" in text
    assert "Environment: Python" in text


def test_verified_table_matches_pyproject_floors():
    """The env-check minimums must not drift from pyproject.toml."""
    from pathlib import Path
    import re

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    deps_block = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    floors = dict(re.findall(r'"([A-Za-z0-9_-]+)>=([0-9.]+)', deps_block))
    for dist, (minimum, _verified) in VERIFIED.items():
        if dist in floors:
            assert floors[dist] == minimum, (
                f"{dist}: env_check minimum {minimum} != pyproject floor "
                f"{floors[dist]} — update VERIFIED in env_check.py"
            )


def test_verified_table_matches_constraints_pins():
    """The env-check 'newest verified' must not drift from constraints.txt."""
    from pathlib import Path
    import re

    constraints = (Path(__file__).resolve().parents[1] / "constraints.txt").read_text()
    pins = dict(re.findall(r"^([A-Za-z0-9_-]+)==([0-9.]+)", constraints, re.MULTILINE))
    for dist, (_minimum, verified) in VERIFIED.items():
        if dist in pins:
            assert pins[dist] == verified, (
                f"{dist}: env_check verified {verified} != constraints pin "
                f"{pins[dist]} — update VERIFIED in env_check.py"
            )
