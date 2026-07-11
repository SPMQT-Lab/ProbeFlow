"""Startup dependency check + crash banner with clear terminal messages.

Two failure modes this module addresses (both version-drift, both otherwise
confusing to diagnose):

* A dependency **older** than ProbeFlow needs (typical in conda environments
  that pip's resolver never touched) — errors surface deep inside numpy/Qt
  with no hint that a version is the cause. ``check_environment`` prints
  "ProbeFlow needs numpy >= 1.26; you have 1.24.3" at startup.
* A dependency **newer** than the last verified set — PySide6/numpy releases
  periodically break behaviour. The startup note names the drift, and
  ``install_crash_banner`` appends the environment versions to any unhandled
  crash so a report immediately shows whether versions are the suspect.

The table below must be updated when ``constraints.txt`` is re-verified —
see docs/maintenance.md ("check regularly" list).
"""

from __future__ import annotations

import sys
import traceback
from importlib import metadata

# (minimum required, newest verified) per distribution. Minimums mirror
# pyproject.toml floors; "newest verified" mirrors constraints.txt (compared
# at major.minor, so verified 6.11.0 accepts any 6.11.x patch release).
VERIFIED: dict[str, tuple[str, str]] = {
    "numpy":        ("1.26", "2.1.3"),
    "scipy":        ("1.11", "1.15.3"),
    "pillow":       ("12.2", "12.2.0"),
    "PySide6":      ("6.7",  "6.11.0"),
    "matplotlib":   ("3.8",  "3.10.0"),
    "shapely":      ("2.0",  "2.1.2"),
    "scikit-image": ("0.22", "0.25.0"),
}

PINNED_INSTALL_HINT = (
    "to reproduce the verified environment:  "
    "pip install -e . -c constraints.txt"
)


def _parse(version: str) -> tuple[int, ...]:
    """Lenient numeric version parse: '6.8.1.2' -> (6, 8, 1, 2).

    Non-numeric tails ('2.1.0rc1') stop the parse at the last clean number —
    good enough for the >=/newer-than comparisons made here.
    """
    parts: list[int] = []
    for token in version.split("."):
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _installed_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def check_environment() -> list[str]:
    """Return human-readable warnings for out-of-range dependency versions.

    Empty list means every checked dependency is inside the verified range.
    Missing distributions are skipped (the import failure elsewhere is the
    clearer error for those).
    """
    warnings: list[str] = []
    for dist, (minimum, verified) in VERIFIED.items():
        installed = _installed_version(dist)
        if installed is None:
            continue
        have = _parse(installed)
        if have < _parse(minimum):
            warnings.append(
                f"ProbeFlow needs {dist} >= {minimum}; you have {installed}. "
                f"Upgrade with:  pip install '{dist}>={minimum}'"
            )
        elif have[:2] > _parse(verified)[:2]:
            warnings.append(
                f"{dist} {installed} is newer than the last verified "
                f"{verified} — if you hit errors or UI glitches, this is the "
                f"first suspect ({PINNED_INSTALL_HINT})."
            )
    return warnings


def report_environment(stream=None) -> list[str]:
    """Print ``check_environment`` warnings to *stream* (default stderr)."""
    warnings = check_environment()
    if warnings:
        out = stream if stream is not None else sys.stderr
        print("ProbeFlow environment check:", file=out)
        for line in warnings:
            print(f"  * {line}", file=out)
    return warnings


def _environment_summary() -> str:
    parts = [f"Python {sys.version.split()[0]}"]
    for dist in VERIFIED:
        installed = _installed_version(dist)
        if installed is not None:
            parts.append(f"{dist} {installed}")
    return ", ".join(parts)


_previous_excepthook = None


def install_crash_banner() -> None:
    """Append an environment banner to any unhandled exception (idempotent).

    The original traceback prints unchanged first; the banner then names the
    installed dependency versions and where to report, so a pasted terminal
    capture is a complete bug report.
    """
    global _previous_excepthook
    if _previous_excepthook is not None:
        return
    _previous_excepthook = sys.excepthook

    def _banner_hook(exc_type, exc, tb):
        _previous_excepthook(exc_type, exc, tb)
        print(
            "\nProbeFlow crashed. Environment:\n"
            f"  {_environment_summary()}\n"
            f"If these differ from constraints.txt, a dependency update is "
            f"the first suspect — {PINNED_INSTALL_HINT}\n"
            "Please report this output at "
            "https://github.com/SPMQT-Lab/ProbeFlow/issues",
            file=sys.stderr,
        )

    sys.excepthook = _banner_hook


def format_exception_with_environment(exc: BaseException) -> str:
    """Traceback text plus the environment summary (for GUI error dialogs)."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return f"{tb}\nEnvironment: {_environment_summary()}"
