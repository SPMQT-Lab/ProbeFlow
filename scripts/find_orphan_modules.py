"""Detect probeflow submodules that are never imported by any other Python file.

Usage (from repo root):
    python scripts/find_orphan_modules.py

A module is "orphaned" when no other .py file in probeflow/ or tests/ contains
an `import` or `from` statement that references it by dotted name.  __init__.py
files are excluded from the orphan list (they are loaded implicitly when their
package is imported).  Known entry-point stubs are also excluded.

Output is grouped:
  ORPHAN   — no inbound references found
  ENTRY    — excluded: loaded by setuptools / CLI, not imported by name
  SKIP     — excluded: __init__.py (implicitly loaded)
"""
from __future__ import annotations

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).parent.parent
PKG  = REPO / "probeflow"

# Files that are entry points (loaded by the runtime, not imported by name).
ENTRY_POINTS = {
    "probeflow/gui/app.py",
    "probeflow/gui/main_window.py",
    "probeflow/cli/parser.py",
    "probeflow/cli/__main__.py",
}

def module_name(path: pathlib.Path) -> str:
    """Convert an absolute path to a dotted module name."""
    rel = path.relative_to(REPO)
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts)


def collect_all_source(exclude: pathlib.Path) -> str:
    """Return concatenated text of all .py files except `exclude`."""
    chunks = []
    for root in (REPO / "probeflow", REPO / "tests"):
        for p in root.rglob("*.py"):
            if p != exclude:
                try:
                    chunks.append(p.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    pass
    return "\n".join(chunks)


def is_referenced(mod_name: str, source_corpus: str) -> bool:
    """Return True if mod_name appears in any import statement in the corpus."""
    # Match:  import probeflow.foo.bar
    #         from probeflow.foo.bar import ...
    #         from probeflow.foo import bar  (partial match on prefix)
    escaped = re.escape(mod_name)
    patterns = [
        rf"(?:^|\s)import\s+{escaped}(?:\s|$|,|#)",
        rf"(?:^|\s)from\s+{escaped}(?:\s+import|\s*$)",
    ]
    for pat in patterns:
        if re.search(pat, source_corpus, re.MULTILINE):
            return True
    return False


def main() -> None:
    candidates = sorted(PKG.rglob("*.py"))

    orphans:  list[tuple[str, str]] = []
    entries:  list[str] = []
    skipped:  list[str] = []

    for path in candidates:
        rel = str(path.relative_to(REPO))
        mod = module_name(path)

        if path.name == "__init__.py":
            skipped.append(rel)
            continue

        if rel in ENTRY_POINTS:
            entries.append(rel)
            continue

        corpus = collect_all_source(exclude=path)
        if not is_referenced(mod, corpus):
            lines = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
            orphans.append((rel, str(lines)))

    print("=" * 60)
    print("ORPHANED MODULES (no inbound import found)")
    print("=" * 60)
    if orphans:
        for rel, lines in orphans:
            print(f"  ORPHAN  {rel:<60} ({lines} lines)")
    else:
        print("  (none found)")

    print()
    print("ENTRY POINTS (excluded from orphan check)")
    print("-" * 60)
    for rel in entries:
        print(f"  ENTRY   {rel}")

    print()
    print(f"Checked {len(candidates)} files; "
          f"{len(orphans)} orphans, {len(entries)} entry points, "
          f"{len(skipped)} __init__.py skipped.")


if __name__ == "__main__":
    sys.exit(main())
