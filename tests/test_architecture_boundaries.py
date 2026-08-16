"""Static tests for the package boundaries documented in docs/architecture.md."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from tests.architecture_policy import (
    ALLOWED_DEPENDENCIES,
    BACKEND_PACKAGES,
    EXCEPTIONS,
    ROOT_PACKAGE,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "probeflow"


@dataclass(frozen=True, order=True)
class ImportEdge:
    source: str
    target: str
    path: str
    line: int
    module: str


def _module_for_path(path: Path) -> tuple[str, bool]:
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = relative.parts
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    return ".".join(parts), is_package


def _owner(module: str) -> str | None:
    parts = module.split(".")
    if parts[0] != "probeflow":
        return None
    return parts[1] if len(parts) > 1 else ROOT_PACKAGE


def _resolve_from_import(node: ast.ImportFrom, source_module: str, is_package: bool) -> str:
    if node.level == 0:
        return node.module or ""

    package = source_module if is_package else source_module.rpartition(".")[0]
    parts = package.split(".")
    parent_count = node.level - 1
    if parent_count:
        parts = parts[:-parent_count]
    if node.module:
        parts.extend(node.module.split("."))
    return ".".join(parts)


def _imports() -> tuple[ImportEdge, ...]:
    edges: set[ImportEdge] = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        source_module, is_package = _module_for_path(path)
        source_owner = _owner(source_module)
        source_path = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [_resolve_from_import(node, source_module, is_package)]
            else:
                continue

            for module in modules:
                target_owner = _owner(module)
                if target_owner is None or target_owner == source_owner:
                    continue
                edges.add(
                    ImportEdge(
                        source=source_owner or ROOT_PACKAGE,
                        target=target_owner,
                        path=source_path,
                        line=node.lineno,
                        module=module,
                    )
                )
    return tuple(sorted(edges))


def _exception_files(source: str, target: str) -> frozenset[str]:
    files: set[str] = set()
    for exception in EXCEPTIONS:
        if exception.source == source and exception.target == target:
            files.update(exception.files)
    return frozenset(files)


def _dependency_violations(edges: tuple[ImportEdge, ...]) -> list[str]:
    violations = []
    for edge in edges:
        allowed_targets = ALLOWED_DEPENDENCIES.get(edge.source)
        if allowed_targets is None:
            violations.append(f"{edge.path}: unclassified package {edge.source!r}")
            continue
        if edge.target in allowed_targets:
            continue
        if edge.path in _exception_files(edge.source, edge.target):
            continue
        violations.append(
            f"{edge.path}:{edge.line}: {edge.source} imports {edge.module} ({edge.target})"
        )
    return violations


def test_package_dependencies_follow_policy() -> None:
    violations = _dependency_violations(_imports())

    assert not violations, "Unapproved package dependencies:\n" + "\n".join(violations)


def test_policy_rejects_synthetic_back_edges() -> None:
    synthetic_edges = (
        ImportEdge("core", "gui", "probeflow/core/new_module.py", 1, "probeflow.gui"),
        ImportEdge("core", "io", "probeflow/core/new_module.py", 2, "probeflow.io"),
    )

    violations = _dependency_violations(synthetic_edges)

    assert len(violations) == 2
    assert all("new_module.py" in violation for violation in violations)


def test_dependency_exceptions_are_current_and_documented() -> None:
    edges = _imports()
    stale = []
    incomplete = []
    for exception in EXCEPTIONS:
        if not exception.reason.strip() or not exception.removal.strip():
            incomplete.append(f"{exception.source} -> {exception.target}")
        for path in exception.files:
            if not (REPO_ROOT / path).is_file():
                stale.append(f"missing file: {path}")
                continue
            if not any(
                edge.source == exception.source
                and edge.target == exception.target
                and edge.path == path
                for edge in edges
            ):
                stale.append(f"unused exception: {path}: {exception.source} -> {exception.target}")

    assert not incomplete, "Undocumented dependency exceptions:\n" + "\n".join(incomplete)
    assert not stale, "Stale dependency exceptions:\n" + "\n".join(stale)


def test_backend_packages_do_not_import_interfaces() -> None:
    violations = [
        edge
        for edge in _imports()
        if edge.source in BACKEND_PACKAGES and edge.target in {"gui", "cli"}
    ]
    assert not violations, f"Backend packages import interface code: {violations}"


def test_pyside6_imports_are_confined_to_gui() -> None:
    violations = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        source_module, _ = _module_for_path(path)
        if _owner(source_module) == "gui":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""] if node.level == 0 else []
            else:
                continue
            if any(module == "PySide6" or module.startswith("PySide6.") for module in modules):
                relative = path.relative_to(REPO_ROOT).as_posix()
                violations.append(f"{relative}:{node.lineno}")

    assert not violations, "PySide6 imports outside probeflow.gui:\n" + "\n".join(violations)
