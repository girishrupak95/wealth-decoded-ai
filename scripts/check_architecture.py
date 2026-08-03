"""Lightweight static architecture audit without importing project modules."""

import ast
from collections.abc import Iterable
from pathlib import Path

PROJECT_PACKAGES = ("agents", "shared", "apps/api/app")
DOCUMENTED_BOUNDARY_EXCEPTIONS = {
    "shared/ai/openai_client.py: shared-to-application import app.config.settings"
}


def module_name(root: Path, path: Path) -> str:
    """Return the importable module name for an audited source file."""
    relative = path.relative_to(root)
    parts = list(relative.with_suffix("").parts)
    if parts[:3] == ["apps", "api", "app"]:
        parts = ["app", *parts[3:]]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def python_files(root: Path) -> Iterable[Path]:
    """Yield production Python files covered by the static import audit."""
    for package in PROJECT_PACKAGES:
        for path in (root / package).rglob("*.py"):
            if "tests" not in path.relative_to(root).parts:
                yield path


def imports_for(path: Path) -> set[str]:
    """Extract absolute project imports without importing or executing code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module)
    return imports


def dependency_graph(root: Path) -> dict[str, set[str]]:
    """Build a graph only among repository modules that can form cycles."""
    modules = {module_name(root, path) for path in python_files(root)}
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for path in python_files(root):
        source = module_name(root, path)
        for imported in imports_for(path):
            if imported in modules:
                graph[source].add(imported)
    return graph


def circular_dependencies(graph: dict[str, set[str]]) -> list[list[str]]:
    """Return stable cycle paths discovered by depth-first traversal."""
    visited: set[str] = set()
    active: list[str] = []
    cycles: list[list[str]] = []

    def visit(module: str) -> None:
        if module in active:
            cycles.append([*active[active.index(module) :], module])
            return
        if module in visited:
            return
        visited.add(module)
        active.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        active.pop()

    for module in sorted(graph):
        visit(module)
    return cycles


def boundary_violations(root: Path) -> list[str]:
    """Check stable one-way package boundaries using source-level imports."""
    violations: list[str] = []
    for path in python_files(root):
        relative = path.relative_to(root)
        imports = imports_for(path)
        if relative.parts[0] == "agents":
            agent_name = relative.parts[1] if len(relative.parts) > 1 else ""
            for imported in imports:
                if imported.startswith("agents.") and not imported.startswith(
                    f"agents.{agent_name}."
                ):
                    violations.append(f"{relative}: cross-agent import {imported}")
        if relative.parts[0] == "shared":
            for imported in imports:
                if imported == "app" or imported.startswith("app.") or imported.startswith("apps."):
                    violations.append(f"{relative}: shared-to-application import {imported}")
    return violations


def main() -> int:
    """Print actionable audit results and return a conventional process status."""
    root = Path(__file__).resolve().parents[1]
    violations = boundary_violations(root)
    unexpected_violations = [
        violation for violation in violations if violation not in DOCUMENTED_BOUNDARY_EXCEPTIONS
    ]
    cycles = circular_dependencies(dependency_graph(root))
    if violations:
        print("Boundary violations:")
        print("\n".join(f"- {violation}" for violation in violations))
    if cycles:
        print("Circular imports:")
        print("\n".join(f"- {' -> '.join(cycle)}" for cycle in cycles))
    if not unexpected_violations and not cycles:
        if violations:
            print("Architecture audit passed with documented boundary exceptions.")
        else:
            print("Architecture audit passed: no boundary violations or circular imports found.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
