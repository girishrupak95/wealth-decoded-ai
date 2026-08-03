"""Stable source-level boundaries for the frozen v1 architecture."""

from pathlib import Path

from scripts.check_architecture import (
    DOCUMENTED_BOUNDARY_EXCEPTIONS,
    boundary_violations,
    circular_dependencies,
    dependency_graph,
)

ROOT = Path(__file__).parents[1]


def test_production_packages_follow_dependency_boundaries() -> None:
    """New boundary violations cannot silently extend the documented architecture exception."""
    assert set(boundary_violations(ROOT)) == DOCUMENTED_BOUNDARY_EXCEPTIONS


def test_production_packages_have_no_circular_imports() -> None:
    """The static production import graph is acyclic."""
    assert circular_dependencies(dependency_graph(ROOT)) == []


def test_sensitive_and_generated_paths_are_ignored() -> None:
    """Repository policy keeps local secrets, environments, and generated media untracked."""
    entries = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert {".env", ".venv/", "generated/"}.issubset(entries)
