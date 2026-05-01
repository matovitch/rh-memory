"""Path helpers for repository scripts."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def project_relative_path(path: str | Path) -> str:
    path = resolve_project_path(path)
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return str(path)
