from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from path_utils import project_relative_path, resolve_project_path


def test_resolve_project_path_keeps_absolute_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"

    assert resolve_project_path(checkpoint) == checkpoint


def test_resolve_project_path_uses_project_root_for_relative_path() -> None:
    resolved = resolve_project_path(Path("scripts/checkpoints/checkpoint.pt"))

    assert resolved == Path(__file__).resolve().parents[1] / "scripts" / "checkpoints" / "checkpoint.pt"


def test_project_relative_path_preserves_external_absolute_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"

    assert project_relative_path(checkpoint) == str(checkpoint)


def test_project_relative_path_relativizes_project_path() -> None:
    checkpoint = Path(__file__).resolve().parents[1] / "scripts" / "checkpoints" / "checkpoint.pt"

    assert project_relative_path(checkpoint) == "scripts/checkpoints/checkpoint.pt"
