"""Cleanup helpers for generated experiment output trees."""

from __future__ import annotations

from pathlib import Path


def remove_empty_directories(root: Path) -> int:
    """Remove empty directories below ``root`` and return the removal count."""
    root = Path(root)
    if not root.is_dir():
        return 0

    removed = 0
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            continue
        removed += 1
    return removed
