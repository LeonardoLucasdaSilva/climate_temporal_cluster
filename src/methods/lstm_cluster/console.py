"""Console output helpers for LSTM cluster experiments."""

from __future__ import annotations


def print_info(message: str, enabled: bool) -> None:
    """Print a progress message when console output is enabled."""
    if enabled:
        print(message)


def print_section(title: str, enabled: bool) -> None:
    """Print a readable section header when console output is enabled."""
    if not enabled:
        return

    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
