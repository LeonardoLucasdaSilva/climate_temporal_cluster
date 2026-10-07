"""Reusable Cartesian-space sampling helpers for random search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
import math
import random


ParameterValue = int | float | str | bool | None | tuple[object, ...]


def validated_parameter_space(
    search_space: Mapping[str, Sequence[ParameterValue]],
) -> dict[str, list[ParameterValue]]:
    """Return a non-empty, insertion-ordered parameter space."""
    if not search_space:
        raise ValueError("The random-search parameter space cannot be empty.")
    normalized: dict[str, list[ParameterValue]] = {}
    for parameter, raw_values in search_space.items():
        if isinstance(raw_values, (str, bytes)):
            raise ValueError(f"{parameter} candidates must be a sequence of values.")
        values = list(raw_values)
        if not values:
            raise ValueError(f"{parameter} must contain at least one candidate.")
        if any(values[:index].count(value) for index, value in enumerate(values)):
            raise ValueError(f"{parameter} cannot contain duplicate candidates.")
        normalized[str(parameter)] = values
    return normalized


def parameter_combination_count(
    search_space: Mapping[str, Sequence[ParameterValue]],
) -> int:
    """Return the number of unique Cartesian combinations."""
    validated = validated_parameter_space(search_space)
    return math.prod(len(values) for values in validated.values())


def generate_parameter_configurations(
    search_space: Mapping[str, Sequence[ParameterValue]],
    *,
    strategy: str,
    n_trials: int | None,
    random_state: int,
) -> list[dict[str, ParameterValue]]:
    """Return deterministic grid or without-replacement random candidates."""
    validated = validated_parameter_space(search_space)
    normalized_strategy = str(strategy).strip().lower()
    if normalized_strategy not in {"grid", "random"}:
        raise ValueError("strategy must be either 'grid' or 'random'.")
    if n_trials is not None:
        if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError("n_trials must be a positive integer or None.")

    names = list(validated)
    combinations = [
        dict(zip(names, values))
        for values in product(*(validated[name] for name in names))
    ]
    count = len(combinations) if n_trials is None else min(n_trials, len(combinations))
    if normalized_strategy == "grid":
        return combinations[:count]
    return random.Random(random_state).sample(combinations, k=count)
