"""Shared pytest fixtures: heap factories and sample keys.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import random

import pytest

from mergeable_heaps import HEAP_CLASSES, MergeableHeap
from reference_heap import ReferenceHeap

NUM_KEYS = 2_000

#: The real heaps, keyed by class name (used as the pytest id).
HEAP_FACTORIES: dict[str, type[MergeableHeap]] = {cls.__name__: cls for cls in HEAP_CLASSES}
#: The real heaps plus the reference heap that validates the contract tests.
CONTRACT_FACTORIES: dict[str, type[MergeableHeap]] = {
    ReferenceHeap.__name__: ReferenceHeap,
    **HEAP_FACTORIES,
}


@pytest.fixture(params=list(CONTRACT_FACTORIES))
def make_heap(request: pytest.FixtureRequest) -> type[MergeableHeap]:
    """Factory for every heap, reference included (the test runs once per heap)."""
    return CONTRACT_FACTORIES[request.param]


@pytest.fixture(params=list(HEAP_FACTORIES))
def make_real_heap(request: pytest.FixtureRequest) -> type[MergeableHeap]:
    """Factory for every real heap, for stress tests the O(n) reference is too slow for."""
    return HEAP_FACTORIES[request.param]


@pytest.fixture(scope="session")
def random_keys() -> list[int]:
    """``NUM_KEYS`` random keys in [0, 1,000,000) (do not mutate: shared by all tests)."""
    rng = random.Random(520)
    return [rng.randrange(1_000_000) for _ in range(NUM_KEYS)]
