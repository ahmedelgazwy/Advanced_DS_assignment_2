"""Measurement helpers shared by the demo and the benchmark.

Provides the heap classes by display name, a timer that pauses the garbage
collector (as ``timeit`` does) so that collection pauses do not distort the
measurements, and a plain-text table formatter.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence, TypeVar

from mergeable_heaps import HEAP_CLASSES, MergeableHeap

T = TypeVar("T")


def heap_factories() -> dict[str, type[MergeableHeap]]:
    """Every heap class, keyed by display name.

    Returns:
        Ordered mapping from heap name to its class (a zero-argument factory).
    """
    return {cls.name: cls for cls in HEAP_CLASSES}


@contextmanager
def gc_paused() -> Iterator[None]:
    """Disable the cyclic garbage collector for the duration of a block.

    Returns:
        Context manager; the previous GC state is restored on exit.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def measure(action: Callable[[], T]) -> tuple[T, float]:
    """Run an action once with the GC paused and time it.

    Time a whole batch of operations inside ``action`` (for example 100,000
    extract-mins), never a single O(1) operation: one call is too short for
    the timer to resolve.

    Args:
        action: Zero-argument function performing the work to time.

    Returns:
        Tuple ``(result of action, wall-clock seconds)``.
    """
    with gc_paused():
        start = time.perf_counter()
        result = action()
        elapsed = time.perf_counter() - start
    return result, elapsed


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render rows as a plain-text table with aligned columns.

    Args:
        headers: Column titles.
        rows: Table rows; cells are converted with ``str``.

    Returns:
        Multi-line string: the header, a rule, then one line per row.
    """
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(text) for text in column) for column in zip(headers, *cells)]
    lines = [headers, ["-" * width for width in widths], *cells]
    return "\n".join(
        "  ".join(text.ljust(width) for text, width in zip(line, widths)).rstrip() for line in lines
    )
