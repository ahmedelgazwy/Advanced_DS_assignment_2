"""Mergeable heaps implemented from scratch: binomial, Fibonacci and pairing heaps.

All heaps share the :class:`MergeableHeap` interface defined in ``base.py``.
``HEAP_CLASSES`` lists every implementation in display order; the demo, the
benchmarks and the tests iterate over it.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from mergeable_heaps.base import Handle, MergeableHeap

#: Every heap implementation, in display order (filled in as the heaps are added).
HEAP_CLASSES: tuple[type[MergeableHeap], ...] = ()

__all__ = ["HEAP_CLASSES", "Handle", "MergeableHeap"]
