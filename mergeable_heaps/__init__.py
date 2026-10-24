"""Mergeable heaps implemented from scratch: binomial, Fibonacci and pairing heaps.

All heaps share the :class:`MergeableHeap` interface defined in ``base.py``.
``HEAP_CLASSES`` lists every implementation in display order; the demo, the
benchmarks and the tests iterate over it.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from mergeable_heaps.base import Handle, MergeableHeap
from mergeable_heaps.binomial_heap import BinomialHeap
from mergeable_heaps.fibonacci_heap import FibonacciHeap
from mergeable_heaps.pairing_heap import PairingHeap

#: Every heap implementation, in display order.
HEAP_CLASSES: tuple[type[MergeableHeap], ...] = (BinomialHeap, FibonacciHeap, PairingHeap)

__all__ = [
    "HEAP_CLASSES",
    "BinomialHeap",
    "FibonacciHeap",
    "Handle",
    "MergeableHeap",
    "PairingHeap",
]
