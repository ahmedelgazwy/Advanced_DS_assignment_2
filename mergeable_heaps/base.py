"""Common interface shared by every mergeable heap.

The binomial, Fibonacci and pairing heaps all implement the same abstract data
type, a *mergeable min-heap* of ``(key, value)`` items, so they all subclass
:class:`MergeableHeap`. The demo, the benchmarks, Dijkstra's algorithm and the
shared unit tests only talk to this interface, which is what lets them treat
the three heaps interchangeably.

``insert`` returns a *handle* to the new item. The caller keeps it to lower
the item's key later (``decrease_key``) or to remove the item (``delete``).

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, Protocol


class Handle(Protocol):
    """What a caller may read from the handle returned by ``insert``.

    Handles are read-only for callers: change a key only through
    :meth:`MergeableHeap.decrease_key`.
    """

    key: Any
    value: Any


class MergeableHeap(ABC):
    """Abstract mergeable min-heap of ``(key, value)`` items.

    Keys can be of any type whose values can be compared with ``<`` (ints,
    floats, strings, tuples, ...). Several items may share the same key; ties
    are broken arbitrarily. Values are carried along untouched.

    Every handle must belong to an item that is still in *this* heap (or in a
    heap that was merged into this one). Handles of items that were already
    removed are detected and rejected with ``ValueError``.
    """

    #: Human-readable name used in tables and plots.
    name: str = "MergeableHeap"

    __slots__ = ()

    @abstractmethod
    def insert(self, key: Any, value: Any = None) -> Handle:
        """Add an item to the heap.

        Args:
            key: Priority of the item (smaller keys come out first).
            value: Payload stored with the key (for example a vertex id).

        Returns:
            Handle to the new item, for ``decrease_key`` and ``delete``.
        """

    @abstractmethod
    def find_min(self) -> Handle:
        """Return an item with the smallest key without removing it.

        Returns:
            Handle to a minimum item.

        Raises:
            IndexError: If the heap is empty.
        """

    @abstractmethod
    def extract_min(self) -> tuple[Any, Any]:
        """Remove an item with the smallest key and return it.

        Returns:
            Tuple ``(key, value)`` of the removed item.

        Raises:
            IndexError: If the heap is empty.
        """

    @abstractmethod
    def decrease_key(self, handle: Handle, new_key: Any) -> None:
        """Lower the key of an item that is in the heap.

        Args:
            handle: Handle returned by ``insert`` for this item.
            new_key: New key; must not be larger than the current key
                (an equal key is allowed and changes nothing).

        Returns:
            None.

        Raises:
            ValueError: If ``new_key`` is larger than the current key, or the
                item was already removed from the heap.
        """

    @abstractmethod
    def delete(self, handle: Handle) -> None:
        """Remove an arbitrary item from the heap.

        Args:
            handle: Handle returned by ``insert`` for this item.

        Returns:
            None.

        Raises:
            ValueError: If the item was already removed from the heap.
        """

    @abstractmethod
    def merge(self, other: MergeableHeap) -> None:
        """Move every item of ``other`` into this heap (destructive union).

        Handles of ``other``'s items stay valid and now refer to items of this
        heap. ``other`` is left empty and can be reused.

        Args:
            other: Heap of the same class as this one.

        Returns:
            None.

        Raises:
            TypeError: If ``other`` is not of the same class as this heap.
            ValueError: If ``other`` is this heap itself.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the heap."""

    @abstractmethod
    def check_invariants(self) -> None:
        """Verify every structural invariant of the heap (used by the tests).

        Walks the whole structure, so it costs O(n).

        Returns:
            None.

        Raises:
            AssertionError: If an invariant is broken (for example heap order,
                a pointer that does not point back, or a wrong item count).
        """

    @abstractmethod
    def _iter_nodes(self) -> Iterator[Any]:
        """Yield every node of the heap exactly once, without recursion.

        Returns:
            Iterator over the node objects (one per item).
        """

    def _node_bytes(self, node: Any) -> int:
        """Bytes owned by one node: the node object itself and its key.

        Values are not counted: they belong to the caller and are often shared.
        Subclasses with more objects per item override this.

        Args:
            node: A node yielded by :meth:`_iter_nodes`.

        Returns:
            Size in bytes as reported by ``sys.getsizeof``.
        """
        return sys.getsizeof(node) + sys.getsizeof(node.key)

    def memory_bytes(self) -> int:
        """Estimate the memory used by the heap.

        Returns:
            Approximate size in bytes of the heap object, every node and every
            key (``sys.getsizeof`` of each object; values are not counted).
        """
        return sys.getsizeof(self) + sum(self._node_bytes(node) for node in self._iter_nodes())

    def __bool__(self) -> bool:
        """Return True if the heap contains at least one item."""
        return len(self) > 0

    def __repr__(self) -> str:
        """Return a short description such as ``Pairing heap(n=3)``."""
        return f"{self.name}(n={len(self)})"

    def insert_all(self, keys: Iterable[Any]) -> list[Handle]:
        """Insert many keys (each with value None), keeping their handles.

        Args:
            keys: Keys to insert, in order.

        Returns:
            List of handles, one per key, in the same order.
        """
        insert = self.insert
        return [insert(key) for key in keys]

    def drain(self) -> Iterator[tuple[Any, Any]]:
        """Remove every item in increasing key order (heap sort).

        Returns:
            Iterator of ``(key, value)`` pairs; the heap is empty once the
            iterator is exhausted.
        """
        while self:
            yield self.extract_min()

    def _check_merge_partner(self, other: MergeableHeap) -> None:
        """Validate the argument of :meth:`merge`.

        Args:
            other: Heap that the caller wants to merge into this one.

        Returns:
            None.

        Raises:
            TypeError: If ``other`` is not of exactly the same class.
            ValueError: If ``other`` is this heap itself.
        """
        if type(other) is not type(self):
            raise TypeError(f"cannot merge {type(other).__name__} into {type(self).__name__}")
        if other is self:
            raise ValueError("cannot merge a heap into itself")

    @staticmethod
    def _check_new_key(current_key: Any, new_key: Any) -> None:
        """Validate the argument of :meth:`decrease_key`.

        Args:
            current_key: Key the item has now.
            new_key: Requested key.

        Returns:
            None.

        Raises:
            ValueError: If ``new_key`` is larger than ``current_key``.
        """
        if current_key < new_key:
            raise ValueError(f"new key {new_key!r} is larger than the current key {current_key!r}")
