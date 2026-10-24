"""A deliberately simple mergeable heap that validates the contract tests.

Items live in an unsorted Python list and ``find_min`` scans it, so every
operation is obviously correct (and slow: O(n)). The contract tests in
``test_common.py`` run against this class as well as against the real heaps.
That shows the tests themselves are right, and they could run before any real
heap existed. Test-only code: never benchmarked.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import sys
from operator import attrgetter
from typing import Any, Iterator

from mergeable_heaps import MergeableHeap

_BY_KEY = attrgetter("key")


class ReferenceEntry:
    """One item of the reference heap; also serves as its handle."""

    __slots__ = ("key", "value", "in_heap")

    def __init__(self, key: Any, value: Any) -> None:
        """Create a live entry.

        Args:
            key: Priority of the item.
            value: Payload stored with the key.
        """
        self.key = key
        self.value = value
        self.in_heap = True


class ReferenceHeap(MergeableHeap):
    """Mergeable heap backed by an unsorted list (O(n) find-min)."""

    name = "Reference (list)"

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        """Create an empty heap."""
        self._entries: list[ReferenceEntry] = []

    def insert(self, key: Any, value: Any = None) -> ReferenceEntry:
        """Append a new entry. See :meth:`MergeableHeap.insert`."""
        entry = ReferenceEntry(key, value)
        self._entries.append(entry)
        return entry

    def find_min(self) -> ReferenceEntry:
        """Scan for the smallest key. See :meth:`MergeableHeap.find_min`."""
        if not self._entries:
            raise IndexError("find_min from an empty heap")
        return min(self._entries, key=_BY_KEY)

    def extract_min(self) -> tuple[Any, Any]:
        """Remove the entry with the smallest key. See :meth:`MergeableHeap.extract_min`."""
        if not self._entries:
            raise IndexError("extract_min from an empty heap")
        entry = min(self._entries, key=_BY_KEY)
        self._remove(entry)
        return entry.key, entry.value

    def decrease_key(self, handle: ReferenceEntry, new_key: Any) -> None:
        """Overwrite the key. See :meth:`MergeableHeap.decrease_key`."""
        self._check_live(handle)
        self._check_new_key(handle.key, new_key)
        handle.key = new_key

    def delete(self, handle: ReferenceEntry) -> None:
        """Remove the entry. See :meth:`MergeableHeap.delete`."""
        self._check_live(handle)
        self._remove(handle)

    def merge(self, other: ReferenceHeap) -> None:
        """Take over ``other``'s entries. See :meth:`MergeableHeap.merge`."""
        self._check_merge_partner(other)
        self._entries.extend(other._entries)
        other._entries = []

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self._entries)

    def check_invariants(self) -> None:
        """Every stored entry is live and stored once. See :meth:`MergeableHeap.check_invariants`."""
        assert all(entry.in_heap for entry in self._entries)
        assert len({id(entry) for entry in self._entries}) == len(self._entries)

    def memory_bytes(self) -> int:
        """Entries and keys plus the list itself. See :meth:`MergeableHeap.memory_bytes`."""
        return super().memory_bytes() + sys.getsizeof(self._entries)

    def _iter_nodes(self) -> Iterator[ReferenceEntry]:
        """Yield every entry. See :meth:`MergeableHeap._iter_nodes`."""
        return iter(self._entries)

    def _remove(self, entry: ReferenceEntry) -> None:
        """Take an entry out of the list and mark it removed.

        Args:
            entry: Live entry of this heap.

        Returns:
            None.
        """
        self._entries.remove(entry)  # entries have no __eq__, so this matches by identity
        entry.in_heap = False

    @staticmethod
    def _check_live(handle: ReferenceEntry) -> None:
        """Reject handles of removed items.

        Args:
            handle: Entry passed by the caller.

        Returns:
            None.

        Raises:
            ValueError: If the entry is no longer in a heap.
        """
        if not handle.in_heap:
            raise ValueError("the item is no longer in the heap")
