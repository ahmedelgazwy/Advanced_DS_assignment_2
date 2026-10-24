"""Pairing heap with two-pass pairing [Fredman, Sedgewick, Sleator & Tarjan 1986].

A pairing heap is a single heap-ordered multiway tree. Its only primitive is
*link*: compare two roots and make the larger one the leftmost child of the
smaller. insert, merge and decrease_key are one link each. All the
restructuring happens in extract_min, which removes the root and combines its
children by *two-pass pairing*: link them in pairs from left to right, then
link the resulting trees one by one from right to left.

Amortized costs (see docs/theory.md, section 3): insert and merge O(1)
[Iacono 2000]; extract_min and delete O(log n); decrease_key between
Omega(log log n) [Fredman 1999] and O(2^(2 sqrt(log log n))) [Pettie 2005].
A single extract_min can cost O(n), for example right after n inserts.

Each node stores its leftmost ``child``, its right sibling ``next`` and
``prev``, which is the left sibling or, for a leftmost child, the parent. The
``prev`` pointer lets decrease_key and delete cut a subtree out in O(1). Each
node is its own handle; a removed node points ``prev`` at itself, which is how
stale handles are recognised.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from mergeable_heaps.base import MergeableHeap


class PairingNode:
    """A node of a pairing heap; this object is also the caller's handle."""

    __slots__ = ("key", "value", "child", "next", "prev")

    def __init__(self, key: Any, value: Any) -> None:
        """Create a one-node tree.

        Args:
            key: Priority of the item.
            value: Payload stored with the key.
        """
        self.key = key
        self.value = value
        self.child: Optional[PairingNode] = None  # leftmost child
        self.next: Optional[PairingNode] = None  # right sibling
        self.prev: Optional[PairingNode] = None  # left sibling, or parent if leftmost


def _link(first: PairingNode, second: PairingNode) -> PairingNode:
    """Link two detached roots: the larger key becomes the leftmost child of the smaller.

    Both roots must have ``prev`` and ``next`` set to None. On equal keys
    ``first`` stays on top.

    Args:
        first: Root of the first tree.
        second: Root of the second tree.

    Returns:
        The root of the combined tree.
    """
    if second.key < first.key:
        first, second = second, first
    leftmost = first.child
    second.next = leftmost
    if leftmost is not None:
        leftmost.prev = second
    second.prev = first
    first.child = second
    return first


def _two_pass_pairing(first: Optional[PairingNode]) -> Optional[PairingNode]:
    """Combine a list of sibling trees into one tree (two-pass pairing).

    Pass 1 links the trees in pairs from left to right; pass 2 links the
    results from right to left into one accumulated tree. Iterative: the
    sibling list can be as long as n - 1.

    Args:
        first: Leftmost tree of a sibling list (may be None).

    Returns:
        Root of the combined tree, with ``prev`` and ``next`` set to None, or
        None if the list was empty.
    """
    pairs: list[PairingNode] = []
    node = first
    while node is not None:
        second = node.next
        if second is None:
            node.prev = None
            pairs.append(node)
            break
        following = second.next
        node.prev = node.next = None
        second.prev = second.next = None
        pairs.append(_link(node, second))
        node = following
    if not pairs:
        return None
    result = pairs.pop()
    while pairs:
        result = _link(pairs.pop(), result)
    return result


class PairingHeap(MergeableHeap):
    """Mergeable min-heap stored as one multiway tree, restructured by two-pass pairing."""

    name = "Pairing heap"

    __slots__ = ("_root", "_size")

    def __init__(self) -> None:
        """Create an empty heap."""
        self._root: Optional[PairingNode] = None
        self._size = 0

    def insert(self, key: Any, value: Any = None) -> PairingNode:
        """Link a new one-node tree with the root. O(1).

        Args:
            key: Priority of the item.
            value: Payload stored with the key.

        Returns:
            The new node, which is its handle.
        """
        node = PairingNode(key, value)
        root = self._root
        self._root = node if root is None else _link(root, node)
        self._size += 1
        return node

    def find_min(self) -> PairingNode:
        """Return the root. See :meth:`MergeableHeap.find_min`.

        Returns:
            Handle of a minimum item, in O(1).

        Raises:
            IndexError: If the heap is empty.
        """
        if self._root is None:
            raise IndexError("find_min from an empty heap")
        return self._root

    def extract_min(self) -> tuple[Any, Any]:
        """Remove the root and combine its children by two-pass pairing.

        O(log n) amortized; O(number of children of the root) actual.

        Returns:
            Tuple ``(key, value)`` of the removed item.

        Raises:
            IndexError: If the heap is empty.
        """
        root = self._root
        if root is None:
            raise IndexError("extract_min from an empty heap")
        self._root = _two_pass_pairing(root.child)
        self._size -= 1
        self._mark_removed(root)
        return root.key, root.value

    def decrease_key(self, handle: PairingNode, new_key: Any) -> None:
        """Lower a key; cut the node's subtree out and link it with the root.

        O(1) actual; the amortized cost is between Omega(log log n) and
        O(2^(2 sqrt(log log n))) because later extract-mins pay for the cut.

        Args:
            handle: Node returned by :meth:`insert`.
            new_key: New key, not larger than the current one.

        Returns:
            None.

        Raises:
            ValueError: If ``new_key`` is larger, or the node was removed.
        """
        self._check_live(handle)
        self._check_new_key(handle.key, new_key)
        decreased = new_key < handle.key
        handle.key = new_key
        if decreased and handle is not self._root:
            self._detach(handle)
            self._root = _link(self._root, handle)

    def delete(self, handle: PairingNode) -> None:
        """Remove any node: cut its subtree, pair its children, link back. O(log n) amortized.

        Args:
            handle: Node returned by :meth:`insert`.

        Returns:
            None.

        Raises:
            ValueError: If the node was already removed.
        """
        self._check_live(handle)
        if handle is self._root:
            self.extract_min()
            return
        self._detach(handle)
        subtree = _two_pass_pairing(handle.child)
        if subtree is not None:
            self._root = _link(self._root, subtree)
        self._size -= 1
        self._mark_removed(handle)

    def merge(self, other: PairingHeap) -> None:
        """Link the two roots. O(1).

        Args:
            other: Pairing heap whose items move into this heap; left empty.

        Returns:
            None.

        Raises:
            TypeError: If ``other`` is not a PairingHeap.
            ValueError: If ``other`` is this heap.
        """
        self._check_merge_partner(other)
        if other._root is not None:
            self._root = other._root if self._root is None else _link(self._root, other._root)
        self._size += other._size
        other._root, other._size = None, 0

    def __len__(self) -> int:
        """Return the number of items in the heap."""
        return self._size

    def root_degree(self) -> int:
        """Number of children of the root (0 for an empty heap).

        Returns:
            Length of the root's child list; this is the work the next
            extract_min must do.
        """
        count = 0
        child = self._root.child if self._root is not None else None
        while child is not None:
            count += 1
            child = child.next
        return count

    def check_invariants(self) -> None:
        """Verify the pairing-heap invariants. O(n).

        Checks: the root has no siblings and no ``prev``; every leftmost child
        points back to its parent and every other child to its left sibling;
        heap order; no live node is marked removed; the node count equals the
        stored size.

        Returns:
            None.

        Raises:
            AssertionError: If an invariant is broken.
        """
        root = self._root
        if root is None:
            assert self._size == 0, "empty heap but non-zero size"
            return
        assert root.prev is None and root.next is None, "the root has a sibling or a prev pointer"
        count = 0
        stack = [root]
        while stack:
            node = stack.pop()
            count += 1
            assert count <= self._size, "more nodes than the stored size (a cycle?)"
            assert node.prev is not node, "a live node is marked as removed"
            previous = node
            child = node.child
            while child is not None:
                assert child.prev is previous, "prev pointer is not the parent or left sibling"
                assert not child.key < node.key, "heap order violated"
                stack.append(child)
                previous, child = child, child.next
        assert count == self._size, "node count differs from the stored size"

    def _iter_nodes(self) -> Iterator[PairingNode]:
        """Yield every node once (depth-first, with an explicit stack).

        Returns:
            Iterator over all nodes of the tree.
        """
        stack = [self._root] if self._root is not None else []
        while stack:
            node = stack.pop()
            yield node
            child = node.child
            while child is not None:
                stack.append(child)
                child = child.next

    @staticmethod
    def _detach(node: PairingNode) -> None:
        """Cut a non-root node (with its subtree) out of its sibling list. O(1).

        Args:
            node: A node that is not the root.

        Returns:
            None.
        """
        previous, following = node.prev, node.next
        if previous.child is node:  # leftmost child: ``previous`` is the parent
            previous.child = following
        else:
            previous.next = following
        if following is not None:
            following.prev = previous
        node.prev = node.next = None

    @staticmethod
    def _mark_removed(node: PairingNode) -> None:
        """Clear a removed node's links and mark it with ``prev = node``.

        Args:
            node: Node that has just been taken out of the heap.

        Returns:
            None.
        """
        node.child = node.next = None
        node.prev = node

    @staticmethod
    def _check_live(handle: PairingNode) -> None:
        """Reject handles of removed nodes.

        Args:
            handle: Node passed by the caller.

        Returns:
            None.

        Raises:
            ValueError: If the node is no longer in a heap.
        """
        if handle.prev is handle:
            raise ValueError("the item is no longer in the heap")
