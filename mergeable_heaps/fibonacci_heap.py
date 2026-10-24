"""Fibonacci heap [Fredman & Tarjan 1987; CLRS 3rd ed., ch. 19].

A Fibonacci heap is a collection of heap-ordered trees whose roots form a
circular doubly linked *root list*, with a pointer to the minimum root. It is
lazy: insert and merge only splice lists in O(1), and all tidying is deferred
to extract_min, which *consolidates* the root list until no two roots have the
same degree. decrease_key cuts a node out of its tree; a node that loses a
second child is cut too (*cascading cut*), which keeps every subtree of a
degree-k node at least F_{k+2} >= phi^k nodes large, so degrees stay
O(log n).

Amortized costs (see docs/theory.md, section 2): insert, find_min, merge and
decrease_key O(1); extract_min and delete O(log n). A single extract_min can
cost O(n), for example right after n inserts.

Each node is its own handle. A removed node has ``left = None``, which is how
stale handles are recognised.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from mergeable_heaps.base import MergeableHeap


class FibonacciNode:
    """A node of a Fibonacci heap; this object is also the caller's handle.

    ``left``/``right`` link the node into a circular list of siblings (or of
    roots); ``child`` points to any one child. ``mark`` records that the node
    has lost a child since it last became a child of another node.
    """

    __slots__ = ("key", "value", "parent", "child", "left", "right", "degree", "mark")

    def __init__(self, key: Any, value: Any) -> None:
        """Create a single node that forms a one-element circular list.

        Args:
            key: Priority of the item.
            value: Payload stored with the key.
        """
        self.key = key
        self.value = value
        self.parent: Optional[FibonacciNode] = None
        self.child: Optional[FibonacciNode] = None
        self.left: Optional[FibonacciNode] = self
        self.right: Optional[FibonacciNode] = self
        self.degree = 0
        self.mark = False


def _splice(first: FibonacciNode, second: FibonacciNode) -> None:
    """Concatenate two circular lists: ``second``'s list goes right after ``first``.

    Args:
        first: Any node of the first list.
        second: Any node of the second list (a different list).

    Returns:
        None.
    """
    first_next = first.right
    second_last = second.left
    first.right = second
    second.left = first
    second_last.right = first_next
    first_next.left = second_last


def _unlink(node: FibonacciNode) -> None:
    """Remove a node from its circular list, leaving it as a one-element list.

    Args:
        node: Node to remove.

    Returns:
        None.
    """
    node.left.right = node.right
    node.right.left = node.left
    node.left = node.right = node


def _siblings(start: FibonacciNode) -> Iterator[FibonacciNode]:
    """Yield every node of a circular list, starting at ``start``.

    The list must not be changed while it is being iterated.

    Args:
        start: Any node of the list.

    Returns:
        Iterator over the nodes of the list.
    """
    node = start
    while True:
        yield node
        node = node.right
        if node is start:
            return


def max_degree_bound(n: int) -> int:
    """Largest degree D(n) any node of an n-item Fibonacci heap can have.

    A node of degree k roots a subtree of at least F_{k+2} nodes (CLRS Lemma
    19.4), so D(n) = max{k : F_{k+2} <= n}, which is at most floor(log_phi n)
    (Corollary 19.5). Integer arithmetic avoids the floating-point rounding of
    log_phi n, which can be off by one near n = phi^k.

    Args:
        n: Number of items in the heap (at least 1).

    Returns:
        The bound as an integer.
    """
    degree, fib_k2, fib_k3 = 0, 1, 2  # F_2 = 1, F_3 = 2
    while fib_k3 <= n:
        degree += 1
        fib_k2, fib_k3 = fib_k3, fib_k2 + fib_k3
    return degree


class FibonacciHeap(MergeableHeap):
    """Mergeable min-heap with O(1) amortized insert, merge and decrease_key."""

    name = "Fibonacci heap"

    __slots__ = ("_min", "_size")

    def __init__(self) -> None:
        """Create an empty heap."""
        self._min: Optional[FibonacciNode] = None  # a root with the smallest key
        self._size = 0

    def insert(self, key: Any, value: Any = None) -> FibonacciNode:
        """Add a one-node tree to the root list. O(1).

        Args:
            key: Priority of the item.
            value: Payload stored with the key.

        Returns:
            The new node, which is its handle.
        """
        node = FibonacciNode(key, value)
        minimum = self._min
        if minimum is None:
            self._min = node
        else:
            right = minimum.right  # insert to the right of the minimum
            node.left, node.right = minimum, right
            minimum.right = right.left = node
            if key < minimum.key:
                self._min = node
        self._size += 1
        return node

    def find_min(self) -> FibonacciNode:
        """Return the minimum root. See :meth:`MergeableHeap.find_min`.

        Returns:
            Handle of a minimum item, in O(1).

        Raises:
            IndexError: If the heap is empty.
        """
        if self._min is None:
            raise IndexError("find_min from an empty heap")
        return self._min

    def extract_min(self) -> tuple[Any, Any]:
        """Remove the minimum root, promote its children, then consolidate.

        O(log n) amortized; O(number of roots + D(n)) actual.

        Returns:
            Tuple ``(key, value)`` of the removed item.

        Raises:
            IndexError: If the heap is empty.
        """
        minimum = self._min
        if minimum is None:
            raise IndexError("extract_min from an empty heap")
        child = minimum.child
        if child is not None:
            for node in _siblings(child):
                node.parent = None
                node.mark = False  # roots are never marked
            _splice(minimum, child)
            minimum.child = None
        if minimum.right is minimum:  # it was the only node left
            self._min = None
        else:
            self._min = minimum.right
            _unlink(minimum)
            self._consolidate()
        self._size -= 1
        minimum.left = minimum.right = None  # marks the handle as removed
        return minimum.key, minimum.value

    def decrease_key(self, handle: FibonacciNode, new_key: Any) -> None:
        """Lower a key; cut the node if heap order breaks. O(1) amortized.

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
        handle.key = new_key
        parent = handle.parent
        if parent is not None and new_key < parent.key:
            self._cut(handle, parent)
            self._cascading_cut(parent)
        if new_key < self._min.key:
            self._min = handle

    def delete(self, handle: FibonacciNode) -> None:
        """Cut the node to the root list, make it the minimum, extract it. O(log n) amortized.

        This is "decrease to minus infinity, then extract_min" without needing
        a minus-infinity key, so it works for any key type.

        Args:
            handle: Node returned by :meth:`insert`.

        Returns:
            None.

        Raises:
            ValueError: If the node was already removed.
        """
        self._check_live(handle)
        parent = handle.parent
        if parent is not None:
            self._cut(handle, parent)
            self._cascading_cut(parent)
        self._min = handle  # consolidation in extract_min recomputes the true minimum
        self.extract_min()

    def merge(self, other: FibonacciHeap) -> None:
        """Concatenate the two root lists. O(1).

        Args:
            other: Fibonacci heap whose items move into this heap; left empty.

        Returns:
            None.

        Raises:
            TypeError: If ``other`` is not a FibonacciHeap.
            ValueError: If ``other`` is this heap.
        """
        self._check_merge_partner(other)
        if other._min is not None:
            if self._min is None:
                self._min = other._min
            else:
                _splice(self._min, other._min)
                if other._min.key < self._min.key:
                    self._min = other._min
        self._size += other._size
        other._min, other._size = None, 0

    def __len__(self) -> int:
        """Return the number of items in the heap."""
        return self._size

    def root_degrees(self) -> list[int]:
        """Degrees of the roots, in root-list order starting at the minimum.

        Returns:
            List of degrees (empty for an empty heap).
        """
        if self._min is None:
            return []
        return [root.degree for root in _siblings(self._min)]

    def check_invariants(self) -> None:
        """Verify the Fibonacci-heap invariants. O(n).

        Checks: every circular list is consistent (``x.right.left is x``);
        roots have no parent and are unmarked; the min pointer is a root with
        the smallest key; every child points to its parent and respects heap
        order; each degree equals the number of children and is at most
        D(n) <= floor(log_phi n); the node count equals the stored size.

        Returns:
            None.

        Raises:
            AssertionError: If an invariant is broken.
        """
        if self._min is None:
            assert self._size == 0, "empty root list but non-zero size"
            return
        bound = max_degree_bound(self._size)
        count = 0
        stack = []
        for root in self._bounded_siblings(self._min):
            assert root.parent is None, "a root has a parent"
            assert not root.mark, "a root is marked"
            assert not root.key < self._min.key, "a root is smaller than the min pointer"
            stack.append(root)
        while stack:
            node = stack.pop()
            count += 1
            assert node.degree <= bound, "degree exceeds D(n)"
            children = 0
            if node.child is not None:
                for child in self._bounded_siblings(node.child):
                    assert child.parent is node, "child's parent pointer is wrong"
                    assert not child.key < node.key, "heap order violated"
                    stack.append(child)
                    children += 1
            assert children == node.degree, "degree does not match the number of children"
        assert count == self._size, "node count differs from the stored size"

    def _iter_nodes(self) -> Iterator[FibonacciNode]:
        """Yield every node once (depth-first, with an explicit stack).

        Returns:
            Iterator over all nodes of all trees.
        """
        if self._min is None:
            return
        stack = list(_siblings(self._min))
        while stack:
            node = stack.pop()
            yield node
            if node.child is not None:
                stack.extend(_siblings(node.child))

    def _bounded_siblings(self, start: FibonacciNode) -> Iterator[FibonacciNode]:
        """Like :func:`_siblings`, but fail instead of looping on a broken list.

        Args:
            start: Any node of a circular list of this heap.

        Returns:
            Iterator over the nodes of the list.

        Raises:
            AssertionError: If the list is longer than the heap or inconsistent.
        """
        steps = 0
        for node in _siblings(start):
            assert node.right.left is node, "circular list pointers are inconsistent"
            steps += 1
            assert steps <= self._size, "circular list does not close"
            yield node

    def _consolidate(self) -> None:
        """Link roots of equal degree until all root degrees differ (CLRS CONSOLIDATE).

        Uses a degree table with D(n) + 2 entries (D(n) <= floor(log_phi n), see
        :func:`max_degree_bound`), then rebuilds the root list from the table
        and finds the new minimum.

        Returns:
            None.
        """
        table: list[Optional[FibonacciNode]] = [None] * (max_degree_bound(self._size) + 2)
        for node in list(_siblings(self._min)):  # copy: linking rewires the list
            degree = node.degree
            other = table[degree]
            while other is not None:
                if other.key < node.key:
                    node, other = other, node
                self._link(other, node)
                table[degree] = None
                degree += 1
                other = table[degree]
            table[degree] = node
        self._min = None
        for root in table:
            if root is None:
                continue
            root.left = root.right = root
            if self._min is None:
                self._min = root
            else:
                _splice(self._min, root)
                if root.key < self._min.key:
                    self._min = root

    @staticmethod
    def _link(child: FibonacciNode, parent: FibonacciNode) -> None:
        """Make the root ``child`` a child of the root ``parent`` (CLRS FIB-HEAP-LINK).

        ``child``'s old root-list pointers are simply overwritten: during
        consolidation the root list is rebuilt from the degree table anyway.

        Args:
            child: Root that goes below (its key must not be smaller).
            parent: Root that stays on top.

        Returns:
            None.
        """
        child.left = child.right = child
        if parent.child is None:
            parent.child = child
        else:
            _splice(parent.child, child)
        child.parent = parent
        child.mark = False
        parent.degree += 1

    def _cut(self, node: FibonacciNode, parent: FibonacciNode) -> None:
        """Move ``node`` (with its subtree) from ``parent``'s children to the root list.

        Args:
            node: Child to cut.
            parent: Its parent.

        Returns:
            None.
        """
        if node.right is node:
            parent.child = None
        else:
            if parent.child is node:
                parent.child = node.right
            _unlink(node)
        parent.degree -= 1
        node.parent = None
        node.mark = False
        _splice(self._min, node)

    def _cascading_cut(self, node: FibonacciNode) -> None:
        """Walk up from a node that just lost a child (CLRS CASCADING-CUT, as a loop).

        An unmarked non-root gets marked and the walk stops; a marked one is cut
        and the walk continues with its parent. Roots are never marked.

        Args:
            node: Node that just lost a child.

        Returns:
            None.
        """
        parent = node.parent
        while parent is not None:
            if not node.mark:
                node.mark = True
                return
            self._cut(node, parent)
            node, parent = parent, parent.parent

    @staticmethod
    def _check_live(handle: FibonacciNode) -> None:
        """Reject handles of removed nodes.

        Args:
            handle: Node passed by the caller.

        Returns:
            None.

        Raises:
            ValueError: If the node is no longer in a heap.
        """
        if handle.left is None:
            raise ValueError("the item is no longer in the heap")
