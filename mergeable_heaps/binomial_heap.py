"""Binomial heap [Vuillemin 1978; CLRS 2nd ed., ch. 19; CLRS 3rd ed., Problem 19-2].

A binomial heap is a list of heap-ordered binomial trees with at most one tree
of each order, kept in increasing order. A tree of order k (B_k) has 2^k
nodes, so the orders present are the 1-bits of n: 13 = 0b1101 items are
stored as B_0, B_2 and B_3. Merging two heaps works like binary addition,
where linking two trees of equal order is a carry.

Costs (see docs/theory.md, section 1): insert O(1) amortized and O(log n)
worst case; find_min O(1) (a pointer to the minimum root is kept);
extract_min, decrease_key, delete and merge O(log n) worst case.

Handles. CLRS decrease-key swaps keys and data between a node and its parent,
which would leave a caller's handle pointing at the wrong item. Here every
node points to a separate :class:`BinomialItem` (the handle) that points back
to its node, and the bubble-up swaps whole items and fixes both pointers.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import sys
from typing import Any, Iterator, Optional

from mergeable_heaps.base import MergeableHeap


class BinomialItem:
    """An item of a binomial heap; this object is the caller's handle.

    ``node`` is the tree node currently holding the item, or None once the
    item has been removed from the heap.
    """

    __slots__ = ("key", "value", "node")

    def __init__(self, key: Any, value: Any) -> None:
        """Create an item that is not yet placed in a node.

        Args:
            key: Priority of the item.
            value: Payload stored with the key.
        """
        self.key = key
        self.value = value
        self.node: Optional[BinomialNode] = None


class BinomialNode:
    """A node of a binomial tree.

    ``child`` is the leftmost (highest-order) child. Children, and the roots of
    the heap, are chained through ``sibling``: a child list runs in decreasing
    order, the root list in increasing order.
    """

    __slots__ = ("item", "parent", "child", "sibling", "degree")

    def __init__(self, item: BinomialItem) -> None:
        """Create a one-node tree (B_0) holding ``item``.

        Args:
            item: Item to store; its back-pointer is set to this node.
        """
        self.item = item
        self.parent: Optional[BinomialNode] = None
        self.child: Optional[BinomialNode] = None
        self.sibling: Optional[BinomialNode] = None
        self.degree = 0
        item.node = self


def _link(child: BinomialNode, parent: BinomialNode) -> None:
    """Make the root ``child`` the leftmost child of the root ``parent``.

    Both roots must have the same degree k; the result is a tree of order k+1.

    Args:
        child: Root that goes below (its key must not be smaller).
        parent: Root that stays on top.

    Returns:
        None.
    """
    child.parent = parent
    child.sibling = parent.child
    parent.child = child
    parent.degree += 1


def _merge_root_lists(
    first: Optional[BinomialNode], second: Optional[BinomialNode]
) -> Optional[BinomialNode]:
    """Merge two root lists sorted by degree into one sorted list.

    Like the merge step of merge sort; trees are not linked yet, so the result
    may contain two trees of the same degree.

    Args:
        first: Head of the first root list (may be None).
        second: Head of the second root list (may be None).

    Returns:
        Head of the merged list.
    """
    if first is None:
        return second
    if second is None:
        return first
    if first.degree <= second.degree:
        head, first = first, first.sibling
    else:
        head, second = second, second.sibling
    tail = head
    while first is not None and second is not None:
        if first.degree <= second.degree:
            tail.sibling, first = first, first.sibling
        else:
            tail.sibling, second = second, second.sibling
        tail = tail.sibling
    tail.sibling = first if first is not None else second
    return head


def _union(
    first: Optional[BinomialNode], second: Optional[BinomialNode]
) -> Optional[BinomialNode]:
    """Union of two root lists: merge them, then link trees of equal degree.

    This is binary addition: each link is a carry (CLRS BINOMIAL-HEAP-UNION).
    Cost O(log n1 + log n2).

    Args:
        first: Head of the first root list (may be None).
        second: Head of the second root list (may be None).

    Returns:
        Head of the resulting root list, with distinct degrees in increasing order.
    """
    head = _merge_root_lists(first, second)
    if head is None:
        return None
    previous: Optional[BinomialNode] = None
    current = head
    following = current.sibling
    while following is not None:
        if current.degree != following.degree or (
            following.sibling is not None and following.sibling.degree == current.degree
        ):
            # Different degrees, or three equal ones in a row: link the last two later.
            previous, current = current, following
        elif not following.item.key < current.item.key:
            current.sibling = following.sibling
            _link(following, current)
        else:
            if previous is None:
                head = following
            else:
                previous.sibling = following
            _link(current, following)
            current = following
        following = current.sibling
    return head


class BinomialHeap(MergeableHeap):
    """Mergeable min-heap stored as a forest of binomial trees."""

    name = "Binomial heap"

    __slots__ = ("_head", "_min", "_size")

    def __init__(self) -> None:
        """Create an empty heap."""
        self._head: Optional[BinomialNode] = None  # root list, increasing degree
        self._min: Optional[BinomialNode] = None  # root with the smallest key
        self._size = 0

    def insert(self, key: Any, value: Any = None) -> BinomialItem:
        """Add an item: a binary-counter increment on the root list.

        The new one-node tree is linked with the leading root while their
        degrees are equal (each link is a carry). Only the carried roots are
        touched, which gives O(1) amortized cost.

        Args:
            key: Priority of the item.
            value: Payload stored with the key.

        Returns:
            The new item, which is its handle.
        """
        item = BinomialItem(key, value)
        carry = BinomialNode(item)
        head = self._head
        while head is not None and head.degree == carry.degree:
            rest = head.sibling
            if carry.item.key < head.item.key:
                _link(head, carry)
            else:
                _link(carry, head)
                carry = head
            head = rest
        carry.sibling = head
        self._head = carry
        # The previous minimum is either still a root, or was linked below
        # ``carry``, in which case carry's key is no larger.
        if self._min is None or not self._min.item.key < carry.item.key:
            self._min = carry
        self._size += 1
        return item

    def find_min(self) -> BinomialItem:
        """Return the item at the minimum root. See :meth:`MergeableHeap.find_min`.

        Returns:
            Handle of a minimum item, in O(1).

        Raises:
            IndexError: If the heap is empty.
        """
        if self._min is None:
            raise IndexError("find_min from an empty heap")
        return self._min.item

    def extract_min(self) -> tuple[Any, Any]:
        """Remove the minimum root and union its children back. O(log n).

        Returns:
            Tuple ``(key, value)`` of the removed item.

        Raises:
            IndexError: If the heap is empty.
        """
        if self._min is None:
            raise IndexError("extract_min from an empty heap")
        item = self._min.item
        self._remove_root(self._min)
        return item.key, item.value

    def decrease_key(self, handle: BinomialItem, new_key: Any) -> None:
        """Lower a key, then bubble the item up its tree. O(log n).

        Args:
            handle: Item returned by :meth:`insert`.
            new_key: New key, not larger than the current one.

        Returns:
            None.

        Raises:
            ValueError: If ``new_key`` is larger, or the item was removed.
        """
        node = self._live_node(handle)
        self._check_new_key(handle.key, new_key)
        handle.key = new_key
        node = self._bubble_up(node, to_root=False)
        if node.parent is None and new_key < self._min.item.key:
            self._min = node

    def delete(self, handle: BinomialItem) -> None:
        """Bubble the item to the root of its tree, then remove that root. O(log n).

        Args:
            handle: Item returned by :meth:`insert`.

        Returns:
            None.

        Raises:
            ValueError: If the item was already removed.
        """
        node = self._live_node(handle)
        self._remove_root(self._bubble_up(node, to_root=True))

    def merge(self, other: BinomialHeap) -> None:
        """Union the two root lists (binary addition). O(log n).

        Args:
            other: Binomial heap whose items move into this heap; left empty.

        Returns:
            None.

        Raises:
            TypeError: If ``other`` is not a BinomialHeap.
            ValueError: If ``other`` is this heap.
        """
        self._check_merge_partner(other)
        self._head = _union(self._head, other._head)
        self._size += other._size
        self._min = self._find_min_root()
        other._head, other._min, other._size = None, None, 0

    def __len__(self) -> int:
        """Return the number of items in the heap."""
        return self._size

    def root_degrees(self) -> list[int]:
        """Orders of the trees in the root list, smallest first.

        Returns:
            List of degrees; they are the positions of the 1-bits of ``len(self)``.
        """
        return [root.degree for root in self._roots()]

    def check_invariants(self) -> None:
        """Verify the binomial-heap invariants. O(n).

        Checks: root degrees strictly increase; every tree of degree k has
        children of degrees k-1, ..., 0 (so it is a B_k with 2^k nodes); parent
        pointers; heap order; item back-pointers; the item count; and that the
        min pointer is a root with the smallest key.

        Returns:
            None.

        Raises:
            AssertionError: If an invariant is broken.
        """
        count = 0
        previous_degree = -1
        stack = []
        root = self._head
        while root is not None:
            assert root.parent is None, "a root has a parent"
            assert root.degree > previous_degree, "root degrees are not strictly increasing"
            assert not root.item.key < self._min.item.key, "a root is smaller than the min pointer"
            previous_degree = root.degree
            stack.append(root)
            root = root.sibling
        assert (self._min is None) == (self._head is None), "min pointer and root list disagree"
        assert self._min is None or self._min in stack, "min pointer is not a root"
        while stack:
            node = stack.pop()
            count += 1
            assert node.item.node is node, "item back-pointer does not point to its node"
            expected_degree = node.degree - 1
            child = node.child
            while child is not None:
                assert child.degree == expected_degree, "child degrees are not k-1, ..., 0"
                assert child.parent is node, "child's parent pointer is wrong"
                assert not child.item.key < node.item.key, "heap order violated"
                stack.append(child)
                expected_degree -= 1
                child = child.sibling
            assert expected_degree == -1, "node degree does not match its number of children"
        assert count == self._size, "node count differs from the stored size"

    def _node_bytes(self, node: BinomialNode) -> int:
        """Bytes owned by one node: the node, its item and the key.

        Args:
            node: A node yielded by :meth:`_iter_nodes`.

        Returns:
            Size in bytes as reported by ``sys.getsizeof``.
        """
        return sys.getsizeof(node) + sys.getsizeof(node.item) + sys.getsizeof(node.item.key)

    def _roots(self) -> Iterator[BinomialNode]:
        """Yield the roots in root-list order (increasing degree).

        Returns:
            Iterator over the roots.
        """
        root = self._head
        while root is not None:
            yield root
            root = root.sibling

    def _children(self, node: BinomialNode) -> Iterator[BinomialNode]:
        """Yield a node's children, highest order first.

        Args:
            node: A node of this heap.

        Returns:
            Iterator over the children.
        """
        child = node.child
        while child is not None:
            yield child
            child = child.sibling

    def _handle_of(self, node: BinomialNode) -> BinomialItem:
        """The item stored at a node, which is the caller's handle.

        Args:
            node: A node of this heap.

        Returns:
            The node's item.
        """
        return node.item

    def _find_min_root(self) -> Optional[BinomialNode]:
        """Scan the root list for the smallest key. O(log n).

        Returns:
            The first root with the smallest key, or None if the heap is empty.
        """
        best = root = self._head
        while root is not None:
            if root.item.key < best.item.key:
                best = root
            root = root.sibling
        return best

    def _bubble_up(self, node: BinomialNode, to_root: bool) -> BinomialNode:
        """Move a node's item towards the root by swapping items with parents.

        Swapping whole items (not keys) keeps every caller's handle valid: both
        back-pointers are updated at each step.

        Args:
            node: Node whose item moves up.
            to_root: If True, move all the way to the root (used by delete);
                otherwise stop as soon as heap order holds.

        Returns:
            The node that holds the item at the end.
        """
        item = node.item
        parent = node.parent
        while parent is not None and (to_root or item.key < parent.item.key):
            parent_item = parent.item
            node.item, parent_item.node = parent_item, node
            parent.item, item.node = item, parent
            node, parent = parent, parent.parent
        return node

    def _remove_root(self, root: BinomialNode) -> None:
        """Unlink a root, union its children back in and refresh the min pointer.

        Args:
            root: A root of this heap.

        Returns:
            None.
        """
        previous, current = None, self._head
        while current is not root:
            previous, current = current, current.sibling
        if previous is None:
            self._head = root.sibling
        else:
            previous.sibling = root.sibling
        # The children are in decreasing order; reverse them into a root list.
        children: Optional[BinomialNode] = None
        child = root.child
        while child is not None:
            following = child.sibling
            child.parent = None
            child.sibling = children
            children = child
            child = following
        self._head = _union(self._head, children)
        self._size -= 1
        self._min = self._find_min_root()
        root.item.node = None  # marks the handle as removed

    @staticmethod
    def _live_node(handle: BinomialItem) -> BinomialNode:
        """Return the node holding a handle's item.

        Args:
            handle: Item passed by the caller.

        Returns:
            The item's current node.

        Raises:
            ValueError: If the item is no longer in a heap.
        """
        node = handle.node
        if node is None:
            raise ValueError("the item is no longer in the heap")
        return node
