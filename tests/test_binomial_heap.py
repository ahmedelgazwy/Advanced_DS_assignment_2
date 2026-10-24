"""Structure tests specific to the binomial heap.

The shared behaviour (ordering, decrease-key, delete, merge, handles) is
covered by ``test_common.py``; these tests check the binomial-tree shape.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import random

import pytest

from mergeable_heaps import BinomialHeap
from mergeable_heaps.binomial_heap import BinomialNode


def one_bits(n: int) -> list[int]:
    """Positions of the 1-bits of ``n``, lowest first (e.g. 13 -> [0, 2, 3])."""
    return [bit for bit in range(n.bit_length()) if n >> bit & 1]


def tree_size(root: BinomialNode) -> int:
    """Number of nodes in the tree rooted at ``root`` (explicit stack)."""
    size, stack = 0, [root]
    while stack:
        node = stack.pop()
        size += 1
        child = node.child
        while child is not None:
            stack.append(child)
            child = child.sibling
    return size


def roots(heap: BinomialHeap) -> list[BinomialNode]:
    """The roots of ``heap`` in root-list order."""
    result, root = [], heap._head
    while root is not None:
        result.append(root)
        root = root.sibling
    return result


def depth(node: BinomialNode) -> int:
    """Number of edges from ``node`` up to its root."""
    levels = 0
    while node.parent is not None:
        node, levels = node.parent, levels + 1
    return levels


def test_root_degrees_are_the_one_bits_of_n_after_every_insert():
    heap = BinomialHeap()
    for n in range(1, 130):
        heap.insert(n)
        assert heap.root_degrees() == one_bits(n)
    heap.check_invariants()


def test_thirteen_items_form_b0_b2_b3():
    heap = BinomialHeap()
    heap.insert_all(range(13))
    assert heap.root_degrees() == [0, 2, 3]


def test_inserting_into_two_to_the_j_minus_one_items_carries_into_one_tree():
    heap = BinomialHeap()
    heap.insert_all(range(15))  # 0b1111: B_0, B_1, B_2, B_3
    assert heap.root_degrees() == [0, 1, 2, 3]
    heap.insert(15)  # 0b10000: four carries
    assert heap.root_degrees() == [4]


def test_every_tree_of_order_k_has_two_to_the_k_nodes():
    heap = BinomialHeap()
    heap.insert_all(random.Random(5).sample(range(10_000), 1_000))
    for _ in range(77):
        heap.extract_min()
    for root in roots(heap):
        assert tree_size(root) == 2 ** root.degree


@pytest.mark.parametrize(("n1", "n2"), [(5, 3), (13, 13), (1, 0), (64, 63)])
def test_merge_is_binary_addition(n1, n2):
    heap, other = BinomialHeap(), BinomialHeap()
    heap.insert_all(range(n1))
    other.insert_all(range(n2))
    heap.merge(other)
    assert heap.root_degrees() == one_bits(n1 + n2)
    heap.check_invariants()


def test_extract_min_keeps_the_one_bit_structure():
    heap = BinomialHeap()
    heap.insert_all(random.Random(6).sample(range(1_000), 200))
    for n in range(199, 0, -1):
        heap.extract_min()
        assert heap.root_degrees() == one_bits(n)


def test_min_pointer_is_always_the_smallest_root():
    rng = random.Random(8)
    heap = BinomialHeap()
    for _ in range(500):
        heap.insert(rng.randrange(100))
        smallest_root = min(root.item.key for root in roots(heap))
        assert heap.find_min().key == smallest_root
        assert heap._min.parent is None


def test_decrease_key_moves_the_item_to_the_root_and_keeps_every_handle_valid():
    heap = BinomialHeap()
    handles = [heap.insert(index, f"v{index}") for index in range(64)]  # a single B_6
    assert heap.root_degrees() == [6]
    deepest = max(handles, key=lambda handle: depth(handle.node))
    assert depth(deepest.node) == 6
    heap.decrease_key(deepest, -1)
    assert deepest.node.parent is None
    assert heap.find_min() is deepest
    heap.check_invariants()  # includes item.node back-pointers
    for index, handle in enumerate(handles):
        assert handle.node.item is handle
        assert handle.value == f"v{index}"


def test_decrease_key_swaps_items_not_keys():
    heap = BinomialHeap()
    parent_item = heap.insert(1, "parent")
    child_item = heap.insert(2, "child")  # linked below parent_item's node
    assert child_item.node.parent is parent_item.node
    parent_node, child_node = parent_item.node, child_item.node
    heap.decrease_key(child_item, 0)
    assert (child_item.key, child_item.value) == (0, "child")
    assert (parent_item.key, parent_item.value) == (1, "parent")
    assert child_item.node is parent_node
    assert parent_item.node is child_node


def test_delete_bubbles_to_the_root_and_marks_the_handle_removed():
    heap = BinomialHeap()
    handles = heap.insert_all(range(32))
    leaf = max(handles, key=lambda handle: depth(handle.node))
    heap.delete(leaf)
    assert leaf.node is None
    assert heap.root_degrees() == one_bits(31)
    heap.check_invariants()
