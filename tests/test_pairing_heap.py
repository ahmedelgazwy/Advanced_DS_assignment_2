"""Structure tests specific to the pairing heap.

The shared behaviour (ordering, decrease-key, delete, merge, handles) is
covered by ``test_common.py``; these tests check the tree shape produced by
link and two-pass pairing, and the ``prev`` pointers used for O(1) cuts.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import math
import random

import pytest

from mergeable_heaps import PairingHeap
from mergeable_heaps.pairing_heap import PairingNode


def children(node: PairingNode) -> list[PairingNode]:
    """The children of ``node``, leftmost first."""
    result, child = [], node.child
    while child is not None:
        result.append(child)
        child = child.next
    return result


def depth(heap: PairingHeap) -> int:
    """Number of edges on the longest root-to-leaf path (explicit stack)."""
    deepest, stack = 0, [(heap.find_min(), 0)]
    while stack:
        node, level = stack.pop()
        deepest = max(deepest, level)
        stack.extend((child, level + 1) for child in children(node))
    return deepest


# --- Shape after inserts and merges -------------------------------------------------


def test_ascending_inserts_hang_every_item_below_the_first():
    heap = PairingHeap()
    heap.insert_all(range(100))
    assert heap.find_min().key == 0
    assert heap.root_degree() == 99
    assert [child.key for child in children(heap.find_min())] == list(range(99, 0, -1))


def test_descending_inserts_build_a_path():
    """Every new key is a new minimum, so the old root becomes its only child."""
    heap = PairingHeap()
    heap.insert_all(range(100, 0, -1))
    assert heap.root_degree() == 1
    assert depth(heap) == 99


def test_merge_is_a_single_link():
    heap, other = PairingHeap(), PairingHeap()
    heap.insert_all([1, 5, 6])
    other.insert_all([3, 4])
    degree_before = heap.root_degree()
    heap.merge(other)
    assert heap.find_min().key == 1
    assert heap.root_degree() == degree_before + 1
    assert children(heap.find_min())[0].key == 3  # the other root is the new leftmost child


# --- Two-pass pairing ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4, 5, 9, 100, 101])
def test_first_extract_min_pairs_the_children(n):
    """After ascending inserts the root has r = n - 1 children, the largest key leftmost.

    Pass 1 forms floor(r/2) pairs (plus one single tree if r is odd); the
    rightmost tree holds the smallest key, so pass 2 links every other tree
    below it. Its degree is then (number of trees - 1) plus 1 if it is a pair:
    floor(r/2) in both cases.
    """
    heap = PairingHeap()
    heap.insert_all(range(n))
    heap.extract_min()
    assert heap.find_min().key == 1
    assert heap.root_degree() == (n - 1) // 2
    heap.check_invariants()


def test_two_pass_pairing_on_a_small_example():
    """Children 4, 3, 2, 1 (left to right) -> pairs 3(4) and 1(2) -> 1 with children 3(4), 2."""
    heap = PairingHeap()
    heap.insert_all(range(5))
    heap.extract_min()
    root = heap.find_min()
    assert root.key == 1
    assert [child.key for child in children(root)] == [3, 2]
    assert [child.key for child in children(children(root)[0])] == [4]


def test_root_degree_halves_on_ascending_input():
    """1,023 children pair into 511 trees; their 511 roots pair into 256 (see the test above).

    Later steps are only roughly halved, because the subtrees stop being uniform.
    """
    heap = PairingHeap()
    heap.insert_all(range(1_024))
    widths = [heap.root_degree()]
    for _ in range(2):
        heap.extract_min()
        widths.append(heap.root_degree())
    assert widths == [1_023, 511, 256]
    heap.check_invariants()


@pytest.mark.parametrize("order", ["ascending", "descending", "random"])
def test_total_pairing_work_of_a_full_drain_is_n_log_n(order):
    """Amortized O(log n) extract-min: the children paired over n extracts total O(n log n)."""
    n = 4_096
    keys = {
        "ascending": list(range(n)),
        "descending": list(range(n, 0, -1)),
        "random": random.Random(9).sample(range(1_000_000), n),
    }[order]
    heap = PairingHeap()
    heap.insert_all(keys)
    work = 0
    while heap:
        work += heap.root_degree()
        heap.extract_min()
    assert work <= 2 * n * math.log2(n)  # measured: at most 1.23 n log2 n


# --- Decrease-key and the prev pointers ----------------------------------------------


def test_decrease_key_of_the_root_keeps_the_structure():
    heap = PairingHeap()
    handles = heap.insert_all(range(10))
    before = [child.key for child in children(heap.find_min())]
    heap.decrease_key(handles[0], -1)
    assert heap.find_min() is handles[0]
    assert [child.key for child in children(heap.find_min())] == before


def test_decrease_key_of_a_leftmost_child_cuts_it():
    heap = PairingHeap()
    handles = heap.insert_all(range(5))  # root 0, children 4, 3, 2, 1
    leftmost = handles[4]
    assert leftmost.prev is handles[0]
    heap.decrease_key(leftmost, -1)
    assert heap.find_min() is leftmost
    assert children(leftmost) == [handles[0]]  # old root linked below
    assert [child.key for child in children(handles[0])] == [3, 2, 1]
    heap.check_invariants()


def test_decrease_key_of_an_inner_sibling_repairs_both_neighbours():
    heap = PairingHeap()
    handles = heap.insert_all(range(5))  # root 0, children 4, 3, 2, 1
    inner = handles[2]
    heap.decrease_key(inner, 1)  # still larger than the root: linked as its leftmost child
    assert handles[1].prev is handles[3]
    assert handles[3].next is handles[1]
    assert children(heap.find_min())[0] is inner
    heap.check_invariants()


def test_decrease_key_with_an_equal_key_does_not_restructure():
    heap = PairingHeap()
    handles = heap.insert_all(range(5))
    heap.decrease_key(handles[2], 2)
    assert [child.key for child in children(heap.find_min())] == [4, 3, 2, 1]


# --- Delete and removed handles ----------------------------------------------------------


def test_delete_an_inner_node_pairs_its_children_back():
    heap = PairingHeap()
    heap.insert_all(range(64))
    heap.extract_min()  # builds a deeper tree
    root = heap.find_min()
    victim = max(
        (node for node in heap._iter_nodes() if node is not root),
        key=lambda node: len(children(node)),
    )
    orphans = {id(child) for child in children(victim)}
    assert orphans
    heap.delete(victim)
    heap.check_invariants()
    assert orphans <= {id(node) for node in heap._iter_nodes()}
    assert len(heap) == 62


def test_removed_nodes_point_prev_at_themselves():
    heap = PairingHeap()
    first = heap.insert(1)
    second = heap.insert(2)
    heap.extract_min()
    heap.delete(second)
    for node in (first, second):
        assert node.prev is node
        assert node.child is None and node.next is None
