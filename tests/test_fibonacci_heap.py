"""Structure tests specific to the Fibonacci heap.

The shared behaviour (ordering, decrease-key, delete, merge, handles) is
covered by ``test_common.py``; these tests check laziness, consolidation,
marks and cascading cuts, and the degree bounds of CLRS 3rd ed., ch. 19.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import math
import random

import pytest

from mergeable_heaps import FibonacciHeap
from mergeable_heaps.fibonacci_heap import FibonacciNode, max_degree_bound

PHI = (1 + math.sqrt(5)) / 2


def fibonacci(k: int) -> int:
    """The k-th Fibonacci number (F_0 = 0, F_1 = 1)."""
    previous, current = 0, 1
    for _ in range(k):
        previous, current = current, previous + current
    return previous


def one_bits(n: int) -> list[int]:
    """Positions of the 1-bits of ``n``, lowest first (e.g. 13 -> [0, 2, 3])."""
    return [bit for bit in range(n.bit_length()) if n >> bit & 1]


def children(node: FibonacciNode) -> list[FibonacciNode]:
    """The children of ``node``, following the circular sibling list."""
    result, child = [], node.child
    while child is not None:
        result.append(child)
        child = child.right
        if child is node.child:
            break
    return result


def subtree_size(node: FibonacciNode) -> int:
    """Number of nodes in the subtree rooted at ``node`` (explicit stack)."""
    size, stack = 0, [node]
    while stack:
        current = stack.pop()
        size += 1
        stack.extend(children(current))
    return size


def child_of_degree(node: FibonacciNode, degree: int) -> FibonacciNode:
    """The unique child of ``node`` with the given degree."""
    (match,) = [child for child in children(node) if child.degree == degree]
    return match


def mixed_workload(seed: int) -> FibonacciHeap:
    """A heap shaped by many inserts, extract-mins and cut-causing decrease-keys."""
    rng = random.Random(seed)
    heap = FibonacciHeap()
    handles = [heap.insert(rng.randrange(1_000_000)) for _ in range(3_000)]
    heap.extract_min()
    for _ in range(3_000):
        choice = rng.random()
        if choice < 0.5:
            handle = handles[rng.randrange(len(handles))]
            if handle.left is not None:  # still in the heap
                heap.decrease_key(handle, handle.key - rng.randrange(1_000_000))
        elif choice < 0.8:
            heap.extract_min()
        else:
            handles.append(heap.insert(rng.randrange(1_000_000)))
    return heap


@pytest.fixture
def binomial_shaped() -> tuple[FibonacciHeap, dict[str, FibonacciNode]]:
    """15 items consolidated into trees of degrees 0-3, with named nodes of the degree-3 tree.

    Names: ``r`` is the degree-3 root, ``a`` its degree-2 child, ``b`` and ``c``
    the degree-1 and degree-0 children of ``a``, ``d`` the child of ``b`` and
    ``e`` the degree-0 child of ``r``.
    """
    heap = FibonacciHeap()
    heap.insert_all(range(16))
    heap.extract_min()  # removes 0 and consolidates the other 15 roots
    (r,) = [root for root in _roots(heap) if root.degree == 3]
    a = child_of_degree(r, 2)
    b, c = child_of_degree(a, 1), child_of_degree(a, 0)
    (d,) = children(b)
    e = child_of_degree(r, 0)
    return heap, {"r": r, "a": a, "b": b, "c": c, "d": d, "e": e}


def _roots(heap: FibonacciHeap) -> list[FibonacciNode]:
    """The roots of ``heap`` in root-list order."""
    result, root = [], heap._min
    while root is not None:
        result.append(root)
        root = root.right
        if root is heap._min:
            break
    return result


# --- Laziness ----------------------------------------------------------------------


def test_insert_only_adds_roots():
    heap = FibonacciHeap()
    heap.insert_all(range(100))
    assert heap.root_degrees() == [0] * 100


def test_merge_only_concatenates_root_lists():
    heap, other = FibonacciHeap(), FibonacciHeap()
    heap.insert_all(range(5))
    other.insert_all(range(7))
    heap.merge(other)
    assert heap.root_degrees() == [0] * 12
    heap.check_invariants()


# --- Consolidation -----------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 16, 17, 100, 1_024, 1_025])
def test_first_extract_min_consolidates_into_binomial_trees(n):
    """With no cuts, consolidation links equal degrees like binary addition."""
    heap = FibonacciHeap()
    heap.insert_all(range(n))
    heap.extract_min()
    assert sorted(heap.root_degrees()) == one_bits(n - 1)
    heap.check_invariants()


def test_root_degrees_are_distinct_after_every_extract_min():
    rng = random.Random(11)
    heap = FibonacciHeap()
    handles = heap.insert_all(rng.randrange(10_000) for _ in range(2_000))
    for _ in range(300):
        heap.extract_min()
        degrees = heap.root_degrees()
        assert len(set(degrees)) == len(degrees)
        for handle in rng.sample(handles, 5):
            if handle.left is not None:
                heap.decrease_key(handle, handle.key - rng.randrange(10_000))


# --- Degree bounds (CLRS Lemma 19.4, Corollary 19.5) ----------------------------------


@pytest.mark.parametrize("k", range(1, 80))
def test_max_degree_bound_is_exact_at_fibonacci_numbers(k):
    assert max_degree_bound(fibonacci(k + 2)) == k
    assert max_degree_bound(fibonacci(k + 2) - 1) == k - 1


def test_max_degree_bound_never_exceeds_log_phi_n():
    for n in range(1, 20_000):
        assert max_degree_bound(n) <= math.log(n) / math.log(PHI) + 1e-9


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_subtree_of_a_degree_k_node_has_at_least_f_k_plus_2_nodes(seed):
    heap = mixed_workload(seed)
    heap.check_invariants()
    for node in heap._iter_nodes():
        assert subtree_size(node) >= fibonacci(node.degree + 2)


@pytest.mark.parametrize("seed", [4, 5])
def test_degrees_stay_within_d_of_n(seed):
    heap = mixed_workload(seed)
    assert max(node.degree for node in heap._iter_nodes()) <= max_degree_bound(len(heap))


# --- Marks and cascading cuts (scripted) --------------------------------------------


def test_fixture_builds_the_expected_shape(binomial_shaped):
    heap, nodes = binomial_shaped
    assert sorted(heap.root_degrees()) == [0, 1, 2, 3]
    assert [nodes[name].degree for name in "rabcde"] == [3, 2, 1, 0, 0, 0]
    assert not any(node.mark for node in heap._iter_nodes())


def test_first_lost_child_marks_the_parent(binomial_shaped):
    heap, nodes = binomial_shaped
    a, c = nodes["a"], nodes["c"]
    heap.decrease_key(c, -10)
    assert c.parent is None
    assert a.mark
    assert a.degree == 1
    assert heap.find_min() is c
    heap.check_invariants()


def test_second_lost_child_cuts_the_marked_parent(binomial_shaped):
    heap, nodes = binomial_shaped
    r, a, b, c, d = (nodes[name] for name in "rabcd")
    heap.decrease_key(c, -10)  # a loses its first child: marked
    heap.decrease_key(b, -20)  # a loses its second child: cascading cut
    assert b.parent is None and not b.mark
    assert a.parent is None and not a.mark
    assert d.parent is b  # the subtree moves with b
    assert r.degree == 2
    assert not r.mark  # roots are never marked
    heap.check_invariants()
    expected = sorted(node.key for node in heap._iter_nodes())
    assert [key for key, _ in heap.drain()] == expected


def test_a_root_is_never_marked_when_it_loses_a_child(binomial_shaped):
    heap, nodes = binomial_shaped
    r, e = nodes["r"], nodes["e"]
    heap.decrease_key(e, -5)
    assert e.parent is None
    assert not r.mark
    assert r.degree == 2
    heap.check_invariants()


def test_decrease_key_that_keeps_heap_order_does_not_cut(binomial_shaped):
    heap, nodes = binomial_shaped
    a, c = nodes["a"], nodes["c"]
    heap.decrease_key(c, a.key)  # equal to the parent's key: still heap-ordered
    assert c.parent is a
    assert not a.mark
    heap.check_invariants()


def test_delete_cuts_the_node_and_cascades(binomial_shaped):
    heap, nodes = binomial_shaped
    r, a, b, c = (nodes[name] for name in "rabc")
    heap.delete(c)  # a marked
    heap.delete(b)  # b's child d is promoted; a is cut from r
    assert c.left is None and b.left is None
    assert a.parent is None
    assert r.degree == 2
    heap.check_invariants()
    assert len(heap) == 13


def test_removed_nodes_lose_their_list_pointers():
    heap = FibonacciHeap()
    handle = heap.insert(1)
    heap.insert(2)
    heap.extract_min()
    assert handle.left is None and handle.right is None
