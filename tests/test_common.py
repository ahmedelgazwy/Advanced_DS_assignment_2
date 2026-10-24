"""Contract tests that every mergeable heap must pass.

Each test runs once per heap through the parametrized factory fixtures in
``conftest.py``: the simple ``ReferenceHeap`` (which validates the tests
themselves) and every real heap. Structure-specific checks live in the
per-heap test files.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Callable

import pytest

import mergeable_heaps
from conftest import NUM_KEYS
from mergeable_heaps import Handle, MergeableHeap
from reference_heap import ReferenceHeap

HeapFactory = Callable[[], MergeableHeap]

_rng = random.Random(1)
KEY_SETS: dict[str, list[Any]] = {
    "random_ints": [_rng.randrange(1_000_000) for _ in range(2_000)],
    "many_duplicates": [_rng.randrange(20) for _ in range(1_000)],
    "negatives_and_zero": [_rng.randrange(-500, 500) for _ in range(1_000)],
    "floats": [_rng.uniform(-1.0, 1.0) for _ in range(1_000)],
    "strings": ["".join(_rng.choices("abc", k=_rng.randrange(6))) for _ in range(1_000)],
    "ascending": list(range(1_000)),
    "descending": list(range(1_000, 0, -1)),
    "all_equal": [7] * 500,
    "single": [42],
}
STRESS_N = 20_000  # well above Python's default recursion limit of 1,000


class OtherHeap(ReferenceHeap):
    """A heap of a different class, for the type check in ``merge``."""


def filled(make_heap: HeapFactory, keys: list[Any]) -> tuple[MergeableHeap, list[Handle]]:
    """Build a heap holding ``keys``; each item's value is its index in ``keys``.

    Args:
        make_heap: Heap factory.
        keys: Keys to insert, in order.

    Returns:
        Tuple ``(heap, handles)`` with one handle per key.
    """
    heap = make_heap()
    handles = [heap.insert(key, index) for index, key in enumerate(keys)]
    return heap, handles


def drained_keys(heap: MergeableHeap) -> list[Any]:
    """Empty the heap and return the keys in the order they came out.

    Args:
        heap: Heap to drain.

    Returns:
        List of keys.
    """
    return [key for key, _ in heap.drain()]


# --- Empty heap and insert ---------------------------------------------------


def test_new_heap_is_empty(make_heap):
    heap = make_heap()
    assert len(heap) == 0
    assert not heap
    heap.check_invariants()
    with pytest.raises(IndexError):
        heap.find_min()
    with pytest.raises(IndexError):
        heap.extract_min()


def test_insert_returns_a_handle_with_key_and_value(make_heap):
    heap = make_heap()
    handle = heap.insert(5, "five")
    assert (handle.key, handle.value) == (5, "five")
    assert len(heap) == 1
    assert heap
    assert heap.find_min() is handle


def test_value_defaults_to_none(make_heap):
    heap = make_heap()
    heap.insert(3)
    assert heap.extract_min() == (3, None)


def test_find_min_tracks_the_running_minimum(make_heap, random_keys):
    heap = make_heap()
    smallest = random_keys[0]
    for key in random_keys[:500]:
        heap.insert(key)
        smallest = min(smallest, key)
        assert heap.find_min().key == smallest


def test_find_min_does_not_remove_the_item(make_heap):
    heap, _ = filled(make_heap, [4, 1, 3])
    assert heap.find_min().key == 1
    assert heap.find_min().key == 1
    assert len(heap) == 3


# --- Extract-min ---------------------------------------------------------------


@pytest.mark.parametrize("keys", list(KEY_SETS.values()), ids=list(KEY_SETS))
def test_drain_returns_the_keys_in_sorted_order(make_heap, keys):
    heap, _ = filled(make_heap, keys)
    heap.check_invariants()
    assert drained_keys(heap) == sorted(keys)
    assert len(heap) == 0
    heap.check_invariants()


def test_extract_min_returns_the_value_stored_with_the_key(make_heap, random_keys):
    heap, _ = filled(make_heap, random_keys)
    for key, index in heap.drain():
        assert random_keys[index] == key


def test_interleaved_inserts_and_extracts_stay_sorted(make_heap):
    rng = random.Random(7)
    heap = make_heap()
    for _ in range(50):
        for _ in range(rng.randrange(1, 40)):
            heap.insert(rng.randrange(1_000))
        extracted = [heap.extract_min()[0] for _ in range(rng.randrange(len(heap) + 1))]
        assert extracted == sorted(extracted)
        if heap and extracted:
            assert heap.find_min().key >= extracted[-1]
        heap.check_invariants()


def test_len_tracks_every_operation(make_heap):
    heap, handles = filled(make_heap, [5, 3, 8, 1])
    assert len(heap) == 4
    heap.extract_min()
    assert len(heap) == 3
    heap.delete(handles[2])
    assert len(heap) == 2
    heap.decrease_key(handles[0], 0)
    assert len(heap) == 2
    other, _ = filled(make_heap, [9, 7])
    heap.merge(other)
    assert len(heap) == 4
    assert len(other) == 0


# --- Decrease-key ---------------------------------------------------------------


def test_decrease_key_to_a_new_minimum(make_heap, random_keys):
    heap, handles = filled(make_heap, random_keys)
    heap.extract_min()  # gives the lazy heaps a real tree structure
    target = max(handles, key=lambda handle: handle.key)
    heap.decrease_key(target, -1)
    assert target.key == -1
    assert heap.find_min() is target
    heap.check_invariants()


def test_decrease_key_of_the_minimum_itself(make_heap):
    heap, handles = filled(make_heap, [2, 5, 9])
    heap.decrease_key(handles[0], -3)
    assert heap.find_min() is handles[0]
    assert drained_keys(heap) == [-3, 5, 9]


def test_many_decrease_keys_keep_every_item_in_order(make_heap, random_keys):
    rng = random.Random(3)
    heap, handles = filled(make_heap, random_keys)
    removed = {heap.extract_min()[1] for _ in range(10)}
    live = [handle for handle in handles if handle.value not in removed]
    for handle in rng.sample(live, 300):
        heap.decrease_key(handle, handle.key - rng.randrange(500_000))
    heap.check_invariants()
    expected = sorted(handle.key for handle in live)
    assert drained_keys(heap) == expected


def test_decrease_key_with_an_equal_key_changes_nothing(make_heap):
    heap, handles = filled(make_heap, [4, 2, 6])
    heap.decrease_key(handles[0], 4)
    heap.check_invariants()
    assert drained_keys(heap) == [2, 4, 6]


def test_decrease_key_rejects_a_larger_key(make_heap):
    heap, handles = filled(make_heap, [5, 7])
    with pytest.raises(ValueError):
        heap.decrease_key(handles[0], 6)
    assert handles[0].key == 5
    heap.check_invariants()
    assert drained_keys(heap) == [5, 7]


def test_decrease_key_of_a_removed_item_is_rejected(make_heap):
    heap, handles = filled(make_heap, [1, 2])
    heap.extract_min()
    with pytest.raises(ValueError):
        heap.decrease_key(handles[0], 0)
    assert drained_keys(heap) == [2]


# --- Delete --------------------------------------------------------------------------


def test_delete_the_minimum(make_heap):
    heap, _ = filled(make_heap, [3, 1, 2])
    heap.delete(heap.find_min())
    heap.check_invariants()
    assert drained_keys(heap) == [2, 3]


def test_delete_arbitrary_items(make_heap, random_keys):
    rng = random.Random(4)
    heap, handles = filled(make_heap, random_keys)
    extracted = heap.extract_min()[1]  # gives the lazy heaps a real tree structure
    live = [handle for handle in handles if handle.value != extracted]
    victims = rng.sample(live, len(live) // 2)
    for handle in victims:
        heap.delete(handle)
    heap.check_invariants()
    victim_ids = {id(handle) for handle in victims}
    expected = sorted(handle.key for handle in live if id(handle) not in victim_ids)
    assert drained_keys(heap) == expected


def test_delete_the_only_item_empties_the_heap(make_heap):
    heap = make_heap()
    heap.delete(heap.insert(1))
    assert len(heap) == 0
    heap.check_invariants()
    with pytest.raises(IndexError):
        heap.find_min()


def test_delete_of_a_removed_item_is_rejected(make_heap):
    heap, handles = filled(make_heap, [1, 2, 3])
    heap.delete(handles[1])
    with pytest.raises(ValueError):
        heap.delete(handles[1])
    with pytest.raises(ValueError):
        heap.decrease_key(handles[1], 0)
    assert drained_keys(heap) == [1, 3]


# --- Merge -----------------------------------------------------------------------------


def test_merge_two_heaps(make_heap, random_keys):
    half = NUM_KEYS // 2
    heap, _ = filled(make_heap, random_keys[:half])
    other, _ = filled(make_heap, random_keys[half:])
    heap.merge(other)
    assert len(heap) == NUM_KEYS
    assert len(other) == 0
    assert not other
    with pytest.raises(IndexError):
        other.find_min()
    heap.check_invariants()
    other.check_invariants()
    assert drained_keys(heap) == sorted(random_keys)


def test_merge_with_empty_heaps(make_heap):
    empty, full = make_heap(), filled(make_heap, [2, 1])[0]
    full.merge(make_heap())
    empty.merge(full)
    assert len(empty) == 2
    assert len(full) == 0
    assert drained_keys(empty) == [1, 2]


def test_merge_keeps_the_smaller_minimum(make_heap):
    heap, _ = filled(make_heap, [5, 6])
    other, _ = filled(make_heap, [3, 9])
    heap.merge(other)
    assert heap.find_min().key == 3


def test_handles_of_the_merged_heap_stay_valid(make_heap):
    heap, _ = filled(make_heap, [10, 20, 30])
    other, other_handles = filled(make_heap, [15, 25, 35])
    heap.merge(other)
    heap.decrease_key(other_handles[2], 5)
    heap.delete(other_handles[0])
    heap.check_invariants()
    assert drained_keys(heap) == [5, 10, 20, 25, 30]


def test_the_emptied_heap_is_reusable_after_merge(make_heap):
    heap, _ = filled(make_heap, [1])
    other, _ = filled(make_heap, [2])
    heap.merge(other)
    other.insert(7)
    other.check_invariants()
    assert drained_keys(other) == [7]
    assert drained_keys(heap) == [1, 2]


def test_merge_rejects_a_heap_of_another_class(make_heap):
    heap = make_heap()
    with pytest.raises(TypeError):
        heap.merge(OtherHeap())


def test_merge_rejects_the_heap_itself(make_heap):
    heap, _ = filled(make_heap, [1, 2])
    with pytest.raises(ValueError):
        heap.merge(heap)
    assert len(heap) == 2


# --- Handles ----------------------------------------------------------------------------


def test_handles_stay_valid_after_many_extracts(make_heap, random_keys):
    heap, handles = filled(make_heap, random_keys)
    extracted = {heap.extract_min()[1] for _ in range(500)}
    remaining = [handle for handle in handles if handle.value not in extracted]
    for handle in remaining[::7]:
        heap.decrease_key(handle, handle.key - 1_000_000)
    heap.check_invariants()
    assert drained_keys(heap) == sorted(handle.key for handle in remaining)


# --- Randomized differential test ------------------------------------------------------


class LiveItems:
    """Handles of the items currently in the heap, with O(1) random pick and removal."""

    def __init__(self) -> None:
        """Create an empty collection."""
        self.handles: list[Handle] = []
        self.position: dict[int, int] = {}  # item id (its value) -> index in handles

    def add(self, handle: Handle) -> None:
        """Track a newly inserted item.

        Args:
            handle: Handle whose value is the item's unique id.
        """
        self.position[handle.value] = len(self.handles)
        self.handles.append(handle)

    def remove(self, item_id: int) -> None:
        """Stop tracking an item (swap with the last handle, then pop).

        Args:
            item_id: Unique id (value) of the removed item.
        """
        index = self.position.pop(item_id)
        last = self.handles.pop()
        if index < len(self.handles):
            self.handles[index] = last
            self.position[last.value] = index

    def pick(self, rng: random.Random) -> Handle:
        """Choose a tracked handle uniformly at random.

        Args:
            rng: Random number generator.

        Returns:
            A handle of an item in the heap.
        """
        return self.handles[rng.randrange(len(self.handles))]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_operation_sequences_match_an_oracle(make_heap, seed):
    """10,000 random operations; the oracle is a plain dict {item id: key}."""
    rng = random.Random(seed)
    heap, oracle, live = make_heap(), {}, LiveItems()
    next_id = 0

    def insert_into(target: MergeableHeap) -> None:
        nonlocal next_id
        key = rng.randrange(1_000_000)
        live.add(target.insert(key, next_id))
        oracle[next_id] = key
        next_id += 1

    for step in range(10_000):
        choice = rng.random()
        if choice < 0.45 or not oracle:
            insert_into(heap)
        elif choice < 0.65:
            key, item_id = heap.extract_min()
            assert key == min(oracle.values())
            assert oracle.pop(item_id) == key
            live.remove(item_id)
        elif choice < 0.85:
            handle = live.pick(rng)
            if rng.random() < 0.5:
                new_key = handle.key - rng.randrange(10)  # small decrease
            else:
                new_key = handle.key - rng.randrange(2_000_000)  # often a new minimum
            heap.decrease_key(handle, new_key)
            oracle[handle.value] = new_key
        elif choice < 0.95:
            handle = live.pick(rng)
            heap.delete(handle)
            del oracle[handle.value]
            live.remove(handle.value)
        else:
            other = make_heap()
            for _ in range(rng.randrange(20)):
                insert_into(other)
            heap.merge(other)
            assert len(other) == 0
        assert len(heap) == len(oracle)
        if step % 100 == 0:
            heap.check_invariants()
    heap.check_invariants()
    assert drained_keys(heap) == sorted(oracle.values())


# --- Helpers and introspection -----------------------------------------------------------


def test_insert_all_returns_handles_in_order(make_heap):
    heap = make_heap()
    handles = heap.insert_all([3, 1, 2])
    assert [handle.key for handle in handles] == [3, 1, 2]
    assert all(handle.value is None for handle in handles)
    assert len(heap) == 3


def test_repr_shows_the_name_and_size(make_heap):
    heap, _ = filled(make_heap, [1, 2])
    assert repr(heap) == f"{heap.name}(n=2)"


def test_memory_grows_with_the_number_of_items(make_heap, random_keys):
    heap = make_heap()
    empty = heap.memory_bytes()
    heap.insert_all(random_keys)
    assert 0 < empty < heap.memory_bytes()


def test_every_item_has_exactly_one_node(make_heap, random_keys):
    heap, _ = filled(make_heap, random_keys)
    heap.extract_min()
    nodes = list(heap._iter_nodes())
    assert len(nodes) == len(heap)
    assert len({id(node) for node in nodes}) == len(nodes)


# --- Stress (real heaps only: the O(n) reference heap is too slow) ------------------------


@pytest.mark.parametrize("ascending", [True, False], ids=["ascending", "descending"])
def test_long_inputs_do_not_hit_the_recursion_limit(make_real_heap, ascending):
    keys = list(range(STRESS_N)) if ascending else list(range(STRESS_N, 0, -1))
    heap, handles = filled(make_real_heap, keys)
    extracted = heap.extract_min()[1]
    live = [handle for handle in handles if handle.value != extracted]
    for handle in live[::2]:
        heap.decrease_key(handle, handle.key - 2 * STRESS_N)
    heap.check_invariants()
    assert drained_keys(heap) == sorted(handle.key for handle in live)


# --- Implementation rules ----------------------------------------------------------------------

LIBRARY_QUEUE = re.compile(
    r"^\s*(?:import|from)\s+(?:heapq|queue|bisect)\b|\bPriorityQueue\s*\(", re.MULTILINE
)


def test_heaps_do_not_use_library_priority_queues():
    package = Path(mergeable_heaps.__file__).parent
    for path in package.glob("*.py"):
        assert not LIBRARY_QUEUE.search(path.read_text(encoding="utf-8")), path.name
