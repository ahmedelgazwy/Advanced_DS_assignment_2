"""Tests for Dijkstra's algorithm on every heap.

Each test runs once per heap (reference heap included) through the
``make_heap`` fixture. The oracle is a heap-free O(V^2) Dijkstra.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import math
import random

import pytest

from mergeable_heaps import HEAP_CLASSES
from mergeable_heaps.dataset import SOURCE, Graph, generate_graph
from mergeable_heaps.dijkstra import dijkstra, distance_checksum
from reference_heap import ReferenceHeap

# CLRS 3rd ed., Figure 24.6: vertices s, t, x, y, z = 0..4.
CLRS_EDGES = [
    (0, 1, 10), (0, 3, 5), (1, 2, 1), (1, 3, 2), (2, 4, 4),
    (3, 1, 3), (3, 2, 9), (3, 4, 2), (4, 0, 7), (4, 2, 6),
]
CLRS_DISTANCES = [0, 8, 9, 5, 7]


def graph_from(num_vertices: int, edges: list[tuple[int, int, int]]) -> Graph:
    """Build a graph from ``(source, target, weight)`` triples."""
    sources, targets, weights = zip(*edges) if edges else ((), (), ())
    return Graph.from_edges(num_vertices, sources, targets, weights)


def quadratic_dijkstra(graph: Graph, source: int) -> list[float]:
    """Reference O(V^2) Dijkstra: repeatedly scan for the closest unfinished vertex."""
    distances = [math.inf] * graph.num_vertices
    distances[source] = 0
    finished = [False] * graph.num_vertices
    for _ in range(graph.num_vertices):
        vertex = min(
            (v for v in range(graph.num_vertices) if not finished[v]),
            key=lambda v: distances[v],
        )
        if distances[vertex] == math.inf:
            break
        finished[vertex] = True
        for target, weight in graph.edges_from(vertex):
            distances[target] = min(distances[target], distances[vertex] + weight)
    return distances


def random_graph(seed: int, num_vertices: int, num_edges: int, max_weight: int) -> Graph:
    """A random graph without the reachability guarantee (some vertices may be unreachable)."""
    rng = random.Random(seed)
    edges = [
        (rng.randrange(num_vertices), rng.randrange(num_vertices), rng.randint(0, max_weight))
        for _ in range(num_edges)
    ]
    return graph_from(num_vertices, edges)


def test_clrs_example(make_heap):
    distances, _ = dijkstra(graph_from(5, CLRS_EDGES), 0, make_heap)
    assert distances == CLRS_DISTANCES


@pytest.mark.parametrize("seed", range(5))
def test_matches_the_quadratic_reference_on_random_graphs(make_heap, seed):
    graph = random_graph(seed, num_vertices=60, num_edges=300, max_weight=20)
    distances, _ = dijkstra(graph, 0, make_heap)
    assert distances == quadratic_dijkstra(graph, 0)


def test_unreachable_vertices_get_infinity(make_heap):
    graph = graph_from(4, [(0, 1, 3), (2, 3, 1)])
    distances, stats = dijkstra(graph, 0, make_heap)
    assert distances == [0, 3, math.inf, math.inf]
    assert stats.extract_mins == 2


def test_zero_weight_edges_and_self_loops(make_heap):
    graph = graph_from(4, [(0, 1, 0), (1, 2, 0), (2, 2, 0), (0, 2, 5), (2, 3, 0)])
    distances, _ = dijkstra(graph, 0, make_heap)
    assert distances == [0, 0, 0, 0]


def test_parallel_edges_use_the_cheapest(make_heap):
    graph = graph_from(2, [(0, 1, 9), (0, 1, 4), (0, 1, 7)])
    distances, stats = dijkstra(graph, 0, make_heap)
    assert distances == [0, 4]
    assert stats.decrease_keys == 1  # 9 -> 4; 7 is no improvement


def test_a_graph_with_one_vertex(make_heap):
    distances, stats = dijkstra(graph_from(1, []), 0, make_heap)
    assert distances == [0]
    assert (stats.inserts, stats.extract_mins, stats.decrease_keys) == (1, 1, 0)


def test_invalid_source_is_rejected(make_heap):
    with pytest.raises(ValueError):
        dijkstra(graph_from(3, [(0, 1, 1)]), 3, make_heap)


def test_operation_counts_are_consistent(make_heap):
    graph = generate_graph(2_000, 20_000)
    distances, stats = dijkstra(graph, SOURCE, make_heap)
    reachable, _ = distance_checksum(distances)
    assert reachable == 2_000  # the generator guarantees reachability
    assert stats.inserts == stats.extract_mins == reachable
    assert stats.edges_scanned == graph.num_edges
    assert stats.inserts - 1 + stats.decrease_keys <= stats.edges_scanned


def test_every_heap_gives_identical_distances_on_a_generated_graph():
    graph = generate_graph(3_000, 30_000, seed=3)
    expected, _ = dijkstra(graph, SOURCE, ReferenceHeap)
    for heap_class in HEAP_CLASSES:
        distances, _ = dijkstra(graph, SOURCE, heap_class)
        assert distances == expected, heap_class.name


def test_distance_checksum():
    assert distance_checksum([0, 3, math.inf, 4]) == (3, 7)
