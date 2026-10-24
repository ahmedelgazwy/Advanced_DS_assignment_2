"""Dijkstra's single-source shortest paths on top of any mergeable heap.

This is the application that motivates decrease_key [Dijkstra 1959; Fredman &
Tarjan 1987]. A vertex is inserted into the heap when it is first reached,
its key is decreased whenever a shorter path to it is found, and it is final
when it is extracted. The run counts every heap operation, so the benchmark
can relate running time to the analysis (docs/theory.md, section 5):

    inserts <= V,   extract_mins <= V,   decrease_keys <= E.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from mergeable_heaps.base import Handle, MergeableHeap
from mergeable_heaps.dataset import Graph


@dataclass
class DijkstraStats:
    """Number of heap operations and edge scans performed by one run."""

    inserts: int = 0
    extract_mins: int = 0
    decrease_keys: int = 0
    edges_scanned: int = 0


def dijkstra(
    graph: Graph, source: int, heap_factory: Callable[[], MergeableHeap]
) -> tuple[list[float], DijkstraStats]:
    """Shortest-path distances from ``source`` to every vertex.

    Edge weights must be non-negative (``Graph`` guarantees this). A finished
    vertex ``v`` always has ``dist[v] <= d``, where ``d`` is the key just
    extracted, so the test ``d + w < dist[v]`` never reopens it; no separate
    "finished" flag is needed.

    Args:
        graph: Directed graph in CSR form.
        source: Start vertex.
        heap_factory: Zero-argument function returning an empty heap
            (for example ``PairingHeap``).

    Returns:
        Tuple ``(distances, stats)``. ``distances[v]`` is the length of a
        shortest path from ``source`` to ``v``, or ``math.inf`` if ``v`` is
        unreachable.

    Raises:
        ValueError: If ``source`` is not a vertex of the graph.
    """
    num_vertices = graph.num_vertices
    if not 0 <= source < num_vertices:
        raise ValueError(f"source {source} is not a vertex (V = {num_vertices})")
    offsets, targets, weights = graph.offsets, graph.targets, graph.weights
    distances: list[float] = [math.inf] * num_vertices
    handles: list[Optional[Handle]] = [None] * num_vertices  # set while a vertex is in the heap
    stats = DijkstraStats(inserts=1)
    heap = heap_factory()
    insert, extract_min, decrease_key = heap.insert, heap.extract_min, heap.decrease_key
    distances[source] = 0
    handles[source] = insert(0, source)
    while heap:
        distance, vertex = extract_min()
        stats.extract_mins += 1
        handles[vertex] = None
        start, end = offsets[vertex], offsets[vertex + 1]
        stats.edges_scanned += end - start
        for index in range(start, end):
            target = targets[index]
            candidate = distance + weights[index]
            if candidate < distances[target]:
                distances[target] = candidate
                handle = handles[target]
                if handle is None:
                    handles[target] = insert(candidate, target)
                    stats.inserts += 1
                else:
                    decrease_key(handle, candidate)
                    stats.decrease_keys += 1
    return distances, stats


def distance_checksum(distances: list[float]) -> tuple[int, int]:
    """Short fingerprint of a distance array, for comparing runs.

    Args:
        distances: Output of :func:`dijkstra`.

    Returns:
        Tuple ``(number of reachable vertices, sum of their distances)``.
    """
    finite = [distance for distance in distances if distance != math.inf]
    return len(finite), int(sum(finite))
