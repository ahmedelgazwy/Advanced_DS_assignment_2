"""Tests for the key and graph generators, the file formats and the generator script.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import json
import sys
from collections import deque

import pytest

from mergeable_heaps.dataset import (
    KEY_BITS,
    MAX_WEIGHT,
    SOURCE,
    Graph,
    generate_graph,
    generate_keys,
    load_graph,
    load_keys,
    load_or_generate_graph,
    load_or_generate_keys,
    save_graph,
    save_keys,
    sha256_of,
)
from scripts import generate_dataset


def reachable_from(graph: Graph, source: int) -> set[int]:
    """Vertices reachable from ``source`` (breadth-first search)."""
    seen, queue = {source}, deque([source])
    while queue:
        vertex = queue.popleft()
        for target, _ in graph.edges_from(vertex):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def edge_list(graph: Graph) -> list[tuple[int, int, int]]:
    """All edges as ``(source, target, weight)`` triples in CSR order."""
    return [
        (source, target, weight)
        for source in range(graph.num_vertices)
        for target, weight in graph.edges_from(source)
    ]


# --- Keys -----------------------------------------------------------------------------


def test_keys_are_deterministic_and_seeded():
    assert generate_keys(1_000) == generate_keys(1_000)
    assert generate_keys(1_000, seed=1) != generate_keys(1_000, seed=2)


def test_keys_are_in_range():
    keys = generate_keys(20_000)
    assert len(keys) == 20_000
    assert all(0 <= key < 2**KEY_BITS for key in keys)
    assert max(keys) > 2 ** (KEY_BITS - 1)  # the whole range is used


def test_smaller_key_sets_are_prefixes_of_larger_ones():
    assert generate_keys(100) == generate_keys(5_000)[:100]


def test_negative_key_count_is_rejected():
    with pytest.raises(ValueError):
        generate_keys(-1)


def test_keys_round_trip_through_a_gzip_file(tmp_path):
    path = tmp_path / "keys.txt.gz"
    keys = generate_keys(1_000)
    assert save_keys(path, keys) == 1_000
    assert load_keys(path) == keys
    assert load_keys(path, limit=10) == keys[:10]


def test_key_files_are_byte_identical_when_regenerated(tmp_path):
    first, second = tmp_path / "a.txt.gz", tmp_path / "b.txt.gz"
    save_keys(first, generate_keys(1_000))
    save_keys(second, generate_keys(1_000))
    assert first.read_bytes() == second.read_bytes()


def test_load_or_generate_uses_the_file_only_if_it_is_large_enough(tmp_path):
    path = tmp_path / "keys.txt.gz"
    assert load_or_generate_keys(path, 50) == (generate_keys(50), False)  # missing file
    save_keys(path, generate_keys(100))
    assert load_or_generate_keys(path, 50) == (generate_keys(50), True)
    assert load_or_generate_keys(path, 200) == (generate_keys(200), False)  # file too small


# --- Graphs ----------------------------------------------------------------------------


def test_graph_has_the_requested_size_and_valid_csr_arrays():
    graph = generate_graph(1_000, 10_000)
    assert (graph.num_vertices, graph.num_edges) == (1_000, 10_000)
    assert len(graph.offsets) == 1_001
    assert graph.offsets[0] == 0 and graph.offsets[-1] == 10_000
    assert all(a <= b for a, b in zip(graph.offsets, graph.offsets[1:]))
    assert graph.memory_bytes() == 4 * (1_001 + 2 * 10_000)


def test_graph_edges_are_valid():
    graph = generate_graph(500, 5_000)
    for source, target, weight in edge_list(graph):
        assert 0 <= target < 500
        assert target != source  # no self-loops
        assert 1 <= weight <= MAX_WEIGHT


def test_every_vertex_is_reachable_from_the_source():
    graph = generate_graph(2_000, 2_000 - 1)  # only the Hamiltonian path
    assert reachable_from(graph, SOURCE) == set(range(2_000))


def test_graph_is_deterministic_and_seeded():
    assert edge_list(generate_graph(300, 3_000)) == edge_list(generate_graph(300, 3_000))
    assert edge_list(generate_graph(300, 3_000, seed=1)) != edge_list(generate_graph(300, 3_000, seed=2))


def test_tiny_graphs():
    single = generate_graph(1, 0)
    assert (single.num_vertices, single.num_edges) == (1, 0)
    pair = generate_graph(2, 5)
    assert all(target != source for source, target, _ in edge_list(pair))


@pytest.mark.parametrize(("vertices", "edges"), [(0, 0), (10, 8), (1, 1)])
def test_impossible_graph_sizes_are_rejected(vertices, edges):
    with pytest.raises(ValueError):
        generate_graph(vertices, edges)


def test_from_edges_groups_edges_by_source_and_keeps_their_order():
    graph = Graph.from_edges(3, [2, 0, 2, 0], [0, 1, 1, 2], [5, 6, 7, 8])
    assert list(graph.edges_from(0)) == [(1, 6), (2, 8)]
    assert list(graph.edges_from(1)) == []
    assert list(graph.edges_from(2)) == [(0, 5), (1, 7)]


@pytest.mark.parametrize(
    ("sources", "targets", "weights"),
    [([0], [3], [1]), ([-1], [0], [1]), ([0], [1], [-5]), ([0, 1], [1], [1, 1])],
    ids=["target_out_of_range", "negative_vertex", "negative_weight", "length_mismatch"],
)
def test_from_edges_rejects_invalid_edges(sources, targets, weights):
    with pytest.raises(ValueError):
        Graph.from_edges(3, sources, targets, weights)


def test_graph_round_trips_through_a_gzip_file(tmp_path):
    path = tmp_path / "graph.txt.gz"
    graph = generate_graph(200, 2_000)
    save_graph(path, graph)
    loaded = load_graph(path)
    assert (loaded.offsets, loaded.targets, loaded.weights) == (
        graph.offsets, graph.targets, graph.weights,
    )


def test_edgeless_graph_round_trips(tmp_path):
    path = tmp_path / "single.txt"
    save_graph(path, generate_graph(1, 0))
    assert load_graph(path).num_vertices == 1


def test_load_or_generate_graph_uses_the_file_only_if_the_size_matches(tmp_path):
    path = tmp_path / "graph.txt.gz"
    expected = edge_list(generate_graph(40, 300))
    graph, from_file = load_or_generate_graph(path, 40, 300)  # missing file
    assert (edge_list(graph), from_file) == (expected, False)
    save_graph(path, generate_graph(40, 300))
    graph, from_file = load_or_generate_graph(path, 40, 300)
    assert (edge_list(graph), from_file) == (expected, True)
    graph, from_file = load_or_generate_graph(path, 50, 300)  # wrong size: generated
    assert (graph.num_vertices, from_file) == (50, False)


def test_graph_file_without_header_is_rejected(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0 1 5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_graph(path)


def test_graph_file_with_a_wrong_edge_count_is_rejected(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("# vertices=2 edges=2\n0 1 5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_graph(path)


# --- Generator script -------------------------------------------------------------------


def test_generator_script_writes_files_and_a_matching_manifest(monkeypatch, tmp_path):
    argv = ["generate_dataset", "--keys", "500", "--vertices", "50", "--edges", "400",
            "--out-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    generate_dataset.main()
    keys_path, graph_path = tmp_path / "keys_500.txt.gz", tmp_path / "graph_50_400.txt.gz"
    assert load_keys(keys_path) == generate_keys(500)
    assert edge_list(load_graph(graph_path)) == edge_list(generate_graph(50, 400))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["keys"]["count"] == 500
    assert (manifest["graph"]["vertices"], manifest["graph"]["edges"]) == (50, 400)
    for path in (keys_path, graph_path):
        assert manifest["sha256"][path.name] == sha256_of(path)
        assert manifest["bytes"][path.name] == path.stat().st_size
