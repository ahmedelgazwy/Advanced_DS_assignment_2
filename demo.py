"""Live demo: the binomial, Fibonacci and pairing heaps side by side.

Three parts:

1. **Operations.** Build each heap from n keys (two halves, then merge), and
   compare insert, merge, extract-min, decrease-key and memory, including the
   first extract-min after the inserts, where the lazy heaps pay their debt.
2. **Shapes.** Put the same 14 keys into each heap, extract the minimum, and
   draw the remaining 13 items as trees.
3. **Dijkstra.** Run shortest paths on a random graph with each heap: the
   distances must be identical, the running times are not.

Usage (from the repository root)::

    python demo.py                             # 1,000,000 keys, 100,000-vertex graph (about 20 s)
    python demo.py --n 100000 --graph-v 10000  # a quick run (about 2 s)

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import argparse
import gc
import random
import time
from itertools import zip_longest
from pathlib import Path

from mergeable_heaps import HEAP_CLASSES, MergeableHeap
from mergeable_heaps.dataset import SOURCE, generate_graph, load_or_generate_keys
from mergeable_heaps.dijkstra import dijkstra
from mergeable_heaps.experiment import format_table, measure

DEFAULT_KEYS_FILE = Path("data/keys_10000000.txt.gz")
SHAPE_KEYS = [9, 4, 14, 1, 11, 6, 2, 13, 8, 3, 12, 7, 5, 10]  # 14 keys in a fixed shuffled order
EDGES_PER_VERTEX = 10
MAX_BATCH = 10_000  # operations per timed batch


def measure_operations(heap_class: type[MergeableHeap], keys: list[int], batch: int) -> dict:
    """Time every heap operation on one heap built from ``keys``.

    The keys are inserted into two heaps of half the size, which are then
    merged. Extract-mins are timed three ways: the first one, a warm-up batch
    (where lazy heaps finish paying for their cheap inserts) and a
    steady-state batch. Decrease-keys mimic Dijkstra: each new key is drawn
    between the current minimum and the old key.

    Args:
        heap_class: Heap to measure.
        keys: Keys to insert (at least 20).
        batch: Operations per timed batch.

    Returns:
        Seconds (or bytes) per measurement, keyed by name.
    """
    half = len(keys) // 2
    heap, other = heap_class(), heap_class()
    handles, first_half = measure(lambda: heap.insert_all(keys[:half]))
    other_handles, second_half = measure(lambda: other.insert_all(keys[half:]))
    handles += other_handles
    _, merge = measure(lambda: heap.merge(other))
    _, first_extract = measure(heap.extract_min)
    _, warm_up = measure(lambda: [heap.extract_min() for _ in range(batch)])
    _, extracts = measure(lambda: [heap.extract_min() for _ in range(batch)])

    rng = random.Random(1)
    current_min = heap.find_min().key
    live = [handle for handle in handles if handle.key > current_min]  # never extracted
    targets = rng.sample(live, min(batch, len(live)))
    new_keys = [rng.randint(current_min, handle.key) for handle in targets]
    _, decreases = measure(
        lambda: [heap.decrease_key(handle, key) for handle, key in zip(targets, new_keys)]
    )
    _, after_decreases = measure(heap.extract_min)
    return {
        "insert": (first_half + second_half) / len(keys),
        "merge": merge,
        "first_extract": first_extract,
        "warm_up": warm_up,
        "extract": extracts / batch,
        "decrease": decreases / max(1, len(targets)),
        "after_decreases": after_decreases,
        "bytes_per_item": heap.memory_bytes() / len(heap),
    }


def show_operations(keys: list[int], source: str) -> None:
    """Part 1: measure and print every operation for every heap.

    Args:
        keys: Keys to insert.
        source: Where the keys came from (for the heading).

    Returns:
        None.
    """
    batch = max(1, min(MAX_BATCH, len(keys) // 10))
    print(f"1. Operations on {len(keys):,} keys ({source})\n")
    results = {}
    for heap_class in HEAP_CLASSES:
        results[heap_class.name] = measure_operations(heap_class, keys, batch)
        gc.collect()
    print(format_table(
        ["Heap", "Insert (us)", "Merge (us)", "Extract-min (us)", "Decrease-key (us)", "Bytes/item"],
        [[name, f"{r['insert'] * 1e6:.2f}", f"{r['merge'] * 1e6:.1f}", f"{r['extract'] * 1e6:.2f}",
          f"{r['decrease'] * 1e6:.2f}", f"{r['bytes_per_item']:.0f}"] for name, r in results.items()],
    ))
    print(f"""
   Insert: mean over n inserts (into two halves); Merge: the one merge of the halves.
   Extract-min and Decrease-key: mean of {batch:,} operations in steady state. New keys
   lie between the current minimum and the old key, as in Dijkstra's algorithm.
   O(1) merges still take microseconds on big heaps: the roots are no longer in
   the CPU cache (on tiny heaps a merge takes about 1 us).
""")
    print("   Paying for laziness: cheap inserts and decrease-keys leave work for later extract-mins\n")
    print(format_table(
        ["Heap", "1st extract-min (ms)", f"Next {batch:,} extract-mins (ms)",
         f"Extract-min after {batch:,} decrease-keys (ms)"],
        [[name, f"{r['first_extract'] * 1e3:,.1f}", f"{r['warm_up'] * 1e3:,.1f}",
          f"{r['after_decreases'] * 1e3:,.2f}"] for name, r in results.items()],
    ))
    print(f"""
   The Fibonacci heap consolidates all n one-node trees in its first extract-min.
   The pairing heap settles over its first few extract-mins (compare the warm-up
   with {batch:,} x the steady-state cost above). The binomial heap is always tidy.
   After the decrease-keys, the lazy heaps must re-link the subtrees they cut out.
""")


def show_shapes() -> None:
    """Part 2: draw each heap after 14 inserts and one extract-min.

    Returns:
        None.
    """
    print(f"2. Shapes: insert {SHAPE_KEYS}, then extract the minimum\n")
    columns, notes = [], []
    for heap_class in HEAP_CLASSES:
        heap = heap_class()
        heap.insert_all(SHAPE_KEYS)
        trees_before = heap.tree_count()
        heap.extract_min()
        notes.append(f"{heap_class.name}: {trees_before} tree(s) after the inserts, "
                     f"{heap.tree_count()} after extract-min")
        columns.append([heap_class.name, "-" * len(heap_class.name), *heap.format_trees().splitlines()])
    width = max(len(line) for column in columns for line in column) + 4
    for parts in zip_longest(*columns, fillvalue=""):
        print("   " + "".join(part.ljust(width) for part in parts).rstrip())
    print()
    for note in notes:
        print(f"   {note}")
    print("""
   13 = 0b1101, so the binomial heap holds trees of 8, 4 and 1 items. The
   Fibonacci heap kept 14 separate roots until extract-min consolidated them
   into the same shape. The pairing heap is always a single tree.
""")


def show_dijkstra(num_vertices: int) -> None:
    """Part 3: run Dijkstra with each heap on the same random graph.

    Args:
        num_vertices: Number of vertices; the graph has 10 edges per vertex.

    Returns:
        None.
    """
    num_edges = EDGES_PER_VERTEX * num_vertices
    print(f"3. Dijkstra from vertex {SOURCE} on a random graph with {num_vertices:,} "
          f"vertices and {num_edges:,} edges\n")
    graph = generate_graph(num_vertices, num_edges)
    rows, reference = [], None
    identical = True
    for heap_class in HEAP_CLASSES:
        (distances, stats), seconds = measure(lambda: dijkstra(graph, SOURCE, heap_class))
        if reference is None:
            reference = distances
        identical &= distances == reference
        rows.append([heap_class.name, f"{seconds:.2f}", f"{stats.inserts:,}",
                     f"{stats.extract_mins:,}", f"{stats.decrease_keys:,}"])
    print(format_table(["Heap", "Time (s)", "Inserts", "Extract-mins", "Decrease-keys"], rows))
    verdict = "identical" if identical else "DIFFERENT (this is a bug)"
    print(f"""
   Shortest-path distances from all three heaps: {verdict}.
   Only about one decrease-key per vertex happens, not one per edge: the
   expected number is O(V log(1 + E/V)) [Goldberg & Tarjan 1996], so the
   Fibonacci heap's O(1) decrease-key has little to win.
""")


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        Namespace with ``n``, ``graph_v`` and ``data``.
    """
    parser = argparse.ArgumentParser(description="Compare the three mergeable heaps live.")
    parser.add_argument("--n", type=int, default=1_000_000, help="number of keys (at least 20)")
    parser.add_argument("--graph-v", type=int, default=100_000,
                        help="vertices of the Dijkstra graph (at least 2)")
    parser.add_argument("--data", type=Path, default=DEFAULT_KEYS_FILE,
                        help="key file; missing keys are generated from the same seed")
    args = parser.parse_args()
    if args.n < 20 or args.graph_v < 2:
        parser.error("--n must be at least 20 and --graph-v at least 2")
    return args


def main() -> None:
    """Run the three parts of the demo.

    Returns:
        None.
    """
    args = parse_args()
    start = time.perf_counter()
    keys, from_file = load_or_generate_keys(args.data, args.n)
    source = f"read from {args.data}" if from_file else "generated in memory"
    show_operations(keys, source)
    show_shapes()
    show_dijkstra(args.graph_v)
    print(f"Demo finished in {time.perf_counter() - start:.1f}s.")


if __name__ == "__main__":
    main()
