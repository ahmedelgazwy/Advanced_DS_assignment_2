"""Benchmark the three mergeable heaps and write the results as CSV files.

Three experiments (see docs/theory.md, section 8, for the hypotheses):

1. **Operations** - for n on a 1-2-5 ladder from 1,000 to 10,000,000 keys (prefixes
   of the key dataset). Each heap is built as two heaps of n/2 keys that are
   then merged. Then, with K = min(100,000, n/10):

   * the first extract-min (where lazy heaps pay for their inserts),
   * a warm-up batch of K extract-mins (the pairing heap keeps settling here),
   * a steady-state batch of K extract-mins, and K find-mins,
   * K Dijkstra-like decrease-keys (new key drawn between the current minimum
     and the old key), then one extract-min that pays for them,
   * K decrease-keys to a new global minimum, then one extract-min,
   * memory per item (first run only).

   Median of ``--repeats`` runs for n <= 1,000,000; one run above.
2. **Key order** - the first extract-min after n inserts in ascending and in
   descending key order (random order is covered by experiment 1).
3. **Dijkstra** - V = 10,000 ... 1,000,000 vertices with E = 10 V edges; all
   heaps must return identical distances.

Outputs (in ``--out-dir``): ``operations.csv``, ``key_order.csv``,
``dijkstra.csv`` and ``environment.json``. Rows are written as soon as they are
measured, so an interrupted run keeps its partial results.

Usage (from the repository root)::

    python -m scripts.run_benchmarks            # full run, about 30 minutes
    python -m scripts.run_benchmarks --quick    # setup check, under a minute

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from mergeable_heaps import HEAP_CLASSES, MergeableHeap
from mergeable_heaps.dataset import SOURCE, load_or_generate_graph, load_or_generate_keys
from mergeable_heaps.dijkstra import dijkstra, distance_checksum
from mergeable_heaps.experiment import measure

MIN_N = 1_000
MIN_VERTICES = 10_000
MAX_BATCH = 100_000  # operations per timed batch
SINGLE_RUN_ABOVE = 1_000_000  # larger sizes are measured once
EDGES_PER_VERTEX = 10
ORDERS = ("ascending", "descending")

OPERATION_FIELDS = [
    "structure", "n", "repeats", "batch", "insert_us", "merge_us", "first_extract_ms",
    "warmup_extract_us", "extract_us", "find_min_us", "decrease_us", "after_decrease_ms",
    "decrease_to_min_us", "after_decrease_to_min_ms", "memory_bytes", "bytes_per_item",
]
KEY_ORDER_FIELDS = ["structure", "order", "n", "insert_us", "trees_before", "first_extract_ms"]
DIJKSTRA_FIELDS = [
    "structure", "vertices", "edges", "repeats", "seconds", "inserts", "extract_mins",
    "decrease_keys", "edges_scanned", "reachable", "distance_sum",
]


def size_ladder(low: int, high: int) -> list[int]:
    """Sizes 1, 2 and 5 times each power of ten within ``[low, high]``.

    Args:
        low: Smallest size.
        high: Largest size.

    Returns:
        Increasing list of sizes.
    """
    sizes, power = [], 1
    while power <= high:
        sizes += [step * power for step in (1, 2, 5) if low <= step * power <= high]
        power *= 10
    return sizes


def batch_size(n: int) -> int:
    """Operations per timed batch for a heap of ``n`` items: min(100,000, n/10).

    Args:
        n: Heap size.

    Returns:
        Batch size (at least 1).
    """
    return max(1, min(MAX_BATCH, n // 10))


def live_sample(handles: list, current_min: object, count: int, rng: random.Random) -> list:
    """Pick handles of items that are certainly still in the heap.

    Every extracted key is at most the current minimum, so a handle whose key
    is strictly larger was never extracted.

    Args:
        handles: All handles returned by the inserts.
        current_min: Smallest key now in the heap.
        count: Number of handles wanted.
        rng: Random number generator.

    Returns:
        Up to ``count`` distinct live handles.
    """
    live = [handle for handle in handles if handle.key > current_min]
    return rng.sample(live, min(count, len(live)))


def run_operations(heap_class: type[MergeableHeap], keys: list[int], with_memory: bool) -> dict:
    """One run of experiment 1 on one heap.

    Args:
        heap_class: Heap to measure.
        keys: The n keys.
        with_memory: Also measure memory (a full traversal, slow for large n).

    Returns:
        Measurements keyed by the ``OPERATION_FIELDS`` names (seconds converted
        to the unit in the name).
    """
    n, batch, rng = len(keys), batch_size(len(keys)), random.Random(len(keys))
    heap, other = heap_class(), heap_class()
    handles, first_half = measure(lambda: heap.insert_all(keys[: n // 2]))
    other_handles, second_half = measure(lambda: other.insert_all(keys[n // 2:]))
    handles += other_handles
    del other_handles
    _, merge = measure(lambda: heap.merge(other))
    _, first_extract = measure(heap.extract_min)
    _, warm_up = measure(lambda: [heap.extract_min() for _ in range(batch)])
    _, extracts = measure(lambda: [heap.extract_min() for _ in range(batch)])
    _, find_mins = measure(lambda: [heap.find_min() for _ in range(batch)])

    current_min = heap.find_min().key
    targets = live_sample(handles, current_min, batch, rng)
    new_keys = [rng.randint(current_min, handle.key) for handle in targets]
    _, decreases = measure(
        lambda: [heap.decrease_key(handle, key) for handle, key in zip(targets, new_keys)]
    )
    _, after_decrease = measure(heap.extract_min)

    current_min = heap.find_min().key
    targets = live_sample(handles, current_min, batch, rng)
    _, to_min = measure(
        lambda: [heap.decrease_key(handle, current_min - 1 - i) for i, handle in enumerate(targets)]
    )
    _, after_to_min = measure(heap.extract_min)

    memory = heap.memory_bytes() if with_memory else None
    result = {
        "insert_us": (first_half + second_half) / n * 1e6,
        "merge_us": merge * 1e6,
        "first_extract_ms": first_extract * 1e3,
        "warmup_extract_us": warm_up / batch * 1e6,
        "extract_us": extracts / batch * 1e6,
        "find_min_us": find_mins / batch * 1e6,
        "decrease_us": decreases / max(1, len(new_keys)) * 1e6,
        "after_decrease_ms": after_decrease * 1e3,
        "decrease_to_min_us": to_min / max(1, len(targets)) * 1e6,
        "after_decrease_to_min_ms": after_to_min * 1e3,
        "memory_bytes": memory if memory is not None else "",
        "bytes_per_item": round(memory / len(heap), 2) if memory is not None else "",
    }
    del heap, handles, targets
    gc.collect()  # nodes form reference cycles; free them before the next run
    return result


def measure_operations(heap_class: type[MergeableHeap], keys: list[int], repeats: int) -> dict:
    """Experiment 1 for one heap and one size: median of ``repeats`` runs.

    Args:
        heap_class: Heap to measure.
        keys: The n keys.
        repeats: Number of runs.

    Returns:
        One ``operations.csv`` row.
    """
    runs = [run_operations(heap_class, keys, with_memory=(repeat == 0)) for repeat in range(repeats)]
    row = {"structure": heap_class.name, "n": len(keys), "repeats": repeats,
           "batch": batch_size(len(keys))}
    for field in OPERATION_FIELDS[4:]:
        values = [run[field] for run in runs if run[field] != ""]
        row[field] = statistics.median(values) if values else ""
    return row


def measure_key_order(heap_class: type[MergeableHeap], n: int, order: str) -> dict:
    """Experiment 2: build from keys in sorted order, then time the first extract-min.

    Args:
        heap_class: Heap to measure.
        n: Number of keys.
        order: ``"ascending"`` or ``"descending"``.

    Returns:
        One ``key_order.csv`` row.
    """
    keys = range(n) if order == "ascending" else range(n, 0, -1)
    heap = heap_class()
    _, build = measure(lambda: heap.insert_all(keys))
    trees_before = heap.tree_count()
    _, first_extract = measure(heap.extract_min)
    del heap
    gc.collect()
    return {"structure": heap_class.name, "order": order, "n": n, "insert_us": build / n * 1e6,
            "trees_before": trees_before, "first_extract_ms": first_extract * 1e3}


def measure_dijkstra(graph, repeats: int) -> list[dict]:
    """Experiment 3: Dijkstra with every heap on one graph.

    Args:
        graph: Graph whose vertices are all reachable from ``SOURCE``.
        repeats: Runs per heap; the median time is reported.

    Returns:
        One ``dijkstra.csv`` row per heap.

    Raises:
        RuntimeError: If two heaps return different distances.
    """
    rows, reference = [], None
    for heap_class in HEAP_CLASSES:
        times = []
        for _ in range(repeats):
            (distances, stats), seconds = measure(lambda: dijkstra(graph, SOURCE, heap_class))
            times.append(seconds)
        if reference is None:
            reference = distances
        elif distances != reference:
            raise RuntimeError(f"{heap_class.name} returned different distances")
        reachable, distance_sum = distance_checksum(distances)
        rows.append({
            "structure": heap_class.name, "vertices": graph.num_vertices,
            "edges": graph.num_edges, "repeats": repeats, "seconds": statistics.median(times),
            "inserts": stats.inserts, "extract_mins": stats.extract_mins,
            "decrease_keys": stats.decrease_keys, "edges_scanned": stats.edges_scanned,
            "reachable": reachable, "distance_sum": distance_sum,
        })
        del distances
        gc.collect()
    return rows


def cpu_name() -> str:
    """Best-effort human-readable CPU model name.

    Returns:
        The CPU brand string, or ``platform.processor()`` as a fallback.
    """
    try:
        import winreg  # Windows only

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def total_ram_bytes() -> int | None:
    """Best-effort physical memory size.

    Returns:
        Bytes of RAM, or None if it cannot be determined.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        pass
    try:
        import ctypes

        kilobytes = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(kilobytes)):
            return kilobytes.value * 1024
    except (AttributeError, OSError):
        pass
    return None


def environment_info(args: argparse.Namespace) -> dict[str, object]:
    """Describe the machine and settings of this run.

    Args:
        args: Parsed command-line options.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "ram_bytes": total_ram_bytes(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "max_n": args.max_n,
        "order_max_n": args.order_max_n,
        "dijkstra_max_v": args.dijkstra_max_v,
        "repeats": args.repeats,
        "max_batch": MAX_BATCH,
        "single_run_above": SINGLE_RUN_ABOVE,
        "edges_per_vertex": EDGES_PER_VERTEX,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options; ``--quick`` shrinks every experiment.

    Args:
        argv: Arguments to parse (default: ``sys.argv[1:]``).

    Returns:
        Namespace with the run settings.
    """
    parser = argparse.ArgumentParser(description="Benchmark the three mergeable heaps.")
    parser.add_argument("--quick", action="store_true", help="small run to check the setup")
    parser.add_argument("--max-n", type=int, default=10_000_000, help="largest heap size")
    parser.add_argument("--order-max-n", type=int, default=None,
                        help="largest size for the key-order experiment (default: --max-n)")
    parser.add_argument("--dijkstra-max-v", type=int, default=1_000_000,
                        help="largest Dijkstra graph (vertices; 0 to skip)")
    parser.add_argument("--repeats", type=int, default=3, help="runs per size up to 1,000,000")
    parser.add_argument("--keys", type=Path, default=Path("data/keys_10000000.txt.gz"))
    parser.add_argument("--graph", type=Path, default=Path("data/graph_1000000_10000000.txt.gz"))
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output folder (default results/, or results/quick/ with --quick)")
    args = parser.parse_args(argv)
    if args.quick:
        args.max_n, args.dijkstra_max_v, args.repeats = 100_000, 20_000, 1
        args.order_max_n = min(args.order_max_n or args.max_n, 100_000)
    if args.order_max_n is None:
        args.order_max_n = args.max_n
    if args.out_dir is None:
        args.out_dir = Path("results/quick" if args.quick else "results")
    return args


def open_csv(path: Path, fields: list[str]) -> tuple[IO[str], csv.DictWriter]:
    """Create a CSV file and write its header.

    Args:
        path: File to create.
        fields: Column names.

    Returns:
        Tuple ``(file handle, writer)``; the caller closes the handle.
    """
    handle = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return handle, writer


def main(argv: list[str] | None = None) -> None:
    """Run the three experiments and write the CSV files.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        None.
    """
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "environment.json").write_text(
        json.dumps(environment_info(args), indent=2) + "\n", encoding="utf-8"
    )
    start = time.perf_counter()

    def elapsed() -> str:
        return f"[{time.perf_counter() - start:6.0f}s]"

    keys, from_file = load_or_generate_keys(args.keys, args.max_n)
    print(f"{len(keys):,} keys {'read from ' + str(args.keys) if from_file else 'generated'}")

    print("\n1. Operations")
    handle, writer = open_csv(args.out_dir / "operations.csv", OPERATION_FIELDS)
    with handle:
        for n in size_ladder(MIN_N, args.max_n):
            repeats = args.repeats if n <= SINGLE_RUN_ABOVE else 1
            subset = keys if n == len(keys) else keys[:n]
            for heap_class in HEAP_CLASSES:
                row = measure_operations(heap_class, subset, repeats)
                writer.writerow(row)
                handle.flush()
                print(f"  n={n:>11,}  {heap_class.name:15} insert {row['insert_us']:5.2f}us  "
                      f"extract {row['extract_us']:6.2f}us  decrease {row['decrease_us']:5.2f}us  "
                      f"{elapsed()}", flush=True)
            del subset
    del keys
    gc.collect()

    print("\n2. Key order")
    handle, writer = open_csv(args.out_dir / "key_order.csv", KEY_ORDER_FIELDS)
    with handle:
        for n in size_ladder(MIN_N, args.order_max_n):
            for order in ORDERS:
                for heap_class in HEAP_CLASSES:
                    row = measure_key_order(heap_class, n, order)
                    writer.writerow(row)
                    handle.flush()
            print(f"  n={n:>11,}  done {elapsed()}", flush=True)

    print("\n3. Dijkstra")
    handle, writer = open_csv(args.out_dir / "dijkstra.csv", DIJKSTRA_FIELDS)
    with handle:
        for vertices in size_ladder(MIN_VERTICES, args.dijkstra_max_v):
            graph, graph_from_file = load_or_generate_graph(
                args.graph, vertices, EDGES_PER_VERTEX * vertices
            )
            repeats = args.repeats if vertices <= SINGLE_RUN_ABOVE // 10 else 1
            for row in measure_dijkstra(graph, repeats):
                writer.writerow(row)
                handle.flush()
                print(f"  V={vertices:>9,}  {row['structure']:15} {row['seconds']:7.2f}s  "
                      f"decrease-keys {row['decrease_keys']:>9,}"
                      f"{'  (graph file)' if graph_from_file else ''}  {elapsed()}", flush=True)
            del graph
            gc.collect()
    print(f"\nDone in {time.perf_counter() - start:.0f}s; results in {args.out_dir}/")


if __name__ == "__main__":
    main()
