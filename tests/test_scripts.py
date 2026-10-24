"""Tests for the benchmark and plotting scripts: helpers and a tiny end-to-end run.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import csv
import math
import random

import numpy as np
import pytest

from mergeable_heaps import HEAP_CLASSES, PairingHeap
from scripts import plot_results, run_benchmarks

FIGURES = ["insert_time", "extract_min_time", "first_extract_min", "decrease_key_time",
           "merge_time", "memory", "extract_warmup", "dijkstra_time"]


def read_rows(path) -> list[dict[str, str]]:
    """Rows of a CSV file."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- Benchmark helpers ------------------------------------------------------------------


def test_size_ladder_uses_1_2_5_steps():
    assert run_benchmarks.size_ladder(1_000, 10_000) == [1_000, 2_000, 5_000, 10_000]
    assert run_benchmarks.size_ladder(10_000, 1_000_000)[-3:] == [200_000, 500_000, 1_000_000]
    assert run_benchmarks.size_ladder(3_000, 4_000) == []


@pytest.mark.parametrize(("n", "batch"), [(5, 1), (1_000, 100), (1_000_000, 100_000),
                                          (10_000_000, 100_000)])
def test_batch_size_is_a_tenth_capped_at_100k(n, batch):
    assert run_benchmarks.batch_size(n) == batch


def test_live_sample_only_returns_items_still_in_the_heap():
    heap = PairingHeap()
    handles = heap.insert_all(random.Random(1).sample(range(10_000), 1_000))
    for _ in range(300):
        heap.extract_min()
    sample = run_benchmarks.live_sample(handles, heap.find_min().key, 500, random.Random(2))
    assert len(sample) == 500
    for handle in sample:
        heap.decrease_key(handle, handle.key)  # raises ValueError for an extracted item


def test_quick_mode_shrinks_every_experiment(tmp_path):
    args = run_benchmarks.parse_args(["--quick", "--out-dir", str(tmp_path)])
    assert (args.max_n, args.order_max_n, args.dijkstra_max_v, args.repeats) == (
        100_000, 100_000, 20_000, 1,
    )


# --- Fitting helpers -----------------------------------------------------------------------


def test_loglog_slope_recovers_known_exponents():
    n = np.array([1e4, 1e5, 1e6, 1e7])
    assert plot_results.loglog_slope(n, 3 * n) == pytest.approx(1.0)
    assert plot_results.loglog_slope(n, np.full(4, 2.0)) == pytest.approx(0.0)
    assert math.isnan(plot_results.loglog_slope(np.array([10.0]), np.array([1.0])))


def test_log2_fit_recovers_a_logarithm():
    n = np.array([1e4, 1e5, 1e6, 1e7])
    a, b, r_squared = plot_results.log2_fit(n, 5 + 2 * np.log2(n))
    assert (a, b, r_squared) == (pytest.approx(5), pytest.approx(2), pytest.approx(1))


@pytest.mark.parametrize(("value", "text"), [(500, "500"), (1_000, "1K"), (20_000, "20K"),
                                             (1_000_000, "1M"), (0.05, "0.05")])
def test_compact_axis_labels(value, text):
    assert plot_results.compact(value) == text


# --- End to end -------------------------------------------------------------------------------


def test_benchmark_and_plots_run_end_to_end(tmp_path):
    out = tmp_path / "results"
    run_benchmarks.main([
        "--max-n", "2000", "--dijkstra-max-v", "10000", "--repeats", "2",
        "--keys", str(tmp_path / "no_keys.gz"), "--graph", str(tmp_path / "no_graph.gz"),
        "--out-dir", str(out),
    ])
    operations = read_rows(out / "operations.csv")
    assert len(operations) == 2 * len(HEAP_CLASSES)  # n = 1,000 and 2,000
    assert all(row["repeats"] == "2" and float(row["bytes_per_item"]) > 0 for row in operations)
    assert len(read_rows(out / "key_order.csv")) == 2 * 2 * len(HEAP_CLASSES)
    dijkstra_rows = read_rows(out / "dijkstra.csv")
    assert len({row["distance_sum"] for row in dijkstra_rows}) == 1  # identical distances

    plot_results.main(["--results", str(out)])
    for name in FIGURES:
        assert (out / "figures" / f"{name}.pdf").stat().st_size > 0
        assert (out / "figures" / f"{name}.png").stat().st_size > 0
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Operations at n = 2,000" in summary and "## Dijkstra at V = 10,000" in summary
    assert len(read_rows(out / "slopes.csv")) == len(plot_results.SLOPE_METRICS) * len(HEAP_CLASSES)
