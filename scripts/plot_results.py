"""Plot the benchmark results and summarize them for the report.

Reads ``operations.csv``, ``key_order.csv``, ``dijkstra.csv`` and
``environment.json`` written by ``scripts.run_benchmarks`` and writes, into the
same results folder:

* ``figures/*.pdf`` (for LaTeX) and ``figures/*.png`` (for the README);
* ``slopes.csv`` - growth rates fitted on log-log axes, and the a + b log2 n fit
  of extract-min;
* ``summary.md`` - tables of the key numbers.

Usage (from the repository root)::

    python -m scripts.plot_results                          # results/
    python -m scripts.plot_results --results results/quick  # the --quick run

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

# Categorical slots 1-3 of the validated reference palette (all pairs pass the
# CVD and normal-vision checks); each heap also has its own marker shape, so
# identity never depends on color alone.
STYLES = {
    "Binomial heap": ("#2a78d6", "o"),
    "Fibonacci heap": ("#eb6834", "s"),
    "Pairing heap": ("#1baf7a", "^"),
}
INK, INK_SECONDARY, INK_MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FIT_MIN_N = 10_000  # below this, fixed per-call overhead hides the growth rate
LABEL_MIN_GAP_PX = 11  # end labels closer than this would collide: rely on the legend

plt.rcParams.update({
    "figure.figsize": (6.4, 3.6),
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})


# --- Reading and fitting -----------------------------------------------------------------


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of rows.

    Args:
        path: CSV file with a header line.

    Returns:
        One dictionary per row (values are strings).
    """
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def series(rows: list[dict[str, str]], name: str, x_field: str, y_field: str,
           **filters: str) -> tuple[np.ndarray, np.ndarray]:
    """One heap's ``(x, y)`` points from a results table, sorted by x.

    Args:
        rows: Rows of a results CSV.
        name: Heap name (the ``structure`` column).
        x_field: Column for x.
        y_field: Column for y; empty cells are skipped.
        **filters: Extra ``column=value`` conditions.

    Returns:
        Tuple of numpy arrays ``(x, y)``.
    """
    points = sorted(
        (float(row[x_field]), float(row[y_field]))
        for row in rows
        if row["structure"] == name and row[y_field] != ""
        and all(row[column] == value for column, value in filters.items())
    )
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def loglog_slope(x: np.ndarray, y: np.ndarray, min_x: float = FIT_MIN_N) -> float:
    """Least-squares slope of log(y) against log(x) for x >= ``min_x``.

    A slope of 1 means linear growth, 0 means constant.

    Args:
        x: Sizes.
        y: Measurements (positive).
        min_x: Smallest size used in the fit.

    Returns:
        The fitted exponent, or NaN with fewer than two usable points.
    """
    keep = (x >= min_x) & (y > 0)
    if keep.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log10(x[keep]), np.log10(y[keep]), 1)[0])


def log2_fit(n: np.ndarray, y: np.ndarray, min_n: float = FIT_MIN_N) -> tuple[float, float, float]:
    """Fit ``y = a + b log2 n`` for n >= ``min_n`` (the O(log n) model).

    Args:
        n: Sizes.
        y: Measurements.
        min_n: Smallest size used in the fit.

    Returns:
        Tuple ``(a, b, r_squared)``; NaNs with fewer than three points.
    """
    keep = n >= min_n
    if keep.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    x, values = np.log2(n[keep]), y[keep]
    b, a = np.polyfit(x, values, 1)
    residual = values - (a + b * x)
    total = values - values.mean()
    r_squared = 1 - float(residual @ residual) / float(total @ total) if total.any() else 1.0
    return float(a), float(b), r_squared


# --- Drawing helpers -----------------------------------------------------------------------


def compact(value: float, _position: object = None) -> str:
    """Format an axis value compactly: 1K, 20K, 1M, 10M.

    Args:
        value: Tick value.
        _position: Unused (matplotlib formatter signature).

    Returns:
        Short label.
    """
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= limit:
            return f"{value / limit:g}{suffix}"
    return f"{value:g}"


def log_x_axis(ax: Axes, label: str) -> None:
    """Log-scale x axis with ticks at powers of ten, labelled 1K ... 10M.

    Args:
        ax: Axes to format.
        label: Axis label.

    Returns:
        None.
    """
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(compact))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(label)
    ax.grid(True, which="major")


def log_y_axis(ax: Axes, label: str, dense: bool = False) -> None:
    """Log-scale y axis with compact labels.

    Args:
        ax: Axes to format.
        label: Axis label.
        dense: Put ticks at 1, 2 and 5 times each power of ten, so that data
            spanning less than a decade still gets labelled ticks.

    Returns:
        None.
    """
    ax.set_yscale("log")
    if dense:
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(label)


def plot_heaps(ax: Axes, rows: list[dict[str, str]], x_field: str, y_field: str,
               label_ends: bool = True, **filters: str) -> None:
    """Draw one line per heap, then label line ends if they do not collide.

    Args:
        ax: Axes to draw on.
        rows: Rows of a results CSV.
        x_field: Column for x.
        y_field: Column for y.
        label_ends: Write each heap's name at its last point when there is room.
        **filters: Extra ``column=value`` conditions for :func:`series`.

    Returns:
        None.
    """
    ends = []
    for name, (color, marker) in STYLES.items():
        x, y = series(rows, name, x_field, y_field, **filters)
        if len(x) == 0:
            continue
        ax.plot(x, y, color=color, marker=marker, markeredgecolor="white", markeredgewidth=1,
                label=name, zorder=3)
        ends.append((name, x[-1], y[-1]))
    if label_ends and ends:
        add_end_labels(ax, ends)


def add_end_labels(ax: Axes, ends: list[tuple[str, float, float]]) -> None:
    """Write series names just right of their last points, unless any two would collide.

    Colliding labels are not nudged apart (that detaches them from their
    lines); the legend carries identity instead.

    Args:
        ax: Axes with the lines already drawn.
        ends: ``(name, x, y)`` of each line's last point.

    Returns:
        None.
    """
    ax.figure.canvas.draw()  # finalize limits so data -> pixel positions are exact
    pixel_y = sorted(ax.transData.transform((x, y))[1] for _, x, y in ends)
    if any(b - a < LABEL_MIN_GAP_PX for a, b in zip(pixel_y, pixel_y[1:])):
        return
    for name, x, y in ends:
        ax.annotate(name.replace(" heap", ""), (x, y), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_SECONDARY, annotation_clip=False)


def legend(ax: Axes) -> None:
    """Add the heap legend in one row below the axes, where it can never cover data.

    Args:
        ax: Axes whose lines are labelled.

    Returns:
        None.
    """
    ax.legend(handlelength=2.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Write a figure as PDF and PNG, then close it.

    Args:
        fig: Figure to save.
        out_dir: Folder for the files.
        name: File name without extension.

    Returns:
        None.
    """
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)


# --- Figures ---------------------------------------------------------------------------------


def plot_insert(ops: list[dict[str, str]], out: Path) -> None:
    """Insert cost per operation against n (H1: constant)."""
    fig, ax = plt.subplots()
    plot_heaps(ax, ops, "n", "insert_us")
    log_x_axis(ax, "heap size n")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("µs per insert")
    ax.set_title("Insert: constant time for all three heaps")
    legend(ax)
    save(fig, out, "insert_time")


def plot_extract(ops: list[dict[str, str]], out: Path) -> None:
    """Steady-state extract-min cost against n with the fitted a + b log2 n model (H3)."""
    fig, ax = plt.subplots()
    plot_heaps(ax, ops, "n", "extract_us", label_ends=False)
    for name, (color, _) in STYLES.items():
        n, y = series(ops, name, "n", "extract_us")
        a, b, _ = log2_fit(n, y)
        if not math.isnan(a):
            grid = np.logspace(math.log10(FIT_MIN_N), math.log10(n.max()), 50)
            ax.plot(grid, a + b * np.log2(grid), color=color, linewidth=1, linestyle=(0, (4, 3)),
                    zorder=2)
    log_x_axis(ax, "heap size n")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("µs per extract-min")
    ax.set_title("Extract-min (steady state) with fitted a + b·log₂ n")
    legend(ax)
    save(fig, out, "extract_min_time")


def plot_first_extract(ops: list[dict[str, str]], order: list[dict[str, str]], out: Path) -> None:
    """First extract-min after n inserts, for random, ascending and descending keys (H2)."""
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.8), sharey=True)
    panels = [("random keys", ops, {}), ("ascending keys", order, {"order": "ascending"}),
              ("descending keys", order, {"order": "descending"})]
    for ax, (title, rows, filters) in zip(axes, panels):
        plot_heaps(ax, rows, "n", "first_extract_ms", label_ends=False, **filters)
        n = np.array(sorted({float(r["n"]) for r in rows}))
        if len(n):
            ref = n / n[0] * 0.01  # a slope-1 guide line, scaled to sit low
            ax.plot(n, ref, color=INK_MUTED, linewidth=1, linestyle=(0, (1, 2)), zorder=1)
        log_x_axis(ax, "n")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.set_title(title, fontweight="normal", fontsize=9)
    axes[0].set_ylabel("ms for the first extract-min")
    axes[0].annotate("slope 1 (Θ(n))", (0.97, 0.05), xycoords="axes fraction", ha="right",
                     fontsize=7.5, color=INK_MUTED)
    fig.suptitle("First extract-min after n inserts: where lazy heaps pay", fontweight="bold",
                 fontsize=10, y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17))
    save(fig, out, "first_extract_min")


def plot_decrease(ops: list[dict[str, str]], out: Path) -> None:
    """Decrease-key cost and the extract-min that pays for it, against n (H4)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    plot_heaps(left, ops, "n", "decrease_us", label_ends=False)
    log_x_axis(left, "heap size n")
    left.set_ylim(bottom=0)
    left.set_ylabel("µs per decrease-key")
    left.set_title("Decrease-key (Dijkstra-like)", fontweight="normal", fontsize=9)
    plot_heaps(right, ops, "n", "after_decrease_ms", label_ends=False)
    log_x_axis(right, "heap size n")
    log_y_axis(right, "ms for the next extract-min")
    right.set_title("The extract-min right after K decrease-keys", fontweight="normal", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    fig.suptitle("Cheap decrease-keys defer work to the next extract-min\n"
                 "(K = min(100,000, n/10) decrease-keys per measurement)", fontweight="bold",
                 fontsize=10, y=0.98)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    save(fig, out, "decrease_key_time")


def plot_merge(ops: list[dict[str, str]], out: Path) -> None:
    """Merge of two halves against n (H5)."""
    fig, ax = plt.subplots()
    plot_heaps(ax, ops, "n", "merge_us")
    log_x_axis(ax, "heap size n (two heaps of n/2 merged)")
    log_y_axis(ax, "µs for one merge", dense=True)
    ax.set_title("Merge: binomial O(log n); Fibonacci and pairing O(1) in theory")
    legend(ax)
    save(fig, out, "merge_time")


def plot_memory(ops: list[dict[str, str]], out: Path) -> None:
    """Bytes per item at the largest n, as horizontal bars (H6)."""
    largest = max(int(r["n"]) for r in ops)
    values = []
    for name in STYLES:
        _, y = series(ops, name, "n", "bytes_per_item", n=str(largest))
        values.append((name, float(y[-1]) if len(y) else 0.0))
    fig, ax = plt.subplots(figsize=(6.4, 1.9))
    positions = range(len(values))
    ax.barh(list(positions), [v for _, v in values], height=0.5,
            color=[STYLES[name][0] for name, _ in values])
    for position, (_, value) in zip(positions, values):
        ax.annotate(f"{value:.0f} B", (value, position), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.set_yticks(list(positions), [name for name, _ in values])
    ax.invert_yaxis()
    ax.grid(False, axis="y")
    ax.set_xlabel(f"bytes per item at n = {compact(largest)} (node, key and, for binomial, item)")
    ax.set_title("Memory per item")
    save(fig, out, "memory")


def plot_dijkstra(dij: list[dict[str, str]], out: Path) -> None:
    """Dijkstra running time against V, and decrease-keys per vertex (H7)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    plot_heaps(left, dij, "vertices", "seconds", label_ends=False)
    log_x_axis(left, "vertices V (E = 10 V)")
    log_y_axis(left, "seconds", dense=True)
    left.set_title("Running time", fontweight="normal", fontsize=9)

    vertices = sorted({float(r["vertices"]) for r in dij})
    per_vertex = [np.mean([float(r["decrease_keys"]) / float(r["vertices"]) for r in dij
                           if float(r["vertices"]) == v]) for v in vertices]
    edges_per_vertex = float(dij[0]["edges"]) / float(dij[0]["vertices"]) if dij else 10.0
    right.plot(vertices, per_vertex, color=INK_SECONDARY, marker="o", markeredgecolor="white",
               markeredgewidth=1, zorder=3)
    right.axhline(edges_per_vertex, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    right.annotate("upper bound: one per edge (E/V)", (vertices[0] if vertices else 1, edges_per_vertex),
                   xytext=(0, -11), textcoords="offset points", fontsize=7.5, color=INK_MUTED)
    log_x_axis(right, "vertices V (E = 10 V)")
    right.set_ylim(0, edges_per_vertex * 1.15)
    right.set_ylabel("decrease-keys per vertex")
    right.set_title("Decrease-keys actually performed (mean of heaps)", fontweight="normal",
                    fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.suptitle("Dijkstra: few decrease-keys, so extract-min decides", fontweight="bold",
                 fontsize=10, y=0.98)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    save(fig, out, "dijkstra_time")


def plot_warmup(ops: list[dict[str, str]], out: Path) -> None:
    """Per-operation cost of the warm-up batch relative to steady state."""
    fig, ax = plt.subplots()
    for name, (color, marker) in STYLES.items():
        n, warm = series(ops, name, "n", "warmup_extract_us")
        _, steady = series(ops, name, "n", "extract_us")
        if len(n):
            ax.plot(n, warm / steady, color=color, marker=marker, markeredgecolor="white",
                    markeredgewidth=1, label=name, zorder=3)
    ax.axhline(1, color=AXIS, linewidth=1)
    log_x_axis(ax, "heap size n")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("warm-up ÷ steady-state cost")
    ax.set_title("The K extract-mins after the first one: still paying for laziness")
    legend(ax)
    save(fig, out, "extract_warmup")


# --- Tables ------------------------------------------------------------------------------------


SLOPE_METRICS = [
    ("operations", "insert_us"), ("operations", "merge_us"), ("operations", "first_extract_ms"),
    ("operations", "extract_us"), ("operations", "decrease_us"),
    ("operations", "after_decrease_ms"), ("operations", "decrease_to_min_us"),
    ("key_order:ascending", "first_extract_ms"), ("key_order:descending", "first_extract_ms"),
    ("dijkstra", "seconds"),
]


def slope_rows(ops, order, dij) -> list[dict[str, object]]:
    """Fitted growth rates for every heap and metric.

    Args:
        ops: ``operations.csv`` rows.
        order: ``key_order.csv`` rows.
        dij: ``dijkstra.csv`` rows.

    Returns:
        ``slopes.csv`` rows: log-log slope, plus a, b and R^2 of the
        a + b log2 n fit for extract-min.
    """
    tables = {"operations": (ops, "n", {}), "dijkstra": (dij, "vertices", {})}
    for direction in ("ascending", "descending"):
        tables[f"key_order:{direction}"] = (order, "n", {"order": direction})
    rows = []
    for table, metric in SLOPE_METRICS:
        data, x_field, filters = tables[table]
        for name in STYLES:
            x, y = series(data, name, x_field, metric, **filters)
            row = {"structure": name, "experiment": table, "metric": metric,
                   "loglog_slope": round(loglog_slope(x, y), 3), "log2_a": "", "log2_b": "",
                   "log2_r2": ""}
            if metric == "extract_us" and table == "operations":
                a, b, r2 = log2_fit(x, y)
                row.update(log2_a=round(a, 3), log2_b=round(b, 3), log2_r2=round(r2, 3))
            rows.append(row)
    return rows


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a Markdown table.

    Args:
        headers: Column titles.
        rows: Cells (converted with ``str``).

    Returns:
        Markdown text.
    """
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def summary_text(env, ops, order, dij, slopes) -> str:
    """Markdown summary: machine, the largest-size tables and the fitted slopes.

    Args:
        env: Parsed ``environment.json``.
        ops: ``operations.csv`` rows.
        order: ``key_order.csv`` rows.
        dij: ``dijkstra.csv`` rows.
        slopes: Output of :func:`slope_rows`.

    Returns:
        Markdown text.
    """
    ram = f"{env['ram_bytes'] / 2**30:.0f} GiB RAM" if env.get("ram_bytes") else "unknown RAM"
    parts = [f"Machine: {env['cpu']}, {ram}, {env['os']}, Python {env['python']}"]
    largest = max(int(r["n"]) for r in ops)
    at_largest = {r["structure"]: r for r in ops if int(r["n"]) == largest}
    number: Callable[[str, str], str] = lambda name, field: f"{float(at_largest[name][field]):,.2f}"
    parts.append(f"## Operations at n = {largest:,}\n")
    parts.append(markdown_table(
        ["Heap", "Insert (µs)", "Merge (µs)", "1st extract-min (ms)", "Warm-up extract (µs)",
         "Extract-min (µs)", "Find-min (µs)", "Decrease-key (µs)", "Next extract (ms)",
         "Decrease to min (µs)", "Next extract (ms)", "Bytes/item"],
        [[name] + [number(name, f) for f in (
            "insert_us", "merge_us", "first_extract_ms", "warmup_extract_us", "extract_us",
            "find_min_us", "decrease_us", "after_decrease_ms", "decrease_to_min_us",
            "after_decrease_to_min_ms")] + [f"{float(at_largest[name]['bytes_per_item']):.0f}"]
         for name in STYLES if name in at_largest],
    ))
    if order:
        top = max(int(r["n"]) for r in order)
        parts.append(f"\n## First extract-min after n = {top:,} sorted inserts (ms)\n")
        cell = {(r["structure"], r["order"]): r for r in order if int(r["n"]) == top}
        parts.append(markdown_table(
            ["Heap", "Ascending", "Trees before", "Descending", "Trees before"],
            [[name, f"{float(cell[name, 'ascending']['first_extract_ms']):,.2f}",
              f"{int(cell[name, 'ascending']['trees_before']):,}",
              f"{float(cell[name, 'descending']['first_extract_ms']):,.2f}",
              f"{int(cell[name, 'descending']['trees_before']):,}"]
             for name in STYLES if (name, "ascending") in cell],
        ))
    if dij:
        top = max(int(r["vertices"]) for r in dij)
        rows = [r for r in dij if int(r["vertices"]) == top]
        parts.append(f"\n## Dijkstra at V = {top:,}, E = {int(rows[0]['edges']):,}\n")
        parts.append(markdown_table(
            ["Heap", "Seconds", "Inserts", "Extract-mins", "Decrease-keys", "Reachable",
             "Distance sum"],
            [[r["structure"], f"{float(r['seconds']):.2f}", f"{int(r['inserts']):,}",
              f"{int(r['extract_mins']):,}", f"{int(r['decrease_keys']):,}",
              f"{int(r['reachable']):,}", f"{int(r['distance_sum']):,}"] for r in rows],
        ))
    parts.append(f"\n## Fitted growth (log-log slope for n >= {FIT_MIN_N:,}; 0 = constant, 1 = linear)\n")
    parts.append(markdown_table(
        ["Experiment", "Metric", *STYLES],
        [[table, metric, *(next(s["loglog_slope"] for s in slopes if s["structure"] == name
                                and s["experiment"] == table and s["metric"] == metric)
                           for name in STYLES)]
         for table, metric in SLOPE_METRICS],
    ))
    fits = [s for s in slopes if s["log2_a"] != ""]
    parts.append("\n## Extract-min fitted as a + b·log2 n (µs)\n")
    parts.append(markdown_table(["Heap", "a", "b", "R²"],
                                [[s["structure"], s["log2_a"], s["log2_b"], s["log2_r2"]] for s in fits]))
    return "\n".join(parts) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options.

    Args:
        argv: Arguments to parse (default: ``sys.argv[1:]``).

    Returns:
        Namespace with ``results``.
    """
    parser = argparse.ArgumentParser(description="Plot and summarize the benchmark results.")
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="folder written by scripts.run_benchmarks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Draw every figure and write ``slopes.csv`` and ``summary.md``.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        None.
    """
    args = parse_args(argv)
    folder = args.results
    ops = read_csv(folder / "operations.csv")
    order = read_csv(folder / "key_order.csv")
    dij = read_csv(folder / "dijkstra.csv")
    env = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
    figures = folder / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plot_insert(ops, figures)
    plot_extract(ops, figures)
    plot_first_extract(ops, order, figures)
    plot_decrease(ops, figures)
    plot_merge(ops, figures)
    plot_memory(ops, figures)
    plot_warmup(ops, figures)
    if dij:
        plot_dijkstra(dij, figures)

    slopes = slope_rows(ops, order, dij)
    with open(folder / "slopes.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slopes[0]))
        writer.writeheader()
        writer.writerows(slopes)
    (folder / "summary.md").write_text(summary_text(env, ops, order, dij, slopes), encoding="utf-8")
    print(f"Figures in {figures}/, tables in {folder / 'summary.md'} and {folder / 'slopes.csv'}")


if __name__ == "__main__":
    main()
