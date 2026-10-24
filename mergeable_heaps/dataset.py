"""Synthetic datasets: random integer keys and random weighted graphs.

Keys. ``generate_keys(n)`` draws n uniform integers in [0, 2^31) from a
seeded Mersenne Twister, one after another, so the first m keys of any larger
dataset are exactly ``generate_keys(m)``. One file of 10 million keys therefore
serves every heap size. Keys may repeat (about 23,000 duplicates at 10 million),
which heaps must handle anyway.

Graphs. ``generate_graph(V, E)`` builds a directed graph whose first V - 1
edges form a random Hamiltonian path starting at vertex ``SOURCE`` = 0, so
every vertex is reachable from the source. The other E - V + 1 edges join
uniformly random pairs of distinct vertices. Weights are uniform integers in
[1, 1000]. The graph is stored in compressed sparse row (CSR) form in three
``array.array`` objects (about 8 bytes per edge), because Python lists of
tuples would need about 1 GB at E = 10 million. numpy is used only to draw the
random numbers and sort the edges by source, never inside a heap.

Files. Text, gzip-compressed, written with ``mtime=0`` and no file name in the
gzip header, so regenerating a file gives byte-identical output (the same
SHA-256 as the published dataset; numpy is pinned in requirements.txt because
its random streams may change between versions).

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import gzip
import hashlib
import io
import random
from array import array
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import IO, Iterable, Iterator, Sequence

import numpy as np

KEY_BITS = 31  # keys are uniform in [0, 2**31)
KEY_SEED = 42
GRAPH_SEED = 7
MAX_WEIGHT = 1_000
SOURCE = 0  # Dijkstra source; the Hamiltonian path starts here
_INT = "i"  # array typecode for 32-bit signed integers (vertex ids, offsets, weights)


class Graph:
    """Directed graph with integer edge weights in compressed sparse row form.

    The out-edges of vertex ``u`` are ``targets[i]`` with weight ``weights[i]``
    for ``i`` in ``range(offsets[u], offsets[u + 1])``.
    """

    __slots__ = ("offsets", "targets", "weights")

    def __init__(self, offsets: array, targets: array, weights: array) -> None:
        """Wrap CSR arrays (use :meth:`from_edges` to build them from an edge list).

        Args:
            offsets: ``V + 1`` non-decreasing positions, from 0 to ``E``.
            targets: ``E`` target vertices.
            weights: ``E`` edge weights.

        Raises:
            ValueError: If the array lengths are inconsistent.
        """
        if len(offsets) < 1 or offsets[0] != 0 or offsets[-1] != len(targets):
            raise ValueError("offsets must start at 0 and end at the number of edges")
        if len(targets) != len(weights):
            raise ValueError("targets and weights must have the same length")
        self.offsets = offsets
        self.targets = targets
        self.weights = weights

    @classmethod
    def from_edges(
        cls,
        num_vertices: int,
        sources: Sequence[int],
        targets: Sequence[int],
        weights: Sequence[int],
    ) -> Graph:
        """Build a graph from parallel edge lists (stable: keeps each vertex's edge order).

        Args:
            num_vertices: Number of vertices V; ids must be in [0, V).
            sources: Source vertex of each edge.
            targets: Target vertex of each edge.
            weights: Non-negative integer weight of each edge.

        Returns:
            The graph in CSR form.

        Raises:
            ValueError: If the lists differ in length, or a vertex id or weight is invalid.
        """
        src = np.asarray(sources, dtype=np.int64)
        dst = np.asarray(targets, dtype=np.int64)
        wgt = np.asarray(weights, dtype=np.int64)
        if not len(src) == len(dst) == len(wgt):
            raise ValueError("sources, targets and weights must have the same length")
        if len(src) and (min(src.min(), dst.min()) < 0 or max(src.max(), dst.max()) >= num_vertices):
            raise ValueError(f"vertex ids must be in [0, {num_vertices})")
        if len(wgt) and wgt.min() < 0:
            raise ValueError("weights must be non-negative")
        order = np.argsort(src, kind="stable")
        offsets = np.zeros(num_vertices + 1, dtype=np.int64)
        np.cumsum(np.bincount(src, minlength=num_vertices), out=offsets[1:])
        return cls(_to_array(offsets), _to_array(dst[order]), _to_array(wgt[order]))

    @property
    def num_vertices(self) -> int:
        """Number of vertices V."""
        return len(self.offsets) - 1

    @property
    def num_edges(self) -> int:
        """Number of edges E."""
        return len(self.targets)

    def edges_from(self, vertex: int) -> Iterator[tuple[int, int]]:
        """Out-edges of one vertex.

        Args:
            vertex: Source vertex.

        Returns:
            Iterator of ``(target, weight)`` pairs.
        """
        for index in range(self.offsets[vertex], self.offsets[vertex + 1]):
            yield self.targets[index], self.weights[index]

    def memory_bytes(self) -> int:
        """Bytes used by the three CSR arrays.

        Returns:
            Total buffer size in bytes.
        """
        return sum(len(part) * part.itemsize for part in (self.offsets, self.targets, self.weights))


def _to_array(values: np.ndarray) -> array:
    """Copy a numpy integer array into a compact ``array.array`` of 32-bit ints.

    Args:
        values: Integers that fit in 32 bits.

    Returns:
        ``array.array`` with typecode ``"i"``.
    """
    result = array(_INT)
    result.frombytes(values.astype(np.int32).tobytes())
    return result


def generate_keys(count: int, seed: int = KEY_SEED) -> list[int]:
    """Uniform random keys in [0, 2^31); the first m keys do not depend on ``count``.

    Args:
        count: Number of keys.
        seed: RNG seed; the same seed always gives the same keys.

    Returns:
        List of ``count`` integers.

    Raises:
        ValueError: If ``count`` is negative.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    draw = random.Random(seed).getrandbits
    return [draw(KEY_BITS) for _ in range(count)]


def generate_graph(
    num_vertices: int, num_edges: int, seed: int = GRAPH_SEED, max_weight: int = MAX_WEIGHT
) -> Graph:
    """Random directed graph in which every vertex is reachable from ``SOURCE``.

    The first V - 1 edges form a random Hamiltonian path that starts at
    ``SOURCE``; the remaining edges join uniformly random pairs of distinct
    vertices (parallel edges are possible, self-loops are not).

    Args:
        num_vertices: Number of vertices V (at least 1).
        num_edges: Number of edges E (at least V - 1; 0 if V is 1).
        seed: RNG seed; the same arguments always give the same graph.
        max_weight: Weights are uniform integers in [1, max_weight].

    Returns:
        The graph in CSR form.

    Raises:
        ValueError: If V < 1, E < V - 1, or E > 0 with a single vertex.
    """
    if num_vertices < 1:
        raise ValueError("the graph needs at least one vertex")
    if num_edges < num_vertices - 1:
        raise ValueError("need at least V - 1 edges to make every vertex reachable")
    if num_vertices == 1 and num_edges > 0:
        raise ValueError("a single vertex cannot have edges without self-loops")
    rng = np.random.default_rng(seed)
    path = np.concatenate(([SOURCE], rng.permutation(np.arange(1, num_vertices))))
    extra = num_edges - (num_vertices - 1)
    extra_sources = rng.integers(0, num_vertices, size=extra)
    extra_targets = rng.integers(0, max(num_vertices - 1, 1), size=extra)
    extra_targets += extra_targets >= extra_sources  # skip the source itself: no self-loops
    sources = np.concatenate((path[:-1], extra_sources))
    targets = np.concatenate((path[1:], extra_targets))
    weights = rng.integers(1, max_weight + 1, size=num_edges)
    return Graph.from_edges(num_vertices, sources, targets, weights)


@contextmanager
def _open_text(path: Path, mode: str) -> Iterator[IO[str]]:
    """Open a UTF-8 text file, transparently gzip-compressed if it ends in .gz.

    Compressed files are written with ``mtime=0`` and no file name in the gzip
    header, so regenerating a dataset gives byte-identical output.

    Args:
        path: File path.
        mode: ``"r"`` to read or ``"w"`` to write.

    Returns:
        Context manager yielding a text handle with ``\\n`` line endings.
    """
    if path.suffix != ".gz":
        with open(path, mode, encoding="utf-8", newline="\n") as handle:
            yield handle
        return
    with open(path, mode + "b") as raw, gzip.GzipFile(
        filename="", mode=mode + "b", fileobj=raw, mtime=0, compresslevel=6
    ) as compressed, io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
        yield handle


def save_keys(path: str | Path, keys: Iterable[int]) -> int:
    """Write keys to a text file, one per line (gzip if the name ends in .gz).

    Args:
        path: Destination file.
        keys: Integer keys.

    Returns:
        Number of keys written.
    """
    count = 0
    with _open_text(Path(path), "w") as handle:
        for key in keys:
            handle.write(f"{key}\n")
            count += 1
    return count


def load_keys(path: str | Path, limit: int | None = None) -> list[int]:
    """Read keys written by :func:`save_keys`.

    Args:
        path: Source file (plain text or .gz).
        limit: Read only the first ``limit`` keys; None reads all.

    Returns:
        List of integers in file order.
    """
    with _open_text(Path(path), "r") as handle:
        return [int(line) for line in islice(handle, limit)]


def save_graph(path: str | Path, graph: Graph) -> None:
    """Write a graph as a header line and one ``source target weight`` line per edge.

    The header is ``# vertices=V edges=E``; edges are written in CSR order
    (grouped by source).

    Args:
        path: Destination file (gzip if the name ends in .gz).
        graph: Graph to write.

    Returns:
        None.
    """
    offsets, targets, weights = graph.offsets, graph.targets, graph.weights
    with _open_text(Path(path), "w") as handle:
        handle.write(f"# vertices={graph.num_vertices} edges={graph.num_edges}\n")
        for source in range(graph.num_vertices):
            lines = [
                f"{source} {targets[index]} {weights[index]}\n"
                for index in range(offsets[source], offsets[source + 1])
            ]
            handle.writelines(lines)


def load_graph(path: str | Path) -> Graph:
    """Read a graph written by :func:`save_graph`.

    Args:
        path: Source file (plain text or .gz).

    Returns:
        The graph in CSR form.

    Raises:
        ValueError: If the header is missing or the edge count does not match it.
    """
    with _open_text(Path(path), "r") as handle:
        header = handle.readline().split()
        if len(header) != 3 or header[0] != "#":
            raise ValueError(f"{path}: missing '# vertices=V edges=E' header")
        num_vertices = int(header[1].removeprefix("vertices="))
        num_edges = int(header[2].removeprefix("edges="))
        if num_edges:
            columns = np.loadtxt(handle, dtype=np.int64, ndmin=2)
        else:
            columns = np.zeros((0, 3), dtype=np.int64)
    if len(columns) != num_edges:
        raise ValueError(f"{path}: header says {num_edges} edges, found {len(columns)}")
    return Graph.from_edges(num_vertices, columns[:, 0], columns[:, 1], columns[:, 2])


def load_or_generate_keys(path: str | Path, count: int) -> tuple[list[int], bool]:
    """Read ``count`` keys from the dataset file, or generate the same keys if it is missing.

    Args:
        path: Key file (see data/README.md).
        count: Number of keys needed.

    Returns:
        Tuple ``(keys, read_from_file)``.
    """
    path = Path(path)
    if path.exists():
        keys = load_keys(path, limit=count)
        if len(keys) == count:
            return keys, True
    return generate_keys(count), False


def sha256_of(path: str | Path) -> str:
    """SHA-256 checksum of a file.

    Args:
        path: File to hash.

    Returns:
        Hex digest string.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
