"""Generate the synthetic datasets used by the benchmarks.

Writes three files to the output directory:

* ``keys_<n>.txt.gz``         - ``n`` random integer keys, one per line;
* ``graph_<V>_<E>.txt.gz``    - a random weighted digraph, one edge per line;
* ``manifest.json``           - counts, seeds, formats and SHA-256 checksums.

Usage (from the repository root)::

    python -m scripts.generate_dataset                              # 10M keys, 1M vertices / 10M edges
    python -m scripts.generate_dataset --keys 100000 --vertices 10000 --edges 100000

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mergeable_heaps.dataset import (
    GRAPH_SEED,
    KEY_BITS,
    KEY_SEED,
    MAX_WEIGHT,
    SOURCE,
    generate_graph,
    generate_keys,
    save_graph,
    save_keys,
    sha256_of,
)

DEFAULT_KEYS = 10_000_000
DEFAULT_VERTICES = 1_000_000
DEFAULT_EDGES = 10_000_000


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        Namespace with ``keys``, ``vertices``, ``edges``, ``key_seed``,
        ``graph_seed`` and ``out_dir``.
    """
    parser = argparse.ArgumentParser(description="Generate the key and graph datasets.")
    parser.add_argument("--keys", type=int, default=DEFAULT_KEYS, help="number of keys")
    parser.add_argument("--vertices", type=int, default=DEFAULT_VERTICES, help="graph vertices")
    parser.add_argument("--edges", type=int, default=DEFAULT_EDGES, help="graph edges")
    parser.add_argument("--key-seed", type=int, default=KEY_SEED, help="seed for the keys")
    parser.add_argument("--graph-seed", type=int, default=GRAPH_SEED, help="seed for the graph")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="output directory")
    return parser.parse_args()


def main() -> None:
    """Generate both dataset files and the manifest.

    Returns:
        None.
    """
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    keys_path = args.out_dir / f"keys_{args.keys}.txt.gz"
    graph_path = args.out_dir / f"graph_{args.vertices}_{args.edges}.txt.gz"
    start = time.perf_counter()

    print(f"Writing {args.keys:,} keys to {keys_path} ...", flush=True)
    save_keys(keys_path, generate_keys(args.keys, args.key_seed))
    print(f"Writing a graph with {args.vertices:,} vertices and {args.edges:,} edges "
          f"to {graph_path} ...", flush=True)
    save_graph(graph_path, generate_graph(args.vertices, args.edges, args.graph_seed))

    paths = (keys_path, graph_path)
    manifest = {
        "description": "Synthetic datasets for the COSC 520 mergeable-heap benchmark",
        "keys": {
            "file": keys_path.name,
            "count": args.keys,
            "seed": args.key_seed,
            "format": "gzip-compressed text, one integer per line",
            "distribution": f"uniform integers in [0, 2^{KEY_BITS})",
        },
        "graph": {
            "file": graph_path.name,
            "vertices": args.vertices,
            "edges": args.edges,
            "seed": args.graph_seed,
            "source": SOURCE,
            "format": "gzip-compressed text: header '# vertices=V edges=E', "
                      "then one 'source target weight' line per edge, grouped by source",
            "construction": f"random Hamiltonian path from vertex {SOURCE} (V - 1 edges) "
                            "plus uniformly random edges between distinct vertices",
            "weights": f"uniform integers in [1, {MAX_WEIGHT}]",
        },
        "sha256": {path.name: sha256_of(path) for path in paths},
        "bytes": {path.name: path.stat().st_size for path in paths},
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done in {time.perf_counter() - start:.1f}s; manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
