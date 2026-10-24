# Dataset

Synthetic data used by the benchmarks: 10 million heap keys and a weighted
graph with 10 million edges for Dijkstra's algorithm.

**Download:** https://github.com/ahmedelgazwy/Advanced_DS_assignment_2/releases/tag/dataset-v1

| File | Contents | Size |
|---|---|---|
| `keys_10000000.txt.gz` | 10,000,000 uniform random integers in [0, 2³¹), one per line | 48 MB |
| `graph_1000000_10000000.txt.gz` | directed graph, 1,000,000 vertices and 10,000,000 weighted edges | 66 MB |
| `manifest.json` | counts, seeds, formats, SHA-256 checksums and sizes of both files | 1 KB |

The large files are not committed to git. Put them in this folder, or
regenerate them (about 50 seconds):

```bash
python -m scripts.generate_dataset
```

Generation is deterministic: the same seeds (and the numpy version pinned in
`requirements.txt`) produce byte-identical files, so the SHA-256 checksums in
`manifest.json` verify a download or a regeneration. No benchmark or test needs
the files: missing data is generated in memory from the same seeds.

## Keys

```
1373158606
239081663
53710184
```

- **Uniform** in [0, 2³¹), drawn one after another from Python's seeded Mersenne Twister (seed 42).
- **Nested.** The first *n* lines are exactly the dataset of size *n*, so one file serves every heap size.
- **Duplicates are possible**, as in real priority queues: 9,977,007 of the 10,000,000 keys are distinct.

## Graph

```
# vertices=1000000 edges=10000000
0 668785 311
0 66342 685
...
```

- The header gives V and E. Each following line is `source target weight`, grouped by source.
- **Every vertex is reachable from vertex 0** (the Dijkstra source). The first V − 1 edges form a random Hamiltonian path that starts at vertex 0.
- The other E − V + 1 edges join uniformly random pairs of distinct vertices: parallel edges are possible, self-loops are not.
- Weights are uniform integers in [1, 1000] (seed 7, numpy PCG64).
- Average out-degree is 10. This is the sparse regime (E = O(V)) in which Fibonacci heaps' O(E + V log V) Dijkstra bound should matter.

The benchmarks also use smaller graphs with the same construction (V = 10⁴ … 10⁶, E = 10V). They are generated in memory from the same seed.

Code: [`mergeable_heaps/dataset.py`](../mergeable_heaps/dataset.py).
