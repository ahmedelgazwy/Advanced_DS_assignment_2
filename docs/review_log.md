# Review log

Record of the author's decisions, reviews and edits, phase by phase. This file
documents which parts of the work are the author's (see README, "AI usage and
authorship").

| Phase | Author's decisions and review | Author's edits | Tests run by the author |
|---|---|---|---|
| Plan | Chose the three structures (binomial, Fibonacci, pairing heaps) and Dijkstra as the application; chose the authorship split; approved the phase plan. | — | — |
| 0. Repository setup | Checked the pinned requirements and the plan; approved. | — | — |
| 1. Theory notes | Reviewed `docs/theory.md` line by line; approved. | — | — |
| 2. Interface and contract tests | Reviewed the interface, the reference heap and the contract tests; approved, including the stale-handle `ValueError` rule. | — | — |
| 3. Binomial heap | Reviewed the binomial heap, its tests, the bug-injection results and the 1M-item timings; approved. | — | — |
| 4. Fibonacci heap | Reviewed the Fibonacci heap (including the exact integer degree bound) and its tests; approved. | — | Ran the Phase 4 tests. |
| 5. Pairing heap | Reviewed the pairing heap and its tests, and the two corrected test expectations; approved. Agreed that Phase 9 must use fixed key orders for the first extract-min and Dijkstra-like decrease-keys (new key between the current minimum and the old key). | — | Ran the Phase 5 tests. |
| 6. Datasets | Reviewed the key and graph datasets, the file formats and the checksum verification; approved. Created the GitHub repository and published the dataset as release `dataset-v1`. | — | — |
| 7. Dijkstra | Reviewed Dijkstra, its tests and the full-graph results (Fibonacci slowest, about 1M decrease-keys for 10M edges); approved. | — | Ran the Phase 7 tests. |
| 8. Demo | Reviewed the demo, the traversal refactor of the three heaps (`_roots`/`_children`, `tree_count`, `format_trees`) and the two measurement fixes (pairing warm-up timed separately; cold-cache merge explained); approved. | — | Ran the demo and the Phase 8 tests. |
| 9. Benchmarks and figures | Reviewed the benchmark design, the full-run results against hypotheses H1–H7 and the figures; approved. | — | — |
| 10. Report | Wrote the motivation for Section 2 ("Why These Structures") in his own words and accepted light edits (naming Dijkstra's algorithm and the Fibonacci heap, one closing sentence on the result, typo fixes). Reviewed the report and removed a sentence of the GenAI section as redundant with the roles listed above it. | Section 2; GenAI section edit | Compiled the report on Overleaf (10 pages). |
