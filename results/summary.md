Machine: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz, 16 GiB RAM, Windows-10-10.0.26200-SP0, Python 3.10.2
## Operations at n = 10,000,000

| Heap | Insert (µs) | Merge (µs) | 1st extract-min (ms) | Warm-up extract (µs) | Extract-min (µs) | Find-min (µs) | Decrease-key (µs) | Next extract (ms) | Decrease to min (µs) | Next extract (ms) | Bytes/item |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Binomial heap | 1.48 | 68.30 | 0.04 | 32.11 | 33.72 | 0.17 | 2.05 | 0.05 | 9.74 | 0.06 | 158 |
| Fibonacci heap | 0.79 | 22.70 | 8,142.28 | 41.78 | 42.54 | 0.11 | 1.74 | 63.09 | 2.37 | 138.69 | 126 |
| Pairing heap | 0.82 | 9.20 | 1,648.11 | 87.82 | 17.50 | 0.10 | 1.75 | 86.45 | 1.90 | 0.01 | 102 |

## First extract-min after n = 10,000,000 sorted inserts (ms)

| Heap | Ascending | Trees before | Descending | Trees before |
|---|---|---|---|---|
| Binomial heap | 0.06 | 8 | 0.02 | 8 |
| Fibonacci heap | 8,142.71 | 10,000,000 | 7,745.44 | 10,000,000 |
| Pairing heap | 4,026.72 | 1 | 0.01 | 1 |

## Dijkstra at V = 1,000,000, E = 10,000,000

| Heap | Seconds | Inserts | Extract-mins | Decrease-keys | Reachable | Distance sum |
|---|---|---|---|---|---|---|
| Binomial heap | 29.34 | 1,000,000 | 1,000,000 | 993,926 | 1,000,000 | 1,490,616,444 |
| Fibonacci heap | 37.53 | 1,000,000 | 1,000,000 | 993,962 | 1,000,000 | 1,490,616,444 |
| Pairing heap | 26.78 | 1,000,000 | 1,000,000 | 993,971 | 1,000,000 | 1,490,616,444 |

## Fitted growth (log-log slope for n >= 10,000; 0 = constant, 1 = linear)

| Experiment | Metric | Binomial heap | Fibonacci heap | Pairing heap |
|---|---|---|---|---|
| operations | insert_us | 0.033 | 0.063 | 0.058 |
| operations | merge_us | 0.237 | 0.201 | 0.128 |
| operations | first_extract_ms | 0.187 | 1.056 | 1.075 |
| operations | extract_us | 0.165 | 0.148 | 0.165 |
| operations | decrease_us | 0.112 | 0.129 | 0.102 |
| operations | after_decrease_ms | 0.162 | 0.854 | 0.862 |
| operations | decrease_to_min_us | 0.207 | 0.099 | 0.111 |
| key_order:ascending | first_extract_ms | 0.18 | 1.028 | 1.025 |
| key_order:descending | first_extract_ms | 0.056 | 1.05 | 0.067 |
| dijkstra | seconds | 1.148 | 1.112 | 1.125 |

## Extract-min fitted as a + b·log2 n (µs)

| Heap | a | b | R² |
|---|---|---|---|
| Binomial heap | -21.386 | 2.227 | 0.899 |
| Fibonacci heap | -23.655 | 2.77 | 0.878 |
| Pairing heap | -12.08 | 1.264 | 0.895 |
