# Mergeable Heaps — COSC 520 Assignment 2

Which priority queue should Dijkstra's algorithm use? This project implements
three mergeable heaps from scratch in Python, analyses their time and space
complexity, and benchmarks them on 10 million keys and a 10-million-edge graph.

| Structure | Module | Key idea |
|---|---|---|
| Binomial heap | `mergeable_heaps/binomial_heap.py` | forest of binomial trees, one per 1-bit of n |
| Fibonacci heap | `mergeable_heaps/fibonacci_heap.py` | lazy root list, marks and cascading cuts; O(1) amortized decrease-key |
| Pairing heap | `mergeable_heaps/pairing_heap.py` | one multiway tree, two-pass pairing on extract-min |

All three are implemented by hand. No library priority queues are used: no
`heapq`, no `queue.PriorityQueue`, no third-party heap packages.

> **Status:** work in progress. Sections below are filled in phase by phase
> (see [docs/review_log.md](docs/review_log.md)).

## Repository layout (planned)

```
.
├── mergeable_heaps/        # the library: base interface, three heaps, Dijkstra, dataset, timing helpers
├── scripts/                # CLIs: generate_dataset, run_benchmarks, plot_results
├── tests/                  # pytest suite
├── demo.py                 # live comparison of the three heaps
├── data/                   # dataset folder (large files downloaded or generated)
├── results/                # benchmark CSVs, figures and summary tables
├── report/                 # ACM-format LaTeX report
├── docs/
│   ├── theory.md           # complexity analysis, formulas, references
│   └── review_log.md       # phase-by-phase review decisions and edits by the author
├── requirements.txt        # pinned dependencies
└── pyproject.toml          # pytest configuration
```

## Setup

Requires **Python 3.10 or newer**. Run the commands from the repository root.

**Windows (PowerShell)**
```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1      # cmd.exe: .venv\Scripts\activate.bat
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No installation of the package itself is needed; every command is run from
the repository root.

## AI usage and authorship

This project was developed with the help of an AI coding assistant (Claude
Code, Anthropic), as allowed by the course policy with disclosure. The
assistant wrote the code, tests and documentation drafts. The author chose the
data structures, directed the work phase by phase, and reviewed and tested
each phase before it was committed; review decisions and the author's own
edits are recorded in [docs/review_log.md](docs/review_log.md). Every
AI-assisted file carries the line `AI-assisted: generated with Claude Code;
reviewed and tested by Ahmed Elgazwy` in its header, and code written by the
author is marked `Author: Ahmed Elgazwy`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` (PowerShell) | Run `Set-ExecutionPolicy -Scope Process Bypass` first, or skip activation and call `.venv\Scripts\python.exe` directly, e.g. `.venv\Scripts\python.exe -m pytest`. |
| `venv` fails with an `ensurepip ... returned non-zero exit status 1` error on Windows | The folder path is probably too long: Windows limits paths to 260 characters and pip's files are deeply nested. Clone into a short folder such as `C:\src`, or [enable long paths](https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation). |
| `ModuleNotFoundError: No module named 'mergeable_heaps'` | Run every command from the repository root, and run scripts as modules: `python -m scripts.run_benchmarks`, not `python scripts/run_benchmarks.py`. |
| `python` starts the wrong version | Check with `python --version`; Python 3.10 or newer is required. |
