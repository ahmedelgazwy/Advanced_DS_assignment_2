"""End-to-end smoke tests of ``demo.py``.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import sys

import pytest

import demo
from mergeable_heaps import HEAP_CLASSES


def run_demo(monkeypatch, *args: str) -> None:
    """Run ``demo.main()`` with the given command-line arguments."""
    monkeypatch.setattr(sys, "argv", ["demo.py", *args])
    demo.main()


def test_demo_runs_end_to_end(monkeypatch, capsys, tmp_path):
    run_demo(monkeypatch, "--n", "2000", "--graph-v", "300", "--data", str(tmp_path / "missing.gz"))
    output = capsys.readouterr().out
    assert "generated in memory" in output
    for section in ("1. Operations", "Paying for laziness", "2. Shapes", "3. Dijkstra"):
        assert section in output
    assert "from all three heaps: identical" in output
    assert all(output.count(heap_class.name) >= 4 for heap_class in HEAP_CLASSES)


def test_demo_shapes_show_thirteen_keys_per_heap(capsys):
    demo.show_shapes()
    output = capsys.readouterr().out
    assert "Fibonacci heap: 14 tree(s) after the inserts, 3 after extract-min" in output
    assert "Binomial heap: 3 tree(s) after the inserts, 3 after extract-min" in output
    assert "Pairing heap: 1 tree(s) after the inserts, 1 after extract-min" in output


@pytest.mark.parametrize("args", [["--n", "10"], ["--graph-v", "1"]])
def test_demo_rejects_tiny_sizes(monkeypatch, args):
    with pytest.raises(SystemExit):
        run_demo(monkeypatch, *args)
