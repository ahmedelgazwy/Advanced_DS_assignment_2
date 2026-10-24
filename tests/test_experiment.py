"""Tests for the shared measurement helpers.

AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
"""

from __future__ import annotations

import gc

from mergeable_heaps import HEAP_CLASSES, MergeableHeap
from mergeable_heaps.experiment import format_table, gc_paused, heap_factories, measure


def test_heap_factories_list_every_heap_class_by_name():
    factories = heap_factories()
    assert list(factories.values()) == list(HEAP_CLASSES)
    assert list(factories) == [cls.name for cls in HEAP_CLASSES]
    assert all(issubclass(cls, MergeableHeap) for cls in factories.values())


def test_gc_paused_restores_an_enabled_collector():
    assert gc.isenabled()
    with gc_paused():
        assert not gc.isenabled()
    assert gc.isenabled()


def test_gc_paused_leaves_a_disabled_collector_disabled():
    gc.disable()
    try:
        with gc_paused():
            assert not gc.isenabled()
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_measure_returns_the_result_and_a_duration():
    result, seconds = measure(lambda: sum(range(1_000)))
    assert result == 499_500
    assert seconds >= 0.0
    assert gc.isenabled()


def test_format_table_aligns_columns():
    table = format_table(["Name", "Value"], [["a", 1], ["long name", 22]])
    lines = table.splitlines()
    assert lines[0] == "Name       Value"
    assert lines[1] == "---------  -----"
    assert lines[3] == "long name  22"
    assert all(line == line.rstrip() for line in lines)
