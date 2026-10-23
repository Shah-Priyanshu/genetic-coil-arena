"""Test the arithmetic demo end to end."""

import pytest

from genetic_coil_arena.protocol import HAS_JSONSCHEMA, verify
from genetic_coil_arena.runtime import TraceLog
from genetic_coil_arena.runtime.demo import (
    DEMO_KEYS,
    build_centre,
    demo_specs,
    main,
    run_demo,
)


def test_run_demo(tmp_path):
    path = tmp_path / "demo.jsonl"
    summary = run_demo(path, verbose=False)
    assert summary["dot"].state == "completed" and summary["dot"].result == 32
    assert len(summary["dot"].subtasks) == 4  # three multiplications and one sum
    assert summary["add"].result == 42
    assert summary["tampered"].state == "failed"
    assert "payload digest mismatch" in summary["tampered"].error
    assert summary["signature_rejections"] == 1

    trace = TraceLog.load(path)
    assert len(trace) == summary["trace_records"]
    envelopes = trace.envelopes()
    assert envelopes and all(e.is_signed for e in envelopes)
    honest = [e for e in envelopes if verify(e, DEMO_KEYS)]
    assert len(honest) == len(envelopes) - 1
    dot_triple = next(
        t for t in trace.triples() if t.spec and t.spec.spec == "arith.dot"
    )
    assert dot_triple.situation["note"].startswith("user asked")


@pytest.mark.parametrize(
    "backend", ["minimal"] + (["jsonschema"] if HAS_JSONSCHEMA else [])
)
def test_demo_with_each_schema_backend(backend):
    centre = build_centre(backend=backend)
    task = centre.submit(
        demo_specs(backend).make("arith.dot", {"x": [1.5, -2], "y": [2, 3]})
    )
    centre.run_until_idle()
    assert centre.result(task) == pytest.approx(-3.0)


def test_planner_rejects_length_mismatch():
    centre = build_centre()
    task = centre.submit(centre.specs.make("arith.dot", {"x": [1, 2], "y": [1]}))
    centre.run_until_idle()
    assert "length mismatch" in centre.status(task).error


def test_main(tmp_path, capsys):
    assert main([str(tmp_path / "t.jsonl")]) == 0
    assert "arith.dot -> completed: 32" in capsys.readouterr().out
