"""Test the append-only trace log: JSONL persistence, replay and training triples."""

import json

import pytest

from genetic_coil_arena.protocol import (
    Envelope,
    GoalSpec,
    complete,
    goal_envelope,
    sign,
)
from genetic_coil_arena.runtime import TRACE_FORMAT, TraceLog, TraceRecord


def conversation(trace_id="tr1"):
    request = goal_envelope(GoalSpec("a.b", {"x": 1}), "centre", "s", "t1", trace_id)
    return [request, complete(request, "s", 2)]


class TestTraceLog:
    def test_records_in_order(self):
        log = TraceLog()
        request, reply = conversation()
        log.snapshot("tr1", {"board": [1, 2]}, clock=0)
        log.record(request)
        log.event("note", "tr1", 1, detail="x")
        log.record(reply)
        assert [r.kind for r in log] == ["situation", "envelope", "event", "envelope"]
        assert [r.seq for r in log.records] == [0, 1, 2, 3]
        assert log.envelopes() == [request, reply]
        assert log.events("note")[0].data == {"event": "note", "detail": "x"}
        assert log.events("other") == []
        assert len(log) == 4

    def test_append_only(self):
        log = TraceLog()
        log.record(conversation()[0])
        records = log.records
        assert isinstance(records, tuple)
        with pytest.raises(AttributeError):
            records[0].seq = 5

    def test_rejects_non_json_before_writing(self):
        log = TraceLog()
        with pytest.raises(TypeError):
            log.snapshot("t", {"x": object()})
        assert len(log) == 0

    def test_situations_keyed_by_trace(self):
        log = TraceLog()
        log.snapshot("a", 1)
        log.snapshot("b", 2)
        log.snapshot("a", 3)
        assert log.situations("a") == [1, 3] and log.situation("a") == 3
        assert log.situation("missing") is None
        assert log.trace_ids() == ["a", "b"]

    def test_filter_by_trace(self):
        log = TraceLog()
        for envelope in conversation("x") + conversation("y"):
            log.record(envelope)
        assert len(log.envelopes("x")) == 2
        assert log.trace_ids() == ["x", "y"]

    def test_replay_into_sink(self):
        log = TraceLog()
        for envelope in conversation():
            log.record(envelope)
        seen = []
        replayed = log.replay(seen.append)
        assert seen == replayed == log.envelopes()
        assert log.replay(trace_id="nope") == []

    def test_triples(self):
        log = TraceLog()
        log.snapshot("tr1", {"s": 1})
        for envelope in conversation("tr1"):
            log.record(envelope)
        log.record(complete(conversation("tr2")[0], "s", 0))  # no goal in tr2
        first, second = log.triples()
        assert first.trace_id == "tr1" and first.situation == {"s": 1}
        assert first.spec.spec == "a.b" and first.spec.params == {"x": 1}
        assert len(first.messages) == 2
        assert second.spec is None and second.situation is None

    def test_record_envelope_accessor(self):
        log = TraceLog()
        record = log.snapshot("t", 1)
        with pytest.raises(ValueError):
            record.envelope()
        with pytest.raises(ValueError):
            TraceRecord.from_dict({"seq": 0, "kind": "bogus", "data": 1})


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        log = TraceLog()
        log.snapshot("tr1", {"s": 1})
        signed = sign(conversation()[0], b"k")
        log.record(signed)
        path = log.save(tmp_path / "sub" / "trace.jsonl")
        lines = path.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0]) == {"format": TRACE_FORMAT, "version": 1}
        assert len(lines) == 3
        loaded = TraceLog.load(path)
        assert loaded.records == log.records
        assert loaded.envelopes()[0].manifest == signed.manifest

    def test_file_backed_appends_immediately(self, tmp_path):
        path = tmp_path / "live.jsonl"
        with TraceLog(path) as log:
            log.record(conversation()[0])
            assert len(path.read_text(encoding="utf-8").splitlines()) == 2
        with TraceLog(path) as resumed:
            assert len(resumed) == 1
            resumed.record(conversation()[1])
            assert resumed.records[-1].seq == 1
        assert len(TraceLog.load(path)) == 2

    def test_unclosed_log_does_not_warn(self, tmp_path):
        log = TraceLog(tmp_path / "x.jsonl")
        log.snapshot("t", 1)
        del log

    @pytest.mark.parametrize(
        "content, match",
        [
            ("", "empty"),
            ('{"format": "other"}\n', "not a"),
            ('{"format": "gca-trace", "version": 9}\n', "version"),
            ('{"format": "gca-trace", "version": 1}\n{"kind": "envelope"}\n', ":2:"),
        ],
    )
    def test_load_rejects_bad_files(self, tmp_path, content, match):
        path = tmp_path / "bad.jsonl"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            TraceLog.load(path)

    def test_replayed_envelopes_equal_originals(self, tmp_path):
        log = TraceLog()
        originals = conversation()
        for envelope in originals:
            log.record(envelope)
        loaded = TraceLog.load(log.save(tmp_path / "t.jsonl"))
        assert loaded.replay() == originals
        assert all(isinstance(e, Envelope) for e in loaded.replay())
