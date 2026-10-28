"""R12c (goal G10b): the paired desk wrapper and the report."""

import datetime as dt
from types import SimpleNamespace

import pytest

from genetic_coil_arena.envs.logtriage.catalogue import DESK_NOW_R12
from genetic_coil_arena.experiments import r11_incident as inc
from genetic_coil_arena.experiments import r12_incident as inc3
from genetic_coil_arena.experiments import r12c_desk as desk4
from genetic_coil_arena.experiments import r12c_report as rep4
from genetic_coil_arena.experiments.r11_pipeline import _final
from genetic_coil_arena.translator import OK, REFUSED, TranslationResult

from .test_r12 import small_catalogue

NOW = DESK_NOW_R12


def base(**kw):
    return {
        "system": "train-ticket",
        "top_k": 3,
        "want_fault_type": True,
        "exclude_services": [],
        **kw,
    }


class FakeDesk:
    """Plays the translator: returns what the 'model' wrote and counts its calls."""

    def __init__(self, params):
        self.params = params
        self.calls = 0
        self.registry = inc3.build_registry_r12()
        self.catalogue = small_catalogue()
        self.domains = {"incident": SimpleNamespace(spec_name=inc.INVESTIGATE)}

    def handle(self, text, report=False):
        self.calls += 1
        spec = self.registry.make(inc.INVESTIGATE, self.params)
        tr = TranslationResult(text, OK, spec=spec, attempts=1)
        return _final(text, "incident", tr, self.catalogue, 0.0, "")


def pair(params, text):
    inner = FakeDesk(params)
    raw, checks = {}, {}
    off = desk4.CheckedDesk(inner, NOW, False, raw, checks)
    on = desk4.CheckedDesk(inner, NOW, True, raw, checks)
    return inner, off.handle(text), on.handle(text), checks


class TestCheckedDesk:
    def test_one_translator_call_serves_both_sides(self):
        inner, a, b, _ = pair(
            base(at="2024-11-22T06:08", system="sock-shop"), "sock shop Nov 22 06:08"
        )
        assert inner.calls == 1
        assert a.status == OK and b.status == OK
        assert a.spec.params["at"] == "2024-11-22T06:08"
        assert b.spec.params["at"] == "--11-22T06:08"
        assert a.entry.ticket == b.entry.ticket == "INC-2002"

    def test_day_only_reading_is_restored(self):
        inner, a, b, _ = pair(
            base(at="2024-12-10T12:03"), "train ticket on the 10th at 12:03"
        )
        assert b.spec.params["at"] == "---10T12:03"
        assert a.status == OK and b.status == OK
        assert a.entry.ticket == b.entry.ticket == "INC-2004"

    def test_clamped_count_is_refused_when_checked(self):
        inner, a, b, _ = pair(
            base(ticket="INC-2001", system="sock-shop", top_k=5),
            "sock shop INC-2001, top 8 suspects",
        )
        assert a.status == OK  # the unchecked desk answers with 5
        assert b.status == REFUSED and b.entry is None

    def test_unchanged_when_nothing_fires(self):
        _, a, b, checks = pair(base(ticket="INC-2004"), "train ticket INC-2004")
        assert a.spec.params == b.spec.params
        assert checks["train ticket INC-2004"].rules == ()

    def test_reports_are_never_checked(self):
        inner = FakeDesk(base(at="2024-11-22", system="sock-shop"))
        on = desk4.CheckedDesk(inner, NOW, True, {}, {})
        res = on.handle("A report on the sock shop incident of Nov 22", report=True)
        assert res.spec.params["at"] == "2024-11-22"


def rec(i, kind="spec", exact=True, status="ok", ticket=None, **kw):
    return {
        "id": i, "kind": kind, "category": "time", "style": "plain", "exact": exact,
        "status": status, "ticket": ticket, "route_ok": True, "decision_ok": True,
        "fields": {}, "gold": {}, **kw,
    }  # fmt: skip


def prow(i, off_exact, on_exact, **kw):
    r = rec(i, exact=on_exact, **kw)
    r["off"] = rec(i, exact=off_exact, **kw)
    r["rules"] = ["year"] if off_exact != on_exact else []
    return r


class TestReport:
    def test_mcnemar(self):
        assert rep4.mcnemar_exact(0, 0) == 1.0
        assert rep4.mcnemar_exact(9, 0) == pytest.approx(2 / 512)
        assert rep4.mcnemar_exact(5, 5) == 1.0
        assert rep4.mcnemar_exact(6, 1) == pytest.approx(2 * 8 / 128)

    def test_silent_wrong(self):
        assert not desk4.silent_wrong(rec("a", ticket="INC-1"), "INC-1")
        assert desk4.silent_wrong(rec("a", ticket="INC-2"), "INC-1")
        assert not desk4.silent_wrong(rec("a", status="refused"), "INC-1")
        assert desk4.silent_wrong(rec("a", kind="refuse"), None)
        assert desk4.silent_wrong(rec("a", kind="clarify"), None)
        assert not desk4.silent_wrong(rec("a", kind="refuse", status="refused"), None)

    def test_paired_counts_and_verdict(self):
        resolves = {f"i{k}": f"INC-{k}" for k in range(40)}
        rows = [prow(f"i{k}", k >= 12, True, ticket=f"INC-{k}") for k in range(40)]
        m = rep4.g10b_metrics(rows, resolves, set())
        assert len(m["paired"]["fixed"]) == 12 and not m["paired"]["broke"]
        assert m["on"]["exact"]["p"] == 1.0 and m["off"]["exact"]["p"] == 0.7
        assert m["paired"]["p"] < 0.05
        v = rep4.verdict(m)
        # no gold run in the synthetic rows: the task criterion is n/a, so not a PASS
        assert v["verdict"] == "NOT PASSED" and not v["kill_reasons"]
        results = [c["result"] for c in v["criteria"]]
        assert results[:3] == ["PASS", "PASS", "PASS"] and results[3] == "n/a"

    def test_kill_when_the_check_breaks_more_than_it_fixes(self):
        resolves = {f"i{k}": f"INC-{k}" for k in range(20)} | {"z": "INC-9"}
        rows = [prow(f"i{k}", True, k >= 3, ticket=f"INC-{k}") for k in range(20)]
        rows.append(prow("z", False, True, ticket="INC-9"))
        m = rep4.g10b_metrics(rows, resolves, set())
        assert len(m["paired"]["broke"]) == 3 and len(m["paired"]["fixed"]) == 1
        assert rep4.verdict(m)["verdict"] == "KILL"

    def test_traps_are_counted_apart(self):
        resolves = {"t": None, "u": "INC-1"}
        rows = [
            prow("t", True, True, kind="refuse", status="ok", ticket="INC-5"),
            prow("u", True, True, ticket="INC-1"),
        ]
        m = rep4.g10b_metrics(rows, resolves, {"t"})
        assert m["on"]["silent_wrong"] == ["t"] == m["on"]["silent_wrong_traps"]

    def test_posthoc_rescoring_of_a_stored_run(self):
        registry = inc3.build_registry_r12()
        schema = registry.get(inc.INVESTIGATE).params_schema
        gold = base(at="--11-22T06:08", system="sock-shop")
        stored = [
            {"id": "a", "kind": "spec", "status": "ok", "exact": False, "gold": gold,
             "text": "sock shop Nov 22 06:08", "pred": {**gold, "at": "2024-11-22T06:08"}},
            {"id": "b", "kind": "refuse", "status": "ok", "text": "sock shop INC-2001 top 8",
             "pred": base(ticket="INC-2001", system="sock-shop", top_k=5), "gold": None},
        ]  # fmt: skip
        out = rep4.posthoc(stored, registry, inc.INVESTIGATE, NOW, schema)
        assert out["fixed"] == ["a"] and out["broke"] == []
        assert out["answered_but_would_refuse"] == ["b"]


def test_clock_is_the_r12_clock():
    assert NOW == dt.datetime(2024, 12, 11, 9, 0, tzinfo=dt.timezone.utc)
