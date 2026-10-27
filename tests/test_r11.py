"""R11 (goal G8): catalogue, spec, sealed set, facts, desk and the end-to-end chain.

Everything runs without a model: the desk is driven by ``KeywordBackend`` (the keyword router
and parser behind the structured-output interface). Checks that need the RCAEval cache skip
when it is absent.
"""

import json
from collections import Counter
from dataclasses import replace

import pytest

from genetic_coil_arena.envs.logtriage import rcaeval
from genetic_coil_arena.envs.logtriage.catalogue import (
    TOLERANCE_S,
    CatalogueEntry,
    IncidentCatalogue,
    ResolveError,
    parse_at,
    utc_text,
)
from genetic_coil_arena.experiments import log_triage3 as t3
from genetic_coil_arena.experiments import r7a_sealed
from genetic_coil_arena.experiments import r11_incident as inc
from genetic_coil_arena.experiments import r11_pipeline as pipe
from genetic_coil_arena.experiments import r11_report as rep
from genetic_coil_arena.experiments import r11_sealed as sealed
from genetic_coil_arena.experiments import r11_team as team
from genetic_coil_arena.experiments.r7a_domains import (
    SQUAD_MISSION,
    build_domains,
    outcome_of,
)
from genetic_coil_arena.protocol.goalspec import SpecValidationError
from genetic_coil_arena.translator import (
    OverlapIndex,
    TraceView,
    exact_match,
    fact_fidelity,
)

T0 = 1_705_500_000.0  # 2024-01-17 14:00 UTC


def synthetic_catalogue() -> IncidentCatalogue:
    """Five incidents: three sock-shop on 2024-01-17, one on the 18th, one boutique."""

    def entry(k: int, code: str, when: float) -> CatalogueEntry:
        services = (
            inc.SOCK_SHOP_SERVICES if code == "SS" else inc.ONLINE_BOUTIQUE_SERVICES
        )
        return CatalogueEntry(f"INC-{1001 + k}", f"{code}/x{k}/2", code, when, services)

    return IncidentCatalogue(
        [
            entry(0, "SS", T0),
            entry(1, "SS", T0 + 3600),
            entry(2, "SS", T0 + 7200),
            entry(3, "SS", T0 + 86400),
            entry(4, "OB", T0 + 60),
        ]
    )


def params(**kw):
    base = {
        "system": "sock-shop",
        "top_k": 3,
        "want_fault_type": True,
        "exclude_services": [],
    }
    return {**base, **kw}


# ------------------------------------------------------------------ catalogue


class TestResolve:
    cat = synthetic_catalogue()

    def test_ticket_resolves(self):
        assert self.cat.resolve(params(ticket="INC-1002")).incident_id == "SS/x1/2"

    def test_time_within_tolerance_names_one_incident(self):
        at = utc_text(T0 + 3600 + 5 * 60)
        assert self.cat.resolve(params(at=at)).ticket == "INC-1002"

    def test_time_outside_tolerance_refuses(self):
        at = utc_text(T0 + 3600 + TOLERANCE_S + 120)
        with pytest.raises(ResolveError) as exc:
            self.cat.resolve(params(at=at))
        assert exc.value.kind == "refuse"

    def test_date_only_with_several_incidents_clarifies(self):
        with pytest.raises(ResolveError) as exc:
            self.cat.resolve(params(at="2024-01-17"))
        assert exc.value.kind == "clarify" and "INC-1001" in exc.value.message

    def test_date_only_with_one_incident_resolves(self):
        assert self.cat.resolve(params(at="2024-01-18")).ticket == "INC-1004"

    def test_system_scopes_the_time_lookup(self):
        # the boutique incident is 60 s after the first shop incident: system decides
        at = utc_text(T0 + 60)
        assert (
            self.cat.resolve(params(at=at, system="online-boutique")).ticket
            == "INC-1005"
        )
        assert self.cat.resolve(params(at=at)).ticket == "INC-1001"

    def test_missing_reference_and_missing_system_clarify(self):
        for p in (
            params(),
            {k: v for k, v in params(ticket="INC-1001").items() if k != "system"},
        ):
            with pytest.raises(ResolveError) as exc:
                self.cat.resolve(p)
            assert exc.value.kind == "clarify"

    @pytest.mark.parametrize(
        "p",
        [
            params(ticket="INC-9999"),  # unknown ticket
            params(ticket="INC-1005"),  # a boutique ticket on the sock shop
            params(ticket="INC-1001", at="2024-01-19"),  # ticket and date disagree
            params(ticket="INC-1001", exclude_services=["billing"]),  # unknown service
            params(at="2024-01-17Tzz"),  # malformed
        ],
    )
    def test_refusals(self, p):
        with pytest.raises(ResolveError) as exc:
            self.cat.resolve(p)
        assert exc.value.kind == "refuse"

    def test_ticket_and_matching_time_resolve(self):
        p = params(ticket="INC-1001", at="2024-01-17")
        assert self.cat.resolve(p).ticket == "INC-1001"

    def test_fingerprint_is_deterministic_and_content_sensitive(self):
        again = synthetic_catalogue()
        assert again.fingerprint() == self.cat.fingerprint()
        moved = [replace(e, inject_time=e.inject_time + 1) for e in again.entries]
        assert IncidentCatalogue(moved).fingerprint() != self.cat.fingerprint()

    def test_parse_at(self):
        assert parse_at("2024-01-17")[0] is None
        assert parse_at("2024-01-17T14:00")[0] == T0
        with pytest.raises(ValueError):
            parse_at("17 January")


@pytest.fixture(scope="module")
def real_catalogue():
    files = rcaeval.list_incidents(None, ("SS", "OB"))
    if not files:
        pytest.skip("RCAEval cache not present")
    return team.load_catalogue()


class TestRealCatalogue:
    def test_is_the_120_test_incidents_with_stable_tickets(self, real_catalogue):
        cat = real_catalogue
        assert len(cat) == 120
        assert [e.ticket for e in cat.entries] == [
            f"INC-{1001 + k}" for k in range(120)
        ]
        times = [e.inject_time for e in cat.entries]
        assert times == sorted(times)
        for e in cat.entries:
            system, case, rep_no = e.incident_id.split("/")
            assert t3.role_of(system, int(rep_no)) == "test"
        assert team.load_catalogue().fingerprint() == cat.fingerprint()
        assert cat.by_ticket["INC-1001"].alert_utc == "2024-01-17T10:20"

    def test_service_lists_match_the_prompt(self, real_catalogue):
        assert tuple(real_catalogue.services_of("sock-shop")) == inc.SOCK_SHOP_SERVICES
        assert (
            tuple(real_catalogue.services_of("online-boutique"))
            == inc.ONLINE_BOUTIQUE_SERVICES
        )


# ------------------------------------------------------------------ spec


class TestSpec:
    registry = inc.build_registry_r11()

    def test_has_at_most_six_params_and_sits_beside_the_r7a_specs(self):
        props = self.registry.get(inc.INVESTIGATE).params_schema["properties"]
        assert len(props) == 6
        assert {SQUAD_MISSION, "plant.diagnose", inc.INVESTIGATE} <= set(
            self.registry.names()
        )

    def test_valid_params_make_a_goal(self):
        spec = self.registry.make(
            inc.INVESTIGATE, params(ticket="INC-1042", system="online-boutique")
        )
        assert spec.params["top_k"] == 3

    @pytest.mark.parametrize(
        "bad",
        [
            params(ticket="INC-42"),
            params(ticket="1042"),
            params(at="20 January"),
            params(top_k=6),
            params(top_k=0),
            params(system="hotel"),
            params(extra=1),
            {"top_k": 3},
        ],
    )
    def test_invalid_params_are_rejected(self, bad):
        with pytest.raises(SpecValidationError):
            self.registry.make(inc.INVESTIGATE, bad)


# ------------------------------------------------------------------ R7a compatibility


class TestR7aStillWorks:
    def test_build_domains_default_is_the_two_r7a_domains(self):
        domains = build_domains(None)
        assert list(domains) == ["squad", "plant"]
        assert all(d.outcome is None for d in domains.values())

    def test_extra_domains_are_appended_with_their_own_outcome(self):
        extra = {"incident": inc.incident_domain(lambda p: {})}
        domains = build_domains(None, extra=extra)
        assert list(domains) == ["squad", "plant", "incident"]
        out = domains["incident"].outcome(
            {"score": 1.0, "public": {"ticket": "INC-1001", "system": "sock-shop"}}
        )
        assert out == {
            "state": "completed",
            "ticket": "INC-1001",
            "system": "sock-shop",
        }

    def test_outcome_of_by_name_is_unchanged(self):
        out = outcome_of(
            "squad", {"score": 3.14159, "food": 1.0, "survivors": 2.0, "steps": 9}
        )
        assert out["score"] == 3.1


# ------------------------------------------------------------------ sealed and dev sets


@pytest.fixture(scope="module")
def sealed_items():
    return sealed.load_sealed()


class TestSealed:
    def test_file_hash_is_pinned(self):
        assert sealed.file_sha256() == sealed.SEALED_SHA256

    def test_load_refuses_a_changed_file(self, tmp_path):
        bad = tmp_path / "x.jsonl"
        bad.write_bytes(sealed.SEALED_PATH.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="sealed"):
            sealed.load_sealed(path=bad)

    def test_size_and_mix(self, sealed_items):
        assert len(sealed_items) >= 120
        kinds = Counter(i.kind for i in sealed_items)
        assert kinds["spec"] >= 100
        assert 0.12 <= (kinds["clarify"] + kinds["refuse"]) / len(sealed_items) <= 0.20
        assert len({i.id for i in sealed_items}) == len(sealed_items)
        assert len({i.text for i in sealed_items}) == len(sealed_items)
        cats = {i.category for i in sealed_items}
        assert {
            "ticket",
            "time",
            "relative",
            "topk",
            "exclude",
            "fault",
            "combo",
        } <= cats

    def test_gold_specs_validate(self, sealed_items):
        registry = inc.build_registry_r11()
        for it in sealed_items:
            if it.kind == "spec":
                registry.make(inc.INVESTIGATE, it.gold)
            else:
                assert it.gold is None

    def test_gold_resolves_to_exactly_one_incident(self, sealed_items, real_catalogue):
        for it in sealed_items:
            if it.kind == "spec":
                assert (
                    real_catalogue.resolve(it.gold).system == it.gold["system"]
                ), it.id

    def test_dev_is_separate_from_sealed(self, sealed_items):
        dev = sealed.load_dev()
        assert len(dev) >= 40
        assert not {i.text for i in dev} & {i.text for i in sealed_items}

    def test_few_shots_and_route_examples_share_nothing_with_held_out_texts(
        self, sealed_items
    ):
        refs = [i.text for i in sealed_items] + [i.text for i in sealed.load_dev()]
        refs7 = [
            i.text
            for i in (
                *r7a_sealed.load_sealed(),
                *r7a_sealed.load_dev(),
                *r7a_sealed.load_old_set(),
            )
        ]
        shots = (
            [e.text for e in inc.INCIDENT_POLICY_EXAMPLES]
            + [e.text for e in inc.INCIDENT_REPORT_EXAMPLES]
            + [t for t, _ in inc.ROUTE_EXAMPLES]
        )
        for refs_ in (refs, refs7):
            index = OverlapIndex(refs_, 4)
            for text in shots:
                score, j = index.max_containment(text)
                assert score < 0.5, (text, refs_[j], score)
        assert not set(shots) & set(refs)


# ------------------------------------------------------------------ facts


def make_view(p, public) -> TraceView:
    return TraceView(inc_goal(p), (), {"state": "completed", **public})


def inc_goal(p):
    return inc.build_registry_r11().make(inc.INVESTIGATE, p)


PUBLIC = {
    "ticket": "INC-1073",
    "system": "sock-shop",
    "suspects": ["orders", "carts-db", "user"],
    "fault_type": "mem",
    "fault_probability": 0.62,
    "onset_utc": "2024-01-20T06:12",
    "excluded_services": ["front-end"],
}
GOOD = (
    "Investigation of the sock-shop incident, requested by time 2024-01-20T06:10 "
    "(ticket INC-1073). The suspected services, most likely first, are orders, carts-db, user; "
    "fault type memory (probability 0.62); onset 06:12 UTC. Excluded: front-end."
)


class TestFacts:
    view = make_view(
        params(at="2024-01-20T06:10", exclude_services=["front-end"]), PUBLIC
    )

    def fidelity(self, text):
        return fact_fidelity(text, inc.incident_facts(self.view))

    def test_a_faithful_summary_states_every_fact(self):
        fid = self.fidelity(GOOD)
        assert fid["all"], fid["facts"]
        assert set(fid["facts"]) == {
            "system", "ticket", "suspects", "at", "fault_type",
            "fault_probability", "onset", "exclusions",
        }  # fmt: skip

    @pytest.mark.parametrize(
        "bad, fact",
        [
            (
                GOOD.replace("orders, carts-db, user", "user, carts-db, orders"),
                "suspects",
            ),
            (GOOD.replace("INC-1073", "INC-1074"), "ticket"),
            (GOOD.replace("sock-shop", "online-boutique"), "system"),
            (GOOD.replace("memory", "cpu"), "fault_type"),
            (GOOD.replace("0.62", "0.26"), "fault_probability"),
            (GOOD.replace("onset 06:12", "onset unknown"), "onset"),
            (
                GOOD.replace(" Excluded: front-end.", " No services excluded."),
                "exclusions",
            ),
            (GOOD.replace("2024-01-20T06:10", "2024-01-21T06:10"), "at"),
        ],
    )
    def test_a_wrong_or_missing_statement_fails_its_fact(self, bad, fact):
        assert not self.fidelity(bad)["facts"][fact]

    def test_suspect_names_do_not_match_longer_service_names(self):
        view = make_view(params(ticket="INC-1073"), {**PUBLIC, "suspects": ["carts"]})
        facts = inc.incident_facts(view)
        text = "system sock-shop INC-1073: suspects carts-db."
        assert not fact_fidelity(text, facts)["facts"]["suspects"]

    def test_no_fault_type_requested_and_unknown_onset(self):
        public = {k: v for k, v in PUBLIC.items() if not k.startswith("fault")}
        public.update(onset_utc=None, excluded_services=[])
        view = make_view(params(ticket="INC-1073", want_fault_type=False), public)
        text = (
            "The sock shop incident INC-1073, requested by ticket INC-1073: suspects orders, "
            "carts-db, user; no fault type was requested; onset unknown; no services excluded."
        )
        fid = fact_fidelity(text, inc.incident_facts(view))
        assert fid["all"], fid["facts"]
        wrong = text.replace("no fault type was requested", "fault type cpu")
        assert not fact_fidelity(wrong, inc.incident_facts(view))["facts"]["fault_type"]


# ------------------------------------------------------------------ keyword baselines


class TestKeyword:
    def test_parses_ticket_system_topk_and_exclusion(self):
        p = inc.keyword_incident(
            "INC-1095 on the sock shop: give me 2 suspects but not the front-end, no fault type."
        )
        assert p == params(
            ticket="INC-1095",
            top_k=2,
            want_fault_type=False,
            exclude_services=["front-end"],
        )

    def test_parses_a_date_and_time(self):
        p = inc.keyword_incident("boutique 19 Jan 20:07 suspects")
        assert p["at"] == "2024-01-19T20:07" and p["system"] == "online-boutique"

    def test_out_of_domain_is_refused(self):
        assert inc.keyword_incident("What is the capital of France?") is None

    def test_router(self):
        assert inc.keyword_route("Let red and green forage for 90 steps") == "squad"
        assert inc.keyword_route("8 bytes per node, reactor only") == "plant"
        assert inc.keyword_route("Investigate INC-1042 on the boutique") == "incident"
        assert inc.keyword_route("Tell me a joke") == "none"


# ------------------------------------------------------------------ the desk (no model)


@pytest.fixture(scope="module")
def stub_desk():
    cat = synthetic_catalogue()
    registry = inc.build_registry_r11()
    domains = build_domains(None)
    domains["incident"] = inc.incident_domain(lambda p: {})
    return pipe.Desk(pipe.KeywordBackend(), registry, domains, cat)


class TestDesk:
    def test_routes_and_translates_an_incident_request(self, stub_desk):
        res = stub_desk.handle("Investigate INC-1002 on the sock shop, top 2 suspects")
        assert (res.domain, res.status) == ("incident", "ok")
        assert res.spec.params["top_k"] == 2 and res.entry.ticket == "INC-1002"

    def test_other_domains_are_routed_and_translated_by_their_own_pack(self, stub_desk):
        res = stub_desk.handle("Let the red and green snakes forage")
        assert res.domain == "squad"

    def test_missing_system_becomes_a_question(self, stub_desk):
        res = stub_desk.handle("Please look at ticket 1090")
        # keyword parser yields no system: the spec cannot be made, so the desk does not run it
        assert res.status != "ok"

    def test_ambiguous_day_becomes_a_clarifying_question(self, stub_desk):
        res = stub_desk.handle("Investigate the sock shop incident on 17 Jan")
        assert res.status == "clarify" and "INC-1001" in res.reason

    def test_unknown_ticket_is_refused_by_the_catalogue(self, stub_desk):
        res = stub_desk.handle("Investigate INC-9999 on the sock shop")
        assert res.status == "refused" and "INC-9999" in res.reason

    def test_out_of_domain_is_refused(self, stub_desk):
        res = stub_desk.handle("Tell me a joke")
        assert (res.domain, res.status) == ("none", "refused")


# ------------------------------------------------------------------ end to end (real incidents)


def faithful_summary(view: TraceView) -> str:
    """A templated summary written from the trace view only (what the model is asked for)."""
    p, o = view.goal.params, view.outcome
    how = (
        f"requested by ticket {p['ticket']}"
        if "ticket" in p
        else f"requested by time {p['at']} (ticket {o['ticket']})"
    )
    fault = (
        f"fault type {o['fault_type']} (probability {o['fault_probability']})"
        if "fault_type" in o
        else "no fault type was requested"
    )
    onset = f"onset {o['onset_utc'][-5:]} UTC" if o["onset_utc"] else "onset unknown"
    excl = (
        f"excluded: {', '.join(o['excluded_services'])}"
        if o["excluded_services"]
        else "no services excluded"
    )
    return (
        f"Investigation of the {o['system']} incident, {how}. The suspected services, most likely "
        f"first, are {', '.join(o['suspects'])}; {fault}; {onset}; {excl}."
    )


@pytest.fixture(scope="module")
def chain(real_catalogue):
    """A small trained team (40 training incidents) behind a keyword-driven desk."""
    state = team.train_metric_specialist(limit_train=40)
    specialist = team.load_specialist(state)
    services = set(inc.SOCK_SHOP_SERVICES) | set(inc.ONLINE_BOUTIQUE_SERVICES)
    executor = team.IncidentExecutor(
        real_catalogue, team.IncidentTeam(specialist, services)
    )
    registry = inc.build_registry_r11()
    domains = build_domains(None)
    domains["incident"] = inc.incident_domain(executor)
    desk = pipe.Desk(pipe.KeywordBackend(), registry, domains, real_catalogue)
    return state, executor, desk


class TestEndToEnd:
    def test_training_never_touches_the_test_split(self, chain):
        state, _, _ = chain
        assert not set(state["train_ids"]) & set(state["test_ids"])
        assert all(i.startswith("TT/") or i.endswith("/1") for i in state["train_ids"])

    def test_three_real_incidents_through_the_whole_chain(self, chain, real_catalogue):
        _, executor, desk = chain
        items = [
            i
            for i in sealed.load_sealed()
            if i.kind == "spec" and i.category in ("ticket", "time", "combo")
        ][:3]
        assert len(items) == 3
        for item in items:
            run = pipe.process_incident_item(desk, item)
            rec, view = run.record, run.view
            assert rec["status"] == "ok" and rec["nl_state"] == "completed"
            entry = real_catalogue.by_ticket[rec["ticket"]]
            # the trace view is what the summariser may see: no label, no task score
            assert set(view.outcome) <= {
                "state", "ticket", "system", "suspects", "fault_type", "fault_probability",
                "onset_utc", "excluded_services",
            }  # fmt: skip
            assert view.outcome["ticket"] == entry.ticket
            assert len(view.outcome["suspects"]) == view.goal.params["top_k"]
            assert [h["codec"] for h in view.headers][0] == "goalspec"
            assert rec["n_envelopes"] >= 2 * len(
                entry.services
            )  # a goal and a result per specialist
            gold_view_ticket = rec["gold_ticket"]
            if rec["exact"]:
                assert rec["ticket"] == gold_view_ticket
                assert rec["nl_score"] == rec["gold_score"]
            # a faithful summary of the view states every fact
            fid = fact_fidelity(faithful_summary(view), inc.incident_facts(view))
            assert fid["all"], (item.text, fid["facts"])

    def test_exclusion_changes_the_candidates(self, chain, real_catalogue):
        _, executor, _ = chain
        entry = real_catalogue.by_ticket["INC-1090"]
        base = executor(params(ticket="INC-1090"))
        top = base["public"]["suspects"][0]
        again = executor(params(ticket="INC-1090", exclude_services=[top]))
        assert top not in again["public"]["suspects"]
        assert again["public"]["excluded_services"] == [top]
        assert entry.system == base["public"]["system"]

    def test_unresolvable_params_fail_the_task_not_the_process(self, chain):
        from genetic_coil_arena.experiments.r7a_pipeline import execute

        _, _, desk = chain
        spec = desk.registry.make(inc.INVESTIGATE, params(ticket="INC-9999"))
        run = execute(desk.domains["incident"], desk.registry, spec)
        assert run.state != "completed" and run.view is None

    def test_exact_match_helper_uses_the_spec_schema(self, chain):
        _, _, desk = chain
        schema = desk.registry.get(inc.INVESTIGATE).params_schema
        a = params(ticket="INC-1090", exclude_services=["a", "b"])
        b = params(ticket="INC-1090", exclude_services=["b", "a"])
        assert exact_match(a, b, schema)


# ------------------------------------------------------------------ the report


class TestReport:
    def records(self):
        rows = []
        for k in range(10):
            ok = k != 0
            rows.append(
                {
                    "id": f"r{k}", "kind": "spec", "category": "ticket", "style": "plain",
                    "text": "t", "gold": {"a": 1}, "routed": "incident", "route_ok": True,
                    "status": "ok", "reason": "", "decision_ok": True, "pred": {"a": 1},
                    "exact": ok, "fields": {"a": ok}, "ticket": "INC-1", "gold_ticket": "INC-1",
                    "nl_score": 1.0 if ok else 0.0, "gold_score": 1.0, "nl_top1": 1.0,
                    "gold_top1": 1.0, "translate_s": 1.0, "team_s": 0.1, "summary_s": 1.0,
                    "total_s": 2.1, "facts": {"x": True, "y": ok}, "all_facts": ok,
                    "round_trip": ok, "summary": "s",
                }
            )  # fmt: skip
        rows.append(
            {
                "id": "c", "kind": "clarify", "category": "clarify", "style": "plain",
                "text": "t", "gold": None, "routed": "incident", "route_ok": True,
                "status": "clarify", "reason": "?", "decision_ok": True, "pred": None,
            }
        )  # fmt: skip
        return rows

    def routes(self):
        return [
            {
                "id": "a",
                "kind": "spec",
                "domain": "squad",
                "routed": "squad",
                "route_ok": True,
            },
            {
                "id": "b",
                "kind": "spec",
                "domain": "plant",
                "routed": "none",
                "route_ok": False,
            },
        ]

    def test_metrics_and_verdict(self):
        m = rep.system_metrics(self.records(), self.routes())
        assert m["exact"]["hits"] == 9 and m["exact"]["n"] == 10
        assert m["routing"]["n"] == 13 and m["routing"]["hits"] == 12
        assert m["round_trip"]["hits"] == 9
        assert m["fidelity"]["facts"] == pytest.approx(19 / 20)
        assert m["task"]["diff"]["mean"] == pytest.approx(-0.1)
        assert m["latency"]["total_p95"] == pytest.approx(2.1)
        v = rep.verdict(m, 9.0)
        assert v["verdict"] == "NOT PASSED" and not v["kill_reasons"]
        rows = {r["criterion"]: r["result"] for r in v["criteria"]}
        assert (
            rows["exact match"] == "PASS"
            and rows["task score, NL vs gold"] == "not met"
        )
        assert rows["domain routing"] == "not met"  # 12 of 13
        assert rows["peak VRAM"] == "PASS"

    def test_kill_rules(self):
        recs = self.records()
        for r in recs[:6]:
            r["exact"] = False
        m = rep.system_metrics(recs, self.routes())
        assert any("exact match" in k for k in rep.verdict(m, None)["kill_reasons"])

    def test_render_mentions_the_analysis_block_and_failures(self):
        recs = self.records()
        m = rep.system_metrics(recs, self.routes())
        report = {
            "generated": "now", "n_sealed": 11, "sealed_sha256": "x", "r7a_sealed_sha256": "y",
            "catalogue": {"n": 120, "fingerprint": "f"}, "team": "t",
            "systems": {
                "s": {
                    "label": "stub", "metrics": m, "verdict": rep.verdict(m, None),
                    "failures": rep.failures(recs),
                }
            },
        }  # fmt: skip
        md = rep.render(report)
        assert "<!-- analysis:start -->" in md and "<!-- analysis:end -->" in md
        assert "failure examples" in md
        json.dumps(m)  # JSON-safe
