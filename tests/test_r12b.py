"""R12b (goal G9b): the attributing newcomer, the v2 centre, the freeze and the verdict."""

import numpy as np
import pytest

from genetic_coil_arena.envs.logtriage import novelty as nv
from genetic_coil_arena.envs.logtriage import novelty2 as n2
from genetic_coil_arena.envs.logtriage.incident import Incident
from genetic_coil_arena.experiments import r12b_report as rep

SS = ["carts", "carts-db", "front-end", "orders", "orders-db", "payment", "user"]
OB = ["adservice", "cartservice", "frontend", "paymentservice", "redis"]
TT = [
    "ts-order-service",
    "ts-order-other-service",
    "ts-order-mongo",
    "ts-payment-service",
    "ts-inside-payment-service",
    "ts-ui-dashboard",
]


def names(services, text, logger=None):
    m = n2.NameMatcher(services)
    return sorted(services[i] for i in m.mentions(text))


class TestNameMatcher:
    def test_host_and_url_forms_name_ordinary_word_services(self):
        assert names(SS, "dial tcp payment:80: connection refused") == ["payment"]
        assert names(SS, "Get http://payment/paymentAuth: timeout") == ["payment"]
        assert names(SS, "connect to user:<n> failed") == ["user"]
        assert names(SS, "java.net.UnknownHostException: orders-db") == ["orders-db"]
        assert names(SS, "java.net.UnknownHostException: shipping") == []  # not in list
        assert names(SS, "lookup payment on 10.0.0.10:53: no such host") == ["payment"]
        assert names(SS, "call payment.sock-shop.svc failed") == ["payment"]

    def test_bare_forms_need_a_distinctive_name(self):
        assert names(SS, "mongo error talking to carts-db") == ["carts-db"]
        assert names(SS, "[carts-db:<n>] org.mongodb.driver.cluster") == ["carts-db"]
        assert names(OB, "rpc error calling paymentservice") == ["paymentservice"]
        assert names(OB, "rpc error calling payment-service") == ["paymentservice"]
        assert names(OB, "cannot reach redis") == []  # an ordinary word, no host form
        assert names(OB, "cannot reach redis:6379") == ["redis"]
        assert names(TT, "POST http://ts-order-service:12031/api/v1") == [
            "ts-order-service"
        ]
        assert names(TT, "ts_order_service refused") == ["ts-order-service"]

    def test_ordinary_words_paths_packages_and_emails_do_not_match(self):
        assert names(SS, "POST /orders HTTP/1.1 500") == []
        assert names(SS, "the user entry was rejected") == []
        assert names(SS, "mail to payment@example.com bounced") == []
        assert (
            names(["rabbitmq", "queue-master"], "com.rabbitmq.client.ShutdownSignal")
            == []
        )
        assert names(["adservice", "frontend"], "failed to retrieve ads") == []

    def test_longer_names_win_over_their_prefixes(self):
        assert names(TT, "calling ts-order-other-service:1") == [
            "ts-order-other-service"
        ]
        assert names(["frontend", "frontend-external"], "to frontend-external") == [
            "frontend-external"
        ]

    def test_shared_cores_and_database_cores_are_not_forms(self):
        assert names(TT, "order:80 failed") == ["ts-order-service"]
        assert names(TT, "order-mongo is down") == []  # a database has no core form
        assert names(TT, "ts-order-mongo is down") == ["ts-order-mongo"]
        assert names(TT, "payment:80 failed") == ["ts-payment-service"]
        amb = ["ts-a-service", "a-service", "x"]
        assert names(amb, "a:80 refused") == []

    def test_forms_listing(self):
        m = n2.NameMatcher(SS)
        f = m.forms(SS.index("payment"))
        assert "payment" in f["host"] and "paymentservice" in f["bare"]
        assert m.forms(SS.index("carts-db"))["bare"] == ["carts-db"]


def make_incident():
    """orders and front-end log errors that name payment; payment logs one of its own."""
    services = ["front-end", "orders", "payment"]
    templates = [
        "ok line",
        "ERROR failed to call http://payment:80/pay",
        "ERROR timeout posting to payment:<n>",
        "ERROR local failure in orders",
        "ERROR payment crashed internally",
        "WARN slow response",
    ]
    t = np.array([-5.0, -5.0, -5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    svc = np.array([0, 1, 2, 0, 1, 1, 2, 0, 2])
    tpl = np.array([0, 0, 0, 1, 2, 3, 4, 1, 5])
    return Incident(
        id="X/payment_loss/2",
        system="X",
        services=services,
        root_service="payment",
        fault_type="loss",
        rep=2,
        t=t,
        svc=svc,
        tpl=tpl,
        templates=templates,
        window=(-5.0, 10.0),
    )


class TestNewcomerV2:
    def test_attribution_counts_and_schema(self):
        inc = make_incident()
        msgs = {s: n2.novelty2_message(inc, i) for i, s in enumerate(inc.services)}
        for m in msgs.values():
            assert n2.novelty2_errors(m) == []
            assert m["n_attr"] == m["own"] + m["about"]
            assert len(m["top"]) <= 3 and all(len(x) <= 80 for x in m["top"])
        # front-end's one template and one of orders' two name payment; payment logged two
        assert msgs["payment"]["about"] == 2
        assert msgs["orders"]["own"] == 1 and msgs["orders"]["about"] == 0
        assert msgs["payment"]["own"] == 2
        assert msgs["orders"]["logged"] == 2
        assert msgs["front-end"]["logged"] == 1 and msgs["front-end"]["n_attr"] == 0
        # G9's count is the number logged by the service itself
        assert sum(m["logged"] for m in msgs.values()) == sum(
            m["own"] + m["about"] for m in msgs.values()
        )
        assert any(x.startswith("orders: ") for x in msgs["payment"]["top"]) or any(
            x.startswith("front-end: ") for x in msgs["payment"]["top"]
        )

    def test_attribution_moved_and_cache_survive_dropping_the_lines(self):
        inc = make_incident()
        first = [n2.novelty2_message(inc, i) for i in range(3)]
        moved = n2.attribution_moved(inc)
        assert moved["templates_redirected"] == 2 and moved["services_moved"] == 3
        inc.t = inc.t[:0]
        inc.svc = inc.svc[:0]
        inc.tpl = inc.tpl[:0]
        assert first == [n2.novelty2_message(inc, i) for i in range(3)]

    def test_schema_rejects_extra_fields_and_long_reports(self):
        ok = {
            "service": "a",
            "own": 1,
            "about": 2,
            "n_attr": 3,
            "logged": 2,
            "top": ["x"],
        }
        assert n2.novelty2_errors(ok) == []
        assert n2.novelty2_errors({**ok, "extra": 1})
        assert n2.novelty2_errors({**ok, "top": ["a"] * 4})
        assert n2.novelty2_errors({**ok, "top": ["a" * 81]})


class TestCentreV2:
    def reports(self, n_attr, logged):
        z = [10.0, 20.0, 5.0]
        metric = [{"score": s} for s in z]
        novelty = [{"n_attr": a, "logged": b} for a, b in zip(n_attr, logged)]
        return metric, novelty

    def test_candidates_and_reference_rows(self):
        assert len(n2.RULES2) == 8
        assert n2.REF_RULE not in n2.RULES2 and n2.ALONE_RULE not in n2.RULES2
        assert nv.RULES["gate45_cap50"] == n2._ALL[n2.REF_RULE]

    def test_reference_counts_logged_and_candidates_count_n_attr(self):
        # z ranks: svc1 first, svc0 second, svc2 third
        metric, novelty = self.reports([0, 0, 4], [3, 0, 0])
        order_ref, _ = n2.joined2_ranking(n2.REF_RULE, metric, novelty)
        order_new, _ = n2.joined2_ranking("a_rr_w1", metric, novelty)
        assert order_ref[0] == 0  # 1/2 + rr(logged) 1 beats svc1's 1
        assert order_new[0] == 2  # 1/3 + rr(n_attr) 1 beats svc1's 1
        assert n2.count_key(n2.REF_RULE) == "logged"
        assert n2.count_key("a_gate45") == "n_attr"

    def test_alone_ranks_by_n_attr_and_old_rule_by_z(self):
        metric, novelty = self.reports([0, 1, 4], [0, 0, 0])
        assert n2.joined2_ranking(n2.ALONE_RULE, metric, novelty)[0][0] == 2
        assert n2.joined2_ranking(nv.OLD_RULE, metric, novelty)[0][0] == 1

    def test_gate_and_cap(self):
        metric, novelty = self.reports([0, 0, 4], [0, 0, 0])
        metric[1]["score"] = 60.0
        _, used = n2.joined2_ranking("a_gate45", metric, novelty)
        assert used is False
        metric[1]["score"] = 20.0
        novelty[2]["n_attr"] = 60
        _, used = n2.joined2_ranking("a_cap50", metric, novelty)
        assert used is False

    def test_definitions_validate_reports(self):
        defs = {(d.name, d.version): d for d in n2.joined2_definitions()}
        assert set(defs) == {(nv.SPEC_LOG_NOVELTY, 2), (nv.SPEC_DECIDE_JOINED, 2)}
        assert n2.ALL_RULES2[0] == nv.OLD_RULE


@pytest.fixture(scope="module")
def tiny():
    from genetic_coil_arena.envs.logtriage.synthetic import synthetic_incident
    from genetic_coil_arena.experiments import log_triage3 as t3
    from genetic_coil_arena.experiments import r11_team as team
    from genetic_coil_arena.experiments import r12_team as jt
    from genetic_coil_arena.experiments import r12b_team as bt
    from genetic_coil_arena.learning.classifiers import MLPClassifier

    train = [synthetic_incident(100 + i) for i in range(12)]
    data = t3.metric_dataset(train)
    model = MLPClassifier(6, epochs=40, seed=0).fit(data.x, data.y)
    services = [f"svc{i}" for i in range(6)]
    old = team.IncidentTeam(t3.MetricFaultSpecialist(model), services)
    joined = bt.JoinedTeamV2(old, services)
    incs = [bt.prepare(synthetic_incident(i, rep=2)) for i in range(4)]
    return old, joined, incs, jt, bt


class TestJoinedTeamV2:
    def test_old_state_hash_is_frozen_by_joining_and_running(self, tiny):
        old, joined, incs, jt, bt = tiny
        before = joined.frozen
        assert jt.old_state_hash(old) == before
        rules = list(n2.ALL_RULES2)
        rec = bt.run_incident_b(old, joined, incs[0], rules)
        assert set(rec["rules"]) == set(rules)
        assert rec["group"] in bt.GROUPS
        assert jt.old_state_hash(old) == before
        joined.assert_frozen()

    def test_metric_rule_reproduces_the_old_team(self, tiny):
        old, joined, incs, jt, bt = tiny
        for inc in incs:
            rec = bt.run_incident_b(old, joined, inc, [nv.OLD_RULE])
            assert rec["rules"][nv.OLD_RULE]["top3"] == rec["old"]["top3"]

    def test_state_hash_catches_a_change_to_the_old_team(self, tiny):
        old, joined, incs, jt, bt = tiny
        before = jt.old_state_hash(old)
        clf = old.model.model
        saved = clf.mean
        clf.mean = saved + 1.0
        try:
            assert jt.old_state_hash(old) != before
            with pytest.raises(RuntimeError, match="old team changed"):
                joined.assert_frozen()
        finally:
            clf.mean = saved
        assert jt.old_state_hash(old) == before

    def test_records_carry_attribution_info(self, tiny):
        old, joined, incs, jt, bt = tiny
        rec = bt.run_incident_b(old, joined, incs[1], [nv.OLD_RULE, n2.REF_RULE])
        assert set(rec["attr"]) == {
            "services_moved",
            "templates_redirected",
            "templates",
        }
        assert set(rec["root_counts"]) == {"own", "about", "logged"}

    def test_pinned_rule_is_a_selectable_candidate(self):
        from genetic_coil_arena.experiments import r12b_team as bt

        assert bt.CHOSEN_RULE_B in n2.RULES2


def rec(
    group, dataset, old, new, joint_old=None, joint_new=None, system="S", fault="cpu"
):
    def r(top1, joint):
        return {
            "top1": top1,
            "in_top3": top1,
            "joint": joint,
            "used_logs": True,
            "changed_top1": False,
        }

    base = r(old, joint_old)
    row = {
        "id": f"{system}/{group}",
        "group": group,
        "dataset": dataset,
        "system": system,
        "fault": fault,
        "old": base,
        "rules": {
            "R": {**r(new, joint_new)},
            n2.REF_RULE: {**r(new, joint_new)},
            n2.ALONE_RULE: {**r(new, joint_new)},
        },
        "baselines": {k: {"top1": old, "in_top3": old} for k in rep.BASELINES},
        "attr": {"services_moved": 1, "templates_redirected": 2, "templates": 5},
    }
    return row


def records(tt, re3, re2, n=60):
    """(old, new) top-1 outcomes per group, repeated to size n."""
    out = []
    for group, ds, (o, w) in (
        ("tt", "re2", tt),
        ("re3_test", "re3", re3),
        ("re2_test", "re2", re2),
    ):
        for i in range(n):
            out.append(
                rec(
                    group,
                    ds,
                    o if i % 2 == 0 else 1,
                    w if i % 2 == 0 else 1,
                    joint_old=1,
                    joint_new=1,
                )
            )
    return out


class TestVerdict:
    def test_pass(self):
        s = rep.summarise(records((1, 1), (0, 1), (1, 1)), "R")
        assert s["verdict"]["verdict"] == "PASS"
        assert [c["result"] for c in s["verdict"]["criteria"]] == ["PASS"] * 4

    def test_kill_on_tt_below_old(self):
        s = rep.summarise(records((1, 0), (0, 1), (1, 1), n=200), "R")
        v = s["verdict"]
        assert v["verdict"] == "KILL"
        assert any("Train Ticket" in k for k in v["kill_reasons"])

    def test_kill_when_the_re3_gain_ci_includes_zero(self):
        s = rep.summarise(records((1, 1), (0, 0), (1, 1)), "R")
        assert s["verdict"]["verdict"] == "KILL"
        assert "RE3 gain" in " ".join(s["verdict"]["kill_reasons"])

    def test_tt_is_scored_on_top1_only_and_breakdowns_exist(self):
        s = rep.summarise(records((1, 1), (0, 1), (1, 1)), "R")
        tt = s["groups"]["tt"]
        assert "joint" not in tt["rules"]["R"]
        assert "by_fault" in tt and "by_fault" not in s["groups"]["re3_test"]
        assert "joint" in s["groups"]["re2_test"]["rules"]["R"]

    def test_render_has_analysis_markers_and_flags_reused_splits(self):
        s = rep.summarise(records((1, 1), (0, 1), (1, 1)), "R")
        report = {
            "generated": "now",
            "rule": "R",
            "hash_before": "a",
            "hash_after": "a",
            "dev": {
                "chosen": "R",
                "table": [
                    {
                        "rule": "R",
                        "top1": 0.5,
                        "re2_top1": 0.5,
                        "re3_top1": 0.5,
                        "re2_joint": 0.5,
                        "used_logs": 0.5,
                    }
                ],
                "reference": [
                    {
                        "rule": r,
                        "top1": 0.5,
                        "re2_top1": 0.5,
                        "re3_top1": 0.5,
                        "re2_joint": 0.5,
                        "used_logs": None,
                    }
                    for r in (nv.OLD_RULE, n2.REF_RULE, n2.ALONE_RULE)
                ],
            },
            "summary": s,
        }
        text = rep.render(report)
        assert "<!-- analysis:start -->" in text and "<!-- analysis:end -->" in text
        assert text.count("REUSED") == 2 and "identical" in text
