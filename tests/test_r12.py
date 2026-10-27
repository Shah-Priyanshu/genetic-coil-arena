"""R12 (goals G9 and G10): the log-novelty newcomer, the joined centre, the grown desk.

Everything runs without a model (the desk is driven by ``KeywordBackend``). Checks that need
the RCAEval cache skip when it is absent.
"""

import datetime as dt
import json
from collections import Counter

import pytest

from genetic_coil_arena.envs.logtriage import rcaeval
from genetic_coil_arena.envs.logtriage.catalogue import (
    DESK_NOW,
    DESK_NOW_R12,
    CatalogueEntry,
    IncidentCatalogue,
    ResolveError,
)
from genetic_coil_arena.experiments import r7a_sealed
from genetic_coil_arena.experiments import r11_incident as inc
from genetic_coil_arena.experiments import r11_pipeline as pipe
from genetic_coil_arena.experiments import r11_sealed as sealed1
from genetic_coil_arena.experiments import r11b_incident as incb
from genetic_coil_arena.experiments import r11b_sealed as sealed2
from genetic_coil_arena.experiments import r12_incident as inc3
from genetic_coil_arena.experiments import r12_sealed as sealed
from genetic_coil_arena.experiments.r7a_domains import build_domains
from genetic_coil_arena.protocol.goalspec import SpecValidationError
from genetic_coil_arena.translator import OverlapIndex
from genetic_coil_arena.translator.distill import frame_ngrams, frame_tokens

from .test_r11 import params

UTC = dt.timezone.utc


def ts(text: str) -> float:
    return dt.datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp()


def small_catalogue() -> IncidentCatalogue:
    """Two RE2-like sock-shop incidents in January, two RE3-like ones in November and a
    Train Ticket pair in December, read at the R12 clock."""
    rows = [
        ("INC-1001", "SS", "2024-01-20T14:21", inc.SOCK_SHOP_SERVICES),
        ("INC-1002", "OB", "2024-01-20T08:46", inc.ONLINE_BOUTIQUE_SERVICES),
        ("INC-2001", "RE3-SS", "2024-11-22T05:42", inc.SOCK_SHOP_SERVICES),
        ("INC-2002", "RE3-SS", "2024-11-22T06:08", inc.SOCK_SHOP_SERVICES),
        ("INC-2003", "RE3-TT", "2024-12-08T06:18", inc3.TRAIN_TICKET_SERVICES),
        ("INC-2004", "RE3-TT", "2024-12-10T12:03", inc3.TRAIN_TICKET_SERVICES),
    ]
    entries = [
        CatalogueEntry(t, f"{c}/x{k}/2", c, ts(w), tuple(s))
        for k, (t, c, w, s) in enumerate(rows)
    ]
    return IncidentCatalogue(entries, now=DESK_NOW_R12)


# ------------------------------------------------------------------ catalogue growth


class TestGrownCatalogue:
    cat = small_catalogue()

    def ticket(self, **kw):
        return self.cat.resolve(params(**kw)).ticket

    def test_new_codes_map_to_the_three_systems(self):
        assert {e.system for e in self.cat.entries} == set(inc3.SYSTEMS_V3)
        assert self.cat.by_ticket["INC-2003"].system == "train-ticket"

    def test_ticket_of_the_new_system_resolves(self):
        assert self.ticket(system="train-ticket", ticket="INC-2004") == "INC-2004"

    def test_a_ticket_of_another_system_refuses(self):
        with pytest.raises(ResolveError) as exc:
            self.cat.resolve(params(system="online-boutique", ticket="INC-2004"))
        assert exc.value.kind == "refuse"

    def test_times_name_the_re3_incidents(self):
        assert self.ticket(at="2024-11-22T06:10") == "INC-2002"
        assert self.ticket(system="train-ticket", at="---10T12:05") == "INC-2004"
        assert self.ticket(system="train-ticket", at="--12-08T06:18") == "INC-2003"

    def test_re2_regression_resolves_as_before(self):
        assert self.ticket(at="---20T14:21") == "INC-1001"
        assert self.ticket(at="2024-01-20T14:21") == "INC-1001"
        assert self.ticket(system="online-boutique", at="--01-20T08:46") == "INC-1002"

    def test_a_busy_day_asks(self):
        with pytest.raises(ResolveError) as exc:
            self.cat.resolve(params(at="2024-11-22"))
        assert exc.value.kind == "clarify" and "INC-2001" in exc.value.message

    def test_unknown_service_of_the_new_system_refuses(self):
        with pytest.raises(ResolveError) as exc:
            self.cat.resolve(
                params(
                    system="train-ticket",
                    ticket="INC-2003",
                    exclude_services=["carts"],
                )
            )
        assert exc.value.kind == "refuse"

    def test_relative_dates_use_the_r12_clock(self):
        # "yesterday" on the R12 desk is 2024-12-10, the day of INC-2004
        assert (
            DESK_NOW_R12.date() - dt.timedelta(days=1)
        ).isoformat() == inc3.YESTERDAY
        assert (
            self.ticket(system="train-ticket", at=f"{inc3.YESTERDAY}T12:03")
            == "INC-2004"
        )

    def test_incidents_after_the_clock_do_not_exist(self):
        assert DESK_NOW_R12 > DESK_NOW
        last = max(e.inject_time for e in self.cat.entries)
        assert last < DESK_NOW_R12.timestamp()

    def test_r11_catalogue_behaviour_is_unchanged(self):
        # the default clock, the two systems and the ticket numbering of R11 stay as they were
        old = IncidentCatalogue(
            [e for e in self.cat.entries if e.code in ("SS", "OB")], now=DESK_NOW
        )
        assert old.now == DESK_NOW
        assert old.resolve(params(ticket="INC-1001")).code == "SS"

    def test_real_catalogue_has_the_re3_tickets(self, real_catalogue):
        assert len(real_catalogue) == 182
        assert real_catalogue.now == DESK_NOW_R12
        assert real_catalogue.by_ticket["INC-1001"].code == "OB"
        assert real_catalogue.by_ticket["INC-2001"].code == "RE3-SS"
        assert real_catalogue.by_ticket["INC-2062"].code == "RE3-TT"
        assert len(real_catalogue.of_system("train-ticket")) == 21
        assert (
            max(e.inject_time for e in real_catalogue.entries)
            < DESK_NOW_R12.timestamp()
        )
        # RE3 was recorded after the R11 clock, in November and December 2024
        re3 = [e for e in real_catalogue.entries if e.code.startswith("RE3")]
        assert all(e.date_utc >= "2024-11-18" for e in re3)


# ------------------------------------------------------------------ spec and prompts


class TestSpecV3:
    def test_is_version_3_and_adds_only_the_system(self):
        d3, d2 = inc3.investigate_definition_v3(), incb.investigate_definition_v2()
        assert d3.version == 3 and d2.version == 2
        p3, p2 = d3.params_schema, d2.params_schema
        assert p3["properties"]["system"]["enum"] == [
            *p2["properties"]["system"]["enum"],
            "train-ticket",
        ]
        for key in p2["properties"]:
            if key != "system":
                assert p3["properties"][key] == p2["properties"][key]
        assert p3["required"] == p2["required"]

    def test_v1_and_v2_stay_as_they_were(self):
        assert inc.investigate_definition().params_schema["properties"]["system"][
            "enum"
        ] == [
            "sock-shop",
            "online-boutique",
        ]
        assert incb.investigate_definition_v2().version == 2

    def test_registry_accepts_train_ticket_only_in_v3(self):
        reg3 = inc3.build_registry_r12()
        reg3.make(inc.INVESTIGATE, {**params(system="train-ticket", ticket="INC-2050")})
        with pytest.raises(SpecValidationError):
            incb.build_registry_r11b().make(
                inc.INVESTIGATE, params(system="train-ticket", ticket="INC-2050")
            )


class TestPromptsV3:
    def test_the_only_system_prompt_differences_are_the_clock_and_the_generated_system_line(
        self,
    ):
        diff = inc3.prompt_diff()
        assert diff["same_line_count"]
        assert len(diff["system_lines_changed"]) == 2
        clock, systems = diff["system_lines_changed"]
        assert inc3.DESK_NOW_TEXT in clock[1] and inc.DESK_NOW in clock[0]
        assert clock[0].replace(inc.DESK_NOW, inc3.DESK_NOW_TEXT) == clock[1]
        assert systems[0].startswith("- system:") and "train-ticket" in systems[1]
        assert systems[1] == systems[0].replace(inc3._V2_CHOICES, inc3.system_choices())

    def test_one_few_shot_changes_its_date_and_nothing_else(self):
        diff = inc3.prompt_diff()
        assert len(diff["examples_changed"]) == 1
        text, old, new = diff["examples_changed"][0]
        assert "Yesterday" in text and old["at"] == "2024-02-21T06:00"
        assert new == {**old, "at": "2024-12-10T06:00"}
        assert diff["n_examples"] == len(incb.INCIDENT_POLICY_EXAMPLES_V2) == 19

    def test_summary_report_and_router_prompts_are_the_v2_ones(self):
        assert inc3.INCIDENT_SUMMARY_PACK_V3 is incb.INCIDENT_SUMMARY_PACK_V2
        assert inc3.INCIDENT_REPORT_PACK_V3.system.endswith(
            incb.INCIDENT_REPORT_NOTE_V2
        )
        assert inc3.INCIDENT_REPORT_EXAMPLES_V3 is incb.INCIDENT_REPORT_EXAMPLES_V2

    def test_shots_validate_against_the_v3_registry(self):
        reg = inc3.build_registry_r12()
        for ex in inc3.INCIDENT_POLICY_EXAMPLES_V3 + inc3.INCIDENT_REPORT_EXAMPLES_V3:
            if ex.params is not None:
                full = {**inc.INCIDENT_DEFAULTS, **ex.params}
                if "system" in full and ex.params.get("top_k", 3) <= 5:
                    reg.make(inc.INVESTIGATE, full)

    def test_pack_is_version_3(self):
        assert inc3.INCIDENT_PACK_V3.version == 3 and inc3.SPEC_VERSION == 3


# ------------------------------------------------------------------ keyword baseline


class TestKeywordV3:
    @pytest.mark.parametrize(
        ("text", "want"),
        [
            ("train ticket 2024-12-08 14:33", {"system": "train-ticket", "at": "2024-12-08T14:33"}),
            ("Train Ticket INC-2045, disregard ts-ui-dashboard.", {"system": "train-ticket", "ticket": "INC-2045", "exclude_services": ["ts-ui-dashboard"]}),
            ("the shop on the 23rd at 21:35", {"system": "sock-shop", "at": "---23T21:35"}),
            ("boutique, November 23rd, 15:13", {"system": "online-boutique", "at": "--11-23T15:13"}),
            ("Train ticket, Dec 8 at 11:36", {"system": "train-ticket", "at": "--12-08T11:36"}),
            ("train ticket yesterday 13:14", {"system": "train-ticket", "at": "2024-12-10T13:14"}),
            ("sock shop on 23 November 2024 at 7:51 pm", {"system": "sock-shop", "at": "2024-11-23T19:51"}),
        ],
    )  # fmt: skip
    def test_reads_the_new_system_months_and_clock(self, text, want):
        got = inc3.keyword_incident_v3(text)
        assert got is not None
        for key, value in want.items():
            assert got[key] == value, (text, key)
        assert "ticket" not in got or got["ticket"].startswith("INC-2")

    def test_train_ticket_is_not_a_ticket_number(self):
        got = inc3.keyword_incident_v3("train ticket 2024-12-08T09:15")
        assert "ticket" not in got

    def test_out_of_domain_is_still_refused(self):
        assert inc3.keyword_incident_v3("What is a good recipe for pancakes?") is None

    def test_v2_parser_is_untouched(self):
        assert (
            incb.keyword_incident_v2("sock shop 20 jan 14:21")["at"] == "--01-20T14:21"
        )

    def test_every_listed_service_is_a_real_name(self):
        assert (
            len(inc3.TRAIN_TICKET_SERVICES)
            == len(set(inc3.TRAIN_TICKET_SERVICES))
            == 68
        )


# ------------------------------------------------------------------ the desk


@pytest.fixture(scope="module")
def desk_v3():
    cat = small_catalogue()
    registry = inc3.build_registry_r12()
    domains = build_domains(None)
    domains["incident"] = inc3.incident_domain_v3(lambda p: {})
    return pipe.Desk(
        pipe.KeywordBackend(inc3.keyword_incident_v3), registry, domains, cat
    )


class TestDeskV3:
    def test_resolves_a_train_ticket_request(self, desk_v3):
        res = desk_v3.handle("Train ticket on the 10th at 12:05, what failed?")
        assert res.status == "ok" and res.spec.version == 3
        assert res.spec.params["system"] == "train-ticket"
        assert res.entry.ticket == "INC-2004"

    def test_a_busy_day_clarifies(self, desk_v3):
        res = desk_v3.handle("sock shop incident on 2024-11-22")
        assert res.status == "clarify"

    def test_uses_the_v3_pack(self, desk_v3):
        assert desk_v3.domains["incident"].pack is inc3.INCIDENT_PACK_V3


# ------------------------------------------------------------------ the sealed set


@pytest.fixture(scope="module")
def sealed_items():
    return sealed.load_sealed()


@pytest.fixture(scope="module")
def real_catalogue():
    if not rcaeval.list_incidents(None, ("SS", "OB")) or not rcaeval.list_incidents(
        None, rcaeval.RE3_SYSTEMS
    ):
        pytest.skip("RCAEval RE2 and RE3 cache not present")
    return inc3.load_catalogue_r12()


def _texts(items):
    return [i.text for i in items]


def _n_words(text: str) -> int:
    return sum(t.isalpha() for t in frame_tokens(text))


def _jaccard(a: str, b: str) -> float:
    ga, gb = frame_ngrams(a, 4), frame_ngrams(b, 4)
    return len(ga & gb) / len(ga | gb)


class TestSealedR12:
    def test_file_hash_is_pinned(self):
        assert sealed.file_sha256(sealed.SEALED_PATH) == sealed.SEALED_SHA256

    def test_older_sealed_hashes_still_pass(self):
        assert sealed1.file_sha256() == sealed1.SEALED_SHA256
        assert sealed2.file_sha256(sealed2.SEALED_PATH) == sealed2.SEALED_SHA256

    def test_load_refuses_a_changed_file(self, tmp_path):
        bad = tmp_path / "x.jsonl"
        bad.write_bytes(sealed.SEALED_PATH.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="sealed"):
            sealed.load_sealed(path=bad)

    def test_size_and_mix(self, sealed_items):
        assert len(sealed_items) == 107
        kinds = Counter(i.kind for i in sealed_items)
        assert kinds == {"spec": 91, "clarify": 6, "refuse": 10}
        assert len({i.id for i in sealed_items}) == len(sealed_items)
        assert len({i.text for i in sealed_items}) == len(sealed_items)
        cats = Counter(i.category for i in sealed_items)
        assert set(cats) >= {
            "ticket", "time", "partial", "relative", "topk", "exclude", "fault", "combo",
            "clarify", "refuse",
        }  # fmt: skip
        assert len({i.style for i in sealed_items}) >= 8

    def test_groups_by_system(self, sealed_items):
        target = sealed.load_resolves_to()
        groups = Counter(sealed.group_of(target[i.id]) for i in sealed_items)
        assert groups == {"re3": 66, "re2": 25, "none": 16}
        tt = [i for i in sealed_items if i.gold and i.gold["system"] == "train-ticket"]
        assert len(tt) >= 20
        # the RE3 relative references all name the clock's "yesterday"
        rel = [i for i in sealed_items if i.category == "relative"]
        assert rel and all(i.gold["at"].startswith(inc3.YESTERDAY) for i in rel)

    def test_gold_specs_validate_and_partial_gold_stays_partial(self, sealed_items):
        registry = inc3.build_registry_r12()
        partial = 0
        for it in sealed_items:
            if it.kind != "spec":
                assert it.gold is None
                continue
            registry.make(inc.INVESTIGATE, it.gold)
            partial += it.gold.get("at", "").startswith("--")
        assert partial >= 25

    def test_gold_resolves_to_exactly_the_documented_incident(
        self, sealed_items, real_catalogue
    ):
        target = sealed.load_resolves_to()
        for it in sealed_items:
            if it.kind == "spec":
                entry = real_catalogue.resolve(it.gold)
                assert entry.ticket == target[it.id], it.id
                assert entry.system == it.gold["system"]
            else:
                assert target[it.id] is None

    def test_clarify_and_refuse_items_are_decided_by_the_desk_policy(
        self, real_catalogue
    ):
        def err(**kw):
            with pytest.raises(ResolveError) as exc:
                real_catalogue.resolve(params(**kw))
            return exc.value.kind

        assert err(system="online-boutique", at="--11-22") == "clarify"
        assert err(system="train-ticket", at="---08") == "clarify"
        assert err(system="sock-shop", at="2024-11-23") == "clarify"
        assert err(system="train-ticket", at="--12-10") == "clarify"
        assert err(system="train-ticket", ticket="INC-2099") == "refuse"
        assert err(system="online-boutique", ticket="INC-2042") == "refuse"
        assert err(system="sock-shop", at="2024-11-22T09:30") == "refuse"
        assert err(system="train-ticket", at="2024-12-09T12:00") == "refuse"
        assert err(system="online-boutique", at="--01-25T10:00") == "refuse"

    def test_dev_is_a_fresh_second_set(self, sealed_items):
        dev = sealed.load_dev()
        assert len(dev) == 41 and Counter(i.kind for i in dev)["clarify"] >= 3
        assert not set(_texts(dev)) & set(_texts(sealed_items))
        registry = inc3.build_registry_r12()
        for it in dev:
            if it.kind == "spec":
                registry.make(inc.INVESTIGATE, it.gold)

    def test_no_text_is_shared_with_the_other_sets(self, sealed_items):
        """No identical frame (numbers masked) and no near-duplicate (4-gram Jaccard < 0.5)."""
        dev = sealed.load_dev()
        r7a = _texts(
            (
                *r7a_sealed.load_sealed(),
                *r7a_sealed.load_dev(),
                *r7a_sealed.load_old_set(),
            )
        )
        others = {
            "v1 sealed": _texts(sealed1.load_sealed()),
            "v1 dev": _texts(sealed1.load_dev()),
            "v2 sealed": _texts(sealed2.load_sealed()),
            "v2 dev": _texts(sealed2.load_dev()),
            "r7a": r7a,
        }
        for name, cands, extra in (
            ("sealed", _texts(sealed_items), {"r12 dev": _texts(dev)}),
            ("dev", _texts(dev), {}),
        ):
            for ref_name, refs in {**others, **extra}.items():
                frames = {" ".join(frame_tokens(t)) for t in refs}
                for t in cands:
                    assert " ".join(frame_tokens(t)) not in frames, (name, ref_name, t)
                    worst = max(_jaccard(t, r) for r in refs)
                    assert worst < 0.5, (name, ref_name, t, worst)

    def test_few_shots_share_nothing_with_the_held_out_texts(self, sealed_items):
        refs = _texts(sealed_items) + _texts(sealed.load_dev())
        shots = [e.text for e in inc3.INCIDENT_POLICY_EXAMPLES_V3] + [
            e.text for e in inc3.INCIDENT_REPORT_EXAMPLES_V3
        ]
        index = OverlapIndex([r for r in refs if _n_words(r) >= 8], 4)
        frames = {" ".join(frame_tokens(r)) for r in refs}
        for text in shots:
            score, j = index.max_containment(text)
            assert score < 0.5, (text, index.references[j], score)
            assert " ".join(frame_tokens(text)) not in frames, text
        blob = inc3.INCIDENT_POLICY_SYSTEM_V3
        assert not any(t in blob for t in refs)


# ------------------------------------------------------------------ G9: the newcomer


@pytest.fixture(scope="module")
def tiny():
    """A tiny trained specialist, a team of six synthetic services and some incidents."""
    from genetic_coil_arena.envs.logtriage.synthetic import synthetic_incident
    from genetic_coil_arena.experiments import log_triage3 as t3
    from genetic_coil_arena.experiments import r11_team as team
    from genetic_coil_arena.experiments import r12_team as jt
    from genetic_coil_arena.learning.classifiers import MLPClassifier

    train = [synthetic_incident(100 + i) for i in range(12)]
    data = t3.metric_dataset(train)
    model = MLPClassifier(6, epochs=40, seed=0).fit(data.x, data.y)
    services = [f"svc{i}" for i in range(6)]
    base = team.IncidentTeam(t3.MetricFaultSpecialist(model), services)
    joined = jt.JoinedTeam(base, services)
    incs = [synthetic_incident(i, rep=2) for i in range(6)]
    return base, joined, incs, jt


class TestNewcomer:
    def test_message_matches_the_new_ext_signal_and_the_schema(self):
        from genetic_coil_arena.envs.logtriage import logsignal as lg
        from genetic_coil_arena.envs.logtriage import novelty as nv
        from genetic_coil_arena.envs.logtriage.synthetic import synthetic_incident

        for seed in range(5):
            incident = synthetic_incident(seed, rep=2)
            want = lg.new_template_scores(incident, "new_ext")
            for s in range(incident.n_services):
                msg = nv.novelty_message(incident, s)
                assert nv.novelty_errors(msg) == []
                assert msg["service"] == incident.services[s]
                assert msg["new_ext"] == want[s]
                assert len(msg["top"]) <= nv.MAX_TOP == 3
                assert all(len(t) <= nv.MAX_CHARS == 80 for t in msg["top"])

    def test_reads_only_its_own_service_and_survives_dropping_the_lines(self):
        from genetic_coil_arena.envs.logtriage import novelty as nv
        from genetic_coil_arena.envs.logtriage.synthetic import synthetic_incident
        from genetic_coil_arena.experiments import r12_team as jt

        incident = synthetic_incident(3, rep=2)
        before = [nv.novelty_message(incident, s) for s in range(incident.n_services)]
        assert max(m["new_ext"] for m in before) > 0
        jt.prepare(incident)  # primes the caches, then drops the line arrays
        assert incident.t.size == 0
        assert before == [
            nv.novelty_message(incident, s) for s in range(incident.n_services)
        ]

    def test_schema_rejects_extra_fields_and_long_reports(self):
        from genetic_coil_arena.envs.logtriage import novelty as nv

        ok = {"service": "a", "new_ext": 2, "top": ["x", "y"]}
        assert nv.novelty_errors(ok) == []
        assert nv.novelty_errors({**ok, "extra": 1})
        assert nv.novelty_errors({**ok, "top": ["a", "b", "c", "d"]})
        assert nv.novelty_errors({**ok, "top": ["z" * 81]})
        assert nv.novelty_errors({**ok, "new_ext": -1})
        assert nv.novelty_errors({"service": "a", "top": []})


class TestJoinedCentre:
    def msgs(self, z, n):
        met = [{"service": f"s{i}", "score": float(v)} for i, v in enumerate(z)]
        nov = [
            {"service": f"s{i}", "new_ext": int(v), "top": []} for i, v in enumerate(n)
        ]
        return met, nov

    def test_candidates_are_the_nine_listed_plus_the_old_rule(self):
        from genetic_coil_arena.envs.logtriage import novelty as nv

        assert list(nv.RULES) == [
            "rr_w0.5", "rr_w1", "rr_w2", "gate40", "gate45", "cap20", "cap50",
            "gate45_cap50", "gate40_cap50",
        ]  # fmt: skip
        assert len(nv.ALL_RULES) <= 10
        assert nv.OLD_RULE not in nv.RULES

    def test_the_old_rule_is_the_stable_metric_ranking(self):
        from genetic_coil_arena.envs.logtriage import novelty as nv
        from genetic_coil_arena.envs.logtriage.team import rule_ranking

        met, nov = self.msgs([5, 9, 9, 1], [7, 0, 0, 3])
        assert nv.joined_ranking("metric", met, nov)[0] == rule_ranking(met)

    def test_log_part_lifts_a_service_the_metrics_miss(self):
        from genetic_coil_arena.envs.logtriage import novelty as nv

        met, nov = self.msgs([12, 10, 8, 3], [0, 0, 5, 0])
        assert nv.joined_ranking("metric", met, nov)[0][0] == 0
        order, used = nv.joined_ranking("rr_w1", met, nov)
        assert order[0] == 2 and used  # 1/3 + 1 beats 1 + 0

    def test_gate_drops_the_log_part_when_a_metric_is_decisive(self):
        from genetic_coil_arena.envs.logtriage import novelty as nv

        met, nov = self.msgs([49, 10, 8, 3], [0, 0, 5, 0])
        for rule in ("gate40", "gate45", "gate45_cap50", "gate40_cap50"):
            order, used = nv.joined_ranking(rule, met, nov)
            assert order[0] == 0 and not used, rule
        order, used = nv.joined_ranking("rr_w1", met, nov)
        assert order[0] == 2 and used  # without a gate the log part still takes over

    def test_cap_treats_a_storm_as_silence(self):
        from genetic_coil_arena.envs.logtriage import novelty as nv

        met, nov = self.msgs([12, 10, 8, 3], [0, 0, 300, 0])
        assert nv.joined_ranking("cap50", met, nov) == ([0, 1, 2, 3], False)
        assert nv.joined_ranking("rr_w1", met, nov)[0][0] == 2

    def test_ties_go_to_the_higher_metric_score_then_the_index(self):
        from genetic_coil_arena.envs.logtriage import novelty as nv

        met, nov = self.msgs([5, 9, 9, 2], [1, 1, 1, 1])
        assert nv.joined_ranking("rr_w2", met, nov)[0] == [1, 2, 0, 3]

    def test_the_centre_sees_only_reports_never_the_dataset(self):
        import inspect

        from genetic_coil_arena.envs.logtriage import novelty as nv

        names = set(inspect.signature(nv.rule_decide_joined).parameters)
        assert names == {"services", "metric_msgs", "novelty_msgs", "rule"}


class TestJoinedTeam:
    def test_old_rule_through_the_joined_centre_equals_the_old_team(self, tiny):
        base, joined, incs, jt = tiny
        for incident in incs:
            old = base.triage(incident, trace_id=f"{incident.id}-o")
            new = joined.triage_rules(incident, ["metric"], trace_id=f"{incident.id}-n")
            assert new["metric"]["ranking"] == old["ranking"]
            assert new["metric"]["fault_type"] == old["fault_type"]

    def test_state_hash_is_frozen_by_joining_and_running(self, tiny):
        base, joined, incs, jt = tiny
        before = joined.frozen
        for incident in incs[:2]:
            joined.triage_rules(incident, ["rr_w1", "gate45_cap50"])
            base.triage(incident)
        assert jt.old_state_hash(base) == before
        joined.assert_frozen()

    def test_state_hash_catches_a_change_to_the_old_team(self, tiny):
        base, joined, incs, jt = tiny
        before = jt.old_state_hash(base)
        clf = base.model.model
        saved = clf.mean
        clf.mean = saved + 1.0
        try:
            assert jt.old_state_hash(base) != before
            with pytest.raises(RuntimeError, match="old team changed"):
                joined.assert_frozen()
        finally:
            clf.mean = saved
        assert jt.old_state_hash(base) == before

    def test_newcomers_join_through_the_command_centre(self, tiny):
        base, joined, incs, jt = tiny
        names = base.cc.registry.names()
        assert sum(n.startswith("novelty:") for n in names) == 6
        assert "rca-centre-joined" in names and "rca-centre" in names
        assert {"rca.log_novelty", "rca.decide_joined"} <= set(base.specs.names())
        assert {"rca.metric_fault_report", "rca.decide_trained"} <= set(
            base.specs.names()
        )
        tid = "trace-check"
        joined.triage_rules(incs[0], ["rr_w1"], trace_id=tid)
        # 6 metric + 6 novelty + 1 decision goals, each a goal and a result envelope
        assert len(base.trace.envelopes(tid)) >= 2 * 13

    def test_run_incident_record_has_the_label_only_in_scores(self, tiny):
        base, joined, incs, jt = tiny
        rec = jt.run_incident(base, joined, incs[0], ["metric", "gate45_cap50"])
        assert rec["dataset"] == "re2" and rec["split"] == "test"
        assert set(rec["rules"]) == {"metric", "gate45_cap50"}
        assert rec["rules"]["metric"]["top3"] == rec["old"]["top3"]
        assert set(rec["baselines"]) == {"baro", "new_ext", "re3_fusion"}
        assert rec["old"]["joint"] in (True, False)
        json.dumps(rec)

    def test_select_rule_takes_the_best_pooled_top1_and_the_earlier_on_ties(self, tiny):
        from genetic_coil_arena.envs.logtriage import novelty as nv

        _, _, _, jt = tiny

        def rec(dataset, hits):
            return {
                "dataset": dataset,
                "old": {"top1": False, "joint": False},
                "rules": {
                    r: {"top1": hits.get(r, False), "used_logs": True, "joint": False}
                    for r in nv.RULES
                },
            }

        recs = [rec("re2", {"cap50": True}), rec("re3", {"cap50": True, "rr_w1": True})]
        assert jt.select_rule(recs)["chosen"] == "cap50"  # 2 of 2 beats 1 of 2
        recs = [rec("re2", {"cap50": True, "gate40": True})]
        assert jt.select_rule(recs)["chosen"] == "gate40"  # the earlier of the tied
        assert jt.CHOSEN_RULE in nv.RULES


# ------------------------------------------------------------------ G10: the executor


class TestJoinedExecutor:
    def catalogue(self, services):
        entries = [
            CatalogueEntry("INC-1001", "synthetic/a/2", "SS", ts("2024-01-20T14:21"), tuple(services)),
            CatalogueEntry("INC-2001", "synthetic/b/2", "RE3-SS", ts("2024-11-22T05:42"), tuple(services)),
        ]  # fmt: skip
        return IncidentCatalogue(entries, now=DESK_NOW_R12)

    def test_score_is_joint_for_re2_and_root_top1_for_re3(self, tiny):
        from genetic_coil_arena.envs.logtriage.synthetic import synthetic_incident
        from genetic_coil_arena.experiments import r12_team as jt

        _, joined, _, _ = tiny
        a, b = synthetic_incident(40, rep=2), synthetic_incident(41, rep=2)
        loaded = {"INC-1001": jt.prepare(a), "INC-2001": jt.prepare(b)}
        ex = jt.JoinedExecutor(
            self.catalogue(a.services), joined, loader=lambda e: loaded[e.ticket]
        )
        for ticket in ("INC-1001", "INC-2001"):
            res = ex(params(system="sock-shop", ticket=ticket))
            incident = loaded[ticket]
            top1 = float(res["public"]["suspects"][0] == incident.root_service)
            assert res["top1"] == top1
            if ticket == "INC-2001":
                assert res["score"] == top1  # no joint accuracy for RE3
            else:
                joint = top1 and res["public"]["fault_type"] == incident.fault_type
                assert res["score"] == float(joint)
            # the public part never carries the label
            assert set(res["public"]) <= {
                "ticket", "system", "suspects", "onset_utc", "excluded_services",
                "fault_type", "fault_probability",
            }  # fmt: skip

    def test_exclusion_removes_a_candidate(self, tiny):
        from genetic_coil_arena.envs.logtriage.synthetic import synthetic_incident
        from genetic_coil_arena.experiments import r12_team as jt

        _, joined, _, _ = tiny
        a = jt.prepare(synthetic_incident(42, rep=2))
        ex = jt.JoinedExecutor(self.catalogue(a.services), joined, loader=lambda e: a)
        top = ex(params(ticket="INC-1001"))["public"]["suspects"][0]
        again = ex(params(ticket="INC-1001", exclude_services=[top]))
        assert top not in again["public"]["suspects"]


# ------------------------------------------------------------------ reports


def g9_records(n2=40, n3=40, old2=0.8, new2=0.8, old3=0.4, new3=0.7):
    """Paired test records with the given top-1 rates (deterministic interleaving)."""
    rows = []
    for ds, n, po, pn in (("re2", n2, old2, new2), ("re3", n3, old3, new3)):
        o = [(k + 0.5) / n < po for k in range(n)]
        w = [(k + 0.5) / n < pn for k in range(n)]
        for k in range(n):
            rows.append(
                {
                    "id": f"{ds}{k}", "system": "SS" if ds == "re2" else "RE3-SS",
                    "dataset": ds, "split": "test", "rep": 2, "root": "a", "fault": "cpu",
                    "old": {"top3": ["a"], "top1": o[k], "in_top3": True, "joint": o[k] if ds == "re2" else None},
                    "rules": {"gate45_cap50": {"top3": ["a"], "top1": w[k], "in_top3": True, "joint": w[k] if ds == "re2" else None, "used_logs": k % 2 == 0, "changed_top1": o[k] != w[k]}},
                    "baselines": {b: {"top3": ["a"], "top1": k % 3 == 0, "in_top3": True} for b in ("baro", "new_ext", "re3_fusion")},
                }
            )  # fmt: skip
    return rows


class TestReportR12:
    def test_pass_when_re3_gains_and_re2_holds(self):
        from genetic_coil_arena.experiments import r12_report as r12

        g9 = r12.g9_summary(g9_records(), "gate45_cap50")
        assert g9["re3"]["diff_top1"]["mean"] == pytest.approx(0.30)
        assert g9["verdict"]["verdict"] == "PASS"

    def test_re3_gain_must_have_a_ci_above_zero(self):
        from genetic_coil_arena.experiments import r12_report as r12

        small = r12.g9_summary(g9_records(old3=0.40, new3=0.45), "gate45_cap50")
        assert small["verdict"]["verdict"] == "KILL"  # the CI includes 0
        assert any("RE3" in k for k in small["verdict"]["kill_reasons"])

    def test_a_clear_re2_loss_kills(self):
        from genetic_coil_arena.experiments import r12_report as r12

        bad = r12.g9_summary(g9_records(n2=200, old2=0.9, new2=0.6), "gate45_cap50")
        assert bad["verdict"]["verdict"] == "KILL"
        assert any("RE2 top-1" in k for k in bad["verdict"]["kill_reasons"])
        rows = {r["criterion"]: r["result"] for r in bad["verdict"]["criteria"]}
        assert rows["RE3 top-1, joined - old"] == "PASS"

    def test_gate_rates_and_fixed_broken_counts(self):
        from genetic_coil_arena.experiments import r12_report as r12

        d = r12.g9_summary(g9_records(), "gate45_cap50")["re3"]
        assert d["logs_used"] == pytest.approx(0.5)
        assert d["fixed"] == 12 and d["broke"] == 0
        assert set(d["baselines"]) == {"baro", "new_ext", "re3_fusion"}

    def test_g9_render_names_the_dev_table_and_the_hash(self):
        from genetic_coil_arena.envs.logtriage import novelty as nv
        from genetic_coil_arena.experiments import r12_report as r12

        row = {"top1": 0.8, "re2_top1": 0.8, "re3_top1": 0.8, "re2_joint": 0.5}
        dev = {
            "chosen": "gate45_cap50",
            "reference": {"rule": "metric", **row},
            "table": [{"rule": r, **row, "used_logs": 0.4} for r in nv.RULES],
        }
        g9 = {
            "rule": "gate45_cap50", "hash_before": "h", "hash_after": "h", "dev": dev,
            "test": r12.g9_summary(g9_records(), "gate45_cap50"),
        }  # fmt: skip
        md = r12.render({"generated": "now", "g9": g9})
        assert "gate45_cap50` (chosen)" in md and "identical" in md
        assert "<!-- analysis:start -->" in md and "Analysis pending" in md

    def test_g10_groups_verdict_and_render(self):
        from genetic_coil_arena.experiments import r11_report as rep
        from genetic_coil_arena.experiments import r12_report as r12

        from .test_r11 import TestReport

        t = TestReport()
        recs = t.records()
        resolves = {
            r["id"]: ("INC-2001" if k < 6 else "INC-1001") for k, r in enumerate(recs)
        }
        resolves["c"] = None
        m = r12.g10_metrics(recs, t.routes(), resolves)
        assert m["group"]["re3"]["exact"]["n"] == 6
        assert m["group"]["re2"]["exact"]["n"] == 4
        assert m["all"]["routing"]["n"] == 13
        v = r12.g10_verdict(m, 9.0)
        assert v["verdict"] in ("PASS", "NOT PASSED", "KILL")
        names = [r["criterion"] for r in v["criteria"]]
        assert any("RE3" in n for n in names) and any("RE2" in n for n in names)
        report = {
            "generated": "now",
            "g10": {
                "n_sealed": 11, "sealed_sha256": "x", "r7a_sealed_sha256": "y",
                "catalogue": {"n": 182, "fingerprint": "f"}, "prompt_audit": "a", "team": "t",
                "systems": {"s": {"label": "stub", "metrics": m, "verdict": v, "failures": rep.failures(recs)}},
            },
        }  # fmt: skip
        md = r12.render(report)
        assert "## G10" in md and "Exact RE3" in md and "failure examples" in md
        json.dumps(m)  # JSON-safe


# ------------------------------------------------------------------ real data (skipped without it)


class TestRealData:
    def test_joined_team_through_the_grown_desk(self, real_catalogue):
        from genetic_coil_arena.experiments import r11_team as team
        from genetic_coil_arena.experiments import r12_team as jt

        state = team.train_metric_specialist(limit_train=40)
        services = (
            set(inc.SOCK_SHOP_SERVICES)
            | set(inc.ONLINE_BOUTIQUE_SERVICES)
            | set(inc3.TRAIN_TICKET_SERVICES)
        )
        base = team.IncidentTeam(team.load_specialist(state), services)
        joined = jt.JoinedTeam(base, services, jt.CHOSEN_RULE)
        executor = jt.JoinedExecutor(real_catalogue, joined)
        registry = inc3.build_registry_r12()
        domains = build_domains(None)
        domains["incident"] = inc3.incident_domain_v3(executor)
        desk = pipe.Desk(
            pipe.KeywordBackend(inc3.keyword_incident_v3),
            registry,
            domains,
            real_catalogue,
        )
        seen = set()
        for item in sealed.load_dev():
            if item.kind != "spec" or item.gold["system"] in seen:
                continue
            seen.add(item.gold["system"])
            rec = pipe.process_incident_item(desk, item).record
            assert rec["status"] == "ok" and rec["nl_state"] == "completed", item.text
            assert rec["gold_state"] == "completed"
        assert seen == set(inc3.SYSTEMS_V3)
        joined.assert_frozen()
