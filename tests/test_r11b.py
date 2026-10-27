"""R11 v2 (goal G8b): partial dates, the v2 spec, prompts, sealed set and the desk.

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
    CatalogueEntry,
    IncidentCatalogue,
    ResolveError,
    parse_ref,
)
from genetic_coil_arena.experiments import r7a_sealed
from genetic_coil_arena.experiments import r11_incident as inc
from genetic_coil_arena.experiments import r11_pipeline as pipe
from genetic_coil_arena.experiments import r11_report as rep
from genetic_coil_arena.experiments import r11_sealed as sealed1
from genetic_coil_arena.experiments import r11_team as team
from genetic_coil_arena.experiments import r11b_incident as incb
from genetic_coil_arena.experiments import r11b_sealed as sealed
from genetic_coil_arena.experiments.r7a_domains import build_domains
from genetic_coil_arena.protocol.goalspec import SpecValidationError
from genetic_coil_arena.translator import OverlapIndex, TraceView, fact_fidelity
from genetic_coil_arena.translator.distill import frame_ngrams, frame_tokens

from .test_r11 import params, synthetic_catalogue

UTC = dt.timezone.utc


def ts(text: str) -> float:
    return dt.datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp()


def catalogue(*rows: tuple[str, str], now=DESK_NOW) -> IncidentCatalogue:
    """``(system code, 'YYYY-MM-DDTHH:MM')`` rows to a catalogue (tickets INC-1001...)."""
    entries = []
    for k, (code, when) in enumerate(rows):
        services = (
            inc.SOCK_SHOP_SERVICES if code == "SS" else inc.ONLINE_BOUTIQUE_SERVICES
        )
        entries.append(
            CatalogueEntry(
                f"INC-{1001 + k}", f"{code}/x{k}/2", code, ts(when), services
            )
        )
    return IncidentCatalogue(entries, now=now)


# ------------------------------------------------------------------ partial dates


class TestPartialDates:
    cat = catalogue(
        ("SS", "2024-01-17T14:00"),  # 1001
        ("SS", "2024-01-17T15:00"),  # 1002
        ("SS", "2024-01-18T14:00"),  # 1003
        ("SS", "2024-02-17T14:00"),  # 1004 (same day of month, another month)
        ("OB", "2024-01-17T14:01"),  # 1005
        ("SS", "2024-01-21T00:03"),  # 1006 just after midnight
        ("SS", "2024-02-23T10:00"),  # 1007 after the desk clock
    )

    def ticket(self, **kw):
        return self.cat.resolve(params(**kw)).ticket

    def error(self, **kw):
        with pytest.raises(ResolveError) as exc:
            self.cat.resolve(params(**kw))
        return exc.value

    def test_parse_ref_forms(self):
        assert parse_ref("---17").day == 17 and parse_ref("---17").month is None
        full = parse_ref("--01-19T21:30")
        assert (full.month, full.day, full.hhmm) == (1, 19, (21, 30)) and full.partial
        assert not parse_ref("2024-01-19").partial
        for bad in ("--13-01", "---32", "--01-17T25:00", "17th", "--1-19", "---7"):
            with pytest.raises(ValueError):
                parse_ref(bad)

    def test_day_of_month_with_time_names_the_one_matching_date(self):
        assert self.ticket(at="---18T14:00") == "INC-1003"
        assert self.ticket(at="---17T15:05") == "INC-1002"  # within 10 minutes
        assert self.ticket(system="online-boutique", at="---17T14:00") == "INC-1005"

    def test_month_and_day_pick_the_month(self):
        assert self.ticket(at="--01-17T14:00") == "INC-1001"
        assert self.ticket(at="--02-17T14:00") == "INC-1004"

    def test_two_matching_dates_ask(self):
        err = self.error(at="---17T14:00")
        assert err.kind == "clarify" and "2024-01-17" in err.message
        assert "2024-02-17" in err.message

    def test_date_without_time_needs_exactly_one_incident(self):
        assert self.ticket(at="---18") == "INC-1003"
        err = self.error(at="--01-17")
        assert err.kind == "clarify" and "INC-1001" in err.message
        assert self.error(at="---17").kind == "clarify"  # two dates and two incidents

    def test_no_match_refuses(self):
        for at in ("---19", "--03-17T14:00", "---17T09:00", "--02-18"):
            assert self.error(at=at).kind == "refuse", at

    def test_incidents_after_the_desk_clock_do_not_exist_yet(self):
        assert self.error(at="---23").kind == "refuse"
        later = catalogue(
            ("SS", "2024-02-23T10:00"), now=dt.datetime(2024, 2, 24, tzinfo=UTC)
        )
        assert later.resolve(params(at="---23")).ticket == "INC-1001"

    def test_a_time_just_across_midnight_matches(self):
        assert self.ticket(at="---20T23:57") == "INC-1006"

    def test_ticket_with_a_partial_date_must_agree(self):
        assert self.ticket(ticket="INC-1003", at="---18T14:00") == "INC-1003"
        assert self.ticket(ticket="INC-1003", at="--01-18") == "INC-1003"
        err = self.error(ticket="INC-1003", at="---17")
        assert err.kind == "refuse" and "INC-1003" in err.message

    def test_impossible_partial_dates_refuse(self):
        for at in ("--13-01", "---32", "--01-17T25:00"):
            assert self.error(at=at).kind == "refuse", at

    def test_full_dates_behave_as_in_v1(self):
        assert self.ticket(at="2024-01-18T14:03") == "INC-1003"
        assert self.error(at="2024-01-19").kind == "refuse"

    def test_the_policy_is_documented_in_the_module(self):
        from genetic_coil_arena.envs.logtriage import catalogue as mod

        doc = mod.__doc__
        assert (
            "---DD" in doc
            and "--MM-DD" in doc
            and "clarify" in doc
            and "DESK_NOW" in doc
        )
        assert DESK_NOW == dt.datetime(2024, 2, 22, 9, 0, tzinfo=UTC)
        assert DESK_NOW.strftime("%A") == "Thursday"


# ------------------------------------------------------------------ the v2 spec


class TestSpecV2:
    registry = incb.build_registry_r11b()

    def make(self, **kw):
        return self.registry.make(inc.INVESTIGATE, params(**kw))

    def test_is_version_2_and_keeps_six_params(self):
        d = self.registry.get(inc.INVESTIGATE)
        assert d.version == 2 and len(d.params_schema["properties"]) == 6
        assert self.registry.versions(inc.INVESTIGATE) == [2]

    @pytest.mark.parametrize(
        "at",
        [
            "---17",
            "---17T10:20",
            "--01-19",
            "--01-19T21:30",
            "2024-01-19",
            "2024-01-19T18:47",
        ],
    )
    def test_accepts_the_three_forms(self, at):
        assert self.make(at=at).params["at"] == at

    @pytest.mark.parametrize(
        "at", ["--1-19", "---7", "17th", "01-19", "-01-19", "---17T10", "--01-19 10:20"]
    )
    def test_rejects_other_text(self, at):
        with pytest.raises(SpecValidationError):
            self.make(at=at)

    def test_v1_stays_as_it_was(self):
        v1 = inc.build_registry_r11()
        assert v1.versions(inc.INVESTIGATE) == [1]
        with pytest.raises(SpecValidationError):
            v1.make(inc.INVESTIGATE, params(at="---17"))

    def test_top_k_stays_bounded(self):
        for k in (0, 6, 9):
            with pytest.raises(SpecValidationError):
                self.make(top_k=k)


# ------------------------------------------------------------------ prompts


class TestPrompts:
    def test_policy_prompt_transcribes_instead_of_completing(self):
        text = incb.INCIDENT_POLICY_SYSTEM_V2
        assert "Never complete it" in text and '"---17"' in text and '"--01-19"' in text
        assert "never lower or raise it" in text  # no clamping of top_k
        assert "never a refusal" in text  # vague requests ask
        assert 'Work out "yesterday", "two days ago" or "last Friday"' not in text
        assert incb.INCIDENT_PACK_V2.version == 2
        assert incb.INCIDENT_REPORT_PACK_V2.version == 2
        assert inc.INCIDENT_PACK.version is None  # v1 untouched

    def test_shots_validate_against_the_v2_registry(self):
        registry = incb.build_registry_r11b()
        for ex in incb.INCIDENT_POLICY_EXAMPLES_V2 + incb.INCIDENT_REPORT_EXAMPLES_V2:
            if ex.params is None:
                continue
            filled = {**inc.INCIDENT_DEFAULTS, **ex.params}
            if "system" not in filled:
                continue  # the checker asks for the system
            top_k = filled["top_k"]
            if not 1 <= top_k <= 5:
                continue  # taught to pass on so that the checker refuses it
            registry.make(inc.INVESTIGATE, filled)

    def test_summary_examples_state_top_k_and_how_the_incident_was_named(self):
        assert "top <goal.top_k> suspects requested" in incb.INCIDENT_SUMMARY_SYSTEM_V2
        for trace, summary in incb.INCIDENT_SUMMARY_EXAMPLES_V2:
            goal = json.loads(trace)["goal"]["params"]
            assert f"top {goal['top_k']} suspects requested" in summary
            if "ticket" in goal:
                assert f"requested by ticket {goal['ticket']}" in summary
            if "at" in goal:
                assert f"requested by time {goal['at']}" in summary

    def test_partial_time_is_a_fact_a_summary_must_keep_verbatim(self):
        p = {**inc.INCIDENT_DEFAULTS, "system": "sock-shop", "at": "---20T14:48"}
        public = {
            "ticket": "INC-1095", "system": "sock-shop", "suspects": ["payment", "user", "orders"],
            "onset_utc": "2024-01-20T14:40", "excluded_services": [],
            "fault_type": "cpu", "fault_probability": 0.5,
        }  # fmt: skip
        goal = incb.build_registry_r11b().make(inc.INVESTIGATE, p)
        view = TraceView(goal, (), {"state": "completed", **public})
        good = (
            "Investigation of the sock-shop incident, requested by time ---20T14:48 (ticket INC-1095), "
            "top 3 suspects requested. The suspected services, most likely first, are payment, user, orders; "
            "fault type cpu (probability 0.5); onset 14:40 UTC; no services excluded."
        )
        assert fact_fidelity(good, inc.incident_facts(view))["facts"]["at"]
        bad = good.replace("---20T14:48", "2024-02-20T14:48")
        assert not fact_fidelity(bad, inc.incident_facts(view))["facts"]["at"]


# ------------------------------------------------------------------ keyword baseline


class TestKeywordV2:
    @pytest.mark.parametrize(
        "text,at",
        [
            ("sock shop on the 17th at 10:20", "---17T10:20"),
            ("boutique incident Jan 19 at 18:47", "--01-19T18:47"),
            ("boutique 19 January 2024 at 18:47", "2024-01-19T18:47"),
            ("boutique 2024-01-19 18:47", "2024-01-19T18:47"),
            ("the boutique yesterday at 04:19", "2024-02-21T04:19"),
            ("boutique two days ago", "2024-02-20"),
            ("sock shop 20 Jan 11:50 pm", "--01-20T23:50"),
            ("boutique on 20 February", "--02-20"),
        ],
    )
    def test_transcribes_without_completing(self, text, at):
        assert incb.keyword_incident_v2(text)["at"] == at

    def test_ticket_numbers_are_not_dates(self):
        p = incb.keyword_incident_v2("Investigate INC-1042 on the boutique")
        assert p["ticket"] == "INC-1042" and "at" not in p

    def test_out_of_domain_is_still_refused(self):
        assert incb.keyword_incident_v2("Tell me a joke") is None


# ------------------------------------------------------------------ the desk (no model)


@pytest.fixture(scope="module")
def desk_v2():
    cat = synthetic_catalogue()
    registry = incb.build_registry_r11b()
    domains = build_domains(None)
    domains["incident"] = incb.incident_domain_v2(lambda p: {})
    backend = pipe.KeywordBackend(incb.keyword_incident_v2)
    return pipe.Desk(backend, registry, domains, cat)


class TestDeskV2:
    def test_a_day_without_a_month_is_resolved_by_the_desk(self, desk_v2):
        res = desk_v2.handle("Sock shop incident on the 18th at 14:00")
        assert res.status == "ok" and res.spec.params["at"] == "---18T14:00"
        assert res.spec.version == 2 and res.entry.ticket == "INC-1004"

    def test_several_incidents_ask(self, desk_v2):
        res = desk_v2.handle("Investigate the sock shop on the 17th")
        assert res.status == "clarify" and "INC-1001" in res.reason

    def test_no_incident_refuses(self, desk_v2):
        res = desk_v2.handle("Investigate the sock shop on the 19th at 10:00")
        assert res.status == "refused"

    def test_uses_the_v2_pack(self, desk_v2):
        assert desk_v2.domains["incident"].pack is incb.INCIDENT_PACK_V2
        assert desk_v2.domains["incident"].summary_pack is incb.INCIDENT_SUMMARY_PACK_V2


# ------------------------------------------------------------------ the sealed set


@pytest.fixture(scope="module")
def sealed_items():
    return sealed.load_sealed()


@pytest.fixture(scope="module")
def real_catalogue():
    if not rcaeval.list_incidents(None, ("SS", "OB")):
        pytest.skip("RCAEval cache not present")
    return team.load_catalogue()


def _texts(items):
    return [i.text for i in items]


def _n_words(text: str) -> int:
    return sum(t.isalpha() for t in frame_tokens(text))


def _jaccard(a: str, b: str) -> float:
    ga, gb = frame_ngrams(a, 4), frame_ngrams(b, 4)
    return len(ga & gb) / len(ga | gb)


class TestSealedV2:
    def test_file_hash_is_pinned(self):
        assert sealed.file_sha256(sealed.SEALED_PATH) == sealed.SEALED_SHA256

    def test_v1_sealed_hash_still_passes(self):
        assert sealed1.file_sha256() == sealed1.SEALED_SHA256
        assert len(sealed1.load_sealed()) == 130

    def test_load_refuses_a_changed_file(self, tmp_path):
        bad = tmp_path / "x.jsonl"
        bad.write_bytes(sealed.SEALED_PATH.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="sealed"):
            sealed.load_sealed(path=bad)

    def test_size_and_mix(self, sealed_items):
        assert len(sealed_items) >= 130
        kinds = Counter(i.kind for i in sealed_items)
        assert kinds["spec"] >= 100
        assert 0.12 <= (kinds["clarify"] + kinds["refuse"]) / len(sealed_items) <= 0.20
        assert len({i.id for i in sealed_items}) == len(sealed_items)
        assert len({i.text for i in sealed_items}) == len(sealed_items)
        cats = Counter(i.category for i in sealed_items)
        assert set(cats) >= {
            "ticket", "time", "partial", "relative", "topk", "exclude", "fault", "combo",
            "clarify", "refuse",
        }  # fmt: skip
        assert cats["partial"] >= 20
        assert len({i.style for i in sealed_items}) >= 7

    def test_gold_specs_validate_and_partial_gold_stays_partial(self, sealed_items):
        registry = incb.build_registry_r11b()
        partial = 0
        for it in sealed_items:
            if it.kind != "spec":
                assert it.gold is None
                continue
            registry.make(inc.INVESTIGATE, it.gold)
            at = it.gold.get("at", "")
            partial += at.startswith("--")
        assert partial >= 40

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

        assert err(system="online-boutique", at="---20") == "clarify"
        assert err(system="online-boutique", at="--02-21") == "clarify"
        assert err(system="sock-shop", at="---17T10:20") == "refuse"
        assert err(system="sock-shop", at="2024-02-21T04:19") == "refuse"
        assert err(system="online-boutique", at="--01-19T12:00") == "refuse"
        assert err(system="sock-shop", ticket="INC-1020") == "refuse"
        assert err(system="online-boutique", ticket="INC-1150") == "refuse"

    def test_dev_is_a_fresh_second_set(self, sealed_items):
        dev = sealed.load_dev()
        assert len(dev) >= 40 and Counter(i.kind for i in dev)["clarify"] >= 3
        assert not set(_texts(dev)) & set(_texts(sealed_items))
        registry = incb.build_registry_r11b()
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
            "r7a": r7a,
        }
        for name, cands, extra in (
            ("sealed", _texts(sealed_items), {"v2 dev": _texts(dev)}),
            ("dev", _texts(dev), {}),
        ):
            for ref_name, refs in {**others, **extra}.items():
                frames = {" ".join(frame_tokens(t)) for t in refs}
                for t in cands:
                    assert " ".join(frame_tokens(t)) not in frames, (name, ref_name, t)
                    worst = max(_jaccard(t, r) for r in refs)
                    assert worst < 0.5, (name, ref_name, t, worst)

    def test_few_shots_share_nothing_with_held_out_texts(self, sealed_items):
        # (the v2 dev set is what the prompts were tuned on, so it may resemble the shots)
        refs = (
            _texts(sealed_items)
            + _texts(sealed1.load_dev())
            + _texts(sealed1.load_sealed())
        )
        refs7 = _texts(
            (
                *r7a_sealed.load_sealed(),
                *r7a_sealed.load_dev(),
                *r7a_sealed.load_old_set(),
            )
        )
        shots = (
            [e.text for e in incb.INCIDENT_POLICY_EXAMPLES_V2]
            + [e.text for e in incb.INCIDENT_REPORT_EXAMPLES_V2]
            + [t for t, _ in incb.INCIDENT_SUMMARY_EXAMPLES_V2]
        )
        shots = [s for s in shots if not s.startswith("{")]
        # containment (the v1 standard) against full-sentence references; very short
        # templates ("Investigate INC-N on the boutique.") sit inside any request, so for
        # them only an identical frame or a Jaccard of 0.6 or more counts as overlap
        for refs_ in (refs, refs7):
            index = OverlapIndex([r for r in refs_ if _n_words(r) >= 8], 4)
            frames = {" ".join(frame_tokens(r)) for r in refs_}
            for text in shots:
                score, j = index.max_containment(text)
                assert score < 0.5, (text, index.references[j], score)
                assert " ".join(frame_tokens(text)) not in frames, text
                assert max(_jaccard(text, r) for r in refs_) < 0.6, text
        held = set(refs) | set(refs7)
        assert not set(shots) & held
        # the held-out sets are not in the prompts either
        blob = incb.INCIDENT_POLICY_SYSTEM_V2 + incb.INCIDENT_SUMMARY_SYSTEM_V2
        assert not any(t in blob for t in _texts(sealed_items))


# ------------------------------------------------------------------ the report


class TestReportV2:
    def test_render_names_the_goal_and_the_sealed_file(self):
        from .test_r11 import TestReport

        t = TestReport()
        recs = t.records()
        m = rep.system_metrics(recs, t.routes())
        report = {
            "generated": "now", "n_sealed": 11, "sealed_sha256": "x", "r7a_sealed_sha256": "y",
            "catalogue": {"n": 120, "fingerprint": "f"}, "team": "t",
            "goal": "G8b", "title": "R11 v2", "sealed_file": "r11b_sealed_test.jsonl",
            "systems": {
                "s": {
                    "label": "stub", "metrics": m, "verdict": rep.verdict(m, None),
                    "failures": rep.failures(recs),
                }
            },
        }  # fmt: skip
        md = rep.render(report)
        assert "(goal G8b)" in md and "r11b_sealed_test.jsonl" in md
        assert "| G8b verdict |" in md and "Analysis pending" in md
