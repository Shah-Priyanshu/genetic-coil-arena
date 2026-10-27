"""R12 (goals G9 and G10): the log-novelty newcomer, the joined centre, the grown desk.

Everything runs without a model (the desk is driven by ``KeywordBackend``). Checks that need
the RCAEval cache skip when it is absent.
"""

import datetime as dt
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
