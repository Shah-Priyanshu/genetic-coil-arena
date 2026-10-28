"""R12c (goal G10b): the literal check, the paired desk, the sealed set and the report."""

import datetime as dt
import random
from collections import Counter

import pytest

from genetic_coil_arena.envs.logtriage import literal_check as lc
from genetic_coil_arena.envs.logtriage import rcaeval
from genetic_coil_arena.envs.logtriage.catalogue import (
    DESK_NOW,
    DESK_NOW_R12,
    ResolveError,
)
from genetic_coil_arena.experiments import r7a_sealed
from genetic_coil_arena.experiments import r11_incident as inc
from genetic_coil_arena.experiments import r11_sealed as sealed1
from genetic_coil_arena.experiments import r11b_sealed as sealed2
from genetic_coil_arena.experiments import r12_incident as inc3
from genetic_coil_arena.experiments import r12_sealed as sealed3
from genetic_coil_arena.experiments import r12c_sealed as sealed
from genetic_coil_arena.protocol.goalspec import SpecValidationError
from genetic_coil_arena.translator.distill import frame_ngrams, frame_tokens

NOW = DESK_NOW_R12


def chk(params, text):
    return lc.check_params(params, text, NOW)


def base(**kw):
    return {
        "system": "train-ticket",
        "top_k": 3,
        "want_fault_type": True,
        "exclude_services": [],
        **kw,
    }


# ------------------------------------------------------------------ the rules


class TestYear:
    def test_drops_a_year_the_text_does_not_name(self):
        r = chk(base(at="2024-11-22T11:59"), "Online boutique on Nov 22 at 11:59")
        assert r.params["at"] == "--11-22T11:59" and r.rules == ("year",)
        assert r.changes == {"at": ("2024-11-22T11:59", "--11-22T11:59")}

    def test_keeps_a_year_the_text_names(self):
        for text in (
            "boutique 22 November 2024 at 11:59",
            "boutique 2024-11-22 11:59",
            "boutique 22/11/24 11:59",
        ):
            assert chk(base(at="2024-11-22T11:59"), text).rules == (), text

    def test_a_ticket_number_or_a_clock_time_is_not_a_year(self):
        r = chk(base(at="2024-11-22"), "INC-2017 is the Nov 22 one, seen at 20:17")
        assert r.params["at"] == "--11-22"

    def test_relative_wording_keeps_the_year(self):
        r = chk(base(at="2024-12-10T12:38"), "train ticket yesterday 12:38")
        assert r.rules == ()
        r = chk(base(at="2024-12-10"), "last Tuesday, train ticket")
        assert r.rules == ()


class TestMonth:
    def test_drops_a_month_the_text_does_not_name(self):
        r = chk(base(at="--01-19T12:55"), "the boutique on the 19th around 12:55")
        assert r.params["at"] == "---19T12:55" and r.rules == ("month",)

    def test_year_and_month_both_unsupported(self):
        r = chk(base(at="2024-12-10T12:03"), "train ticket on the 10th at 12:03")
        assert r.params["at"] == "---10T12:03" and r.rules == ("year", "month")

    @pytest.mark.parametrize(
        "text",
        [
            "boutique, February 20th",
            "train ticket isue of 8 decemeber around 16:19",
            "sock shop jan 20 4:58 am",
            "shop 20/1 at 04:58",
            "boutique on 20.01. 04:58",
            "the sock shop on the 20th of last month",
            "train ticket 8 Dec",
            "boutique sept 3",
        ],
    )
    def test_keeps_a_month_the_text_names_or_may_name(self, text):
        assert chk(base(at="--01-20"), text).rules == (), text

    def test_a_clock_time_is_not_a_numeric_date(self):
        r = chk(base(at="--01-20T04:58"), "shop on the 20th at 04:58")
        assert r.params["at"] == "---20T04:58"


class TestRelative:
    def test_two_days_ago_replaces_a_bare_day(self):
        r = chk(base(at="---20T23:49"), "Two days ago, 23:49, boutique - root cause?")
        assert r.params["at"] == "2024-12-09T23:49" and r.rules == ("relative",)

    def test_words_and_digits(self):
        for text, want in (
            ("train ticket three days ago", "2024-12-08"),
            ("train ticket 3 days ago", "2024-12-08"),
            ("train ticket yesterday", "2024-12-10"),
            ("train ticket the day before yesterday", "2024-12-09"),
        ):
            r = chk(base(at="---01"), text)
            assert r.params["at"] == want, text

    def test_right_value_is_left_alone(self):
        assert chk(base(at="2024-12-10T12:38"), "yesterday 12:38").rules == ()

    def test_never_overrides_an_explicit_date_or_two_wordings(self):
        assert chk(base(at="--12-10"), "yesterday, Dec 10").rules == ()
        assert chk(base(at="---05"), "yesterday and three days ago").rules == ()

    def test_does_not_add_at(self):
        r = chk(base(ticket="INC-2050"), "INC-2050, from yesterday")
        assert "at" not in r.params and r.rules == ()


class TestTopK:
    @pytest.mark.parametrize(
        "text,want",
        [
            ("INC-1 top 8 suspects", 8),
            ("INC-1, top-4 please", 4),
            ("INC-1, give me five suspects", 5),
            ("INC-1 the 5 most likely culprits", 5),
            ("INC-1: top ten", 10),
            ("INC-1, two likely suspects", 2),
            ("INC-1, the 3 top candidates", 3),
            ("INC-1, one suspect only", 1),
        ],
    )
    def test_text_count_wins(self, text, want):
        r = chk(base(top_k=3 if want != 3 else 5), text)
        assert r.params["top_k"] == want and r.rules == ("top_k",)

    def test_clamped_count_becomes_invalid_not_clamped(self):
        registry = inc3.build_registry_r12()
        r = chk(base(top_k=5, ticket="INC-2001"), "INC-2001: the top 8 suspects")
        assert r.params["top_k"] == 8
        with pytest.raises(SpecValidationError):
            registry.make(inc.INVESTIGATE, r.params)

    @pytest.mark.parametrize(
        "text",
        [
            "INC-2001 leave out these two services",
            "INC-2001, I need it by 12:30",
            "top 3 and also top 4",
            "INC-2001 on the 3rd at 14:21",
            "just INC-2001",
        ],
    )
    def test_no_stated_count_changes_nothing(self, text):
        assert chk(base(top_k=3), text).rules == (), text

    def test_missing_top_k_takes_the_text_count(self):
        params = base()
        del params["top_k"]
        assert chk(params, "top 2 for INC-2001").params["top_k"] == 2


# ------------------------------------------------------------------ never adds information

TEXTS = [
    "Boutique on Nov 22 at 14:07, five suspects and leave out adservice",
    "train ticket yesterday 12:38, top 2",
    "the 19th around 12:55 on the shop",
    "INC-2017 please, three suspects",
    "Two days ago, 23:49, boutique",
    "sock shop 2024-01-19 21:22",
    "nothing to see here",
    "top 12 for the 3rd",
]
ATS = [
    None,
    "2024-12-10",
    "2024-11-22T14:07",
    "--11-22T14:07",
    "--01-19",
    "---19T12:55",
    "---20T23:49",
    "2024-01-19T21:22",
]


class TestNeverAdds:
    def test_random_params_and_texts(self):
        rng = random.Random(7)
        for _ in range(400):
            text = rng.choice(TEXTS)
            params = base(top_k=rng.choice([1, 3, 5, 8]))
            at = rng.choice(ATS)
            if at:
                params["at"] = at
            if rng.random() < 0.4:
                params["ticket"] = "INC-2017"
            before = dict(params)
            res = chk(params, text)
            assert params == before  # the input is never modified
            assert set(res.params) <= set(params) | {"top_k"}
            assert ("at" in res.params) == ("at" in params)
            for key in ("system", "ticket", "exclude_services", "want_fault_type"):
                assert res.params.get(key) == params.get(key)
            a, b = params.get("at"), res.params.get("at")
            if a != b and "relative" not in res.rules:
                # without a relative wording only the year, then the month, may go
                if a[0].isdigit():
                    allowed = {f"--{a[5:]}", f"---{a[8:]}"}
                else:
                    allowed = {f"---{a[5:]}"}
                assert b in allowed, (a, b)
            if res.params.get("top_k") != params.get("top_k"):
                assert res.params["top_k"] == lc.stated_count(text.lower())

    def test_idempotent(self):
        for text in TEXTS:
            for at in ATS:
                params = base(at=at) if at else base()
                once = chk(params, text)
                assert chk(once.params, text).rules == ()

    def test_gold_of_every_spec_item_is_a_fixed_point(self):
        """The check never changes a gold spec of the earlier v2/v3 sets or of R12c."""
        sets = [
            (sealed2.load_sealed(), DESK_NOW),
            (sealed2.load_dev(), DESK_NOW),
            (sealed3.load_sealed(), DESK_NOW_R12),
            (sealed3.load_dev(), DESK_NOW_R12),
            (sealed.load_sealed(), DESK_NOW_R12),
            (sealed.load_dev(), DESK_NOW_R12),
        ]
        n = 0
        for items, now in sets:
            for it in items:
                if it.kind != "spec":
                    continue
                n += 1
                res = lc.check_params(it.gold, it.text, now)
                assert res.params == it.gold and res.rules == (), (it.id, it.text, res)
        assert n > 300


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


def _jaccard(a: str, b: str) -> float:
    ga, gb = frame_ngrams(a, 4), frame_ngrams(b, 4)
    return len(ga & gb) / max(1, len(ga | gb))


class TestSealedR12c:
    def test_file_hash_is_pinned(self):
        assert sealed.file_sha256(sealed.SEALED_PATH) == sealed.SEALED_SHA256

    def test_load_refuses_a_changed_file(self, tmp_path):
        bad = tmp_path / "x.jsonl"
        bad.write_bytes(sealed.SEALED_PATH.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="sealed"):
            sealed.load_sealed(path=bad)

    def test_earlier_sealed_hashes_still_pass(self):
        assert sealed3.file_sha256(sealed3.SEALED_PATH) == sealed3.SEALED_SHA256
        assert sealed2.file_sha256(sealed2.SEALED_PATH) == sealed2.SEALED_SHA256

    def test_size_and_mix(self, sealed_items):
        assert len(sealed_items) == 106
        assert Counter(i.kind for i in sealed_items) == {
            "spec": 84,
            "clarify": 8,
            "refuse": 14,
        }
        assert len({i.id for i in sealed_items}) == len({i.text for i in sealed_items})
        cats = Counter(i.category for i in sealed_items)
        assert set(cats) >= {
            "ticket", "time", "partial", "relative", "topk", "exclude", "fault",
            "combo", "clarify", "refuse",
        }  # fmt: skip
        assert len({i.style for i in sealed_items}) >= 8
        assert len(sealed.load_dev()) == 31

    def test_weighted_toward_dates_and_counts(self, sealed_items):
        spec = [i for i in sealed_items if i.kind == "spec"]
        at = [i for i in spec if "at" in i.gold]
        assert len(at) >= 50
        assert sum(i.gold["at"].startswith("--") for i in at) >= 25  # no year
        assert sum(i.gold["at"].startswith("---") for i in at) >= 12  # day only
        assert sum(i.gold["at"].startswith("20") for i in at) >= 10  # year given
        assert sum(i.category == "relative" for i in spec) >= 8
        assert sum(i.gold["top_k"] != 3 for i in spec) >= 15

    def test_guess_traps(self, sealed_items):
        traps = sealed.load_traps()
        by_id = {i.id: i for i in sealed_items}
        assert len(traps) == 13
        kinds = Counter(by_id[t].kind for t in traps)
        assert kinds == {"clarify": 5, "refuse": 6, "spec": 2}
        for t in traps:
            assert "20th" in by_id[t].text or by_id[t].kind == "refuse"

    def test_gold_specs_validate(self, sealed_items):
        registry = inc3.build_registry_r12()
        for it in sealed_items:
            if it.kind == "spec":
                registry.make(inc.INVESTIGATE, it.gold)
            else:
                assert it.gold is None

    def test_gold_resolves_to_exactly_the_documented_incident(
        self, sealed_items, real_catalogue
    ):
        target = sealed.load_resolves_to()
        for it in sealed_items:
            if it.kind == "spec":
                assert real_catalogue.resolve(it.gold).ticket == target[it.id], it.id
            else:
                assert target[it.id] is None

    def test_the_traps_are_traps(self, real_catalogue):
        """The boutique's 20th names two dates; a guessed month gives one incident."""
        base_p = {
            "system": "online-boutique",
            "top_k": 3,
            "want_fault_type": True,
            "exclude_services": [],
        }
        with pytest.raises(ResolveError) as exc:
            real_catalogue.resolve({**base_p, "at": "---20"})
        assert exc.value.kind == "clarify"
        assert real_catalogue.resolve({**base_p, "at": "--02-20"}).ticket == "INC-1118"

    def test_clarify_and_refuse_items_are_not_answerable(
        self, sealed_items, real_catalogue
    ):
        """Every clarify/refuse item's best-case reading does not resolve to an incident."""
        by_text = {i.text: i for i in sealed_items}
        cases = {
            "Sock shop, 22 Nov.": ("sock-shop", "--11-22"),
            "Investigate the train ticket incident of December 8.": ("train-ticket", "--12-08"),
            "Train ticket, two days ago at 10:00.": ("train-ticket", "2024-12-09T10:00"),
            "sock shop four days ago 11:00": ("sock-shop", "2024-12-07T11:00"),
            "Any sock shop trouble on 2024-11-30, about 09:00? Diagnose it.": ("sock-shop", "2024-11-30T09:00"),
        }  # fmt: skip
        for text, (system, at) in cases.items():
            assert text in by_text, text
            with pytest.raises(ResolveError):
                real_catalogue.resolve({"system": system, "at": at, "top_k": 3})

    def test_dev_is_a_second_set(self, sealed_items):
        dev = sealed.load_dev()
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
            "r12 sealed": _texts(sealed3.load_sealed()),
            "r12 dev": _texts(sealed3.load_dev()),
            "r7a": r7a,
        }
        for name, cands, extra in (
            ("sealed", _texts(sealed_items), {"r12c dev": _texts(dev)}),
            ("dev", _texts(dev), {}),
        ):
            for ref_name, refs in {**others, **extra}.items():
                frames = {" ".join(frame_tokens(t)) for t in refs}
                for t in cands:
                    assert " ".join(frame_tokens(t)) not in frames, (name, ref_name, t)
                    worst = max(_jaccard(t, r) for r in refs)
                    assert worst < 0.5, (name, ref_name, t, worst)

    def test_the_prompts_are_the_r12_ones(self):
        """G10b changes no prompt: the v3 pack is what the desk uses."""
        audit = inc3.prompt_diff()
        assert audit["summary_pack_same"] and audit["router_other_lines_same"]


def test_clock_matches_r12():
    assert NOW == dt.datetime(2024, 12, 11, 9, 0, tzinfo=dt.timezone.utc)
