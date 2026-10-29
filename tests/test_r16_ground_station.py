"""R16 ground station: the swarm.behavior spec, site map, state check, BEHAVIOR_CMD
packets, the translator wrapper, the keyword baseline and scoring."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from genetic_coil_arena.envs.swarm import ground as gs
from genetic_coil_arena.experiments import r16_ground_station as r16
from genetic_coil_arena.experiments import r16_report
from genetic_coil_arena.protocol.goalspec import SpecValidationError

REG = r16.build_registry()


def make(params):
    return REG.make(gs.SPEC_NAME, gs.fill_behavior_defaults(params), version=1)


def test_behavior_ids_match_openswarm_firmware():
    assert gs.BEHAVIORS == {"hover": 0, "formation": 1, "boids": 2, "search_grid": 3,
                            "perimeter": 4, "person_follow": 5, "light_show": 6,
                            "rth": 7}  # fmt: skip
    assert gs.SHAPES == {"line": 0, "v": 1, "grid": 2, "circle": 3}


def test_defaults_filled_per_behaviour():
    assert make({"behavior": "formation"}).params == {
        "behavior": "formation", "shape": "v", "spacing_m": 1.5}  # fmt: skip
    assert make({"behavior": "hover"}).params == {"behavior": "hover"}
    pf = make({"behavior": "person_follow", "distance_m": 5}).params
    assert pf == {"behavior": "person_follow", "distance_m": 5, "alt_m": 2.0,
                  "min_conf": 0.7}  # fmt: skip


@pytest.mark.parametrize(
    "params",
    [
        {"behavior": "boids", "alt_m": 5},  # a param boids does not have
        {"behavior": "perimeter", "area": "barn", "spacing_m": 2},
        {"behavior": "formation", "spacing_m": 0.5},  # below the 0.8 m separation
        {"behavior": "hover", "alt_m": 50},
        {"behavior": "search_grid"},  # no area
        {"behavior": "search_grid", "area": "lake"},
        {"behavior": "land"},
        {"alt_m": 3},
    ],
)
def test_schema_refuses(params):
    with pytest.raises(SpecValidationError):
        make(params)


def test_find_place_aliases_and_articles():
    assert gs.find_place("Parking Lot").name == "car park"
    assert gs.find_place("the old barn").name == "barn"
    assert gs.find_place("The North Field").name == "north field"
    assert gs.find_place("lake") is None


def _state(n=6, vision=True, battery=90.0, indoor=False):
    return gs.SwarmState(
        tuple(gs.Drone(i, battery if i == 0 else 90.0, vision and i == 0)
              for i in range(n)), indoor,
    )  # fmt: skip


def test_check_rules():
    assert gs.check({"behavior": "rth"}, _state()).ok
    assert not gs.check({"behavior": "person_follow"}, _state(vision=False)).ok
    assert not gs.check({"behavior": "formation", "shape": "grid"}, _state(n=3)).ok
    assert gs.check({"behavior": "formation", "shape": "line"}, _state(n=2)).ok
    assert not gs.check({"behavior": "hover", "alt_m": 4}, _state(indoor=True)).ok
    assert gs.check({"behavior": "hover", "alt_m": 2.5}, _state(indoor=True)).ok
    assert not gs.check({"behavior": "search_grid", "area": "school"}, _state()).ok
    assert not gs.check({"behavior": "boids"}, _state(battery=8)).ok
    assert gs.check({"behavior": "rth"}, _state(battery=8)).ok
    low = gs.check({"behavior": "boids"}, _state(battery=15))
    assert low.ok and low.warnings
    assert not gs.check({"behavior": "rth"}, gs.SwarmState()).ok


def test_encode_fits_and_decodes():
    worst = [
        {"behavior": "boids", "cohesion": 4.55, "separation": 3.35, "alignment": 2.25},
        {"behavior": "search_grid", "area": "highway", "alt_m": 29.5},
        {
            "behavior": "perimeter",
            "area": "east woods",
            "radius_m": 99.5,
            "alt_m": 29.5,
        },
        {
            "behavior": "person_follow",
            "distance_m": 9.5,
            "alt_m": 29.5,
            "min_conf": 0.95,
        },
        {"behavior": "formation", "shape": "circle", "spacing_m": 9.5, "alt_m": 29.5},
        {"behavior": "light_show", "pattern": "sparkle", "bpm": 199},
    ]
    for p in worst:
        pkt = gs.encode(make(p).params)
        assert len(pkt) == 2 + gs.PARAMS_MAX
        assert gs.decode(pkt)["behavior"] == p["behavior"]
    grid = gs.decode(gs.encode({"behavior": "search_grid", "area": "car park"}))
    assert grid["params"] == {"a": 5, "b": [50, -20, 90, 20]} and grid["count"] == 2
    assert gs.decode(gs.encode({"behavior": "rth"})) == {
        "behavior": "rth", "count": 0, "params": {}}  # fmt: skip


class FakeBackend:
    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, prompts, **kw):
        body = self.replies.pop(0)
        return [
            SimpleNamespace(text=json.dumps(body), latency_s=0.0, n_prompt_tokens=0)
        ]


def translate(*replies, text="x"):
    return r16.GroundTranslator(FakeBackend(replies), REG).translate(text, r16.pack())


def test_translator_resolves_places_and_fills_defaults():
    res = translate(
        {"params": {"behavior": "Search Grid", "area": "parking lot"}},
        text="search the parking lot",
    )
    assert res.status == "ok"
    assert res.spec.params == {
        "behavior": "search_grid",
        "area": "car park",
        "alt_m": 5.0,
    }


def test_translator_asks_for_missing_or_unknown_places():
    assert translate({"params": {"behavior": "perimeter"}}).status == "clarify"
    res = translate({"params": {"behavior": "search_grid", "area": "the lake"}})
    assert res.status == "clarify" and "lake" in res.reason


def test_translator_refuses_after_invalid_retry():
    text = "fly in formation 0.3 m apart"
    bad = {"params": {"behavior": "formation", "spacing_m": 0.3}}
    assert translate(bad, bad, text=text).status == "refused"
    # a model that clamps to the 0.8 m limit on the retry is stopped by the literal check
    clamped = {"params": {"behavior": "formation", "spacing_m": 0.8}}
    assert translate(bad, clamped, text=text).status == "refused"
    assert translate({"refusal": "not a swarm command"}).status == "refused"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("all units RTH", {"behavior": "rth"}),
        ("line up 2 m apart", {"behavior": "formation", "shape": "line", "spacing_m": 2.0}),
        ("sweep the parking lot at 10 m",
         {"behavior": "search_grid", "area": "car park", "alt_m": 10.0}),
        ("what's the weather", None),
    ],
)  # fmt: skip
def test_keyword_baseline(text, expected):
    assert r16.keyword_params(text) == expected


def test_score_and_correct():
    ok = SimpleNamespace(
        status="ok", spec=SimpleNamespace(params={"behavior": "boids"})
    )
    item = {"expect": "ok", "params": {"behavior": "boids", "cohesion": 1}}
    assert r16.score(ok, item) == "exact"
    assert (
        r16.score(ok, {"expect": "ok", "params": {"behavior": "rth"}})
        == "wrong command"
    )
    assert r16.score(ok, {"expect": "refuse"}) == "wrong command"
    ask = SimpleNamespace(status="clarify", spec=None)
    assert r16.correct(r16.score(ask, {"expect": "clarify"}), "clarify")


def test_report_verdict():
    def run(tr, exact, wrong):
        return {"summary": {"split": "sealed", "translator": tr, "n": 100,
                            "exact": (exact, exact - 0.05, exact + 0.05),
                            "asked_when_unclear": (1, 1, 1),
                            "refused_when_should": (1, 1, 1),
                            "wrong_commands": (wrong, 0.0, wrong + 0.02),
                            "median_latency_s": 1.0}, "rows": []}  # fmt: skip

    md, data = r16_report.render(
        [run(r16_report.MAIN, 0.95, 0.02), run("keyword", 0.6, 0.2)]
    )
    assert data["g14"]["passed"] and "G14 PASSED" in md
    md, data = r16_report.render(
        [run(r16_report.MAIN, 0.95, 0.05), run("keyword", 0.6, 0.2)]
    )
    assert not data["g14"]["passed"]


def test_label_guide_lists_the_site_map():
    guide = (Path(r16.__file__).parent / "data" / "r16_label_guide.md").read_text(
        encoding="utf-8"
    )
    for place in gs.SITE:
        assert f"| {place.name} | {', '.join(place.aliases)} |" in guide
    for b in gs.BEHAVIORS:
        assert f"`{b}`" in guide


def test_sealed_set_is_pinned_and_every_gold_command_is_valid():
    from genetic_coil_arena.experiments import r16_sealed

    assert r16_sealed.file_sha256(r16_sealed.SEALED_PATH) == r16_sealed.SEALED_SHA256
    for items in (r16_sealed.load_sealed(), r16_sealed.load_dev()):
        assert len({i["id"] for i in items}) == len(items)
        for i in items:
            assert i["expect"] in ("ok", "clarify", "refuse")
            if i["expect"] == "ok":
                make(i["params"])
            else:
                assert "params" not in i
    sealed = {i["text"] for i in r16_sealed.load_sealed()}
    assert not sealed & {i["text"] for i in r16_sealed.load_dev()}
    assert not sealed & {e.text for e in r16.EXAMPLES}


def test_spoken_numbers_and_literal_check():
    assert gs.spoken_numbers("one hundred and forty bpm") == [140.0]
    assert sorted(gs.spoken_numbers("spacing five metres, height 10 m")) == [5.0, 10.0]
    assert gs.spoken_numbers("twenty-five m, a hundred feet") == [25.0, 100.0]
    ok = {"behavior": "perimeter", "area": "home", "radius_m": 6.1}
    assert gs.unsupported(ok, "orbit home 20 ft out") == []
    assert (
        gs.unsupported({"behavior": "person_follow", "min_conf": 0.8}, "80% sure") == []
    )
    clamped = {"behavior": "formation", "shape": "v", "spacing_m": 10}
    assert len(gs.unsupported(clamped, "get in a V at 15 m altitude")) == 1
    invented = {"behavior": "formation", "shape": "line", "spacing_m": 2}
    assert len(gs.unsupported(invented, "formation with 2 m between each drone")) == 1
    assert gs.unsupported(
        {"behavior": "search_grid", "area": "north field"}, "sweep the hill"
    )


def test_translator_question_beats_refusal_and_literals_are_checked():
    both = {"refusal": "not enough detail", "clarify": "Which area?"}
    assert translate(both).status == "clarify"
    clamped = {"params": {"behavior": "formation", "shape": "v", "spacing_m": 10}}
    res = translate(clamped, clamped, text="get in a V at 15 m altitude")
    assert res.status == "refused" and res.attempts == 2
    moved = {"params": {"behavior": "formation", "shape": "v", "alt_m": 15}}
    res = translate(clamped, moved, text="get in a V at 15 m altitude")
    assert res.status == "ok" and res.spec.params["alt_m"] == 15


def test_v3_schema_order_question_refusals_and_snapped_literals():
    props = list(gs.swarm_behavior_definition().params_schema["properties"])
    assert props[0] == "behavior" and props[-1] == "alt_m"  # alt_m is shared: last
    asks = {"refusal": "The lake is not on the site map. Which mapped area?"}
    assert translate(asks).status == "clarify"
    assert translate({"refusal": "Not a swarm command."}).status == "refused"
    near = {"params": {"behavior": "perimeter", "area": "home", "radius_m": 6.08}}
    res = translate(near, text="orbit home 20 ft out")
    assert res.status == "ok" and res.spec.params["radius_m"] == 6.1
    snapped = gs.snap_literals({"behavior": "hover", "alt_m": 6.096}, "at twenty feet")
    assert snapped["alt_m"] == 6.1
    assert gs.snap_literals({"min_conf": 0.9}, "confidence 0.9") == {"min_conf": 0.9}


def test_v4_every_stated_length_sets_the_param_its_words_name():
    assert gs.with_digits("Everybody line up, six-feet apart.") == (
        "everybody line up , 6 feet apart ."
    )
    assert gs.stated_lengths("perimeter of the barn with 3 m spacing") == [
        ("3 m", 3.0, ("spacing_m",))
    ]
    assert gs.stated_lengths("altitude of twenty ft") == [("20 ft", 6.1, ("alt_m",))]
    assert gs.stated_lengths("hover at 12 m") == [("12 m", 12.0, ())]
    moved = {"behavior": "perimeter", "area": "barn", "radius_m": 3}
    assert gs.misplaced_lengths(moved, "perimeter of the barn with 3 m spacing")
    dropped = {"behavior": "formation", "shape": "line"}
    assert gs.misplaced_lengths(dropped, "line up six feet apart")
    placed = {"behavior": "formation", "shape": "line", "spacing_m": 1.83}
    assert gs.misplaced_lengths(placed, "line up six feet apart") == []
    # a length equal to the behaviour's default is placed (the command flies the same)
    assert gs.misplaced_lengths({"behavior": "search_grid", "area": "field"},
                                "search the field at 5 m") == []  # fmt: skip
    res = translate({"params": moved}, text="perimeter of the barn with 3 m spacing")
    assert res.status == "clarify" and "spacing_m" in res.reason
    res = translate({"params": placed}, text="line up six feet apart")
    assert res.status == "ok"
