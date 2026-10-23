"""The squad protocol on the object layer (``experiments.squad_runtime``).

Uses freshly initialised (untrained) networks, so it needs no trained parameters.
"""

import numpy as np
import pytest

from genetic_coil_arena.experiments.learned_protocol import (
    HAND_CODED,
    LEARNED,
    MODES,
    Nets,
    TrainSettings,
    policies_for,
)
from genetic_coil_arena.experiments.squad_runtime import (
    CENTRE,
    DEFAULT_KEYS,
    FALLBACK,
    LEAD,
    SQUAD_FORAGE,
    SquadRuntime,
    compare_episode,
    fast_path_episode,
    hold_last_hook,
    snake_names,
    squad_specs,
)
from genetic_coil_arena.protocol import (
    SpecValidationError,
    envelopes_to_codes,
    is_goal,
    verify,
)
from genetic_coil_arena.runtime import TraceLog

SETTINGS = TrainSettings(radius=3, max_steps=40)
NAMES = snake_names(3)
SEEDS = (11, 12)


def make_policies(mode: str = LEARNED, run: int = 0):
    theta = Nets.build(SETTINGS, mode).init(run)
    return policies_for(mode, SETTINGS, theta)


def make_runtime(mode: str = LEARNED, **kwargs) -> SquadRuntime:
    specialist, centre = make_policies(mode)
    return SquadRuntime(specialist, centre, SETTINGS.squad_config(), **kwargs)


# ------------------------------------------------------------------ GoalSpec
class TestSpec:
    def test_make_validates(self):
        specs = squad_specs()
        goal = specs.make(
            SQUAD_FORAGE, {"episode_seed": 3, "max_steps": 50, "assignment": NAMES}
        )
        assert goal.spec == SQUAD_FORAGE and goal.version == 1

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"episode_seed": -1},
            {"episode_seed": 1.5},
            {"episode_seed": 1, "max_steps": 0},
            {"episode_seed": 1, "max_steps": 10_000},
            {"episode_seed": 1, "assignment": NAMES[:2]},
            {"episode_seed": 1, "assignment": [NAMES[0]] * 3},
            {"episode_seed": 1, "colour": "red"},
        ],
    )
    def test_rejects_bad_params(self, params):
        with pytest.raises(SpecValidationError):
            squad_specs().make(SQUAD_FORAGE, params)

    def test_result_schema(self):
        specs = squad_specs()
        goal = specs.make(SQUAD_FORAGE, {"episode_seed": 0})
        good = {"team_food": 3, "food": [1, 1, 1], "steps": 9, "survivors": 2}
        assert specs.result_errors(goal, {**good, "return": 3.2}) == []
        assert specs.result_errors(goal, good)  # return missing


# ------------------------------------------------------------------ equivalence
class TestEquivalence:
    @pytest.mark.parametrize("mode", MODES)
    def test_matches_fast_path_step_for_step(self, mode):
        runtime = make_runtime(mode)
        specialist, centre = runtime_policies(mode)
        for seed in SEEDS:
            out = runtime.run_episode(seed)
            assert out.state == "completed", out.error
            fast = fast_path_episode(specialist, centre, SETTINGS.squad_config(), seed)
            row = compare_episode(out, fast)
            assert row["identical"], row
            assert out.result["team_food"] == fast["team_food"]
            assert out.result["steps"] == fast["steps"] == len(out.history["actions"])
            assert out.result["return"] == pytest.approx(fast["return"])

    def test_permuted_assignment_routes_envelopes(self):
        runtime = make_runtime()
        assignment = [NAMES[2], NAMES[0], NAMES[1]]
        out = runtime.run_episode(SEEDS[0], assignment=assignment)
        specialist, centre = runtime_policies(LEARNED)
        fast = fast_path_episode(specialist, centre, SETTINGS.squad_config(), SEEDS[0])
        assert compare_episode(out, fast)["identical"]
        codes = [
            e
            for e in runtime.trace.envelopes(out.trace_id)
            if e.payload_type == "codes"
        ]
        first = [e for e in codes if e.clock == codes[0].clock]
        assert [e.recipients[0] for e in first] == assignment

    def test_max_steps_from_goal(self):
        out = make_runtime().run_episode(SEEDS[0], max_steps=7)
        assert out.result["steps"] <= 7 and len(out.history["actions"]) <= 7

    def test_unknown_member_fails_task(self):
        out = make_runtime().run_episode(
            SEEDS[0], assignment=[NAMES[0], NAMES[1], "intruder"]
        )
        assert out.state == "failed" and "intruder" in out.error

    def test_runtime_is_reusable(self):
        runtime = make_runtime()
        a = runtime.run_episode(SEEDS[0])
        runtime.run_episode(SEEDS[1])
        again = runtime.run_episode(SEEDS[0])
        assert a.result == again.result
        for key in ("codes", "heard", "actions"):
            assert np.array_equal(a.history[key], again.history[key])


def runtime_policies(mode: str):
    """Fresh policies with the same parameters as ``make_runtime(mode)``."""
    return make_policies(mode)


# ------------------------------------------------------------------ trace
class TestTrace:
    def test_every_message_is_a_signed_codes_envelope(self, tmp_path):
        trace = TraceLog(tmp_path / "trace.jsonl")
        runtime = make_runtime(trace=trace)
        out = runtime.run_episode(SEEDS[0])
        trace.close()
        loaded = TraceLog.load(tmp_path / "trace.jsonl")
        envelopes = loaded.envelopes(out.trace_id)
        codes = [e for e in envelopes if e.payload_type == "codes"]
        assert sum(is_goal(e) for e in envelopes) == 1
        assert len(codes) == int(out.history["alive"].sum())
        for e in codes:
            assert e.sender == LEAD and e.task_id == out.task_id
            assert e.payload_encoding["V"] == 8 and e.payload_encoding["L"] == 1
            assert verify(e, {LEAD: DEFAULT_KEYS[LEAD]})
        assert all(
            verify(e, DEFAULT_KEYS) for e in envelopes if e.sender in (CENTRE, LEAD)
        )
        # The codes in the trace are the codes the lead computed, step by step
        steps = sorted({e.clock for e in codes})
        assert len(steps) == len(out.history["codes"])
        for t, clock in enumerate(steps):
            for e in (e for e in codes if e.clock == clock):
                code = envelopes_to_codes([[e]])[0, 0, 0]
                assert code == out.history["codes"][t, NAMES.index(e.recipients[0])]

    def test_situations_and_triples(self):
        runtime = make_runtime()
        out = runtime.run_episode(SEEDS[0])
        situations = runtime.trace.situations(out.trace_id)
        assert len(situations) == out.result["steps"] + 1
        assert [s["step"] for s in situations] == list(range(len(situations)))
        assert situations[-1]["done"] and not situations[0]["done"]
        assert set(situations[0]) >= {"heads", "food", "health", "alive", "score"}
        (triple,) = runtime.trace.triples()
        assert triple.spec.spec == SQUAD_FORAGE
        assert triple.spec.params["episode_seed"] == SEEDS[0]
        assert triple.situation == situations[-1]
        # A step's snapshot shares its clock with that step's code envelopes
        snaps = [r for r in runtime.trace.records if r.kind == "situation"]
        code_clocks = {
            e.clock for e in runtime.trace.envelopes() if e.payload_type == "codes"
        }
        assert {r.clock for r in snaps[:-1]} == code_clocks


# ------------------------------------------------------------------ tamper
class TestTamper:
    def test_tampered_envelope_is_rejected_and_falls_back(self):
        targets = [(1, NAMES[0]), (3, NAMES[0])]
        runtime = make_runtime(tamper=targets)
        out = runtime.run_episode(SEEDS[0])
        assert out.state == "completed"
        assert len(out.tampered) == 2 and len(out.rejections) == 2
        assert {r["reason"] for r in out.rejections} == {"payload digest mismatch"}
        assert {r["fallback"] for r in out.rejections} == {FALLBACK}
        events = runtime.trace.events("signature_rejected")
        assert len(events) == 2 and events[0].trace_id == out.trace_id
        # The snake acted on the last verified code: the fast path replays it
        specialist, centre = runtime_policies(LEARNED)
        replay = fast_path_episode(
            specialist,
            centre,
            SETTINGS.squad_config(),
            SEEDS[0],
            hook=hold_last_hook([(1, 0), (3, 0)]),
        )
        assert compare_episode(out, replay)["identical"]
        heard = out.history["heard"][:, 0]
        assert heard[2] == out.history["codes"][0, 0]  # step 1's code never arrived
        assert heard[4] == heard[3]

    def test_hand_coded_tamper_changes_nothing_else(self):
        runtime = make_runtime(HAND_CODED, tamper=[(2, NAMES[1])])
        out = runtime.run_episode(SEEDS[1])
        assert len(out.rejections) == 1
        assert out.state == "completed"


def test_hold_last_hook():
    hook = hold_last_hook([(1, 0)])
    m0 = np.array([[[3], [4]]])
    m1 = np.array([[[5], [6]]])
    assert hook(m0, 0).tolist() == m0.tolist()
    assert hook(m1, 1).tolist() == [[[3], [6]]]
