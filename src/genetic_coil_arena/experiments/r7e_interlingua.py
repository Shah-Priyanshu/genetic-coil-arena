"""R7e probe: can independently trained squads interoperate through a shared meaning space?

Exploratory probe, not a goal test. G12 found that every (b) squad of G7/G7c speaks its
own dialect: another squad's decoder reads only 24% of its eater sets. Each decoder also
maps the dialect onto a human-defined meaning space (status: eat or do not eat, and
situation: the hand code's direction and near/far of the receiver's food). This probe
uses that space as an interlingua. Squad A's centre drives squad B's specialists, and a
bridge translates every symbol:

- read A: the running naive-Bayes log-odds of each receiver's status, built from A's
  table ``P(symbol | status, situation)`` over the symbols that receiver got so far;
- write B: B's writer table gives the symbol for (decoded status, current situation).

Conditions per pair (A, B), on the 6 held-out goals:

- ``native``: B's centre and B's specialists (the ceiling);
- ``raw``: A's centre talks to B's specialists with no bridge;
- ``bridged``: A's centre through the bridge;
- ``hand``: the hand-coded goal centre (d), whose status is explicit, written in B's
  dialect (a hand-written centre speaking a learned language);
- ``oracle``: the true status written in B's dialect (the ceiling of any writer-based
  bridge).

Pairs: the G12 sample (34 (b) runs), each run A paired with the next run B. Decoders are
fitted on training-goal episodes only, exactly as in G12.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np

from ..envs.squad import SquadEnv, run_episodes
from ..envs.squad.engine import ALIVE
from ..evaluation import mean_ci
from .r7b_goal_conditioned import (
    CENTRE_GOAL,
    Goal,
    GoalArrays,
    GoalHandCodedCentre,
    GoalTrainSettings,
    _GoalAware,
    goal_sets,
    mission_score,
    policies_for,
)
from .r7b_report import HELDOUT, SEEN
from .r7c_reliability import FLAG_BELOW
from .r7d_legibility import (
    FIT_SEED_BASE,
    N_SITUATIONS,
    TEST_SEED_BASE,
    SymbolRecorder,
    episode_seeds,
    fit_decoder,
    play,
)

CONDITIONS = ("native", "raw", "bridged", "hand", "oracle")
FIT_PER_GOAL, TEST_PER_GOAL = 40, 50


def status_table(rec: dict[str, np.ndarray], vocab: int) -> np.ndarray:
    """(2, S, V) log P(symbol | status, situation), add-one smoothed, from fit episodes."""
    status = np.broadcast_to(rec["eaters"][:, None, :], rec["symbols"].shape)
    live = rec["live"]
    t = np.ones((2, N_SITUATIONS, vocab))
    np.add.at(
        t,
        (status[live].astype(np.int64), rec["situation"][live], rec["symbols"][live]),
        1,
    )
    return np.log(t / t.sum(axis=2, keepdims=True))


class Bridge:
    """Hook state: reads A's symbols into a running status, writes B's symbols."""

    def __init__(
        self,
        rec: SymbolRecorder,
        writer: np.ndarray,
        reader: tuple[np.ndarray, float] | None = None,
        status: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ):
        self.rec, self.writer = rec, writer
        self.reader, self.status = reader, status
        self.llr: np.ndarray | None = None

    def __call__(self, messages: np.ndarray, step: int, mask: Any = None) -> np.ndarray:
        s = self.rec.situation
        assert s is not None
        sym = np.asarray(messages)[:, :, 0]
        if self.status is not None:
            eat = self.status(sym, s)
        else:
            assert self.reader is not None
            table, prior = self.reader
            if self.llr is None:
                self.llr = np.zeros(sym.shape)
            self.llr += table[1, s, sym] - table[0, s, sym]
            eat = self.llr + prior > 0
        out = np.array(messages, copy=True)
        out[:, :, 0] = self.writer[eat.astype(np.int64), s]
        return out


def play_pair(
    settings: GoalTrainSettings,
    specialist: Any,
    centre: Any,
    goals: Sequence[Goal],
    seeds: Sequence[int],
    hook_for: Callable[[SymbolRecorder, GoalArrays], Bridge] | None = None,
) -> float:
    """Mean mission score of B's specialists driven by ``centre`` (optionally bridged)."""
    arrays = GoalArrays.from_goals(goals)
    for policy in (specialist, centre):
        if isinstance(policy, _GoalAware):
            policy.set_goals(arrays)
    rec = SymbolRecorder(0)
    hook = hook_for(rec, arrays) if hook_for else None
    out = run_episodes(
        SquadEnv(settings.squad_config()), specialist, centre, list(seeds), hook, rec
    )
    alive = out["death_reason"] == ALIVE
    return float(mission_score(out["food"], alive, arrays).mean())


def fit_dialect(
    t: dict, settings: GoalTrainSettings
) -> tuple[np.ndarray, float, np.ndarray]:
    """(status table, prior log-odds, writer) of one squad, from training-goal episodes."""
    vocab = settings.squad_config().message.vocab_size
    run = int(t["run"])
    goals = [g for g in goal_sets()[SEEN] for _ in range(FIT_PER_GOAL)]
    rec = play(
        t["mode"], settings, t["theta"], goals,
        episode_seeds(FIT_SEED_BASE, run, len(goals)),
    )  # fmt: skip
    dec = fit_decoder(rec, vocab)
    return status_table(rec, vocab), dec.prior, dec.writer


def evaluate_pair(job: tuple[dict, dict, dict[str, Any]]) -> dict[str, Any]:
    a, b, settings_d = job
    settings = GoalTrainSettings(**settings_d)
    table_a, prior_a, _ = fit_dialect(a, settings)
    _, _, writer_b = fit_dialect(b, settings)
    goals = [g for g in goal_sets()[HELDOUT] for _ in range(TEST_PER_GOAL)]
    seeds = episode_seeds(TEST_SEED_BASE, int(b["run"]), len(goals))
    spec_b, centre_b = policies_for(CENTRE_GOAL, settings, b["theta"])
    _, centre_a = policies_for(CENTRE_GOAL, settings, a["theta"])

    def run(centre: Any, hook_for: Any = None) -> float:
        return play_pair(settings, spec_b, centre, goals, seeds, hook_for)

    scores = {
        "native": run(centre_b),
        "raw": run(centre_a),
        "bridged": run(
            centre_a, lambda rec, arr: Bridge(rec, writer_b, (table_a, prior_a))
        ),
        "hand": run(
            GoalHandCodedCentre(),
            lambda rec, arr: Bridge(rec, writer_b, status=lambda sym, s: sym > 0),
        ),
        "oracle": run(
            centre_b,
            lambda rec, arr: Bridge(
                rec, writer_b, status=lambda sym, s, e=arr.eaters: e
            ),
        ),
    }
    return {"a": int(a["run"]), "b": int(b["run"]), "scores": scores}


def pairs(trained: Sequence[dict]) -> list[tuple[dict, dict]]:
    """The G12 sample in run order, each run paired with the next (cyclically)."""
    sample = sorted(
        (
            t
            for t in trained
            if t["mode"] == CENTRE_GOAL and t["best_val_mission"] >= FLAG_BELOW
        ),
        key=lambda t: int(t["run"]),
    )
    return [(sample[i], sample[(i + 1) % len(sample)]) for i in range(len(sample))]


def summarise(rows: Sequence[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    native = np.array([r["scores"]["native"] for r in rows])
    for c in CONDITIONS:
        v = np.array([r["scores"][c] for r in rows])
        m, ci = mean_ci(list(v))
        frac = v / native
        fm, fci = mean_ci(list(frac))
        out[c] = {"mean": m, "ci": ci, "of_native": fm, "of_native_ci": fci}
    gap = np.array([r["scores"]["bridged"] - r["scores"]["raw"] for r in rows])
    out["bridged_minus_raw"] = dict(zip(("mean", "ci"), mean_ci(list(gap))))
    return out


def render(rep: dict[str, Any]) -> str:
    s = rep["summary"]
    lines = [
        "# R7e probe: independently trained squads through a shared meaning space",
        "",
        "Exploratory probe (not a pre-registered goal). Squad A's centre drives squad B's "
        "specialists on the 6 held-out goals; the bridge reads A's symbols into "
        "eat/do-not-eat with A's G12 decoder and writes them in B's dialect with B's G12 "
        f"writer. {rep['n_pairs']} pairs (each G12-sample run with the next), "
        f"{TEST_PER_GOAL} episodes per goal. Mean ± 95% t-CI over pairs.",
        "",
        "| Condition | Mission score | Share of B's native score |",
        "|---|---|---|",
    ]
    labels = {
        "native": "B's own centre (native)",
        "raw": "A's centre, no bridge",
        "bridged": "A's centre through the bridge",
        "hand": "hand-coded centre written in B's dialect",
        "oracle": "true status written in B's dialect",
    }
    for c in CONDITIONS:
        r = s[c]
        lines.append(
            f"| {labels[c]} | {r['mean']:.2f} ± {r['ci']:.2f} "
            f"| {100 * r['of_native']:.0f}% ± {100 * r['of_native_ci']:.0f} |"
        )
    g = s["bridged_minus_raw"]
    lines += [
        "",
        f"Bridge gain over no bridge: {g['mean']:+.2f} ± {g['ci']:.2f}.",
        "",
        "Reproduce: `python scripts/r7e_interlingua.py --from-json runs:r7b3-train-* "
        "runs:r7b4-train-* runs:r7b5-train-* runs:r7c-train-*`.",
        "",
    ]
    return "\n".join(lines)
