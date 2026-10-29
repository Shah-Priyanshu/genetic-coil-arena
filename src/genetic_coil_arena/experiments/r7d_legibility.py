"""R7d: can a monitor read and write the squad's learned language? (goal G12)

In G7 and G7c, condition (b) trained a command centre that tells three goal-agnostic
specialists, through one 3-bit symbol per step, which of them should eat. The goal is
only at the centre, so it must travel in those symbols. The roadmap's reverse translator
has to decode such payloads for monitoring. Every trained squad has its own dialect
(symbol assignments differ between runs), so the monitor learns one decoder per squad
from episodes on the 15 *training* goals only and is tested on the 6 held-out goals.

Meaning space. The hand-designed code of condition (d) supplies the meanings:

- ``status``: eat or do not eat, from the GoalSpec;
- ``situation``: R6's direction and near/far of the receiver's nearest own-colour food
  (``HandCodedCentre``, 8 values), computed from the centre's observation.

The decoder is a table of symbol counts from the fit episodes:

- read: naive Bayes over the symbols a live receiver gets in the first ``WINDOW``
  steps (add-one smoothing, prior log-odds from the fit episodes) gives each snake's
  status. The predicted eaters (or the highest-scoring snake, if none is predicted)
  are the episode's decoded ``eaters``. Caution is decoded the same way from all
  receivers' symbols. The decoded GoalSpec is built through ``SpecRegistry.make``.
- write: every step the writer sends the receiver the most frequent fit-episode symbol
  for (status, situation), ties to the lowest symbol. One snake at a time is steered
  with the "eat" and with the "do not eat" writer on the held-out episodes. Its write
  efficacy is ``(food_eat - food_no) / (food of intact eaters - food of intact
  non-eaters)``.

Controls:

- read null: the decoder fitted with eater labels shuffled across fit episodes;
- write null: ``N_RANDOM_WRITERS`` pairs of random writer tables;
- (d), the hand-coded centre: its own code, an upper reference;
- (a), the goal-blind centre: its symbols carry no goal;
- (c): the specialists see the goal, so the symbols need not carry it;
- cross-run transfer: one squad's decoder applied to another squad's symbols.

Verdict, on the (b) runs whose best training-goal validation is at least 12 (G7c's
rule, ``FLAG_BELOW``): pass if the held-out eater-set recovery averages at least
``READ_PASS`` and the write efficacy at least ``WRITE_PASS``; kill if recovery
averages below ``READ_KILL`` or efficacy below ``WRITE_KILL``; otherwise not passed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..envs.squad import HandCodedCentre, SquadEnv, run_episodes
from ..evaluation import mean_ci
from .learned_protocol import SEED_STRIDE
from .r7a_domains import CAUTIONS, COLOURS, build_registry
from .r7b_goal_conditioned import (
    BLIND,
    BOTH_GOAL,
    CENTRE_GOAL,
    CONDITION,
    HAND_GOAL,
    MODES,
    Goal,
    GoalArrays,
    GoalTrainSettings,
    _GoalAware,
    goal_sets,
    make_goal_spec,
    policies_for,
)
from .r7b_report import HELDOUT, SEEN
from .r7c_reliability import FLAG_BELOW

WINDOW = 30
FIT_SEED_BASE = 5_000_000_000
TEST_SEED_BASE = 5_500_000_000
N_SITUATIONS = 8
READ_PASS, READ_KILL = 0.90, 0.60
WRITE_PASS, WRITE_KILL = 0.50, 0.25
NO, EAT = 0, 1


@dataclass(frozen=True)
class LegibilitySettings:
    """Episode counts of one run's fit and test (``window`` is fixed by G12)."""

    window: int = WINDOW
    fit_per_goal: int = 40
    test_per_goal: int = 50
    n_random_writers: int = 10


# ------------------------------------------------------------------ episodes
class SymbolRecorder:
    """``run_episodes`` observer: symbols, situations and live receivers of the first
    ``window`` steps; ``situation`` always holds the current step's (for a writer)."""

    def __init__(self, window: int):
        self.window = window
        self.hand = HandCodedCentre()
        self.situation: np.ndarray | None = None
        self._m: list[np.ndarray] = []
        self._s: list[np.ndarray] = []
        self._a: list[np.ndarray] = []

    def __call__(self, step: int, obs: Any, sent: Any, done: np.ndarray) -> None:
        self.situation = self.hand.send(obs)[:, :, 0]
        if step >= self.window:
            return
        live = obs.centre["alive"] & ~done[:, None]
        symbols = np.zeros(live.shape, np.int64) if sent is None else sent[:, :, 0]
        self._m.append(np.asarray(symbols, dtype=np.int64))
        self._s.append(self.situation.astype(np.int64))
        self._a.append(np.asarray(live, dtype=bool))

    def stacked(self) -> dict[str, np.ndarray]:
        """(B, T, N) ``symbols``, ``situation`` and ``live``."""
        return {
            "symbols": np.stack(self._m, axis=1),
            "situation": np.stack(self._s, axis=1),
            "live": np.stack(self._a, axis=1),
        }


def episode_seeds(base: int, run: int, n: int) -> list[int]:
    if not 1 <= n <= SEED_STRIDE:
        raise ValueError(f"n must lie in [1, {SEED_STRIDE}]")
    return [base + run * SEED_STRIDE + i for i in range(n)]


def play(
    mode: str,
    settings: GoalTrainSettings,
    theta: np.ndarray,
    goals: Sequence[Goal],
    seeds: Sequence[int],
    window: int = WINDOW,
    writer: tuple[int, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Episodes ``seeds`` (episode b plays ``goals[b]``), optionally with ``writer`` =
    (receiver, table over situations) replacing that receiver's symbols every step."""
    if len(goals) != len(seeds):
        raise ValueError("one goal per episode seed")
    specialist, centre = policies_for(mode, settings, theta)
    arrays = GoalArrays.from_goals(goals)
    for policy in (specialist, centre):
        if isinstance(policy, _GoalAware):
            policy.set_goals(arrays)
    rec = SymbolRecorder(window)
    hook = None
    if writer is not None:
        receiver, table = writer

        def hook(messages: np.ndarray, step: int, mask: Any = None) -> np.ndarray:
            assert rec.situation is not None
            out = np.array(messages, copy=True)
            out[:, receiver, 0] = table[rec.situation[:, receiver]]
            return out

    env = SquadEnv(settings.squad_config())
    out = run_episodes(env, specialist, centre, list(seeds), hook, rec)
    return {
        **rec.stacked(),
        "eaters": arrays.eaters,
        "caution": arrays.caution,
        "food": np.asarray(out["food"], dtype=float),
    }


# ------------------------------------------------------------------ decoder
def symbol_counts(symbols: np.ndarray, live: np.ndarray, vocab: int) -> np.ndarray:
    """(B, T, N) symbols of live receivers -> (B, N, vocab) counts."""
    b, _, n = symbols.shape
    out = np.zeros((b, n, vocab))
    ep, _, recv = np.nonzero(live)
    np.add.at(out, (ep, recv, symbols[live]), 1)
    return out


@dataclass(frozen=True)
class Decoder:
    """One squad's dialect, fitted on its fit episodes."""

    status_llr: np.ndarray  # (V,) log P(symbol | eat) - log P(symbol | no)
    prior: float  # log-odds of eat among the fit receivers
    caution_logp: np.ndarray  # (C, V) log P(symbol | caution)
    caution_prior: np.ndarray  # (C,) log P(caution)
    writer: np.ndarray  # (2, S) symbol to send for (status, situation)

    def eaters(self, counts: np.ndarray) -> np.ndarray:
        """(B, N, V) counts -> (B, N) decoded eaters, never an empty set."""
        score = counts @ self.status_llr + self.prior
        pred = score > 0
        empty = ~pred.any(axis=1)
        pred[empty, score[empty].argmax(axis=1)] = True
        return pred

    def caution(self, counts: np.ndarray) -> np.ndarray:
        """(B, N, V) counts -> (B,) decoded caution index."""
        ll = counts.sum(axis=1) @ self.caution_logp.T + self.caution_prior
        return ll.argmax(axis=1)

    def message_balanced_accuracy(
        self, symbols: np.ndarray, live: np.ndarray, eaters: np.ndarray
    ) -> float:
        """Status from a single symbol: mean of the eat and do-not-eat recall."""
        truth = np.broadcast_to(eaters[:, None, :], symbols.shape)[live]
        pred = self.status_llr[symbols[live]] + self.prior > 0
        if truth.all() or not truth.any():
            return float("nan")
        return float((pred[truth].mean() + (~pred[~truth]).mean()) / 2)


def fit_decoder(
    rec: dict[str, np.ndarray],
    vocab: int,
    shuffle: np.random.Generator | None = None,
) -> Decoder:
    """Fit on recorded episodes; ``shuffle`` permutes the goals across episodes first
    (the read null)."""
    eaters, caution = rec["eaters"], rec["caution"]
    if shuffle is not None:
        order = shuffle.permutation(len(eaters))
        eaters, caution = eaters[order], caution[order]
    counts = symbol_counts(rec["symbols"], rec["live"], vocab)
    pe = counts[eaters].sum(axis=0) + 1.0
    pn = counts[~eaters].sum(axis=0) + 1.0
    llr = np.log(pe / pe.sum()) - np.log(pn / pn.sum())
    share = float(np.clip(eaters.mean(), 1e-3, 1 - 1e-3))
    per_ep = counts.sum(axis=1)
    n_c = len(CAUTIONS)
    pc = np.stack([per_ep[caution == k].sum(axis=0) + 1.0 for k in range(n_c)])
    freq = np.bincount(caution, minlength=n_c) + 1.0
    status = np.broadcast_to(eaters[:, None, :], rec["symbols"].shape)
    live = rec["live"]
    table = np.ones((2, N_SITUATIONS, vocab))
    np.add.at(
        table,
        (status[live].astype(np.int64), rec["situation"][live], rec["symbols"][live]),
        1,
    )
    return Decoder(
        status_llr=llr,
        prior=float(np.log(share / (1 - share))),
        caution_logp=np.log(pc / pc.sum(axis=1, keepdims=True)),
        caution_prior=np.log(freq / freq.sum()),
        writer=table.argmax(axis=2).astype(np.int64),
    )


def read_scores(dec: Decoder, rec: dict[str, np.ndarray], vocab: int) -> dict:
    counts = symbol_counts(rec["symbols"], rec["live"], vocab)
    pred = dec.eaters(counts)
    exact = (pred == rec["eaters"]).all(axis=1)
    caution = dec.caution(counts) == rec["caution"]
    return {
        "eaters_exact": float(exact.mean()),
        "receiver_acc": float((pred == rec["eaters"]).mean()),
        "message_bal_acc": dec.message_balanced_accuracy(
            rec["symbols"], rec["live"], rec["eaters"]
        ),
        "caution_acc": float(caution.mean()),
        "spec_exact": float((exact & caution).mean()),
    }


def decoded_specs(
    dec: Decoder, rec: dict[str, np.ndarray], vocab: int
) -> list[dict[str, Any]]:
    """The decoded goals as validated ``squad.mission`` GoalSpec params."""
    counts = symbol_counts(rec["symbols"], rec["live"], vocab)
    registry = build_registry()
    out = []
    for eat, c in zip(dec.eaters(counts), dec.caution(counts)):
        goal = Goal(tuple(col for col, e in zip(COLOURS, eat) if e), CAUTIONS[c])
        out.append(make_goal_spec(goal, registry).params)
    return out


# ------------------------------------------------------------------ write test
def write_efficacy(
    foods: dict[str, np.ndarray], intact: dict[str, np.ndarray]
) -> dict[str, float]:
    """``foods[arm]``: (N,) mean food of each steered receiver under the eat and no
    writers; ``intact`` the unsteered episodes."""
    f, e = intact["food"], intact["eaters"]
    gap = float(f[e].mean() - f[~e].mean())
    eat, no = float(np.mean(foods["eat"])), float(np.mean(foods["no"]))
    return {
        "food_eat": eat,
        "food_no": no,
        "intact_eater": float(f[e].mean()),
        "intact_non": float(f[~e].mean()),
        "efficacy": (eat - no) / gap if gap > 0 else float("nan"),
    }


def steer(
    mode: str,
    settings: GoalTrainSettings,
    theta: np.ndarray,
    goals: Sequence[Goal],
    seeds: Sequence[int],
    tables: np.ndarray,
) -> dict[str, np.ndarray]:
    """(N,) mean food of each receiver steered with ``tables[EAT]`` and ``tables[NO]``."""
    n = settings.squad_config().n_snakes
    out: dict[str, list[float]] = {"eat": [], "no": []}
    for arm, row in (("eat", EAT), ("no", NO)):
        for r in range(n):
            got = play(mode, settings, theta, goals, seeds, 1, (r, tables[row]))
            out[arm].append(float(got["food"][:, r].mean()))
    return {k: np.asarray(v) for k, v in out.items()}


# ------------------------------------------------------------------ one run
def evaluate_run(job: tuple[dict, dict[str, Any], LegibilitySettings]) -> dict:
    """Fit, read and write for one trained run (``read_trained`` row)."""
    t, settings_d, ls = job
    settings = GoalTrainSettings(**settings_d)
    vocab = settings.squad_config().message.vocab_size
    mode, run, theta = t["mode"], int(t["run"]), t["theta"]
    fit_goals = [g for g in goal_sets()[SEEN] for _ in range(ls.fit_per_goal)]
    test_goals = [g for g in goal_sets()[HELDOUT] for _ in range(ls.test_per_goal)]
    fit_seeds = episode_seeds(FIT_SEED_BASE, run, len(fit_goals))
    test_seeds = episode_seeds(TEST_SEED_BASE, run, len(test_goals))
    fit = play(mode, settings, theta, fit_goals, fit_seeds, ls.window)
    test = play(mode, settings, theta, test_goals, test_seeds, ls.window)
    dec = fit_decoder(fit, vocab)
    null = fit_decoder(fit, vocab, shuffle=np.random.default_rng([run, 7]))
    write = write_efficacy(
        steer(mode, settings, theta, test_goals, test_seeds, dec.writer), test
    )
    rng = np.random.default_rng([run, 11])
    random_eff = [
        write_efficacy(
            steer(
                mode, settings, theta, test_goals, test_seeds,
                rng.integers(0, vocab, size=(2, N_SITUATIONS)),
            ),
            test,
        )["efficacy"]
        for _ in range(ls.n_random_writers)
    ]  # fmt: skip
    return {
        "mode": mode,
        "run": run,
        "best_val": float(t["best_val_mission"]),
        "read": read_scores(dec, test, vocab),
        "read_null": read_scores(null, test, vocab)["eaters_exact"],
        "write": write,
        "write_null": float(np.mean(random_eff)) if random_eff else float("nan"),
        "writer": dec.writer.tolist(),
        "decoded_specs": decoded_specs(dec, test, vocab)[:: ls.test_per_goal],
        # kept in memory for cross-run transfer, dropped from the JSON
        "_decoder": dec,
        "_test_counts": symbol_counts(test["symbols"], test["live"], vocab),
        "_test_eaters": test["eaters"],
    }


# ------------------------------------------------------------------ report
def primary(rows: Sequence[dict]) -> list[dict]:
    """The G12 sample: (b) runs with best training-goal validation >= ``FLAG_BELOW``."""
    return [r for r in rows if r["mode"] == CENTRE_GOAL and r["best_val"] >= FLAG_BELOW]


def g12_verdict(read: Sequence[float], write: Sequence[float]) -> dict[str, Any]:
    r, w = float(np.mean(read)), float(np.mean(write))
    if r >= READ_PASS and w >= WRITE_PASS:
        verdict = "PASS"
    elif r < READ_KILL or w < WRITE_KILL:
        verdict = "KILL"
    else:
        verdict = "NOT PASSED"
    return {
        "verdict": verdict,
        "read": r,
        "read_ci": mean_ci(list(read))[1],
        "write": w,
        "write_ci": mean_ci(list(write))[1],
        "n": len(read),
    }


def transfer(rows: Sequence[dict]) -> dict[str, float]:
    """Eater-set recovery when run i's decoder reads run j's held-out symbols."""
    k = len(rows)
    m = np.full((k, k), np.nan)
    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            pred = a["_decoder"].eaters(b["_test_counts"])
            m[i, j] = (pred == b["_test_eaters"]).all(axis=1).mean()
    off = m[~np.eye(k, dtype=bool)]
    return {
        "own": float(np.mean(np.diag(m))),
        "other": float(np.mean(off)) if off.size else float("nan"),
        "other_max": float(np.max(off)) if off.size else float("nan"),
        "distinct_writers": len({json.dumps(r["writer"]) for r in rows}),
        "n": k,
    }


def _summary(rows: Sequence[dict]) -> dict[str, Any]:
    def col(f: Any) -> list[float]:
        return [float(f(r)) for r in rows]

    cols = {
        "eaters_exact": col(lambda r: r["read"]["eaters_exact"]),
        "read_null": col(lambda r: r["read_null"]),
        "message_bal_acc": col(lambda r: r["read"]["message_bal_acc"]),
        "caution_acc": col(lambda r: r["read"]["caution_acc"]),
        "spec_exact": col(lambda r: r["read"]["spec_exact"]),
        "efficacy": col(lambda r: r["write"]["efficacy"]),
        "write_null": col(lambda r: r["write_null"]),
        "food_eat": col(lambda r: r["write"]["food_eat"]),
        "food_no": col(lambda r: r["write"]["food_no"]),
        "intact_eater": col(lambda r: r["write"]["intact_eater"]),
        "intact_non": col(lambda r: r["write"]["intact_non"]),
    }
    return {
        k: {"mean": mean_ci(v)[0], "ci": mean_ci(v)[1], "values": v}
        for k, v in cols.items()
    }


def build_report(
    rows: Sequence[dict], settings_d: dict[str, Any], ls: LegibilitySettings
) -> dict[str, Any]:
    prim = primary(rows)
    if not prim:
        raise ValueError("no (b) run passes the G12 sample rule")
    g12 = g12_verdict(
        [r["read"]["eaters_exact"] for r in prim],
        [r["write"]["efficacy"] for r in prim],
    )
    groups: dict[str, list[dict]] = {
        f"(b), G12 sample (n = {len(prim)})": prim,
        "(b), all runs": [r for r in rows if r["mode"] == CENTRE_GOAL],
    }
    for mode in (BOTH_GOAL, HAND_GOAL, BLIND):
        got = [r for r in rows if r["mode"] == mode]
        if got:
            groups[CONDITION[mode]] = got
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    return {
        "g12": g12,
        "rule": {
            "window": ls.window,
            "flag_below": FLAG_BELOW,
            "read_pass": READ_PASS,
            "read_kill": READ_KILL,
            "write_pass": WRITE_PASS,
            "write_kill": WRITE_KILL,
        },
        "groups": {k: _summary(v) for k, v in groups.items()},
        "group_n": {k: len(v) for k, v in groups.items()},
        "transfer": transfer(prim),
        "runs": sorted(clean, key=lambda r: (MODES.index(r["mode"]), r["run"])),
        "settings": settings_d,
        "legibility": asdict(ls),
        "goals": {w: [g.label() for g in goal_sets()[w]] for w in (SEEN, HELDOUT)},
    }


def _pm(s: dict[str, Any], scale: float = 1.0, digits: int = 1) -> str:
    return f"{scale * s['mean']:.{digits}f} ± {scale * s['ci']:.{digits}f}"


def render(rep: dict[str, Any]) -> str:
    g = rep["g12"]
    ls = rep["legibility"]
    lines = [
        "# R7d: can a monitor read and write the squad's learned language? (goal G12)",
        "",
        f"**G12 verdict: {g['verdict']}.** On held-out goals, per-squad decoders fitted on "
        f"training-goal episodes recover the exact eater set in {100 * g['read']:.1f}% "
        f"± {100 * g['read_ci']:.1f} of episodes (pass ≥ {100 * READ_PASS:.0f}%, kill < "
        f"{100 * READ_KILL:.0f}%), and the decoded writer steers a snake with efficacy "
        f"{g['write']:.2f} ± {g['write_ci']:.2f} (pass ≥ {WRITE_PASS:.2f}, kill < "
        f"{WRITE_KILL:.2f}); n = {g['n']} (b) runs with best training-goal validation "
        f"≥ {FLAG_BELOW:g}.",
        "",
        f"Decoders read the first {ls['window']} steps of each live receiver's symbols; "
        f"fit: {ls['fit_per_goal']} episodes per training goal; test: "
        f"{ls['test_per_goal']} per held-out goal; {ls['n_random_writers']} random writer "
        "pairs per run for the null. Means ± 95% t-CI over training runs.",
        "",
        "## Read and write, held-out goals",
        "",
        "| Runs | n | Eater set exact | Shuffled-label null | Single symbol (bal. acc.) "
        "| Caution | Whole spec | Write efficacy | Random writers |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in rep["groups"].items():
        lines.append(
            f"| {name} | {rep['group_n'][name]} | {_pm(s['eaters_exact'], 100)}% "
            f"| {_pm(s['read_null'], 100)}% | {_pm(s['message_bal_acc'], 100)}% "
            f"| {_pm(s['caution_acc'], 100)}% | {_pm(s['spec_exact'], 100)}% "
            f"| {_pm(s['efficacy'], 1, 2)} | {_pm(s['write_null'], 1, 2)} |"
        )
    lines += [
        "",
        "Food of the steered snake (mean per episode) against the intact squad:",
        "",
        '| Runs | Written "eat" | Written "do not eat" | Intact eaters | Intact non-eaters |',
        "|---|---|---|---|---|",
    ]
    for name, s in rep["groups"].items():
        lines.append(
            f"| {name} | {_pm(s['food_eat'], 1, 2)} | {_pm(s['food_no'], 1, 2)} "
            f"| {_pm(s['intact_eater'], 1, 2)} | {_pm(s['intact_non'], 1, 2)} |"
        )
    t = rep["transfer"]
    lines += [
        "",
        "## Dialects",
        "",
        f"Among the {t['n']} sample squads there are {t['distinct_writers']} distinct writer "
        f"tables. A squad's own decoder recovers {100 * t['own']:.1f}% of its held-out eater "
        f"sets; another squad's decoder {100 * t['other']:.1f}% on average (best pair "
        f"{100 * t['other_max']:.1f}%).",
        "",
        "## Runs",
        "",
        "| Condition | Run | Best validation | Eater set exact | Write efficacy | Writer, eat (situations 0-7) | Writer, do not eat |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rep["runs"]:
        eat, no = r["writer"][EAT], r["writer"][NO]
        lines.append(
            f"| {CONDITION[r['mode']]} | {r['run']} | {r['best_val']:.1f} "
            f"| {100 * r['read']['eaters_exact']:.1f}% | {r['write']['efficacy']:.2f} "
            f"| {''.join(map(str, eat))} | {''.join(map(str, no))} |"
        )
    lines += [
        "",
        "Situation s = 2 x direction + far of the receiver's nearest own-colour food "
        "(R6's hand code, `HandCodedCentre`).",
        "",
        f"Training settings: `{json.dumps(rep['settings'])}`.",
        "",
        "Reproduce: `python scripts/r7d_legibility.py --from-json "
        "runs:r7b3-train-* runs:r7b4-train-* runs:r7b5-train-* runs:r7c-train-*`.",
        "",
    ]
    return "\n".join(lines)
