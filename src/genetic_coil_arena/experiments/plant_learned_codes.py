"""Learned codes against hand-designed encodings on the plant floor (R8 step 3, goal G2).

Glue between ``envs.plant`` and the step-2 replicate machinery. For each learned method
(``envs.plant.learned_codes``: end-to-end, residual VQ, local classifier) and byte budget
it

1. chooses the method's hyperparameters on the tuning block only (``tune_hp``; the centre's
   own settings are fixed to the ones step 2 chose on that block for the hand features);
2. on each of the 10 run-level replicates trains the encoder on the training runs, trains
   and calibrates the centre exactly as in step 2 (5% false alarms on validation runs) and
   replays the test runs (``run_cell``), plus two ablations: one node's message zeroed, and
   messages shuffled across runs;
3. pairs each learned score with the best hand design of step 2 per replicate and applies
   the G2 rule (``g2_tests``, ``g2_verdict``).

Everything that is not training is pure: merging partial results (``merge_partials``), the
statistics and the report (``build_report``) work on plain dicts, so partial runs on
different machines can be combined.
"""

import dataclasses
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy import stats as scipy_stats

from ..envs.plant.centres import DecodedCentre
from ..envs.plant.encoders import UplinkEncoder
from ..envs.plant.learned_codes import (
    DEFAULT_HP,
    METHODS,
    Featuriser,
    fit_encoder,
)
from ..envs.plant.replicates import Prepared, Protocol, Replicate, summarise
from ..envs.plant.rieth import RiethData
from ..envs.plant.tep import NODES
from ..evaluation.stats import holm_correction

G2_BUDGETS = (1, 2, 4)
REPORT_BUDGETS = (1, 2, 4, 8)
HAND = ("alarms", "summary", "features")
METHOD_LABELS = {
    "e2e": "(L1) end-to-end encoder + centre, binary bottleneck",
    "vq": "(L2) residual k-means codebook (unsupervised)",
    "local": "(L3) local classifier, quantised top posteriors",
}
HAND_LABELS = {
    "alarms": "(a) threshold alarms (1 B alarm byte per node)",
    "summary": "(b) quantised summary stats",
    "features": "(c) quantised hand features (PCA/FFT)",
    "oracle": "(e) oracle (all raw variables, no byte limit)",
}
VARIANTS = ("base", *(f"zero_{n}" for n in NODES), "shuffle")

# Step 2's centre for the hand features, chosen on the tuning block (tuning.tune_centre)
CENTRE_FULL: dict[str, Any] = {
    "classifier": "mlp",
    "hidden": 64,
    "l2": 1e-3,
    "max_iter": 200,
    "context": 6,
    "precision": "float32",
}


@dataclass(frozen=True)
class Scale:
    """How much work a run does: the centre, the hyperparameter grids and the overrides."""

    name: str
    centre: dict[str, Any]
    grids: dict[str, tuple[tuple[str, tuple[Any, ...]], ...]]
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


SCALES = {
    "full": Scale(
        "full",
        CENTRE_FULL,
        {
            "e2e": (
                ("bottleneck", ("ste", "gumbel")),
                ("hidden", (64, 128, 256)),
                ("epochs", (20, 40, 80)),
                ("lr", (2e-3, 1e-3, 5e-3)),
            ),
            "vq": (("whiten", (False, True)), ("dim", (8, 16, 32))),
            "local": (
                ("hidden", (32, 64, 128)),
                ("l2", (1e-3, 1e-2, 1e-1)),
                ("max_iter", (60, 100, 200)),
            ),
        },
    ),
    "smoke": Scale(
        "smoke",
        {**CENTRE_FULL, "hidden": 32, "max_iter": 40, "context": 2},
        {
            "e2e": (("bottleneck", ("ste", "gumbel")),),
            "vq": (("whiten", (False, True)),),
            "local": (("hidden", (32, 64)),),
        },
        {
            "e2e": {"epochs": 6},
            "vq": {},
            "local": {"max_iter": 40, "folds": 2},
        },
    ),
}
TOLERANCE = 0.002


# ---------------------------------------------------------------------------- one cell
def make_prepared(
    rep: Replicate,
    encoder: UplinkEncoder,
    x_train: Optional[np.ndarray],
    seed: int,
    method: str,
) -> Prepared:
    """A ``Prepared`` cell around a fitted learned encoder (the centre reads its decoding)."""
    n_classes, target = rep.protocol.n_classes, rep.protocol.target_far

    def make_centre(params: dict[str, Any]) -> DecodedCentre:
        rest = {k: v for k, v in params.items() if k != "classifier"}
        return DecodedCentre(
            encoder, n_classes, params["classifier"], seed, target_far=target, **rest
        )

    prep = Prepared(method, encoder.budget, seed, encoder, make_centre)
    prep.x_train = (
        x_train
        if x_train is not None
        else encoder.decode(encoder.encode_windows(rep.train["windows"]))
    )
    prep.x_val = encoder.decode(encoder.encode_windows(rep.val["windows"]))
    return prep


def featuriser_of(rep: Replicate) -> Featuriser:
    """The replicate's local-feature standardiser, fitted once on its training windows."""
    cached = getattr(rep, "_featuriser", None)
    if cached is None:
        cached = Featuriser(rep.layout).fit(rep.train["windows"])
        rep._featuriser = cached  # type: ignore[attr-defined]
    return cached


def build_cell(
    rep: Replicate,
    method: str,
    budget: int,
    hp: dict[str, Any],
    seed: int,
    scale: Scale,
) -> Prepared:
    """Fit the encoder on a replicate's training runs and wrap it for the centre."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected {METHODS}")
    hp = {**DEFAULT_HP[method], **scale.overrides.get(method, {}), **hp}
    enc, x_train = fit_encoder(
        method,
        budget,
        rep.layout,
        featuriser_of(rep),
        rep.train,
        rep.protocol.n_classes,
        hp,
        seed,
    )
    return make_prepared(rep, enc, x_train, seed, method)


def zero_node(node: int) -> Callable[..., np.ndarray]:
    """Hook: node ``node`` sends all-zero bytes."""

    def hook(messages: np.ndarray, step: int, mask: Any = None) -> np.ndarray:
        out = messages.copy()
        out[:, node, :] = 0
        return out

    return hook


def shuffle_runs(seed: int = 0) -> Callable[..., np.ndarray]:
    """Hook: every episode receives the messages of another episode (a fixed derangement)."""
    cache: dict[int, np.ndarray] = {}

    def hook(messages: np.ndarray, step: int, mask: Any = None) -> np.ndarray:
        n = messages.shape[0]
        if n not in cache:
            perm = np.random.default_rng(seed).permutation(n)
            source = np.empty(n, np.int64)
            source[perm] = np.roll(perm, 1) if n > 1 else perm
            cache[n] = source
        return messages[cache[n]]

    return hook


def ablation_hooks(seed: int = 0) -> dict[str, Callable[..., np.ndarray]]:
    hooks = {f"zero_{node}": zero_node(i) for i, node in enumerate(NODES)}
    hooks["shuffle"] = shuffle_runs(seed)
    return hooks


def run_cell(
    rep: Replicate,
    method: str,
    budget: int,
    hp: dict[str, Any],
    seed_index: int,
    scale: Scale,
    ablate: bool,
) -> list[dict[str, Any]]:
    """Train one cell on a replicate and return its rows (base, plus ablations if asked)."""
    start = time.perf_counter()
    seed = 100 * int(rep.index or 0) + seed_index
    prep = build_cell(rep, method, budget, hp, seed, scale)
    centre = rep.fit_centre(prep, scale.centre)
    assert prep.x_val is not None and prep.encoder is not None
    sample = prep.encoder.encode_windows(rep.val["windows"][:256])
    variants: dict[str, Optional[Callable[..., np.ndarray]]] = {"base": None}
    if ablate:
        variants.update(ablation_hooks(seed))
    rows = []
    for variant, hook in variants.items():
        out = rep.evaluate(
            method, budget, seed, scale.centre, prep=prep, centre=centre, hook=hook
        )
        out.update(
            method=method,
            variant=variant,
            seed_index=seed_index,
            hp=dict(hp),
            msg_bytes=int(sample.shape[-1]),
            max_symbol=int(sample.max()),
        )
        rows.append(out)
    rows[0]["seconds"] = time.perf_counter() - start
    return rows


def tune_hp(
    rep: Replicate,
    method: str,
    budget: int,
    scale: Scale,
    seed: int = 0,
    tol: float = TOLERANCE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Staged coordinate search of a method's hyperparameters on the tuning block.

    The score is the validation macro-F1 at the centre's calibrated threshold, as in
    ``tuning.tune_centre``; among candidates within ``tol`` the earliest listed wins.
    """
    best = {**DEFAULT_HP[method], **scale.overrides.get(method, {})}
    trace: list[dict[str, Any]] = []
    current: Optional[float] = None
    for name, values in scale.grids[method]:
        scores: list[tuple[Any, float]] = []
        for value in values:
            if current is not None and value == best[name]:
                scores.append((value, current))
                continue
            hp = {**best, name: value}
            t0 = time.perf_counter()
            prep = build_cell(rep, method, budget, hp, seed, scale)
            centre = rep.fit_centre(prep, scale.centre)
            score = rep.validation_f1(centre, prep)
            trace.append(
                {
                    "stage": name,
                    "value": value,
                    "val_f1": float(score),
                    "seconds": time.perf_counter() - t0,
                }
            )
            scores.append((value, score))
            if current is None and value == best[name]:
                current = score
        top = max(s for _, s in scores)
        pick = next(v for v, s in scores if s >= top - tol)
        best[name] = pick
        current = dict(scores)[pick]
    return best, trace


# ------------------------------------------------------------------------------ workers
_STATE: dict[str, Any] = {}


def init_worker(
    cache_dir: Optional[str],
    protocol: dict,
    scale: str,
    torch_threads: int,
    methods: Sequence[str],
) -> None:
    """Process initialiser: load the data once, cap torch's threads."""
    if "e2e" in methods:
        import torch

        torch.set_num_threads(max(1, torch_threads))
    _STATE["data"] = RiethData.from_cache(Path(cache_dir) if cache_dir else None)
    _STATE["protocol"] = Protocol(**protocol)
    _STATE["scale"] = SCALES[scale]
    _STATE["replicate"] = None


def _replicate(index: Optional[int]) -> Replicate:
    cached = _STATE.get("replicate")
    if cached is None or cached[0] != index:
        _STATE["replicate"] = None
        _STATE["replicate"] = (
            index,
            Replicate(_STATE["data"], _STATE["protocol"], index),
        )
    return _STATE["replicate"][1]


def tune_task(task: tuple[str, int]) -> dict[str, Any]:
    method, budget = task
    start = time.perf_counter()
    hp, trace = tune_hp(_replicate(None), method, budget, _STATE["scale"])
    return {
        "method": method,
        "budget": budget,
        "hp": hp,
        "trace": trace,
        "seconds": time.perf_counter() - start,
    }


def cell_task(task: tuple[str, int, int, int, dict, bool]) -> list[dict[str, Any]]:
    method, budget, replicate, seed_index, hp, ablate = task
    return run_cell(
        _replicate(replicate), method, budget, hp, seed_index, _STATE["scale"], ablate
    )


def run_groups(
    groups: Sequence[tuple[str, int]],
    replicates: Sequence[int],
    seeds: int,
    ablate_budgets: Iterable[int],
    workers: int,
    protocol: dict,
    scale: str,
    cache_dir: Optional[str],
    torch_threads: int,
    on_group_done: Callable[[dict[str, Any]], None],
    log: Callable[[str], None] = print,
    tuned: Optional[dict[tuple[str, int], dict[str, Any]]] = None,
) -> None:
    """Tune and evaluate (method, budget) groups on one process pool.

    A group's cells start as soon as its tuning is done; ablations run for encoder seed 0 only; ``on_group_done`` gets the group's
    ``{"method", "budget", "tuning", "rows", "seconds"}`` when its last cell finishes.
    """
    ablate_set = set(ablate_budgets)
    methods = sorted({m for m, _ in groups})
    waiting = {g: len(replicates) * seeds for g in groups}
    state: dict[tuple[str, int], dict[str, Any]] = {
        g: {"rows": [], "tuning": (tuned or {}).get(g)} for g in groups
    }
    started = {g: time.perf_counter() for g in groups}
    total = sum(waiting.values())
    done_cells = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(cache_dir, protocol, scale, torch_threads, methods),
    ) as pool:
        pending: dict[Any, tuple[str, tuple[str, int]]] = {}

        def submit_cells(group: tuple[str, int], hp: dict[str, Any]) -> None:
            method, budget = group
            for r in replicates:
                for s in range(seeds):
                    task = (method, budget, r, s, hp, budget in ablate_set and s == 0)
                    pending[pool.submit(cell_task, task)] = ("cell", group)

        for group in groups:
            if state[group]["tuning"] is None:
                pending[pool.submit(tune_task, group)] = ("tune", group)
            else:
                submit_cells(group, state[group]["tuning"]["hp"])
        log(f"{total} cells in {len(groups)} groups on {workers} workers")
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                kind, group = pending.pop(future)
                result = future.result()
                if kind == "tune":
                    state[group]["tuning"] = result
                    best = max([t["val_f1"] for t in result["trace"]] or [float("nan")])
                    log(
                        f"  tuned {group}: {result['hp']} val F1 {best:.3f} "
                        f"({result['seconds']:.0f}s)"
                    )
                    submit_cells(group, result["hp"])
                    continue
                state[group]["rows"].extend(result)
                done_cells += 1
                waiting[group] -= 1
                if done_cells % 5 == 0 or done_cells == total:
                    log(
                        f"  {done_cells}/{total} cells, {time.perf_counter() - t0:.0f}s"
                    )
                if waiting[group] == 0:
                    on_group_done(
                        {
                            "method": group[0],
                            "budget": group[1],
                            "tuning": state[group]["tuning"],
                            "rows": state[group]["rows"],
                            "seconds": time.perf_counter() - started[group],
                        }
                    )


# ------------------------------------------------------------------------------- merging
def merge_partials(partials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine partial result files into one result.

    Rows are keyed by (method, budget, replicate, seed_index, variant); a later partial
    replaces an earlier one with the same key. All partials must share the protocol and the
    scale.

    Raises:
        ValueError: If there are no partials or they disagree on protocol or scale
    """
    if not partials:
        raise ValueError("Nothing to merge")
    first = partials[0]
    for p in partials[1:]:
        if p["protocol"] != first["protocol"] or p["scale"] != first["scale"]:
            raise ValueError("Partial results come from different protocols or scales")
    rows: dict[tuple, dict[str, Any]] = {}
    tuning: dict[str, Any] = {}
    cpu = 0.0
    commits = sorted({str(p.get("commit")) for p in partials})
    for p in partials:
        cpu += float(p.get("seconds", 0.0))
        for r in p["rows"]:
            key = (
                r["method"],
                int(r["budget"]),
                int(r["replicate"]),
                int(r["seed_index"]),
                r["variant"],
            )
            rows[key] = r
        t = p.get("tuning")
        if t:
            tuning[f"{t['method']}|{t['budget']}"] = t
    return {
        "protocol": first["protocol"],
        "scale": first["scale"],
        "commits": commits,
        "partials": len(partials),
        "cpu_seconds": cpu,
        "tuning": tuning,
        "rows": [rows[k] for k in sorted(rows)],
    }


def normalise_baseline(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """Step 2's per-cell rows (17 faults, tuned centre) in the learned rows' shape.

    The 1 B alarm byte counts at every budget; the oracle has budget 0.
    """
    out = []
    for r in baseline["rows"]:
        if r.get("fault_set") != "diagnosable" or r.get("variant") != "tuned":
            continue
        enc, budget = r["encoding"], int(r["budget"])
        if enc == "alarms":
            budgets = REPORT_BUDGETS
        elif enc in ("summary", "features"):
            budgets = (budget,)
        elif enc == "oracle":
            budgets = (0,)
        else:
            continue
        for b in budgets:
            out.append({**r, "method": enc, "budget": b, "variant": "base"})
    return out


# ---------------------------------------------------------------------------- statistics
def by_replicate(
    rows: Sequence[dict[str, Any]],
    method: str,
    budget: int,
    metric: str,
    variant: str = "base",
) -> dict[int, float]:
    """Replicate -> metric, averaged over a replicate's encoder seeds (NaN skipped)."""
    acc: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if (
            r["method"] == method
            and int(r["budget"]) == budget
            and r["variant"] == variant
            and r.get(metric) is not None
            and np.isfinite(r[metric])
        ):
            acc[int(r["replicate"])].append(float(r[metric]))
    return {k: float(np.mean(v)) for k, v in sorted(acc.items())}


def best_hand(
    baseline_rows: Sequence[dict[str, Any]], budget: int, metric: str
) -> Optional[tuple[str, dict[int, float]]]:
    """The hand design (a, b or c) with the best mean ``metric`` at a budget."""
    options = {enc: by_replicate(baseline_rows, enc, budget, metric) for enc in HAND}
    options = {e: v for e, v in options.items() if v}
    if not options:
        return None
    enc = max(options, key=lambda e: float(np.mean(list(options[e].values()))))
    return enc, options[enc]


def paired(a: dict[int, float], b: dict[int, float]) -> np.ndarray:
    """Differences ``a - b`` over the replicates both have."""
    common = sorted(set(a) & set(b))
    return np.array([a[k] - b[k] for k in common], dtype=float)


def g2_tests(
    rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    metric: str = "macro_f1",
    budgets: Sequence[int] = G2_BUDGETS,
    methods: Sequence[str] = METHODS,
) -> list[dict[str, Any]]:
    """Each learned method against the best hand design, paired by replicate.

    A test ``win`` needs the 95% t-interval of the mean difference to lie above zero. The
    Holm-adjusted p-value (family: all returned tests) is reported next to it.
    """
    tests = []
    for budget in budgets:
        bar = best_hand(baseline_rows, budget, metric)
        for method in methods:
            learned = by_replicate(rows, method, budget, metric)
            if bar is None or not learned:
                continue
            enc, hand = bar
            diff = paired(learned, hand)
            if diff.size == 0:
                continue
            mean, half = summarise(diff)
            if diff.size > 1 and np.ptp(diff) > 0:
                p = float(scipy_stats.ttest_1samp(diff, 0.0).pvalue)
            else:
                p = 1.0
            tests.append(
                {
                    "method": method,
                    "budget": budget,
                    "n": int(diff.size),
                    "learned": float(np.mean([learned[k] for k in learned])),
                    "hand": enc,
                    "hand_mean": float(np.mean(list(hand.values()))),
                    "diff": mean,
                    "half": half,
                    "p": p,
                    "win": bool(np.isfinite(half) and mean - half > 0),
                }
            )
    if tests:
        adjusted = holm_correction([t["p"] for t in tests])
        for t, a in zip(tests, adjusted):
            t["p_holm"] = float(a)
            t["win_holm"] = bool(t["win"] and a < 0.05)
    return tests


def g2_verdict(
    tests: Sequence[dict[str, Any]],
    budgets: Sequence[int] = G2_BUDGETS,
    methods: Sequence[str] = METHODS,
    n_required: int = 10,
) -> tuple[str, str]:
    """Apply the G2 rule: ``("PASS" | "KILL" | "INCOMPLETE", explanation)``.

    PASS: at least one (method, budget in 1/2/4 B) beats the best hand design with the 95%
    CI of the paired difference over ``n_required`` replicates above zero. KILL: every
    (method, budget) was tested on ``n_required`` replicates and none did. Otherwise the
    evidence is incomplete.
    """
    full = [t for t in tests if t["budget"] in budgets and t["n"] >= n_required]
    wins = [t for t in full if t["win"]]
    if wins:
        text = "; ".join(
            f"{t['method']} at {t['budget']} B: {t['diff']:+.3f} ± {t['half']:.3f} "
            f"over {t['hand']}"
            for t in wins
        )
        return "PASS", f"A learned code beats the best hand design: {text}."
    seen = {(t["method"], t["budget"]) for t in full}
    missing = [(m, b) for m in methods for b in budgets if (m, b) not in seen]
    if missing:
        return (
            "INCOMPLETE",
            f"No significant win yet, but {len(missing)} of {len(methods) * len(budgets)} "
            f"(method, budget) cells lack {n_required} replicates.",
        )
    return (
        "KILL",
        f"None of the {len(full)} (method, budget) comparisons at {', '.join(f'{b} B' for b in budgets)} "
        f"has a paired 95% CI above zero over {n_required} replicates.",
    )


# ------------------------------------------------------------------------------- report
def _fmt(values: Iterable[float], digits: int = 3) -> str:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return "-"
    mean, half = summarise(arr)
    if np.isfinite(half):
        return f"{mean:.{digits}f} ± {half:.{digits}f}"
    return f"{mean:.{digits}f}"


def _cell(rows, method, budget, metric, variant="base", digits=3) -> str:
    return _fmt(by_replicate(rows, method, budget, metric, variant).values(), digits)


def _metric_table(
    rows, baseline, metric: str, digits: int, budgets: Sequence[int]
) -> list[str]:
    head = "| Encoding | " + " | ".join(f"{b} B" for b in budgets) + " |"
    lines = [head, "|---|" + "---|" * len(budgets)]
    for enc in HAND:
        cells = [_cell(baseline, enc, b, metric, digits=digits) for b in budgets]
        lines.append(f"| {HAND_LABELS[enc]} | {' | '.join(cells)} |")
    for m in METHODS:
        cells = [_cell(rows, m, b, metric, digits=digits) for b in budgets]
        lines.append(f"| {METHOD_LABELS[m]} | {' | '.join(cells)} |")
    cells = [_cell(baseline, "oracle", 0, metric, digits=digits)] + [""] * (
        len(budgets) - 1
    )
    lines.append(f"| {HAND_LABELS['oracle']} | {' | '.join(cells)} |")
    return lines + [""]


def _test_table(tests: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "| Method | Budget | Learned | Best hand design | Difference (learned - hand) | p | p (Holm) | n | 95% CI excludes 0 above |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in tests:
        lines.append(
            f"| {METHOD_LABELS[t['method']]} | {t['budget']} B | {t['learned']:.3f} | "
            f"({t['hand']}) {t['hand_mean']:.3f} | {t['diff']:+.3f} ± {t['half']:.3f} | "
            f"{t['p']:.3g} | {t['p_holm']:.3g} | {t['n']} | {'yes' if t['win'] else 'no'} |"
        )
    return lines + [""]


def merged_verdict(
    merged: dict[str, Any], tests: Sequence[dict[str, Any]]
) -> tuple[str, str]:
    """``g2_verdict`` for a merged result; a reduced-scale run can only ever be SMOKE."""
    verdict, why = g2_verdict(tests, n_required=int(merged["protocol"]["n_replicates"]))
    if merged["scale"] != "full":
        return (
            "SMOKE",
            f"Reduced-scale pipeline check, not evidence (the rule would read {verdict}). {why}",
        )
    return verdict, why


def best_method(rows, budget: int) -> Optional[str]:
    """The learned method with the best mean calibrated macro-F1 at a budget."""
    scores = {
        m: np.mean(list(v.values()))
        for m in METHODS
        if (v := by_replicate(rows, m, budget, "macro_f1"))
    }
    return max(scores, key=lambda m: scores[m]) if scores else None


def _ablation_table(rows, method: str, budget: int) -> list[str]:
    lines = [
        "| Condition | Macro-F1 | Drop from intact | Test false alarms | Detection delay |",
        "|---|---|---|---|---|",
    ]
    base = by_replicate(rows, method, budget, "macro_f1")
    for variant in VARIANTS:
        f1 = by_replicate(rows, method, budget, "macro_f1", variant)
        if not f1:
            continue
        drop = summarise(-paired(f1, base)) if variant != "base" else (0.0, 0.0)
        label = {"base": "intact"}.get(
            variant,
            variant.replace("zero_", "node zeroed: ").replace(
                "shuffle", "messages shuffled across runs"
            ),
        )
        lines.append(
            f"| {label} | {_fmt(f1.values())} | "
            f"{'-' if variant == 'base' else f'{drop[0]:+.3f} ± {drop[1]:.3f}'} | "
            f"{_fmt(by_replicate(rows, method, budget, 'false_alarm_rate', variant).values())} | "
            f"{_fmt(by_replicate(rows, method, budget, 'detection_delay', variant).values(), 1)} |"
        )
    return lines + [""]


def build_report(
    merged: dict[str, Any], baseline: dict[str, Any], baseline_name: str
) -> str:
    """The Markdown report of a merged result against step 2's baseline."""
    rows = merged["rows"]
    base_rows = normalise_baseline(baseline)
    n_rep = len({r["replicate"] for r in rows if r["variant"] == "base"})
    smoke = merged["scale"] != "full"
    n_required = int(merged["protocol"]["n_replicates"])
    tests = g2_tests(rows, base_rows)
    matched = g2_tests(rows, base_rows, "macro_f1_matched")
    verdict, why = merged_verdict(merged, tests)
    lines = [
        "# Plant floor: learned codes against hand-designed encodings (R8, step 3, goal G2)",
        "",
    ]
    if smoke:
        lines += [
            f"> **{merged['scale'].upper()} RUN: a pipeline check on a reduced budget of "
            f"work, not evidence.** Do not read the numbers or the verdict below as the result of G2.",
            "",
        ]
    lines += [
        f"- Provenance: commits {', '.join(c[:9] for c in merged['commits'])}; scale `{merged['scale']}`; "
        f"{merged['partials']} partial result files; {merged['cpu_seconds'] / 60:.0f} CPU-minutes of "
        f"tuning and cell work; {n_rep} replicates with results.",
        f"- Protocol and data: those of step 2 (`{baseline_name}`): Rieth et al. 2017 runs, 17 diagnosable "
        f"faults, window 20 samples, one uplink per 10, {n_required} disjoint run-level replicates "
        "(the same runs as step 2, so differences are paired by replicate), centre calibrated on validation "
        "runs to 5% false alarms. Every number is mean ± 95% t-CI over replicates (a replicate's value "
        "averages its encoder seeds).",
        "- Learned codes: each node encodes its local window (per variable mean, std, slope, min, max and "
        "the log energy of 3 FFT bands, standardised on the training windows) into exactly `budget` bytes; "
        "the centre is step 2's calibrated MLP on the decoded messages, stacking the last 6 uplinks, with the "
        "settings step 2 chose on the tuning block for the hand features (hidden 64, L2 1e-3, 200 iterations). "
        "Encoder hyperparameters were chosen on the tuning block only.",
        "  - **L1 end-to-end**: per-node MLP, `8 B` sign bits through a straight-through (or logistic-noise "
        "binary-concrete) bottleneck, trained jointly with a classifier head by class-balanced cross-entropy "
        "(torch, CPU); the head is then discarded and the centre is re-trained on the frozen code.",
        "  - **L2 residual k-means**: per node, PCA of the local features and `B` stages of 256-centroid "
        "k-means on the residual; one byte per stage; no labels.",
        "  - **L3 local classifier**: each node trains its own fault classifier and sends its top `B` classes, "
        "one byte each: 5 bits of class id and 3 bits of posterior. The centre trains on cross-fitted messages "
        "(classifiers that never saw the run), because a classifier is over-confident on its own training runs.",
        "",
        "## Verdict (goal G2)",
        "",
        "G2 passes if, at 1, 2 or 4 B per uplink, a learned code beats the best hand design's macro-F1 at "
        "the calibrated 5% false-alarm rate with the 95% CI of the paired difference over 10 replicates "
        "excluding 0; it is killed if no budget shows a significant win.",
        "",
        f"**{verdict}.** {why}",
        "",
    ]
    if tests:
        top = max(tests, key=lambda t: t["diff"])
        lines += [
            f"- Largest difference: {top['method']} at {top['budget']} B, {top['diff']:+.3f} ± {top['half']:.3f} "
            f"(macro-F1 {top['learned']:.3f} against ({top['hand']}) {top['hand_mean']:.3f}).",
            f"- Multiple comparisons: {len(tests)} tests are run (methods x budgets); "
            f"{sum(t.get('win_holm', False) for t in tests)} survive Holm correction at 5%. "
            "The G2 rule as written uses the unadjusted 95% CI; read a single marginal win with that in mind.",
        ]
        same = [
            (a["method"], a["budget"])
            for a, b in zip(tests, matched)
            if a["win"] != b["win"]
        ]
        lines.append(
            "- Re-thresholded to exactly 5% test false alarms (not deployable, removes calibration error), "
            + (
                "the verdict is unchanged."
                if not same
                else "the win/no-win outcome changes for "
                + ", ".join(f"{m} at {b} B" for m, b in same)
                + "."
            )
        )
        lines.append("")
    lines += [
        "## Learned against the best hand design (macro-F1 at the calibrated threshold)",
        "",
    ]
    lines += _test_table(tests)
    lines += ["### Same comparison at exactly 5% test false alarms", ""]
    lines += _test_table(matched)
    lines += ["## Macro-F1 by budget (17 faults, calibrated 5% threshold)", ""]
    lines += _metric_table(rows, base_rows, "macro_f1", 3, REPORT_BUDGETS)
    lines += ["## Realised test false-alarm rate (target 5%)", ""]
    lines += _metric_table(rows, base_rows, "false_alarm_rate", 3, REPORT_BUDGETS)
    lines += ["## Detection delay in samples (mean over detected fault runs)", ""]
    lines += _metric_table(rows, base_rows, "detection_delay", 1, REPORT_BUDGETS)
    lines += ["## Macro-F1 re-thresholded to exactly 5% test false alarms", ""]
    lines += _metric_table(rows, base_rows, "macro_f1_matched", 3, REPORT_BUDGETS)
    lines += ["## Ablations of the best learned method", ""]
    for budget in G2_BUDGETS:
        method = best_method(rows, budget)
        if method is None:
            continue
        if not by_replicate(rows, method, budget, "macro_f1", "shuffle"):
            continue
        lines += [
            f"### {METHOD_LABELS[method]} at {budget} B",
            "",
            "The centre is trained on intact messages; the intervention only changes what it receives at "
            "test time. The method shown is the one with the best test macro-F1 at this budget (chosen "
            "after the fact, for description only).",
            "",
        ]
        lines += _ablation_table(rows, method, budget)
    lines += ["## Chosen hyperparameters (tuning block)", ""]
    lines += [
        "| Method | Budget | Hyperparameters | Best tuning-block validation macro-F1 |",
        "|---|---|---|---|",
    ]
    for key in sorted(
        merged["tuning"], key=lambda k: (k.split("|")[0], int(k.split("|")[1]))
    ):
        t = merged["tuning"][key]
        best = max([x["val_f1"] for x in t["trace"]] or [float("nan")])
        keep = {
            k: v
            for k, v in t["hp"].items()
            if k in {name for name, _ in SCALES[merged["scale"]].grids[t["method"]]}
        }
        lines.append(f"| {t['method']} | {t['budget']} B | {keep} | {best:.3f} |")
    lines += [
        "",
        "## Checks that ran with the experiment",
        "",
        f"- Largest message: {max(r['msg_bytes'] for r in rows)} B per node at most "
        f"(rows with a message longer than their budget: "
        f"{sum(r['msg_bytes'] > r['budget'] for r in rows)}); largest symbol "
        f"{max(r['max_symbol'] for r in rows)} (a byte holds 0..255).",
        "",
        "## Caveats",
        "",
        "- One simulator, one sampling interval; replicates differ in simulator seeds only. The hand-design "
        "scores are step 2's numbers on the same replicates, not re-run.",
        "- The centre's settings come from step 2's search with hand features; a search with the learned "
        "codes (a different input distribution) might help them a little. The encoders got a staged "
        "coordinate search of at most about 10 fits per (method, budget) on the tuning block; the hand "
        "designs are fixed designs with step 2's centre search.",
        "- The learned codes (L1, L3) use the fault labels of the training runs to shape the message; the "
        "hand designs (a)-(c) are unsupervised and only their centre sees labels. A win therefore says "
        "that a label-trained node code beats these particular unsupervised designs, not that no "
        "hand-designed supervised code (for example a per-node discriminant) could match it.",
        "- Learned encoders are trained on 20 faulty runs per fault (340 faulty runs plus 40 fault-free, in "
        "all), the same data the hand designs' PCA and thresholds used.",
        "- Windows within a run are autocorrelated; replicates, not windows, are the unit of the CIs.",
        "- The tuning block is disjoint from every replicate; the best-method choice in the ablation uses "
        "test scores.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/plant_learned_codes.py --methods e2e,vq,local --budgets 1,2,4,8 --replicates 10 --workers 8 --out docs/results/plant_learned_codes.md",
        "# merge partial result files into the report, no computation:",
        "python scripts/plant_learned_codes.py --from-json <dir or files> --out docs/results/plant_learned_codes.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def read_partials(sources: Iterable[Path]) -> list[dict[str, Any]]:
    """Load partial result files (``kind`` ``plant_learned_codes_partial``) from files or folders."""
    paths: list[Path] = []
    for src in sources:
        src = Path(src)
        paths += sorted(src.rglob("*.json")) if src.is_dir() else [src]
    out = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("kind") == "plant_learned_codes_partial":
            out.append(data)
    return out


def default_protocol(replicates: int = 10) -> dict[str, Any]:
    p = dataclasses.asdict(Protocol(n_replicates=replicates))
    p.pop("faults")
    return p


def torch_threads_for(workers: int, requested: int = 1) -> tuple[int, int]:
    """``(processes, torch threads)`` with processes x threads <= workers."""
    workers = max(1, int(workers))
    threads = max(1, min(int(requested), workers))
    return max(1, workers // threads), threads


def env_workers(default: int) -> int:
    """``GCA_WORKERS`` from the environment when set and valid, else ``default``."""
    raw = os.environ.get("GCA_WORKERS", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else int(default)
