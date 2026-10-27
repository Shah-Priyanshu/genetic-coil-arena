"""R10 step 4 (goal G4d): do LLM log specialists earn their place where the signal is in the logs?

RCAEval RE3 injects code-level faults (``<service>_f<k>``), which leave new error, exception
and start-up lines in the root service's logs rather than moving its resource metrics. This
module holds the data split, the non-LLM systems (the bar an LLM team has to beat), the gate
statistics, and the LLM conditions' glue; ``log_triage4_report`` writes the verdict.

**Split.** Repetition 1 of every case is the *dev* split (prompts, fusion weights and the
choice of the best non-LLM system), repetitions 2 to 4 are the *test* split (every reported
number). :func:`assert_split` checks the two sets and the tests pin it. A test incident is a
repetition of a dev case (same service and fault), so dev choices can partly memorise cases;
the weights fitted here are few (a handful of integers), which limits that.

**Labels.** The root-cause service is the case name's prefix. The fault id (``f1`` ...) is
specific to the service and has no cross-service meaning, so a model cannot name it and there
is no joint accuracy: RE3 is scored on root-cause top-1 and top-3.
"""

import itertools
import random
import zlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..envs.logtriage import heuristics as hz
from ..envs.logtriage import logsignal as lg
from ..envs.logtriage import rcaeval
from ..envs.logtriage.incident import Incident
from . import log_triage as lt

SYSTEMS4 = rcaeval.RE3_SYSTEMS
NO_FAULT = "n/a"
FUSION_WEIGHTS = (0.0, 0.5, 1.0, 2.0)
FUSION_PARTS = ("baro_metrics", "new_ext", "new_any_mass", "baro_logs")


# ------------------------------------------------------------------------ data split


def role4(system: str, rep: int) -> str:
    """``dev`` (repetition 1), ``test`` (the other repetitions) or ``other`` (not RE3)."""
    if system not in SYSTEMS4:
        return "other"
    return "dev" if rep in rcaeval.DEV_REPS else "test"


def key_of(inc: Incident) -> tuple[str, str, int]:
    return inc.id, inc.system, inc.rep


def assert_split(
    dev: Sequence[tuple[str, str, int]], test: Sequence[tuple[str, str, int]]
) -> None:
    """Raise if an incident is in both sets or either set breaks the repetition rule."""
    both = {k[0] for k in dev} & {k[0] for k in test}
    if both:
        raise ValueError(f"{len(both)} incident(s) in dev and test: {sorted(both)[:3]}")
    for name, keys in (("dev", dev), ("test", test)):
        for iid, system, rep in keys:
            if role4(system, rep) != name:
                raise ValueError(f"{iid} (system {system}, rep {rep}) is not {name}")


def load_incidents(
    root: Optional[Path] = None,
    systems: Sequence[str] = SYSTEMS4,
    limit: Optional[int] = None,
    shard: Optional[tuple[int, int]] = None,
) -> tuple[list[Incident], list[Incident]]:
    """(dev, test) incidents of RE3; ``shard=(i, n)`` keeps every n-th incident from i.

    Sharding and ``limit`` both pick incidents by a hash of the id, so the choice does not
    depend on the file order.
    """
    files = rcaeval.list_incidents(root, tuple(systems))
    incs = [rcaeval.load_incident(f) for f in files]
    incs.sort(key=lambda i: i.id)
    if limit and limit < len(incs):
        incs = sorted(incs, key=lambda i: zlib.crc32(i.id.encode()))[:limit]
        incs.sort(key=lambda i: i.id)
    if shard is not None:
        k, n = shard
        incs = [i for i in incs if zlib.crc32(i.id.encode()) % n == k]
    dev = [i for i in incs if role4(i.system, i.rep) == "dev"]
    test = [i for i in incs if role4(i.system, i.rep) == "test"]
    assert_split([key_of(i) for i in dev], [key_of(i) for i in test])
    return dev, test


def describe(incs: Sequence[Incident]) -> dict[str, Any]:
    """Counts and size statistics per system (the data description of the report)."""
    out: dict[str, Any] = {}
    by_sys: dict[str, list[Incident]] = defaultdict(list)
    for i in incs:
        by_sys[i.system].append(i)
    for system, group in sorted(by_sys.items()):
        roots = sorted({i.root_service for i in group})
        out[system] = {
            "n": len(group),
            "root_services": roots,
            "n_candidate_services": int(np.median([i.n_services for i in group])),
            "faults": {
                f: sum(1 for i in group if i.fault_type == f)
                for f in sorted({i.fault_type for i in group})
            },
            "reps": {
                r: sum(1 for i in group if i.rep == r)
                for r in sorted({i.rep for i in group})
            },
            "cases": len({(i.root_service, i.fault_type) for i in group}),
            "median_lines": int(np.median([i.n_lines for i in group])),
            "median_templates": int(np.median([len(i.templates) for i in group])),
        }
    return out


# ------------------------------------------------------------- non-LLM systems


def tie_ranking(scores: Sequence[float], inc: Incident, salt: str = "") -> list[int]:
    """Service indices, best first; equal scores are ordered by a hash of the incident id.

    A fixed index order would hand ties to whichever service sorts first; hashing the id
    keeps the choice deterministic but unrelated to the label.
    """
    s = np.asarray(scores, dtype=np.float64)
    rng = random.Random(zlib.crc32((inc.id + salt).encode()))
    noise = np.array([rng.random() for _ in range(s.size)])
    return [int(i) for i in np.lexsort((noise, -s))]


def part_scores(inc: Incident, name: str) -> np.ndarray:
    if name == "baro_metrics":
        return hz.baro_metrics(inc)
    if name == "baro_logs":
        return hz.baro_logs(inc)
    if name == "surprise_logs":
        return hz.surprise_sum(inc)
    return lg.new_template_scores(inc, name)


NON_LLM_PARTS = (
    "baro_metrics",
    "baro_logs",
    "surprise_logs",
    *lg.LOG_SCORERS,
)


def fused_scores(parts: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted sum of the reciprocal ranks of each part (zero score = no credit)."""
    total = np.zeros(next(iter(parts.values())).size)
    for name, w in weights.items():
        if w:
            s = parts[name]
            total += w * np.where(s > 0, lg.reciprocal_ranks(s), 0.0)
    return total


def weight_grid() -> list[dict[str, float]]:
    grid = []
    for combo in itertools.product(FUSION_WEIGHTS, repeat=len(FUSION_PARTS)):
        if any(combo):
            grid.append(dict(zip(FUSION_PARTS, combo)))
    return grid


def fusion_label(weights: dict[str, float]) -> str:
    return "fuse(" + ",".join(f"{k}={v:g}" for k, v in weights.items() if v) + ")"


def top1_of(ranking: Sequence[int], inc: Incident) -> int:
    return int(inc.services[ranking[0]] == inc.root_service)


def pick_fusion(
    dev: Sequence[Incident], cache: dict[str, dict[str, np.ndarray]]
) -> dict:
    """The fusion weights with the best dev top-1 (ties: fewer parts, then grid order).

    ``cache`` maps incident id to its part scores (see :func:`all_parts`).
    """
    best: Optional[tuple[float, int, int]] = None
    chosen: dict[str, float] = {}
    for n, w in enumerate(weight_grid()):
        hits = [
            top1_of(tie_ranking(fused_scores(cache[i.id], w), i, "fuse"), i)
            for i in dev
        ]
        key = (float(np.mean(hits)), -sum(1 for v in w.values() if v), -n)
        if best is None or key > best:
            best, chosen = key, w
    return chosen


def all_parts(incs: Sequence[Incident]) -> dict[str, dict[str, np.ndarray]]:
    return {i.id: {p: part_scores(i, p) for p in NON_LLM_PARTS} for i in incs}


def non_llm_records(
    incs: Sequence[Incident],
    cache: dict[str, dict[str, np.ndarray]],
    fusion: dict[str, float],
) -> list[lt.Record]:
    """One record per (non-LLM system, incident); ``extra`` holds the fusion label."""
    records = []
    systems: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = {
        p: (lambda parts, p=p: parts[p]) for p in NON_LLM_PARTS
    }
    systems["fusion"] = lambda parts: fused_scores(parts, fusion)
    for inc in incs:
        parts = cache[inc.id]
        for name, fn in systems.items():
            order = tie_ranking(fn(parts), inc, name)
            records.append(
                lt.score_record(
                    inc,
                    "none",
                    f"nonllm_{name}",
                    [inc.services[i] for i in order],
                    NO_FAULT,
                    extra={"fusion": fusion_label(fusion)} if name == "fusion" else {},
                )
            )
    return records


# ------------------------------------------------------------------------ the gate


def gate_rows(incs: Sequence[Incident]) -> list[dict[str, Any]]:
    return [lg.gate_stats(i) for i in incs]


def gate_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per system: how often the root's logs hold a new warning/error template, and how
    often that count puts the root strictly first (no tie)."""
    out: dict[str, Any] = {}
    by_sys: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    for system, group in sorted(by_sys.items()):
        out[system] = {
            "n": len(group),
            "root_has_new_err": float(np.mean([r["root_new_err"] > 0 for r in group])),
            "root_new_err_mean": float(np.mean([r["root_new_err"] for r in group])),
            "others_new_err_mean": float(np.mean([r["others_mean"] for r in group])),
            "root_ranked_first": float(
                np.mean([r["root_strictly_first"] for r in group])
            ),
        }
    return out
