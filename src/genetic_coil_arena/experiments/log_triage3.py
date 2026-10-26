"""R10 step 3: trained fault-type specialists in the typed team (goal G4c).

Step 2 found fault type to be the bottleneck: a rule centre that picks the service by the metric
anomaly score matches BARO (88% top-1) but only gets about 45% joint accuracy, because the
strongest metric kind says little about disk, loss and socket faults. Here the fault type comes
from a *trained* specialist, which sends a typed report (:mod:`..envs.logtriage.trained`):

* **metric fault specialist** (CPU): per-kind features of the service's own metrics
  (:mod:`..envs.logtriage.fault_features`) into a small numpy classifier;
* **log fault specialist** (GPU): a LoRA adapter on Qwen3-1.7B reads the service's own log view
  around the reported onset and answers with a letter, scored by closed-set log-probabilities.

**Data.** Training uses RCAEval RE2 Train Ticket (all repetitions) and Sock Shop / Online Boutique
repetition 1 (the dev split). The test split, SS/OB repetitions 2 and 3, is never fitted on,
tuned on or used to select anything: :func:`assert_no_test_leak` checks the incident ids and
the system/repetition rule, and the tests pin it. Model selection is leave-one-system-out
cross-validation over the three training systems (TT, SS-dev, OB-dev); the fusion rule of the
two specialists is chosen on the cross-validated predictions.

Caveat: a test incident is a repetition of a dev case (same service, same fault), so models
trained on SS/OB-dev may partly memorise cases; the leave-one-system-out numbers measure
cross-system transfer instead, and a TT-only metric specialist is run as a diagnostic.
"""

import random
import time
import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..envs.logtriage import fault_features as ff
from ..envs.logtriage import rcaeval
from ..envs.logtriage import team as tm
from ..envs.logtriage import trained as tr
from ..envs.logtriage.incident import Incident
from ..envs.logtriage.rcaeval import FAULT_TYPES
from ..envs.logtriage.views import TokenCounter
from ..learning.classifiers import MLPClassifier
from . import log_triage as lt

TRAIN_SYSTEMS = ("TT", "SS", "OB")
TEST_SYSTEMS = ("SS", "OB")
FAULT_INDEX = {f: i for i, f in enumerate(FAULT_TYPES)}
TEAMS3 = (
    "team_trained_metric",
    "team_trained_log",
    "team_trained_both",
    "team_trained_metric_ttonly",
)
ABLATIONS3 = (
    "drop_root",
    "shuffle_root",
    "drop_other",
    "shuffle_other",
    "drop_fault_root",
    "shuffle_fault_root",
)
METRIC_GRID: tuple[dict[str, Any], ...] = (
    {"hidden": 0, "l2": 0.01, "epochs": 400},
    {"hidden": 0, "l2": 0.1, "epochs": 400},
    {"hidden": 0, "l2": 1.0, "epochs": 400},
    {"hidden": 16, "l2": 0.01, "epochs": 300, "n_models": 5},
    {"hidden": 16, "l2": 0.1, "epochs": 300, "n_models": 5},
    {"hidden": 32, "l2": 0.1, "epochs": 300, "n_models": 5},
    {"hidden": 32, "l2": 1.0, "epochs": 300, "n_models": 5},
)


# ------------------------------------------------------------------------- data split


def role_of(system: str, rep: int) -> str:
    """``train`` (Train Ticket, any repetition; SS/OB dev repetitions) or ``test`` (SS/OB else)."""
    if system == "TT":
        return "train"
    if system in TEST_SYSTEMS:
        return "train" if rep in rcaeval.DEV_REPS else "test"
    return "other"


Key = tuple[str, str, int]  # (incident id, system, repetition)


def file_key(path: Path) -> Key:
    """(id, system, rep) of a converted incident file, from its name alone."""
    system = path.parent.name
    case, rep = path.stem.rsplit("__", 1)
    return f"{system}/{case}/{rep}", system, int(rep)


def inc_key(inc: Incident) -> Key:
    return inc.id, inc.system, inc.rep


def assert_no_test_leak(train: Sequence[Key], test: Sequence[Key]) -> None:
    """Raise if a test incident is in the training set, or either set breaks the split rule."""
    both = {k[0] for k in train} & {k[0] for k in test}
    if both:
        raise ValueError(
            f"{len(both)} incident(s) in both train and test: {sorted(both)[:3]}"
        )
    for name, keys in (("train", train), ("test", test)):
        for iid, system, rep in keys:
            if role_of(system, rep) != name:
                raise ValueError(
                    f"{iid} (system {system}, rep {rep}) is not {name} data"
                )


def paths_by_role(files: Sequence[Path]) -> dict[str, list[Path]]:
    """Group converted incident files (``<case>__<rep>.npz`` under ``incidents/<system>``)."""
    out: dict[str, list[Path]] = {"train": [], "test": []}
    for f in files:
        _, system, rep = file_key(f)
        role = role_of(system, rep)
        if role in out:
            out[role].append(f)
    return out


def load_split(
    root: Optional[Path] = None, limit_train: Optional[int] = None
) -> tuple[list[Path], list[Path]]:
    """(training files, test files) of the RCAEval cache; checks the split before returning."""
    files = rcaeval.list_incidents(root, TRAIN_SYSTEMS)
    parts = paths_by_role(files)
    train, test = parts["train"], parts["test"]
    assert_no_test_leak([file_key(f) for f in train], [file_key(f) for f in test])
    if limit_train and limit_train < len(train):
        train = sorted(train, key=lambda q: zlib.crc32(q.name.encode()))[:limit_train]
        train.sort()
    return train, test


def describe(incs: Sequence[Incident]) -> dict[str, Any]:
    """Counts per system, repetition and fault type."""
    return {
        "n": len(incs),
        "by_system": dict(Counter(i.system for i in incs)),
        "by_system_rep": dict(Counter(f"{i.system}/{i.rep}" for i in incs)),
        "by_fault": dict(Counter(i.fault_type for i in incs)),
    }


# --------------------------------------------------------------------- metric specialist


@dataclass
class MetricDataset:
    """Root-service feature rows of the training incidents."""

    x: np.ndarray
    y: np.ndarray
    ids: list[str]
    groups: list[str]  # the system, for leave-one-system-out

    def subset(self, mask: np.ndarray) -> "MetricDataset":
        keep = np.flatnonzero(mask)
        return MetricDataset(
            self.x[keep],
            self.y[keep],
            [self.ids[i] for i in keep],
            [self.groups[i] for i in keep],
        )


def metric_dataset(incs: Sequence[Incident]) -> MetricDataset:
    """Features of each incident's root-cause service; labels are the fault types."""
    x, y, ids, groups = [], [], [], []
    for inc in incs:
        x.append(ff.service_features(inc, inc.root_service))
        y.append(FAULT_INDEX[inc.fault_type])
        ids.append(inc.id)
        groups.append(inc.system)
    return MetricDataset(np.stack(x), np.asarray(y), ids, groups)


def _make(cfg: Mapping[str, Any], seed: int) -> MLPClassifier:
    return MLPClassifier(len(FAULT_TYPES), seed=seed, **dict(cfg))


def loso(
    data: MetricDataset, cfg: Mapping[str, Any], seed: int = 0
) -> tuple[np.ndarray, dict[str, float]]:
    """Leave-one-system-out: out-of-fold probabilities (N, 6) and accuracy per held-out system."""
    groups = np.asarray(data.groups)
    oof = np.zeros((len(data.y), len(FAULT_TYPES)))
    acc: dict[str, float] = {}
    for g in sorted(set(data.groups)):
        held = groups == g
        model = _make(cfg, seed).fit(data.x[~held], data.y[~held])
        oof[held] = model.predict_proba(data.x[held])
        acc[g] = float(np.mean(oof[held].argmax(axis=1) == data.y[held]))
    return oof, acc


def select_metric_model(
    data: MetricDataset,
    grid: Sequence[Mapping[str, Any]] = METRIC_GRID,
    seed: int = 0,
    log: Callable[[str], None] = lambda s: None,
) -> dict[str, Any]:
    """Pick the config with the best mean leave-one-system-out accuracy (ties: simpler first).

    Returns the table, the chosen config and the out-of-fold probabilities of the chosen one.
    """
    table, best, best_oof = [], None, None
    for k, cfg in enumerate(grid):
        oof, acc = loso(data, cfg, seed)
        mean = float(np.mean(list(acc.values())))
        table.append({"config": dict(cfg), "per_system": acc, "mean": mean})
        log(f"metric cv {k + 1}/{len(grid)} {dict(cfg)}: mean {mean:.3f} {acc}")
        if best is None or mean > table[best]["mean"] + 1e-9:
            best, best_oof = k, oof
    assert best is not None and best_oof is not None
    return {
        "table": table,
        "chosen": table[best]["config"],
        "chosen_index": best,
        "oof": best_oof,
    }


class MetricFaultSpecialist:
    """The trained metric fault specialist: step 2's metric report plus a fault distribution."""

    def __init__(self, model: MLPClassifier, name: str = "trained"):
        self.model = model
        self.name = name

    def probs(self, inc: Incident, service: int) -> np.ndarray:
        feats = ff.service_features(inc, inc.services[service])
        return self.model.predict_proba(feats[None])[0]

    def messages(self, inc: Incident) -> list[dict[str, Any]]:
        """One typed report per service; each reads only its own service's metrics."""
        feats = np.stack([ff.service_features(inc, n) for n in inc.services])
        probs = self.model.predict_proba(feats)
        return [
            tr.trained_metric_message(tm.metric_message(inc, s), probs[s])
            for s in range(inc.n_services)
        ]


# ------------------------------------------------------------------ log specialist (LoRA)

LOG_SYSTEM_PROMPT = (
    "You are a log monitor in a microservice incident-response team. One fault was "
    "injected into the service you monitor at the alert time."
)
_FAULT_HELP = {
    "cpu": "CPU hog",
    "mem": "memory exhaustion",
    "disk": "disk I/O stress",
    "delay": "network delay",
    "loss": "packet loss",
    "socket": "socket exhaustion",
}


def fault_prompt(service: str, view: str) -> str:
    """The log fault specialist's user message: its own log view and the closed answer set."""
    options = "\n".join(
        f"{lt.FAULT_LETTERS[i]}. {k}: {_FAULT_HELP[k]}"
        for i, k in enumerate(FAULT_TYPES)
    )
    return (
        f"{view}\n\nWhich fault type was injected into service '{service}'? You see only "
        f"this service's own logs.\n{options}\n\nAnswer with the letter only."
    )


def chat_text(tokenizer: Any, user: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": LOG_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@dataclass
class LogSettings:
    view_tokens: int = 1500
    before_s: float = 120.0
    after_s: float = 300.0
    jitters: tuple[float, ...] = (0.0, -20.0, 20.0)  # onset shifts used as augmentation


def log_prompt_for(
    inc: Incident,
    service: int,
    onset: Optional[float],
    counter: TokenCounter,
    cfg: LogSettings,
) -> str:
    view = tm.log_view(
        inc, service, onset, cfg.view_tokens, counter, cfg.before_s, cfg.after_s
    )
    return fault_prompt(inc.services[service], view)


def log_training_examples(
    inc: Incident, counter: TokenCounter, cfg: LogSettings
) -> list[tuple[str, int]]:
    """(prompt, fault index) examples of the root-cause service: one per onset jitter."""
    s = inc.services.index(inc.root_service)
    onset = tm.metric_message(inc, s)["onset_s"]
    out = []
    for j in cfg.jitters:
        shifted = (
            None if onset is None and j == 0 else (0.0 if onset is None else onset) + j
        )
        out.append(
            (log_prompt_for(inc, s, shifted, counter, cfg), FAULT_INDEX[inc.fault_type])
        )
    return out


def fault_letter_ids(tokenizer: Any) -> list[int]:
    ids = []
    for letter in lt.FAULT_LETTERS:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"{letter!r} is not a single token: {enc}")
        ids.append(int(enc[0]))
    return ids


def letter_probs(logprobs: Sequence[Sequence[float]]) -> np.ndarray:
    """Renormalise log-probabilities over the six answer letters."""
    z = np.asarray(logprobs, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def score_prompts(
    backend: Any,
    adapter: Optional[str],
    prompts: Sequence[str],
    batch: int = 4,
    log: Callable[[str], None] = lambda s: None,
    log_every_s: float = 120.0,
) -> np.ndarray:
    """Fault distribution (N, 6) of each prompt under ``adapter`` (closed-set letter scoring)."""
    tok = backend.tokenizer
    ids = fault_letter_ids(tok)
    chats = [chat_text(tok, p) for p in prompts]
    order = sorted(
        range(len(chats)), key=lambda i: len(chats[i])
    )  # similar lengths together
    out = np.zeros((len(chats), len(FAULT_TYPES)))
    last = time.time()
    for lo in range(0, len(order), batch):
        part = order[lo : lo + batch]
        lp = backend.next_token_logprobs(
            [chats[i] for i in part], ids, adapter, chunk_size=2048
        )
        out[part] = letter_probs(lp)
        if time.time() - last > log_every_s:
            log(f"scored {lo + len(part)}/{len(order)}")
            last = time.time()
    return out


def train_log_adapter(
    backend: Any,
    name: str,
    examples: Sequence[tuple[str, int]],
    *,
    rank: int = 8,
    epochs: int = 2,
    batch: int = 4,
    lr: float = 2e-4,
    seed: int = 0,
    target_modules: Sequence[str] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
    log: Callable[[str], None] = print,
) -> Any:
    """LoRA on the answer letter (cross-entropy on the letter and the stop token)."""
    backend.add_adapter(
        name,
        r=rank,
        alpha=2 * rank,
        dropout=0.05,
        target_modules=list(target_modules),
        seed=seed,
    )
    tok = backend.tokenizer
    chats = [chat_text(tok, p) for p, _ in examples]
    labels = [lt.FAULT_LETTERS[y] for _, y in examples]
    rng = random.Random(seed)
    steps = max(1, epochs * len(examples) // batch)
    order: list[list[int]] = []
    while len(order) < steps:
        idx = list(range(len(examples)))
        rng.shuffle(idx)
        order += [idx[lo : lo + batch] for lo in range(0, len(idx) - batch + 1, batch)]
    order = order[:steps]

    def step(i: int) -> Any:
        rows = order[i]
        inputs = backend.build_inputs(
            [chats[r] for r in rows], [labels[r] for r in rows]
        )
        return backend.training_loss(inputs)

    backend.set_gradient_checkpointing(True)
    try:
        stats = backend.fit(
            name,
            step,
            steps=steps,
            lr=lr,
            warmup=min(20, steps // 5),
            log_every=10,
            logger=log,
            label=f"[{name}]",
        )
    finally:
        backend.set_gradient_checkpointing(False)
    return stats


# ----------------------------------------------------------------------- fusion choice


@dataclass
class FusionChoice:
    rule: str
    table: dict[str, dict[str, float]]
    stack: Optional[dict[str, Any]] = None
    n: int = 0


def select_fusion(
    pm: np.ndarray,
    pl: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    rules: Sequence[str] = tr.FUSIONS,
) -> FusionChoice:
    """Choose the fusion rule with the best mean per-system accuracy on out-of-fold reports.

    ``pm``, ``pl``: the metric and log specialists' cross-validated fault distributions for
    the root service of each training incident, shape (N, 6). The stacked rule is itself
    evaluated leave-one-system-out. Ties go to the earlier rule in :data:`trained.FUSIONS`.
    """
    g = np.asarray(groups)
    table: dict[str, dict[str, float]] = {}
    if len(set(groups)) < 2:  # a stack cannot be evaluated leave-one-system-out
        rules = [r for r in rules if r != "stack"]
    for rule in rules:
        per: dict[str, float] = {}
        for system in sorted(set(groups)):
            held = g == system
            stack = None
            if rule == "stack":
                stack = tr.fit_stack(pm[~held], pl[~held], y[~held])
            pred = [
                int(np.argmax(tr.fuse(rule, a, b, stack)))
                for a, b in zip(pm[held], pl[held])
            ]
            per[system] = float(np.mean(np.asarray(pred) == y[held]))
        per["mean"] = float(np.mean(list(per.values())))
        table[rule] = per
    best = max(rules, key=lambda r: (table[r]["mean"], -tr.FUSIONS.index(r)))
    stack = tr.fit_stack(pm, pl, y) if best == "stack" else None
    return FusionChoice(rule=best, table=table, stack=stack, n=len(y))


# -------------------------------------------------------------------------- the teams

Pool3 = dict[str, list[tuple[dict[str, Any], Optional[dict[str, Any]]]]]


def build_pool3(
    per_incident: Mapping[
        str, tuple[Sequence[str], list[dict[str, Any]], Optional[list[dict[str, Any]]]]
    ],
) -> Pool3:
    """service -> [(metric report, log report or None)] over incidents, for shuffles."""
    pool: Pool3 = {}
    for services, mm, lm in per_incident.values():
        for i, name in enumerate(services):
            pool.setdefault(name, []).append((mm[i], lm[i] if lm else None))
    return pool


def check_messages(
    mm: Sequence[Mapping[str, Any]], lm: Optional[Sequence[Mapping[str, Any]]]
) -> None:
    """Raise if any typed report violates its schema."""
    for m in mm:
        errs = tr.trained_metric_errors(m)
        if errs:
            raise ValueError(f"invalid metric fault report: {errs}")
    for m in lm or []:
        errs = tr.trained_log_errors(m)
        if errs:
            raise ValueError(f"invalid log fault report: {errs}")


def _fault_report_ablation(
    mode: str,
    idx: int,
    name: str,
    mm: list[dict[str, Any]],
    lm: Optional[list[dict[str, Any]]],
    pool: Optional[Pool3],
    rng: random.Random,
    own: tuple[dict[str, Any], Optional[dict[str, Any]]],
) -> None:
    """Replace only the fault reports of service ``idx`` (scores and onset stay intact)."""
    if mode == "drop":
        mm[idx] = tr.trained_metric_message(
            {k: v for k, v in mm[idx].items() if k not in ("fault", "fault_probs")},
            None,
        )
        if lm is not None:
            lm[idx] = tr.empty_trained_log_message(name)
        return
    alt = [p for p in (pool or {}).get(name, []) if p != own]
    if alt:
        a_m, a_l = rng.choice(alt)
        mm[idx] = dict(
            mm[idx], fault=a_m["fault"], fault_probs=list(a_m["fault_probs"])
        )
        if lm is not None and a_l is not None:
            lm[idx] = dict(a_l, service=name)


def team3_records(
    inc: Incident,
    team: str,
    mm: Sequence[dict[str, Any]],
    lm: Optional[Sequence[dict[str, Any]]],
    rule: str,
    stack: Optional[Mapping[str, Any]],
    pool: Optional[Pool3] = None,
    ablate: bool = True,
    model: str = "trained",
) -> list[lt.Record]:
    """The team's record and, if ``ablate``, its message ablations.

    ``rule`` is the fusion rule of the centre (``metric`` for the metric-only team, ``log``
    for the log-only team). Whole-service ablations empty or swap both of a service's
    reports; the ``*_fault_root`` ablations touch only the root service's fault reports.
    """
    check_messages(mm, lm)
    t0 = time.perf_counter()
    ranking, fault = tr.rule_decide_trained(inc.services, mm, lm, rule, stack)
    latency = time.perf_counter() - t0
    nbytes = sum(tm.nbytes(m) for m in mm) + sum(tm.nbytes(m) for m in lm or [])
    extra = {"bytes": nbytes, "fusion": rule, "centre_model": "rule"}
    records = [
        lt.score_record(
            inc, model, team, ranking, fault, latency_s=latency, extra=extra
        )
    ]
    if not ablate:
        return records
    rng = random.Random(zlib.crc32(inc.id.encode()))
    root = inc.services.index(inc.root_service)
    others = [i for i in range(inc.n_services) if i != root]
    picks = {"root": root, "other": rng.choice(others)}
    for label, idx in picks.items():
        name = inc.services[idx]
        for mode in ("drop", "shuffle"):
            m2, l2 = list(mm), list(lm) if lm is not None else None
            if mode == "drop":
                m2[idx] = tr.empty_trained_metric_message(name)
                if l2 is not None:
                    l2[idx] = tr.empty_trained_log_message(name)
            else:
                own = (mm[idx], lm[idx] if lm else None)
                alt = [p for p in (pool or {}).get(name, []) if p != own]
                if alt:
                    a_m, a_l = rng.choice(alt)
                    m2[idx] = dict(a_m, service=name)
                    if l2 is not None and a_l is not None:
                        l2[idx] = dict(a_l, service=name)
            r, f = tr.rule_decide_trained(inc.services, m2, l2, rule, stack)
            records.append(
                lt.score_record(
                    inc,
                    model,
                    f"{team}:{mode}_{label}",
                    r,
                    f,
                    extra={"fusion": rule, "ablated_service": name},
                )
            )
    for mode in ("drop", "shuffle"):
        m2, l2 = list(mm), list(lm) if lm is not None else None
        _fault_report_ablation(
            mode, root, inc.services[root], m2, l2, pool, rng,
            (mm[root], lm[root] if lm else None),
        )  # fmt: skip
        r, f = tr.rule_decide_trained(inc.services, m2, l2, rule, stack)
        records.append(
            lt.score_record(
                inc,
                model,
                f"{team}:{mode}_fault_root",
                r,
                f,
                extra={"fusion": rule, "ablated_service": inc.services[root]},
            )
        )
    return records


def strip_fault_reports(mm: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The metric reports without their trained fault fields (for the log-only team)."""
    return [
        tr.trained_metric_message(
            {k: v for k, v in m.items() if k not in ("fault", "fault_probs")}, None
        )
        for m in mm
    ]
