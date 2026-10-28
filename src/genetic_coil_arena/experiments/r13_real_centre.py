"""R13 (goals G11a, G11b): does the G4c team replicate on fresh incidents, and can a learned centre rank?

Data: RCAEval **RE1** (metrics only, resource faults cpu/mem/disk/delay/loss; Sock Shop, Online
Boutique, Train Ticket; 125 incidents each), nothing of which earlier goals used.

* **G11a**: the frozen G4c team (the ``r11-team`` fault specialist, the rule centre; no refit,
  threshold or policy change) runs on every RE1 incident through :class:`IncidentTeam` and is
  compared with BARO (``log_triage2.run_baro``'s rule: the strongest metric kind of BARO's top
  service gives the fault) on joint accuracy.
* **G11b**: a small numpy ranker over per-service report features, trained on RE1 only, picks the
  root service. :class:`ConditionalLogit` is a softmax over the services of one incident. The
  hyperparameters are chosen by leave-one-system-out cross-validation on RE1 and the ranker is
  tested once on RE2 (SS/OB repetitions 2 and 3, Train Ticket all repetitions).

The ranker reads only what the specialists send (BARO score, the frozen fault specialist's six
probabilities, per-metric-kind anomaly maxima); features are indexed by metric kind, never by name.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..envs.logtriage import fault_features as ff
from ..envs.logtriage import heuristics as hz
from ..envs.logtriage import rcaeval
from ..envs.logtriage import team as tm
from ..envs.logtriage.incident import Incident
from ..envs.logtriage.rcaeval import FAULT_TYPES, RE1_SYSTEMS
from ..evaluation.stats import paired_bootstrap_ci
from . import log_triage3 as t3

N_BOOT = 10_000
RE1_SYS = ("SS", "OB", "TT")  # RE1-<system>, the leave-one-system-out folds
# SHA-256 of the frozen G4c fault specialist (``r11-team``'s model dict); pinned before any run
FROZEN_MODEL_SHA256: Optional[str] = (
    "953524c1ce6905fc7de0d49e5682479794e040001a0dfcd8137698e9c69934d6"
)
STD_FLOOR = 0.5

# G11b configs (pre-registered): feature set x L2 on the weights
FEATURE_SETS = ("compact", "all")
L2_GRID = (0.01, 0.1, 1.0)
CONFIGS: tuple[dict[str, Any], ...] = tuple(
    {"features": f, "l2": l2} for f in FEATURE_SETS for l2 in L2_GRID
)
TRAIN_STEPS = 300
TRAIN_LR = 0.05


# ------------------------------------------------------------------------------ data


def ensure_re1(cache_dir: Optional[Path], log: Callable[[str], None] = print) -> None:
    """Fetch and convert RE1 when any system is missing (about 0.4 GB)."""
    base = rcaeval.cache_dir(cache_dir)
    missing = tuple(
        s
        for s in RE1_SYSTEMS
        if not (rcaeval.incident_dir(base, s) / "_done.json").exists()
    )
    if missing:
        log(f"RE1 missing in {base}: fetching {','.join(missing)}")
        rcaeval.fetch(cache_dir, missing, keep_raw=False, log=log)


def load_metrics_view(path: Path) -> Incident:
    """An incident with metrics only: the (large) line arrays of RE2 files are never read.

    ``meta["has_logs"]`` is set False, so any log method refuses it (``NoLogsError``).
    """
    with np.load(path) as data:
        meta = json.loads(bytes(data["meta"]).decode("utf-8"))
    meta["meta"] = {**meta["meta"], "has_logs": False}
    return Incident(
        id=meta["id"],
        system=meta["system"],
        services=meta["services"],
        root_service=meta["root_service"],
        fault_type=meta["fault_type"],
        rep=meta["rep"],
        t=np.zeros(0, dtype=np.float32),
        svc=np.zeros(0, dtype=np.int16),
        tpl=np.zeros(0, dtype=np.int32),
        templates=[],
        window=(0.0, 0.0),
        meta=meta["meta"],
        metrics=meta.get("metrics", {}),
    )


def load_re1(
    cache_dir: Optional[Path] = None, limit: Optional[int] = None
) -> list[Incident]:
    """All RE1 incidents (a stable subsample of ``limit`` per system for smoke runs)."""
    out: list[Incident] = []
    for system in RE1_SYSTEMS:
        files = rcaeval.list_incidents(cache_dir, (system,))
        if not files:
            raise FileNotFoundError(f"no converted {system} incidents")
        if limit:
            files = files[:: max(1, len(files) // limit)][:limit]
        out += [load_metrics_view(f) for f in files]
    return out


def load_re2_test(
    cache_dir: Optional[Path] = None, limit: Optional[int] = None
) -> list[Incident]:
    """The G11b test set: SS/OB repetitions 2-3 (120) and Train Ticket, all repetitions (90)."""
    ssob = t3.paths_by_role(rcaeval.list_incidents(cache_dir, ("SS", "OB")))["test"]
    tt = rcaeval.list_incidents(cache_dir, ("TT",))
    files = list(ssob) + list(tt)
    if not ssob or not tt:
        raise FileNotFoundError("no converted RE2 incidents: run fetch_rcaeval.py")
    if limit:
        files = files[:: max(1, len(files) // limit)][:limit]
    return [load_metrics_view(f) for f in files]


def re1_statistics(cache_dir: Optional[Path] = None) -> dict[str, Any]:
    """Per-system RE1 counts, kinds and parse failures from the converter's ``_done.json``."""
    base = rcaeval.cache_dir(cache_dir)
    out: dict[str, Any] = {}
    for system in RE1_SYSTEMS:
        info = json.loads(
            (rcaeval.incident_dir(base, system) / "_done.json").read_text("utf-8")
        )
        out[system] = {
            "n_incidents": info["n_incidents"],
            "n_cases_found": info["n_cases_found"],
            "n_failed": len(info["failed"]),
            "failed": info["failed"],
            "metric_kinds": info["metric_kinds"],
            "logs": info["logs"],
            "kind_alias": info["kind_alias"],
        }
    return out


def kind_coverage(incs: Sequence[Incident]) -> dict[str, Any]:
    """Per metric kind: share of incidents whose root service has it, share of services."""
    root = dict.fromkeys(ff.KINDS, 0)
    svc = dict.fromkeys(ff.KINDS, 0)
    n_svc = 0
    for inc in incs:
        n_svc += len(inc.services)
        for s in inc.services:
            for k in inc.metrics.get(s, {}):
                if k in svc:
                    svc[k] += 1
        for k in inc.metrics.get(inc.root_service, {}):
            if k in root:
                root[k] += 1
    n = max(len(incs), 1)
    return {
        "n_incidents": len(incs),
        "mean_services": n_svc / n,
        "root_has_kind": {k: root[k] / n for k in ff.KINDS},
        "service_has_kind": {k: svc[k] / max(n_svc, 1) for k in ff.KINDS},
    }


# ------------------------------------------------------------------------ frozen team


def model_sha256(state: Mapping[str, Any]) -> str:
    text = json.dumps(state["model"], sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_frozen(state: Mapping[str, Any]) -> str:
    digest = model_sha256(state)
    if FROZEN_MODEL_SHA256 is not None and digest != FROZEN_MODEL_SHA256:
        raise ValueError(
            f"fault specialist differs from the frozen G4c model: {digest} != "
            f"{FROZEN_MODEL_SHA256}"
        )
    return digest


def baro_decision(inc: Incident) -> tuple[list[str], str]:
    """BARO on the metrics: ranking and the fault of the strongest kind of its top service."""
    order = hz.ranking(hz.baro_metrics(inc))
    rows = inc.metric_rows(inc.services[order[0]], min_z=0.0, top=1)
    fault = (
        hz.METRIC_FAULT.get(rows[0][0], tm.DEFAULT_FAULT) if rows else tm.DEFAULT_FAULT
    )
    return [inc.services[i] for i in order], fault


def rule_top(inc: Incident) -> str:
    """The rule centre's top service (ranking by the metric messages' scores)."""
    msgs = [tm.metric_message(inc, s) for s in range(inc.n_services)]
    return inc.services[tm.rule_ranking(msgs)[0]]


# ------------------------------------------------------------------------ ranker features

BASE_NAMES = ("logscore", "rel_to_max", "zscore", "rank_inv", "is_top")
PROB_NAMES = tuple(f"p_{f}" for f in FAULT_TYPES)
KIND_NAMES = tuple(f"logz:{k}" for k in ff.KINDS)
KIND_REL_NAMES = tuple(f"logz_rel:{k}" for k in ff.KINDS)
ALL_NAMES = BASE_NAMES + PROB_NAMES + KIND_NAMES + KIND_REL_NAMES
N_COMPACT = len(BASE_NAMES) + len(PROB_NAMES)


def feature_names(feature_set: str) -> tuple[str, ...]:
    if feature_set == "compact":
        return ALL_NAMES[:N_COMPACT]
    if feature_set == "all":
        return ALL_NAMES
    raise ValueError(f"unknown feature set {feature_set!r}")


def fault_probs(model: Any, inc: Incident) -> np.ndarray:
    """The frozen specialist's six probabilities for every service, shape (S, 6)."""
    feats = np.stack([ff.service_features(inc, s) for s in inc.services])
    return np.asarray(model.predict_proba(feats))


def incident_features(inc: Incident, probs: np.ndarray) -> np.ndarray:
    """Per-service report features, shape (S, len(ALL_NAMES)); see :data:`ALL_NAMES`."""
    n = len(inc.services)
    logz = np.zeros((n, len(ff.KINDS)))
    for i, s in enumerate(inc.services):
        for j, k in enumerate(ff.KINDS):
            row = inc.metrics.get(s, {}).get(k)
            if row is not None:
                logz[i, j] = np.log1p(min(max(float(row[0]), 0.0), 1e4))
    raw = np.array(
        [
            max((float(v[0]) for v in inc.metrics.get(s, {}).values()), default=0.0)
            for s in inc.services
        ]
    )
    score = np.log1p(np.minimum(np.maximum(raw, 0.0), 1e4))
    order = np.argsort(-score, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    out = np.zeros((n, len(ALL_NAMES)))
    out[:, 0] = score
    out[:, 1] = score - score.max()
    out[:, 2] = (score - score.mean()) / (score.std() + 0.5)
    out[:, 3] = 1.0 / (1.0 + rank)
    out[:, 4] = (rank == 0).astype(float)
    out[:, 5 : 5 + len(FAULT_TYPES)] = probs
    k0 = len(BASE_NAMES) + len(PROB_NAMES)
    out[:, k0 : k0 + len(ff.KINDS)] = logz
    out[:, k0 + len(ff.KINDS) :] = logz - logz.max(axis=0, keepdims=True)
    return out


class RankData:
    """Padded per-incident feature tensors: ``x`` (N, Smax, F), ``mask`` (N, Smax), ``y`` (N,)."""

    def __init__(self, incs: Sequence[Incident], model: Any):
        self.incs = list(incs)
        feats = [incident_features(i, fault_probs(model, i)) for i in self.incs]
        smax = max(f.shape[0] for f in feats)
        self.x = np.zeros((len(feats), smax, len(ALL_NAMES)))
        self.mask = np.zeros((len(feats), smax), dtype=bool)
        for k, f in enumerate(feats):
            self.x[k, : f.shape[0]] = f
            self.mask[k, : f.shape[0]] = True
        self.y = np.array([i.services.index(i.root_service) for i in self.incs])
        self.systems = np.array([i.system.split("-")[-1] for i in self.incs])
        self.probs = [f[:, 5 : 5 + len(FAULT_TYPES)] for f in feats]

    def subset(self, sel: np.ndarray) -> "RankData":
        out = object.__new__(RankData)
        idx = np.flatnonzero(sel)
        out.incs = [self.incs[i] for i in idx]
        out.x, out.mask, out.y = self.x[idx], self.mask[idx], self.y[idx]
        out.systems = self.systems[idx]
        out.probs = [self.probs[i] for i in idx]
        return out


# --------------------------------------------------------------------------- the ranker


class ConditionalLogit:
    """Softmax over the services of one incident: score = standardised features @ w.

    Trained full-batch with Adam on the mean negative log-likelihood of the root service plus
    ``0.5 * l2 * |w|^2``. There is no bias (a softmax over services ignores it). Features are
    standardised with the training rows' mean and std (std floored at :data:`STD_FLOOR`).
    """

    def __init__(self, feature_set: str = "all", l2: float = 0.1):
        self.feature_set = feature_set
        self.l2 = l2
        self.names = feature_names(feature_set)
        self.mean = np.zeros(len(self.names))
        self.std = np.ones(len(self.names))
        self.w = np.zeros(len(self.names))

    def _x(self, data: RankData) -> np.ndarray:
        return data.x[:, :, : len(self.names)]

    def fit(self, data: RankData) -> "ConditionalLogit":
        x = self._x(data)
        rows = x[data.mask]
        self.mean = rows.mean(axis=0)
        self.std = np.maximum(rows.std(axis=0), STD_FLOOR)
        z = (x - self.mean) / self.std
        z[~data.mask] = 0.0
        n = len(z)
        w = np.zeros(z.shape[2])
        m, v = np.zeros_like(w), np.zeros_like(w)
        onehot = np.zeros(data.mask.shape)
        onehot[np.arange(n), data.y] = 1.0
        for t in range(1, TRAIN_STEPS + 1):
            s = np.where(data.mask, z @ w, -np.inf)
            s -= s.max(axis=1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=1, keepdims=True)
            grad = np.einsum("ns,nsf->f", p - onehot, z) / n + self.l2 * w
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            w -= TRAIN_LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
        self.w = w
        return self

    def scores(self, data: RankData) -> np.ndarray:
        z = (self._x(data) - self.mean) / self.std
        return np.where(data.mask, z @ self.w, -np.inf)

    def top(self, data: RankData) -> np.ndarray:
        """Index of the top-ranked service of every incident (ties: the first)."""
        return self.scores(data).argmax(axis=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_set": self.feature_set,
            "l2": self.l2,
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "w": self.w.tolist(),
            "steps": TRAIN_STEPS,
            "lr": TRAIN_LR,
            "std_floor": STD_FLOOR,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ConditionalLogit":
        out = cls(d["feature_set"], d["l2"])
        out.mean, out.std, out.w = (
            np.asarray(d[k], float) for k in ("mean", "std", "w")
        )
        return out


def ranker_sha256(d: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def baro_top(data: RankData) -> np.ndarray:
    """The rule centre's top service per incident (feature ``rank_inv`` = 1 marks BARO's top)."""
    return np.array([i.services.index(rule_top(i)) for i in data.incs])


def loso_table(data: RankData) -> list[dict[str, Any]]:
    """Leave-one-system-out top-1 on RE1 for every config (and BARO's, on the same incidents)."""
    base = baro_top(data)
    rows = []
    for cfg in CONFIGS:
        per: dict[str, Any] = {}
        for held in RE1_SYS:
            test = data.systems == held
            model = ConditionalLogit(cfg["features"], cfg["l2"]).fit(data.subset(~test))
            sub = data.subset(test)
            hit = (model.top(sub) == sub.y).astype(float)
            ref = (base[test] == sub.y).astype(float)
            per[held] = {
                "top1": float(hit.mean()),
                "baro": float(ref.mean()),
                "n": int(test.sum()),
            }
        rows.append(
            {
                **cfg,
                "per_system": per,
                "mean_top1": float(np.mean([v["top1"] for v in per.values()])),
            }
        )
    return rows


def choose_config(table: Sequence[Mapping[str, Any]]) -> int:
    """Highest mean held-out top-1; ties go to the compact set, then the larger L2."""
    best = max(
        range(len(table)),
        key=lambda i: (
            round(table[i]["mean_top1"], 9),
            table[i]["features"] == "compact",
            table[i]["l2"],
        ),
    )
    return int(best)


def loso_pooled(data: RankData, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Pooled and per-system paired bootstrap (learned - BARO top-1) of the chosen config."""
    base = baro_top(data)
    hits = np.zeros(len(data.y))
    for held in RE1_SYS:
        test = data.systems == held
        model = ConditionalLogit(cfg["features"], cfg["l2"]).fit(data.subset(~test))
        hits[test] = model.top(data.subset(test)) == data.y[test]
    ref = (base == data.y).astype(float)
    out = {"pooled": paired(hits, ref)}
    for s in RE1_SYS:
        out[s] = paired(hits[data.systems == s], ref[data.systems == s])
    return out


# ------------------------------------------------------------------------------ stats


def paired(a: Sequence[float], b: Sequence[float], seed: int = 0) -> dict[str, Any]:
    """Mean of a, mean of b and the paired bootstrap CI of a - b (10,000 resamples), in [0, 1]."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    point, lo, hi = paired_bootstrap_ci(x, y, n_boot=N_BOOT, seed=seed)
    return {
        "n": int(x.size),
        "a": float(x.mean()),
        "b": float(y.mean()),
        "diff": point,
        "lo": lo,
        "hi": hi,
    }


def confusion(true: Sequence[str], pred: Sequence[str]) -> dict[str, dict[str, int]]:
    out = {t: dict.fromkeys(FAULT_TYPES, 0) for t in FAULT_TYPES}
    for t, p in zip(true, pred):
        out[t][p] += 1
    return out


def g11a_verdict(res: Mapping[str, Any]) -> dict[str, Any]:
    """Pass: pooled joint difference >= 5 points with CI above 0 (and each system alike)."""
    pooled = res["joint"]["pooled"]
    per = {s: res["joint"][s] for s in RE1_SYS}

    def ok(r: Mapping[str, Any]) -> bool:
        return r["diff"] >= 0.05 and r["lo"] > 0

    pooled_ok = ok(pooled)
    each = {s: ok(r) for s, r in per.items()}
    if pooled["lo"] <= 0:
        label = "kill"
    elif pooled_ok and all(each.values()):
        label = "passed"
    elif pooled_ok:
        label = "partial"
    else:
        label = "not met"
    return {"verdict": label, "pooled_ok": pooled_ok, "per_system_ok": each}


def g11b_verdict(res: Mapping[str, Any]) -> dict[str, Any]:
    """Pass: pooled top-1 gain >= 3 points with CI above 0 and every system's CI low > -5."""
    pooled = res["top1"]["pooled"]
    per = {s: res["top1"][s] for s in RE1_SYS}
    pooled_ok = pooled["diff"] >= 0.03 and pooled["lo"] > 0
    each = {s: r["lo"] > -0.05 for s, r in per.items()}
    if pooled["lo"] <= 0:
        label = "kill"
    elif pooled_ok and all(each.values()):
        label = "passed"
    else:
        label = "not met"
    return {"verdict": label, "pooled_ok": pooled_ok, "non_inferior": each}
