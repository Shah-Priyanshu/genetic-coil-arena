"""R13b probe: how many labelled incidents does the fault specialist need on a new system?

Exploratory probe, not a goal test. G11a replicated the frozen G4c fault specialist on
375 fresh RE1 incidents (+9.3 joint over BARO pooled), but not on Online Boutique, where
RE1 records few of the metric kinds it learned from. A deployment would label a few
incidents of its own. Here the frozen specialist gets a small adapter fitted on ``k``
labelled incidents of the target system:

- input at the root service: the specialist's six log-probabilities and the log z-score
  of every metric kind (``fault_features.KINDS``), plus a bias;
- a softmax regression over the six fault types, initialised to reproduce the frozen
  specialist exactly and pulled back towards it by an L2 penalty ``LAMBDA`` (so k = 0 is
  the frozen specialist);
- ``scratch``: the same model from zero weights, so the specialist is no prior.

For each RE1 system, k in ``KS``, and ``N_DRAWS`` random draws of k training incidents,
the rest of that system's incidents are the test. Measures: fault accuracy at the true
root, and joint accuracy with BARO's top service (the rule centre ranks by BARO, so this
is G11a's joint). References: the frozen specialist and BARO's fault rule on the same
test incidents.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..envs.logtriage import fault_features as ff
from ..envs.logtriage.rcaeval import FAULT_TYPES
from ..evaluation import mean_ci
from .r13_real_centre import baro_decision, fault_probs

KS = (0, 5, 10, 20, 40)
N_DRAWS = 30
LAMBDA = 1.0
STEPS, LR = 400, 0.1
N_F = len(FAULT_TYPES)


def incident_row(model: Any, inc: Any) -> dict[str, Any]:
    """Adapter input at the root service, label, and BARO's decision."""
    probs = fault_probs(model, inc)
    i = inc.services.index(inc.root_service)
    logz = np.zeros(len(ff.KINDS))
    for j, k in enumerate(ff.KINDS):
        row = inc.metrics.get(inc.root_service, {}).get(k)
        if row is not None:
            logz[j] = np.log1p(min(max(float(row[0]), 0.0), 1e4))
    ranking, b_fault = baro_decision(inc)
    return {
        "system": inc.system.split("-")[-1],
        "x": np.concatenate([np.log(np.clip(probs[i], 1e-6, 1.0)), logz]),
        "y": FAULT_TYPES.index(inc.fault_type),
        "baro_top_ok": ranking[0] == inc.root_service,
        "baro_fault_ok": b_fault == inc.fault_type,
    }


def _features(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((len(x), 1))], axis=1)


def prior_weights(n_in: int) -> np.ndarray:
    """(n_in + 1, 6): logits equal the frozen log-probabilities, everything else 0."""
    w = np.zeros((n_in + 1, N_F))
    w[:N_F, :N_F] = np.eye(N_F)
    return w


def fit_adapter(
    x: np.ndarray, y: np.ndarray, w0: np.ndarray, lam: float = LAMBDA
) -> np.ndarray:
    """Softmax regression by gradient descent, L2 towards ``w0``."""
    w = w0.copy()
    if len(x) == 0:
        return w
    f = _features(x)
    onehot = np.eye(N_F)[y]
    for _ in range(STEPS):
        z = f @ w
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        grad = f.T @ (p - onehot) / len(x) + lam * (w - w0) / len(x)
        w -= LR * grad
    return w


def predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (_features(x) @ w).argmax(axis=1)


def run_system(rows: Sequence[dict], seed: int = 0) -> dict[str, Any]:
    x = np.stack([r["x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    top_ok = np.array([r["baro_top_ok"] for r in rows])
    baro_f = np.array([r["baro_fault_ok"] for r in rows])
    w_prior = prior_weights(x.shape[1])
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"n": len(rows), "by_k": {}}
    for k in KS:
        acc: dict[str, list[float]] = {m: [] for m in (
            "adapted_fault", "adapted_joint", "scratch_fault", "scratch_joint",
            "frozen_joint", "baro_joint")}  # fmt: skip
        for _ in range(N_DRAWS if k else 1):
            train = (
                rng.choice(len(rows), size=k, replace=False) if k else np.zeros(0, int)
            )
            test = np.setdiff1d(np.arange(len(rows)), train)
            wa = fit_adapter(x[train], y[train], w_prior)
            ws = fit_adapter(x[train], y[train], np.zeros_like(w_prior))
            for name, w in (("adapted", wa), ("scratch", ws)):
                ok = predict(w, x[test]) == y[test]
                acc[f"{name}_fault"].append(float(ok.mean()))
                acc[f"{name}_joint"].append(float((ok & top_ok[test]).mean()))
            frozen = predict(w_prior, x[test]) == y[test]
            acc["frozen_joint"].append(float((frozen & top_ok[test]).mean()))
            acc["baro_joint"].append(float((baro_f[test] & top_ok[test]).mean()))
        out["by_k"][k] = {
            m: mean_ci(v) if len(v) > 1 else (v[0], 0.0) for m, v in acc.items()
        }
    return out


def render(rep: dict[str, Any]) -> str:
    lines = [
        "# R13b probe: how many labelled incidents does the fault specialist need on a new system?",
        "",
        "Exploratory probe (not a pre-registered goal). RE1 (375 incidents, metrics only). "
        "The frozen G4c fault specialist plus a softmax adapter fitted on k labelled "
        "incidents of the target system (L2 towards the frozen model), tested on the rest. "
        f"Mean ± 95% t-CI over {N_DRAWS} random draws of the k incidents. Joint = BARO's top "
        "service right and the fault type right.",
        "",
        "| System | k | Adapted joint | Scratch joint | Frozen joint | BARO joint | Adapted fault acc. |",
        "|---|---|---|---|---|---|---|",
    ]

    def pm(v: Sequence[float]) -> str:
        return f"{100 * v[0]:.1f} ± {100 * v[1]:.1f}"

    for system, res in rep["systems"].items():
        for k, r in res["by_k"].items():
            lines.append(
                f"| {system} (n = {res['n']}) | {k} | {pm(r['adapted_joint'])} "
                f"| {pm(r['scratch_joint'])} | {pm(r['frozen_joint'])} "
                f"| {pm(r['baro_joint'])} | {pm(r['adapted_fault'])} |"
            )
    lines += [
        "",
        f"Adapter: input = 6 log-probabilities + {len(ff.KINDS)} kind log z-scores + bias; "
        f"L2 λ = {LAMBDA}; {STEPS} gradient steps at {LR}. Frozen model SHA-256 "
        f"`{rep['model_sha256'][:16]}`.",
        "",
        "Reproduce: `python scripts/r13b_kshot.py --team-from runs:r11-team`.",
        "",
    ]
    return "\n".join(lines)
