"""Can a learned code express faults it was never trained on? (R8 step 4, goal G2b)

A designed class-id code has a fixed vocabulary: a node can only name the faults it was
trained to name. Does a learned code carry information that lets the system handle faults it
never saw, without retraining the sensors (the node encoders)? That is extensibility.

For each of several fixed seen/unseen splits of the 17 diagnosable faults (12 seen, 5 unseen
by default) and each of 10 run-level replicates (replicate ``r`` uses split ``r % n_splits``):

1. the encoders and the centre are trained on the *seen* faults and normal data only (runs of
   unseen faults are dropped from training and validation altogether), the centre is
   calibrated to 5% false alarms on seen-only validation runs, and the encoders are frozen;
2. **detection**: the alarm rate of that centre on test runs of the unseen faults at its
   calibrated threshold, and the detection delay; how it maps unseen faults to known classes;
3. **separability**: a linear probe and k-means (normalised mutual information) on the
   frozen decoded messages of the unseen faults;
4. **few-shot extension**: with the encoders frozen, the centre is retrained on the seen data
   plus ``k`` in {1, 3, 10} labelled training runs of each unseen fault (a fresh 18-class
   centre, calibrated the same way); macro-F1 over all 17 faults and over the unseen ones.

Codes: ``local`` (the designed class-id code: top classes of a per-node classifier), ``e2e``,
``e2e_aux`` (e2e plus a reconstruction term), ``vq``, ``features`` (the hand-designed PCA/FFT
code) and ``oracle`` (no byte limit: float32 local features plus local posteriors; a true
upper bound in the seen-fault setting). Hyperparameters are chosen on the tuning block with
its own split, scored on seen-fault validation macro-F1.

Everything that is not training is pure (``merge_partials``, ``g2b_tests``, ``g2b_verdict``,
``build_report``), so partial runs on different machines combine.
"""

import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy import stats as scipy_stats

from ..envs.plant.centres import _LearnedCentre
from ..envs.plant.classifiers import make_classifier
from ..envs.plant.encoders import make_encoder
from ..envs.plant.learned_codes import (
    DEFAULT_HP,
    NOVEL_METHODS,
    Featuriser,
    LocalClassifierEncoder,
    fit_encoder,
    kmeans,
    nearest,
)
from ..envs.plant.replicates import DIAGNOSABLE_FAULTS, Protocol, Replicate, summarise
from ..envs.plant.rieth import RiethData, to_runs
from ..envs.plant.runner import macro_f1
from ..envs.plant.tep import ONSET_TEST
from ..envs.plant.windows import labelled_windows
from ..evaluation.stats import holm_correction
from .plant_learned_codes import CENTRE_FULL, TOLERANCE, Scale

PARTIAL_KIND = "plant_novel_faults_partial"
CODES = ("local", "e2e", "e2e_aux", "vq", "features", "oracle")
LEARNED = ("e2e", "e2e_aux", "vq")
G2B_LEARNED = ("e2e", "e2e_aux")
G2B_BUDGETS = (1, 2, 4)
REPORT_BUDGETS = (1, 2, 4, 8)
SHOTS = (1, 3, 10)
G2B_SHOT = 3
N_SPLITS = 5
N_UNSEEN = 5
SPLIT_SEED = 20261024
CODE_LABELS = {
    "local": "designed class-id code (local classifier)",
    "e2e": "(L1) end-to-end learned",
    "e2e_aux": "(L1+) end-to-end + reconstruction",
    "vq": "(L2) residual k-means (unsupervised)",
    "features": "(c) quantised hand features (PCA/FFT)",
    "oracle": "oracle (float32 local features + posteriors)",
}


# ------------------------------------------------------------------------------- splits
def make_splits(
    faults: Sequence[int] = DIAGNOSABLE_FAULTS,
    n_splits: int = N_SPLITS,
    n_unseen: int = N_UNSEEN,
    seed: int = SPLIT_SEED,
) -> list[tuple[int, ...]]:
    """``n_splits`` distinct random sets of ``n_unseen`` faults to hold out (sorted ids).

    Raises:
        ValueError: If the sizes are impossible or there are fewer distinct sets than
            ``n_splits``
    """
    faults = tuple(faults)
    if not 0 < n_unseen < len(faults):
        raise ValueError("n_unseen must leave at least one seen and one unseen fault")
    from math import comb

    if comb(len(faults), n_unseen) < n_splits:
        raise ValueError("Fewer distinct splits exist than requested")
    rng = np.random.default_rng(seed)
    out: list[tuple[int, ...]] = []
    while len(out) < n_splits:
        pick = tuple(sorted(int(f) for f in rng.choice(faults, n_unseen, False)))
        if pick not in out:
            out.append(pick)
    return out


@dataclass(frozen=True)
class Setting:
    """One seen/unseen split in class space (class 0 is normal, faults are 1..n)."""

    faults: tuple[int, ...]
    unseen: tuple[int, ...]

    def __post_init__(self) -> None:
        if not set(self.unseen) <= set(self.faults) or not self.unseen:
            raise ValueError("unseen must be a non-empty subset of the faults")
        if len(self.unseen) == len(self.faults):
            raise ValueError("At least one fault must be seen")

    @property
    def n_classes(self) -> int:
        return len(self.faults) + 1

    @property
    def seen(self) -> tuple[int, ...]:
        return tuple(f for f in self.faults if f not in self.unseen)

    @property
    def seen_classes(self) -> np.ndarray:
        """Full class index of each seen fault, in order (compact class c is entry c - 1)."""
        return np.array([self.faults.index(f) + 1 for f in self.seen], dtype=np.int64)

    @property
    def unseen_classes(self) -> np.ndarray:
        return np.array([self.faults.index(f) + 1 for f in self.unseen], dtype=np.int64)

    @property
    def n_seen_classes(self) -> int:
        return len(self.seen) + 1

    @property
    def compact(self) -> np.ndarray:
        """Full class -> compact class of the seen-only problem (-1 for unseen faults)."""
        out = np.full(self.n_classes, -1, dtype=np.int64)
        out[0] = 0
        out[self.seen_classes] = np.arange(1, len(self.seen) + 1)
        return out


def run_classes(d: dict[str, np.ndarray]) -> np.ndarray:
    """Fault class of every run (0 for fault-free runs): the largest label in the run."""
    out = np.zeros(int(d["run"].max()) + 1, dtype=np.int64)
    np.maximum.at(out, d["run"], d["labels"])
    return out


def seen_view(
    d: dict[str, np.ndarray], setting: Setting
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Windows of the runs that are not unseen faults, with compact labels.

    Whole runs of an unseen fault are dropped, including their fault-free start, so
    nothing of an unseen fault reaches training or calibration.
    """
    rc = run_classes(d)
    keep = ~np.isin(rc[d["run"]], setting.unseen_classes)
    view = {
        "windows": d["windows"][keep],
        "labels": setting.compact[d["labels"][keep]],
        "run": d["run"][keep],
    }
    return keep, view


# ------------------------------------------------------------------------------ metrics
def nmi(labels: np.ndarray, clusters: np.ndarray) -> float:
    """Normalised mutual information (arithmetic mean of the entropies) of two labelings."""
    a = np.unique(labels, return_inverse=True)[1]
    b = np.unique(clusters, return_inverse=True)[1]
    n = len(a)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1.0)
    pab = table / n
    pa, pb = pab.sum(axis=1), pab.sum(axis=0)
    nz = pab > 0
    mi = float((pab[nz] * np.log(pab[nz] / np.outer(pa, pb)[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    denom = 0.5 * (ha + hb)
    return mi / denom if denom > 0 else 0.0


def separability(
    x: np.ndarray,
    labels: np.ndarray,
    run: np.ndarray,
    unseen_classes: np.ndarray,
    seed: int = 0,
) -> dict[str, float]:
    """How well frozen messages separate the unseen faults, without labels or with a probe.

    Uses the fault windows of the unseen faults only. ``probe_acc`` is a linear probe
    trained on every other run and tested on the rest (runs, not windows, so a run never
    sits on both sides); ``nmi`` is the normalised mutual information of a k-means with one
    cluster per unseen fault (best of 3 starts) against the true faults.
    """
    m = np.isin(labels, unseen_classes)
    xs, ys, rs = x[m].astype(np.float64), labels[m], run[m]
    sd = xs.std(axis=0)
    z = ((xs - xs.mean(axis=0)) / np.where(sd > 1e-8, sd, 1.0)).astype(np.float32)
    y = np.searchsorted(np.sort(unseen_classes), ys)
    n_class = len(unseen_classes)
    half = np.isin(rs, np.unique(rs)[::2])
    clf = make_classifier("logreg", n_class, seed, l2=1e-2, max_iter=100)
    clf.fit(z[half], y[half])
    probe = float((clf.predict(z[~half]) == y[~half]).mean())
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(3):
        centres = kmeans(z, n_class, rng, iters=30)
        assign = nearest(z, centres)
        inertia = float(((z - centres[assign]) ** 2).sum())
        if inertia < best_inertia:
            best, best_inertia = assign, inertia
    assert best is not None
    return {"probe_acc": probe, "nmi": nmi(y, best), "chance": 1.0 / n_class}


def run_detection(
    alarm: np.ndarray,
    labels: np.ndarray,
    run: np.ndarray,
    n_seen: np.ndarray,
    run_mask: np.ndarray,
    onset: int = ONSET_TEST,
) -> dict[str, float]:
    """Window alarm rate, run detection rate and mean delay over the runs in ``run_mask``.

    Only windows that have seen a fault count (``labels > 0``). A run is detected when any of
    them alarms; its delay is the samples from the onset to the first such window.
    """
    in_runs = run_mask[run] & (labels > 0)
    rate = float(alarm[in_runs].mean()) if in_runs.any() else float("nan")
    detected, delays = [], []
    for r in np.flatnonzero(run_mask):
        w = np.flatnonzero((run == r) & (labels > 0))
        if w.size == 0:
            continue
        hit = np.flatnonzero(alarm[w])
        detected.append(hit.size > 0)
        if hit.size:
            delays.append(float(n_seen[w[hit[0]]] - onset))
    return {
        "alarm_rate": rate,
        "detect_rate": float(np.mean(detected)) if detected else float("nan"),
        "delay": float(np.mean(delays)) if delays else float("nan"),
    }


def nearest_known(
    fault_c: np.ndarray,
    alarm: np.ndarray,
    labels: np.ndarray,
    setting: Setting,
) -> dict[str, list[float]]:
    """How the seen-only centre treats each unseen fault: the known class it maps it to.

    Returns ``{unseen fault id: [modal known fault id, share of its alarms that go there,
    alarm rate]}`` over the fault windows of that unseen fault.
    """
    out: dict[str, list[float]] = {}
    for fault, cls in zip(setting.unseen, setting.unseen_classes):
        w = labels == cls
        hit = w & alarm
        if not w.any():
            continue
        if not hit.any():
            out[str(fault)] = [-1.0, float("nan"), 0.0]
            continue
        counts = np.bincount(fault_c[hit], minlength=setting.n_seen_classes)
        modal = int(counts.argmax())
        out[str(fault)] = [
            float(setting.seen[modal - 1]),
            float(counts[modal] / hit.sum()),
            float(hit.sum() / w.sum()),
        ]
    return out


def fingerprint(obj: Any) -> str:
    """Hash of every array and scalar reachable from ``obj`` (to prove encoders stay frozen)."""
    h = hashlib.sha1()

    def walk(o: Any, depth: int) -> None:
        if isinstance(o, np.ndarray):
            h.update(np.ascontiguousarray(o).tobytes())
        elif isinstance(o, (list, tuple)):
            for item in o:
                walk(item, depth)
        elif isinstance(o, dict):
            for key in sorted(o, key=str):
                walk(o[key], depth)
        elif isinstance(o, (int, float, str, bool, type(None))):
            h.update(repr(o).encode())
        elif hasattr(o, "__dict__") and depth < 8:
            walk(vars(o), depth + 1)

    walk(obj, 0)
    return h.hexdigest()


# ----------------------------------------------------------------------------- the codes
class FullOracle:
    """No byte limit: standardised float32 local features plus each node's full posteriors.

    The posteriors are those of per-node classifiers trained on the seen faults, so this is
    a true upper bound for the seen-fault problem: it has everything every code can send.
    """

    def __init__(self, featuriser: Featuriser, local: LocalClassifierEncoder):
        self.featuriser, self.local = featuriser, local

    def posteriors(
        self, local: LocalClassifierEncoder, feats: np.ndarray
    ) -> np.ndarray:
        n = self.featuriser.layout.n_nodes
        out = [
            m.predict_proba(feats[:, i, self.featuriser.mask[i]]).astype(np.float32)
            for i, m in enumerate(local.models[:n])
        ]
        return np.stack(out, axis=1).reshape(len(feats), -1)

    def decode(self, windows: np.ndarray) -> np.ndarray:
        feats = self.featuriser(windows)
        return np.concatenate(
            [feats.reshape(len(feats), -1), self.posteriors(self.local, feats)], axis=1
        )

    def crossfit(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        run: np.ndarray,
        folds: int = 3,
        seed: int = 0,
    ) -> np.ndarray:
        """``decode`` of training windows with posteriors from classifiers that never saw the run."""
        feats = self.featuriser(windows)
        rng = np.random.default_rng(seed)
        runs = np.unique(run)
        group = dict(zip(rng.permutation(runs), np.arange(len(runs)) % folds))
        fold = np.array([group[r] for r in run])
        post = np.zeros((len(feats), self.local.n_features), np.float32)
        for f in range(folds):
            held = fold == f
            if not held.any() or held.all():
                continue
            twin = LocalClassifierEncoder(
                self.local.budget,
                self.local.layout,
                self.featuriser,
                self.local.n_classes,
                **self.local.params,
            ).fit(feats[~held], labels[~held], seed + 10 * (f + 1))
            post[held] = self.posteriors(twin, feats[held])
        return np.concatenate([feats.reshape(len(feats), -1), post], axis=1)


@dataclass
class Coded:
    """A fitted, frozen code: ``decode(windows)`` is what the centre reads."""

    code: str
    budget: int
    frozen: Any  # everything that must not change while the centre is extended
    decode: Callable[[np.ndarray], np.ndarray]
    x_seen_train: Optional[np.ndarray] = (
        None  # cross-fitted decoding of seen training windows
    )


def build_coded(
    rep: Replicate,
    setting: Setting,
    code: str,
    budget: int,
    hp: dict[str, Any],
    seed: int,
    scale: Scale,
    seen_train: dict[str, np.ndarray],
) -> Coded:
    """Fit a code on the seen-only training windows (labels are seen-fault classes)."""
    n_classes = setting.n_seen_classes
    if code == "features":
        normal = seen_train["windows"][seen_train["labels"] == 0]
        enc = make_encoder("features", budget, rep.layout, seed).fit(normal)
        return Coded(code, budget, enc, lambda w: enc.decode(enc.encode_windows(w)))
    featuriser = Featuriser(rep.layout).fit(seen_train["windows"])
    if code == "oracle":
        base = {**DEFAULT_HP["local"], **scale.overrides.get("local", {})}
        feats = featuriser(seen_train["windows"])
        local = LocalClassifierEncoder(
            1,
            rep.layout,
            featuriser,
            n_classes,
            hidden=base["hidden"],
            l2=base["l2"],
            max_iter=base["max_iter"],
        ).fit(feats, seen_train["labels"], seed)
        oracle = FullOracle(featuriser, local)
        x_cross = oracle.crossfit(
            seen_train["windows"],
            seen_train["labels"],
            seen_train["run"],
            int(base["folds"]),
            seed,
        )
        return Coded(code, 0, oracle, oracle.decode, x_cross)
    if code not in NOVEL_METHODS:
        raise ValueError(f"Unknown code {code!r}; expected one of {CODES}")
    merged = {**DEFAULT_HP[code], **scale.overrides.get(code, {}), **hp}
    enc, x_cross = fit_encoder(
        code, budget, rep.layout, featuriser, seen_train, n_classes, merged, seed
    )
    return Coded(
        code, budget, enc, lambda w: enc.decode(enc.encode_windows(w)), x_cross
    )


def centre_params(code: str, hp: dict[str, Any], scale: Scale) -> dict[str, Any]:
    """The centre's settings; only the oracle (a much wider input) may tune its hidden size."""
    params = dict(scale.centre)
    if code == "oracle" and "hidden" in hp and params["classifier"] == "mlp":
        params["hidden"] = hp["hidden"]
    return params


def make_centre(
    n_classes: int, params: dict[str, Any], seed: int, target_far: float
) -> _LearnedCentre:
    rest = {k: v for k, v in params.items() if k != "classifier"}
    return _LearnedCentre(
        n_classes, params["classifier"], seed, target_far=target_far, **rest
    )


@dataclass
class Stage1:
    """The seen-only system of one cell: frozen code, calibrated centre, seen features."""

    coded: Coded
    centre: _LearnedCentre
    params: dict[str, Any]
    keep_train: np.ndarray
    x_train: np.ndarray
    train: dict[str, np.ndarray]
    x_val: np.ndarray
    val: dict[str, np.ndarray]


def fit_stage1(
    rep: Replicate,
    setting: Setting,
    code: str,
    budget: int,
    hp: dict[str, Any],
    seed: int,
    scale: Scale,
) -> Stage1:
    """Train the code and the centre on the seen faults and normal data only."""
    keep, train = seen_view(rep.train, setting)
    _, val = seen_view(rep.val, setting)
    coded = build_coded(rep, setting, code, budget, hp, seed, scale, train)
    x_train = (
        coded.x_seen_train
        if coded.x_seen_train is not None
        else coded.decode(train["windows"])
    )
    x_val = coded.decode(val["windows"])
    params = centre_params(code, hp, scale)
    centre = make_centre(
        setting.n_seen_classes, params, seed, rep.protocol.target_far
    ).fit_features(
        x_train,
        train["labels"],
        train["run"],
        {"features": x_val, "labels": val["labels"], "run": val["run"]},
    )
    return Stage1(coded, centre, params, keep, x_train, train, x_val, val)


def seen_val_f1(s1: Stage1, setting: Setting) -> float:
    """Macro-F1 over the seen faults on seen validation windows at the calibrated threshold."""
    margin, fault = s1.centre.score_features(s1.x_val, s1.val["run"])
    pred = np.where(margin > s1.centre.tau, fault, 0)
    return macro_f1(s1.val["labels"], pred, setting.n_seen_classes)[0]


# ------------------------------------------------------------------------------- one cell
def test_windows(rep: Replicate) -> dict[str, np.ndarray]:
    """Labelled windows of a replicate's test runs (all 17 faults), built once."""
    cached = getattr(rep, "_novel_test", None)
    if cached is None:
        p = rep.protocol
        runs = to_runs(rep.data, rep.normaliser, rep.selection, "test", tuple(p.faults))
        cached = labelled_windows(
            runs, p.class_of_fault, rep.layout, p.window, p.stride
        )
        rep._novel_test = cached  # type: ignore[attr-defined]
    return cached


def extension_runs(
    train: dict[str, np.ndarray], setting: Setting, k: int, replicate: int
) -> np.ndarray:
    """Training run ids the centre may see labelled: ``k`` per unseen fault, nested in ``k``."""
    rc = run_classes(train)
    chosen = []
    for cls in setting.unseen_classes:
        ids = np.flatnonzero(rc == cls)
        order = np.random.default_rng([4242, replicate, int(cls)]).permutation(ids)
        chosen.append(order[:k])
    return np.concatenate(chosen)


def extend_centre(
    rep: Replicate,
    setting: Setting,
    s1: Stage1,
    x_train_all: np.ndarray,
    k: int,
    replicate: int,
    seed: int,
) -> _LearnedCentre:
    """Retrain only the centre (full class set) on seen data plus ``k`` runs per unseen fault.

    The code is not touched: ``x_train_all`` are the frozen decodings of the training windows.
    """
    train = rep.train
    pick = np.isin(train["run"], extension_runs(train, setting, k, replicate))
    use = s1.keep_train | pick
    centre = make_centre(
        setting.n_classes, s1.params, seed, rep.protocol.target_far
    ).fit_features(
        x_train_all[use],
        train["labels"][use],
        train["run"][use],
        {"features": s1.x_val, "labels": s1.val["labels"], "run": s1.val["run"]},
    )
    return centre


def macro_f1_split(
    labels: np.ndarray, pred: np.ndarray, setting: Setting
) -> tuple[float, float, float]:
    """Macro-F1 over all faults, the unseen ones and the seen ones (full class space)."""
    _, per = macro_f1(labels, pred, setting.n_classes)
    unseen = per[setting.unseen_classes - 1]
    seen = per[setting.seen_classes - 1]
    return float(per.mean()), float(unseen.mean()), float(seen.mean())


def run_cell(
    rep: Replicate,
    setting: Setting,
    code: str,
    budget: int,
    hp: dict[str, Any],
    replicate: int,
    split: int,
    scale: Scale,
    shots: Sequence[int] = SHOTS,
) -> dict[str, Any]:
    """Train one code on one replicate/split and measure the four things (see the module doc)."""
    start = time.perf_counter()
    seed = 100 * int(replicate)
    s1 = fit_stage1(rep, setting, code, budget, hp, seed, scale)
    before = fingerprint(s1.coded.frozen)
    test = test_windows(rep)
    x_test = s1.coded.decode(test["windows"])
    lab, run = test["labels"], test["run"]
    centre = s1.centre

    # (i) detection, seen-fault quality and the nearest-known-class failure
    margin, fault_c = centre.score_features(x_test, run)
    alarm = margin > centre.tau
    rc = run_classes(test)
    unseen_runs = np.isin(rc, setting.unseen_classes)
    seen_fault_runs = (rc > 0) & ~unseen_runs
    keep_w = ~unseen_runs[run]
    pred_c = np.where(alarm, fault_c, 0)
    seen_f1 = macro_f1(
        setting.compact[lab[keep_w]], pred_c[keep_w], setting.n_seen_classes
    )[0]
    det_unseen = run_detection(alarm, lab, run, test["n_seen"], unseen_runs)
    det_seen = run_detection(alarm, lab, run, test["n_seen"], seen_fault_runs)
    mapping = nearest_known(fault_c, alarm, lab, setting)
    shares = [v[1] for v in mapping.values() if np.isfinite(v[1])]

    # (iii) separability of the frozen messages of the unseen faults
    sep = separability(x_test, lab, run, setting.unseen_classes, seed)

    # (ii) few-shot extension, with the code frozen
    full_pred0 = np.where(alarm, setting.seen_classes[np.maximum(fault_c, 1) - 1], 0)
    out_shots: dict[str, dict[str, float]] = {}
    all0, un0, se0 = macro_f1_split(lab, full_pred0, setting)
    out_shots["0"] = {"f1_all": all0, "f1_unseen": un0, "f1_seen": se0}
    x_train_all = s1.coded.decode(rep.train["windows"])
    x_train_all[s1.keep_train] = s1.x_train
    for k in shots:
        ext = extend_centre(rep, setting, s1, x_train_all, k, replicate, seed)
        m, f = ext.score_features(x_test, run)
        pred = np.where(m > ext.tau, f, 0)
        a, u, s = macro_f1_split(lab, pred, setting)
        far = float((m[lab == 0] > ext.tau).mean())
        out_shots[str(k)] = {"f1_all": a, "f1_unseen": u, "f1_seen": s, "far": far}
    after = fingerprint(s1.coded.frozen)
    if before != after:
        raise AssertionError("The code changed while the centre was extended")

    row: dict[str, Any] = {
        "code": code,
        "budget": int(budget),
        "replicate": int(replicate),
        "split": int(split),
        "unseen": [int(f) for f in setting.unseen],
        "seen_f1": float(seen_f1),
        "far": float((margin[lab == 0] > centre.tau).mean()),
        "unseen_alarm_rate": det_unseen["alarm_rate"],
        "unseen_detect_rate": det_unseen["detect_rate"],
        "unseen_delay": det_unseen["delay"],
        "seen_alarm_rate": det_seen["alarm_rate"],
        "seen_delay": det_seen["delay"],
        "map_share": float(np.mean(shares)) if shares else float("nan"),
        "mapping": mapping,
        "probe_acc": sep["probe_acc"],
        "nmi": sep["nmi"],
        "chance": sep["chance"],
        "frozen_ok": True,
        "hp": dict(hp),
        "seconds": time.perf_counter() - start,
    }
    for k, v in out_shots.items():
        for name, val in v.items():
            row[f"{name}_k{k}"] = val
    return row


# --------------------------------------------------------------------------------- tuning
def staged_search(
    start: dict[str, Any],
    grids: Sequence[tuple[str, Sequence[Any]]],
    score: Callable[[dict[str, Any]], float],
    tol: float = TOLERANCE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Coordinate search as in step 3: the earliest value within ``tol`` of the best wins."""
    best = dict(start)
    trace: list[dict[str, Any]] = []
    current: Optional[float] = None
    for name, values in grids:
        scores: list[tuple[Any, float]] = []
        for value in values:
            if current is not None and value == best[name]:
                scores.append((value, current))
                continue
            t0 = time.perf_counter()
            sc = float(score({**best, name: value}))
            trace.append(
                {
                    "stage": name,
                    "value": value,
                    "val_f1": sc,
                    "seconds": time.perf_counter() - t0,
                }
            )
            scores.append((value, sc))
            if current is None and value == best[name]:
                current = sc
        top = max(s for _, s in scores)
        pick = next(v for v, s in scores if s >= top - tol)
        best[name] = pick
        current = dict(scores)[pick]
    return best, trace


def tune_hp(
    rep: Replicate,
    setting: Setting,
    code: str,
    budget: int,
    scale: Scale,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose a code's hyperparameters on the tuning block, scored on seen-fault F1."""
    grids = scale.grids.get(code, ())
    if code == "oracle":
        start = {"hidden": scale.centre.get("hidden", 64)}
    else:
        start = {**DEFAULT_HP.get(code, {}), **scale.overrides.get(code, {})}
    if not grids:
        return {}, []

    def score(hp: dict[str, Any]) -> float:
        return seen_val_f1(
            fit_stage1(rep, setting, code, budget, hp, seed, scale), setting
        )

    return staged_search(start, grids, score)


SCALES = {
    "full": Scale(
        "full",
        CENTRE_FULL,
        {
            "e2e": (
                ("bottleneck", ("ste", "gumbel")),
                ("hidden", (64, 128)),
                ("epochs", (20, 40)),
                ("lr", (2e-3, 5e-3)),
            ),
            "e2e_aux": (
                ("bottleneck", ("ste", "gumbel")),
                ("hidden", (64, 128)),
                ("epochs", (20, 40)),
                ("lr", (2e-3, 5e-3)),
            ),
            "vq": (("whiten", (False, True)), ("dim", (8, 16, 32))),
            "local": (("hidden", (32, 64, 128)), ("l2", (1e-3, 1e-2, 1e-1))),
            "oracle": (("hidden", (64, 256)),),
        },
    ),
    "smoke": Scale(
        "smoke",
        {**CENTRE_FULL, "hidden": 32, "max_iter": 40, "context": 2},
        {
            "e2e": (("bottleneck", ("ste", "gumbel")),),
            "e2e_aux": (("bottleneck", ("ste",)),),
            "vq": (("whiten", (False, True)),),
            "local": (("hidden", (32, 64)),),
            "oracle": (),
        },
        {
            "e2e": {"epochs": 6},
            "e2e_aux": {"epochs": 6},
            "vq": {},
            "local": {"max_iter": 40, "folds": 2},
        },
    ),
}


# ------------------------------------------------------------------------------ workers
_STATE: dict[str, Any] = {}


def init_worker(
    cache_dir: Optional[str],
    protocol: dict,
    scale: str,
    torch_threads: int,
    codes: Sequence[str],
    splits: Sequence[Sequence[int]],
) -> None:
    """Process initialiser: load the data once, cap torch's threads."""
    if any(c in ("e2e", "e2e_aux") for c in codes):
        import torch

        torch.set_num_threads(max(1, torch_threads))
    _STATE["data"] = RiethData.from_cache(Path(cache_dir) if cache_dir else None)
    _STATE["protocol"] = Protocol(**protocol)
    _STATE["scale"] = SCALES[scale]
    _STATE["splits"] = [tuple(s) for s in splits]
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


def _setting(split: int) -> Setting:
    return Setting(tuple(_STATE["protocol"].faults), _STATE["splits"][split])


def tune_task(task: tuple[str, int]) -> dict[str, Any]:
    code, budget = task
    start = time.perf_counter()
    tuning_split = len(_STATE["splits"]) - 1  # the last split is reserved for tuning
    hp, trace = tune_hp(
        _replicate(None), _setting(tuning_split), code, budget, _STATE["scale"]
    )
    return {
        "code": code,
        "budget": budget,
        "hp": hp,
        "trace": trace,
        "seconds": time.perf_counter() - start,
    }


def cell_task(task: tuple[str, int, int, dict]) -> dict[str, Any]:
    code, budget, replicate, hp = task
    split = replicate % (len(_STATE["splits"]) - 1)
    return run_cell(
        _replicate(replicate),
        _setting(split),
        code,
        budget,
        hp,
        replicate,
        split,
        _STATE["scale"],
    )


def run_groups(
    groups: Sequence[tuple[str, int]],
    replicates: Sequence[int],
    workers: int,
    protocol: dict,
    scale: str,
    cache_dir: Optional[str],
    torch_threads: int,
    splits: Sequence[Sequence[int]],
    on_group_done: Callable[[dict[str, Any]], None],
    log: Callable[[str], None] = print,
) -> None:
    """Tune and evaluate (code, budget) groups on one process pool (cells start after tuning)."""
    codes = sorted({c for c, _ in groups})
    waiting = {g: len(replicates) for g in groups}
    state: dict[tuple[str, int], dict[str, Any]] = {
        g: {"rows": [], "tuning": None} for g in groups
    }
    started = {g: time.perf_counter() for g in groups}
    total = sum(waiting.values())
    done_cells, t0 = 0, time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(cache_dir, protocol, scale, torch_threads, codes, splits),
    ) as pool:
        pending: dict[Any, tuple[str, tuple[str, int]]] = {}
        for group in groups:
            pending[pool.submit(tune_task, group)] = ("tune", group)
        log(f"{total} cells in {len(groups)} groups on {workers} workers")
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                kind, group = pending.pop(future)
                result = future.result()
                if kind == "tune":
                    state[group]["tuning"] = result
                    log(f"  tuned {group}: {result['hp']} ({result['seconds']:.0f}s)")
                    for r in replicates:
                        task = (group[0], group[1], r, result["hp"])
                        pending[pool.submit(cell_task, task)] = ("cell", group)
                    continue
                state[group]["rows"].append(result)
                done_cells += 1
                waiting[group] -= 1
                if done_cells % 5 == 0 or done_cells == total:
                    log(
                        f"  {done_cells}/{total} cells, {time.perf_counter() - t0:.0f}s"
                    )
                if waiting[group] == 0:
                    on_group_done(
                        {
                            "code": group[0],
                            "budget": group[1],
                            "tuning": state[group]["tuning"],
                            "rows": state[group]["rows"],
                            "seconds": time.perf_counter() - started[group],
                        }
                    )


# ------------------------------------------------------------------------------- merging
def merge_partials(partials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine partial result files; later rows replace earlier ones with the same key.

    Raises:
        ValueError: If there are no partials or they disagree on protocol, scale or splits
    """
    if not partials:
        raise ValueError("Nothing to merge")
    first = partials[0]
    for p in partials[1:]:
        for key in ("protocol", "scale", "splits"):
            if p[key] != first[key]:
                raise ValueError(f"Partial results disagree on {key}")
    rows: dict[tuple, dict[str, Any]] = {}
    tuning: dict[str, Any] = {}
    cpu = 0.0
    for p in partials:
        cpu += float(p.get("seconds", 0.0))
        for r in p["rows"]:
            rows[(r["code"], int(r["budget"]), int(r["replicate"]))] = r
        t = p.get("tuning")
        if t:
            tuning[f"{t['code']}|{t['budget']}"] = t
    return {
        "protocol": first["protocol"],
        "scale": first["scale"],
        "splits": first["splits"],
        "commits": sorted({str(p.get("commit")) for p in partials}),
        "partials": len(partials),
        "cpu_seconds": cpu,
        "tuning": tuning,
        "rows": [rows[k] for k in sorted(rows)],
    }


def read_partials(sources: Iterable[Path]) -> list[dict[str, Any]]:
    """Load partial result files (``kind`` ``plant_novel_faults_partial``) from files or folders."""
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
        if isinstance(data, dict) and data.get("kind") == PARTIAL_KIND:
            out.append(data)
    return out


# -------------------------------------------------------------------------- statistics
def by_replicate(
    rows: Sequence[dict[str, Any]], code: str, budget: int, metric: str
) -> dict[int, float]:
    """Replicate -> metric for one cell (NaN skipped)."""
    out: dict[int, float] = {}
    for r in rows:
        v = r.get(metric)
        if (
            r["code"] == code
            and int(r["budget"]) == budget
            and v is not None
            and np.isfinite(v)
        ):
            out[int(r["replicate"])] = float(v)
    return dict(sorted(out.items()))


def paired(a: dict[int, float], b: dict[int, float]) -> np.ndarray:
    common = sorted(set(a) & set(b))
    return np.array([a[k] - b[k] for k in common], dtype=float)


def _paired_test(diff: np.ndarray) -> tuple[float, float, float]:
    mean, half = summarise(diff)
    if diff.size > 1 and np.ptp(diff) > 0:
        p = float(scipy_stats.ttest_1samp(diff, 0.0).pvalue)
    else:
        p = 1.0
    return mean, half, p


def g2b_tests(
    rows: Sequence[dict[str, Any]],
    metric: str = f"f1_unseen_k{G2B_SHOT}",
    budgets: Sequence[int] = REPORT_BUDGETS,
    learned: Sequence[str] = LEARNED,
    designed: str = "local",
) -> list[dict[str, Any]]:
    """Each learned code against the designed class-id code on one metric, paired by replicate.

    Replicates differ in split and in runs, so the interval covers both. A test is a ``win``
    when the 95% t-interval of the mean difference lies above zero (``loss`` below).
    """
    tests = []
    for budget in budgets:
        base = by_replicate(rows, designed, budget, metric)
        for code in learned:
            cand = by_replicate(rows, code, budget, metric)
            diff = paired(cand, base)
            if diff.size == 0:
                continue
            mean, half, p = _paired_test(diff)
            ok = bool(np.isfinite(half))
            tests.append(
                {
                    "code": code,
                    "budget": budget,
                    "n": int(diff.size),
                    "learned": float(np.mean(list(cand.values()))),
                    "designed": float(np.mean(list(base.values()))),
                    "diff": mean,
                    "half": half,
                    "p": p,
                    "win": bool(ok and mean - half > 0),
                    "loss": bool(ok and mean + half < 0),
                }
            )
    if tests:
        adjusted = holm_correction([t["p"] for t in tests])
        for t, a in zip(tests, adjusted):
            t["p_holm"] = float(a)
            t["win_holm"] = bool(t["win"] and a < 0.05)
    return tests


def g2b_verdict(
    tests: Sequence[dict[str, Any]],
    budgets: Sequence[int] = G2B_BUDGETS,
    pass_codes: Sequence[str] = G2B_LEARNED,
    kill_codes: Sequence[str] = LEARNED,
    kill_budgets: Sequence[int] = REPORT_BUDGETS,
    n_required: int = 10,
) -> tuple[str, str]:
    """Apply the G2b rule: ``("PASS" | "KILL" | "INCONCLUSIVE" | "INCOMPLETE", why)``.

    PASS: at 1, 2 or 4 B, e2e or e2e_aux beats the designed code on few-shot (k = 3)
    unseen-fault macro-F1 with the 95% CI of the paired difference over replicates (which
    span the splits) above zero. KILL: the designed code is at least as good (mean difference
    <= 0) as every learned code at every budget. INCONCLUSIVE: some learned code is ahead on
    average but no CI excludes 0 as required; INCOMPLETE: a cell is missing or has fewer than
    ``n_required`` replicates.
    """
    full = [t for t in tests if t["n"] >= n_required]
    wins = [
        t
        for t in full
        if t["win"] and t["code"] in pass_codes and t["budget"] in budgets
    ]
    if wins:
        text = "; ".join(
            f"{t['code']} at {t['budget']} B: {t['diff']:+.3f} ± {t['half']:.3f} "
            f"(F1 {t['learned']:.3f} against {t['designed']:.3f})"
            for t in wins
        )
        return (
            "PASS",
            f"A learned code beats the designed class-id code at k = {G2B_SHOT}: {text}.",
        )
    seen = {(t["code"], t["budget"]) for t in full}
    need = [(c, b) for c in kill_codes for b in kill_budgets]
    missing = [cb for cb in need if cb not in seen]
    if missing:
        return (
            "INCOMPLETE",
            f"No qualifying win yet, but {len(missing)} of {len(need)} (code, budget) "
            f"comparisons lack {n_required} replicates.",
        )
    ahead = [t for t in full if t["code"] in kill_codes and t["diff"] > 0]
    if not ahead:
        return (
            "KILL",
            f"The designed code is at least as good as every learned code at every budget "
            f"({len(need)} comparisons, mean paired differences all <= 0).",
        )
    best = max(ahead, key=lambda t: t["diff"])
    return (
        "INCONCLUSIVE",
        f"{len(ahead)} learned (code, budget) cells are ahead on average (best: {best['code']} "
        f"at {best['budget']} B, {best['diff']:+.3f} ± {best['half']:.3f}) but none "
        "qualifies for a PASS (e2e or e2e_aux at 1, 2 or 4 B with the CI above 0).",
    )


def oracle_check(
    rows: Sequence[dict[str, Any]], metric: str = "seen_f1"
) -> list[dict[str, Any]]:
    """Oracle minus every coded cell on the seen-fault problem, paired by replicate."""
    ref = by_replicate(rows, "oracle", 0, metric)
    out = []
    for code in CODES:
        if code == "oracle":
            continue
        for budget in REPORT_BUDGETS:
            diff = paired(ref, by_replicate(rows, code, budget, metric))
            if diff.size == 0:
                continue
            mean, half, _ = _paired_test(diff)
            out.append(
                {
                    "code": code,
                    "budget": budget,
                    "n": int(diff.size),
                    "diff": mean,
                    "half": half,
                    "dominates": bool(np.isfinite(half) and mean - half > 0),
                    "beaten": bool(np.isfinite(half) and mean + half < 0),
                }
            )
    return out


# ----------------------------------------------------------------------------- the report
def _cell(rows, code: str, budget: int, metric: str, digits: int = 3) -> str:
    values = list(by_replicate(rows, code, budget, metric).values())
    if not values:
        return ""
    mean, half = summarise(values)
    if not np.isfinite(half):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {half:.{digits}f}"


def _table(
    rows, metric: str, codes: Sequence[str] = CODES, digits: int = 3
) -> list[str]:
    head = ["Code", *(f"{b} B" for b in REPORT_BUDGETS)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for code in codes:
        if code == "oracle":
            cells = [_cell(rows, code, 0, metric, digits), "", "", ""]
        else:
            cells = [_cell(rows, code, b, metric, digits) for b in REPORT_BUDGETS]
        if any(cells):
            lines.append("| " + " | ".join([CODE_LABELS[code], *cells]) + " |")
    return lines


def _mapping_counts(rows, code: str, budget: int) -> tuple[Counter, int]:
    pairs: Counter = Counter()
    total = 0
    for r in rows:
        if r["code"] != code or int(r["budget"]) != budget:
            continue
        for unseen, (known, _share, rate) in r["mapping"].items():
            if known >= 0 and rate > 0:
                pairs[(int(unseen), int(known))] += 1
            total += 1
    return pairs, total


def build_report(merged: dict[str, Any], name: str = "") -> str:
    rows = merged["rows"]
    tests = g2b_tests(rows)
    verdict, why = g2b_verdict(tests)
    oracle = oracle_check(rows)
    p = merged["protocol"]
    smoke = merged["scale"] == "smoke"
    n_rep = len({int(r["replicate"]) for r in rows})
    lines = [
        "# Plant floor: can learned codes express faults they never saw? "
        "(R8, step 4, goal G2b)",
        "",
    ]
    if smoke:
        lines += [
            "> **SMOKE RUN: reduced scale. A pipeline check, not evidence.**",
            "",
        ]
    lines += [
        f"- Provenance: commits {', '.join(c[:9] for c in merged['commits'])}; scale "
        f"`{merged['scale']}`; {merged['partials']} partial result files; "
        f"{merged['cpu_seconds'] / 60:.0f} CPU-minutes; {n_rep} replicates with results.",
        f"- Protocol: Rieth et al. 2017 runs, window {p['window']} samples, one uplink per "
        f"{p['stride']}, disjoint run-level replicates as in step 3, centre calibrated on "
        f"validation runs to {100 * p['target_far']:.0f}% false alarms. Every number is "
        "mean ± 95% t-CI over replicates; replicate `r` uses split `r % "
        f"{len(merged['splits']) - 1}`, so the intervals span splits and runs.",
        "- Seen/unseen splits (unseen faults; the last split tunes hyperparameters only): "
        + "; ".join(
            f"split {i}{' (tuning)' if i == len(merged['splits']) - 1 else ''}: "
            + ", ".join(str(f) for f in s)
            for i, s in enumerate(merged["splits"])
        )
        + ".",
        "- Training sees only seen faults and normal data: runs of unseen faults are dropped "
        "from training and validation. Codes and centre are trained and calibrated on that; "
        "the codes are then frozen (checked by a hash of all encoder state before and after "
        "each extension). Few-shot: the centre is retrained (all 17 faults) on the seen "
        "training data plus `k` labelled training runs of each unseen fault.",
        "- Codes: designed class-id code (`local`: each node sends its top classes and "
        "quantised posteriors, over the seen vocabulary), `e2e` (end-to-end sign bits), "
        "`e2e_aux` (e2e plus a per-node linear reconstruction of the standardised features "
        "from the bits, weight 1, fixed rather than tuned), `vq` (unsupervised residual "
        "k-means), `features` (hand-designed PCA/FFT), and `oracle` (no byte limit: float32 "
        "local features plus each node's full seen-class posteriors). Hyperparameters were "
        "chosen on the tuning block for seen-fault validation macro-F1 only.",
        "",
        "## Verdict (goal G2b)",
        "",
        f"G2b passes if, at 1 to 4 B, a learned code (e2e or e2e_aux) beats the designed "
        f"class-id code on few-shot unseen-fault macro-F1 at k = {G2B_SHOT}, with the 95% CI of "
        "the paired difference over replicates and splits excluding 0; it is killed if the "
        "designed code is at least as good as every learned code at every budget.",
        "",
        f"**{verdict}.** {why}",
        "",
        f"## Learned against the designed code, unseen-fault macro-F1 at k = {G2B_SHOT}",
        "",
        "| Code | Budget | Learned | Designed | Difference | p | p (Holm) | n | CI above 0 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in tests:
        lines.append(
            f"| {CODE_LABELS[t['code']]} | {t['budget']} B | {t['learned']:.3f} | "
            f"{t['designed']:.3f} | {t['diff']:+.3f} ± {t['half']:.3f} | {t['p']:.3g} | "
            f"{t['p_holm']:.3g} | {t['n']} | {'yes' if t['win'] else 'no'} |"
        )
    for k in SHOTS:
        for label, metric in (
            (f"Macro-F1 on the unseen faults alone, k = {k}", f"f1_unseen_k{k}"),
            (f"Macro-F1 over all 17 faults, k = {k}", f"f1_all_k{k}"),
        ):
            lines += ["", f"## {label}", ""] + _table(rows, metric)
    lines += [
        "",
        "## Before extension (k = 0): macro-F1 over all 17 faults, unseen faults scored 0",
        "",
        *_table(rows, "f1_all_k0"),
        "",
        "## Detection of unseen faults by the seen-only centre",
        "",
        "Alarm rate on the fault windows of unseen faults at the calibrated 5% "
        "false-alarm threshold (seen-fault alarm rate for reference):",
        "",
        *_table(rows, "unseen_alarm_rate"),
        "",
        "Seen faults, same measure:",
        "",
        *_table(rows, "seen_alarm_rate"),
        "",
        "Fraction of unseen-fault runs detected:",
        "",
        *_table(rows, "unseen_detect_rate"),
        "",
        "Detection delay on detected unseen-fault runs (samples):",
        "",
        *_table(rows, "unseen_delay", digits=1),
        "",
        "Realised false-alarm rate on normal test windows (target 0.05):",
        "",
        *_table(rows, "far"),
        "",
        "## Separability of the frozen messages of the unseen faults",
        "",
        "Linear-probe accuracy (probe trained on half the runs, tested on the rest; chance "
        "is 0.2 for 5 unseen faults):",
        "",
        *_table(rows, "probe_acc"),
        "",
        "Normalised mutual information of k-means (one cluster per unseen fault) against "
        "the true faults:",
        "",
        *_table(rows, "nmi"),
        "",
        "## Seen-fault macro-F1 (is the oracle now an upper bound?)",
        "",
        "Macro-F1 over the 12 seen faults on test runs, seen-only system:",
        "",
        *_table(rows, "seen_f1"),
        "",
    ]
    if oracle:
        dom = [o for o in oracle if o["dominates"]]
        beat = [o for o in oracle if o["beaten"]]
        lines.append(
            f"The oracle is significantly above {len(dom)} of {len(oracle)} coded cells and "
            f"significantly below {len(beat)}."
            + (
                " Cells where a code beats it: "
                + ", ".join(
                    f"{o['code']} {o['budget']} B ({-o['diff']:+.3f})" for o in beat
                )
                + "."
                if beat
                else " No coded cell beats it, so it is an upper bound here."
            )
        )
    lines += [
        "",
        "## How the designed code fails on unseen faults",
        "",
        "With a fixed vocabulary the seen-only centre can only name a known class. "
        "`map_share` is the share of an unseen fault's alarms that go to its single most "
        "frequent known class (1 means it is always mapped to the same known fault).",
        "",
        *_table(rows, "map_share"),
        "",
        "Most frequent (unseen fault -> known fault) mappings of the designed code at "
        "1 B, over replicates (count of replicates in which the pair occurred):",
        "",
    ]
    pairs, total = _mapping_counts(rows, "local", 1)
    top = ", ".join(f"{u} -> {k} ({c})" for (u, k), c in pairs.most_common(8))
    lines.append(top or "(no data)")
    lines += ["", "## Hyperparameters chosen on the tuning block", ""]
    for key, t in sorted(merged["tuning"].items()):
        lines.append(f"- {key}: {json.dumps(t['hp'])}")
    return "\n".join(lines) + "\n"
