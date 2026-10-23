"""Communication metrics: is information sent, is it used, and how is it structured?

All quantities are in bits. Messages follow the conventions in ``interventions``; any
array whose last axis is the message (length L or dimension D) is accepted and
flattened to one row per message.

Signalling and listening are measured separately because a sender can transmit
information that receivers ignore (Lowe et al. 2019, "On the Pitfalls of Measuring
Emergent Communication"):

- positive signalling: messages carry information about the sender's situation
  (``mutual_information`` well above its shuffled-label null);
- positive listening: receivers' actions depend on the message (``causal_influence``)
  and performance drops when messages are intervened on (``runner.run_ablations``).

Structure: topographic similarity (Brighton & Kirby 2006), positional and
bag-of-symbols disentanglement (Chaabouni et al. 2020).
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional

import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial.distance import pdist

from .interventions import get_intervention, step_rng
from .stats import mean_ci

LN2 = float(np.log(2.0))
CORRECTIONS = ("none", "miller_madow")


def flatten_messages(messages: Any) -> np.ndarray:
    """One row per message: (B, R, L) -> (B*R, L); 1-D input becomes one column."""
    arr = np.asarray(messages)
    if arr.ndim == 0:
        raise ValueError("Messages must have at least one dimension")
    if arr.ndim == 1:
        return arr[:, None]
    return arr.reshape(-1, arr.shape[-1])


def _row_ids(rows: np.ndarray) -> np.ndarray:
    """Integer label per distinct row."""
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    return np.asarray(inverse).reshape(-1)


def as_labels(values: Any, n: Optional[int] = None) -> np.ndarray:
    """Integer labels for a target variable: one label per row (or per scalar)."""
    arr = np.asarray(values)
    if arr.ndim <= 1:
        _, labels = np.unique(arr.reshape(-1), return_inverse=True)
        labels = np.asarray(labels).reshape(-1)
    else:
        rows = arr.reshape(n if n is not None else arr.shape[0], -1)
        labels = _row_ids(rows)
    if n is not None and labels.size != n:
        raise ValueError(f"Expected {n} target values, got {labels.size}")
    return labels


def entropy_from_counts(counts: Any, correction: str = "none") -> float:
    """Plug-in entropy of a histogram, optionally with the Miller-Madow bias correction."""
    if correction not in CORRECTIONS:
        raise ValueError(f"correction must be one of {CORRECTIONS}, got {correction!r}")
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        raise ValueError("Entropy needs at least one observation")

    n = c.sum()
    p = c / n
    h = float(-(p * np.log(p)).sum())
    if correction == "miller_madow":
        h += (c.size - 1) / (2 * n)
    return h / LN2 + 0.0  # + 0.0 turns -0.0 (a single symbol) into 0.0


def entropy(labels: Any, correction: str = "none") -> float:
    """Entropy of a label sequence."""
    _, counts = np.unique(np.asarray(labels).reshape(-1), return_counts=True)
    return entropy_from_counts(counts, correction)


def message_entropy(messages: Any, correction: str = "none") -> float:
    """Entropy H(M) of whole messages (each row is one symbol of the joint alphabet)."""
    return entropy(_row_ids(flatten_messages(messages)), correction)


def symbol_entropy(messages: Any) -> np.ndarray:
    """Entropy of the symbol at each message position, shape (L,)."""
    rows = flatten_messages(messages)
    return np.array([entropy(rows[:, pos]) for pos in range(rows.shape[1])])


def _mi_from_labels(x: np.ndarray, y: np.ndarray, correction: str = "none") -> float:
    joint = x.astype(np.int64) * (int(y.max()) + 1) + y
    return float(
        entropy(x, correction) + entropy(y, correction) - entropy(joint, correction)
    )


def mutual_information(
    messages: Any, targets: Any, correction: str = "miller_madow"
) -> float:
    """I(M; T) between messages and a target variable (e.g. what the sender observed).

    The Miller-Madow correction removes most of the plug-in estimator's upward bias on
    small samples; the result can then be slightly negative when the true value is 0.
    """
    m = _row_ids(flatten_messages(messages))
    t = as_labels(targets, m.size)
    return _mi_from_labels(m, t, correction)


def mi_null_test(
    messages: Any,
    targets: Any,
    n_shuffles: int = 200,
    seed: int = 0,
    correction: str = "miller_madow",
) -> dict[str, float]:
    """Mutual information compared with its null distribution under shuffled targets.

    Returns:
        mi, null_mean, null_std, null_q95 and a one-sided permutation p_value
    """
    m = _row_ids(flatten_messages(messages))
    t = as_labels(targets, m.size)
    observed = _mi_from_labels(m, t, correction)

    rng = np.random.default_rng(seed)
    null = np.array(
        [_mi_from_labels(m, rng.permutation(t), correction) for _ in range(n_shuffles)]
    )
    return {
        "mi": observed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_q95": float(np.quantile(null, 0.95)),
        "p_value": float((1 + np.sum(null >= observed)) / (1 + n_shuffles)),
    }


def channel_usage(messages: Any, vocab_size: int) -> dict[str, float]:
    """Bits actually used per message against the channel capacity L * log2(V).

    Entropy is estimated from the sample, so it can never exceed log2(number of messages).
    """
    rows = flatten_messages(messages)
    capacity = float(rows.shape[1] * np.log2(vocab_size))
    used = message_entropy(rows)
    return {
        "capacity_bits": capacity,
        "entropy_bits": used,
        "utilization": used / capacity if capacity > 0 else 0.0,
        "distinct_messages": float(np.unique(rows, axis=0).shape[0]),
        "symbol_entropy_bits": float(symbol_entropy(rows).sum()),
    }


def quantize_continuous(messages: Any, n_bins: int = 8) -> np.ndarray:
    """Discretise continuous messages per dimension at batch quantiles.

    Returns int64 codes in [0, n_bins) with the input's shape, so the discrete metrics
    (entropy, mutual information, disentanglement) can be applied to continuous channels.
    """
    arr = np.asarray(messages, dtype=float)
    flat = arr.reshape(-1, arr.shape[-1])
    codes = np.empty(flat.shape, dtype=np.int64)
    inner = np.linspace(0, 1, n_bins + 1)[1:-1]
    for dim in range(flat.shape[1]):
        edges = np.quantile(flat[:, dim], inner)
        codes[:, dim] = np.searchsorted(edges, flat[:, dim], side="right")
    return codes.reshape(arr.shape)


def topographic_similarity(
    meanings: Any,
    messages: Any,
    meaning_metric: str = "hamming",
    message_metric: str = "hamming",
) -> float:
    """Spearman correlation between pairwise meaning distances and message distances.

    1 means similar situations get similar messages. NaN when either set of distances is
    constant (e.g. a constant sender), where the correlation is undefined.
    """
    rows = flatten_messages(messages)
    attrs = np.asarray(meanings).reshape(rows.shape[0], -1)
    if rows.shape[0] < 3:
        raise ValueError("Topographic similarity needs at least 3 messages")

    d_meaning = pdist(attrs, metric=meaning_metric)
    d_message = pdist(rows, metric=message_metric)
    if np.ptp(d_meaning) == 0 or np.ptp(d_message) == 0:
        return float("nan")
    rho, _ = scipy_stats.spearmanr(d_meaning, d_message)
    return float(rho)


def _information_gap(attributes: np.ndarray, representation: np.ndarray) -> float:
    """Mean over non-constant columns of (top MI - second MI) / column entropy."""
    attr_labels = [as_labels(attributes[:, i]) for i in range(attributes.shape[1])]
    gaps = []
    for col in range(representation.shape[1]):
        column = as_labels(representation[:, col])
        h = entropy(column)
        if h <= 0:
            continue
        mis = sorted((_mi_from_labels(column, a) for a in attr_labels), reverse=True)
        gaps.append((mis[0] - mis[1]) / h)
    return float(np.mean(gaps)) if gaps else float("nan")


def positional_disentanglement(meanings: Any, messages: Any) -> float:
    """posdis: does each message position encode a single attribute?

    1 for a perfectly position-wise compositional code; NaN if every position is
    constant. Needs at least 2 attributes.
    """
    rows = flatten_messages(messages)
    attrs = np.asarray(meanings).reshape(rows.shape[0], -1)
    if attrs.shape[1] < 2:
        raise ValueError("posdis needs at least 2 attributes")
    return _information_gap(attrs, rows)


def bag_of_symbols_disentanglement(
    meanings: Any, messages: Any, vocab_size: int, exclude_symbols: Sequence[int] = ()
) -> float:
    """bosdis: does each symbol's count encode a single attribute, regardless of position?"""
    rows = flatten_messages(messages)
    attrs = np.asarray(meanings).reshape(rows.shape[0], -1)
    if attrs.shape[1] < 2:
        raise ValueError("bosdis needs at least 2 attributes")
    symbols = [v for v in range(vocab_size) if v not in set(exclude_symbols)]
    counts = np.stack([(rows == v).sum(axis=1) for v in symbols], axis=1)
    return _information_gap(attrs, counts)


def causal_influence(
    listener: Callable[[np.ndarray], np.ndarray],
    messages: np.ndarray,
    candidates: Optional[np.ndarray] = None,
    weights: Optional[Sequence[float]] = None,
    mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Causal influence of communication (CIC) on receivers' actions.

    With each receiver's situation held fixed, every candidate message is delivered in
    turn and the receiver's action distribution recorded; CIC is the mutual information
    I(A; M) between the intervened message and the action, averaged over episodes.

    Args:
        listener: Maps messages (B, R, L|D) to action probabilities (B, R, A) for the
            current, fixed situations
        messages: Observed messages; their distinct rows and frequencies are the default
            candidates and weights (the sender's empirical message distribution)
        candidates: Messages to intervene with, shape (K, L|D)
        weights: Probability of each candidate (default uniform when candidates given)
        mask: Optional (B, R) bool array of live receivers; only they are averaged
            over (and supply the default candidates)

    Returns:
        cic_bits (mean over live receivers), per_receiver_bits (list; NaN for a
        receiver never live) and n_candidates
    """
    arr = np.asarray(messages)
    if arr.ndim != 3:
        raise ValueError(f"Messages must have shape (B, R, L|D), got {arr.shape}")
    live = _live_mask(mask, arr)

    if candidates is None:
        cand, counts = np.unique(arr[live], axis=0, return_counts=True)
        w = counts / counts.sum()
    else:
        cand = np.asarray(candidates, dtype=arr.dtype).reshape(-1, arr.shape[-1])
        w = (
            np.full(len(cand), 1.0 / len(cand))
            if weights is None
            else np.asarray(weights, dtype=float) / np.sum(weights)
        )
    if len(w) != len(cand):
        raise ValueError("weights must match the number of candidates")
    keep = (
        w > 0
    )  # Zero-weight candidates add nothing and could divide by a zero marginal
    cand, w = cand[keep], w[keep]
    if cand.shape[0] == 0:
        raise ValueError("At least one candidate needs positive weight")

    probs = []
    for candidate in cand:
        intervened = np.broadcast_to(candidate, arr.shape).copy()
        p = np.asarray(listener(intervened), dtype=float)
        if p.ndim != 3 or p.shape[:2] != arr.shape[:2]:
            raise ValueError(f"listener must return shape (B, R, A), got {p.shape}")
        probs.append(p)
    p_given_m = np.stack(probs)  # (K, B, R, A)
    marginal = np.tensordot(w, p_given_m, axes=1)  # (B, R, A)

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p_given_m > 0, p_given_m * np.log(p_given_m / marginal), 0.0)
    cic = np.tensordot(w, terms.sum(axis=-1), axes=1) / LN2  # (B, R)
    return {
        "cic_bits": _live_mean(cic, live),
        "per_receiver_bits": _per_receiver(cic, live),
        "n_candidates": int(len(cand)),
    }


def _live_mask(mask: Optional[np.ndarray], arr: np.ndarray) -> np.ndarray:
    """(B, R) bool live mask; all True when ``mask`` is None."""
    if mask is None:
        return np.ones(arr.shape[:2], dtype=bool)
    live = np.asarray(mask, dtype=bool)
    if live.shape != arr.shape[:2]:
        raise ValueError(f"mask must have shape {arr.shape[:2]}, got {live.shape}")
    if not live.any():
        raise ValueError("mask has no live receivers")
    return live


def _live_mean(values: np.ndarray, live: np.ndarray) -> float:
    return float(values[live].mean())


def _per_receiver(values: np.ndarray, live: np.ndarray) -> list[float]:
    counts = live.sum(axis=0)
    sums = np.where(live, values, 0.0).sum(axis=0)
    return [float(s / c) if c else float("nan") for s, c in zip(sums, counts)]


def _js_divergence_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence along the last axis, in bits (0 to 1)."""
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * np.log(p / m), 0.0).sum(axis=-1)
        kl_q = np.where(q > 0, q * np.log(q / m), 0.0).sum(axis=-1)
    return 0.5 * (kl_p + kl_q) / LN2


def intervention_influence(
    listener: Callable[[np.ndarray], np.ndarray],
    messages: np.ndarray,
    intervention: str = "shuffle_batch",
    seed: int = 0,
    n_repeats: int = 1,
    mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """How much an intervention on messages changes receivers' action distributions.

    Complements ``causal_influence`` when the message space is too large to enumerate:
    the listener sees the observed messages and then intervened ones in the same
    situations, and the Jensen-Shannon divergence between the two action
    distributions is averaged over episodes, receivers and repeats.

    Args:
        listener: Maps messages (B, R, L|D) to action probabilities (B, R, A)
        messages: Observed messages
        intervention: Name of an intervention from ``interventions.INTERVENTIONS``
        seed: Seed for the intervention generator
        n_repeats: Independent intervention draws to average over
        mask: Optional (B, R) bool array of live receivers; it is passed to the
            intervention (which then acts among live entries only) and only live
            receivers are averaged over

    Returns:
        js_bits (mean), per_receiver_bits and changed_fraction (share of live
        (episode, receiver) pairs whose most likely action changed)
    """
    arr = np.asarray(messages)
    live = _live_mask(mask, arr)
    fn = get_intervention(intervention)
    base = np.asarray(listener(arr), dtype=float)
    if base.ndim != 3 or base.shape[:2] != arr.shape[:2]:
        raise ValueError(f"listener must return shape (B, R, A), got {base.shape}")

    js, changed = [], []
    for repeat in range(n_repeats):
        intervened = np.asarray(
            listener(fn(arr, step_rng(seed, 0, repeat), None, mask)), dtype=float
        )
        js.append(_js_divergence_bits(base, intervened))
        changed.append(base.argmax(axis=-1) != intervened.argmax(axis=-1))
    js_arr = np.mean(js, axis=0)  # (B, R)
    return {
        "js_bits": _live_mean(js_arr, live),
        "per_receiver_bits": _per_receiver(js_arr, live),
        "changed_fraction": _live_mean(np.mean(changed, axis=0), live),
    }


def steps_to_threshold(
    steps: Sequence[float], values: Sequence[float], threshold: float, window: int = 1
) -> Optional[float]:
    """First step at which the trailing ``window``-point average reaches ``threshold``."""
    s = np.asarray(steps, dtype=float)
    v = np.asarray(values, dtype=float)
    if s.shape != v.shape:
        raise ValueError("steps and values must have the same length")
    if v.size == 0:
        return None

    smoothed = np.convolve(v, np.ones(window) / window, mode="full")[: v.size]
    counts = np.minimum(np.arange(1, v.size + 1), window)
    smoothed = smoothed * window / counts  # Average over the points seen so far
    hits = np.nonzero(smoothed >= threshold)[0]
    return float(s[hits[0]]) if hits.size else None


def learnability(
    curves: Sequence[tuple[Sequence[float], Sequence[float]]],
    reference: float,
    fraction: float = 0.8,
    window: int = 1,
) -> dict[str, Any]:
    """How fast fresh learners reach ``fraction`` of a reference score on a fixed protocol.

    Args:
        curves: One (steps, values) learning curve per fresh learner (seed)
        reference: Score of the reference (e.g. the jointly trained agent); higher is better
        fraction: Fraction of the reference that counts as learned
        window: Smoothing window for the curves

    Returns:
        threshold, fraction_reached, mean_steps and ci over learners that reached it,
        and steps_per_curve (None where never reached)
    """
    threshold = fraction * reference
    per_curve = [steps_to_threshold(s, v, threshold, window) for s, v in curves]
    reached = [s for s in per_curve if s is not None]
    mean, half = mean_ci(reached) if reached else (float("nan"), float("nan"))
    return {
        "threshold": threshold,
        "fraction_reached": len(reached) / len(per_curve) if per_curve else 0.0,
        "mean_steps": mean,
        "ci": half,
        "steps_per_curve": per_curve,
    }


def communication_metrics(
    messages: Any,
    targets: Any = None,
    meanings: Any = None,
    vocab_size: Optional[int] = None,
    n_shuffles: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """The standard set of message-side metrics, so every experiment reports the same ones.

    Continuous messages should be passed through ``quantize_continuous`` first.

    Args:
        messages: Discrete messages, (B, R, L) or (N, L)
        targets: What the messages should be about (sender's situation), one per message
        meanings: Attribute vectors per message for structure metrics (>= 2 attributes)
        vocab_size: Channel vocabulary size, for capacity and bosdis
        n_shuffles: Permutations for the mutual-information null
        seed: Seed for the permutation null

    Returns:
        Flat dict of metric name -> value (NaN where undefined)
    """
    rows = flatten_messages(messages)
    out: dict[str, float] = {"message_entropy_bits": message_entropy(rows)}
    if vocab_size is not None:
        usage = channel_usage(rows, vocab_size)
        out["capacity_bits"] = usage["capacity_bits"]
        out["utilization"] = usage["utilization"]
    if targets is not None:
        test = mi_null_test(rows, targets, n_shuffles=n_shuffles, seed=seed)
        out["mi_bits"] = test["mi"]
        out["mi_null_q95_bits"] = test["null_q95"]
        out["mi_p_value"] = test["p_value"]
    if meanings is not None:
        out["topsim"] = topographic_similarity(meanings, rows)
        attrs = np.asarray(meanings).reshape(rows.shape[0], -1)
        if attrs.shape[1] >= 2:
            out["posdis"] = positional_disentanglement(attrs, rows)
            if vocab_size is not None:
                out["bosdis"] = bag_of_symbols_disentanglement(attrs, rows, vocab_size)
    return out


def listening_summary(
    cic: dict[str, Any],
    ablations: dict[str, dict[str, Any]],
    min_cic_bits: float = 0.01,
) -> dict[str, Any]:
    """Combine CIC and ablation results into a positive-listening verdict.

    Listening is positive when receivers' actions depend on messages (CIC above
    ``min_cic_bits``) and at least one ablation's performance drop has a CI above zero.
    """
    dropped: list[str] = [
        name for name, effect in ablations.items() if effect.get("listening")
    ]
    return {
        "cic_bits": cic["cic_bits"],
        "ablations_with_drop": dropped,
        "positive_listening": bool(cic["cic_bits"] > min_cic_bits and dropped),
    }
