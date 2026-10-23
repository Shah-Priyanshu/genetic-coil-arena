"""Roll out an encoder and a centre over plant episodes, and score the diagnoses.

``run_episodes`` mirrors the squad runner: it plays one batched episode per seed to the
end, applies an optional message ``hook`` between send and delivery (an ``evaluation``
``MessageHook`` fits: ``hook(messages, step, mask)``), and returns per-window labels and
diagnoses plus the environment's delay and false-alarm tracking. ``diagnosis_metrics``
turns those into the headline numbers:

- macro-F1 over the fault classes, from the confusion of every window of every episode
  (normal windows count as false positives for the class they are mistaken for);
- detection rate and delay: a fault run is detected at its first faulty window where the
  centre raises any alarm, the delay is samples from the onset to that window's end;
- false-alarm rate: the share of normal windows (the normal run and the pre-onset part of
  fault runs) that raise an alarm.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional

import numpy as np

from ..base import Observations
from .env import PlantEnv

Hook = Callable[..., np.ndarray]
Observer = Callable[[int, Observations, Optional[np.ndarray], np.ndarray], Any]


def run_episodes(
    env: PlantEnv,
    encoder: Optional[Any],
    centre: Any,
    seeds: Sequence[int],
    hook: Optional[Hook] = None,
    observer: Optional[Observer] = None,
) -> dict[str, np.ndarray]:
    """Play one batched episode per seed to the end.

    Args:
        env: The environment (its budget fixes the uplink shape)
        encoder: Node encoder with ``encode(specialist_obs)``; ``None`` sends nothing
        centre: Has ``diagnose(messages, raw)``; ``raw`` is passed only if
            ``centre.needs_raw`` (the oracle)
        seeds: One episode per seed (``seed % env.n_runs`` picks the run)
        hook: Applied to the messages between send and delivery as
            ``hook(messages, step, mask)``, ``mask`` = (B, N) senders of unfinished episodes
        observer: Called as ``observer(step, obs, sent, done)`` with the messages as sent

    Returns:
        ``labels``, ``diagnoses`` and ``n_seen`` (B, K) with ``valid`` (B, K) marking
        real windows; ``margins`` and ``fault_pred`` (B, K) from learned centres (the
        normal margin and best fault class, NaN / 0 for others); per-episode ``episode_class``, ``onset``, ``detection_delay``,
        ``diagnosis_delay`` (NaN if never), ``false_alarms``, ``normal_windows``;
        ``bytes_per_node`` (the encoder's declared bytes per node message, mean over nodes)
    """
    obs = env.reset(seeds)
    b, k_max = len(seeds), env.spec.max_steps
    labels = np.zeros((b, k_max), dtype=np.int64)
    diagnoses = np.zeros((b, k_max), dtype=np.int64)
    n_seen = np.zeros((b, k_max), dtype=np.int64)
    valid = np.zeros((b, k_max), dtype=bool)
    margins = np.full((b, k_max), np.nan)
    fault_pred = np.zeros((b, k_max), dtype=np.int64)
    done = np.zeros(b, dtype=bool)
    step = 0
    n_nodes = env.spec.n_specialists
    while not done.all():
        sent: Optional[np.ndarray]
        if encoder is None:
            sent = None
        else:
            sent = encoder.encode(obs.specialists)
        if observer is not None:
            observer(step, obs, sent, done)
        delivered = sent
        if hook is not None and sent is not None:
            delivered = hook(sent, step, np.repeat(~done[:, None], n_nodes, axis=1))
        messages = (
            delivered if delivered is not None else env.config.message.zeros(b, n_nodes)
        )
        raw = obs.specialists if getattr(centre, "needs_raw", False) else None
        diagnosis = np.asarray(centre.diagnose(messages, raw), dtype=np.int64)
        live = ~done
        if getattr(centre, "last_margin", None) is not None:
            margins[live, step] = centre.last_margin[live]
            fault_pred[live, step] = centre.last_fault[live]
        obs, _, done, info = env.step(None, delivered, diagnosis)
        labels[live, step] = info["label"][live]
        diagnoses[live, step] = info["diagnosis"][live]
        n_seen[live, step] = info["n_seen"][live]
        valid[live, step] = True
        step += 1
    sent_bytes = 0.0 if encoder is None else float(np.mean(encoder.bytes_used))
    return {
        "labels": labels,
        "diagnoses": diagnoses,
        "n_seen": n_seen,
        "valid": valid,
        "margins": margins,
        "fault_pred": fault_pred,
        "episode_class": env.episode_class.copy(),
        "onset": env.onset.copy(),
        "detection_delay": env.detection_delay,
        "diagnosis_delay": env.diagnosis_delay,
        "false_alarms": env.false_alarms.copy(),
        "normal_windows": env.normal_windows.copy(),
        "bytes_per_node": np.float64(sent_bytes),
    }


def macro_f1(
    labels: np.ndarray, preds: np.ndarray, n_classes: int
) -> tuple[float, np.ndarray]:
    """Macro-F1 over the fault classes 1..n_classes-1 and the per-class F1 (class 0 excluded)."""
    f1 = np.zeros(n_classes - 1)
    for c in range(1, n_classes):
        tp = float(np.sum((preds == c) & (labels == c)))
        fp = float(np.sum((preds == c) & (labels != c)))
        fn = float(np.sum((preds != c) & (labels == c)))
        denom = 2 * tp + fp + fn
        f1[c - 1] = 2 * tp / denom if denom else 0.0
    return float(f1.mean()), f1


def metrics_from_arrays(
    labels: np.ndarray,
    diagnoses: np.ndarray,
    n_seen: np.ndarray,
    valid: np.ndarray,
    onset: np.ndarray,
    episode_class: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    """Headline numbers from per-window arrays (all (B, K) except the per-episode ones).

    Returns:
        ``macro_f1``, ``f1_per_class`` (fault classes), ``false_alarm_rate``,
        ``detection_rate`` (fraction of fault runs detected), ``detection_delay`` (mean
        samples over detected runs, NaN if none), ``diagnosis_rate`` and
        ``diagnosis_delay`` likewise for the first correct class
    """
    f1, per_class = macro_f1(labels[valid], diagnoses[valid], n_classes)
    normal = valid & (labels == 0)
    far = float((normal & (diagnoses != 0)).sum()) / max(int(normal.sum()), 1)
    if not normal.any():
        far = float("nan")
    fault_run = episode_class != 0
    out: dict[str, Any] = {
        "macro_f1": f1,
        "f1_per_class": per_class,
        "false_alarm_rate": far,
    }
    rows = np.arange(labels.shape[0])
    for name, hit in (
        ("detection", diagnoses != 0),
        ("diagnosis", diagnoses == labels),
    ):
        reached = valid & (labels != 0) & hit
        first = reached.argmax(axis=1)
        delay = np.where(reached.any(axis=1), n_seen[rows, first] - onset, np.nan)
        delay = delay[fault_run].astype(np.float64)
        found = np.isfinite(delay)
        out[f"{name}_rate"] = float(found.mean()) if delay.size else float("nan")
        out[f"{name}_delay"] = (
            float(delay[found].mean()) if found.any() else float("nan")
        )
    return out


def diagnosis_metrics(result: dict[str, np.ndarray], n_classes: int) -> dict[str, Any]:
    """Headline numbers of one rollout (see the module docstring and ``metrics_from_arrays``).

    The environment's own delay and false-alarm tracking is in ``result`` too
    (``detection_delay``, ``false_alarms``, ...); the tests check the two agree.
    """
    return metrics_from_arrays(
        result["labels"],
        result["diagnoses"],
        result["n_seen"],
        result["valid"],
        result["onset"],
        result["episode_class"],
        n_classes,
    )


def matched_far_tau(result: dict[str, np.ndarray], target: float) -> float:
    """Normal-margin threshold that gives ``target`` false alarms on this rollout's normal windows.

    This looks at the rollout's own labels, so it is an operating point no deployed
    centre could pick: it is for comparing encodings at the same false-alarm rate.
    """
    margins = result["margins"]
    normal = result["valid"] & (result["labels"] == 0)
    if np.isnan(margins[normal]).any() or not normal.any():
        raise ValueError("The rollout has no margins; only learned centres have them")
    return float(np.quantile(margins[normal], 1.0 - target))


def rethreshold(result: dict[str, np.ndarray], tau: float) -> dict[str, np.ndarray]:
    """The same rollout re-decided with another normal margin ``tau`` (no environment calls)."""
    if np.isnan(result["margins"][result["valid"]]).any():
        raise ValueError("The rollout has no margins; only learned centres have them")
    out = dict(result)
    out["diagnoses"] = np.where(
        result["margins"] > tau, result["fault_pred"], 0
    ).astype(np.int64)
    return out
