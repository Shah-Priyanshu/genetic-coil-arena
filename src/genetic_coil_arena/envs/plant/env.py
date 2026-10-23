"""Plant-floor sensor squad: four edge nodes uplink to a gateway centre that diagnoses faults.

The shared batched API (``envs.base``) is reused with the roles of the channel reversed
relative to the squad: the **specialists are the four sensor nodes** (reactor, separator,
stripper, compressor/utilities) and the message channel runs **node -> centre** (the
uplink), still shaped ``(B, N, L)`` with N = 4 senders in the receiver slot, so message
hooks, ablations and codecs apply unchanged. The centre is the gateway: it sees only the
uplink messages and acts by *diagnosing* the plant state.

Communication does not change the plant, so episodes replay recorded Tennessee Eastman
runs (``tep``): one run per episode, one window every ``stride`` samples, one step per
window. Per step, with ``obs`` from ``reset``/``step``:

    messages = nodes.encode(obs.specialists)       # (B, N, L) int64, L = byte budget
    diagnosis = centre.diagnose(messages)          # (B,) class, 0 normal else a fault
    obs, rewards, dones, info = env.step(None, messages, diagnosis)

The environment checks the budget (shape, dtype and symbol range via ``MessageSpec``)
and scores the diagnosis against the label of the window just sent. The nodes take no
action on the plant (``n_actions = 1``, a no-op). It tracks per episode the detection
delay (samples from the fault onset to the first alarm), the diagnosis delay (to the first
correct class) and the false alarms on normal windows.

The centre's observation holds only ``uplink``; labels, the clock and the onset are never
in an observation (labels are in ``info``, for scoring and training).
"""

import copy
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ...messages import MessageSpec
from ..base import EnvSpec, MultiAgentEnv, Observations, StepResult
from . import budget as budget_mod
from .tep import FAULT_IDS, NODES, Normaliser, TepData
from .windows import (
    NO_ONSET,
    NodeLayout,
    Run,
    build_runs,
    labelled_windows,
    make_layout,
)

N_ACTIONS = 1  # The nodes only report; no action changes the plant


@dataclass(frozen=True)
class PlantConfig:
    """Plant-floor squad parameters.

    Args:
        budget: Bytes per node message (L); 0 means no uplink. Studied: 0, 8, 16, 51
        window: Samples per node window (W); 20 samples = 1 hour
        stride: Samples between uplinks; one uplink and one diagnosis per window
        split: Replayed split, "test" (960 samples, fault from sample 160) or "train"
            (480 samples, faulty throughout)
        faults: Fault ids present; classes are normal (0) then these, in this order
        include_normal: Add the normal run as an episode (needed for false alarms)
    """

    budget: int = 16
    window: int = 20
    stride: int = 10
    split: str = "test"
    faults: tuple[int, ...] = FAULT_IDS
    include_normal: bool = True
    message: MessageSpec = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if self.split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {self.split!r}")
        faults = tuple(int(f) for f in self.faults)
        if len(set(faults)) != len(faults) or any(f not in FAULT_IDS for f in faults):
            raise ValueError(f"faults must be distinct ids in 1..21, got {self.faults}")
        if not faults and not self.include_normal:
            raise ValueError("Need at least one run")
        object.__setattr__(self, "faults", faults)
        object.__setattr__(
            self, "message", budget_mod.message_spec(self.budget, len(NODES))
        )

    @property
    def classes(self) -> tuple[int, ...]:
        """Class -> fault id: ``(0, *faults)``; class 0 is normal operation."""
        return (0,) + self.faults

    @property
    def n_classes(self) -> int:
        return len(self.faults) + 1

    @property
    def class_of_fault(self) -> dict[int, int]:
        return {f: i + 1 for i, f in enumerate(self.faults)}


class PlantEnv(MultiAgentEnv):
    """Batched replay of Tennessee Eastman runs with a bandwidth-limited uplink.

    Specialist observations (B, N, ...):
        window: (Vmax, W) float32, the node's last W samples as z-scores, zero padded
        mask: (Vmax,) bool, which rows are real variables
    Centre observations (B, ...):
        uplink: (N, L) int64, the messages delivered by the last step (zeros before the
            first step and while a node is silent); shape (N, 0) at budget 0

    ``reset(seeds)`` maps each seed to a run, ``seed % n_runs`` (run 0 is the normal run
    when included, then the faults in ``config.faults`` order), so any seed list works and
    ``range(n_runs)`` plays every run once. The reward is shared: 1 for a correct
    diagnosis of the window just sent, else 0. An episode ends after its last window.

    Normalisation statistics come from the normal *training* run only (``normaliser``),
    whichever split is replayed.

    Pass ``runs`` (already normalised ``windows.Run`` objects, e.g. from
    ``rieth.to_runs``) to replay an arbitrary set of simulation runs instead of the
    Braatz data: ``data``, ``normaliser`` and ``config.split`` are then not used to
    build episodes, and ``config.faults`` must cover the runs' faults.
    """

    def __init__(
        self,
        config: Optional[PlantConfig] = None,
        data: Optional[TepData] = None,
        normaliser: Optional[Normaliser] = None,
        runs: Optional[Sequence[Run]] = None,
    ):
        self.config = config or PlantConfig()
        c = self.config
        self.layout: NodeLayout = make_layout()
        self.normaliser: Optional[Normaliser]
        if runs is None:
            self.data: Optional[TepData] = (
                TepData.from_cache() if data is None else data
            )
            self.normaliser = normaliser or Normaliser.fit(self.data.normal_train)
            self.runs: list[Run] = build_runs(
                self.data, self.normaliser, c.split, c.faults, c.include_normal
            )
        else:
            self.data, self.normaliser = data, normaliser
            self.runs = list(runs)
            if not self.runs:
                raise ValueError("A split needs at least one run")
        flat = labelled_windows(
            self.runs, c.class_of_fault, self.layout, c.window, c.stride
        )
        counts = np.bincount(flat["run"], minlength=len(self.runs))
        k_max = int(np.max(counts))
        n_runs = len(self.runs)
        n, v, w = self.layout.n_nodes, self.layout.max_vars, c.window
        self._windows = np.zeros((n_runs, k_max, n, v, w), dtype=np.float32)
        self._labels = np.zeros((n_runs, k_max), dtype=np.int64)
        self._n_seen = np.zeros((n_runs, k_max), dtype=np.int64)
        start = 0
        for r, count in enumerate(counts.tolist()):
            sl = slice(start, start + count)
            self._windows[r, :count] = flat["windows"][sl]
            self._labels[r, :count] = flat["labels"][sl]
            self._n_seen[r, :count] = flat["n_seen"][sl]
            self._windows[r, count:] = flat["windows"][start + count - 1]  # frozen tail
            self._labels[r, count:] = flat["labels"][start + count - 1]
            self._n_seen[r, count:] = flat["n_seen"][start + count - 1]
            start += count
        self._n_windows = counts.astype(np.int64)
        self._run_onset = np.array([r.onset for r in self.runs], dtype=np.int64)
        self._run_class = np.array(
            [0 if r.fault == 0 else c.class_of_fault[r.fault] for r in self.runs],
            dtype=np.int64,
        )

        self.spec = self._make_spec(k_max)
        self._batch = 0

    def _make_spec(self, k_max: int) -> EnvSpec:
        msg = self.config.message
        n, v, w = self.layout.n_nodes, self.layout.max_vars, self.config.window
        return EnvSpec(
            n_specialists=n,
            n_actions=N_ACTIONS,
            specialist_obs={
                "window": ((v, w), np.float32),
                "mask": ((v,), np.bool_),
            },
            centre_obs={"uplink": ((n,) + msg.payload_shape, msg.dtype)},
            message=msg,
            max_steps=k_max,
        )

    def with_budget(self, budget: int) -> "PlantEnv":
        """The same episodes on a channel of another byte budget, sharing the window arrays.

        Building an environment materialises every window of every run; this copy only
        swaps the message channel, so one set of runs serves all budgets.
        """
        other = copy.copy(self)
        other.config = dataclasses.replace(self.config, budget=budget)
        other.spec = other._make_spec(self.spec.max_steps)
        other._batch = 0
        return other

    # ------------------------------------------------------------------ properties
    @property
    def batch_size(self) -> int:
        """Number of parallel episodes since the last reset."""
        return self._batch

    @property
    def n_runs(self) -> int:
        """Distinct episodes ``reset`` can start (seeds wrap around)."""
        return len(self.runs)

    @property
    def run_faults(self) -> np.ndarray:
        """Fault id of each run (0 for the normal run)."""
        return np.array([r.fault for r in self.runs], dtype=np.int64)

    @property
    def run_onsets(self) -> np.ndarray:
        """Onset of each run: first faulty sample, ``NO_ONSET`` (-1) for the normal run."""
        return self._run_onset.copy()

    @property
    def detection_delay(self) -> np.ndarray:
        """(B,) samples from the onset to the first alarm window; NaN if none yet."""
        return self._delay(self._detect)

    @property
    def diagnosis_delay(self) -> np.ndarray:
        """(B,) samples from the onset to the first correct fault class; NaN if none."""
        return self._delay(self._diagnose)

    @staticmethod
    def _delay(raw: np.ndarray) -> np.ndarray:
        return np.where(raw < 0, np.nan, raw).astype(np.float64)

    # ------------------------------------------------------------------ episode loop
    def reset(self, seeds: Sequence[int]) -> Observations:
        """Start one episode per seed; the same seed always gives the same episode."""
        seeds = [int(s) for s in seeds]
        if not seeds:
            raise ValueError("reset needs at least one seed")
        b = len(seeds)
        self._batch = b
        self._run = np.array([s % self.n_runs for s in seeds], dtype=np.int64)
        self._k = np.zeros(b, dtype=np.int64)
        self.done = np.zeros(b, dtype=np.bool_)
        self.episode_class = self._run_class[self._run].copy()
        self.onset = self._run_onset[self._run].copy()
        self._detect = np.full(b, -1, dtype=np.int64)
        self._diagnose = np.full(b, -1, dtype=np.int64)
        self.false_alarms = np.zeros(b, dtype=np.int64)
        self.normal_windows = np.zeros(b, dtype=np.int64)
        self.correct_windows = np.zeros(b, dtype=np.int64)
        self.last_message = self.config.message.zeros(b, len(NODES))
        return self.observe()

    def _cursor(self) -> np.ndarray:
        return np.minimum(self._k, self._n_windows[self._run] - 1)

    def observe(self) -> Observations:
        """Current observations (fresh arrays, safe to keep)."""
        if self._batch == 0:
            raise RuntimeError("Call reset() before observe()")
        windows = self._windows[self._run, self._cursor()]
        mask = np.broadcast_to(
            self.layout.mask, (self._batch,) + self.layout.mask.shape
        )
        return Observations(
            specialists={"window": windows.copy(), "mask": mask.copy()},
            centre={"uplink": self.last_message.copy()},
        )

    def labels(self) -> np.ndarray:
        """(B,) class of the window now being observed (for scoring, never observed)."""
        return self._labels[self._run, self._cursor()].copy()

    def _check_uplink(self, messages: Optional[np.ndarray]) -> np.ndarray:
        spec = self.config.message
        n = len(NODES)
        if messages is None:
            return spec.zeros(self._batch, n)  # Silent nodes
        if spec.kind == "none":
            arr = np.asarray(messages)
            if arr.shape != (self._batch, n, 0):
                raise ValueError(
                    "Budget is 0 bytes: messages must be None or shape "
                    f"({self._batch}, {n}, 0), got {arr.shape}"
                )
            return arr.astype(np.int64, copy=False)
        return spec.validate(messages, self._batch, n)

    def step(
        self,
        actions: Optional[np.ndarray] = None,
        messages: Optional[np.ndarray] = None,
        diagnosis: Optional[np.ndarray] = None,
    ) -> StepResult:
        """Deliver the uplink, score the centre's diagnosis and move to the next window.

        Args:
            actions: (B, N) zeros, or ``None``; the nodes' only action is a no-op
            messages: (B, N, L) int64 uplink in [0, 256) with L the byte budget, or
                ``None`` for silent nodes; anything else raises ``ValueError``
            diagnosis: (B,) class of the window just sent in [0, n_classes), or ``None``
                for "no alarm" (class 0)

        Returns:
            StepResult(obs, rewards (B, N), dones (B,), info) where ``info`` has
            ``label``, ``diagnosis``, ``correct``, ``alarm``, ``false_alarm`` (all (B,)),
            ``n_seen`` (samples seen at the window just scored), ``detection_delay`` and
            ``diagnosis_delay`` (float, NaN until reached), ``steps`` (windows done)
        """
        if self._batch == 0:
            raise RuntimeError("Call reset() before step()")
        b, n = self._batch, len(NODES)
        if actions is not None:
            self.validate_actions(actions)
        uplink = self._check_uplink(messages)
        classes = self.config.n_classes
        if diagnosis is None:
            diag = np.zeros(b, dtype=np.int64)
        else:
            diag = np.asarray(diagnosis)
            if diag.shape != (b,) or not np.issubdtype(diag.dtype, np.integer):
                raise ValueError(f"diagnosis must be an integer array of shape ({b},)")
            if diag.size and (np.min(diag) < 0 or np.max(diag) >= classes):
                raise ValueError(f"diagnosis must lie in [0, {classes})")
            diag = diag.astype(np.int64, copy=False)

        active = ~self.done
        cursor = self._cursor()
        label = self._labels[self._run, cursor]
        n_seen = self._n_seen[self._run, cursor]
        self.last_message[active] = uplink[active]

        correct = (diag == label) & active
        alarm = (diag != 0) & active
        faulty = (label != 0) & active
        false_alarm = alarm & ~faulty
        first_alarm = alarm & faulty & (self._detect < 0)
        first_correct = correct & faulty & (self._diagnose < 0)
        self._detect[first_alarm] = (n_seen - self.onset)[first_alarm]
        self._diagnose[first_correct] = (n_seen - self.onset)[first_correct]
        self.false_alarms += false_alarm
        self.normal_windows += (label == 0) & active
        self.correct_windows += correct

        self._k = np.where(active, self._k + 1, self._k)
        self.done = self._k >= self._n_windows[self._run]
        team = correct.astype(np.float32)
        info = {
            "label": label.copy(),
            "diagnosis": diag.copy(),
            "correct": correct,
            "alarm": alarm,
            "false_alarm": false_alarm,
            "n_seen": n_seen.copy(),
            "detection_delay": self.detection_delay,
            "diagnosis_delay": self.diagnosis_delay,
            "steps": self._k.copy(),
        }
        rewards = np.repeat(team[:, None], n, axis=1)
        return StepResult(self.observe(), rewards, self.done.copy(), info)

    # ------------------------------------------------------------------ training data
    def labelled_windows(
        self, split: str = "train", stride: Optional[int] = None
    ) -> dict[str, np.ndarray]:
        """Labelled windows of a split for fitting a centre (the normal run included).

        Uses the same normaliser and label rule as the episodes. Pass the training split
        to fit anything that learns or calibrates; ``stride`` defaults to the env's.
        """
        c = self.config
        if self.data is None or self.normaliser is None:
            raise ValueError(
                "This environment was built from runs; window them directly"
            )
        runs = build_runs(self.data, self.normaliser, split, c.faults, True)
        out = labelled_windows(
            runs,
            c.class_of_fault,
            self.layout,
            c.window,
            c.stride if stride is None else stride,
        )
        out["mask"] = self.layout.mask.copy()
        return out


__all__ = ["N_ACTIONS", "NO_ONSET", "PlantConfig", "PlantEnv"]
