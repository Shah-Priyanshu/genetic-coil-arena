"""Runs, node layout and sliding windows for the plant environment (pure numpy).

A *run* is one recorded TEP time series with its fault id and onset. A *window* is the
last ``window`` normalised samples of a node's variables when ``n_seen`` samples have been
recorded; windows are taken every ``stride`` samples, so window ``k`` has seen
``window + stride * k`` samples.

Label rule (the same for training and evaluation): window ``k`` carries the run's fault
class iff it contains at least one faulty sample, that is ``n_seen > onset`` (the last
sample, index ``n_seen - 1``, is at or after the onset). Otherwise it is normal (class 0).
For a test run (onset 160) the window that has seen exactly 160 samples is the last
normal one and the window that has seen 160 + stride samples is the first faulty one.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .tep import (
    NODE_VARIABLES,
    NODES,
    ONSET_TEST,
    ONSET_TRAIN,
    Normaliser,
    TepData,
)

NO_ONSET = -1  # Onset value of a run that never becomes faulty


@dataclass(frozen=True)
class NodeLayout:
    """Which columns each node observes, padded to the widest node.

    Args:
        index: (N, Vmax) int64 column of each node variable; padding points at column 0
        mask: (N, Vmax) bool, True for real variables
    """

    index: np.ndarray
    mask: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.index.shape[0])

    @property
    def max_vars(self) -> int:
        return int(self.index.shape[1])

    @property
    def n_vars(self) -> np.ndarray:
        """(N,) number of real variables per node."""
        return self.mask.sum(axis=1)


def make_layout(
    node_variables: Mapping[str, Sequence[int]] = NODE_VARIABLES,
    nodes: Sequence[str] = NODES,
) -> NodeLayout:
    """Layout of the given nodes (default: the four TEP nodes)."""
    width = max(len(node_variables[n]) for n in nodes)
    index = np.zeros((len(nodes), width), dtype=np.int64)
    mask = np.zeros((len(nodes), width), dtype=np.bool_)
    for i, node in enumerate(nodes):
        cols = list(node_variables[node])
        index[i, : len(cols)] = cols
        mask[i, : len(cols)] = True
    return NodeLayout(index=index, mask=mask)


@dataclass(frozen=True)
class Run:
    """One normalised recorded run.

    Args:
        fault: Fault id, 0 for a normal run
        series: (T, 52) float32 z-scores
        onset: First faulty sample (0-based), ``NO_ONSET`` for a normal run
    """

    fault: int
    series: np.ndarray
    onset: int


def build_runs(
    data: TepData,
    normaliser: Normaliser,
    split: str,
    faults: Sequence[int],
    include_normal: bool = True,
) -> list[Run]:
    """The runs of one split, normalised with ``normaliser`` (normal run first)."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    test = split == "test"
    onset = ONSET_TEST if test else ONSET_TRAIN
    runs: list[Run] = []
    if include_normal:
        raw = data.normal_test if test else data.normal_train
        runs.append(Run(0, normaliser.transform(raw), NO_ONSET))
    table = data.fault_test if test else data.fault_train
    for fault in faults:
        if fault not in table:
            raise KeyError(f"Fault {fault} is not in the data set")
        runs.append(Run(int(fault), normaliser.transform(table[fault]), onset))
    if not runs:
        raise ValueError("A split needs at least one run")
    return runs


def window_end_counts(n_samples: int, window: int, stride: int) -> np.ndarray:
    """``n_seen`` of every window of a series: window, window + stride, ... <= n_samples."""
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1")
    if n_samples < window:
        raise ValueError(f"A series of {n_samples} samples is shorter than the window")
    return np.arange(window, n_samples + 1, stride, dtype=np.int64)


def extract_windows(
    series: np.ndarray, layout: NodeLayout, window: int, stride: int
) -> np.ndarray:
    """All windows of a (T, 52) series as (K, N, Vmax, W) float32 (padding is zero)."""
    if series.ndim != 2:
        raise ValueError(f"series must be (T, variables), got {series.shape}")
    ends = window_end_counts(series.shape[0], window, stride)
    view = sliding_window_view(series, window, axis=0)  # (T - W + 1, 52, W)
    picked = view[ends - window]  # window starting at n_seen - window
    out = picked[:, layout.index, :]  # (K, N, Vmax, W)
    out = out * layout.mask[None, :, :, None]
    return np.ascontiguousarray(out, dtype=np.float32)


def window_labels(n_seen: np.ndarray, onset: int, class_index: int) -> np.ndarray:
    """Class of each window: ``class_index`` once it has seen a faulty sample, else 0."""
    n_seen = np.asarray(n_seen, dtype=np.int64)
    if onset == NO_ONSET:
        return np.zeros(n_seen.shape, dtype=np.int64)
    return np.where(n_seen > onset, class_index, 0).astype(np.int64)


def labelled_windows(
    runs: Sequence[Run],
    class_of_fault: Mapping[int, int],
    layout: NodeLayout,
    window: int,
    stride: int,
    min_start: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Windows of several runs with labels, concatenated.

    Args:
        runs: The runs to window
        class_of_fault: Fault id -> class index (0 is normal and must not be a key)
        layout: Node layout
        window: Window length W
        stride: Samples between windows
        min_start: Drop windows that have seen fewer than this many samples

    Returns:
        ``windows`` (n, N, Vmax, W) float32, ``labels`` (n,) int64, ``run`` (n,) the
        index into ``runs``, ``n_seen`` (n,) int64
    """
    parts: dict[str, list[np.ndarray]] = {
        "windows": [],
        "labels": [],
        "run": [],
        "n_seen": [],
    }
    for r, run in enumerate(runs):
        wins = extract_windows(run.series, layout, window, stride)
        seen = window_end_counts(run.series.shape[0], window, stride)
        cls = 0 if run.fault == 0 else class_of_fault[run.fault]
        labels = window_labels(seen, run.onset, cls)
        keep = (
            np.ones(seen.shape, dtype=bool) if min_start is None else seen >= min_start
        )
        parts["windows"].append(wins[keep])
        parts["labels"].append(labels[keep])
        parts["run"].append(np.full(int(keep.sum()), r, dtype=np.int64))
        parts["n_seen"].append(seen[keep])
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}
