"""Tennessee Eastman Process (TEP) data: variables, node partition, loading, normalising.

The classic Braatz-group benchmark (Downs and Vogel 1993 simulator; Chiang, Russell and
Braatz 2001 data) has 22 operating modes: the normal mode ``d00`` and faults ``d01`` to
``d21``. Every sample is 52 columns, XMEAS(1..41) then XMV(1..11), taken every 3 minutes.

Files (one recorded run per mode and split):

- ``d00.dat`` (normal training): stored transposed, 52 rows by 500 columns;
- ``d00_te.dat`` (normal test): 960 samples, no fault;
- ``dNN.dat`` (fault training): 480 samples; the fault acts from the first sample (it was
  introduced at the start of the run, so there is no normal prefix);
- ``dNN_te.dat`` (fault test): 960 samples; normal for samples 0..159 and the fault
  acts from sample 160 (introduced after 8 hours).

The 52 variables are split over four edge nodes by the unit operation (or service) that
owns the sensor or actuator, so each node is a plausible LoRaWAN end device:

- ``reactor``: the four reactant feed flows and valves, the reactor feed rate, pressure,
  level and temperature, and the reactor-feed gas analyser (stream 6);
- ``separator``: the purge rate, the product separator temperature, level, pressure and
  underflow, the purge gas analyser (stream 9), and the purge and separator liquid
  valves;
- ``stripper``: the stripper level, pressure, underflow and temperature, the product
  analyser (stream 11) and the stripper product valve;
- ``compressor_utilities``: the recycle flow, the compressor work and recycle valve,
  the stripper steam flow and valve, and both cooling-water circuits (outlet
  temperatures and valves).
"""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

N_VARIABLES = 52
SAMPLE_MINUTES = 3  # One TEP sample every 3 minutes
FAULT_IDS = tuple(range(1, 22))  # IDV(1..21)
# Faults that no data-driven method separates from normal operation in this data set
UNOBSERVABLE_FAULTS = (3, 9, 15)
ONSET_TEST = 160  # First faulty sample of a test run (0-based): 8 h at 3 min
ONSET_TRAIN = 0  # Fault training runs are faulty from their first sample
TRAIN_NORMAL_SAMPLES = 500
TRAIN_FAULT_SAMPLES = 480
TEST_SAMPLES = 960

NODES = ("reactor", "separator", "stripper", "compressor_utilities")

FAULT_DESCRIPTIONS = {
    1: "A/C feed ratio, B composition constant (stream 4), step",
    2: "B composition, A/C ratio constant (stream 4), step",
    3: "D feed temperature (stream 2), step",
    4: "Reactor cooling water inlet temperature, step",
    5: "Condenser cooling water inlet temperature, step",
    6: "A feed loss (stream 1), step",
    7: "C header pressure loss, reduced availability (stream 4), step",
    8: "A, B, C feed composition (stream 4), random variation",
    9: "D feed temperature (stream 2), random variation",
    10: "C feed temperature (stream 4), random variation",
    11: "Reactor cooling water inlet temperature, random variation",
    12: "Condenser cooling water inlet temperature, random variation",
    13: "Reaction kinetics, slow drift",
    14: "Reactor cooling water valve, sticking",
    15: "Condenser cooling water valve, sticking",
    16: "Unknown",
    17: "Unknown",
    18: "Unknown",
    19: "Unknown",
    20: "Unknown",
    21: "Valve for stream 4 fixed at its steady-state position, constant position",
}


@dataclass(frozen=True)
class TepVariable:
    """One of the 52 TEP columns.

    Args:
        index: Column in the data files (0-based)
        tag: Published tag, ``XMEAS(n)`` or ``XMV(n)``
        name: Published description
        unit: Engineering unit
        node: The edge node that owns the variable
    """

    index: int
    tag: str
    name: str
    unit: str
    node: str


def _build_variables() -> tuple[TepVariable, ...]:
    # (name, unit, node) per column, from the Downs and Vogel (1993) tables
    xmeas = [
        ("A feed (stream 1)", "kscmh", "reactor"),
        ("D feed (stream 2)", "kg/h", "reactor"),
        ("E feed (stream 3)", "kg/h", "reactor"),
        ("A and C feed (stream 4)", "kscmh", "reactor"),
        ("Recycle flow (stream 8)", "kscmh", "compressor_utilities"),
        ("Reactor feed rate (stream 6)", "kscmh", "reactor"),
        ("Reactor pressure", "kPa gauge", "reactor"),
        ("Reactor level", "%", "reactor"),
        ("Reactor temperature", "deg C", "reactor"),
        ("Purge rate (stream 9)", "kscmh", "separator"),
        ("Product separator temperature", "deg C", "separator"),
        ("Product separator level", "%", "separator"),
        ("Product separator pressure", "kPa gauge", "separator"),
        ("Product separator underflow (stream 10)", "m3/h", "separator"),
        ("Stripper level", "%", "stripper"),
        ("Stripper pressure", "kPa gauge", "stripper"),
        ("Stripper underflow (stream 11)", "m3/h", "stripper"),
        ("Stripper temperature", "deg C", "stripper"),
        ("Stripper steam flow", "kg/h", "compressor_utilities"),
        ("Compressor work", "kW", "compressor_utilities"),
        ("Reactor cooling water outlet temperature", "deg C", "compressor_utilities"),
        ("Separator cooling water outlet temperature", "deg C", "compressor_utilities"),
    ]
    xmeas += [
        (f"Reactor feed analysis, component {c} (stream 6)", "mol%", "reactor")
        for c in "ABCDEF"
    ]
    xmeas += [
        (f"Purge gas analysis, component {c} (stream 9)", "mol%", "separator")
        for c in "ABCDEFGH"
    ]
    xmeas += [
        (f"Product analysis, component {c} (stream 11)", "mol%", "stripper")
        for c in "DEFGH"
    ]
    xmv = [
        ("D feed flow (stream 2)", "reactor"),
        ("E feed flow (stream 3)", "reactor"),
        ("A feed flow (stream 1)", "reactor"),
        ("A and C feed flow (stream 4)", "reactor"),
        ("Compressor recycle valve", "compressor_utilities"),
        ("Purge valve (stream 9)", "separator"),
        ("Separator pot liquid flow (stream 10)", "separator"),
        ("Stripper liquid product flow (stream 11)", "stripper"),
        ("Stripper steam valve", "compressor_utilities"),
        ("Reactor cooling water flow", "compressor_utilities"),
        ("Condenser cooling water flow", "compressor_utilities"),
    ]
    out = [
        TepVariable(i, f"XMEAS({i + 1})", name, unit, node)
        for i, (name, unit, node) in enumerate(xmeas)
    ]
    out += [
        TepVariable(len(xmeas) + j, f"XMV({j + 1})", name, "%", node)
        for j, (name, node) in enumerate(xmv)
    ]
    return tuple(out)


VARIABLES = _build_variables()
assert len(VARIABLES) == N_VARIABLES

# Columns of each node, in data order
NODE_VARIABLES: dict[str, tuple[int, ...]] = {
    node: tuple(v.index for v in VARIABLES if v.node == node) for node in NODES
}
assert sorted(i for cols in NODE_VARIABLES.values() for i in cols) == list(
    range(N_VARIABLES)
)

SOURCE_URL = "https://web.mit.edu/braatzgroup/TE_process.zip"
SOURCE_LICENSE = (
    "University of Illinois / Large Scale Systems Research Laboratory (Prof. Richard "
    "Braatz), 1998-2002: free of charge, permission to deal with the Software and data "
    "without restriction (copy, modify, merge, publish, distribute, sublicense, sell), "
    "subject to keeping the copyright notice and disclaimers (NCSA-style open-source "
    "licence, see readme.txt in the archive)"
)


def default_cache_dir() -> Path:
    """Cache directory outside the repo: ``$GCA_TEP_DIR`` or the per-user cache.

    ``%LOCALAPPDATA%\\genetic_coil_arena\\tep`` on Windows, ``$XDG_CACHE_HOME`` (or
    ``~/.cache``) ``/genetic_coil_arena/tep`` elsewhere.
    """
    override = os.environ.get("GCA_TEP_DIR")
    if override:
        return Path(override)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "genetic_coil_arena" / "tep"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "genetic_coil_arena" / "tep"


def _fault_file(fault: int, test: bool) -> str:
    return f"d{fault:02d}_te.dat" if test else f"d{fault:02d}.dat"


@dataclass
class TepData:
    """The recorded runs of one TEP data set (raw engineering units).

    Args:
        normal_train: (500, 52) normal operation, the only data the normaliser and every
            hand-designed encoder calibration may use
        normal_test: (960, 52) normal operation, held out
        fault_train: Fault id -> (480, 52) run, faulty from sample 0
        fault_test: Fault id -> (960, 52) run, faulty from sample ``ONSET_TEST``
        source: Provenance (url, licence, checksums) when loaded from the cache
    """

    normal_train: np.ndarray
    normal_test: np.ndarray
    fault_train: dict[int, np.ndarray]
    fault_test: dict[int, np.ndarray]
    source: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("normal_train", "normal_test"):
            arr = getattr(self, name)
            if arr.ndim != 2 or arr.shape[1] != N_VARIABLES:
                raise ValueError(f"{name} must be (T, {N_VARIABLES}), got {arr.shape}")
        for name in ("fault_train", "fault_test"):
            for fault, arr in getattr(self, name).items():
                if arr.ndim != 2 or arr.shape[1] != N_VARIABLES:
                    raise ValueError(
                        f"{name}[{fault}] must be (T, {N_VARIABLES}), got {arr.shape}"
                    )

    @property
    def faults(self) -> tuple[int, ...]:
        """Fault ids present in both splits."""
        return tuple(sorted(set(self.fault_train) & set(self.fault_test)))

    @classmethod
    def from_directory(cls, directory: Path) -> "TepData":
        """Load ``d00.dat`` ... ``d21_te.dat`` from a directory (see ``fetch``).

        Raises:
            FileNotFoundError: If a file is missing (run ``scripts/fetch_tep.py``)
        """
        directory = Path(directory)

        def load(name: str) -> np.ndarray:
            path = directory / name
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found; run `python scripts/fetch_tep.py` to download "
                    "the Tennessee Eastman data into the cache"
                )
            arr = np.loadtxt(path, dtype=np.float64)
            if arr.shape[0] == N_VARIABLES and arr.shape[1] != N_VARIABLES:
                arr = arr.T  # d00.dat is stored as 52 rows by 500 columns
            return np.ascontiguousarray(arr)

        source: dict = {}
        info = directory / "SOURCE.json"
        if info.exists():
            source = json.loads(info.read_text(encoding="utf-8"))
        return cls(
            normal_train=load("d00.dat"),
            normal_test=load("d00_te.dat"),
            fault_train={f: load(_fault_file(f, False)) for f in FAULT_IDS},
            fault_test={f: load(_fault_file(f, True)) for f in FAULT_IDS},
            source=source,
        )

    @classmethod
    def from_cache(cls, cache_dir: Optional[Path] = None) -> "TepData":
        """Load from the cache directory (``default_cache_dir()`` when ``None``)."""
        return cls.from_directory(
            default_cache_dir() if cache_dir is None else cache_dir
        )


@dataclass(frozen=True)
class Normaliser:
    """Per-variable z-scoring with statistics from normal training data only.

    Args:
        mean: (52,) per-variable mean
        std: (52,) per-variable standard deviation (floored, never zero)
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, normal_train: np.ndarray) -> "Normaliser":
        """Fit on the normal training run (the only data the statistics may see)."""
        x = np.asarray(normal_train, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError("Need a (T >= 2, variables) normal training array")
        mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=1)
        floor = 1e-8 * np.maximum(1.0, np.abs(mean))
        return cls(mean=mean, std=np.maximum(std, floor))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """z-scores as float32; the fitted statistics are never updated."""
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.std).astype(
            np.float32
        )


def synthetic_tep(
    seed: int = 0, faults: Sequence[int] = FAULT_IDS, strength: float = 4.0
) -> TepData:
    """A small stand-in for the real data with the same shapes and onsets.

    Normal operation is stationary AR(1) noise around per-variable levels. Fault ``f``
    shifts a fixed random subset of variables by ``strength`` standard deviations from
    the onset (sample 0 of a training run, ``ONSET_TEST`` of a test run), ramping in
    over a few samples. For tests and demos where the download is unavailable; carries
    no process physics.
    """
    rng = np.random.default_rng(seed)
    level = rng.normal(0.0, 50.0, size=N_VARIABLES)
    scale = rng.uniform(0.1, 5.0, size=N_VARIABLES)

    def noise(n: int) -> np.ndarray:
        e = rng.normal(size=(n, N_VARIABLES))
        out = np.empty_like(e)
        out[0] = e[0]
        for t in range(1, n):
            out[t] = 0.7 * out[t - 1] + np.sqrt(1 - 0.49) * e[t]
        return out

    def run(n: int, shift: Optional[np.ndarray], onset: int) -> np.ndarray:
        z = noise(n)
        if shift is not None:
            ramp = np.clip((np.arange(n) - onset + 1) / 5.0, 0.0, 1.0)
            z = z + ramp[:, None] * shift[None, :]
        return level + scale * z

    shifts = {}
    shift_rng = np.random.default_rng(1000 + seed)
    for f in FAULT_IDS:
        shift = np.zeros(N_VARIABLES)
        picked = shift_rng.choice(N_VARIABLES, size=6, replace=False)
        shift[picked] = strength * shift_rng.choice([-1.0, 1.0], size=6)
        shifts[f] = shift
    wanted = tuple(faults)
    return TepData(
        normal_train=run(TRAIN_NORMAL_SAMPLES, None, 0),
        normal_test=run(TEST_SAMPLES, None, 0),
        fault_train={
            f: run(TRAIN_FAULT_SAMPLES, shifts[f], ONSET_TRAIN) for f in wanted
        },
        fault_test={f: run(TEST_SAMPLES, shifts[f], ONSET_TEST) for f in wanted},
        source={"url": "synthetic", "license": "n/a"},
    )


def describe_mapping(
    variables: Sequence[TepVariable] = VARIABLES,
) -> Mapping[str, list[str]]:
    """Node -> list of ``"XMEAS(7) Reactor pressure [kPa gauge]"`` strings."""
    out: dict[str, list[str]] = {node: [] for node in NODES}
    for v in variables:
        out[v.node].append(f"{v.tag} {v.name} [{v.unit}]")
    return out
