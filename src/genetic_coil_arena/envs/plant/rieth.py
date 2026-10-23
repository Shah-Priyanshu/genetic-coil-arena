"""Rieth et al. (2017) Tennessee Eastman data: hundreds of independent runs per fault.

"Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation",
Harvard Dataverse, doi:10.7910/DVN/6C3JR1 (Rieth, Amsel, Tran and Cook, Pacific Science &
Engineering Group). It reruns the Downs and Vogel simulator with 500 different random
number generator states per condition and ships four R data frames:

- fault-free training: 500 runs of 500 samples (25 h), no fault;
- fault-free testing: 500 runs of 960 samples (48 h), no fault;
- faulty training: faults 1..20, 500 runs each of 500 samples, fault introduced after 1 h;
- faulty testing: faults 1..20, 500 runs each of 960 samples, fault introduced after 8 h.

Every sample has the same 52 variables as the Braatz data (``tep.VARIABLES``), one every
3 minutes. Fault 21 of the Braatz set is not part of this data. The training and testing
files were generated from non-overlapping seeds, so a run never appears in both.

``fetch_rieth`` downloads the four ``.RData`` files and converts them once into dense
float32 ``.npy`` arrays (see ``CACHE_FILES``), so normal use needs no R tooling. This
module loads that cache and cuts it into disjoint run-level splits.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .tep import N_VARIABLES, Normaliser, default_cache_dir
from .windows import NO_ONSET, Run

DOI = "10.7910/DVN/6C3JR1"
SOURCE_URL = "https://doi.org/10.7910/DVN/6C3JR1"
RIETH_FAULTS = tuple(range(1, 21))  # IDV(1..20); there is no fault 21
N_RUNS = 500  # Simulation runs per condition and file
TRAIN_SAMPLES = 500
TEST_SAMPLES = 960
# First faulty sample (0-based) of a faulty run, i.e. the number of normal samples before
# it. Verified from the data by ``fetch_rieth.verify_onsets`` (step faults 4, 5, 6, 7).
ONSET_TRAIN = 20
ONSET_TEST = 160
KINDS = ("training", "testing")

# Array name -> (file name, shape). Faulty arrays are (fault, run, sample, variable),
# fault-free arrays are (run, sample, variable); faults are 1..20 in order, runs 1..500.
CACHE_FILES: dict[str, tuple[str, tuple[int, ...]]] = {
    "fault_free_training": (
        "fault_free_training.npy",
        (N_RUNS, TRAIN_SAMPLES, N_VARIABLES),
    ),
    "fault_free_testing": (
        "fault_free_testing.npy",
        (N_RUNS, TEST_SAMPLES, N_VARIABLES),
    ),
    "faulty_training": (
        "faulty_training.npy",
        (len(RIETH_FAULTS), N_RUNS, TRAIN_SAMPLES, N_VARIABLES),
    ),
    "faulty_testing": (
        "faulty_testing.npy",
        (len(RIETH_FAULTS), N_RUNS, TEST_SAMPLES, N_VARIABLES),
    ),
}
SOURCE_FILE = "SOURCE.json"


def rieth_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """``rieth2017`` inside the TEP cache (``$GCA_TEP_DIR`` or the per-user cache)."""
    return (default_cache_dir() if cache_dir is None else Path(cache_dir)) / "rieth2017"


def is_converted(directory: Path) -> bool:
    """True if every converted array is present with the right size (cheap check)."""
    directory = Path(directory)
    if not (directory / SOURCE_FILE).exists():
        return False
    for filename, shape in CACHE_FILES.values():
        path = directory / filename
        want = int(np.prod(shape)) * 4 + 128  # float32 payload plus the .npy header
        if not path.exists() or not want <= path.stat().st_size <= want + 4096:
            return False
    return True


@dataclass
class RiethData:
    """The converted Rieth et al. runs (raw engineering units, float32).

    Args:
        fault_free_training: (500, 500, 52) normal runs, ``[run, sample, variable]``
        fault_free_testing: (500, 960, 52)
        faulty_training: (20, 500, 500, 52) ``[fault - 1, run, sample, variable]``; each
            run is normal for ``ONSET_TRAIN`` samples, then faulty
        faulty_testing: (20, 500, 960, 52); normal for ``ONSET_TEST`` samples, then faulty
        source: Provenance (url, licence, checksums) from ``SOURCE.json``
    Arrays may be read-only memory maps; indexing a few runs reads only those runs.
    """

    fault_free_training: np.ndarray
    fault_free_testing: np.ndarray
    faulty_training: np.ndarray
    faulty_testing: np.ndarray
    source: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, (_, shape) in CACHE_FILES.items():
            got = getattr(self, name).shape
            # Subsets (fewer runs) are allowed; the fault, sample and variable axes are not
            if got[-2:] != shape[-2:] or len(got) != len(shape):
                raise ValueError(f"{name} must be shaped like {shape}, got {got}")

    @property
    def faults(self) -> tuple[int, ...]:
        return RIETH_FAULTS[: self.faulty_training.shape[0]]

    @property
    def n_runs(self) -> int:
        """Simulation runs per condition (the smallest of the four arrays)."""
        return int(
            min(
                self.fault_free_training.shape[0],
                self.fault_free_testing.shape[0],
                self.faulty_training.shape[1],
                self.faulty_testing.shape[1],
            )
        )

    @classmethod
    def from_directory(cls, directory: Path, mmap: bool = True) -> "RiethData":
        """Load the converted arrays.

        Args:
            directory: ``rieth_cache_dir()``-style directory
            mmap: Memory-map the (up to 2 GB) arrays instead of reading them

        Raises:
            FileNotFoundError: If the cache is not converted (run ``scripts/fetch_rieth.py``)
        """
        directory = Path(directory)
        if not is_converted(directory):
            raise FileNotFoundError(
                f"{directory} has no converted Rieth et al. data; run "
                "`python scripts/fetch_rieth.py` to download and convert it"
            )
        arrays = {
            name: np.load(directory / filename, mmap_mode="r" if mmap else None)
            for name, (filename, _) in CACHE_FILES.items()
        }
        source = json.loads((directory / SOURCE_FILE).read_text(encoding="utf-8"))
        return cls(source=source, **arrays)

    @classmethod
    def from_cache(
        cls, cache_dir: Optional[Path] = None, mmap: bool = True
    ) -> "RiethData":
        """Load from ``rieth_cache_dir(cache_dir)``."""
        return cls.from_directory(rieth_cache_dir(cache_dir), mmap=mmap)

    # ------------------------------------------------------------------ run access
    def normal_runs(self, kind: str, run_ids: np.ndarray) -> np.ndarray:
        """(n, T, 52) float32 fault-free runs by 0-based run index."""
        arr = self._pick(kind, "fault_free")
        out: np.ndarray = np.asarray(
            arr[np.asarray(run_ids, dtype=np.int64)], dtype=np.float32
        )
        return out

    def fault_runs(self, kind: str, fault: int, run_ids: np.ndarray) -> np.ndarray:
        """(n, T, 52) float32 runs of one fault by 0-based run index."""
        if fault not in self.faults:
            raise KeyError(f"Fault {fault} is not in the Rieth et al. data (1..20)")
        arr = self._pick(kind, "faulty")[fault - 1]
        out: np.ndarray = np.asarray(
            arr[np.asarray(run_ids, dtype=np.int64)], dtype=np.float32
        )
        return out

    def _pick(self, kind: str, which: str) -> np.ndarray:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        array: np.ndarray = getattr(self, f"{which}_{kind}")
        return array


# --------------------------------------------------------------------------- run splits
POOL_CODE = {"train_normal": 1, "train_fault": 2, "test_normal": 3, "test_fault": 4}


@dataclass(frozen=True)
class RunSelection:
    """Which simulation runs one replicate uses (0-based indices into the arrays).

    Training runs come from the ``training`` files; validation and test runs from the
    ``testing`` files, which are longer (960 samples) and so match deployment.

    Args:
        train_normal: (n,) indices into ``fault_free_training``
        train_fault: Fault -> (n,) indices into ``faulty_training[fault - 1]``
        val_normal: (n,) indices into ``fault_free_testing``
        val_fault: Fault -> (n,) indices into ``faulty_testing[fault - 1]``
        test_normal, test_fault: Likewise, disjoint from the validation runs
    """

    train_normal: np.ndarray
    train_fault: dict[int, np.ndarray]
    val_normal: np.ndarray
    val_fault: dict[int, np.ndarray]
    test_normal: np.ndarray
    test_fault: dict[int, np.ndarray]

    def keys(self, part: str) -> set[tuple[str, int, int]]:
        """Simulation runs of a part (``train``, ``val`` or ``test``) as unique keys.

        A key is ``(file kind, fault, run index)`` with fault 0 for fault-free runs, so
        two selections share a simulation run iff their key sets intersect.
        """
        kind = "training" if part == "train" else "testing"
        normal = getattr(self, f"{part}_normal")
        faulty = getattr(self, f"{part}_fault")
        out = {(kind, 0, int(r)) for r in normal}
        for fault, runs in faulty.items():
            out |= {(kind, int(fault), int(r)) for r in runs}
        return out


@dataclass(frozen=True)
class RunSplits:
    """Disjoint run-level splits for a hyperparameter-tuning stage and n replicates.

    Every (file, fault) pool of 500 runs is permuted once with ``split_seed``; the
    permutation is cut into consecutive blocks: block 0 for tuning, block ``1 + i`` for
    replicate ``i``. Blocks never overlap, so no simulation run is used by two
    replicates, by a replicate and the tuning stage, or twice within one replicate (the
    validation and test runs of a replicate are different blocks' slices of the same
    permutation). Hyperparameters are chosen on the tuning stage's validation runs and
    never see a replicate's test runs.
    """

    faults: tuple[int, ...]
    n_replicates: int
    train_runs: int
    val_runs: int
    test_runs: int
    train_normal_runs: int
    val_normal_runs: int
    test_normal_runs: int
    split_seed: int = 0
    n_available: int = N_RUNS

    def __post_init__(self) -> None:
        if any(f not in RIETH_FAULTS for f in self.faults) or not self.faults:
            raise ValueError("faults must be distinct ids in 1..20")
        if min(self.train_runs, self.val_runs, self.test_runs) < 1:
            raise ValueError("Every split needs at least one run per fault")
        blocks = self.n_replicates + 1  # + the tuning block
        need = {
            "faulty_training": blocks * self.train_runs,
            "fault_free_training": blocks * self.train_normal_runs,
            "faulty_testing": blocks * (self.val_runs + self.test_runs),
            "fault_free_testing": blocks
            * (self.val_normal_runs + self.test_normal_runs),
        }
        for pool, count in need.items():
            if count > self.n_available:
                raise ValueError(
                    f"{pool}: {blocks} blocks need {count} runs, only {self.n_available} "
                    "available; use fewer replicates or runs"
                )

    def selection(self, replicate: Optional[int]) -> RunSelection:
        """Runs of replicate ``replicate`` (0-based), or of the tuning stage if ``None``."""
        if replicate is not None and not 0 <= replicate < self.n_replicates:
            raise ValueError(f"replicate must lie in [0, {self.n_replicates})")
        block = 0 if replicate is None else replicate + 1
        n_test_pool = self.val_runs + self.test_runs
        n_test_normal = self.val_normal_runs + self.test_normal_runs

        def perm(pool: str, fault: int) -> np.ndarray:
            rng = np.random.default_rng([self.split_seed, POOL_CODE[pool], fault])
            return rng.permutation(self.n_available)

        def cut(order: np.ndarray, size: int) -> np.ndarray:
            return order[block * size : (block + 1) * size]

        train_fault = {
            f: cut(perm("train_fault", f), self.train_runs) for f in self.faults
        }
        val_fault, test_fault = {}, {}
        for f in self.faults:
            both = cut(perm("test_fault", f), n_test_pool)
            val_fault[f], test_fault[f] = both[: self.val_runs], both[self.val_runs :]
        both_n = cut(perm("test_normal", 0), n_test_normal)
        return RunSelection(
            train_normal=cut(perm("train_normal", 0), self.train_normal_runs),
            train_fault=train_fault,
            val_normal=both_n[: self.val_normal_runs],
            val_fault=val_fault,
            test_normal=both_n[self.val_normal_runs :],
            test_fault=test_fault,
        )


def to_runs(
    data: RiethData,
    normaliser: Normaliser,
    selection: RunSelection,
    part: str,
    faults: tuple[int, ...],
) -> list[Run]:
    """The normalised ``Run`` objects of one part (``train``, ``val`` or ``test``).

    Fault-free runs come first, then the faults in ``faults`` order, runs in index order.
    Training runs have their fault from sample ``ONSET_TRAIN``, validation and test runs
    from ``ONSET_TEST``.
    """
    if part not in ("train", "val", "test"):
        raise ValueError(f"part must be 'train', 'val' or 'test', got {part!r}")
    kind = "training" if part == "train" else "testing"
    onset = ONSET_TRAIN if part == "train" else ONSET_TEST
    normal_ids = getattr(selection, f"{part}_normal")
    fault_ids = getattr(selection, f"{part}_fault")
    runs = [
        Run(0, normaliser.transform(series), NO_ONSET)
        for series in data.normal_runs(kind, np.sort(normal_ids))
    ]
    for fault in faults:
        for series in data.fault_runs(kind, fault, np.sort(fault_ids[fault])):
            runs.append(Run(int(fault), normaliser.transform(series), onset))
    return runs


def synthetic_rieth(
    n_runs: int = 8, n_faults: int = 3, seed: int = 0, strength: float = 4.0
) -> RiethData:
    """A small stand-in with the real shapes and onsets but few runs and faults.

    Normal operation is stationary AR(1) noise around per-variable levels, different in
    every run. Fault ``f`` (1..``n_faults``) shifts a fixed random subset of six variables
    by ``strength`` standard deviations from sample ``ONSET_TRAIN`` (training runs) or
    ``ONSET_TEST`` (testing runs). For tests and demos where the download is unavailable;
    carries no process physics.
    """
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    level = rng.normal(0.0, 50.0, size=N_VARIABLES)
    scale = rng.uniform(0.1, 5.0, size=N_VARIABLES)
    shift_rng = np.random.default_rng(1000 + seed)
    shifts = []
    for _ in range(n_faults):
        shift = np.zeros(N_VARIABLES)
        picked = shift_rng.choice(N_VARIABLES, size=6, replace=False)
        shift[picked] = strength * shift_rng.choice([-1.0, 1.0], size=6)
        shifts.append(shift)

    def runs(prefix: tuple[int, ...], samples: int, onset: int) -> np.ndarray:
        e = rng.normal(size=prefix + (samples, N_VARIABLES))
        z = lfilter([np.sqrt(1 - 0.49)], [1.0, -0.7], e, axis=-2)
        if len(prefix) == 2:  # faulty: (fault, run)
            for f in range(prefix[0]):
                z[f, :, onset:, :] += shifts[f]
        out: np.ndarray = (level + scale * z).astype(np.float32)
        return out

    return RiethData(
        fault_free_training=runs((n_runs,), TRAIN_SAMPLES, ONSET_TRAIN),
        fault_free_testing=runs((n_runs,), TEST_SAMPLES, ONSET_TEST),
        faulty_training=runs((n_faults, n_runs), TRAIN_SAMPLES, ONSET_TRAIN),
        faulty_testing=runs((n_faults, n_runs), TEST_SAMPLES, ONSET_TEST),
        source={"url": "synthetic", "license": "n/a"},
    )
