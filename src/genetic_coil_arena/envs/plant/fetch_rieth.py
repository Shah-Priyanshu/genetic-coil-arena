"""Download, verify and convert the Rieth et al. (2017) Tennessee Eastman data.

The four ``.RData`` files (about 1.4 GB) come from Harvard Dataverse, doi:10.7910/DVN/6C3JR1.
Each is checked against its pinned size, the MD5 that Dataverse publishes and a SHA-256
taken at download time, so a truncated or altered file fails loudly. They are converted
once, with the dependency-free reader in ``rdata``, into dense float32 ``.npy`` arrays
(about 3.2 GB, see ``rieth.CACHE_FILES``) plus ``SOURCE.json``, which records the source,
the licence, the checksums of every raw file and array, and the fault onsets measured
from the data. Nothing is written inside the repo. Converting the largest file needs about
5 GB of memory.
"""

import datetime
import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .rdata import data_frame_columns, read_rdata
from .rieth import (
    CACHE_FILES,
    DOI,
    N_RUNS,
    ONSET_TEST,
    ONSET_TRAIN,
    RIETH_FAULTS,
    SOURCE_FILE,
    SOURCE_URL,
    is_converted,
    rieth_cache_dir,
)
from .tep import N_VARIABLES, VARIABLES

DOWNLOAD_URL = "https://dataverse.harvard.edu/api/access/datafile/{id}"
CITATION = (
    "Rieth, C. A., Amsel, B. D., Tran, R. and Cook, M. B. (2017). Additional Tennessee "
    "Eastman Process Simulation Data for Anomaly Detection Evaluation. Harvard Dataverse, "
    "V1. https://doi.org/10.7910/DVN/6C3JR1"
)
LICENSE = (
    "Public domain dedication (Harvard Dataverse terms of use for this dataset): the "
    "contributors dedicated the work to the public domain and waived their rights "
    "worldwide under copyright law; it may be copied, modified, distributed and "
    "performed for any lawful purpose without asking permission. No warranty and no "
    "implied endorsement by Pacific Science & Engineering Group, Inc. or the authors."
)
COLUMNS = (
    ["faultNumber", "simulationRun", "sample"]
    + [f"xmeas_{i}" for i in range(1, 42)]
    + [f"xmv_{i}" for i in range(1, 12)]
)


@dataclass(frozen=True)
class RawFile:
    """One published ``.RData`` file and its pinned checksums."""

    key: str
    name: str
    file_id: int
    doi: str
    size: int
    md5: str
    sha256: str
    faulty: bool
    n_samples: int


RAW_FILES: dict[str, RawFile] = {
    f.key: f
    for f in (
        RawFile(
            "fault_free_training",
            "TEP_FaultFree_Training.RData",
            3031241,
            "doi:10.7910/DVN/6C3JR1/TMBUVJ",
            24678017,
            "ec126484534331f85001d8c4ebce6d17",
            "12d1055b852763fac09ef36bd7c9612a800c8bf7b6f75f344d12119206f9e940",
            False,
            500,
        ),
        RawFile(
            "fault_free_testing",
            "TEP_FaultFree_Testing.RData",
            3031240,
            "doi:10.7910/DVN/6C3JR1/Q4NOFD",
            47327663,
            "38ad9810fc871026157086ae2c2f0ee9",
            "4f45afafa469793eeb7203fb9ed10ed0b1724c73c9c95537f15a0889ade0ebd4",
            False,
            960,
        ),
        RawFile(
            "faulty_training",
            "TEP_Faulty_Training.RData",
            3031242,
            "doi:10.7910/DVN/6C3JR1/6QOWUU",
            494063194,
            "c5f594d54c47e620ff877feb58407fda",
            "058df8bd90c5f575f44d2f6085bdd3bc3662e3e9cc25a50e34af42be3748952b",
            True,
            500,
        ),
        RawFile(
            "faulty_testing",
            "TEP_Faulty_Testing.RData",
            3031243,
            "doi:10.7910/DVN/6C3JR1/IL0XP0",
            836882037,
            "556bdb64c83021bc0c5f92e427753565",
            "c2f1aabc77498efda4d5f9ad81ecc829e2fcd98752b84cf5b2a06558cabacc36",
            True,
            960,
        ),
    )
}


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check_raw(raw: RawFile, path: Path) -> None:
    """Verify a downloaded file's size, MD5 and SHA-256.

    Raises:
        ValueError: On any mismatch
    """
    if path.stat().st_size != raw.size:
        raise ValueError(f"{path.name}: {path.stat().st_size} bytes != {raw.size}")
    if _digest(path, "md5") != raw.md5:
        raise ValueError(f"{path.name}: MD5 differs from the published checksum")
    if _digest(path, "sha256") != raw.sha256:
        raise ValueError(f"{path.name}: SHA-256 differs from the pinned checksum")


def _download(url: str, dest: Path, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "genetic-coil-arena"})
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        with open(part, "wb") as out:
            while chunk := response.read(1 << 22):
                out.write(chunk)
    part.replace(dest)


def _dense(raw: RawFile, path: Path) -> np.ndarray:
    """Read one RData file into a dense float32 array (see ``rieth.CACHE_FILES``)."""
    frames = read_rdata(path, np.float32)
    if list(frames) != [raw.key]:
        raise ValueError(
            f"{path.name}: expected one object {raw.key!r}, got {list(frames)}"
        )
    cols = data_frame_columns(frames.pop(raw.key))
    if list(cols) != COLUMNS:
        raise ValueError(f"{path.name}: unexpected columns {list(cols)[:6]}...")
    fault = cols.pop("faultNumber").astype(np.int64)
    run = cols.pop("simulationRun").astype(np.int64)
    sample = cols.pop("sample").astype(np.int64)
    n_rows = len(fault)
    faults = np.unique(fault)
    n_faults = len(faults) if raw.faulty else 1
    want_faults = np.array(RIETH_FAULTS) if raw.faulty else np.array([0])
    if not np.array_equal(faults, want_faults):
        raise ValueError(
            f"{path.name}: fault ids {faults.tolist()} != {want_faults.tolist()}"
        )
    if n_rows != n_faults * N_RUNS * raw.n_samples:
        raise ValueError(f"{path.name}: {n_rows} rows is not a complete run grid")
    # Rows must be ordered fault, run, sample: check, and reorder only if they are not
    grid = (n_faults, N_RUNS, raw.n_samples)
    in_order = (
        np.array_equal(fault.reshape(grid)[:, 0, 0], want_faults)
        and np.array_equal(sample.reshape(grid)[0, 0], np.arange(1, raw.n_samples + 1))
        and np.array_equal(run.reshape(grid)[0, :, 0], np.arange(1, N_RUNS + 1))
        and bool((fault.reshape(grid) == want_faults[:, None, None]).all())
        and bool((run.reshape(grid) == np.arange(1, N_RUNS + 1)[None, :, None]).all())
        and bool((sample.reshape(grid) == np.arange(1, raw.n_samples + 1)).all())
    )
    order = None if in_order else np.lexsort((sample, run, fault))
    dense = np.empty((n_rows, N_VARIABLES), dtype=np.float32)
    for j, name in enumerate(COLUMNS[3:]):
        col = cols.pop(name)
        dense[:, j] = col if order is None else col[order]
    shape = CACHE_FILES[raw.key][1]
    return dense.reshape(shape)


def verify_onsets(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    """First faulty sample (0-based) of the faulty training and testing runs, from the data.

    Faults 4, 5, 6 and 7 are step changes that act on their very first sample. For each,
    the mean over the 500 runs of the largest absolute z-score (against the fault-free
    runs of the same file) is compared with the fault-free runs' own value: the excess is
    zero before the fault and jumps by several z at its first sample. The onset is the
    first sample whose excess exceeds 1.0, which must follow at least ten samples with an
    excess below 0.5.

    Returns:
        ``{"training": n, "testing": n}``: the number of normal samples before the fault

    Raises:
        ValueError: If a fault shows no clean onset or the four faults disagree
    """
    found: dict[str, int] = {}
    for kind in ("training", "testing"):
        free = np.asarray(arrays[f"fault_free_{kind}"], dtype=np.float64)
        mean, std = free.mean(axis=(0, 1)), np.maximum(free.std(axis=(0, 1)), 1e-9)
        baseline = np.abs((free - mean) / std).max(axis=2).mean(axis=0)  # (T,)
        onsets = []
        for fault in (4, 5, 6, 7):
            runs = np.asarray(arrays[f"faulty_{kind}"][fault - 1], dtype=np.float64)
            excess = np.abs((runs - mean) / std).max(axis=2).mean(axis=0) - baseline
            jump = int(np.argmax(excess > 1.0))
            if jump < 10 or excess[jump] <= 1.0 or np.abs(excess[:jump]).max() >= 0.5:
                raise ValueError(f"No clean onset for fault {fault} ({kind})")
            onsets.append(jump)
        if len(set(onsets)) != 1:
            raise ValueError(
                f"Faults 4, 5, 6, 7 disagree on the {kind} onset: {onsets}"
            )
        found[kind] = onsets[0]
    return found


def is_complete(directory: Path) -> bool:
    """True if the cache holds every converted array and ``SOURCE.json``."""
    return is_converted(directory)


def fetch_rieth(
    cache_dir: Optional[Path] = None,
    force: bool = False,
    keep_raw: bool = True,
    timeout: float = 120.0,
    progress: Callable[[str], None] = print,
) -> Path:
    """Make sure the converted Rieth et al. arrays are in the cache; return its path.

    Args:
        cache_dir: TEP cache root (default ``tep.default_cache_dir()``); the data goes
            to its ``rieth2017`` subdirectory
        force: Download and convert again even if a converted cache exists
        keep_raw: Keep the ``.RData`` downloads in ``rieth2017/raw`` (else delete them)
        timeout: Per-request timeout in seconds
        progress: Receives one line per step

    Raises:
        ValueError: If a download fails verification or the data is not what the
            documentation promises (nothing partial is left marked as complete)
        OSError: If Dataverse is unreachable
    """
    target = rieth_cache_dir(cache_dir)
    if not force and is_converted(target):
        return target
    raw_dir = target / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (target / SOURCE_FILE).unlink(missing_ok=True)  # incomplete until the end

    for raw in RAW_FILES.values():
        path = raw_dir / raw.name
        if path.exists():
            try:
                check_raw(raw, path)
                progress(f"{raw.name}: already downloaded and verified")
                continue
            except ValueError:
                path.unlink()
        progress(f"{raw.name}: downloading {raw.size / 1e6:.0f} MB")
        try:
            _download(DOWNLOAD_URL.format(id=raw.file_id), path, timeout)
        except (urllib.error.URLError, OSError) as error:
            raise OSError(
                f"Could not download {raw.name} from Dataverse: {error}"
            ) from error
        try:
            check_raw(raw, path)
        except ValueError:
            path.unlink()
            raise

    arrays: dict[str, np.ndarray] = {}
    array_info: dict[str, dict] = {}
    for key, raw in RAW_FILES.items():
        progress(f"{raw.name}: converting")
        dense = _dense(raw, raw_dir / raw.name)
        filename = CACHE_FILES[key][0]
        np.save(target / filename, dense)
        array_info[key] = {
            "file": filename,
            "shape": list(dense.shape),
            "dtype": "float32",
            "sha256": _digest(target / filename, "sha256"),
        }
        arrays[key] = np.load(target / filename, mmap_mode="r")
        del dense
    onsets = verify_onsets(arrays)
    if onsets != {"training": ONSET_TRAIN, "testing": ONSET_TEST}:
        raise ValueError(
            f"Measured onsets {onsets} differ from the documented "
            f"{{'training': {ONSET_TRAIN}, 'testing': {ONSET_TEST}}}"
        )
    record = {
        "dataset": "Additional Tennessee Eastman Process Simulation Data for Anomaly "
        "Detection Evaluation (Rieth et al. 2017)",
        "doi": DOI,
        "url": SOURCE_URL,
        "citation": CITATION,
        "license": LICENSE,
        "retrieved": datetime.date.today().isoformat(),
        "runs_per_condition": N_RUNS,
        "faults": list(RIETH_FAULTS),
        "variables": [v.tag for v in VARIABLES],
        "samples": {"training": 500, "testing": 960},
        "first_faulty_sample_0based": onsets,
        "subset": "none (all 500 runs of every condition)",
        "raw_files": {
            r.key: {
                "name": r.name,
                "dataverse_file_id": r.file_id,
                "doi": r.doi,
                "bytes": r.size,
                "md5": r.md5,
                "sha256": r.sha256,
            }
            for r in RAW_FILES.values()
        },
        "arrays": array_info,
    }
    (target / SOURCE_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    if not keep_raw:
        for raw in RAW_FILES.values():
            (raw_dir / raw.name).unlink(missing_ok=True)
    return target
