"""Download and convert the Rieth et al. (2017) Tennessee Eastman data into the cache.

"Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation",
Harvard Dataverse, doi:10.7910/DVN/6C3JR1: 500 independent simulation runs per condition
(fault-free and faults 1..20), a training file (500 samples, fault after sample 20) and a
testing file (960 samples, fault after sample 160). The four .RData files (1.4 GB) are
checked against pinned sizes and checksums, converted once to float32 .npy arrays (3.2 GB)
with a dependency-free reader (no R, no pyreadr), and the fault onsets are measured from
the data. The source, licence and checksums are recorded in ``SOURCE.json``. Nothing is
written inside the repo.

    python scripts/fetch_rieth.py                      # default cache
    python scripts/fetch_rieth.py --cache-dir D:/data/tep --drop-raw
    python scripts/fetch_rieth.py --check              # verify only, no network

The data goes to the ``rieth2017`` folder of the TEP cache: %LOCALAPPDATA%\\genetic_coil_arena\\tep
on Windows, otherwise ~/.cache/genetic_coil_arena/tep; override with --cache-dir or
$GCA_TEP_DIR. Converting the largest file needs about 5 GB of memory.
"""

import argparse
import sys
from pathlib import Path

from genetic_coil_arena.envs.plant.fetch_rieth import (
    CITATION,
    LICENSE,
    fetch_rieth,
    is_complete,
)
from genetic_coil_arena.envs.plant.rieth import rieth_cache_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=None, help="TEP cache root")
    parser.add_argument(
        "--force", action="store_true", help="Convert again even if the cache is valid"
    )
    parser.add_argument(
        "--check", action="store_true", help="Only check the cache, never download"
    )
    parser.add_argument(
        "--drop-raw", action="store_true", help="Delete the .RData downloads afterwards"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="Seconds per request"
    )
    args = parser.parse_args()

    target = rieth_cache_dir(args.cache_dir)
    if args.check:
        ok = is_complete(target)
        print(
            f"{target}: {'converted and complete' if ok else 'missing or incomplete'}"
        )
        return 0 if ok else 1

    path = fetch_rieth(
        args.cache_dir,
        force=args.force,
        keep_raw=not args.drop_raw,
        timeout=args.timeout,
    )
    print(f"Rieth et al. data ready in {path}")
    print(f"cite:    {CITATION}")
    print(f"licence: {LICENSE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
