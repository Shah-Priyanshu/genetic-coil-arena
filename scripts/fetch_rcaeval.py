"""Download and convert RCAEval RE2 (per-service logs of injected microservice faults), RE3 and RE1.

Pham et al. (2025), Zenodo record 14590730, CC-BY-4.0. Only the logs and the injection
time of each case are kept: the archives (SS 0.25 GB, OB 1.2 GB, TT 2.8 GB) are checked
against Zenodo's MD5, each incident is converted once to a compressed array file of
templated lines, and the source, licence and checksums go to ``SOURCE.json``. Nothing is
written inside the repo.

    python scripts/fetch_rcaeval.py                    # SS and OB
    python scripts/fetch_rcaeval.py --systems SS,OB,TT --drop-raw
    python scripts/fetch_rcaeval.py --systems RE3-SS,RE3-OB,RE3-TT   # RE3 (0.5 GB, code faults)
    python scripts/fetch_rcaeval.py --systems RE1-SS,RE1-OB,RE1-TT   # RE1 (0.4 GB, metrics only, no logs)
    python scripts/fetch_rcaeval.py --check            # list the cache, no network

The cache is %LOCALAPPDATA%\\genetic_coil_arena\\rcaeval on Windows, otherwise
~/.cache/genetic_coil_arena/rcaeval; override with --cache-dir or $GCA_RCAEVAL_DIR.
"""

import argparse
import sys
from pathlib import Path

from genetic_coil_arena.envs.logtriage.rcaeval import (
    ARCHIVES,
    CITATION,
    DEFAULT_SYSTEMS,
    LICENSE,
    cache_dir,
    fetch,
    list_incidents,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--systems",
        default=",".join(DEFAULT_SYSTEMS),
        help="comma list of SS,OB,TT (RE2) and RE3-SS,RE3-OB,RE3-TT (RE3), RE1-SS,RE1-OB,RE1-TT (RE1, metrics only)",
    )
    parser.add_argument("--check", action="store_true", help="only count the cache")
    parser.add_argument("--drop-raw", action="store_true", help="delete the zips after")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    systems = tuple(s for s in args.systems.split(",") if s)
    for s in systems:
        if s not in ARCHIVES:
            parser.error(f"unknown system {s!r}; choose from {sorted(ARCHIVES)}")
    root = cache_dir(args.cache_dir)
    if args.check:
        files = list_incidents(args.cache_dir, systems)
        print(f"{root}: {len(files)} converted incidents for {','.join(systems)}")
        return 0 if files else 1
    fetch(args.cache_dir, systems, keep_raw=not args.drop_raw, timeout=args.timeout)
    print(f"RCAEval RE2 logs ready in {root}")
    print(f"cite:    {CITATION}")
    print(f"licence: {LICENSE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
