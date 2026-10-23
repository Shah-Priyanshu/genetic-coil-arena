"""Download the Tennessee Eastman Process benchmark data into a cache outside the repo.

The classic Braatz-group files d00-d21 (training and test, 52 variables, fault at sample
160 of every test run) come from https://web.mit.edu/braatzgroup/TE_process.zip, with a
public mirror as fallback. Each file is checked against a pinned SHA-256 and its row and
column count. The source URL, licence and checksums are recorded in ``SOURCE.json`` next
to the data. Nothing is written inside the repo.

    python scripts/fetch_tep.py                       # default cache
    python scripts/fetch_tep.py --cache-dir D:/data/tep --force
    python scripts/fetch_tep.py --check               # verify only, no network

Default cache: %LOCALAPPDATA%\\genetic_coil_arena\\tep on Windows, otherwise
~/.cache/genetic_coil_arena/tep; override with --cache-dir or $GCA_TEP_DIR.
"""

import argparse
import sys
from pathlib import Path

from genetic_coil_arena.envs.plant.fetch import fetch, is_complete
from genetic_coil_arena.envs.plant.tep import (
    SOURCE_LICENSE,
    SOURCE_URL,
    default_cache_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=None, help="Target directory")
    parser.add_argument(
        "--force", action="store_true", help="Download again even if the cache is valid"
    )
    parser.add_argument(
        "--check", action="store_true", help="Only verify the cache, never download"
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Seconds per request"
    )
    args = parser.parse_args()

    target = args.cache_dir or default_cache_dir()
    if args.check:
        ok = is_complete(target)
        print(f"{target}: {'complete and verified' if ok else 'missing or corrupt'}")
        return 0 if ok else 1

    path = fetch(target, force=args.force, timeout=args.timeout)
    print(f"TEP data verified in {path}")
    print(f"source:  {SOURCE_URL}")
    print(f"licence: {SOURCE_LICENSE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
