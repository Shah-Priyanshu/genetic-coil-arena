"""R13b probe: k-shot adaptation of the frozen G4c fault specialist on each RE1 system.

Thin wrapper around ``genetic_coil_arena.experiments.r13b_kshot``; see that module.

    python scripts/r13b_kshot.py --team-from runs:r11-team --out docs/results/r13b_kshot.md
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r11_team as team  # noqa: E402
from genetic_coil_arena.experiments import r13_real_centre as r13  # noqa: E402
from genetic_coil_arena.experiments import r13b_kshot as probe  # noqa: E402
from genetic_coil_arena.experiments.r7a_step2 import (  # noqa: E402
    find_file,
    resolve_runs,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--team-from", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--split", choices=probe.SPLITS, default="random")
    p.add_argument("--out", type=Path, default=Path("docs/results/r13b_kshot.md"))
    args = p.parse_args()
    cache = Path(args.cache_dir) if args.cache_dir else None
    state = json.loads(
        find_file(resolve_runs([args.team_from]), "team.json").read_text(
            encoding="utf-8"
        )
    )
    digest = r13.assert_frozen(state)
    model = team.load_specialist(state).model
    r13.ensure_re1(cache)
    start = time.time()
    rows = [probe.incident_row(model, inc) for inc in r13.load_re1(cache)]
    print(
        f"[r13b] {len(rows)} incidents featurised in {time.time() - start:.0f} s",
        flush=True,
    )
    if r13.model_sha256(state) != digest:
        raise SystemExit("the frozen model changed")
    systems = {}
    for i, system in enumerate(sorted({r["system"] for r in rows})):
        part = [r for r in rows if r["system"] == system]
        systems[system] = probe.run_system(part, i, args.split)
        print(f"[r13b] {system} done", flush=True)
    rep = {"systems": systems, "model_sha256": digest, "split": args.split}
    md = probe.render(rep)
    js = json.dumps(
        rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(js, encoding="utf-8")
    run_dir = os.environ.get("GCA_RUN_DIR")
    if run_dir:
        (Path(run_dir) / args.out.name).write_text(md, encoding="utf-8")
        (Path(run_dir) / args.out.with_suffix(".json").name).write_text(
            js, encoding="utf-8"
        )
    print(md, flush=True)


if __name__ == "__main__":
    main()
