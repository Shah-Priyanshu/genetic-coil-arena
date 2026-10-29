"""R17: a leader that plans OpenSwarm's person search from the mission.

train    ES-train planners (blind or conditioned), one process per run
oracle   tune each held-out mission's plan alone (the room a planner could win)
report   evaluate every system on the held-out missions -> docs/results/r17_mission.{md,json}

python scripts/r17_mission.py train --kind conditioned --runs 0:8 --workers 8
python scripts/r17_mission.py oracle --missions 0:40 --workers 8
python scripts/r17_mission.py report --from runs:r17-* --out docs/results/r17_mission.md
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r17_mission as r17  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402


def _span(text: str) -> range:
    a, b = text.split(":")
    return range(int(a), int(b))


def _out() -> Path:
    return Path(os.environ.get("GCA_RUN_DIR", "outputs/r17"))


def _train_one(args: tuple) -> str:
    run, settings, out = args
    res = r17.train_run(run, settings)
    stem = r17.save_run(res, out)
    print(f"[done] run {run}: best val {res['best_val']:.3f} at generation "
          f"{res['best_generation']}, {res['seconds']:.0f} s", flush=True)  # fmt: skip
    return str(stem)


def _oracle_one(args: tuple) -> str:
    index, generations, version, out = args
    res = r17.tune_mission(index, generations=generations, version=version)
    prefix = "r17_oracle_m" if version == 1 else "r17b_oracle_m"
    path = out / f"{prefix}{index}.json"
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(
        f"[done] mission {index}: val {res['val']:.3f} from {res['start']}", flush=True
    )
    return str(path)


def train(args: argparse.Namespace) -> None:
    settings = replace(r17.PlannerSettings(), conditioned=args.kind == "conditioned",
                       generations=args.generations, version=args.version)  # fmt: skip
    out = _out()
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(run, settings, out) for run in _span(args.runs)]
    with ProcessPoolExecutor(args.workers) as pool:
        for stem in pool.map(_train_one, tasks):
            print(f"[saved] {stem}", flush=True)


def oracle(args: argparse.Namespace) -> None:
    out = _out()
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(i, args.generations, args.version, out) for i in _span(args.missions)]
    with ProcessPoolExecutor(args.workers) as pool:
        for path in pool.map(_oracle_one, tasks):
            print(f"[saved] {path}", flush=True)


def report(args: argparse.Namespace) -> None:
    folders = resolve_sources(args.sources)
    blind = r17.load_runs(folders, "blind")
    cond = r17.load_runs(folders, "conditioned")
    oracle_plans = r17.load_oracle(folders)
    print(f"[load] blind {len(blind)}, conditioned {len(cond)}, "
          f"oracle {len(oracle_plans)} missions", flush=True)  # fmt: skip
    plans = r17.system_plans(blind, cond, oracle_plans)
    tasks = r17.eval_tasks(plans, args.episodes)
    cells, start = {}, time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (key, res) in enumerate(pool.map(r17.eval_cell, tasks, chunksize=8), 1):
            cells[key] = res
            if i % 200 == 0 or i == len(tasks):
                print(f"[progress] {i}/{len(tasks)} cells, "
                      f"{time.perf_counter() - start:.0f} s", flush=True)  # fmt: skip
    rep = r17.summarise(cells, plans)
    rep["examples"] = r17.example_plans(plans)
    rep["plans"] = r17.plan_table(plans)
    md = r17.render(rep)
    js = json.dumps(rep, indent=1, default=float)
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    workers = int(os.environ.get("GCA_WORKERS", 4))
    t = sub.add_parser("train")
    t.add_argument("--kind", choices=["blind", "conditioned"], required=True)
    t.add_argument("--runs", default="0:8")
    t.add_argument("--version", type=int, choices=[1, 2], default=1,
                   help="2: step 2 missions and the hand-off (experiments/r17b_handoff.py)")  # fmt: skip
    t.add_argument("--generations", type=int, default=r17.PlannerSettings.generations)
    t.add_argument("--workers", type=int, default=workers)
    o = sub.add_parser("oracle")
    o.add_argument("--missions", default=f"0:{r17.N_TEST}")
    o.add_argument("--generations", type=int, default=300)
    o.add_argument("--version", type=int, choices=[1, 2], default=1)
    o.add_argument("--workers", type=int, default=workers)
    r = sub.add_parser("report")
    r.add_argument("--from", dest="sources", nargs="+", required=True)
    r.add_argument("--episodes", type=int, default=r17.TEST_EPISODES)
    r.add_argument("--workers", type=int, default=workers)
    r.add_argument("--out", type=Path, default=Path("docs/results/r17_mission.md"))
    args = p.parse_args()
    {"train": train, "oracle": oracle, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
