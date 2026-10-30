"""R17 step 4: what the scout brain must reach before a search works (experiments/r17e_requirements.py).

tune     tune each held-out mission's plan per (camera, frames, fa_mult)
report   score default, hand and tuned plans -> docs/results/r17e_requirements.{md,json}

python scripts/r17e_requirements.py tune --camera "K210 224=docs/results/r15b_aerial_scout.json" --frames 3
python scripts/r17e_requirements.py report --camera "K210 224=docs/results/r15b_aerial_scout.json" --from runs:r17e-tune-*
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r17e_requirements as rq  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402


def _cameras(specs: list[str]) -> dict[str, dict]:
    out = {}
    for spec in specs:
        name, path = spec.split("=", 1)
        out[name] = rq.camera_table(Path(path))
    return out


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def tune(args: argparse.Namespace) -> None:
    out = Path(os.environ.get("GCA_RUN_DIR", "outputs/r17e"))
    out.mkdir(parents=True, exist_ok=True)
    a, b = (int(x) for x in args.missions.split(":"))
    for name, cam in _cameras(args.camera).items():
        todo = [(cam, f, x, i, args.generations)
                for f in args.frames for x in args.fa_mults for i in range(a, b)]  # fmt: skip
        start = time.perf_counter()
        with ProcessPoolExecutor(args.workers) as pool:
            for k, rec in enumerate(pool.map(rq.tune_task, todo), 1):
                rec["camera"] = name
                path = out / (f"r17e_tuned_{_slug(name)}_f{rec['frames']:g}"
                              f"_x{rec['fa_mult']:g}_m{rec['mission']}.json")  # fmt: skip
                path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
                if k % 20 == 0 or k == len(todo):
                    print(f"[progress] {name}: {k}/{len(todo)} tuned, "
                          f"{time.perf_counter() - start:.0f} s", flush=True)  # fmt: skip


def report(args: argparse.Namespace) -> None:
    cameras = _cameras(args.camera)
    tuned = {}
    for folder in resolve_sources(args.sources):
        for f in sorted(Path(folder).rglob("r17e_tuned_*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            if rec["camera"] in cameras:
                key = (rec["camera"], rec["frames"], rec["fa_mult"], rec["mission"])
                tuned[key] = rec
    print(f"[load] {len(tuned)} tuned missions", flush=True)
    todo = rq.eval_tasks(cameras, tuned, args.episodes)
    cells, start = {}, time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for k, (key, res) in enumerate(pool.map(rq.eval_task, todo, chunksize=8), 1):
            cells[key] = res
            if k % 200 == 0 or k == len(todo):
                print(f"[progress] {k}/{len(todo)} cells, "
                      f"{time.perf_counter() - start:.0f} s", flush=True)  # fmt: skip
    rep = rq.summarise(cells)
    md = rq.render(rep, args.episodes)
    js = json.dumps(rep, indent=1)
    outs = [args.out]
    if os.environ.get("GCA_RUN_DIR"):
        outs.append(Path(os.environ["GCA_RUN_DIR"]) / args.out.name)
    for out in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        out.with_suffix(".json").write_text(js, encoding="utf-8")
    print(md, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    workers = int(os.environ.get("GCA_WORKERS", 4))
    t = sub.add_parser("tune")
    t.add_argument("--camera", action="append", required=True, help="name=report.json")
    t.add_argument("--frames", type=float, nargs="+", default=list(rq.FRAMES))
    t.add_argument("--fa-mults", type=float, nargs="+", default=list(rq.FA_MULTS))
    t.add_argument("--missions", default="0:40")
    t.add_argument("--generations", type=int, default=200)
    t.add_argument("--workers", type=int, default=workers)
    r = sub.add_parser("report")
    r.add_argument("--camera", action="append", required=True, help="name=report.json")
    r.add_argument("--from", dest="sources", nargs="+", required=True)
    r.add_argument("--episodes", type=int, default=rq.EPISODES)
    r.add_argument("--workers", type=int, default=workers)
    r.add_argument(
        "--out", type=Path, default=Path("docs/results/r17e_requirements.md")
    )
    args = p.parse_args()
    tune(args) if args.cmd == "tune" else report(args)


if __name__ == "__main__":
    main()
