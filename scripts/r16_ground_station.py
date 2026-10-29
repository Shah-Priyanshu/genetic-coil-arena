"""R16 (G14): the plain-language ground station for OpenSwarm.

run      the dev or sealed request set through one translator (GPU for ollama:<tag>)
report   merge run outputs into docs/results/r16_ground_station.{md,json}

python scripts/r16_ground_station.py run --translator keyword --split dev
python scripts/r16_ground_station.py run --translator ollama:qwen2.5-coder:14b --split sealed
python scripts/r16_ground_station.py report --from runs:r16-* --out docs/results/r16_ground_station.md
"""

import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from genetic_coil_arena.experiments import r16_ground_station as r16  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402

DATA = Path(r16.__file__).parent / "data"
KEEP = "30m"
CTX = {"num_ctx": 4096}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def items_for(split: str) -> list[dict[str, Any]]:
    if split in ("sealed", "sealed-b"):
        from genetic_coil_arena.experiments import r16_sealed

        return r16_sealed.load_sealed(b=split == "sealed-b")
    return r16.load_items(DATA / "r16_dev.jsonl")


def run_items(translator: Any, items: list[dict], tag: str) -> list[dict]:
    rows = []
    for k, item in enumerate(items):
        res = translator.translate(item["text"], r16.pack())
        outcome = r16.score(res, item)
        rows.append({
            "id": item["id"], "text": item["text"], "expect": item["expect"],
            "gold": item.get("params"), "tags": item.get("tags", []),
            "status": res.status, "params": res.spec.params if res.spec else None,
            "reason": res.reason, "raw": res.raw, "attempts": res.attempts,
            "latency_s": res.latency_s, "outcome": outcome,
            "correct": r16.correct(outcome, item["expect"]),
        })  # fmt: skip
        if (k + 1) % 25 == 0:
            ok = sum(r["correct"] for r in rows)
            log(f"{tag}: {k + 1}/{len(items)}, {ok} correct")
    return rows


def run(args: argparse.Namespace) -> None:
    registry = r16.build_registry()
    items = items_for(args.split)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kind = args.translator.split(":", 1)[0]
    if kind == "keyword":
        tr = r16.GroundTranslator(r16.KeywordGroundBackend(), registry)
        rows = run_items(tr, items, args.translator)
    elif kind == "ollama":
        from r7a_vertical_slice import OllamaServer

        from genetic_coil_arena.backends.ollama import DEFAULT_HOST, OllamaBackend

        model = args.translator.split(":", 1)[1]
        with OllamaServer(DEFAULT_HOST) as server:
            server.require(model)
            server.unload_all()
            backend = OllamaBackend(model, keep_alive=KEEP, options=CTX)
            try:
                tr = r16.GroundTranslator(backend, registry, keep_alive=KEEP)
                tr.translate("Hover.", r16.pack())  # load the model before timing
                rows = run_items(tr, items, args.translator)
            finally:
                backend.unload()
    else:
        raise SystemExit(f"unknown translator {args.translator!r}")
    stem = f"r16_{args.split}_{args.translator.replace(':', '_').replace('/', '_')}"
    with open(out / f"{stem}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = summarise(rows)
    summary.update(translator=args.translator, split=args.split)
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(json.dumps(summary))


def summarise(rows: list[dict]) -> dict[str, Any]:
    from genetic_coil_arena.translator.metrics import wilson

    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["expect"]][r["outcome"]] += 1
    n_ok = sum(by["ok"].values())
    wrong = sum(c["wrong command"] for c in by.values())
    lat = sorted(r["latency_s"] for r in rows)
    return {
        "n": len(rows),
        "by_expect": {k: dict(v) for k, v in by.items()},
        "exact": wilson(by["ok"]["exact"], n_ok),
        "asked_when_unclear": wilson(
            by["clarify"]["asked"], sum(by["clarify"].values())
        ),
        "refused_when_should": wilson(
            by["refuse"]["refused"], sum(by["refuse"].values())
        ),
        "wrong_commands": wilson(wrong, len(rows)),
        "correct": wilson(sum(r["correct"] for r in rows), len(rows)),
        "median_latency_s": lat[len(lat) // 2] if lat else 0.0,
    }


def report(args: argparse.Namespace) -> None:
    from genetic_coil_arena.experiments import r16_report

    runs = []
    for src in resolve_sources(args.sources):
        for f in sorted(Path(src).glob("r16_*.json")):
            rows_path = f.with_suffix(".jsonl")
            rows = [json.loads(x) for x in rows_path.read_text(encoding="utf-8").splitlines()
                    if x.strip()]  # fmt: skip
            runs.append(
                {"summary": json.loads(f.read_text(encoding="utf-8")), "rows": rows}
            )
    md, data = r16_report.render(runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(data, indent=1), encoding="utf-8"
    )
    run_dir = os.environ.get("GCA_RUN_DIR")
    if run_dir:
        (Path(run_dir) / args.out.name).write_text(md, encoding="utf-8")
    print(md, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--translator", required=True, help="ollama:<tag> or keyword")
    r.add_argument("--split", choices=["dev", "sealed", "sealed-b"], default="dev")
    r.add_argument("--out", default=os.environ.get("GCA_RUN_DIR", "outputs/r16"))
    m = sub.add_parser("report")
    m.add_argument("--from", dest="sources", nargs="+", required=True)
    m.add_argument(
        "--out", type=Path, default=Path("docs/results/r16_ground_station.md")
    )
    args = p.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
