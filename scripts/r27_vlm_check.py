"""R27: a ground-station vision-language model as the final check (experiments/r27_vlm_check.py).

python scripts/r27_vlm_check.py dev --model 8b      # both prompt variants on the dev spots; also 2b
python scripts/r27_vlm_check.py test --model 8b     # the variant chosen on dev, once; also 2b
python scripts/r27_vlm_check.py report

GPU (12 GB): the 8B runs in 4-bit. Scores go to ``D:/Projects/genetic-coil-arena-data/r27_vlm_check``;
``report`` writes ``docs/results/r27_vlm_check.{md,json}``.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\hf_cache")

import numpy as np  # noqa: E402

from genetic_coil_arena.experiments import r27_vlm_check as r27  # noqa: E402


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save_labels(split: str, data: dict) -> None:
    keys = {k: v for k, v in data.items() if k in ("label", "source", "frame", "chain")}
    np.savez(r27.OUT_DIR / f"{split}_labels.npz", **keys)


def load_labels(split: str) -> dict:
    return dict(np.load(r27.OUT_DIR / f"{split}_labels.npz"))


def chosen() -> str:
    dev = load_labels("dev")
    scores = {v: np.load(r27.score_path("8b", v, "dev")) for v in r27.PROMPTS}
    return r27.choose(scores, dev)[0]


def cmd_dev(args) -> None:
    r27.OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = r27.dev_set()
    save_labels("dev", dev)
    log(
        f"dev: {len(dev['label'])} spots, {dict(zip(*np.unique(dev['label'], return_counts=True)))}"
    )
    m = r27.VLMCheck(args.model)
    for v in r27.PROMPTS:
        s = m.score(dev["crops"], v, log=log)
        np.save(r27.score_path(args.model, v, "dev"), s)
        log(f"{args.model} {v} dev: {len(s)} scores")


def cmd_test(args) -> None:
    variant = chosen()
    out = r27.score_path(args.model, variant, "test")
    if out.exists():
        raise SystemExit(f"{out} exists: the test spots are scored once per model")
    test = r27.test_set(r27.dev_set())
    save_labels("test", test)
    log(f"test: {len(test['label'])} spots ({test['dropped']} dropped near dev spots), "
        f"variant {variant}")  # fmt: skip
    s = r27.VLMCheck(args.model).score(test["crops"], variant, log=log)
    np.save(out, s)
    (r27.OUT_DIR / "test_dropped.json").write_text(json.dumps({"dropped": test["dropped"]}))
    log(f"{args.model} {variant} test: {len(s)} scores")


def cmd_report(_args) -> None:
    variant = chosen()
    dev, test = load_labels("dev"), load_labels("test")
    rep: dict = {"variant": variant, "dev": {}, "test": {},
                 "test_dropped": json.loads((r27.OUT_DIR / "test_dropped.json").read_text())["dropped"]}  # fmt: skip
    for model in r27.MODELS:
        dev_s = {v: np.load(r27.score_path(model, v, "dev")) for v in r27.PROMPTS
                 if r27.score_path(model, v, "dev").exists()}  # fmt: skip
        rep["dev"][model] = {v: r27.evaluate(s, dev, r27.threshold_at(s, dev["label"]))
                             for v, s in dev_s.items()}  # fmt: skip
        tp = r27.score_path(model, variant, "test")
        if tp.exists() and variant in dev_s:
            thr = r27.threshold_at(dev_s[variant], dev["label"])
            rep["test"][model] = r27.evaluate(np.load(tp), test, thr)
    rep["verdict"] = r27.verdict(rep["test"]["8b"])
    out = Path("docs/results/r27_vlm_check.md")
    out.write_text(r27.render(rep), "utf-8")
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1), "utf-8")
    log(f"verdict: {rep['verdict']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("dev", "test"):
        c = sub.add_parser(name)
        c.add_argument("--model", choices=sorted(r27.MODELS), required=True)
    sub.add_parser("report")
    args = p.parse_args()
    {"dev": cmd_dev, "test": cmd_test, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
