"""R26: compile R19's confirmer for MaixCAM (cv181x, BF16) with tpu-mlir and run it in its simulator.

Runs in WSL Ubuntu in R24's tpu-mlir venv (``/mnt/d/venvs/tpumlir``, see
``scripts/r24_maixcam_sim.py``). Standalone on purpose: it imports nothing from this
package. The ONNX and the sample's views come from ``scripts/r26_confirmer_npu.py``.

wsl -d Ubuntu -- /mnt/d/venvs/tpumlir/bin/python scripts/r26_confirmer_npu_sim.py compile
wsl -d Ubuntu -- /mnt/d/venvs/tpumlir/bin/python scripts/r26_confirmer_npu_sim.py simulate --split test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT_DIR = Path("/mnt/d/Projects/genetic-coil-arena-data/r26_confirmer_npu")
SIZE = 336
MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]  # DINOv2's
STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]
VARIANT = "bf16"
TOLERANCE = "0.99,0.9"  # model_deploy's own top-vs-BF16 check
CHUNK = 4


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], cwd: Path, logfile: Path) -> None:
    with logfile.open("w") as f:
        res = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if res.returncode:
        raise SystemExit(f"{cmd[0]} failed, see {logfile}")


def cvimodel() -> Path:
    return OUT_DIR / VARIANT / f"confirmer336_{VARIANT}.cvimodel"


def cmd_compile(args) -> None:
    work = OUT_DIR / VARIANT
    work.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    img = str(OUT_DIR / f"{args.test_input}_first.png")
    run(["model_transform.py", "--model_name", "confirmer336",
         "--model_def", str(OUT_DIR / "confirmer336.onnx"),
         "--input_shapes", f"[[1,3,{SIZE},{SIZE}]]", "--mean", ",".join(f"{m:.3f}" for m in MEAN),
         "--scale", ",".join(f"{1 / s:.6f}" for s in STD), "--pixel_format", "rgb",
         "--test_input", img, "--test_result", "top_outputs.npz", "--mlir", "model.mlir"],
        work, work / "transform.log")  # fmt: skip
    run(["model_deploy.py", "--mlir", "model.mlir", "--quantize", VARIANT.upper(),
         "--processor", "cv181x", "--fuse_preprocess", "--customization_format", "RGB_PLANAR",
         "--test_input", img, "--test_reference", "top_outputs.npz", "--tolerance", TOLERANCE,
         "--model", cvimodel().name], work, work / "deploy.log")  # fmt: skip
    info = subprocess.run(["model_tool", "--info", cvimodel().name], cwd=work,
                          capture_output=True, text=True).stdout  # fmt: skip
    (work / "model_info.txt").write_text(info, "utf-8")
    ion = re.search(r"ION Memory Size: \(([\d.]+) MB\)", info)
    sims = re.findall(r"min_similarity = \(([\d.]+), ([\d.]+), ([\d.inf]+)\)",
                      (work / "deploy.log").read_text("utf-8", "replace"))  # fmt: skip
    rec = {"variant": VARIANT, "size": SIZE, "tpu_mlir": "1.30.2", "processor": "cv181x",
           "quant": "BF16, fused uint8 RGB preprocessing",
           "cvimodel_bytes": cvimodel().stat().st_size,
           "ion_mb": float(ion.group(1)) if ion else None,
           "deploy_min_similarity": [list(map(float, s)) for s in sims],
           "seconds": time.perf_counter() - start}  # fmt: skip
    (OUT_DIR / f"{VARIANT}_compile.json").write_text(json.dumps(rec, indent=1), "utf-8")
    log(json.dumps(rec))


def _sim_chunk(task: tuple) -> tuple[int, list[float], float]:
    split, a, b = task
    devnull = os.open(os.devnull, os.O_WRONLY)  # the simulator prints a line per run
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        import tpu_mlir  # noqa: F401  (puts pyruntime_cvi on sys.path; must come first)

        # isort: split
        import pyruntime_cvi as rt

        m = rt.Model(str(cvimodel()), output_all_tensors=False)
        views = np.load(OUT_DIR / f"{split}_views.npy", mmap_mode="r")
        out, t0 = [], time.perf_counter()
        for i in range(a, b):
            m.inputs[0].data[:] = np.ascontiguousarray(views[i : i + 1].transpose(0, 3, 1, 2))
            m.forward()
            out.append(float(np.asarray(m.outputs[0].data, np.float32).reshape(-1)[0]))
        return a, out, time.perf_counter() - t0
    finally:
        os.dup2(saved, 1)


def cmd_simulate(args) -> None:
    path = OUT_DIR / f"{args.split}_{VARIANT}.npy"
    if args.split == "test" and path.exists():
        raise SystemExit(f"{path} exists: test-dev is simulated once")
    n = len(np.load(OUT_DIR / f"{args.split}_views.npy", mmap_mode="r"))
    tasks = [(args.split, a, min(a + CHUNK, n)) for a in range(0, n, CHUNK)]
    res, done, busy, start = np.full(n, np.nan, np.float32), 0, 0.0, time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for a, out, secs in pool.map(_sim_chunk, tasks):
            res[a : a + len(out)] = out
            done, busy = done + len(out), busy + secs
            if done % (CHUNK * 25) < CHUNK or done == n:
                log(f"[progress] {args.split}: {done}/{n} views, {busy / done:.0f} s per view "
                    f"per worker, {time.perf_counter() - start:.0f} s")  # fmt: skip
    assert not np.isnan(res).any()
    np.save(path, res)
    log(f"SIM_DONE {args.split}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--test-input", choices=("val", "test"), default="val",
                   help="the sample whose first view model_deploy checks")  # fmt: skip
    s = sub.add_parser("simulate")
    s.add_argument("--split", choices=("val", "test"), required=True)
    s.add_argument("--workers", type=int, default=int(os.environ.get("GCA_WORKERS", 20)))
    args = p.parse_args()
    {"compile": cmd_compile, "simulate": cmd_simulate}[args.cmd](args)


if __name__ == "__main__":
    main()
