"""R24: compile the 448-px scout brain for MaixCAM (cv181x) with tpu-mlir and run it in its simulator.

Runs in WSL Ubuntu in its own venv (Python 3.10, tpu-mlir 1.30.2): ``/mnt/d/venvs/tpumlir``.
tpu-mlir's bundled glibc 2.35 (``libc.so.6``, ``libm.so.6``, ``libpthread.so.0`` in
``tpu_mlir/lib/third_party``) clashes with Ubuntu 24.04's and is moved aside; it also
needs onnx, onnxruntime, onnxsim, psutil, loguru, pydantic, fire, astunparse, pydot,
openpyxl and CPU torch, which its wheel does not declare. Standalone on purpose: it
imports nothing from this package. Scoring is ``scripts/r24_maixcam.py`` in the main venv.

wsl -d Ubuntu -- /mnt/d/venvs/tpumlir/bin/python scripts/r24_maixcam_sim.py calib
wsl -d Ubuntu -- /mnt/d/venvs/tpumlir/bin/python scripts/r24_maixcam_sim.py compile --model r15b_coco_a0.5_seed0_s448
wsl -d Ubuntu -- /mnt/d/venvs/tpumlir/bin/python scripts/r24_maixcam_sim.py simulate --model r15b_coco_a0.5_seed0_s448 --split val
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

DATA_DIR = Path("/mnt/d/Projects/genetic-coil-arena-data")
OUT_DIR = DATA_DIR / "r24_maixcam"
SIZE = 448
MEAN, STD = [123.7, 116.3, 103.5], [58.4, 57.1, 57.4]  # r15b.input_fn
N_CALIB, CALIB_SEED = 500, 23  # R23's rule, on the 448 train tiles
TOLERANCE = "0.85,0.45"  # model_deploy's own top-vs-int8 sanity check
CHUNK = 64


def tiles(split: str) -> np.ndarray:
    return np.load(DATA_DIR / f"visdrone_tiles{SIZE}" / f"{split}_images.npy", mmap_mode="r")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def calib_dir() -> Path:
    return OUT_DIR / "calib"


def cmd_calib(args) -> None:
    """``N_CALIB`` random train tiles (seed ``CALIB_SEED``) as lossless PNGs."""
    from PIL import Image

    tr = tiles("train")
    pick = np.sort(np.random.default_rng(CALIB_SEED).choice(len(tr), N_CALIB, replace=False))
    calib_dir().mkdir(parents=True, exist_ok=True)
    for i in pick:
        Image.fromarray(np.asarray(tr[i])).save(calib_dir() / f"t{i:06d}.png")
    (OUT_DIR / "calib_index.json").write_text(json.dumps([int(i) for i in pick]), "utf-8")
    log(f"{len(pick)} calibration tiles")


def run(cmd: list[str], cwd: Path, logfile: Path) -> None:
    with open(logfile, "w", encoding="utf-8") as f:
        res = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if res.returncode:
        raise SystemExit(f"{cmd[0]} failed, see {logfile}")


def cmd_compile(args) -> None:
    work = OUT_DIR / args.model
    work.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    img = str(sorted(calib_dir().glob("*.png"))[0])
    scale = ",".join(f"{1 / s:.6f}" for s in STD)
    run(["model_transform.py", "--model_name", args.model, "--model_def", args.onnx,
         "--input_shapes", f"[[1,3,{SIZE},{SIZE}]]", "--mean", ",".join(map(str, MEAN)),
         "--scale", scale, "--pixel_format", "rgb", "--test_input", img,
         "--test_result", "top_outputs.npz", "--mlir", "model.mlir"], work, work / "transform.log")  # fmt: skip
    run(["run_calibration.py", "model.mlir", "--dataset", str(calib_dir()),
         "--input_num", str(N_CALIB), "-o", "cali_table"], work, work / "calib.log")  # fmt: skip
    cvimodel = f"{args.model}.cvimodel"
    run(["model_deploy.py", "--mlir", "model.mlir", "--quantize", "INT8",
         "--calibration_table", "cali_table", "--processor", "cv181x", "--fuse_preprocess",
         "--customization_format", "RGB_PLANAR", "--test_input", img,
         "--test_reference", "top_outputs.npz", "--tolerance", TOLERANCE,
         "--model", cvimodel], work, work / "deploy.log")  # fmt: skip
    info = subprocess.run(["model_tool", "--info", cvimodel], cwd=work, capture_output=True,
                          text=True).stdout  # fmt: skip
    (work / "model_info.txt").write_text(info, "utf-8")
    ion = re.search(r"ION Memory Size: \(([\d.]+) MB\)", info)
    shared = re.search(r"shared_gmem_size: (\d+)", info)
    sims = re.findall(r"min_similarity = \(([\d.]+), ([\d.]+), ([\d.inf]+)\)",
                      (work / "deploy.log").read_text("utf-8", "replace"))  # fmt: skip
    rec = {"model": args.model, "size": SIZE, "tpu_mlir": "1.30.2", "processor": "cv181x",
           "quant": "INT8 symmetric, fused uint8 RGB preprocessing",
           "calibration": {"tiles": N_CALIB, "seed": CALIB_SEED, "split": "train"},
           "onnx": args.onnx, "cvimodel_bytes": (work / cvimodel).stat().st_size,
           "ion_mb": float(ion.group(1)) if ion else None,
           "shared_gmem_bytes": int(shared.group(1)) if shared else None,
           "deploy_min_similarity": [list(map(float, s)) for s in sims],
           "seconds": time.perf_counter() - start}  # fmt: skip
    (OUT_DIR / f"{args.model}_compile.json").write_text(json.dumps(rec, indent=1), "utf-8")
    log(json.dumps(rec))


def _sim_chunk(task: tuple) -> int:
    model, split, a, b, path = task
    devnull = os.open(os.devnull, os.O_WRONLY)  # the simulator prints a line per run
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        import tpu_mlir  # noqa: F401  (puts pyruntime_cvi on sys.path; must come first)

        # isort: split
        import pyruntime_cvi as rt

        m = rt.Model(str(OUT_DIR / model / f"{model}.cvimodel"), output_all_tensors=False)
        names = [o.name for o in m.outputs]
        assert names[0].startswith("heat"), names
        src = tiles(split)
        out = np.lib.format.open_memmap(path, mode="r+")
        for i in range(a, b):
            m.inputs[0].data[:] = np.ascontiguousarray(src[i : i + 1].transpose(0, 3, 1, 2))
            m.forward()
            out[i] = np.concatenate([np.asarray(o.data, np.float32)[0] for o in m.outputs], 0)
        out.flush()
    finally:
        os.dup2(saved, 1)
    return b - a


def cmd_simulate(args) -> None:
    path = OUT_DIR / f"{args.model}_{args.split}.npy"
    if args.split == "test" and path.exists():
        raise SystemExit(f"{path} exists: test-dev is simulated once")
    n, g = len(tiles(args.split)), SIZE // 8
    if args.split == "train":  # speed and memory probe only
        n, path = min(n, args.limit), OUT_DIR / f"{args.model}_train_probe.npy"
    np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, 5, g, g)).flush()
    tasks = [(args.model, args.split, a, min(a + CHUNK, n), str(path)) for a in range(0, n, CHUNK)]
    done, start = 0, time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for k in pool.map(_sim_chunk, tasks):
            done += k
            if done % (CHUNK * 40) < CHUNK or done == n:
                log(f"[progress] {args.model} {args.split}: {done}/{n} tiles, "
                    f"{time.perf_counter() - start:.0f} s")  # fmt: skip
    log(f"SIM_DONE {args.model} {args.split}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calib")
    c = sub.add_parser("compile")
    c.add_argument("--model", required=True)
    c.add_argument("--onnx", required=True, help="the split-head ONNX (R23c's export)")
    s = sub.add_parser("simulate")
    s.add_argument("--model", required=True)
    s.add_argument("--split", choices=("val", "test", "train"), required=True)
    s.add_argument("--workers", type=int, default=int(os.environ.get("GCA_WORKERS", 16)))
    s.add_argument("--limit", type=int, default=256, help="train probe: tiles to run")
    args = p.parse_args()
    {"calib": cmd_calib, "compile": cmd_compile, "simulate": cmd_simulate}[args.cmd](args)


if __name__ == "__main__":
    main()
