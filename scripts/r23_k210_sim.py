"""R23: compile the scout brain for the K210 with nncase and run it in nncase's simulator.

Runs in its own venv (Python 3.10; nncase 1.9.0 is the last release with the K210
target): ``D:\\venvs\\nncase`` with numpy<2, onnx and nncase; on Windows nncase.dll also
needs ``libomp140.x86_64.dll`` beside it (Intel's libiomp5md.dll exports the same
ordinals). Standalone on purpose: it imports nothing from this package. Scoring is
``scripts/r23_k210.py`` in the main venv.

D:\\venvs\\nncase\\Scripts\\python.exe scripts/r23_k210_sim.py compile --model r15b_coco_a0.5_seed0
D:\\venvs\\nncase\\Scripts\\python.exe scripts/r23_k210_sim.py simulate --model r15b_coco_a0.5_seed0 --split test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATA_DIR = Path(r"D:\Projects\genetic-coil-arena-data")
OUT_DIR = DATA_DIR / "r23_k210"
MEAN, STD = [123.7, 116.3, 103.5], [58.4, 57.1, 57.4]  # r15b.input_fn
N_CALIB, CALIB_SEED = 500, 23
CHUNK = 256


def tiles(size: int, split: str) -> np.ndarray:
    return np.load(DATA_DIR / f"visdrone_tiles{size}" / f"{split}_images.npy", mmap_mode="r")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def onnx_path(model: str, onnx: str | None) -> Path:
    return Path(onnx) if onnx else DATA_DIR / "r15_models" / f"{model}.onnx"


def calib_pick(n: int, index: str | None) -> np.ndarray:
    """``N_CALIB`` random train tiles, or the tile indices saved at ``index`` (R23c)."""
    if index:
        return np.sort(np.load(index))
    return np.sort(np.random.default_rng(CALIB_SEED).choice(n, N_CALIB, replace=False))


def _compile(model: str, size: int, onnx_file: str | None = None,
             index: str | None = None) -> None:  # fmt: skip
    """In a child process, so the compiler's own prints (memory table) can be captured."""
    import nncase
    import onnx

    tr = tiles(size, "train")
    pick = calib_pick(len(tr), index)
    calib = np.ascontiguousarray(tr[pick].transpose(0, 3, 1, 2))
    co = nncase.CompileOptions()
    co.target, co.preprocess, co.swapRB = "k210", True, False
    co.input_shape, co.input_type, co.input_range = [1, 3, size, size], "uint8", [0, 255]
    co.mean, co.std = MEAN, STD
    co.input_layout = co.output_layout = "NCHW"
    co.quant_type = co.w_quant_type = "uint8"
    co.dump_ir = co.dump_asm = False
    co.dump_dir = str(OUT_DIR / "dump")
    c = nncase.Compiler(co)
    m = onnx.shape_inference.infer_shapes(onnx.load(onnx_path(model, onnx_file)))
    c.import_onnx(m.SerializeToString(), nncase.ImportOptions())
    ptq = nncase.PTQTensorOptions()
    ptq.samples_count = len(calib)
    ptq.set_tensor_data(calib.tobytes())
    c.use_ptq(ptq)
    c.compile()
    (OUT_DIR / f"{model}.kmodel").write_bytes(c.gencode_tobytes())


def cmd_compile(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cmd = [sys.executable, __file__, "_compile", "--model", args.model, "--size", str(args.size)]
    for flag in ("onnx", "calib_index"):
        if getattr(args, flag):
            cmd += [f"--{flag.replace('_', '-')}", getattr(args, flag)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    (OUT_DIR / f"{args.model}_compile.log").write_text(res.stdout + res.stderr, "utf-8")
    if res.returncode:
        raise SystemExit(f"compile failed:\n{res.stderr[-2000:]}")
    memory = {k: int(b) for k, b in re.findall(r"^(\S+)\s+[\d.]+ [KMG]?B\s+\((\d+) B\)",
                                                 res.stdout, re.M)}  # fmt: skip
    rec = {"model": args.model, "size": args.size, "nncase": "1.9.0.20230322",
           "target": "k210", "quant": "uint8 weights and activations",
           "calibration": {"tiles": N_CALIB, "seed": CALIB_SEED, "split": "train",
                           "index": args.calib_index}, "onnx": args.onnx,
           "kmodel_bytes": (OUT_DIR / f"{args.model}.kmodel").stat().st_size,
           "memory": memory, "seconds": time.perf_counter() - start}  # fmt: skip
    (OUT_DIR / f"{args.model}_compile.json").write_text(json.dumps(rec, indent=1), "utf-8")
    log(json.dumps(rec))


def _sim_chunk(task: tuple) -> int:
    import nncase

    model, size, split, a, b, path, shape = task
    sim = nncase.Simulator()
    sim.load_model((OUT_DIR / f"{model}.kmodel").read_bytes())
    src = tiles(size, split)
    out = np.lib.format.open_memmap(path, mode="r+")
    for i in range(a, b):
        x = np.ascontiguousarray(src[i : i + 1].transpose(0, 3, 1, 2))
        sim.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(x))
        sim.run()
        # several outputs (R23c: heatmap, then size and offsets) join along channels
        out[i] = np.concatenate([sim.get_output_tensor(k).to_numpy()[0]
                                 for k in range(sim.outputs_size)], axis=0)  # fmt: skip
    out.flush()
    return b - a


def cmd_simulate(args) -> None:
    path = OUT_DIR / f"{args.model}_{args.split}.npy"
    if args.split == "test" and path.exists():
        raise SystemExit(f"{path} exists: test-dev is simulated once")
    src = tiles(args.size, args.split)
    g = args.size // 8
    shape = (len(src), 5, g, g)
    np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape).flush()
    tasks = [(args.model, args.size, args.split, a, min(a + CHUNK, len(src)), str(path), shape)
             for a in range(0, len(src), CHUNK)]  # fmt: skip
    done, start = 0, time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for n in pool.map(_sim_chunk, tasks):
            done += n
            if done % (CHUNK * 40) < CHUNK or done == len(src):
                log(f"[progress] {args.model} {args.split}: {done}/{len(src)} tiles, "
                    f"{time.perf_counter() - start:.0f} s")  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("compile", "_compile", "simulate"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--size", type=int, default=224)
        if name != "simulate":
            s.add_argument("--onnx", help="the ONNX file (default: r15_models/<model>.onnx)")
            s.add_argument("--calib-index", help=".npy of train tile indices to calibrate on")
        if name == "simulate":
            s.add_argument("--split", choices=("val", "test"), required=True)
            s.add_argument("--workers", type=int, default=int(os.environ.get("GCA_WORKERS", 20)))
    args = p.parse_args()
    if args.cmd == "_compile":
        _compile(args.model, args.size, args.onnx, args.calib_index)
    else:
        {"compile": cmd_compile, "simulate": cmd_simulate}[args.cmd](args)


if __name__ == "__main__":
    main()
