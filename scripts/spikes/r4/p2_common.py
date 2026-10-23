"""Shared helpers for Part 2 (run with venv_llm python)."""
import os, json, time, statistics, subprocess
from pathlib import Path

HERE = Path(__file__).parent
os.environ["HF_HOME"] = str(HERE / "hf")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUTF8"] = "1"

import torch  # noqa: E402

torch.set_num_threads(2)
MODEL = "Qwen/Qwen3-1.7B"
DEV = "cuda"


def driver_used_mib():
    """Total GPU memory in use at driver level (all processes), MiB."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out.splitlines()[0])


def mem():
    torch.cuda.synchronize()
    return {"allocated_MiB": torch.cuda.memory_allocated() / 2**20,
            "reserved_MiB": torch.cuda.memory_reserved() / 2**20,
            "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
            "driver_used_MiB": driver_used_mib()}


def timeit(fn, warm=5, iters=20):
    """Median/mean/min/max of fn() in ms with cuda sync."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return {"median_ms": statistics.median(ts), "mean_ms": statistics.mean(ts),
            "min_ms": min(ts), "max_ms": max(ts), "n": iters}


def load_base(dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    t = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map=DEV)
    torch.cuda.synchronize()
    return tok, model, time.perf_counter() - t


def make_ids(tok, L, bs=1):
    text = ("The squad of three snakes red green and blue cooperates to eat food on the board. " * 80)
    ids = tok(text, return_tensors="pt").input_ids[:, :L]
    assert ids.shape[1] == L, ids.shape
    return ids.repeat(bs, 1).to(DEV)


def kv_tensors(cache):
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(k, v) for k, v in cache]


def kv_bytes(cache):
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kv_tensors(cache))


def save(name, obj):
    p = HERE / "results" / name
    p.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    print("saved", p)
