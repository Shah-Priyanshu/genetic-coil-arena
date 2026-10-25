"""R7a step 2: an honest G3 translator, and a distilled small model that fits beside the specialists.

Sub-commands (each is one runner job; see ``docs/gpu_runner.md``):

    para-gen      qwen3-coder:30b writes paraphrases of gold specs in style cards -> para_raw.jsonl
    para-verify   qwen2.5-coder:14b re-translates each paraphrase; keeps those that recover the
                  gold outcome; builds the training pool (templates + paraphrases + retries),
                  with the n-gram leak filter against the sealed/dev/step-1 sets -> pool.jsonl
    train         LoRA on Qwen3-1.7B (rank, data size), then dev + step-1 score for selection
    eval-lora     best run by dev score, evaluated once on the sealed set (+ VRAM, latency)
    eval-ollama   14B with the refusal policy, gemma4:e4b and the regex baseline on the sealed
                  set, the 14B writes summaries of every translator's traces, round trip of the
                  three non-distilled translators
    eval-lora-rt  round trip of the distilled model (the HF model runs apart from Ollama)
    aggregate     docs/results/r7a_step2.{md,json} with a G3 verdict per translator

Nothing here reads results on the sealed set before ``eval-*``; prompts, training data and
checkpoint selection use the step-1 set and the dev set only. Ollama servers started by
a job are stopped by it, and models are unloaded (keep_alive 0) when it finishes.
Adapters go to ``$R7A2_STORE`` (default ``$GCA_RUN_DIR``), not into git.

    python scripts/r7a_step2.py train --pool runs:r7a2-para-verify --rank 8 --n 1000
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r7a_vertical_slice import (  # noqa: E402
    OllamaServer,
    VramMonitor,
    clean,
    log,
)

from genetic_coil_arena.backends.ollama import DEFAULT_HOST, OllamaBackend  # noqa: E402
from genetic_coil_arena.envs.plant.tep import TepData  # noqa: E402
from genetic_coil_arena.experiments import r7a_train_data as td  # noqa: E402
from genetic_coil_arena.experiments.r7a_domains import (  # noqa: E402
    build_domains,
    build_registry,
)
from genetic_coil_arena.experiments.r7a_pipeline import keyword_translator  # noqa: E402
from genetic_coil_arena.experiments.r7a_policy import with_policy_packs  # noqa: E402
from genetic_coil_arena.experiments.r7a_report import (  # noqa: E402
    evaluate_forward,
    forward_metrics,
    reverse_metrics,
    select_items,
)
from genetic_coil_arena.experiments.r7a_sealed import (  # noqa: E402
    SEALED_SHA256,
    load_dev,
    load_old_set,
    load_sealed,
)
from genetic_coil_arena.experiments.r7a_step2 import (  # noqa: E402
    find_file,
    reconstruct_rows,
    rehydrate,
    resolve_runs,
    select_best_run,
    strip_views,
    with_lora_packs,
    write_summaries,
)
from genetic_coil_arena.translator import (  # noqa: E402
    PolicyTranslator,
    ReverseTranslator,
    TranslationResult,
    format_prompt,
    percentile,
)

WRITER = "qwen2.5-coder:14b"  # verifies paraphrases, translates, writes summaries
GENERATOR = "qwen3-coder:30b"
SMALL = "gemma4:e4b"
BASE = "Qwen/Qwen3-1.7B"
# The policy prompts (system text, about ten few-shots, a retry) can pass 2048 tokens; Ollama
# would silently cut the start of the prompt, so ask for more context.
CTX = {"num_ctx": 4096}
ALL_LINEAR = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def run_dir(arg: Optional[str]) -> Path:
    """Where this job writes its (small) artifacts."""
    path = Path(arg or os.environ.get("GCA_RUN_DIR") or "outputs/r7a_step2")
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_dir() -> Path:
    """Where big files (adapters) go: ``$R7A2_STORE``, else the run folder (never git)."""
    path = Path(
        os.environ.get("R7A2_STORE")
        or os.environ.get("GCA_RUN_DIR")
        or "outputs/r7a_step2"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(clean(obj), indent=1, allow_nan=False), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def appender(path: Path):
    """A function that appends one JSON row to ``path`` (flushed, so a restart can resume)."""
    handle = path.open("a", encoding="utf-8")

    def write(row: dict[str, Any]) -> None:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()

    return write


def held_out_texts() -> list[str]:
    """Texts the training data must stay away from: sealed, dev and step-1 sets."""
    return [i.text for i in (*load_sealed(), *load_dev(), *load_old_set())]


def make_domains(args: argparse.Namespace, execute: bool) -> tuple[Any, dict[str, Any]]:
    registry = build_registry()
    plant = TepData.from_cache(None) if execute else None
    domains = build_domains(plant, args.squad_seeds, args.plant_seeds)
    return registry, domains


def latency_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    warm = [r["latency_s"] for r in records[1:]]
    calls = [c for r in records[1:] for c in r["call_latencies"]]
    first = (
        records[0]["call_latencies"][0]
        if records and records[0]["call_latencies"]
        else None
    )
    return {
        "cold_first_call_s": first,
        "warm_item_p50_s": percentile(warm, 50),
        "warm_item_p95_s": percentile(warm, 95),
        "warm_call_p50_s": percentile(calls, 50),
        "warm_call_p95_s": percentile(calls, 95),
        "max_prompt_tokens": max((r["n_prompt_tokens"] for r in records), default=0),
        "mean_attempts": sum(r["attempts"] for r in records) / max(1, len(records)),
    }


def progress_logger(tag: str, every: int = 20):
    def progress(k: int, n: int, ident: str) -> None:
        if k % every == 0 or k == n:
            log(f"{tag}: {k}/{n} ({ident})")

    return progress


# ---------------------------------------------------------------------- para-gen


class FakeGenerator:
    """Scripted stand-in for the paraphrase model (``--fake``): reorders the words."""

    def generate(self, prompts: list[str], **kw: Any) -> list[Any]:
        from types import SimpleNamespace

        out = []
        for prompt in prompts:
            flavour = [x for x in prompt.splitlines() if x.startswith("Meaning")]
            texts = [f"fake paraphrase {i} of {len(flavour)}" for i in range(3)]
            out.append(
                SimpleNamespace(text=json.dumps({"texts": texts}), latency_s=0.0)
            )
        return out


def pick_bases(n_per_domain: int, seed: int) -> list[td.Record]:
    """Gold meanings to paraphrase: spec, refusal and clarify records (80/10/10)."""
    pool = td.make_templates(max(400, n_per_domain * 2), seed=seed)
    refs = held_out_texts()
    pool, _ = td.filter_overlap(pool, refs)
    rng = random.Random(seed)
    out = []
    for domain in ("squad", "plant"):
        for kind, share in (("spec", 0.8), ("refuse", 0.1), ("clarify", 0.1)):
            cands = [
                r for r in pool
                if r.domain == domain and r.kind == kind and "report" not in r.style
            ]  # fmt: skip
            rng.shuffle(cands)
            out += cands[: round(n_per_domain * share)]
    rng.shuffle(out)
    return out


def cmd_para_gen(args: argparse.Namespace) -> None:
    out = run_dir(args.out)
    bases = pick_bases(args.bases // 2, args.seed)
    (out / "bases.jsonl").write_text(
        "".join(b.to_json() + "\n" for b in bases), encoding="utf-8"
    )
    raw_path = out / "para_raw.jsonl"
    done = {r["base_id"] for r in read_jsonl(raw_path)}
    log(f"{len(bases)} bases, {len(done)} already done")
    styles = list(td.STYLE_CARDS)

    def run(backend: Any) -> None:
        td.generate_paraphrases(
            backend, bases, styles, args.per_call, args.seed, sink=appender(raw_path),
            done_ids=done, progress=lambda k, n: log(f"para-gen {k}/{n}") if k % 10 == 0 or k == n else None,
        )  # fmt: skip

    if args.fake:
        run(FakeGenerator())
    else:
        with OllamaServer(args.host) as server:
            server.require(args.model)
            server.unload_all()
            backend = OllamaBackend(
                args.model, host=args.host, keep_alive="30m", timeout=1200,
                options={"num_ctx": 4096},
            )  # fmt: skip
            run(backend)
            backend.unload()
    n_rows = sum(1 for r in read_jsonl(raw_path) if r["text"])
    log(f"wrote {n_rows} raw paraphrases to {raw_path}")


# ---------------------------------------------------------------------- para-verify


def cmd_para_verify(args: argparse.Namespace) -> None:
    out = run_dir(args.out)
    sources = resolve_runs([args.raw])
    bases = [
        td.Record.from_json(x)
        for x in find_file(sources, "bases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x
    ]
    base_by_id = {b.id: b for b in bases}
    raw = [r for r in read_jsonl(find_file(sources, "para_raw.jsonl")) if r["text"]]
    registry, domains = make_domains(args, execute=False)
    domains = with_policy_packs(domains)
    packs = {name: d.pack for name, d in domains.items()}
    ver_path = out / "para_verified.jsonl"
    done = {(r["base_id"], r["text"]): r for r in read_jsonl(ver_path)}
    todo = [r for r in raw if (r["base_id"], r["text"]) not in done]
    log(
        f"{len(raw)} raw paraphrases, {len(done)} verified before, {len(todo)} to verify"
    )
    if todo and not args.fake:
        with OllamaServer(args.host) as server:
            server.require(args.model)
            server.unload_all()
            backend = OllamaBackend(
                args.model, host=args.host, keep_alive="30m", timeout=900, options=CTX
            )
            translator = PolicyTranslator(backend, registry)
            td.verify_paraphrases(
                translator, packs, base_by_id, todo, sink=appender(ver_path),
                progress=lambda k, n: log(f"para-verify {k}/{n}") if k % 50 == 0 or k == n else None,
            )  # fmt: skip
            backend.unload()
    elif todo:  # fake: an oracle that accepts every other paraphrase
        sink = appender(ver_path)
        for i, row in enumerate(todo):
            sink({**row, "verified": i % 2 == 0, "verifier_status": "fake"})
    verified = read_jsonl(ver_path)
    build_pool(args, out, registry, base_by_id, verified)


def build_pool(
    args: argparse.Namespace,
    out: Path,
    registry: Any,
    base_by_id: dict[str, td.Record],
    verified: list[dict[str, Any]],
) -> None:
    refs = held_out_texts()
    templates = td.make_templates(args.templates, seed=args.seed + 7)
    paras = td.paraphrase_records(base_by_id, verified)
    kept_t, dropped_t = td.filter_overlap(templates, refs)
    kept_p, dropped_p = td.filter_overlap(paras, refs)
    retries = td.retry_examples(
        registry, kept_t + kept_p, seed=args.seed, n=args.retries
    )
    pool = kept_t + kept_p + retries
    td.assert_no_leak(pool, refs)
    (out / "pool.jsonl").write_text(
        "".join(r.to_json() + "\n" for r in pool), encoding="utf-8"
    )
    by_style: dict[str, list[int]] = {}
    for row in verified:
        by_style.setdefault(row["style"], [0, 0])
        by_style[row["style"]][0] += 1
        by_style[row["style"]][1] += int(bool(row.get("verified")))
    by_kind: dict[str, list[int]] = {}
    for row in verified:
        kind = base_by_id[row["base_id"]].kind
        by_kind.setdefault(kind, [0, 0])
        by_kind[kind][0] += 1
        by_kind[kind][1] += int(bool(row.get("verified")))
    stats = {
        "sealed_sha256": SEALED_SHA256,
        "n_pool": len(pool),
        "templates": len(kept_t),
        "paraphrases": len(kept_p),
        "retries": len(retries),
        "raw_paraphrases": len(verified),
        "verified_paraphrases": sum(bool(r.get("verified")) for r in verified),
        "accept_by_style": {
            k: {"n": v[0], "accepted": v[1], "rate": v[1] / v[0]}
            for k, v in sorted(by_style.items())
        },
        "accept_by_kind": {
            k: {"n": v[0], "accepted": v[1], "rate": v[1] / v[0]}
            for k, v in sorted(by_kind.items())
        },
        "dropped_by_overlap": {
            "templates": len(dropped_t),
            "paraphrases": len(dropped_p),
        },
        "overlap_threshold": 0.5,
        "kinds": {
            k: sum(r.kind == k for r in pool)
            for k in ("spec", "refuse", "clarify", "retry")
        },
        "sources": {
            k: sum(r.source == k for r in pool)
            for k in ("template", "paraphrase", "retry")
        },
    }
    write_json(out / "pool_stats.json", stats)
    log(
        f"pool: {stats['n_pool']} records ({stats['sources']}), overlap drops {stats['dropped_by_overlap']}"
    )


# ---------------------------------------------------------------------- hf helpers


def open_hf(args: argparse.Namespace) -> Any:
    from genetic_coil_arena.backends.hf_peft import HFPeftBackend

    be = HFPeftBackend(
        args.base, memory_fraction=0.85, num_threads=4,
        local_files_only=not args.online, dtype="bfloat16",
    )  # fmt: skip
    be.load()
    return be


def eval_language(
    be: Any,
    adapter: str,
    items: list[Any],
    domains: dict[str, Any],
    registry: Any,
    tag: str,
) -> list[dict[str, Any]]:
    from genetic_coil_arena.backends import AdapterTextBackend

    translator = PolicyTranslator(AdapterTextBackend(be, adapter), registry)
    doms = with_lora_packs(domains)
    return evaluate_forward(
        lambda it: translator.translate(it.text, doms[it.domain].pack),
        items, doms, registry, progress_logger(tag, 40), execute_tasks=False,
    )  # fmt: skip


# ---------------------------------------------------------------------- train


def cmd_train(args: argparse.Namespace) -> None:
    out = run_dir(args.out)
    name = args.name or f"r{args.rank}-n{args.n}"
    if name in "lora_":  # PEFT warns that such a name can re-initialise the weights
        raise SystemExit(
            f"adapter name {name!r} is a substring of 'lora_'; pick another"
        )
    sources = resolve_runs([args.pool])
    pool = [
        td.Record.from_json(x)
        for x in find_file(sources, "pool.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if x
    ]
    refs = held_out_texts()
    td.assert_no_leak(pool, refs)  # never train on anything that overlaps held-out text
    chosen = td.select_training(pool, args.n, args.seed, args.paraphrase_share)
    log(f"{name}: {len(chosen)} training records "
        f"({sum(r.source == 'paraphrase' for r in chosen)} paraphrases, "
        f"{sum(r.kind == 'retry' for r in chosen)} retries)")  # fmt: skip
    registry, domains = make_domains(args, execute=False)

    be = open_hf(args)
    torch = be.torch
    be.add_adapter(
        name, r=args.rank, alpha=2 * args.rank, dropout=0.05,
        target_modules=ALL_LINEAR, seed=args.seed,
    )  # fmt: skip
    steps = max(1, args.epochs * len(chosen) // args.batch)
    order: list[list[td.Record]] = []
    rng = random.Random(args.seed)
    while len(order) < steps:
        idx = list(range(len(chosen)))
        rng.shuffle(idx)
        for lo in range(0, len(idx) - args.batch + 1, args.batch):
            order.append([chosen[i] for i in idx[lo : lo + args.batch]])
    spec_of = td.SPEC_NAMES

    def step(i: int) -> Any:
        batch = order[i]
        inputs = be.build_inputs(
            [format_prompt(spec_of[r.domain], r.text) for r in batch],
            [r.reply for r in batch],
        )
        return be.training_loss(inputs)

    # All-linear fp32 LoRA stores fp32 copies of every linear's input; checkpointing keeps a
    # batch of 16 well inside the 12 GB card (without it the run needs more than 10 GB).
    be.set_gradient_checkpointing(not args.no_checkpoint)
    start = time.time()
    stats = be.fit(
        name, step, steps=steps, lr=args.lr, warmup=min(20, steps // 5),
        log_every=25, logger=log, label=f"[{name}]",
    )  # fmt: skip
    train_s = time.time() - start
    be.set_gradient_checkpointing(False)
    log(
        f"{name}: trained {steps} steps in {train_s:.0f}s, loss {stats.mean_last_loss:.4f}, peak {stats.peak_memory_mib:.0f} MiB"
    )

    items = load_dev() + load_old_set()
    if args.limit:
        items = select_items(items, args.limit, args.seed)
    records = eval_language(be, name, items, domains, registry, f"{name} dev")
    fwd = forward_metrics(records)
    ex = fwd["exact"]
    score = ex["all"]["exact"]
    adapter_path = store_dir() / f"{name}.pt"
    be.save_adapter(name, adapter_path)
    result = {
        "name": name,
        "config": {
            "rank": args.rank, "n": len(chosen), "n_requested": args.n, "epochs": args.epochs,
            "batch": args.batch, "lr": args.lr, "steps": steps, "seed": args.seed,
            "paraphrase_share": args.paraphrase_share, "base": args.base,
            "target_modules": list(ALL_LINEAR),
            "n_paraphrases": sum(r.source == "paraphrase" for r in chosen),
            "n_retries": sum(r.kind == "retry" for r in chosen),
        },  # fmt: skip
        "train": {
            "final_loss": stats.final_loss, "mean_last_loss": stats.mean_last_loss,
            "wall_s": train_s, "peak_mib": stats.peak_memory_mib, "n_trainable": stats.n_trainable,
            "loss_curve": stats.losses[:: max(1, len(stats.losses) // 100)],
        },  # fmt: skip
        "dev": {
            "score": score,
            "exact_dev": ex.get("dev", {}).get("exact"),
            "exact_step1_set": {k: v.get("exact") for k, v in ex.items() if k in ("indist", "heldout")},
            "n_valid": ex["all"]["n"],
            "refusal": fwd["refusal"],
            "clarify": fwd["clarify"],
            "status_counts": fwd["status_counts"],
            "latency": latency_block(records),
        },  # fmt: skip
        "adapter_path": str(adapter_path),
        "adapter_mib": adapter_path.stat().st_size / 2**20,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(out / "dev.json", result)
    log(
        f"{name}: dev+step-1 exact {score:.1%} on {ex['all']['n']} valid items; adapter {result['adapter_mib']:.0f} MiB at {adapter_path}"
    )
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ---------------------------------------------------------------------- eval-lora


def load_adapter_backend(
    args: argparse.Namespace, best: dict[str, Any]
) -> tuple[Any, str]:
    be = open_hf(args)
    name = best["name"]
    path = Path(best["adapter_path"])
    if not path.exists():  # moved store: fall back to the configured one
        path = store_dir() / path.name
    be.load_adapter(name, path)
    be.set_adapter(name)
    return be, name


def cmd_eval_lora(args: argparse.Namespace) -> None:
    out = run_dir(args.out)
    sources = resolve_runs(args.runs)
    dev_files = [s / "dev.json" for s in sources if (s / "dev.json").exists()]
    best = select_best_run(dev_files)
    log(
        f"best run by dev score: {best['name']} ({best['dev']['score']:.1%}); considered {len(dev_files)}"
    )
    write_json(out / "best.json", best)
    items = load_sealed()
    if args.limit:
        items = select_items(items, args.limit, 0)
    registry, domains = make_domains(args, execute=True)
    vram = VramMonitor()
    base_mib = vram.read()
    be, name = load_adapter_backend(args, best)
    torch = be.torch
    weights_mib = (
        torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    )
    be.reset_peak_memory()
    vram.start()
    from genetic_coil_arena.backends import AdapterTextBackend

    translator = PolicyTranslator(AdapterTextBackend(be, name), registry)
    doms = with_lora_packs(domains)
    cache: dict[str, Any] = {}
    records = evaluate_forward(
        lambda it: translator.translate(it.text, doms[it.domain].pack),
        items, doms, registry, progress_logger("lora sealed", 20), cache=cache,
    )  # fmt: skip
    peak_smi = vram.stop()
    result = {
        "name": "lora",
        "label": f"Qwen3-1.7B + LoRA ({best['name']})",
        "model": BASE,
        "config": best["config"],
        "dev": best["dev"],
        "records": strip_views(records),
        "forward": forward_metrics(records),
        "latency": latency_block(records),
        "vram": {
            "weights_mib": weights_mib, "peak_allocated_mib": be.peak_memory_mib(),
            "baseline_smi_mib": base_mib, "peak_smi_mib": peak_smi if vram.available else None,
            "model_smi_mib": (peak_smi - base_mib) if (vram.available and base_mib is not None) else None,
        },  # fmt: skip
        "sealed_sha256": SEALED_SHA256,
        "smoke": bool(args.limit),
    }
    write_json(out / "lora_forward.json", result)
    ex = result["forward"]["exact"]["sealed"]
    log(f"lora: sealed exact {ex['exact']:.1%} (n={ex['n']}), p95 {result['latency']['warm_item_p95_s']:.2f}s, "
        f"peak allocated {result['vram']['peak_allocated_mib']:.0f} MiB")  # fmt: skip


# ---------------------------------------------------------------------- eval-ollama


class _ItemShim:
    def __init__(self, domain: str, text: str):
        self.domain, self.text = domain, text


def cmd_eval_ollama(args: argparse.Namespace) -> None:
    out = run_dir(args.out)
    items = load_sealed()
    if args.limit:
        items = select_items(items, args.limit, 0)
    registry, base_domains = make_domains(args, execute=True)
    domains = with_policy_packs(base_domains)
    cache: dict[str, Any] = {}
    systems: dict[str, dict[str, Any]] = {}
    records_by_sys: dict[str, list[dict[str, Any]]] = {}
    vram = VramMonitor()

    def forward_llm(server: OllamaServer, key: str, model: str) -> Any:
        server.require(model)
        server.unload_all()
        time.sleep(1.0)
        base = vram.read()
        vram.start()
        backend = OllamaBackend(model, host=args.host, keep_alive="30m", options=CTX)
        translator = PolicyTranslator(backend, registry)
        recs = evaluate_forward(
            lambda it: translator.translate(it.text, domains[it.domain].pack),
            items, domains, registry, progress_logger(f"{model} sealed", 20), cache=cache,
        )  # fmt: skip
        peak = vram.stop()
        records_by_sys[key] = recs
        systems[key] = {
            "name": key,
            "label": model + " (refusal policy)",
            "model": model,
            "records": strip_views(recs),
            "forward": forward_metrics(recs),
            "latency": latency_block(recs),
            "vram": {
                "baseline_smi_mib": base, "peak_smi_mib": peak if vram.available else None,
                "model_smi_mib": (peak - base) if (vram.available and base is not None) else None,
            },  # fmt: skip
            "smoke": bool(args.limit),
        }
        return translator

    with OllamaServer(args.host) as server:
        forward_llm(server, "gemma", args.small)
        w_translator = forward_llm(server, "o14", args.writer)
        # regex baseline (no model)
        kw = keyword_translator(registry, base_domains)
        recs = evaluate_forward(kw, items, base_domains, registry, cache=cache)
        records_by_sys["regex"] = recs
        systems["regex"] = {
            "name": "regex", "label": "keyword/regex parser", "model": None,
            "records": strip_views(recs), "forward": forward_metrics(recs),
            "smoke": bool(args.limit),
        }  # fmt: skip
        # the writer (14B, still resident) summarises every translator's own traces
        lora_src = resolve_runs([args.lora_forward]) if args.lora_forward else []
        if lora_src:
            lora = json.loads(
                find_file(lora_src, "lora_forward.json").read_text(encoding="utf-8")
            )
            records_by_sys["lora"] = rehydrate(
                lora["records"], domains, registry, cache
            )
        writer = ReverseTranslator(w_translator, keep_alive="30m")
        summaries: dict[str, dict[str, Any]] = {}
        for key, recs in records_by_sys.items():
            log(f"writing summaries for {key}")
            summaries[key] = write_summaries(
                writer, recs, domains, progress_logger(f"summaries {key}", 20)
            )
        write_json(out / "summaries.json", summaries)

        def rt(key: str, translate: Any) -> None:
            rows = reconstruct_rows(
                translate, records_by_sys[key], summaries[key], domains, registry,
                progress_logger(f"round trip {key}", 20),
            )  # fmt: skip
            systems[key]["round_trip_rows"] = rows
            systems[key]["reverse"] = reverse_metrics(rows)

        rt("o14", lambda text, pack: w_translator.translate(text, pack))
        kw_domains = base_domains

        def kw_translate(text: str, pack: Any) -> TranslationResult:
            dom = next(
                d for d in kw_domains.values() if d.pack.spec_name == pack.spec_name
            )
            return keyword_translator(registry, kw_domains)(_ItemShim(dom.name, text))

        rt("regex", kw_translate)
        # gemma reads the summaries (swap models: gemma first needs to be loaded again)
        server.unload_all()
        gemma_be = OllamaBackend(
            args.small, host=args.host, keep_alive="30m", options=CTX
        )
        gemma_tr = PolicyTranslator(gemma_be, registry)
        rt("gemma", lambda text, pack: gemma_tr.translate(text, pack))
        gemma_be.unload()
    result = {
        "systems": systems, "sealed_sha256": SEALED_SHA256, "smoke": bool(args.limit),
        "writer": args.writer, "lora_included": bool(lora_src),
    }  # fmt: skip
    write_json(out / "ollama_eval.json", result)
    for key, s in systems.items():
        ex = s["forward"]["exact"]["sealed"]
        log(f"{key}: sealed exact {ex['exact']:.1%} (n={ex['n']})")


# ---------------------------------------------------------------------- eval-lora-rt


def cmd_eval_lora_rt(args: argparse.Namespace) -> None:
    out = run_dir(args.out)
    registry, base_domains = make_domains(args, execute=True)
    domains = with_policy_packs(base_domains)
    lora_src = resolve_runs([args.lora_forward])
    ollama_src = resolve_runs([args.ollama])
    lora = json.loads(
        find_file(lora_src, "lora_forward.json").read_text(encoding="utf-8")
    )
    best = json.loads(find_file(lora_src, "best.json").read_text(encoding="utf-8"))
    summaries = json.loads(
        find_file(ollama_src, "summaries.json").read_text(encoding="utf-8")
    )["lora"]
    cache: dict[str, Any] = {}
    records = rehydrate(lora["records"], domains, registry, cache)
    be, name = load_adapter_backend(args, best)
    from genetic_coil_arena.backends import AdapterTextBackend

    translator = PolicyTranslator(AdapterTextBackend(be, name), registry)
    doms = with_lora_packs(base_domains)
    # reconstruct_rows reads domain.report_pack and domain.facts: take both from the right places
    from dataclasses import replace

    mixed = {n: replace(d, report_pack=doms[n].report_pack) for n, d in domains.items()}
    rows = reconstruct_rows(
        lambda text, pack: translator.translate(text, pack),
        records, summaries, mixed, registry, progress_logger("round trip lora", 20),
    )  # fmt: skip
    write_json(
        out / "lora_rt.json",
        {"round_trip_rows": rows, "reverse": reverse_metrics(rows)},
    )
    rt = reverse_metrics(rows)["all"]
    log(f"lora round trip {rt['round_trip']:.1%} (n={rt['n']})")


# ---------------------------------------------------------------------- aggregate


def cmd_aggregate(args: argparse.Namespace) -> None:
    from genetic_coil_arena.experiments.r7a_step2_report import build_report, render

    run_dir(args.out)
    sources = resolve_runs(args.runs)

    def maybe(name: str) -> Optional[dict[str, Any]]:
        try:
            return json.loads(find_file(sources, name).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    lora = maybe("lora_forward.json")
    ollama = maybe("ollama_eval.json")
    lora_rt = maybe("lora_rt.json")
    if lora and lora_rt:
        lora["round_trip_rows"] = lora_rt["round_trip_rows"]
        lora["reverse"] = lora_rt["reverse"]
    runs = [
        json.loads(p.read_text(encoding="utf-8"))
        for s in sources
        if (p := s / "dev.json").exists()
    ]
    pool_stats = maybe("pool_stats.json")
    report = build_report(lora, ollama, runs, pool_stats)
    report["generated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    report["host"] = {"machine": platform.node(), "python": platform.python_version()}
    report["command"] = "python " + " ".join(sys.argv)
    md = render(report)
    stem = Path(args.stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    write_json(stem.with_suffix(".json"), report)
    rd = os.environ.get("GCA_RUN_DIR")
    if rd:
        for ext in (".md", ".json"):
            shutil.copy(stem.with_suffix(ext), Path(rd) / (stem.name + ext))
    log(f"wrote {stem}.md and .json")


# ---------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument(
        "--out", default=None, help="artifact folder (default $GCA_RUN_DIR)"
    )
    ap.add_argument("--squad-seeds", type=int, default=8)
    ap.add_argument("--plant-seeds", type=int, default=5)
    ap.add_argument("--base", default=BASE)
    ap.add_argument(
        "--online", action="store_true", help="allow Hugging Face downloads"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("para-gen")
    p.add_argument("--model", default=GENERATOR)
    p.add_argument("--bases", type=int, default=360, help="gold meanings to paraphrase")
    p.add_argument("--per-call", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fake", action="store_true", help="scripted generator (no Ollama)")
    p.set_defaults(fn=cmd_para_gen)

    p = sub.add_parser("para-verify")
    p.add_argument("--raw", default="runs:r7a2-para-gen")
    p.add_argument("--model", default=WRITER)
    p.add_argument(
        "--templates",
        type=int,
        default=14000,
        help="valid templated examples per domain",
    )
    p.add_argument("--retries", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fake", action="store_true")
    p.set_defaults(fn=cmd_para_verify)

    p = sub.add_parser("train")
    p.add_argument("--pool", default="runs:r7a2-para-verify")
    p.add_argument("--name", default=None)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--paraphrase-share", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--limit", type=int, default=0, help="subsample the dev items (smoke)"
    )
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="no gradient checkpointing (needs >10 GB)",
    )
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval-lora")
    p.add_argument("--runs", nargs="+", default=["runs:r7a2-train-*"])
    p.add_argument(
        "--limit", type=int, default=0, help="subsample the sealed items (smoke only)"
    )
    p.set_defaults(fn=cmd_eval_lora)

    p = sub.add_parser("eval-ollama")
    p.add_argument("--writer", default=WRITER)
    p.add_argument("--small", default=SMALL)
    p.add_argument("--lora-forward", default="runs:r7a2-eval-lora")
    p.add_argument("--limit", type=int, default=0, help="smoke only")
    p.set_defaults(fn=cmd_eval_ollama)

    p = sub.add_parser("eval-lora-rt")
    p.add_argument("--lora-forward", default="runs:r7a2-eval-lora")
    p.add_argument("--ollama", default="runs:r7a2-eval-ollama")
    p.set_defaults(fn=cmd_eval_lora_rt)

    p = sub.add_parser("aggregate")
    p.add_argument("--runs", nargs="+", default=["runs:r7a2-*"])
    p.add_argument("--stem", default="docs/results/r7a_step2")
    p.set_defaults(fn=cmd_aggregate)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
