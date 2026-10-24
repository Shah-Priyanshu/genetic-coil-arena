"""R9: do latent messages beat JSON between LoRA specialists on one base model?

Go/no-go test for latent messaging with real models. Two LoRA specialists on one shared
Qwen3-1.7B process a synthetic invoice pipeline: the extractor A reads a raw snippet,
the normalizer B must emit canonical JSON without ever seeing it. A's message is JSON
text, or ``k`` soft tokens built from A's hidden states, or A's KV cache; a single LoRA
that reads the snippet is the non-communicating reference. The kill criterion: latent
messages must match JSON accuracy with at least 30% savings in message size or latency,
or JSON stays the default.

    python scripts/r9_latent_vs_json.py --seeds 0 1 2 --out docs/results/r9_latent_vs_json.md

Trained adapters, per-run result files and logs go to ``$GCA_RUN_DIR`` (default
``outputs/r9_latent_vs_json``); a killed run resumes from the cached result files. Use
``--report-only`` to rebuild the Markdown from them. The "Analysis" section of the
report is hand-written and carried over between rebuilds.

The model needs a GPU with about 10 GB free (it caps itself at 85% of the card) and
torch, transformers and peft (see the ``[llm]`` extra requested in the R9 report).
"""

import argparse
import datetime
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from genetic_coil_arena.evaluation import mean_ci

REPO = Path(__file__).resolve().parent.parent
ANALYSIS_START = "<!-- analysis:start -->"
ANALYSIS_END = "<!-- analysis:end -->"
MATCH_MARGIN = 0.02  # latent "matches" JSON if it is at most 2 points below it
SAVINGS_TARGET = 0.30
FIELDS = ("vendor_canonical", "date_iso", "amount_cents", "currency_iso", "invoice_id")
FIELD_HEADERS = ("vendor", "date", "amount", "currency", "invoice id")
MAIN_LATENT = ("latent_k1", "latent_k4", "latent_k16")
EXTRA_LATENT = ("latent_k16_last", "latent_k16_L28", "latent_k16_e2e")
ORDER = ("single", "json", *MAIN_LATENT, "kv_prefix")
LABELS = {
    "single": "single model (no message)",
    "json": "JSON text",
    "latent_k1": "latent k=1",
    "latent_k4": "latent k=4",
    "latent_k16": "latent k=16",
    "kv_prefix": "KV prefix",
    "latent_k16_L28": "latent k=16, last layer (28)",
    "latent_k16_last": "latent k=16, last-k positions",
    "latent_k16_e2e": "latent k=16, sender trained end to end",
}


def log(message: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


# ------------------------------------------------------------------------------ running


def build_conditions(args: argparse.Namespace, seed: int) -> list[Any]:
    from genetic_coil_arena.experiments.latent_vs_json import Condition

    conditions = [Condition("single"), Condition("json")]
    conditions += [Condition("latent", k=k) for k in args.ks]
    conditions.append(Condition("kv"))
    if args.extras and seed == args.extras_seed:
        for k in args.extra_ks:
            conditions += [
                Condition("latent", k=k, pooling="last", tag="last"),
                Condition("latent", k=k, layer=args.last_layer, tag="L28"),
                Condition("latent", k=k, train_sender=True, tag="e2e"),
            ]
    return conditions


def run_experiments(args: argparse.Namespace, run_dir: Path) -> None:
    from genetic_coil_arena.backends import HFPeftBackend
    from genetic_coil_arena.experiments.latent_vs_json import Experiment, Settings

    settings = Settings(
        steps=args.steps,
        latent_steps=args.latent_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        projector_lr=args.projector_lr,
        layer=args.layer,
        n_test=args.n_test,
        n_ablation=args.n_ablation,
        eval_batch=args.eval_batch,
        n_latency=args.n_latency,
        log_every=args.log_every,
    )
    backend = HFPeftBackend(
        args.base, memory_fraction=0.85, local_files_only=args.offline
    )
    backend.load()
    log(f"loaded {args.base} on {backend.device}; memory cap {backend.memory_fraction}")
    (run_dir / "hardware.json").write_text(
        json.dumps(hardware(), indent=1), encoding="utf-8"
    )
    experiment = Experiment(backend, settings, run_dir, log=log)
    start = time.time()
    for seed in args.seeds:
        experiment.run_seed(seed, build_conditions(args, seed))
        log(f"seed {seed} done; {(time.time() - start) / 60:.1f} min elapsed")


# --------------------------------------------------------------------------- aggregation


def load_results(run_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """``{condition: {seed: result}}`` from the per-run result files."""
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for path in sorted((run_dir / "results").glob("seed*_*.json")):
        if path.name.endswith("_extractor_train.json"):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        out.setdefault(result["condition"], {})[result["seed"]] = result
    return out


def values(
    results: dict[str, dict[int, dict[str, Any]]], cond: str, getter: Any
) -> list[float]:
    """``getter(result)`` for every seed of a condition, in seed order."""
    return [getter(r) for _, r in sorted(results.get(cond, {}).items())]


def pm(vals: list[float], scale: float = 100.0, digits: int = 1) -> str:
    """``mean ± 95% CI half-width`` (the CI needs at least two seeds)."""
    if not vals:
        return "n/a"
    mean, half = mean_ci(vals)
    if math.isnan(half):
        return f"{mean * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} ± {half * scale:.{digits}f}"


def exact(split: str) -> Any:
    return lambda r: r["eval"][split]["exact"]


def field_acc(split: str, name: str) -> Any:
    return lambda r: r["eval"][split]["fields"][name]


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def latency_of(r: dict[str, Any]) -> float:
    return r["latency"]["total_mean_s"]


def human_bytes(n: float) -> str:
    if n < 1024:
        return f"{n:.0f} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KiB"
    return f"{n / 1024**2:.1f} MiB"


def size_text(saving: float) -> str:
    """A size saving as 'saves 40%' or, when the message is bigger, '158x larger'."""
    ratio = 1 - saving
    return f"{ratio:.0f}x larger" if ratio >= 1 else f"saves {saving * 100:.0f}%"


def paired_saving(results: dict, cond: str, ref: str, getter: Any) -> list[float]:
    """Per-seed ``1 - getter(cond) / getter(ref)`` over the seeds both have."""
    seeds = sorted(set(results.get(cond, {})) & set(results.get(ref, {})))
    return [1 - getter(results[cond][s]) / getter(results[ref][s]) for s in seeds]


def paired_difference(results: dict, cond: str, ref: str, getter: Any) -> list[float]:
    seeds = sorted(set(results.get(cond, {})) & set(results.get(ref, {})))
    return [getter(results[cond][s]) - getter(results[ref][s]) for s in seeds]


def message_bytes(results: dict, cond: str, wire: str = "bf16") -> list[float]:
    """Mean message bytes per record, per seed (both splits pooled)."""
    out = []
    for _, r in sorted(results.get(cond, {}).items()):
        ev = r["eval"] if wire == "bf16" else r.get("eval_int8", r["eval"])
        sizes = [ev[s]["message_bytes"] for s in ("id_test", "heldout_test")]
        out.append(sum(sizes) / len(sizes))
    return out


def kill_criterion(results: dict) -> dict[str, Any]:
    """Evaluate the R9 rule for each main latent configuration.

    A configuration passes when its mean full-record exact match is at most
    ``MATCH_MARGIN`` below JSON's on both test splits, and it saves at least
    ``SAVINGS_TARGET`` of JSON's message size or of its end-to-end latency.
    """
    verdicts: dict[str, Any] = {}
    for cond in MAIN_LATENT:
        if cond not in results or "json" not in results:
            continue
        diffs = {
            split: paired_difference(results, cond, "json", exact(split))
            for split in ("id_test", "heldout_test")
        }
        mean_diff = {s: sum(d) / len(d) for s, d in diffs.items()}
        size = paired_saving(
            results,
            cond,
            "json",
            lambda r: sum(
                r["eval"][s]["message_bytes"] for s in ("id_test", "heldout_test")
            ),
        )
        size_int8 = []
        for seed in sorted(set(results[cond]) & set(results["json"])):
            sent = results[cond][seed]
            wire = sum(
                sent["eval_int8"][s]["message_bytes"]
                for s in ("id_test", "heldout_test")
            )
            base = sum(
                results["json"][seed]["eval"][s]["message_bytes"]
                for s in ("id_test", "heldout_test")
            )
            size_int8.append(1 - wire / base)
        latency = paired_saving(results, cond, "json", latency_of)
        matches = all(d >= -MATCH_MARGIN for d in mean_diff.values())
        size_ok = sum(size) / len(size) >= SAVINGS_TARGET
        latency_ok = sum(latency) / len(latency) >= SAVINGS_TARGET
        verdicts[cond] = {
            "accuracy_difference": mean_diff,
            "accuracy_difference_ci": {
                s: mean_ci(d)[1] for s, d in diffs.items() if len(d) > 1
            },
            "matches_accuracy": matches,
            "size_saving": sum(size) / len(size),
            "size_saving_int8": sum(size_int8) / len(size_int8),
            "latency_saving": sum(latency) / len(latency),
            "size_ok": size_ok,
            "latency_ok": latency_ok,
            "passes": matches and (size_ok or latency_ok),
        }
    return verdicts


# ------------------------------------------------------------------------------- report


def hardware() -> dict[str, str]:
    info = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": f"{platform.processor() or platform.machine()} ({os.cpu_count()} logical cores)",
    }
    try:
        import peft
        import torch
        import transformers

        info["torch"] = torch.__version__
        info["transformers"] = transformers.__version__
        info["peft"] = peft.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = f"{props.name}, {props.total_memory / 2**30:.0f} GiB"
    except ImportError:
        pass
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO,
        )
        info["commit"] = out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["commit"] = "unknown"
    return info


def carried_analysis(out: Path) -> str:
    if out.exists():
        text = out.read_text(encoding="utf-8")
        if ANALYSIS_START in text and ANALYSIS_END in text:
            return text.split(ANALYSIS_START, 1)[1].split(ANALYSIS_END, 1)[0].strip()
    return "_(not written yet)_"


def render_report(
    results: dict, meta: dict[str, str], command: str, analysis: str
) -> str:
    if not results:
        raise SystemExit("no result files found; run the experiment first")
    some = next(iter(next(iter(results.values())).values()))
    st = some["settings"]
    n_seeds = len(results.get("json", {}))
    verdict = kill_criterion(results)
    lines: list[str] = []
    add = lines.append

    add("# R9: latent messages vs JSON between LoRA specialists")
    add("")
    add(
        "Go/no-go test for latent messaging with real models. Two LoRA specialists share "
        "one base model (Qwen3-1.7B). The extractor A reads a raw "
        "invoice-like snippet; the normalizer B must emit canonical JSON "
        "`{vendor_canonical, date_iso, amount_cents, currency_iso, invoice_id}` and never "
        "sees the snippet, only A's message."
    )
    add("")
    add(
        "**Kill criterion:** latent messages must match JSON accuracy with at least 30% "
        "savings in message size or latency. If they don't, JSON stays the default."
    )
    add("")

    # ---- verdict
    add("## Verdict")
    add("")
    any_pass = any(v["passes"] for v in verdict.values())
    add(
        "**The kill criterion is "
        + ("MET" if any_pass else "NOT MET")
        + (
            "; latent messaging passes the go/no-go."
            if any_pass
            else "; JSON stays the default."
        )
        + "**"
    )
    add("")
    add(
        f"Rule applied to each main latent configuration (layer 14, last-k positions): mean "
        f"full-record exact match at most {MATCH_MARGIN * 100:.0f} points below JSON's on both "
        f"test splits, **and** a saving of at least {SAVINGS_TARGET * 100:.0f}% of JSON's "
        "message size or of its end-to-end latency. Savings are per seed, then averaged "
        f"({n_seeds} seeds)."
    )
    add("")
    rows = []
    for cond, v in verdict.items():
        d = v["accuracy_difference"]
        rows.append(
            [
                LABELS[cond],
                f"{d['id_test'] * 100:+.1f}",
                f"{d['heldout_test'] * 100:+.1f}",
                "yes" if v["matches_accuracy"] else "**no**",
                f"{size_text(v['size_saving'])} (int8: {size_text(v['size_saving_int8'])})",
                f"{v['latency_saving'] * 100:+.0f}%",
                "**pass**" if v["passes"] else "fail",
            ]
        )
    lines += table(
        [
            "Latent config",
            "Δ exact, ID (pts)",
            "Δ exact, held-out (pts)",
            "Matches JSON",
            "Message size vs JSON",
            "Latency saving",
            "Criterion",
        ],
        rows,
    )
    add("")
    add(
        "Message size is bytes per record against JSON's generated text (about 100 bytes). "
        "Latency saving is `1 - latent / JSON` of the mean end-to-end time per record."
    )
    add("")

    # ---- analysis
    add("## Analysis")
    add("")
    add(ANALYSIS_START)
    add(analysis)
    add(ANALYSIS_END)
    add("")

    # ---- results
    add("## Results")
    add("")
    add(
        f"Mean ± 95% Student-t CI over {n_seeds} seeds (`evaluation.stats.mean_ci`; with "
        f"3 seeds the t multiplier is 4.3, so intervals are wide). Test sets: "
        f"{st['n_test']} records per split, identical for every seed. **In-distribution** "
        "= fresh records with training vendors and training date/amount formats. "
        "**Held-out** = vendors never seen in training, written in date and amount formats "
        "never seen in training."
    )
    add("")
    add("### Full-record exact match (%)")
    add("")
    rows = []
    for cond in ORDER:
        if cond in results:
            rows.append(
                [
                    LABELS[cond],
                    pm(values(results, cond, exact("id_test"))),
                    pm(values(results, cond, exact("heldout_test"))),
                    ", ".join(
                        f"{100 * v:.1f}"
                        for v in values(results, cond, exact("heldout_test"))
                    ),
                ]
            )
    lines += table(
        ["Condition", "In-distribution", "Held-out", "Held-out, per seed"], rows
    )
    add("")
    for split, title in (("heldout_test", "Held-out"), ("id_test", "In-distribution")):
        add(f"### Per-field exact match, {title.lower()} (%)")
        add("")
        rows = []
        for cond in ORDER:
            if cond in results:
                rows.append(
                    [LABELS[cond]]
                    + [pm(values(results, cond, field_acc(split, f))) for f in FIELDS]
                )
        lines += table(["Condition", *FIELD_HEADERS], rows)
        add("")
    if "json" in results:
        add("### Stage A alone (JSON condition)")
        add("")
        add(
            "How well the extractor copies the four raw fields, scored against the "
            "snippet's gold raw fields. This bounds the JSON pipeline from above."
        )
        add("")
        rows = []
        for split in ("id_test", "heldout_test"):
            rows.append(
                [split.replace("_test", "")]
                + [
                    pm(
                        values(
                            results,
                            "json",
                            lambda r, f=f, s=split: r["eval"][s]["stage_a"]["fields"][
                                f
                            ],
                        )
                    )
                    for f in ("vendor", "date", "amount", "invoice_id")
                ]
                + [
                    pm(
                        values(
                            results,
                            "json",
                            lambda r, s=split: r["eval"][s]["stage_a"]["exact"],
                        )
                    )
                ]
            )
        lines += table(
            ["Split", "vendor", "date", "amount", "invoice id", "all four"], rows
        )
        add("")

    add("### Message size and end-to-end latency")
    add("")
    add(
        "Message size is per record, bytes on the wire: JSON text is UTF-8 bytes of what A "
        "generated; a latent message is k × 2048 × 2 bytes in bfloat16 (k × 2048 + 2k as "
        "int8 with a float16 scale per vector); KV is the cache of the snippet "
        "(112 KiB per token). Latency is wall clock from the raw snippet to the canonical "
        f"JSON string for one record at a time (batch size 1), {2 * st['n_latency']} records, "
        "greedy decoding; the latent path goes through an envelope and a `TensorStore`. "
        "The CPU was busy with another experiment and decoding is launch-bound, so absolute "
        "times are upper bounds; ratios are the useful number."
    )
    add("")
    rows = []
    for cond in ORDER:
        if cond not in results:
            continue
        sizes = message_bytes(results, cond)
        int8 = (
            human_bytes(sum(message_bytes(results, cond, "int8")) / n_seeds)
            if cond.startswith("latent")
            else "n/a"
        )
        tokens = values(
            results,
            cond,
            lambda r: sum(
                r["eval"][s]["message_tokens"] for s in ("id_test", "heldout_test")
            )
            / 2,
        )
        stage = [
            sum(v) / len(v)
            for v in zip(
                *[
                    (
                        r["latency"]["stage_mean_s"]["sender"],
                        r["latency"]["stage_mean_s"]["handoff"],
                        r["latency"]["stage_mean_s"]["receiver"],
                    )
                    for _, r in sorted(results[cond].items())
                ]
            )
        ]
        saving = (
            pm(paired_saving(results, cond, "json", latency_of), 100, 0) + "%"
            if cond not in ("json",) and "json" in results
            else "reference"
        )
        rows.append(
            [
                LABELS[cond],
                human_bytes(sum(sizes) / len(sizes)) if cond != "single" else "none",
                int8,
                f"{sum(tokens) / len(tokens):.0f}" if cond != "single" else "none",
                pm(values(results, cond, latency_of), 1, 2),
                f"{stage[0]:.2f} / {stage[1]:.3f} / {stage[2]:.2f}",
                saving,
            ]
        )
    lines += table(
        [
            "Condition",
            "Message (bf16 / text)",
            "Message (int8)",
            "Message tokens",
            "Latency per record (s)",
            "Sender / handoff / receiver (s)",
            "Latency saving vs JSON",
        ],
        rows,
    )
    add("")

    add("### Training cost")
    add("")
    add(
        f"{st['steps']} optimiser steps ({st['latent_steps']} for latent receivers) of {st['batch_size']} records per adapter "
        f"(AdamW, lr {st['lr']:g}, LoRA r={st['lora_r']} on q/v). A latent receiver also "
        "trains its projector; A's own LoRA is trained once per seed and reused by the JSON, "
        "latent and KV conditions."
    )
    add("")
    rows = []
    for cond in ORDER:
        if cond not in results:
            continue
        wall = values(results, cond, lambda r: r["train"]["receiver"]["wall_s"])
        loss = values(results, cond, lambda r: r["train"]["receiver"]["mean_last_loss"])
        params = values(results, cond, lambda r: r["train"]["receiver"]["n_trainable"])
        peak = values(
            results, cond, lambda r: r["train"]["receiver"]["peak_memory_mib"]
        )
        rows.append(
            [
                LABELS[cond] + (" (receiver B)" if cond not in ("single",) else ""),
                f"{st['steps']}",
                pm(wall, 1, 0),
                f"{sum(loss) / len(loss):.3f}",
                f"{sum(params) / len(params) / 1e6:.1f} M",
                f"{max(peak) / 1024:.1f} GiB",
            ]
        )
    ext = values(results, "json", lambda r: r["train"]["extractor"]["wall_s"])
    if ext:
        rows.append(
            ["extractor A (shared)", f"{st['steps']}", pm(ext, 1, 0), "", "1.6 M", ""]
        )
    lines += table(
        [
            "Adapter",
            "Steps",
            "Wall time (s)",
            "Final loss",
            "Trainable params",
            "Peak GPU memory",
        ],
        rows,
    )
    total = 0.0
    for per in results.values():
        for r in per.values():
            total += r["train"]["receiver"]["wall_s"]
    for r in results.get("json", {}).values():
        total += r["train"]["extractor"]["wall_s"]
    add("")
    add(
        f"Total GPU training time of the whole study (every adapter, every seed, extras "
        f"included): **{total / 60:.0f} min**."
    )
    add("")

    add("### Does B use the message? Ablations")
    add("")
    add(
        "Full-record exact match (%) on the first "
        f"{st['n_ablation']} records of each split, with the message replaced by: nothing "
        "(zeros), another record's message from the same batch (shuffle), or Gaussian noise "
        "with the batch's per-dimension mean and variance (noise; latent via "
        "`evaluation.interventions.random_matched`, KV per layer and tensor). "
        "For JSON only shuffling makes sense. 'intact' is the same subset without "
        "intervention. A receiver that used the message collapses."
    )
    add("")
    rows = []
    for cond in ORDER:
        if cond not in results or "ablations" not in some_result(results, cond):
            continue
        names = list(some_result(results, cond)["ablations"])
        row = [LABELS[cond]]
        for name in ("intact", "zero", "shuffle", "noise"):
            if name in names:
                cell = " / ".join(
                    pm(
                        values(
                            results,
                            cond,
                            lambda r, n=name, s=split: r["ablations"][n][s]["exact"],
                        ),
                        100,
                        0,
                    )
                    for split in ("id_test", "heldout_test")
                )
            else:
                cell = "n/a"
            row.append(cell)
        rows.append(row)
    lines += table(
        ["Condition", "intact (ID / held-out)", "zero", "shuffle", "noise"], rows
    )
    add("")

    add("### Teacher-forced NLL under message ablations")
    add("")
    add(
        "Mean negative log-likelihood (nats per target token) of the gold canonical JSON, "
        "held-out records, with the message intact or ablated. Unlike exact match it moves "
        "while accuracy is still zero: the gap between 'intact' and 'zero' is what the "
        "receiver extracts from the message. Single model: no message."
    )
    add("")
    rows = []
    for cond in ORDER:
        if cond in results and "nll" in some_result(results, cond):
            names = list(some_result(results, cond)["nll"])
            row = [LABELS[cond]]
            for name in ("intact", "zero", "shuffle", "noise"):
                if name in names:
                    vals = values(
                        results, cond, lambda r, n=name: r["nll"][n]["heldout_test"]
                    )
                    row.append(pm(vals, 1, 3))
                else:
                    row.append("n/a")
            rows.append(row)
    lines += table(["Condition", "intact", "zero", "shuffle", "noise"], rows)
    add("")

    add("### int8 messages")
    add("")
    add(
        "The same trained receivers, with each latent message quantised to int8 (one "
        "absmax scale per vector) and dequantised before B reads it. Exact match (%), "
        "ID / held-out."
    )
    add("")
    rows = []
    for cond in MAIN_LATENT:
        if cond in results:
            row = [LABELS[cond]]
            for key in ("eval", "eval_int8"):
                row.append(
                    " / ".join(
                        pm(
                            values(
                                results,
                                cond,
                                lambda r, k=key, s=split: r[k][s]["exact"],
                            ),
                            100,
                            1,
                        )
                        for split in ("id_test", "heldout_test")
                    )
                )
            rows.append(row)
    lines += table(["Condition", "bfloat16", "int8"], rows)
    add("")

    extras = [c for c in EXTRA_LATENT if c in results]
    if extras:
        add("### One-seed ablations: pooling, layer, sender")
        add("")
        add(
            "Seed 0 only, k=16. 'Main' is the same seed's layer-14, attention-pool, frozen-"
            "sender run. Exact match (%) ID / held-out, final training loss and teacher-"
            "forced NLL per token on held-out records (intact message)."
        )
        add("")
        ablation_seed = min(results[extras[0]])
        rows = []
        for cond in ("latent_k16", *extras):
            r = results.get(cond, {}).get(ablation_seed)
            if r is None:
                continue
            nll = r.get("nll", {}).get("intact", {}).get("heldout_test")
            rows.append(
                [
                    LABELS[cond] + (" (main)" if cond == "latent_k16" else ""),
                    f"{r['eval']['id_test']['exact'] * 100:.1f} / {r['eval']['heldout_test']['exact'] * 100:.1f}",
                    f"{r['train']['receiver']['mean_last_loss']:.3f}",
                    "n/a" if nll is None else f"{nll:.3f}",
                ]
            )
        lines += table(
            ["Configuration", "Exact (ID / held-out)", "Final loss", "Held-out NLL"],
            rows,
        )
        add("")

    checks = [
        r.get("protocol_roundtrip_identical")
        for per in results.values()
        for r in per.values()
        if "protocol_roundtrip_identical" in r
    ]
    if checks:
        add(
            "Protocol check: for "
            f"{sum(bool(c) for c in checks)} of {len(checks)} latent runs the outputs through "
            "the envelope and `TensorStore` path were identical to the direct path on a probe "
            "batch of 8 held-out records."
        )
        add("")

    # ---- method
    add("## Method")
    add("")
    add(
        "- **Data** (`experiments/records.py`, seeded): invoice-like snippets with a vendor "
        "(48 training vendors; aliases: case, legal suffix, `&`, one-character typos), a date "
        "(4 training formats), an amount (4 training formats, 5 currencies) and an invoice id, "
        "plus distractor dates, amounts and ids. Held-out records use 24 unseen vendors, 4 "
        "unseen date formats (`14.03.2024` with day above 12, `14-Mar-24`, "
        "`March 14th, 2024`, `20240314`) and 3 unseen amount formats (`1.234,50 €`, "
        "`1 234,50 EUR`, `1234.50€`). Snippets average 91 tokens; A's raw-field JSON 46 "
        "tokens and 104 bytes; the canonical JSON 52 tokens."
    )
    add(
        f"- **Model:** Qwen3-1.7B, bf16, sdpa attention. LoRA r={st['lora_r']}, "
        f"alpha={st['lora_alpha']}, on q and v, one adapter per specialist, switched with "
        "`set_adapter`. `lm_head` is applied on supervised positions only (chunked and "
        "recomputed in the backward pass); sequences over 130 tokens (A and the single model) "
        "use gradient checkpointing. Process capped at 85% of GPU memory."
    )
    add(
        "- **JSON:** A is trained to emit raw-field JSON; B is trained on gold raw JSON and "
        "at test time reads A's generated text."
    )
    add(
        "- **Latent-k:** A frozen (the JSON condition's extractor). Its `hidden_states[14]` "
        "over the whole prompt are standardized per coordinate (statistics measured once on "
        "96 training snippets; layer-14 states have a few massive coordinates) and k learned "
        "queries attend over them (Perceiver-style pooling), giving a k x 2048 message in "
        "bfloat16 on the wire. A receiver-side projector (MLP, learned position, RMSNorm "
        "with a gain starting at the embedding scale) turns it into k soft tokens placed "
        "after 'Message:' in B's input, as the JSON text would be. Trained: B's LoRA "
        "(initialised from the JSON receiver's) and the projector. Attention pooling is the "
        "main design because a linear probe (see Analysis) finds that the last positions of "
        "layer 14 carry currency and vendor but not the digits; last-k is a one-seed "
        "ablation."
    )
    add(
        "- **KV prefix:** B's LoRA is trained while attending to A's KV cache of the snippet "
        "(A's adapter applied to the snippet; B continues after it)."
    )
    add("- **Single model:** one LoRA reads the snippet and writes the canonical JSON.")
    add(
        "- **Seeds** change the LoRA and projector initialisation, the training records and "
        "their order. The test sets are shared. Greedy decoding."
    )
    add("")

    add("## Reproduce")
    add("")
    add("```")
    add(command)
    add("```")
    add("")
    add(
        "Environment: Python with torch, transformers and peft (venv_llm of the R4 spike), "
        "`PYTHONPATH=src`, `PYTHONUTF8=1`, `HF_HOME` pointing at a cache that holds "
        "`Qwen/Qwen3-1.7B`. Per-run result files and adapters are written to "
        "`$GCA_RUN_DIR`."
    )
    add("")
    add("## Hardware and software")
    add("")
    for key in (
        "gpu",
        "cpu",
        "os",
        "python",
        "torch",
        "transformers",
        "peft",
        "commit",
    ):
        if key in meta:
            add(f"- {key}: {meta[key]}")
    add("")
    add(
        "**Caveat: busy CPU.** Another experiment was using all CPU cores during these "
        "runs. HF eager decoding at batch size 1 is launch-bound (about 100 ms per token "
        "here, 75 ms in R4), so every latency above is an upper bound and noisier than on an "
        "idle machine. Savings are ratios of times measured back to back under the same load, "
        "which cancels part of it but not all: a faster CPU would shrink the sender's share "
        "of every condition alike."
    )
    add("")
    return "\n".join(lines) + "\n"


def some_result(results: dict, cond: str) -> dict[str, Any]:
    return next(iter(results[cond].values()))


# --------------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--steps", type=int, default=200, help="steps for extractor, JSON, KV, single"
    )
    parser.add_argument("--latent-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--projector-lr", type=float, default=1e-3)
    parser.add_argument(
        "--layer", type=int, default=14, help="hidden-state layer to send"
    )
    parser.add_argument("--last-layer", type=int, default=28)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--extra-ks", type=int, nargs="+", default=[16])
    parser.add_argument("--no-extras", dest="extras", action="store_false")
    parser.add_argument("--extras-seed", type=int, default=0)
    parser.add_argument(
        "--n-test", type=int, default=200, help="records per test split"
    )
    parser.add_argument("--n-ablation", type=int, default=100)
    parser.add_argument("--eval-batch", type=int, default=100)
    parser.add_argument(
        "--n-latency", type=int, default=5, help="records per split timed"
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--offline", action="store_true", help="never touch the network"
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out", default="docs/results/r9_latent_vs_json.md")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="tiny run to check the plumbing (20 steps)"
    )
    args = parser.parse_args()

    if args.smoke:
        args.seeds, args.steps, args.latent_steps = [0], 20, 20
        args.n_test, args.n_ablation = 32, 32
        args.n_latency, args.log_every, args.ks, args.extra_ks = 1, 10, [4], [4]
    run_dir = Path(
        args.run_dir
        or os.environ.get("GCA_RUN_DIR")
        or REPO / "outputs" / "r9_latent_vs_json"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "hardware.json"
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out

    if not args.report_only:
        run_experiments(args, run_dir)

    results = load_results(run_dir)
    command = (
        "python scripts/r9_latent_vs_json.py "
        f"--seeds {' '.join(map(str, args.seeds))} --steps {args.steps} --latent-steps {args.latent_steps} "
        f"--batch-size {args.batch_size} --n-test {args.n_test} "
        f"--out docs/results/r9_latent_vs_json.md"
    )
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.exists()
        else hardware()
    )
    report = render_report(results, meta, command, carried_analysis(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    log(f"wrote {out}")
    summary = {
        "verdict": kill_criterion(results),
        "conditions": sorted(results),
        "seeds": sorted({s for per in results.values() for s in per}),
    }
    (out.with_suffix(".json")).write_text(
        json.dumps(summary, indent=1, default=float), encoding="utf-8"
    )
    print(json.dumps(summary["verdict"], indent=1, default=float), file=sys.stdout)


if __name__ == "__main__":
    main()
