"""R7a step 2: assemble and render ``docs/results/r7a_step2.{md,json}``.

``build_report`` takes the saved outputs of the evaluation jobs (the distilled model's
forward run and round trip, the Ollama systems' runs, the training sweep's dev files and
the training-pool statistics) and returns one JSON-safe report with a G3 verdict for each
translator; ``render`` writes it as Markdown. No model runs here.
"""

import json
import math
from typing import Any, Optional

from .r7a_report import verdict
from .r7a_sealed import SEALED_SHA256

DISTILLED_VRAM_MIB = 4096.0  # the target: fits beside the specialists on a shared GPU
ORDER = ("o14", "lora", "gemma", "regex")
LABELS = {
    "o14": "qwen2.5-coder:14b (refusal policy)",
    "lora": "Qwen3-1.7B + LoRA (distilled)",
    "gemma": "gemma4:e4b (refusal policy)",
    "regex": "keyword/regex parser",
}


def pct(x: Optional[float], digits: int = 1) -> str:
    return (
        "n/a"
        if x is None or (isinstance(x, float) and math.isnan(x))
        else f"{100 * x:.{digits}f}%"
    )


def pm(mean: Optional[float], hw: Optional[float], digits: int = 2) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.{digits}f}" + ("" if hw is None else f" ± {hw:.{digits}f}")


def sec(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f} s"


def system_verdict(sys_: dict[str, Any]) -> dict[str, Any]:
    """The G3 / G6 verdict rows for one translator, read from the sealed split."""
    lat = sys_.get("latency") or {}
    vram = sys_.get("vram") or {}
    if sys_["name"] == "lora":
        used_mib = vram.get("peak_allocated_mib")
    else:
        used_mib = vram.get("model_smi_mib")
    gb = None if used_mib is None else used_mib / 1024
    v = verdict(
        sys_["forward"], sys_.get("reverse"), lat.get("warm_item_p95_s"), gb,
        bool(sys_.get("smoke")), split="sealed",
    )  # fmt: skip
    if sys_["name"] == "lora" and used_mib is not None:
        ok = used_mib <= DISTILLED_VRAM_MIB
        v["rows"].append(
            {
                "criterion": "Step-2 target: inference memory beside the specialists",
                "target": f"<= {DISTILLED_VRAM_MIB / 1024:.0f} GB (torch peak allocated)",
                "value": f"{used_mib / 1024:.2f} GB peak allocated"
                + (
                    f"; {vram['model_smi_mib'] / 1024:.2f} GB by nvidia-smi incl. CUDA context"
                    if vram.get("model_smi_mib") is not None
                    else ""
                ),
                "pass": ok,
                "kill": False,
            }
        )
        if not ok and "PASS" == v["label"]:
            v["label"] = "NOT PASSED (no kill criterion hit)"
    return v


def build_report(
    lora: Optional[dict[str, Any]],
    ollama: Optional[dict[str, Any]],
    runs: list[dict[str, Any]],
    pool_stats: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """The combined report (JSON-safe)."""
    systems: dict[str, dict[str, Any]] = {}
    if ollama:
        systems.update(ollama["systems"])
    if lora:
        systems["lora"] = lora
    for s in systems.values():
        s["verdict"] = system_verdict(s)
    sweep = sorted(
        (
            {
                "name": r["name"], "rank": r["config"]["rank"], "n": r["config"]["n"],
                "paraphrases": r["config"].get("n_paraphrases"), "steps": r["config"]["steps"],
                "dev_score": r["dev"]["score"], "final_loss": r["train"]["mean_last_loss"],
                "wall_min": r["train"]["wall_s"] / 60, "peak_train_mib": r["train"]["peak_mib"],
                "adapter_mib": r.get("adapter_mib"),
                "refusal_precision": r["dev"]["refusal"].get("precision"),
                "refusal_recall": r["dev"]["refusal"].get("recall"),
            }
            for r in runs
        ),
        key=lambda x: (x["rank"], x["n"]),
    )  # fmt: skip
    smoke = any(bool(s.get("smoke")) for s in systems.values())
    return {
        "sealed_sha256": SEALED_SHA256,
        "smoke": smoke,
        "systems": {k: systems[k] for k in ORDER if k in systems},
        "sweep": sweep,
        "pool": pool_stats,
        "selected": None if not lora else lora.get("config"),
    }


def _fmt_ci(block: Optional[dict[str, Any]]) -> str:
    if not block:
        return "n/a"
    lo, hi = block["exact_ci"]
    return f"{pct(block['exact'])} ({pct(lo, 0)}-{pct(hi, 0)}, n={block['n']})"


def render(report: dict[str, Any]) -> str:
    systems = report["systems"]
    lines: list[str] = []
    w = lines.append
    w(
        "# R7a step 2: an honest G3 translator, and a distilled model that fits beside the specialists"
    )
    w("")
    w(
        f"Generated {report.get('generated', '')} on {report.get('host', {}).get('machine', '')} "
        f"(Python {report.get('host', {}).get('python', '')}). Command: `{report.get('command', '')}`."
    )
    if report["smoke"]:
        w("")
        w(
            "> **SMOKE RUN.** A subsample; the numbers check the pipeline and are not the evaluation."
        )
    w("")
    w("## What changed since step 1")
    w("")
    w(
        "- **Sealed test set.** 174 hand-written items (`experiments/data/r7a_sealed_test.jsonl`), written from the specs before any step-2 prompt or model work, none labelled by a model. SHA-256 `"
        + report["sealed_sha256"]
        + "`, asserted by `tests/test_r7a_sealed.py` and by every script that loads it. Tuning (prompts, training data, checkpoint choice) used the step-1 set and a separate dev set only; each translator was run on the sealed set once."
    )
    w(
        "- **Refusal policy** (`translator/policy.py`). Parse first, validate after: the model extracts into a relaxed schema; code fills documented defaults, asks for a missing mandatory choice, and validates. Refusal only for out-of-domain, contradictory or unsatisfiable requests (a value the schema forbids is sent back once, then refused). A new `clarify` outcome returns a question for an in-domain but ambiguous request; it is neither an exact match nor a refusal and is reported separately."
    )
    w(
        "- **Distilled translator.** A LoRA on Qwen3-1.7B that writes the compact policy reply (params that differ from the default; a refusal; or a question), with the same schema check and one retry at inference. Training data: templates plus paraphrases from qwen3-coder:30b, each kept only if qwen2.5-coder:14b recovers the gold outcome from it, plus retry-format examples; anything whose phrasing frame overlaps the sealed, dev or step-1 sets (4-gram containment >= 0.5) is dropped, and the training job refuses to start if any survives."
    )
    w("")
    ev = next((s for s in systems.values() if "forward" in s), None)
    if ev:
        ex = ev["forward"]["exact"]["sealed"]
        w(
            f"Sealed set: {ev['forward']['n_items']} items, {ex['n']} with a gold spec, the rest gold refusals or gold clarifications; squad and plant."
        )
    w("")
    w("## Verdict")
    w("")
    w(
        "| Translator | G3 verdict | Exact match (sealed) | Round trip | Warm p95 | Inference memory |"
    )
    w("|---|---|---|---|---|---|")
    for key, s in systems.items():
        v = s["verdict"]
        ex = s["forward"]["exact"].get("sealed")
        rt = (s.get("reverse") or {}).get("all")
        lat = s.get("latency") or {}
        vr = s.get("vram") or {}
        mem = "n/a"
        if key == "lora" and vr.get("peak_allocated_mib") is not None:
            mem = f"{vr['peak_allocated_mib'] / 1024:.2f} GB"
        elif vr.get("model_smi_mib") is not None:
            mem = f"{vr['model_smi_mib'] / 1024:.1f} GB"
        w(
            f"| {LABELS.get(key, key)} | **{v['label']}** | {_fmt_ci(ex)} | {pct(rt['round_trip']) if rt else 'n/a'} | {sec(lat.get('warm_item_p95_s'))} | {mem} |"
        )
    w("")
    for key, s in systems.items():
        w(f"### {LABELS.get(key, key)}")
        w("")
        w("| Criterion | Target | Measured | Result |")
        w("|---|---|---|---|")
        for r in s["verdict"]["rows"]:
            res = (
                "n/a"
                if r["pass"] is None
                else ("PASS" if r["pass"] else ("KILL" if r["kill"] else "not met"))
            )
            w(f"| {r['criterion']} | {r['target']} | {r['value']} | {res} |")
        w("")
    w("## GoalSpec accuracy on the sealed set")
    w("")
    w(
        "Exact match is over the items with a gold spec (params equal after sorting unordered arrays; 95% Wilson interval). A clarifying question or a refusal on such an item counts as a miss."
    )
    w("")
    w("| Translator | All | Squad | Plant |")
    w("|---|---|---|---|")
    for key, s in systems.items():
        e = s["forward"]["exact"]
        w(
            f"| {LABELS.get(key, key)} | {_fmt_ci(e.get('sealed'))} | {_fmt_ci(e.get('sealed/squad'))} | {_fmt_ci(e.get('sealed/plant'))} |"
        )
    w("")
    fields = sorted(
        {
            f
            for s in systems.values()
            for f in s["forward"]["exact"].get("sealed", {}).get("field_accuracy", {})
        }
    )
    w("Per-field accuracy:")
    w("")
    w("| Translator | " + " | ".join(fields) + " |")
    w("|---|" + "---|" * len(fields))
    for key, s in systems.items():
        fa = s["forward"]["exact"].get("sealed", {}).get("field_accuracy", {})
        w(
            f"| {LABELS.get(key, key)} | "
            + " | ".join(pct(fa.get(f)) if f in fa else "-" for f in fields)
            + " |"
        )
    w("")
    w("## Refusals and clarifications")
    w("")
    w(
        "Refusal precision and recall are against the gold refusals (out of domain, contradictory, unsatisfiable). The clarify columns count questions asked back: the rate over all items, the share of valid-spec items that got a question instead of a spec (counted against exactness), and recall of the gold clarify items."
    )
    w("")
    w(
        "| Translator | Refusal precision | Refusal recall | TP / FP / FN | Clarify rate | Clarify on valid-spec items | Clarify recall | Statuses |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for key, s in systems.items():
        r, c = s["forward"]["refusal"], s["forward"]["clarify"]
        w(
            f"| {LABELS.get(key, key)} | {pct(r['precision'])} | {pct(r['recall'])} | {int(r['tp'])} / {int(r['fp'])} / {int(r['fn'])} | {pct(c['rate'])} ({int(c['asked'])}) | {pct(c['on_spec_rate'])} ({int(c['on_spec'])}) | {pct(c['recall'])} ({int(c['tp'])}/{int(c['gold'])}) | {s['forward']['status_counts']} |"
        )
    w("")
    w("## Task score driven from natural language vs the gold spec")
    w("")
    w(
        "Both runs go through the command centre on the same episodes; an item whose translation failed scores 0 on every seed. Mean over items per seed, then mean ± 95% t-interval over seeds; the last columns are the paired difference (NL minus gold) and whether it lies within its CI (or is >= 0)."
    )
    w("")
    w("| Translator | Domain | n | NL | Gold | NL - gold | Within CI |")
    w("|---|---|---|---|---|---|---|")
    for key, s in systems.items():
        for tkey, t in s["forward"]["task"].items():
            if not tkey.startswith("all/"):
                continue
            w(
                f"| {LABELS.get(key, key)} | {tkey.split('/')[1]} | {t['n_items']} | {pm(t['nl_mean'], t['nl_ci'])} | {pm(t['gold_mean'], t['gold_ci'])} | {pm(t['diff_mean'], t['diff_ci'])} | {'yes' if t['within_ci'] else 'no'} |"
            )
    w("")
    w("## Round trip and summary fidelity")
    w("")
    w(
        "The 14B model writes a status summary from each translator's own executed trace; the translator under test then reads the summary back. Round trip: the reconstruction equals the forward spec. Fidelity (G6): the share of required facts the summary states without contradiction (regular expressions)."
    )
    w("")
    w(
        "| Translator | n | Round trip (95% CI) | Summary field fidelity | All facts in summary |"
    )
    w("|---|---|---|---|---|")
    for key, s in systems.items():
        rt = (s.get("reverse") or {}).get("all")
        if not rt:
            w(f"| {LABELS.get(key, key)} | - | n/a | n/a | n/a |")
            continue
        w(
            f"| {LABELS.get(key, key)} | {rt['n']} | {pct(rt['round_trip'])} ({pct(rt['round_trip_ci'][0], 0)}-{pct(rt['round_trip_ci'][1], 0)}) | {pct(rt['field_fidelity'])} | {pct(rt['all_facts_rate'])} |"
        )
    w("")
    w("## Latency and memory")
    w("")
    w(
        "Cold is the first call (model already resident for the distilled model; for Ollama it follows an unload and includes the load). Warm is every later item, retries included."
    )
    w("")
    w("| Translator | Cold first call | Warm item p50 / p95 | Mean attempts | Memory |")
    w("|---|---|---|---|---|")
    for key, s in systems.items():
        lat = s.get("latency")
        if not lat:
            w(f"| {LABELS.get(key, key)} | - | - | - | - |")
            continue
        vr = s["vram"]
        if key == "lora":
            mem = (
                f"weights {vr['weights_mib'] / 1024:.2f} GB, peak allocated {vr['peak_allocated_mib'] / 1024:.2f} GB"
                + (
                    f", nvidia-smi delta {vr['model_smi_mib'] / 1024:.2f} GB"
                    if vr.get("model_smi_mib") is not None
                    else ""
                )
            )
        else:
            mem = (
                "n/a"
                if vr.get("model_smi_mib") is None
                else f"{vr['model_smi_mib'] / 1024:.1f} GB (nvidia-smi over baseline)"
            )
        cold = lat.get("cold_first_call_s")
        w(
            f"| {LABELS.get(key, key)} | {sec(cold)} | {sec(lat['warm_item_p50_s'])} / {sec(lat['warm_item_p95_s'])} | {lat['mean_attempts']:.2f} | {mem} |"
        )
    w("")
    w("## Distilled model: rank and data-size sweep")
    w("")
    w(
        "Dev score is exact match on the valid items of the dev set plus the step-1 set (never the sealed set); the run with the best dev score was evaluated on the sealed set."
    )
    w("")
    w(
        "| Run | Rank | Train records (paraphrases) | Steps | Dev score | Refusal P / R (dev) | Final loss | Train time | Adapter |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    for r in report["sweep"]:
        best = (
            " **(selected)**"
            if report.get("selected")
            and r["rank"] == report["selected"]["rank"]
            and r["n"] == report["selected"]["n"]
            else ""
        )
        adapter = f"{r['adapter_mib']:.0f} MiB" if r["adapter_mib"] else "n/a"
        w(
            f"| {r['name']}{best} | {r['rank']} | {r['n']} ({r['paraphrases']}) | {r['steps']} | {pct(r['dev_score'])} | {pct(r['refusal_precision'])} / {pct(r['refusal_recall'])} | {r['final_loss']:.4f} | {r['wall_min']:.0f} min | {adapter} |"
        )
    w("")
    pool = report.get("pool")
    if pool:
        w("## Training-data pipeline")
        w("")
        w(
            f"Pool: {pool['n_pool']} records ({pool['sources']}; kinds {pool['kinds']}). Paraphrases: {pool['raw_paraphrases']} generated by qwen3-coder:30b, {pool['verified_paraphrases']} kept after the 14B round-trip check. Dropped by the n-gram leak filter: {pool['dropped_by_overlap']}."
        )
        w("")
        w("| Style card | Generated | Verified | Rate |")
        w("|---|---|---|---|")
        for style, a in pool["accept_by_style"].items():
            w(f"| {style} | {a['n']} | {a['accepted']} | {pct(a['rate'], 0)} |")
        w("")
        w(
            "Verification rate by kind: "
            + ", ".join(
                f"{k} {pct(a['rate'], 0)} of {a['n']}"
                for k, a in pool["accept_by_kind"].items()
            )
            + "."
        )
        w("")
    w("## Failures on the sealed set")
    w("")
    for key, s in systems.items():
        recs = s["records"]
        bad = [
            r for r in recs
            if (r["kind"] == "spec" and not r.get("exact"))
            or (r["kind"] == "refuse" and not r["refused"])
            or (r["kind"] == "clarify" and not r["clarified"])
        ]  # fmt: skip
        w(
            f"**{LABELS.get(key, key)}**: {len(bad)} of {len(recs)} items not as expected. First 12:"
        )
        w("")
        for r in bad[:12]:
            why = f" ({r['reason'][:90]})" if r.get("reason") else ""
            w(
                f'- `{r["id"]}` [{r["kind"]} -> {r["status"]}] "{r["text"]}"  gold `{json.dumps(r["gold"])}`  got `{json.dumps(r["pred"])}`{why}'
            )
        w("")
    w("## Limitations")
    w("")
    w(
        "- The sealed set is written by one author who also wrote the training templates, the prompts and the label conventions. The n-gram test keeps frames apart, not ideas: the same person's notion of 'indirect' or 'inverted wording' shapes both sides, and the 10 gold clarify items encode one author's view of 'ambiguous'."
    )
    w(
        "- The prompts were written after the sealed set existed (the brief required it first), without running them on it. I avoided phrasing conventions that only the sealed items exercise, but I cannot prove my prompt choices were independent of having seen them."
    )
    w(
        "- Paraphrases are kept only when the 14B model recovers the gold spec, which also skews the surviving phrasing toward what that model reads easily; the 14B is also one of the translators evaluated. The labels themselves never came from a model."
    )
    w(
        "- Round trip uses summaries written by the 14B from each translator's own trace, so a translator is also judged by how well it reads the 14B's summary style; the fixed summary template makes this easier than free-form monitoring text. Fidelity is a regular-expression check."
    )
    w(
        "- Memory: the distilled model is bf16 on one process; 'beside the specialists' assumes they share this base model (the HF/PEFT design) and does not include the specialists' own adapters, KV caches or the CUDA context (reported separately by nvidia-smi)."
    )
    w(
        "- Latency was measured on a machine that may have run other work; Ollama VRAM is the GPU's total used memory over the pre-load baseline."
    )
    return "\n".join(lines) + "\n"
