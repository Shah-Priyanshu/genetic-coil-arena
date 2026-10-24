"""R7a: the end-to-end vertical slice, evaluated (goal G3, and G6's summary fidelity).

natural-language instruction -> translator -> validated GoalSpec -> command centre ->
specialists -> trace -> reverse translator, on two domains (squad missions and the
plant-floor diagnosis configuration) with one translator.

Systems compared on the same instructions: the primary model (default qwen2.5-coder:14b
through Ollama structured outputs), a smaller model (gemma4:e4b) and a keyword/regex
parser. Only the primary model also writes trace summaries and is scored on the round
trip. Writes ``<out>.md`` and ``<out>.json``.

    python scripts/r7a_vertical_slice.py --out docs/results/r7a_vertical_slice
    python scripts/r7a_vertical_slice.py --limit 20 --out smoke/r7a      # smoke run

Ollama: if the server is unreachable this starts ``ollama serve`` and stops it at the end
(only if it started it); models are unloaded when done. Needs the Tennessee Eastman data
(``python scripts/fetch_tep.py``). Nothing here needs more than ~10 GB of VRAM.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

from genetic_coil_arena.backends.ollama import (  # noqa: E402
    DEFAULT_HOST,
    OllamaBackend,
    OllamaError,
    http_transport,
)
from genetic_coil_arena.envs.plant.tep import TepData  # noqa: E402
from genetic_coil_arena.experiments.r7a_domains import (  # noqa: E402
    build_domains,
    build_registry,
)
from genetic_coil_arena.experiments.r7a_evalset import build_evalset  # noqa: E402
from genetic_coil_arena.experiments.r7a_pipeline import keyword_translator  # noqa: E402
from genetic_coil_arena.experiments.r7a_report import (  # noqa: E402
    evaluate_forward,
    evaluate_reverse,
    forward_metrics,
    reverse_metrics,
    select_items,
    verdict,
)
from genetic_coil_arena.translator import (  # noqa: E402
    ReverseTranslator,
    Translator,
    percentile,
)

PRIMARY = "qwen2.5-coder:14b"
SMALL = "gemma4:e4b"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------- ollama


def find_ollama() -> Optional[str]:
    exe = shutil.which("ollama")
    if exe:
        return exe
    local = os.environ.get("LOCALAPPDATA", "")
    for cand in (Path(local) / "Programs" / "Ollama" / "ollama.exe",):
        if cand.exists():
            return str(cand)
    return None


class OllamaServer:
    """Use the running Ollama server, or start one and stop it afterwards."""

    def __init__(self, host: str):
        self.host = host
        self.call = http_transport(host)
        self.proc: Optional[subprocess.Popen] = None

    def up(self) -> bool:
        try:
            self.call("GET", "/api/version", None, 5.0)
            return True
        except OllamaError:
            return False

    def __enter__(self) -> "OllamaServer":
        if self.up():
            log(f"Ollama already running at {self.host}")
            return self
        exe = find_ollama()
        if exe is None:
            raise SystemExit("Ollama is unreachable and no `ollama` executable was found")
        log(f"starting `{exe} serve`")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        for _ in range(120):
            if self.up():
                return self
            time.sleep(0.5)
        self.proc.terminate()
        raise SystemExit("started `ollama serve` but it never answered")

    def __exit__(self, *exc: Any) -> None:
        try:
            for name in self.loaded():
                self.call("POST", "/api/generate", {"model": name, "keep_alive": 0}, 60.0)
        except OllamaError:
            pass
        if self.proc is not None:
            log("stopping the `ollama serve` this script started")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def models(self) -> list[str]:
        return [m["name"] for m in self.call("GET", "/api/tags", None, 20.0).get("models", [])]

    def loaded(self) -> list[str]:
        reply = self.call("GET", "/api/ps", None, 10.0)
        return [m.get("name") or m.get("model", "") for m in reply.get("models", [])]

    def require(self, model: str) -> None:
        have = self.models()
        if model not in have:
            raise SystemExit(f"model {model!r} is not installed in Ollama (have: {have})")

    def unload_all(self) -> None:
        for name in self.loaded():
            self.call("POST", "/api/generate", {"model": name, "keep_alive": 0}, 60.0)
        for _ in range(30):
            if not self.loaded():
                return
            time.sleep(0.5)


class VramMonitor:
    """Polls ``nvidia-smi`` for the GPU's used memory (MiB) in a thread."""

    def __init__(self, every: float = 0.5):
        self.every = every
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.available = shutil.which("nvidia-smi") is not None

    def read(self) -> Optional[float]:
        if not self.available:
            return None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout  # fmt: skip
            return max(float(x) for x in out.split())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def start(self) -> None:
        self.peak = 0.0
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                used = self.read()
                if used is not None:
                    self.peak = max(self.peak, used)
                self._stop.wait(self.every)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.peak


# ---------------------------------------------------------------------- run one system


def run_llm(
    server: OllamaServer,
    model: str,
    items: list[Any],
    domains: dict[str, Any],
    registry: Any,
    host: str,
    full: bool,
    vram: VramMonitor,
) -> dict[str, Any]:
    server.require(model)
    server.unload_all()
    time.sleep(1.0)
    baseline = vram.read()
    vram.start()
    backend = OllamaBackend(model, host=host, keep_alive="10m")
    translator = Translator(backend, registry)
    log(f"{model}: forward translation of {len(items)} items")

    def progress(k: int, n: int, ident: str) -> None:
        if k % 10 == 0 or k == n:
            log(f"{model}: forward {k}/{n} ({ident})")

    records = evaluate_forward(
        lambda it: translator.translate(it.text, domains[it.domain].pack),
        items, domains, registry, progress,
    )  # fmt: skip
    out: dict[str, Any] = {"model": model, "records": records}
    out["forward"] = forward_metrics(records)
    first = records[0]["call_latencies"][0] if records and records[0]["call_latencies"] else None
    warm_items = [r["latency_s"] for r in records[1:]]
    warm_calls = [c for r in records[1:] for c in r["call_latencies"]]
    out["latency"] = {
        "cold_first_call_s": first,
        "warm_item_p50_s": percentile(warm_items, 50),
        "warm_item_p95_s": percentile(warm_items, 95),
        "warm_call_p50_s": percentile(warm_calls, 50),
        "warm_call_p95_s": percentile(warm_calls, 95),
        "max_prompt_tokens": max((r["n_prompt_tokens"] for r in records), default=0),
    }
    if full:
        log(f"{model}: reverse translation (summary + round trip)")
        reverse = ReverseTranslator(translator, keep_alive="10m")
        rows = evaluate_reverse(
            reverse, records, domains, registry,
            lambda k, n, i: log(f"{model}: reverse {k}/{n} ({i})") if k % 10 == 0 or k == n else None,
        )  # fmt: skip
        out["reverse_rows"] = rows
        out["reverse"] = reverse_metrics(rows)
    peak = vram.stop()
    out["vram"] = {
        "baseline_mib": baseline,
        "peak_mib": peak if vram.available else None,
        "model_mib": (peak - baseline) if (baseline is not None and vram.available) else None,
    }
    try:
        backend.unload()
    except OllamaError:
        pass
    return out


# ---------------------------------------------------------------------- report


def pct(x: Optional[float], digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}%"


def pm(mean: Optional[float], hw: Optional[float], digits: int = 2) -> str:
    if mean is None:
        return "n/a"
    return f"{mean:.{digits}f}" + ("" if hw is None else f" ± {hw:.{digits}f}")


def render(report: dict[str, Any]) -> str:
    systems = report["systems"]
    primary = systems[0]
    lines = []
    w = lines.append
    w("# R7a vertical slice: natural language to GoalSpec to command centre to trace to summary")
    w("")
    w(
        f"Generated {report['generated']} on {report['host']['machine']} (Python {report['host']['python']}). "
        f"Command: `{report['command']}`."
    )
    if report["smoke"]:
        w("")
        w(
            f"> **SMOKE RUN.** {report['n_items']} of {report['n_items_full']} instructions, a stratified subsample. "
            "The numbers below check that the pipeline runs; they are not the evaluation, and no verdict is given."
        )
    w("")
    w("## What was run")
    w("")
    w(
        "One translator class (`translator.Translator`) serves two domains. A domain supplies only data: a "
        "`SpecDefinition` in a shared `SpecRegistry`, a system prompt with few-shot examples, an executor that "
        "honours the GoalSpec by wrapping the public squad or plant APIs, and the facts a summary must state."
    )
    w("")
    w("- **Squad** (`squad.mission`): which colours forage, caution (low/normal/high), step limit 50-300, food target 0-40. Non-eater snakes wander; caution sets the wander rate; the target ends the mission. Task score: eater food (capped at the target) plus 0.5 per eater alive at the end, mean over the seeds.")
    w("- **Plant floor** (`plant.diagnose`): byte budget (8/16/51), target false-alarm rate (0.1% to 5%), active nodes. The feature encoder and the calibrated centre are trained for the ordered budget and rate with the silenced nodes' windows zeroed. Task score: macro-F1 over the fault classes, mean over fit seeds.")
    w("")
    c = report["evalset"]
    w(
        f"Evaluation set: {c['n']} instructions ({c['invalid']} invalid or unsatisfiable, {pct(c['invalid'] / c['n'], 0)}). "
        "Gold GoalSpecs come from the generator's own parameters (phrase banks and templates), never from a model. "
        "`indist` uses plain templated phrasings close to the few-shot examples; `heldout` uses terse, verbose, indirect "
        "and number-in-words phrasings, `everyone except X` negations and other unit spellings that appear in no few-shot."
    )
    w("")
    w(f"Seeds: squad missions run on {report['seeds']['squad']} held-out episodes (seeds from 2,000,000,000), plant runs on {report['seeds']['plant']} fit-and-replay repeats. Each model call is retried once with the validation error before a typed failure.")
    w("")
    w("## Verdict")
    w("")
    v = report["verdict"]
    w(f"**{v['label']}** (primary model {primary['model']}).")
    w("")
    w("| Criterion | Target | Measured | Result |")
    w("|---|---|---|---|")
    for r in v["rows"]:
        res = "n/a" if r["pass"] is None else ("PASS" if r["pass"] else ("KILL" if r["kill"] else "not met"))
        w(f"| {r['criterion']} | {r['target']} | {r['value']} | {res} |")
    w("")
    w("## GoalSpec accuracy")
    w("")
    w("Exact match is over the instructions with a gold spec (params equal after sorting unordered arrays); 95% Wilson intervals.")
    w("")
    w("| System | In-distribution | Held-out | Held-out squad | Held-out plant |")
    w("|---|---|---|---|---|")
    for s in systems:
        ex = s["forward"]["exact"]

        def cell(key: str, ex: dict = ex) -> str:
            b = ex.get(key)
            if not b:
                return "n/a"
            return f"{pct(b['exact'])} ({pct(b['exact_ci'][0], 0)}-{pct(b['exact_ci'][1], 0)}, n={b['n']})"

        w(f"| {s['label']} | {cell('indist')} | {cell('heldout')} | {cell('heldout/squad')} | {cell('heldout/plant')} |")
    w("")
    w("Per-field accuracy, held-out split:")
    w("")
    fields = sorted({f for s in systems for f in s["forward"]["exact"].get("heldout", {}).get("field_accuracy", {})})
    w("| System | " + " | ".join(fields) + " |")
    w("|---|" + "---|" * len(fields))
    for s in systems:
        fa = s["forward"]["exact"].get("heldout", {}).get("field_accuracy", {})
        w(f"| {s['label']} | " + " | ".join(pct(fa.get(f)) if f in fa else "-" for f in fields) + " |")
    w("")
    w("Per-field accuracy, in-distribution split:")
    w("")
    w("| System | " + " | ".join(fields) + " |")
    w("|---|" + "---|" * len(fields))
    for s in systems:
        fa = s["forward"]["exact"].get("indist", {}).get("field_accuracy", {})
        w(f"| {s['label']} | " + " | ".join(pct(fa.get(f)) if f in fa else "-" for f in fields) + " |")
    w("")
    w("## Refusals")
    w("")
    w("A refusal is any outcome without a validated GoalSpec (the model refused, or its reply never validated). Gold refusals are the invalid requests.")
    w("")
    w("| System | Precision | Recall | TP / FP / FN | Statuses |")
    w("|---|---|---|---|---|")
    for s in systems:
        r = s["forward"]["refusal"]
        w(f"| {s['label']} | {pct(r['precision'])} | {pct(r['recall'])} | {int(r['tp'])} / {int(r['fp'])} / {int(r['fn'])} | {s['forward']['status_counts']} |")
    w("")
    w("## Task score driven from natural language vs the gold spec")
    w("")
    w("Both runs go through the command centre on the same episodes. An instruction whose translation failed scores 0 on every seed. Mean over instructions per seed, then mean ± 95% t-interval over seeds; the last column is the paired difference (NL minus gold) over seeds.")
    w("")
    w("| System | Split / domain | n | NL | Gold | NL - gold | Within CI |")
    w("|---|---|---|---|---|---|---|")
    for s in systems:
        for key, t in s["forward"]["task"].items():
            if key.startswith("all/") or True:
                w(f"| {s['label']} | {key} | {t['n_items']} | {pm(t['nl_mean'], t['nl_ci'])} | {pm(t['gold_mean'], t['gold_ci'])} | {pm(t['diff_mean'], t['diff_ci'])} | {'yes' if t['within_ci'] else 'no'} |")
    w("")
    if "reverse" in primary:
        rv = primary["reverse"]
        w("## Reverse translator (primary model only)")
        w("")
        w("The summary is written from the executed trace (GoalSpec, envelope headers, outcome), then translated back into a GoalSpec by the forward translator. Round trip: the reconstruction equals the forward spec. Field fidelity: the share of required facts (every parameter plus the outcome score) the summary states without a contradiction, checked by regular expressions against the GoalSpec and the outcome.")
        w("")
        w("| Group | n | Round trip (95% CI) | Field fidelity | All facts in summary | Per fact |")
        w("|---|---|---|---|---|---|")
        for key in ("all", "indist", "heldout", "squad", "plant"):
            b = rv.get(key)
            if not b:
                continue
            per = ", ".join(f"{k} {pct(x, 0)}" for k, x in b["per_fact"].items())
            w(f"| {key} | {b['n']} | {pct(b['round_trip'])} ({pct(b['round_trip_ci'][0], 0)}-{pct(b['round_trip_ci'][1], 0)}) | {pct(b['field_fidelity'])} | {pct(b['all_facts_rate'])} | {per} |")
        w("")
        w(f"Summary latency p50 / p95: {primary['reverse']['summary_latency']['p50']:.2f} / {primary['reverse']['summary_latency']['p95']:.2f} s.")
        w("")
        w("Sample summaries:")
        w("")
        for row in primary["reverse_rows"][:6]:
            w(f"- `{row['id']}` (round trip {'ok' if row['round_trip'] else 'FAILED'}, fidelity {pct(row['fidelity'], 0)}): {row['summary']}")
        w("")
    w("## Latency and memory")
    w("")
    w("Cold is the first call after unloading every model (it includes the model load); warm is every later call. Item latency includes a retry when one happened.")
    w("")
    w("| System | Cold first call | Warm item p50 / p95 | Warm call p50 / p95 | Peak VRAM over baseline | Max prompt tokens |")
    w("|---|---|---|---|---|---|")
    for s in systems:
        lat = s.get("latency")
        if not lat:
            w(f"| {s['label']} | - | - | - | - | - |")
            continue
        vr = s["vram"]
        mem = "n/a" if vr["model_mib"] is None else f"{vr['model_mib'] / 1024:.1f} GB (peak {vr['peak_mib'] / 1024:.1f} GB total, baseline {vr['baseline_mib'] / 1024:.1f} GB)"
        w(f"| {s['label']} | {lat['cold_first_call_s']:.1f} s | {lat['warm_item_p50_s']:.2f} / {lat['warm_item_p95_s']:.2f} s | {lat['warm_call_p50_s']:.2f} / {lat['warm_call_p95_s']:.2f} s | {mem} | {lat['max_prompt_tokens']} |")
    w("")
    w("## Failures of the primary model")
    w("")
    bad = [r for r in primary["records"] if (r["gold"] is not None and not r.get("exact")) or (r["gold"] is None and not r["refused"])]
    w(f"{len(bad)} of {len(primary['records'])} instructions were wrong (a valid request not matched exactly, or an invalid one accepted). First {min(len(bad), 15)}:")
    w("")
    for r in bad[:15]:
        why = " (" + r["reason"][:100] + ")" if r["reason"] else ""
        gold, got = json.dumps(r["gold"]), json.dumps(r["pred"])
        text = r["text"]
        w(f'- `{r["id"]}` [{r["status"]}] "{text}"  gold `{gold}`  got `{got}`{why}')
    w("")
    w("## Limitations")
    w("")
    w("- The evaluation set is synthetic (templates plus hand-written phrase banks); its held-out split changes style, not the label conventions, which the system prompts state. A different author could phrase things differently, and the prompts' conventions (for example `low` caution for 'take risks') are mine.")
    w("- The prompts were adjusted once after a 20-instruction smoke run drawn from this same set (the rate-conversion rule of the plant prompt, and the status-report examples that let the forward translator read a summary). No other tuning was done on the set, but those 20 items are not fully fresh.")
    w("- The keyword baseline was written against the in-distribution phrasing, so its in-distribution score is an upper bound for rules of that kind; its held-out score is what such rules do on new styles.")
    w("- The few-shot examples of the summary pack fix the summary's phrasing (it names the parameters in a fixed pattern), which both helps the fact check and makes the round trip easier than free-form monitoring text would be. Fidelity is checked by regular expressions, not by a human or a second model.")
    w("- Squad non-eaters wander (blind to their own food) rather than hold still, because snakes cannot stop. Plant silencing trains the centre on zeroed windows for the silenced nodes; the trace of a plant task holds only the goal and result envelopes, because the sensor uplinks run on the batched array path.")
    w("- Latency was measured on a machine that may run other jobs; the VRAM peak is the GPU's total used memory over the pre-load baseline and includes other processes that started in between.")
    return "\n".join(lines) + "\n"


def clean(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items() if k != "nl_view"}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/results/r7a_vertical_slice", help="path without extension")
    ap.add_argument("--primary", default=PRIMARY)
    ap.add_argument("--small", default=SMALL, help="second model; 'none' skips it")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--limit", type=int, default=0, help="stratified subsample size (0 = all)")
    ap.add_argument("--squad-seeds", type=int, default=8)
    ap.add_argument("--plant-seeds", type=int, default=5)
    ap.add_argument("--set-seed", type=int, default=7)
    ap.add_argument("--per-split", type=int, default=36, help="valid items per domain and split")
    ap.add_argument("--no-reverse", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    registry = build_registry()
    domains = build_domains(TepData.from_cache(None), args.squad_seeds, args.plant_seeds)
    full_items = build_evalset(args.set_seed, args.per_split)
    items = select_items(full_items, args.limit, args.set_seed) if args.limit else full_items
    smoke = bool(args.limit) and args.limit < len(full_items)
    log(f"{len(items)} instructions ({sum(i.invalid for i in items)} invalid), smoke={smoke}")

    systems: list[dict[str, Any]] = []
    vram = VramMonitor()
    with OllamaServer(args.host) as server:
        models = [(args.primary, True)]
        if args.small.lower() != "none":
            models.append((args.small, False))
        for model, full in models:
            res = run_llm(server, model, items, domains, registry, args.host, full and not args.no_reverse, vram)
            res["label"] = model
            systems.append(res)
    log("keyword baseline")
    kw_records = evaluate_forward(keyword_translator(registry, domains), items, domains, registry)
    systems.append({"model": "keyword/regex", "label": "keyword/regex parser", "records": kw_records, "forward": forward_metrics(kw_records)})

    primary = systems[0]
    vram_gb = None if primary["vram"]["model_mib"] is None else primary["vram"]["model_mib"] / 1024
    report = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "command": "python " + " ".join(sys.argv),
        "host": {"machine": platform.node(), "python": platform.python_version()},
        "smoke": smoke,
        "n_items": len(items),
        "n_items_full": len(full_items),
        "evalset": {"n": len(items), "invalid": sum(i.invalid for i in items)},
        "seeds": {"squad": args.squad_seeds, "plant": args.plant_seeds},
        "systems": systems,
        "verdict": verdict(
            primary["forward"], primary.get("reverse"),
            primary["latency"]["warm_item_p95_s"], vram_gb, smoke,
        ),  # fmt: skip
        "elapsed_s": time.time() - t0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(render(report), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(clean(report), indent=1, allow_nan=False), encoding="utf-8")
    run_dir = os.environ.get("GCA_RUN_DIR")
    if run_dir:
        for ext in (".md", ".json"):
            shutil.copy(out.with_suffix(ext), Path(run_dir) / (out.name + ext))
    log(f"wrote {out}.md and .json in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
