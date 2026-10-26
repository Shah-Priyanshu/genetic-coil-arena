"""R10 step 3: trained fault-type specialists in the typed team (goal G4c).

Sub-commands (each one is a runner job, see ``docs/goals.md`` row G4c):

    data      fetch (optional) and check the RCAEval cache; write the train/test split
    metric    CPU: leave-one-system-out model selection and training of the metric fault
              specialist on TT + SS/OB-dev; writes metric_model.json and oof_metric.jsonl
    lora      GPU: LoRA on Qwen3-1.7B for one fold (``--fold tt|ss|ob`` holds that system out
              and scores its root-service log views; ``final`` trains on everything and scores
              every service of every test incident)
    teams     CPU: fuse (rule chosen on cross-validated predictions of the training data),
              run BARO, the step-2 metric team and the trained teams on the test split
    report    merge everything into docs/results/r10_step3.{md,json} with the G4c verdict

    python scripts/r10_step3.py data --fetch
    python scripts/r10_step3.py metric --out outputs/r10s3
    python scripts/r10_step3.py lora --fold final --out outputs/r10s3 --smoke
    python scripts/r10_step3.py teams --metric-from outputs/r10s3 --lora-from outputs/r10s3
    python scripts/r10_step3.py report --from "runs:r10s3-*" --out docs/results/r10_step3.md

Test incidents (SS/OB repetitions 2 and 3) are never used to fit or select anything; the split
is checked in ``data`` and again wherever a training set is built. Adapters go to
``$R10S3_STORE`` (default: the run folder), never into git. Under the runner artifacts go to
``$GCA_RUN_DIR``.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import r10_log_triage as s1  # noqa: E402  (log, out_dir, resolve_sources)

log = s1.log


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def find_files(sources: list[str], pattern: str) -> list[Path]:
    out: list[Path] = []
    for src in s1.resolve_sources(sources):
        out += sorted(src.rglob(pattern)) if src.is_dir() else [src]
    return out


def store_dir() -> Path:
    path = Path(
        os.environ.get("R10S3_STORE")
        or os.environ.get("GCA_RUN_DIR")
        or "outputs/r10s3"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def split_files(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    from genetic_coil_arena.experiments import log_triage3 as t3

    train, test = t3.load_split(args.cache_dir, getattr(args, "limit_train", None))
    if not train or not test:
        raise SystemExit(
            "RCAEval cache incomplete (need TT, SS and OB): run "
            "`python scripts/fetch_rcaeval.py --systems SS,OB,TT`"
        )
    if not any(f.parent.name == "TT" for f in train):
        raise SystemExit("no Train Ticket incidents cached: fetch with --systems TT")
    return train, test


# ------------------------------------------------------------------------------ data


def cmd_data(args: argparse.Namespace) -> int:
    from genetic_coil_arena.envs.logtriage import rcaeval
    from genetic_coil_arena.experiments import log_triage3 as t3

    if args.fetch:
        rcaeval.fetch(args.cache_dir, ("SS", "OB", "TT"))
    train, test = split_files(args)
    tk = [t3.file_key(f) for f in train]
    sk = [t3.file_key(f) for f in test]
    t3.assert_no_test_leak(tk, sk)

    def summary(keys: list[t3.Key]) -> dict[str, Any]:
        from collections import Counter

        faults = Counter(rcaeval.parse_case(k[0].split("/")[1])[1] for k in keys)
        return {
            "n": len(keys),
            "by_system_rep": dict(Counter(f"{k[1]}/{k[2]}" for k in keys)),
            "by_fault": dict(faults),
        }

    out = s1.out_dir(args.out)
    result = {
        "train": summary(tk),
        "test": summary(sk),
        "leak_check": "passed: no id in both sets; train = TT (any rep) + SS/OB rep 1; "
        "test = SS/OB reps 2-3",
        "train_ids": sorted(k[0] for k in tk),
        "test_ids": sorted(k[0] for k in sk),
        "commit": commit(),
    }
    write_json(out / "data_summary.json", result)
    log(f"train {result['train']}")
    log(f"test  {result['test']}")
    log(result["leak_check"])
    return 0


# ---------------------------------------------------------------------------- metric


def load_incidents(files: list[Path], what: str) -> list[Any]:
    from genetic_coil_arena.envs.logtriage import rcaeval

    out = []
    started = time.time()
    for k, path in enumerate(files):
        out.append(rcaeval.load_incident(path))
        if (k + 1) % 25 == 0 or k == len(files) - 1:
            log(f"{what}: loaded {k + 1}/{len(files)} ({time.time() - started:.0f}s)")
    return out


def light(inc: Any) -> Any:
    """Drop the line arrays (the metric specialist reads metrics only) to save memory."""
    inc.t = inc.t[:0]
    inc.svc = inc.svc[:0]
    inc.tpl = inc.tpl[:0]
    return inc


def cmd_metric(args: argparse.Namespace) -> int:
    from genetic_coil_arena.envs.logtriage import fault_features as ff
    from genetic_coil_arena.envs.logtriage import heuristics as hz
    from genetic_coil_arena.envs.logtriage.rcaeval import FAULT_TYPES
    from genetic_coil_arena.experiments import log_triage3 as t3

    out = s1.out_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_files, test_files = split_files(args)
    t3.assert_no_test_leak(
        [t3.file_key(f) for f in train_files], [t3.file_key(f) for f in test_files]
    )
    incs = [light(i) for i in load_incidents(train_files, "train")]
    data = t3.metric_dataset(incs)
    log(f"{len(data.y)} training rows, {data.x.shape[1]} features, {t3.describe(incs)}")
    sel = t3.select_metric_model(data, seed=args.seed, log=log)
    chosen = sel["chosen"]
    oof = sel["oof"]
    final = t3._make(chosen, args.seed).fit(data.x, data.y)
    tt = np.asarray(data.groups) == "TT"
    ttonly = t3._make(chosen, args.seed).fit(data.x[tt], data.y[tt])

    # the step-2 heuristic (strongest metric kind) on the same training rows, for reference
    heur = []
    for inc in incs:
        rows = inc.metric_rows(inc.root_service, min_z=0.0, top=1)
        heur.append(
            hz.METRIC_FAULT.get(rows[0][0], "delay") == inc.fault_type
            if rows
            else False
        )
    groups = np.asarray(data.groups)
    per_fault: dict[str, dict[str, Any]] = {}
    for f_i, f in enumerate(FAULT_TYPES):
        sel_f = data.y == f_i
        per_fault[f] = {
            "n": int(sel_f.sum()),
            "oof_accuracy": float(np.mean(oof[sel_f].argmax(axis=1) == f_i)),
            "heuristic_accuracy": float(np.mean(np.asarray(heur)[sel_f])),
        }
    result = {
        "features": ff.feature_names(),
        "grid": sel["table"],
        "chosen": chosen,
        "oof_per_system": sel["table"][sel["chosen_index"]]["per_system"],
        "heuristic_per_system": {
            g: float(np.mean(np.asarray(heur)[groups == g]))
            for g in sorted(set(groups))
        },
        "per_fault": per_fault,
        "n_train": int(len(data.y)),
        "n_ttonly": int(tt.sum()),
        "model": final.to_dict(),
        "model_ttonly": ttonly.to_dict(),
        "commit": commit(),
        "seed": args.seed,
    }
    write_json(out / "metric_model.json", result)
    with open(out / "oof_metric.jsonl", "w", encoding="utf-8") as fh:
        for i, iid in enumerate(data.ids):
            fh.write(
                json.dumps(
                    {
                        "incident": iid,
                        "system": data.groups[i],
                        "fault": FAULT_TYPES[int(data.y[i])],
                        "probs": [round(float(p), 5) for p in oof[i]],
                    }
                )
                + "\n"
            )
    log(
        f"chosen {chosen}: leave-one-system-out accuracy {result['oof_per_system']}; "
        f"metric-kind heuristic {result['heuristic_per_system']}"
    )
    return 0


# ------------------------------------------------------------------------------ lora


def cmd_lora(args: argparse.Namespace) -> int:
    from genetic_coil_arena.backends.hf_peft import HFPeftBackend
    from genetic_coil_arena.envs.logtriage import team as tm
    from genetic_coil_arena.envs.logtriage.rcaeval import FAULT_TYPES
    from genetic_coil_arena.envs.logtriage.views import HFCounter
    from genetic_coil_arena.experiments import log_triage3 as t3

    out = s1.out_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    held = None if args.fold == "final" else args.fold.upper()
    train_files, test_files = split_files(args)
    t3.assert_no_test_leak(
        [t3.file_key(f) for f in train_files], [t3.file_key(f) for f in test_files]
    )
    fit_files = [f for f in train_files if f.parent.name != held]
    held_files = [f for f in train_files if f.parent.name == held]
    if args.smoke:
        fit_files = sorted(fit_files, key=lambda q: q.name)[
            :: max(1, len(fit_files) // 24)
        ]
        held_files = held_files[:: max(1, len(held_files) // 8)]
        test_files = test_files[:: max(1, len(test_files) // 4)][:4]
    log(
        f"fold {args.fold}: {len(fit_files)} fit incidents, {len(held_files)} held-out, "
        f"{len(test_files)} test incidents to score (final only)"
    )
    cfg = t3.LogSettings(view_tokens=args.view_tokens)
    be = HFPeftBackend(
        args.base, memory_fraction=0.85, num_threads=4, dtype="bfloat16",
        local_files_only=not args.online,
    )  # fmt: skip
    be.load()
    counter = HFCounter(be.tokenizer, "qwen3")
    started = time.time()
    examples: list[tuple[str, int]] = []
    from genetic_coil_arena.envs.logtriage import rcaeval

    for k, path in enumerate(fit_files):
        inc = rcaeval.load_incident(path)
        examples += t3.log_training_examples(inc, counter, cfg)
        if (k + 1) % 20 == 0 or k == len(fit_files) - 1:
            log(
                f"examples: {k + 1}/{len(fit_files)} incidents ({time.time() - started:.0f}s)"
            )
    name = f"r10s3{args.fold}"
    t_train = time.time()
    stats = t3.train_log_adapter(
        be, name, examples, rank=args.rank, epochs=args.epochs, batch=args.batch,
        lr=args.lr, seed=args.seed, log=log,
    )  # fmt: skip
    train_s = time.time() - t_train
    path = store_dir() / f"{name}.pt"
    be.save_adapter(name, path)
    log(
        f"trained {stats.steps} steps in {train_s:.0f}s, loss {stats.mean_last_loss:.4f}; saved {path}"
    )

    rows: list[dict[str, Any]] = []
    acc: Optional[float] = None
    sc_started = time.time()
    if held is not None:  # out-of-fold: the held-out system's root-service views
        prompts, meta = [], []
        for path_i in held_files:
            inc = rcaeval.load_incident(path_i)
            s = inc.services.index(inc.root_service)
            onset = tm.metric_message(inc, s)["onset_s"]
            prompts.append(t3.log_prompt_for(inc, s, onset, counter, cfg))
            meta.append((inc.id, inc.system, inc.fault_type, inc.services[s]))
        probs = t3.score_prompts(be, name, prompts, batch=args.score_batch, log=log)
        for (iid, system, fault, svc), p in zip(meta, probs):
            rows.append(
                {"kind": "oof", "fold": args.fold, "incident": iid, "system": system,
                 "fault": fault, "service": svc, "probs": [round(float(x), 5) for x in p]}
            )  # fmt: skip
        acc = float(
            np.mean(
                [FAULT_TYPES[int(np.argmax(r["probs"]))] == r["fault"] for r in rows]
            )
        )
        log(
            f"held-out {held}: fault accuracy {acc:.3f} on {len(rows)} root-service views"
        )
    else:  # final: every service of every test incident
        for k, path_i in enumerate(test_files):
            inc = rcaeval.load_incident(path_i)
            prompts = []
            for s in range(inc.n_services):
                onset = tm.metric_message(inc, s)["onset_s"]
                prompts.append(t3.log_prompt_for(inc, s, onset, counter, cfg))
            probs = t3.score_prompts(be, name, prompts, batch=args.score_batch)
            for s, p in enumerate(probs):
                rows.append(
                    {"kind": "test", "fold": "final", "incident": inc.id, "system": inc.system,
                     "service": inc.services[s], "probs": [round(float(x), 5) for x in p]}
                )  # fmt: skip
            if (k + 1) % 5 == 0 or k == len(test_files) - 1:
                log(
                    f"scored {k + 1}/{len(test_files)} test incidents ({time.time() - sc_started:.0f}s)"
                )
    with open(out / f"logprobs_{args.fold}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    peak = be.peak_memory_mib()
    write_json(
        out / f"lora_{args.fold}.json",
        {
            "fold": args.fold,
            "held_out": held,
            "n_fit_incidents": len(fit_files),
            "n_examples": len(examples),
            "jitters": list(cfg.jitters),
            "view_tokens": cfg.view_tokens,
            "rank": args.rank, "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
            "steps": stats.steps, "train_s": train_s, "final_loss": stats.final_loss,
            "mean_last_loss": stats.mean_last_loss,
            "loss_curve": stats.losses[:: max(1, len(stats.losses) // 100)],
            "peak_mib": peak, "score_s": time.time() - sc_started,
            "heldout_accuracy": acc, "n_heldout": len(rows) if held else 0,
            "adapter_path": str(path), "smoke": bool(args.smoke), "base": args.base,
            "commit": commit(), "python": platform.python_version(),
            "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    )  # fmt: skip
    log(f"done; peak VRAM {peak:.0f} MiB; wall {time.time() - started:.0f}s")
    return 0


# ----------------------------------------------------------------------------- teams


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]
    return rows


def cmd_teams(args: argparse.Namespace) -> int:
    from genetic_coil_arena.envs.logtriage import rcaeval
    from genetic_coil_arena.envs.logtriage import trained as tr
    from genetic_coil_arena.experiments import log_triage as lt
    from genetic_coil_arena.experiments import log_triage2 as t2
    from genetic_coil_arena.experiments import log_triage3 as t3
    from genetic_coil_arena.learning.classifiers import MLPClassifier

    out = s1.out_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mfiles = find_files(args.metric_from, "metric_model.json")
    if not mfiles:
        raise SystemExit("no metric_model.json found in --metric-from")
    model_state = json.loads(mfiles[0].read_text(encoding="utf-8"))
    spec = t3.MetricFaultSpecialist(MLPClassifier.from_dict(model_state["model"]))
    spec_tt = t3.MetricFaultSpecialist(
        MLPClassifier.from_dict(model_state["model_ttonly"]), "trained-tt"
    )
    oof_m = {
        r["incident"]: r
        for r in read_rows(find_files(args.metric_from, "oof_metric.jsonl"))
    }
    lora_rows = read_rows(find_files(args.lora_from, "logprobs_*.jsonl"))
    oof_l = {r["incident"]: r for r in lora_rows if r["kind"] == "oof"}
    test_l: dict[tuple[str, str], list[float]] = {
        (r["incident"], r["service"]): r["probs"]
        for r in lora_rows
        if r["kind"] == "test"
    }

    # ---- fusion rule: chosen on cross-validated predictions of the training data only
    ids = sorted(set(oof_m) & set(oof_l))
    fusion: dict[str, Any] = {"n_oof": len(ids)}
    rule, stack = "metric", None
    if ids:
        faults = list(rcaeval.FAULT_TYPES)
        pm = np.array([oof_m[i]["probs"] for i in ids])
        pl = np.array([oof_l[i]["probs"] for i in ids])
        y = np.array([faults.index(oof_m[i]["fault"]) for i in ids])
        choice = t3.select_fusion(pm, pl, y, [oof_m[i]["system"] for i in ids])
        rule, stack = choice.rule, choice.stack
        fusion.update(
            chosen=rule,
            table=choice.table,
            n_by_system={
                s: sum(oof_m[i]["system"] == s for i in ids)
                for s in sorted({oof_m[i]["system"] for i in ids})
            },
        )
        log(f"fusion rule {rule} (n={len(ids)}): {choice.table[rule]}")
    else:
        log("no out-of-fold log predictions: the log and fused teams are skipped")
    fusion["stack"] = stack
    write_json(out / "fusion.json", fusion)
    has_logs = bool(test_l)

    # ---- the teams on the test split
    _, test_files = split_files(args)
    if args.limit:
        test_files = test_files[:: max(1, len(test_files) // args.limit)][: args.limit]
    cfg = t2.Settings2()
    per: dict[str, dict[str, Any]] = {}
    for k, path in enumerate(test_files):
        inc = rcaeval.load_incident(path)
        assert t3.role_of(inc.system, inc.rep) == "test", inc.id
        mm_plain = t2.metric_messages(inc)
        mm = spec.messages(inc)
        mm_tt = spec_tt.messages(inc)
        lm = None
        if has_logs:
            if all((inc.id, n) in test_l for n in inc.services):
                lm = [
                    tr.trained_log_message(n, test_l[(inc.id, n)]) for n in inc.services
                ]
            else:
                log(f"{inc.id}: no log reports; log teams skipped for it")
        per[inc.id] = {"inc": inc, "plain": mm_plain, "mm": mm, "tt": mm_tt, "lm": lm}
        if (k + 1) % 20 == 0 or k == len(test_files) - 1:
            log(f"messages: {k + 1}/{len(test_files)}")
    plain_pool = t2.build_pool(
        {i: (v["inc"].services, v["plain"], None) for i, v in per.items()}
    )
    pool_m = t3.build_pool3(
        {i: (v["inc"].services, v["mm"], None) for i, v in per.items()}
    )
    pool_tt = t3.build_pool3(
        {i: (v["inc"].services, v["tt"], None) for i, v in per.items()}
    )
    pool_l = t3.build_pool3(
        {
            i: (v["inc"].services, t3.strip_fault_reports(v["mm"]), v["lm"])
            for i, v in per.items()
            if v["lm"]
        }
    )
    pool_b = t3.build_pool3(
        {i: (v["inc"].services, v["mm"], v["lm"]) for i, v in per.items() if v["lm"]}
    )
    rows: list[dict[str, Any]] = []
    for k, v in enumerate(per.values()):
        inc = v["inc"]
        recs: list[lt.Record] = [t2.run_baro(inc)]
        recs += t2.team_records(
            inc, "team_metric_rule", v["plain"], None, "none", None, None, cfg,
            pool=plain_pool, ablate=not args.no_ablate,
        )  # fmt: skip
        recs += t3.team3_records(
            inc,
            "team_trained_metric",
            v["mm"],
            None,
            "metric",
            None,
            pool_m,
            not args.no_ablate,
        )
        recs += t3.team3_records(
            inc,
            "team_trained_metric_ttonly",
            v["tt"],
            None,
            "metric",
            None,
            pool_tt,
            not args.no_ablate,
        )
        if v["lm"]:
            recs += t3.team3_records(
                inc, "team_trained_log", t3.strip_fault_reports(v["mm"]), v["lm"], "log", None,
                pool_l, not args.no_ablate,
            )  # fmt: skip
            recs += t3.team3_records(
                inc,
                "team_trained_both",
                v["mm"],
                v["lm"],
                rule,
                stack,
                pool_b,
                not args.no_ablate,
            )
        rows += [r.to_dict() for r in recs]
        if (k + 1) % 20 == 0 or k == len(per) - 1:
            log(f"teams: {k + 1}/{len(per)}")
    lt.append_jsonl(out / "results.jsonl", rows)
    write_json(
        out / "summary_teams.json",
        {
            "n_incidents": len(per),
            "fusion": rule,
            "has_logs": has_logs,
            "commit": commit(),
            "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    )
    log(f"records in {out / 'results.jsonl'}")
    return 0


# ---------------------------------------------------------------------------- report


def cmd_report(args: argparse.Namespace) -> int:
    from genetic_coil_arena.experiments import log_triage as lt
    from genetic_coil_arena.experiments import log_triage3_report as rp3
    from genetic_coil_arena.experiments import log_triage_report as rp

    results = find_files(args.sources, "results*.jsonl")
    if not results:
        raise SystemExit("no results files found")
    records: list[dict[str, Any]] = []
    for p in results:
        records += lt.read_jsonl(p)

    def first_json(pattern: str) -> Optional[dict[str, Any]]:
        files = find_files(args.sources, pattern)
        return json.loads(files[0].read_text(encoding="utf-8")) if files else None

    extras: dict[str, Any] = {
        "metric": first_json("metric_model.json"),
        "fusion": first_json("fusion.json"),
        "data": first_json("data_summary.json"),
        "lora": {
            p.stem.replace("lora_", ""): json.loads(p.read_text(encoding="utf-8"))
            for p in find_files(args.sources, "lora_*.json")
        },
    }
    if extras["metric"]:
        for key in ("model", "model_ttonly"):
            extras["metric"].pop(key, None)
    out = Path(args.out)
    report, payload = rp3.build_report(
        records, extras, analysis=rp.carried_analysis(out), note=args.note
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(report)
    log(f"wrote {out} and {out.with_suffix('.json')} from {len(results)} file(s)")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", default=None)
        p.add_argument("--cache-dir", type=Path, default=None)
        p.add_argument("--limit-train", type=int, default=None)
        p.add_argument("--seed", type=int, default=0)

    d = sub.add_parser("data")
    common(d)
    d.add_argument(
        "--fetch", action="store_true", help="download and convert missing systems"
    )
    m = sub.add_parser("metric")
    common(m)
    lo = sub.add_parser("lora")
    common(lo)
    lo.add_argument("--fold", required=True, choices=["tt", "ss", "ob", "final"])
    lo.add_argument("--base", default="Qwen/Qwen3-1.7B")
    lo.add_argument("--online", action="store_true")
    lo.add_argument("--rank", type=int, default=8)
    lo.add_argument("--epochs", type=int, default=2)
    lo.add_argument("--batch", type=int, default=4)
    lo.add_argument("--lr", type=float, default=2e-4)
    lo.add_argument("--view-tokens", type=int, default=1500)
    lo.add_argument("--score-batch", type=int, default=4)
    lo.add_argument(
        "--smoke", action="store_true", help="a handful of incidents (not a result)"
    )
    te = sub.add_parser("teams")
    common(te)
    te.add_argument("--metric-from", nargs="+", required=True)
    te.add_argument("--lora-from", nargs="*", default=[])
    te.add_argument("--limit", type=int, default=None)
    te.add_argument("--no-ablate", action="store_true")
    rep = sub.add_parser("report")
    rep.add_argument("--from", dest="sources", nargs="+", required=True)
    rep.add_argument("--out", default=str(REPO / "docs" / "results" / "r10_step3.md"))
    rep.add_argument("--note", default=None)
    args = parser.parse_args(argv)
    return {
        "data": cmd_data,
        "metric": cmd_metric,
        "lora": cmd_lora,
        "teams": cmd_teams,
        "report": cmd_report,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
