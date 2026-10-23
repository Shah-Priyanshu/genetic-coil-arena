"""Part 1: translator feasibility against Ollama structured outputs.

usage: python p1_run.py MODEL [--limit N] [--think false|omit] [--num-thread N] [--abort-over-s 10]
Writes results/p1_<model>.json
"""
import argparse, json, math, statistics, sys, time, urllib.request, urllib.error
from pathlib import Path

import jsonschema

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from p1_dataset import SYSTEM_PROMPT, FEWSHOT, SCHEMA  # noqa: E402

HOST = "http://localhost:11434"
FIELDS = ["eaters", "region", "avoid", "tempo", "hold"]


def post(path, body, timeout=600):
    req = urllib.request.Request(HOST + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path, timeout=10):
    with urllib.request.urlopen(HOST + path, timeout=timeout) as r:
        return json.loads(r.read())


def build_messages(text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t, g in FEWSHOT:
        msgs.append({"role": "user", "content": t})
        msgs.append({"role": "assistant", "content": json.dumps(g, separators=(",", ":"))})
    msgs.append({"role": "user", "content": text})
    return msgs


def norm(field, v):
    return sorted(v) if field in ("eaters", "avoid") and isinstance(v, list) else v


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--think", default="false", choices=["false", "omit"])
    ap.add_argument("--num-thread", type=int, default=4)
    ap.add_argument("--abort-over-s", type=float, default=0.0,
                    help="after 6 calls, abort if median wall latency exceeds this")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(HERE / "p1_testset.jsonl", encoding="utf-8")]
    if a.limit:
        items = items[: a.limit]
    validator = jsonschema.Draft202012Validator(SCHEMA)

    options = {"temperature": 0, "seed": 0, "num_predict": 200, "num_ctx": 2048,
               "num_thread": a.num_thread}
    recs = []
    first = None
    aborted = None
    for i, it in enumerate(items):
        body = {"model": a.model, "messages": build_messages(it["text"]), "format": SCHEMA,
                "stream": False, "options": options, "keep_alive": "10m"}
        if a.think == "false":
            body["think"] = False
        t0 = time.perf_counter()
        try:
            r = post("/api/chat", body)
            err = None
        except urllib.error.HTTPError as e:
            r, err = {}, f"HTTP {e.code}: {e.read().decode()[:300]}"
        wall = time.perf_counter() - t0
        raw = (r.get("message") or {}).get("content", "")
        parsed, valid, verr = None, False, None
        try:
            parsed = json.loads(raw)
            verrs = list(validator.iter_errors(parsed))
            valid = not verrs
            if verrs:
                verr = verrs[0].message[:160]
        except Exception as e:  # noqa: BLE001
            verr = f"json: {e}"
        gold = it["gold"]
        fm = {f: (parsed is not None and isinstance(parsed, dict) and f in parsed
                  and norm(f, parsed[f]) == norm(f, gold[f])) for f in FIELDS}
        rec = {"id": it["id"], "kind": it["kind"], "heldout": it["heldout"], "text": it["text"],
               "gold": gold, "raw": raw, "parsed": parsed, "valid": valid, "valid_err": verr,
               "field_ok": fm, "exact": all(fm.values()), "wall_s": wall, "error": err,
               "ollama": {k: r.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count",
                                                "prompt_eval_duration", "eval_count", "eval_duration")},
               "thinking": (r.get("message") or {}).get("thinking")}
        recs.append(rec)
        if i == 0:
            try:
                ps = get("/api/ps")
                first = {"wall_s": wall, "load_duration_s": (r.get("load_duration") or 0) / 1e9,
                         "ps": ps}
            except Exception as e:  # noqa: BLE001
                first = {"wall_s": wall, "ps_error": str(e)}
            print(f"[{a.model}] first call wall={wall:.2f}s load={first.get('load_duration_s')}", flush=True)
            if err:
                print("ERR", err, flush=True)
                aborted = f"first call failed: {err}"
                break
        if a.abort_over_s and i == 6:
            med = statistics.median(x["wall_s"] for x in recs[1:])
            print(f"[{a.model}] median of calls 2-7 = {med:.2f}s", flush=True)
            if med > a.abort_over_s:
                aborted = f"median latency {med:.1f}s over {a.abort_over_s}s threshold after 7 calls"
                break
        if (i + 1) % 20 == 0:
            ex = sum(x["exact"] for x in recs)
            print(f"[{a.model}] {i+1}/{len(items)} exact so far {ex}/{len(recs)}", flush=True)

    # unload and verify
    unload_ok = None
    try:
        post("/api/generate", {"model": a.model, "keep_alive": 0}, timeout=60)
        time.sleep(1.5)
        unload_ok = not any(m.get("name") == a.model or m.get("model") == a.model
                            for m in get("/api/ps").get("models", []))
    except Exception as e:  # noqa: BLE001
        unload_ok = f"error: {e}"

    n = len(recs)

    def acc(rs, key):
        return sum(1 for x in rs if (x[key] if key in ("exact", "valid") else x["field_ok"][key])) / len(rs) if rs else None

    summary = {"n": n}
    if n:
        summary["exact"] = acc(recs, "exact")
        summary["exact_ci95"] = wilson(sum(x["exact"] for x in recs), n)
        summary["schema_valid"] = acc(recs, "valid")
        summary["per_field"] = {f: acc(recs, f) for f in FIELDS}
        ho = [x for x in recs if x["heldout"]]
        nh = [x for x in recs if not x["heldout"]]
        summary["heldout_n"] = len(ho)
        summary["heldout_exact"] = acc(ho, "exact") if ho else None
        summary["heldout_exact_ci95"] = wilson(sum(x["exact"] for x in ho), len(ho)) if ho else None
        summary["seen_combo_exact"] = acc(nh, "exact") if nh else None
        for kind in ("templated", "paraphrase"):
            ks = [x for x in recs if x["kind"] == kind]
            summary[f"exact_{kind}"] = acc(ks, "exact") if ks else None
            summary[f"n_{kind}"] = len(ks)
        lat = [x["wall_s"] for x in recs[1:]]
        if lat:
            summary["latency_s"] = {"n": len(lat), "p50": pct(lat, .5), "p95": pct(lat, .95),
                                    "mean": statistics.mean(lat), "max": max(lat), "min": min(lat)}
        eg = [x["ollama"]["eval_count"] for x in recs if x["ollama"].get("eval_count")]
        pg = [x["ollama"]["prompt_eval_count"] for x in recs if x["ollama"].get("prompt_eval_count")]
        summary["mean_output_tokens"] = statistics.mean(eg) if eg else None
        summary["mean_prompt_tokens"] = statistics.mean(pg) if pg else None
        summary["errors"] = sum(1 for x in recs if x["error"])
        summary["thinking_nonempty"] = sum(1 for x in recs if x.get("thinking"))
    out = {"model": a.model, "options": options, "think_param": a.think, "first_call": first,
           "aborted": aborted, "unloaded": unload_ok, "summary": summary, "records": recs}
    fn = HERE / "results" / f"p1_{a.model.replace(':', '_').replace('/', '_')}{a.tag}.json"
    fn.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("records",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
