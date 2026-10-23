import json, glob
from pathlib import Path
HERE = Path(__file__).parent
rows = []
for fn in sorted(glob.glob(str(HERE / "results" / "p1_*.json"))):
    d = json.load(open(fn, encoding="utf-8"))
    if "model" not in d:
        continue
    s = d["summary"]; fc = d.get("first_call") or {}
    ps = ((fc.get("ps") or {}).get("models") or [{}])[0]
    if not s.get("n"):
        print(d["model"], "no results", d.get("aborted")); continue
    pf = s["per_field"]; lat = s.get("latency_s", {})
    print(f"\n## {d['model']}  n={s['n']} aborted={d.get('aborted')} unloaded={d.get('unloaded')}")
    print(f"exact {s['exact']:.3f} CI{s['exact_ci95']}  valid {s['schema_valid']:.3f}  heldout({s['heldout_n']}) {s['heldout_exact']} CI{s['heldout_exact_ci95']}  seen {s['seen_combo_exact']}")
    print(f"templated {s['exact_templated']} (n={s['n_templated']}) paraphrase {s['exact_paraphrase']} (n={s['n_paraphrase']})")
    print("fields", {k: round(v, 3) for k, v in pf.items()})
    print(f"first call wall {fc.get('wall_s'):.1f}s load {fc.get('load_duration_s'):.1f}s; ps size {ps.get('size')} size_vram {ps.get('size_vram')}")
    print(f"latency n={lat.get('n')} p50 {lat.get('p50'):.3f} p95 {lat.get('p95'):.3f} mean {lat.get('mean'):.3f} max {lat.get('max'):.3f}; out_tok {s['mean_output_tokens']:.1f} prompt_tok {s['mean_prompt_tokens']:.0f} errors {s['errors']} think_nonempty {s['thinking_nonempty']}")
    # error taxonomy
    bad = [r for r in d["records"] if not r["exact"]]
    from collections import Counter
    c = Counter()
    for r in bad:
        for f, ok in r["field_ok"].items():
            if not ok:
                c[f] += 1
    print("wrong fields among", len(bad), "misses:", dict(c))
