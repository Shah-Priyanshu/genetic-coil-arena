"""R7a step 2: evaluation glue for the sealed-set comparison of four translators.

The round trip needs a summary writer and a translator that reads the summary. The
distilled model cannot write summaries, so the writer is fixed (the 14B model) and
reads each translator's *own* executed trace; the translator under test then re-reads the
summary. That splits the step-1 ``evaluate_reverse`` in two: :func:`write_summaries`
(writer model resident) and :func:`reconstruct_rows` (translator under test resident).

Records are JSON-safe (the trace view is dropped when saving and rebuilt by
:func:`rehydrate`, which re-executes the translated specs through the cache).
"""

import glob
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from ..protocol.goalspec import SpecRegistry
from ..translator import (
    ReverseTranslator,
    TranslationPack,
    TranslationResult,
    exact_match,
    fact_fidelity,
)
from .r7a_domains import Domain
from .r7a_policy import policy_packs
from .r7a_report import _cached_execute, _num


def lora_pack(domain: Domain) -> TranslationPack:
    """The pack of the distilled model: the spec name is its whole system prompt.

    The distilled model has the task and reply format in its weights; only the defaults
    and the questions (code-side policy) are needed here.
    """
    policy, _ = policy_packs(
        domain.name
    )  # defaults and questions live in the policy packs
    return TranslationPack(
        policy.spec_name,
        policy.spec_name,
        (),
        policy.version,
        defaults=policy.defaults,
        questions=policy.questions,
    )


def with_lora_packs(domains: dict[str, Domain]) -> dict[str, Domain]:
    """The domains with the distilled model's packs (forward and report alike)."""
    from dataclasses import replace

    return {
        name: replace(d, pack=lora_pack(d), report_pack=lora_pack(d))
        for name, d in domains.items()
    }


def strip_views(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records without the (non-JSON) trace views."""
    return [{k: v for k, v in r.items() if k != "nl_view"} for r in records]


def rehydrate(
    records: Sequence[dict[str, Any]],
    domains: dict[str, Domain],
    registry: SpecRegistry,
    cache: dict[str, Any],
) -> list[dict[str, Any]]:
    """Put the trace view back on saved records whose translation produced a spec."""
    out = []
    for rec in records:
        rec = dict(rec)
        if rec.get("pred") is not None and rec.get("gold") is not None:
            domain = domains[rec["domain"]]
            spec = registry.make(domain.spec_name, rec["pred"])
            run = _cached_execute(cache, domain, registry, spec, rec["text"])
            rec["nl_view"] = run.view
        out.append(rec)
    return out


# ---------------------------------------------------------------------- round trip


def write_summaries(
    writer: ReverseTranslator,
    records: Sequence[dict[str, Any]],
    domains: dict[str, Domain],
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, dict[str, Any]]:
    """Summary text and latency for every record that has a trace view."""
    todo = [r for r in records if r.get("nl_view") is not None]
    out = {}
    for k, rec in enumerate(todo):
        domain = domains[rec["domain"]]
        res = writer.summarise(rec["nl_view"], domain.summary_pack)
        out[rec["id"]] = {
            "summary": res.text,
            "latency_s": res.latency_s,
            "error": res.error,
        }
        if progress:
            progress(k + 1, len(todo), rec["id"])
    return out


def reconstruct_rows(
    translate: Callable[[str, TranslationPack], TranslationResult],
    records: Sequence[dict[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    domains: dict[str, Domain],
    registry: SpecRegistry,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict[str, Any]]:
    """Re-translate each summary and compare with the forward spec; check the facts.

    ``translate(summary, report_pack)`` is the translator under test. A missing or empty
    summary counts as a failed round trip.
    """
    todo = [r for r in records if r.get("nl_view") is not None]
    rows = []
    for k, rec in enumerate(todo):
        domain = domains[rec["domain"]]
        schema = registry.get(domain.spec_name).params_schema
        view = rec["nl_view"]
        summ = summaries.get(rec["id"], {})
        text = (summ.get("summary") or "").strip()
        recon = translate(text, domain.report_pack) if text else None
        fid = fact_fidelity(text, domain.facts(view)) if text else None
        match = bool(
            recon is not None
            and recon.spec is not None
            and recon.spec.spec == view.goal.spec
            and exact_match(recon.spec.params, view.goal.params, schema)
        )
        rows.append(
            {
                "id": rec["id"],
                "domain": rec["domain"],
                "split": rec["split"],
                "summary": text,
                "summary_latency_s": float(summ.get("latency_s") or 0.0),
                "facts": {} if fid is None else fid["facts"],
                "fidelity": 0.0 if fid is None else _num(fid["fidelity"]),
                "all_facts": False if fid is None else bool(fid["all"]),
                "recon_status": None if recon is None else recon.status,
                "recon_params": None if recon is None else recon.params,
                "recon_latency_s": 0.0 if recon is None else recon.latency_s,
                "round_trip": match,
            }
        )
        if progress:
            progress(k + 1, len(todo), rec["id"])
    return rows


# ---------------------------------------------------------------------- jobs' files


def resolve_runs(tokens: Sequence[str], runs_root: Optional[Path] = None) -> list[Path]:
    """Files, folders, globs and ``runs:<pattern>`` tokens to existing paths.

    ``runs:<pattern>`` means the ``artifacts`` folders of sibling jobs: the runner gives
    each job ``$GCA_RUN_DIR = <results>/runs/<job id>/artifacts``, so the siblings are
    ``<results>/runs/<pattern>/artifacts`` (the same layout ``plant_learned_codes`` uses).
    """
    out: list[Path] = []
    for tok in tokens:
        if tok.startswith("runs:"):
            if runs_root is None:
                run_dir = os.environ.get("GCA_RUN_DIR")
                if not run_dir:
                    raise SystemExit("'runs:' needs GCA_RUN_DIR (the runner sets it)")
                runs_root = Path(run_dir).resolve().parents[1]
            out += [
                j / "artifacts" for j in sorted(runs_root.glob(tok[len("runs:") :]))
            ]
        elif any(c in tok for c in "*?["):
            out += [Path(p) for p in sorted(glob.glob(tok))]
        else:
            out.append(Path(tok))
    return [p for p in out if p.exists()]


def find_file(sources: Sequence[Path], name: str) -> Path:
    """The first ``name`` found directly in any of the source folders."""
    for src in sources:
        cand = src / name if src.is_dir() else src
        if cand.name == name and cand.exists():
            return cand
    raise FileNotFoundError(f"{name} not found in {[str(s) for s in sources]}")


def select_best_run(dev_files: Sequence[Path]) -> dict[str, Any]:
    """The training run with the best dev score (ties: fewer trained steps, then name)."""
    runs = []
    for path in dev_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        runs.append(
            (data["dev"]["score"], -data["config"]["steps"], data["name"], data)
        )
    if not runs:
        raise FileNotFoundError("no dev.json found among the training runs")
    best = max(runs, key=lambda x: (x[0], x[1], x[2]))
    return best[3]
