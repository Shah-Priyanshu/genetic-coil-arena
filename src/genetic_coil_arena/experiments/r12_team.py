"""R12 (goal G9): the log-novelty specialist joins the G4c team, behind the same command centre.

:class:`JoinedTeam` adds the newcomer to an existing :class:`~.r11_team.IncidentTeam` without
touching it: the two new goals are registered in the team's spec registry, one
``novelty:<service>`` specialist per service and the joined centre are registered with the same
:class:`CommandCentre` and router, and the old specialists, the old centre and the old rule keep
answering ``rca.decide_trained`` exactly as before. :func:`old_state_hash` fingerprints
everything the old team is made of (the fault specialist's weights, the sources of its handlers,
its specs and its specialists); :meth:`JoinedTeam.assert_frozen` recomputes it after the join and
after every run.

:class:`JoinedExecutor` is the ``rca.investigate`` handler of the grown desk (G10): the same
resolution, caching and public result as :class:`~.r11_team.IncidentExecutor`, with the joined
team behind it and a task score that is root top-1 for RE3 incidents (their fault ids are
service-specific, so there is no joint accuracy) and joint accuracy for RE2 incidents.

:func:`run_incident` and :func:`select_rule` are the G9 experiment: per incident the old team and
the joined team (every candidate rule on dev, the pinned one on test) plus the non-LLM baselines
of G4d, as JSON-safe records that ``r12_report`` turns into the verdict.
"""

import hashlib
import inspect
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..envs.logtriage import novelty as nv
from ..envs.logtriage import trained as tr
from ..envs.logtriage.catalogue import CatalogueEntry, IncidentCatalogue, utc_text
from ..envs.logtriage.incident import Incident
from ..envs.logtriage.rcaeval import FAULT_TYPES
from ..runtime.router import RoutingRule
from ..runtime.specialist import FunctionSpecialist
from . import log_triage4 as t4
from .r11_team import IncidentExecutor, IncidentTeam, light

JOINED_CENTRE = "rca-centre-joined"
NOVELTY_PREFIX = "novelty:"
# The rule the pooled dev split picked (see ``select_rule``); pinned before the test run.
# ``scripts/r12_growth.py team`` fails if a rerun of the dev selection disagrees.
CHOSEN_RULE: Optional[str] = "gate45_cap50"


# ------------------------------------------------------------------------ the frozen team


def old_state_hash(team: IncidentTeam) -> str:
    """SHA-256 over what the old team is: weights, handler sources, specs, specialists.

    Newcomers (``novelty:*`` and the joined centre) and the specs they add are excluded, so
    the hash is the same before and after the join if nothing old was touched.
    """
    new_specs = {nv.SPEC_LOG_NOVELTY, nv.SPEC_DECIDE_JOINED}
    specs = [
        [d.name, d.version, d.params_schema, d.result_schema]
        for d in sorted(team.specs.definitions(), key=lambda d: d.key)
        if d.name not in new_specs
    ]
    specialists = [
        [r.capabilities.name, list(r.capabilities.accepts)]
        for r in team.cc.registry.records()
        if not r.capabilities.name.startswith(NOVELTY_PREFIX)
        and r.capabilities.name != JOINED_CENTRE
    ]
    router = [
        [rule.spec, rule.specialist]
        for rule in team.router.rules
        if not rule.specialist.startswith(NOVELTY_PREFIX)
        and rule.specialist != JOINED_CENTRE
    ]
    payload = {
        "model": team.model.model.to_dict(),
        "decide_source": inspect.getsource(IncidentTeam._decide),
        "metric_source": inspect.getsource(IncidentTeam._metric),
        "specs": specs,
        "specialists": specialists,
        "router": router,
    }
    text = json.dumps(payload, sort_keys=True, default=_jsonable)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot hash {type(value).__name__}")


def prepare(inc: Incident) -> Incident:
    """Cache what the newcomer and the baselines read from the lines, then drop the lines."""
    inc.stats()
    nv.novelty_message(inc, 0)
    return light(inc)


class JoinedTeam:
    """The G4c team plus the log-novelty newcomer and the joined centre.

    Args:
        team: The old team (its command centre receives the newcomer)
        services: Services to register a novelty specialist for (union over the systems)
        rule: The joined centre's fusion rule for :meth:`triage` (one of :data:`nv.ALL_RULES`)
    """

    def __init__(
        self, team: IncidentTeam, services: Sequence[str], rule: str = nv.OLD_RULE
    ):
        if rule not in self._all_rules():
            raise ValueError(f"unknown rule {rule!r}")
        self.team = team
        self.rule = rule
        self.frozen = old_state_hash(team)
        for definition in self._definitions():
            team.specs.register(definition)
        for name in sorted(set(services)):
            team.cc.register(
                FunctionSpecialist(
                    f"{NOVELTY_PREFIX}{name}",
                    {nv.SPEC_LOG_NOVELTY: self._novelty},
                    description=f"log-novelty specialist of {name}",
                )
            )
            team.router.add_rule(
                RoutingRule(
                    nv.SPEC_LOG_NOVELTY,
                    f"{NOVELTY_PREFIX}{name}",
                    when=lambda g, s=name: g.params["service"] == s,
                )
            )
        team.cc.register(
            FunctionSpecialist(
                JOINED_CENTRE,
                {nv.SPEC_DECIDE_JOINED: self._decide},
                description="joined rule centre (metric + log novelty)",
            )
        )
        team.router.add_rule(RoutingRule(nv.SPEC_DECIDE_JOINED, JOINED_CENTRE))
        self.assert_frozen()

    # hooks that G9b's ``JoinedTeamV2`` overrides (newcomer v2, its rules and centre)

    @staticmethod
    def _all_rules() -> Sequence[str]:
        return nv.ALL_RULES

    @staticmethod
    def _definitions() -> list[Any]:
        return nv.joined_definitions()

    @staticmethod
    def _ranking(
        rule: str, mm: Sequence[Mapping[str, Any]], nn: Sequence[Mapping[str, Any]]
    ) -> tuple[list[int], bool]:
        return nv.joined_ranking(rule, mm, nn)

    def assert_frozen(self) -> None:
        """Raise if anything of the old team changed since before the join."""
        now = old_state_hash(self.team)
        if now != self.frozen:
            raise RuntimeError(
                "the old team changed: state hash differs from before the join"
            )

    # handlers: the only place that touches incident data

    def _novelty(self, params: dict[str, Any]) -> dict[str, Any]:
        inc = self.team.incidents[params["incident"]]
        return nv.novelty_message(inc, inc.services.index(params["service"]))

    def _decide(self, params: dict[str, Any]) -> dict[str, Any]:
        ranking, fault, _ = nv.rule_decide_joined(
            params["candidates"],
            params["metric_reports"],
            params["novelty_reports"],
            params["rule"],
        )
        return {"ranking": ranking, "fault_type": fault}

    def triage_rules(
        self,
        inc: Incident,
        rules: Sequence[str],
        exclude: Sequence[str] = (),
        trace_id: Optional[str] = None,
    ) -> dict[str, dict[str, Any]]:
        """Run the specialists once and the joined centre once per rule.

        Returns ``rule -> {ranking, fault_type, fault_prob, onset_s, used_logs}``; the
        keys ``n_envelopes`` and ``wall_s`` are those of the whole run.
        """
        start = time.perf_counter()
        team = self.team
        team.incidents[inc.id] = inc
        services = [s for s in inc.services if s not in set(exclude)]
        if len(services) < 2:
            raise ValueError("a decision needs at least two candidate services")
        tid = trace_id or inc.id
        make = team.specs.make
        mm = team._run_all(
            [
                make(tr.SPEC_METRIC_FAULT, {"incident": inc.id, "service": s})
                for s in services
            ],
            tid,
        )
        nn = team._run_all(
            [
                make(nv.SPEC_LOG_NOVELTY, {"incident": inc.id, "service": s})
                for s in services
            ],
            tid,
        )
        decisions = team._run_all(
            [
                make(
                    nv.SPEC_DECIDE_JOINED,
                    {
                        "incident": inc.id,
                        "candidates": services,
                        "metric_reports": mm,
                        "novelty_reports": nn,
                        "rule": rule,
                    },
                )
                for rule in rules
            ],
            tid,
        )
        self.assert_frozen()
        wall = time.perf_counter() - start
        n_env = len(team.trace.envelopes(tid))
        out = {}
        for rule, decision in zip(rules, decisions):
            ranking = list(decision["ranking"])
            top = next(m for m in mm if m["service"] == ranking[0])
            fault = decision["fault_type"]
            out[rule] = {
                "ranking": ranking,
                "fault_type": fault,
                "fault_prob": float(top["fault_probs"][FAULT_TYPES.index(fault)]),
                "onset_s": top["onset_s"],
                "used_logs": self._ranking(rule, mm, nn)[1],
                "n_envelopes": n_env,
                "wall_s": wall,
            }
        return out

    def triage(
        self, inc: Incident, exclude: Sequence[str] = (), trace_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Same interface and result as :meth:`IncidentTeam.triage`, under ``self.rule``."""
        return self.triage_rules(inc, [self.rule], exclude, trace_id)[self.rule]

    def close(self) -> None:
        self.team.close()


# ------------------------------------------------------------------------ the desk handler


class JoinedExecutor(IncidentExecutor):
    """``rca.investigate`` over the grown catalogue, answered by the joined team.

    The score of a result is root top-1 for an RE3 incident and joint accuracy for an RE2
    one; ``top1`` is root top-1 for both.
    """

    def __init__(
        self,
        catalogue: IncidentCatalogue,
        team: JoinedTeam,
        loader: Optional[Callable[[CatalogueEntry], Incident]] = None,
    ):
        super().__init__(catalogue, team, loader)  # type: ignore[arg-type]

    def _default_loader(self, entry: CatalogueEntry) -> Incident:
        from ..envs.logtriage import rcaeval

        return prepare(rcaeval.load_incident(self.catalogue.files[entry.ticket]))

    def _run(
        self, entry: CatalogueEntry, params: Mapping[str, Any], exclude: list[str]
    ) -> dict[str, Any]:
        inc = self.incident(entry)
        res = self.team.triage(inc, exclude, trace_id=f"{entry.ticket}-{self.runs}")
        onset = res["onset_s"]
        public: dict[str, Any] = {
            "ticket": entry.ticket,
            "system": entry.system,
            "suspects": res["ranking"][: int(params["top_k"])],
            "onset_utc": (
                None if onset is None else utc_text(entry.inject_time + float(onset))
            ),
            "excluded_services": exclude,
        }
        if params["want_fault_type"]:
            public["fault_type"] = res["fault_type"]
            public["fault_probability"] = round(res["fault_prob"], 2)
        top1 = float(res["ranking"][0] == inc.root_service)
        if entry.code.startswith("RE3"):
            score = top1
        else:
            score = float(top1 and res["fault_type"] == inc.fault_type)
        # the label enters only these scores, never ``public``
        return {
            "score": score,
            "top1": top1,
            "per_seed": [score],
            "wall_s": round(float(res["wall_s"]), 4),
            "n_envelopes": int(res["n_envelopes"]),
            "public": public,
        }

    def evaluate_all(self) -> dict[str, Any]:
        """Top-1 (all) and joint accuracy (RE2) of the joined team on every catalogued incident."""
        rows = []
        for entry in self.catalogue.entries:
            res = self(
                {
                    "system": entry.system,
                    "ticket": entry.ticket,
                    "top_k": 3,
                    "want_fault_type": True,
                    "exclude_services": [],
                }
            )
            rows.append((entry.code.startswith("RE3"), res["top1"], res["score"]))
        re2 = [r for r in rows if not r[0]]
        re3 = [r for r in rows if r[0]]

        def mean(xs: list[float]) -> Optional[float]:
            return float(np.mean(xs)) if xs else None

        return {
            "n": len(rows),
            "re2_top1": mean([r[1] for r in re2]),
            "re2_joint": mean([r[2] for r in re2]),
            "re3_top1": mean([r[1] for r in re3]),
        }


# ------------------------------------------------------------------------ the G9 experiment

TOP_N = 3


def dataset_of(inc: Incident) -> str:
    return "re3" if inc.system.startswith("RE3") else "re2"


def split_of(inc: Incident) -> str:
    """``dev`` (repetition 1) or ``test`` (the other repetitions) of either dataset."""
    return "dev" if inc.rep in (1,) else "test"


def load_incident_files(
    cache_dir: Optional[Path] = None,
    splits: Sequence[str] = ("dev", "test"),
    datasets: Sequence[str] = ("re2", "re3"),
) -> list[Path]:
    """Converted incident files of the requested datasets and splits, sorted by id."""
    from ..envs.logtriage import rcaeval

    systems: list[str] = []
    if "re2" in datasets:
        systems += ["SS", "OB"]
    if "re3" in datasets:
        systems += list(rcaeval.RE3_SYSTEMS)
    files = rcaeval.list_incidents(cache_dir, tuple(systems))
    out = []
    for f in files:
        rep = int(f.stem.rsplit("__", 1)[1])
        if ("dev" if rep == 1 else "test") in splits:
            out.append(f)
    return sorted(out, key=lambda p: (p.parent.name, p.name))


def _top(ranking: Sequence[str], n: int = TOP_N) -> list[str]:
    return list(ranking[:n])


def run_incident(
    team: IncidentTeam,
    joined: JoinedTeam,
    inc: Incident,
    rules: Sequence[str],
) -> dict[str, Any]:
    """One incident through the old team, the joined team and the non-LLM baselines.

    ``inc`` must come from :func:`prepare`. The label is read only here, for the record.
    """
    re3 = dataset_of(inc) == "re3"
    old = team.triage(inc, trace_id=f"{inc.id}-old")
    new = joined.triage_rules(inc, rules, trace_id=f"{inc.id}-joined")
    root = inc.root_service

    def entry(res: Mapping[str, Any]) -> dict[str, Any]:
        top1 = bool(res["ranking"][0] == root)
        return {
            "top3": _top(res["ranking"]),
            "top1": top1,
            "in_top3": root in _top(res["ranking"]),
            "joint": (
                None if re3 else bool(top1 and res["fault_type"] == inc.fault_type)
            ),
            "fault_type": res["fault_type"],
        }

    parts = {p: t4.part_scores(inc, p) for p in ("baro_metrics", "new_ext")}
    fusion = t4.fused_scores(parts, {"baro_metrics": 0.5, "new_ext": 0.5})
    baselines = {}
    for name, scores in (
        ("baro", parts["baro_metrics"]),
        ("new_ext", parts["new_ext"]),
        ("re3_fusion", fusion),
    ):
        order = t4.tie_ranking(scores, inc, name)
        names = [inc.services[i] for i in order]
        baselines[name] = {
            "top3": _top(names),
            "top1": names[0] == root,
            "in_top3": root in _top(names),
        }
    rec: dict[str, Any] = {
        "id": inc.id,
        "system": inc.system,
        "dataset": "re3" if re3 else "re2",
        "split": split_of(inc),
        "rep": inc.rep,
        "root": root,
        "fault": None if re3 else inc.fault_type,
        "n_services": inc.n_services,
        "old": entry(old),
        "rules": {},
        "baselines": baselines,
        "n_envelopes": new[rules[0]]["n_envelopes"],
    }
    for rule in rules:
        e = entry(new[rule])
        e["used_logs"] = new[rule]["used_logs"]
        e["changed_top1"] = new[rule]["ranking"][0] != old["ranking"][0]
        rec["rules"][rule] = e
    return rec


def select_rule(
    dev_records: Sequence[Mapping[str, Any]],
    candidates: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """The candidate with the best pooled dev top-1 (ties: the earlier rule), plus the table."""
    table = []
    for rule in candidates or nv.RULES:
        rows = [r["rules"][rule] for r in dev_records]
        by = {
            ds: [r["rules"][rule]["top1"] for r in dev_records if r["dataset"] == ds]
            for ds in ("re2", "re3")
        }
        table.append(
            {
                "rule": rule,
                "top1": float(np.mean([x["top1"] for x in rows])) if rows else None,
                "re2_top1": float(np.mean(by["re2"])) if by["re2"] else None,
                "re3_top1": float(np.mean(by["re3"])) if by["re3"] else None,
                "re2_joint": _mean(
                    [
                        r["rules"][rule]["joint"]
                        for r in dev_records
                        if r["dataset"] == "re2"
                    ]
                ),
                "used_logs": (
                    float(np.mean([x["used_logs"] for x in rows])) if rows else None
                ),
            }
        )
    best = max(range(len(table)), key=lambda i: (table[i]["top1"] or 0.0, -i))
    ref = {
        "rule": nv.OLD_RULE,
        "top1": _mean([r["old"]["top1"] for r in dev_records]),
        "re2_top1": _mean(
            [r["old"]["top1"] for r in dev_records if r["dataset"] == "re2"]
        ),
        "re3_top1": _mean(
            [r["old"]["top1"] for r in dev_records if r["dataset"] == "re3"]
        ),
        "re2_joint": _mean(
            [r["old"]["joint"] for r in dev_records if r["dataset"] == "re2"]
        ),
    }
    return {"chosen": table[best]["rule"], "table": table, "reference": ref}


def _mean(values: Sequence[Any]) -> Optional[float]:
    return float(np.mean([bool(v) for v in values])) if values else None


__all__ = [
    "CHOSEN_RULE",
    "JOINED_CENTRE",
    "JoinedExecutor",
    "JoinedTeam",
    "dataset_of",
    "load_incident_files",
    "old_state_hash",
    "prepare",
    "run_incident",
    "select_rule",
    "split_of",
]
