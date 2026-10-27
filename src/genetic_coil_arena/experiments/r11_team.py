"""R11: the G4c trained team behind the incident desk, through the object layer.

Three pieces:

* :func:`train_metric_specialist` reruns the G4c recipe (``log_triage3``): the fault-type
  classifier is trained on Train Ticket (any repetition) plus Sock Shop / Online Boutique
  repetition 1, the config is chosen by leave-one-system-out cross-validation, and the test
  split (SS/OB repetitions 2 and 3, the catalogued incidents) is never fitted on.
* :class:`IncidentTeam` runs one incident the way ``log_triage2_runtime.RuntimeTeam`` does:
  every service's metric specialist (code, plus the trained fault classifier) is a
  :class:`FunctionSpecialist` behind a :class:`CommandCentre` with signed-free in-process
  envelopes (in-process bus) and a :class:`TraceLog`; a ``rca.decide_trained`` goal carries the typed reports
  to the rule centre. Services the request excluded get no specialist call and are not
  candidates.
* :class:`IncidentExecutor` is the ``rca.investigate`` handler of the desk: it resolves the
  request's reference through the catalogue, runs the team and returns a result whose
  ``public`` part is all a summary may see. The ground-truth label only enters ``score`` and
  ``top1`` (the task score), never ``public``.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from ..envs.logtriage import rcaeval
from ..envs.logtriage import team as tm
from ..envs.logtriage import trained as tr
from ..envs.logtriage.catalogue import CatalogueEntry, IncidentCatalogue, utc_text
from ..envs.logtriage.incident import Incident
from ..envs.logtriage.rcaeval import FAULT_TYPES
from ..learning.classifiers import MLPClassifier
from ..runtime.command_centre import CommandCentre
from ..runtime.router import RoutingRule, RuleBasedRouter
from ..runtime.specialist import FunctionSpecialist
from ..runtime.trace import TraceLog
from . import log_triage3 as t3

CENTRE_NAME = "rca-centre"


# ------------------------------------------------------------------------ training


def light(inc: Incident) -> Incident:
    """Drop the line arrays (the metric specialist reads metrics only)."""
    inc.t = inc.t[:0]
    inc.svc = inc.svc[:0]
    inc.tpl = inc.tpl[:0]
    return inc


def train_metric_specialist(
    cache_dir: Optional[Path] = None,
    seed: int = 0,
    limit_train: Optional[int] = None,
    log: Callable[[str], None] = lambda s: None,
) -> dict[str, Any]:
    """Train the G4c metric fault specialist; returns a JSON-safe state dict.

    Exactly the ``r10_step3.py metric`` recipe: TT (any repetition) + SS/OB repetition 1,
    leave-one-system-out selection over :data:`log_triage3.METRIC_GRID`, then a fit of the
    chosen config on all training rows. The split is checked (``assert_no_test_leak``)
    before anything is loaded.
    """
    train_files, test_files = t3.load_split(cache_dir, limit_train)
    t3.assert_no_test_leak(
        [t3.file_key(f) for f in train_files], [t3.file_key(f) for f in test_files]
    )
    incs = []
    for k, path in enumerate(train_files):
        incs.append(light(rcaeval.load_incident(path)))
        if (k + 1) % 50 == 0 or k + 1 == len(train_files):
            log(f"train: loaded {k + 1}/{len(train_files)}")
    data = t3.metric_dataset(incs)
    log(f"{len(data.y)} training rows, {data.x.shape[1]} features")
    sel = t3.select_metric_model(data, seed=seed, log=log)
    model = t3._make(sel["chosen"], seed).fit(data.x, data.y)
    return {
        "model": model.to_dict(),
        "chosen": sel["chosen"],
        "oof_per_system": sel["table"][sel["chosen_index"]]["per_system"],
        "grid": sel["table"],
        "n_train": int(len(data.y)),
        "train_ids": sorted(data.ids),
        "test_ids": sorted(t3.file_key(f)[0] for f in test_files),
        "seed": seed,
        "recipe": "G4c: TT + SS/OB repetition 1; leave-one-system-out selection",
    }


def load_catalogue(cache_dir: Optional[Path] = None) -> IncidentCatalogue:
    """The catalogue of the 120 RE2 SS/OB test incidents (repetitions 2 and 3)."""
    files = t3.paths_by_role(rcaeval.list_incidents(cache_dir, ("SS", "OB")))["test"]
    if not files:
        raise FileNotFoundError(
            "no converted RE2 test incidents: run `python scripts/fetch_rcaeval.py`"
        )
    return IncidentCatalogue.from_files(files)


def load_specialist(state: Mapping[str, Any]) -> t3.MetricFaultSpecialist:
    """The trained specialist from :func:`train_metric_specialist`'s state."""
    return t3.MetricFaultSpecialist(MLPClassifier.from_dict(dict(state["model"])))


# ------------------------------------------------------------------------ the team


class IncidentTeam:
    """Per-service metric specialists, the trained fault specialist and the rule centre.

    Args:
        specialist: The trained metric fault specialist
        services: Services to register a specialist for (union over the systems)
        incidents: ``incident id -> Incident`` store the handlers read from (filled by the
            caller before :meth:`triage`)
    """

    def __init__(
        self,
        specialist: t3.MetricFaultSpecialist,
        services: Sequence[str],
        trace_path: Optional[Path] = None,
    ):
        self.model = specialist
        self.incidents: dict[str, Incident] = {}
        self.trace = TraceLog(trace_path)
        self.specs = tr.trained_registry()
        self.router = RuleBasedRouter()
        self.cc = CommandCentre(self.specs, router=self.router, trace=self.trace)
        for name in sorted(set(services)):
            self.cc.register(
                FunctionSpecialist(
                    f"metric:{name}",
                    {tr.SPEC_METRIC_FAULT: self._metric},
                    description=f"metric fault specialist of {name}",
                )
            )
            self.router.add_rule(
                RoutingRule(
                    tr.SPEC_METRIC_FAULT,
                    f"metric:{name}",
                    when=lambda g, s=name: g.params["service"] == s,
                )
            )
        self.cc.register(
            FunctionSpecialist(
                CENTRE_NAME,
                {tr.SPEC_DECIDE_TRAINED: self._decide},
                description="rule centre (G4c fusion rule 'metric')",
            )
        )

    # handlers: the only place that touches incident data

    def _metric(self, params: dict[str, Any]) -> dict[str, Any]:
        inc = self.incidents[params["incident"]]
        s = inc.services.index(params["service"])
        return tr.trained_metric_message(
            tm.metric_message(inc, s), self.model.probs(inc, s)
        )

    def _decide(self, params: dict[str, Any]) -> dict[str, Any]:
        ranking, fault = tr.rule_decide_trained(
            params["candidates"], params["metric_reports"], None, "metric"
        )
        return {"ranking": list(ranking), "fault_type": fault}

    def _run_all(self, goals: list[Any], trace_id: str) -> list[Any]:
        ids = [self.cc.submit(g, trace_id=trace_id) for g in goals]
        self.cc.run_until_idle()
        out = []
        for task_id in ids:
            status = self.cc.status(task_id)
            if status.state != "completed":
                raise RuntimeError(f"{status.spec} {status.state}: {status.error}")
            out.append(status.result)
        return out

    def triage(
        self, inc: Incident, exclude: Sequence[str] = (), trace_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Run one incident; excluded services are neither asked nor candidates."""
        start = time.perf_counter()
        self.incidents[inc.id] = inc
        services = [s for s in inc.services if s not in set(exclude)]
        if len(services) < 2:
            raise ValueError("a decision needs at least two candidate services")
        tid = trace_id or inc.id
        make = self.specs.make
        mm = self._run_all(
            [
                make(tr.SPEC_METRIC_FAULT, {"incident": inc.id, "service": s})
                for s in services
            ],
            tid,
        )
        (decision,) = self._run_all(
            [
                make(
                    tr.SPEC_DECIDE_TRAINED,
                    {
                        "incident": inc.id,
                        "candidates": services,
                        "metric_reports": mm,
                    },
                )
            ],
            tid,
        )
        ranking = list(decision["ranking"])
        top = next(m for m in mm if m["service"] == ranking[0])
        fault = decision["fault_type"]
        return {
            "ranking": ranking,
            "fault_type": fault,
            "fault_prob": float(top["fault_probs"][FAULT_TYPES.index(fault)]),
            "onset_s": top["onset_s"],
            "n_envelopes": len(self.trace.envelopes(tid)),
            "wall_s": time.perf_counter() - start,
        }

    def close(self) -> None:
        self.trace.close()


# ------------------------------------------------------------------------ the desk handler


class IncidentExecutor:
    """``rca.investigate`` params to a result (cached by params, like the R7a executors).

    Args:
        catalogue: Tickets and times of the incidents the desk can investigate
        team: The G4c trained team
        loader: ``CatalogueEntry -> Incident`` (default: the catalogue's converted file,
            line arrays dropped)
    """

    def __init__(
        self,
        catalogue: IncidentCatalogue,
        team: IncidentTeam,
        loader: Optional[Callable[[CatalogueEntry], Incident]] = None,
    ):
        self.catalogue = catalogue
        self.team = team
        self._load = loader or self._default_loader
        self._store: dict[str, Incident] = {}
        self._cache: dict[str, dict[str, Any]] = {}
        self.runs = 0

    def _default_loader(self, entry: CatalogueEntry) -> Incident:
        return light(rcaeval.load_incident(self.catalogue.files[entry.ticket]))

    def incident(self, entry: CatalogueEntry) -> Incident:
        if entry.ticket not in self._store:
            self._store[entry.ticket] = self._load(entry)
        return self._store[entry.ticket]

    def __call__(self, params: Mapping[str, Any]) -> dict[str, Any]:
        entry = self.catalogue.resolve(params)  # ResolveError fails the task
        exclude = sorted(params.get("exclude_services") or [])
        key = f"{entry.ticket}|{params['top_k']}|{params['want_fault_type']}|{exclude}"
        if key not in self._cache:
            self._cache[key] = self._run(entry, params, exclude)
            self.runs += 1
        return dict(self._cache[key])

    def evaluate_all(self) -> dict[str, Any]:
        """Top-1 and joint accuracy of the team on every catalogued incident (all defaults)."""
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
            rows.append((res["top1"], res["score"]))
        n = len(rows)
        return {
            "n": n,
            "top1": sum(r[0] for r in rows) / n if n else None,
            "joint": sum(r[1] for r in rows) / n if n else None,
        }

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
        joint = float(top1 and res["fault_type"] == inc.fault_type)
        # the label enters only these scores, never ``public``
        return {
            "score": joint,
            "top1": top1,
            "per_seed": [joint],
            "wall_s": round(float(res["wall_s"]), 4),
            "n_envelopes": int(res["n_envelopes"]),
            "public": public,
        }
