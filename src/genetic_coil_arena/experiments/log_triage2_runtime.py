"""The R10 step-2 team through the object layer (CommandCentre, envelopes, TraceLog).

The same decisions as the batch path in :mod:`.log_triage2`, but every report travels as a
schema-validated GoalSpec result between specialists and the centre:

1. the driver submits one ``rca.metric_report`` goal per service; each is routed to that
   service's metric specialist (code), which reads only its own service's metrics;
2. the centre relays each reported onset in an ``rca.log_report`` goal to that service's
   log specialist (a model), which reads only its own logs around that onset;
3. one ``rca.decide`` goal carries all reports to the centre specialist (rule or LLM).

Every goal, result and state change is in the trace. The domain is only in the handlers.
"""

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from ..envs.logtriage import messages as msgs
from ..envs.logtriage import team as tm
from ..envs.logtriage.incident import Incident
from ..envs.logtriage.rcaeval import FAULT_TYPES
from ..runtime.command_centre import CommandCentre
from ..runtime.router import RoutingRule, RuleBasedRouter
from ..runtime.specialist import FunctionSpecialist
from ..runtime.trace import TraceLog
from . import log_triage as lt
from . import log_triage2 as t2

CENTRE_NAME = "rca-centre"


class RuntimeTeam:
    """Specialists for the services of the given incidents, behind one command centre.

    Args:
        incidents: Incidents the team can work on (looked up by id from goal params)
        log_reasoner: Model behind the log specialists
        cfg: Step-2 settings
        centre: ``rule`` or ``llm`` (then ``centre_reasoner`` is required)
        trace_path: Optional JSONL file for the trace
    """

    def __init__(
        self,
        incidents: Sequence[Incident],
        log_reasoner: lt.Reasoner,
        cfg: Optional[t2.Settings2] = None,
        centre: str = "rule",
        centre_reasoner: Optional[lt.Reasoner] = None,
        trace_path: Optional[Path] = None,
    ):
        if centre == "llm" and centre_reasoner is None:
            raise ValueError("an LLM centre needs centre_reasoner")
        self.cfg = cfg or t2.Settings2()
        self.incidents = {inc.id: inc for inc in incidents}
        self.log_reasoner = log_reasoner
        self.centre = centre
        self.centre_reasoner = centre_reasoner
        self.trace = TraceLog(trace_path)
        self.specs = tm.team_registry()
        self.router = RuleBasedRouter()
        self.cc = CommandCentre(self.specs, router=self.router, trace=self.trace)
        self.log_tokens = {"prompt": 0, "new": 0}
        self.log_latency = 0.0
        services = sorted({s for inc in incidents for s in inc.services})
        for name in services:
            self._add(f"metric:{name}", tm.SPEC_METRIC, name, self._metric)
            self._add(f"log:{name}", tm.SPEC_LOG, name, self._log)
        self.cc.register(
            FunctionSpecialist(
                CENTRE_NAME, {tm.SPEC_DECIDE: self._decide}, description="the centre"
            )
        )

    def _add(self, specialist: str, spec: str, service: str, handler: Any) -> None:
        self.cc.register(
            FunctionSpecialist(specialist, {spec: handler}, description=service)
        )
        self.router.add_rule(
            RoutingRule(
                spec, specialist, when=lambda g, s=service: g.params["service"] == s
            )
        )

    # -- handlers (the only place that touches domain data)

    def _inc(self, params: dict[str, Any]) -> tuple[Incident, int]:
        inc = self.incidents[params["incident"]]
        return inc, inc.services.index(params["service"])

    def _metric(self, params: dict[str, Any]) -> dict[str, Any]:
        inc, s = self._inc(params)
        return tm.metric_message(inc, s)

    def _log(self, params: dict[str, Any]) -> dict[str, Any]:
        inc, s = self._inc(params)
        view = tm.log_view(
            inc,
            s,
            params["onset_s"],
            self.cfg.log_view_tokens,
            self.log_reasoner.counter,
            self.cfg.before_s,
            self.cfg.after_s,
        )
        start = time.perf_counter()
        (w,) = self.log_reasoner.write(
            [tm.log_prompt(inc.services[s], view)],
            self.cfg.log_new_tokens,
            True,
            schema=tm.LOG_SCHEMA,
        )
        self.log_latency += time.perf_counter() - start
        self.log_tokens["prompt"] += w.n_prompt
        self.log_tokens["new"] += w.n_new
        clean = tm.clean_log_message(msgs.parse_json_object(w.text), inc.services[s])
        return clean or tm.empty_log_message(inc.services[s])

    def _decide(self, params: dict[str, Any]) -> dict[str, Any]:
        inc = self.incidents[params["incident"]]
        mm = params["metric_reports"]
        lm = params.get("log_reports") or None
        if self.centre == "rule":
            ranking, faults = tm.rule_decide(
                params["candidates"], mm, lm, self.cfg.policy
            )
            fault = faults["chosen"]
        else:
            assert self.centre_reasoner is not None
            dec = t2.llm_decide(inc, mm, lm, self.centre_reasoner, self.cfg)
            ranking, fault = dec.ranking, dec.fault
        if fault not in FAULT_TYPES:
            fault = tm.metric_fault(mm[0]) if mm else tm.DEFAULT_FAULT
        return {"ranking": list(ranking), "fault_type": fault}

    # -- driving one incident

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

    def triage(self, inc: Incident, hybrid: bool = True) -> dict[str, Any]:
        """Run one incident; returns the decision, the reports and the plumbing counts."""
        before = len(self.trace)
        start = time.perf_counter()
        trace_id = f"{inc.id}"
        self.trace.snapshot(
            trace_id, {"incident": inc.id, "services": inc.services}, self.cc.clock
        )
        make = self.specs.make
        mm = self._run_all(
            [
                make(tm.SPEC_METRIC, {"incident": inc.id, "service": s})
                for s in inc.services
            ],
            trace_id,
        )
        lm: Optional[list[Any]] = None
        if hybrid:
            lm = self._run_all(
                [
                    make(
                        tm.SPEC_LOG,
                        {"incident": inc.id, "service": s, "onset_s": m["onset_s"]},
                    )
                    for s, m in zip(inc.services, mm)
                ],
                trace_id,
            )
        params: dict[str, Any] = {
            "incident": inc.id,
            "candidates": list(inc.services),
            "metric_reports": mm,
        }
        if lm is not None:
            params["log_reports"] = lm
        (decision,) = self._run_all([make(tm.SPEC_DECIDE, params)], trace_id)
        envelopes = self.trace.envelopes(trace_id)
        return {
            "incident": inc.id,
            "ranking": decision["ranking"],
            "fault_type": decision["fault_type"],
            "metric_reports": mm,
            "log_reports": lm,
            "n_envelopes": len(envelopes),
            "envelope_bytes": sum(len(e.to_json().encode("utf-8")) for e in envelopes),
            "report_bytes": t2.message_bytes(mm, lm)["bytes"],
            "n_trace_records": len(self.trace) - before,
            "ticks": self.cc.clock,
            "wall_s": time.perf_counter() - start,
        }

    def close(self) -> None:
        self.trace.close()
