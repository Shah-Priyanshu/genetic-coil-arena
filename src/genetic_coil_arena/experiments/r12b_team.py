"""R12b (goal G9b): the attributing newcomer joins the frozen G4c team.

:class:`JoinedTeamV2` is :class:`~.r12_team.JoinedTeam` with ``rca.log_novelty`` v2 (one
``novelty:<service>`` specialist per service, reporting ``own``, ``about``, ``n_attr``,
``logged`` and examples, see ``envs/logtriage/novelty2.py``) and a centre ``rca.decide_joined``
v2 whose rules count ``n_attr``. The old team is hashed and asserted frozen exactly as in G9
(the new specs and specialists reuse the G9 names, so the hash ignores them).

Groups of incidents: ``dev`` (SS/OB and RE3 repetition 1: the rule is chosen here), ``tt`` (all
RE2 Train Ticket incidents: fresh, never ranked in any test), and the two *reused* test splits
``re2_test`` (SS/OB repetitions 2-3, used by G9) and ``re3_test`` (RE3 repetitions 2-4, used by
G4d, G4e and G9). The verdict logic is in ``r12b_report``.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from ..envs.logtriage import novelty as nv
from ..envs.logtriage import novelty2 as n2
from ..envs.logtriage import rcaeval
from ..envs.logtriage.incident import Incident
from . import r12_team as jt
from .r11_team import IncidentTeam

# Pinned after the dev check and before the first run on a fresh or test group.
CHOSEN_RULE_B: Optional[str] = "a_gate45_cap50"

GROUPS = ("dev", "tt", "re2_test", "re3_test")
SYSTEMS = ("SS", "OB", "TT", *rcaeval.RE3_SYSTEMS)


class JoinedTeamV2(jt.JoinedTeam):
    """The G4c team plus the attributing newcomer and the v2 joined centre."""

    @staticmethod
    def _all_rules() -> Sequence[str]:
        return n2.ALL_RULES2

    @staticmethod
    def _definitions() -> list[Any]:
        return n2.joined2_definitions()

    @staticmethod
    def _ranking(
        rule: str, mm: Sequence[Mapping[str, Any]], nn: Sequence[Mapping[str, Any]]
    ) -> tuple[list[int], bool]:
        return n2.joined2_ranking(rule, mm, nn)

    def _novelty(self, params: dict[str, Any]) -> dict[str, Any]:
        inc = self.team.incidents[params["incident"]]
        return n2.novelty2_message(inc, inc.services.index(params["service"]))

    def _decide(self, params: dict[str, Any]) -> dict[str, Any]:
        ranking, fault, _ = n2.rule_decide_joined2(
            params["candidates"],
            params["metric_reports"],
            params["novelty_reports"],
            params["rule"],
        )
        return {"ranking": ranking, "fault_type": fault}


def prepare(inc: Incident) -> Incident:
    """Cache what the newcomer v2 and the baselines read from the lines, then drop the lines."""
    inc.stats()
    n2.attribution(inc)
    nv.novelty_message(inc, 0)
    from .r11_team import light

    return light(inc)


def group_of(inc: Incident) -> str:
    if inc.system == "TT":
        return "tt"
    if inc.rep == 1:
        return "dev"
    return "re3_test" if inc.system.startswith("RE3") else "re2_test"


def group_of_path(path: Path) -> str:
    """The group of a converted incident file, from its folder and repetition."""
    system = path.parent.name
    rep = int(path.stem.rsplit("__", 1)[1])
    if system == "TT":
        return "tt"
    if rep == 1:
        return "dev"
    return "re3_test" if system.startswith("RE3") else "re2_test"


def load_files(groups: Sequence[str], cache_dir: Optional[Path] = None) -> list[Path]:
    """Converted incident files of the requested groups, sorted by system and name."""
    files = rcaeval.list_incidents(cache_dir, SYSTEMS)
    out = [f for f in files if group_of_path(f) in groups]
    return sorted(out, key=lambda p: (p.parent.name, p.name))


def run_incident_b(
    team: IncidentTeam,
    joined: JoinedTeamV2,
    inc: Incident,
    rules: Sequence[str],
) -> dict[str, Any]:
    """One incident through the old team, the v2 joined team (``rules``) and the baselines."""
    rec = jt.run_incident(team, joined, inc, rules)
    rec["group"] = group_of(inc)
    rec["attr"] = n2.attribution_moved(inc)
    a = n2.attribution(inc)
    root = inc.services.index(inc.root_service)
    rec["root_counts"] = {
        "own": len(a.own.get(root, [])),
        "about": len(a.about.get(root, [])),
        "logged": a.logged.get(root, 0),
    }
    return rec


def rule_row(records: Sequence[Mapping[str, Any]], rule: str) -> dict[str, Any]:
    """Pooled and per-dataset top-1 of one rule (``metric`` = the old team) over records."""
    key = "old" if rule == nv.OLD_RULE else None

    def get(r: Mapping[str, Any]) -> Mapping[str, Any]:
        return r["old"] if key else r["rules"][rule]

    def mean(xs: Sequence[Any]) -> Optional[float]:
        return sum(bool(x) for x in xs) / len(xs) if xs else None

    re2 = [r for r in records if r["dataset"] == "re2"]
    re3 = [r for r in records if r["dataset"] == "re3"]
    return {
        "rule": rule,
        "top1": mean([get(r)["top1"] for r in records]),
        "re2_top1": mean([get(r)["top1"] for r in re2]),
        "re3_top1": mean([get(r)["top1"] for r in re3]),
        "re2_joint": mean([get(r)["joint"] for r in re2]),
        "used_logs": (
            None if key else mean([r["rules"][rule]["used_logs"] for r in records])
        ),
    }


def dev_table(dev_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Selection among the 8 candidates plus the reference rows (never selectable)."""
    sel = jt.select_rule(dev_records, tuple(n2.RULES2))
    return {
        "chosen": sel["chosen"],
        "table": sel["table"],
        "reference": [
            rule_row(dev_records, r) for r in (nv.OLD_RULE, n2.REF_RULE, n2.ALONE_RULE)
        ],
    }


__all__ = [
    "CHOSEN_RULE_B",
    "GROUPS",
    "JoinedTeamV2",
    "dev_table",
    "group_of",
    "load_files",
    "prepare",
    "rule_row",
    "run_incident_b",
]
