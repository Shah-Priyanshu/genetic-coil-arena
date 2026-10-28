"""R12c (goal G10b): desk v4 = the R12 desk plus the literal check, scored in pairs.

:class:`CheckedDesk` wraps a :class:`~.r11_pipeline.Desk` (or ``KeywordDesk``). The wrapped desk
translates a request once; the wrapper caches that raw result and returns it either as it is
(check off, i.e. the G10 desk) or after :func:`~..envs.logtriage.literal_check.check_params`
and a new validation and resolution (check on, desk v4). :func:`process_pair` runs one item
through the whole chain both ways on the same raw translation, so the effect of the check is
exactly paired and costs no extra model call. The row is the check-on record with the check-off
record under ``off``, plus the raw and the checked params, the rules that fired and the changes.

``silent_wrong`` is the pre-registered error that matters most: the desk ran on an incident
other than the one the request names, or answered a request that should have been questioned or
refused.
"""

import dataclasses
import datetime as dt
from collections.abc import Mapping
from typing import Any, Optional

from ..envs.logtriage.literal_check import CheckResult, check_params
from ..protocol.goalspec import SpecValidationError
from ..translator import INVALID, OK, TranslationResult
from . import r11_pipeline as pipe
from .r7a_domains import Domain
from .r11_pipeline import DeskResult


class CheckedDesk:
    """A desk whose incident translations are optionally checked against the request text.

    Args:
        inner: The desk (``Desk`` or ``KeywordDesk``) that does the translating
        now: The desk clock, passed to the check
        check: Whether to apply the literal check
        raw: Shared cache ``(text, report) -> DeskResult`` of the inner desk (pass the same
            dict to the check-off and the check-on wrapper to share one translator run)
        checks: Shared dict ``text -> CheckResult`` filled when the check runs
    """

    def __init__(
        self,
        inner: Any,
        now: dt.datetime,
        check: bool,
        raw: dict[tuple[str, bool], DeskResult],
        checks: dict[str, CheckResult],
    ):
        self.inner = inner
        self.now = now
        self.check = check
        self.raw = raw
        self.checks = checks

    @property
    def registry(self) -> Any:
        return self.inner.registry

    @property
    def domains(self) -> dict[str, Domain]:
        return self.inner.domains  # type: ignore[no-any-return]

    @property
    def catalogue(self) -> Any:
        return self.inner.catalogue

    def route(self, text: str) -> tuple[str, float, str]:
        return self.inner.route(text)  # type: ignore[no-any-return]

    def handle(self, text: str, report: bool = False) -> DeskResult:
        key = (text, report)
        if key not in self.raw:
            self.raw[key] = self.inner.handle(text, report=report)
        res = self.raw[key]
        if not self.check or report:  # status reports are never checked
            return res
        return self._checked(text, res)

    def _checked(self, text: str, res: DeskResult) -> DeskResult:
        tr = res.translation
        if res.domain != "incident" or tr is None or tr.status != OK or tr.spec is None:
            return res
        outcome = check_params(tr.spec.params, text, self.now)
        self.checks[text] = outcome
        if not outcome.rules:
            return res
        dom = self.inner.domains["incident"]
        try:
            spec = self.registry.make(dom.spec_name, outcome.params)
            tr2 = dataclasses.replace(tr, spec=spec)
        except SpecValidationError as exc:
            tr2 = TranslationResult(
                text,
                INVALID,
                reason=f"literal check: {exc}",
                errors=tuple(exc.errors),
                attempts=tr.attempts,
                latency_s=tr.latency_s,
                call_latencies=tr.call_latencies,
                n_prompt_tokens=tr.n_prompt_tokens,
                raw=tr.raw,
            )
        return pipe._final(
            text, res.domain, tr2, self.catalogue, res.route_s, res.route_raw
        )


def process_pair(
    inner: Any,
    now: dt.datetime,
    item: Any,
    summariser: Optional[Any],
    raw: dict[tuple[str, bool], DeskResult],
) -> dict[str, Any]:
    """One item through the chain with the check on (the row) and off (``row["off"]``)."""
    checks: dict[str, CheckResult] = {}
    off = CheckedDesk(inner, now, False, raw, checks)
    on = CheckedDesk(inner, now, True, raw, checks)
    off_rec = pipe.process_incident_item(off, item, None).record
    on_rec = pipe.process_incident_item(on, item, summariser).record
    outcome = checks.get(item.text)
    row = dict(on_rec)
    row["off"] = off_rec
    row["raw_pred"] = off_rec["pred"]
    row["checked_pred"] = on_rec["pred"]
    row["rules"] = list(outcome.rules) if outcome else []
    row["changes"] = (
        {k: list(v) for k, v in (outcome.changes or {}).items()} if outcome else {}
    )
    return row


def side(row: Mapping[str, Any], which: str) -> Mapping[str, Any]:
    """The check-off (``off``) or check-on (``on``) record of a paired row."""
    return (
        row["off"] if which == "off" else {k: v for k, v in row.items() if k != "off"}
    )


def silent_wrong(rec: Mapping[str, Any], resolves_to: Optional[str]) -> bool:
    """The desk answered, but on the wrong incident or a request it should not have answered."""
    if rec["status"] != OK:
        return False
    if rec["kind"] != "spec":
        return True
    return rec.get("ticket") != resolves_to
