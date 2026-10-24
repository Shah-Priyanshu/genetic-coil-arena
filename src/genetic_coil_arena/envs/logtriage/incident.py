"""An incident: time-ordered, templated log lines from several services, plus its label.

Lines are stored as three parallel arrays (time relative to the alert in seconds, service
index, template index), so an incident with 10^5 lines stays small and every statistic
the baselines and the specialists need is a vectorised count.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .templating import severity


@dataclass
class TemplateStats:
    """Per (service, template) counts before (t < 0) and after (t >= 0) the alert.

    Attributes:
        svc, tpl: Service and template indices, one entry per distinct pair
        pre, post: Line counts before and from the alert on
        first_post: Time of the first line at or after the alert (NaN if none)
        sev: Severity of the template (0 plain, 1 warning-like, 2 error-like)
    """

    svc: np.ndarray
    tpl: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    first_post: np.ndarray
    sev: np.ndarray

    def __len__(self) -> int:
        return int(self.svc.size)

    def surprise(self) -> np.ndarray:
        """Score of how unexpected each pair is: a log-ratio of post to pre counts,
        weighted by how many lines are involved and by severity.

        New or surging templates score high; templates that vanished score half as much
        (a crashed or blocked service stops logging its usual lines).
        """
        post = self.post.astype(np.float64)
        pre = self.pre.astype(np.float64)
        lift = np.log((post + 0.5) / (pre + 0.5))
        mass = np.log1p(np.maximum(post, pre))
        weight = 1.0 + self.sev.astype(np.float64)
        return np.where(lift > 0, lift * mass, -0.5 * lift * mass) * weight


@dataclass
class Incident:
    """One incident.

    Attributes:
        id: Unique name, e.g. ``SS/payment_mem/2``
        system: Source system (``SS``, ``OB``, ``synthetic``)
        services: Services that have a log stream; the candidates for the root cause
        root_service: Label, one of ``services``
        fault_type: Label (``cpu``, ``mem``, ``disk``, ``delay``, ``loss``, ``socket``)
        rep: Repetition number of the injection (1 is the dev split)
        t: Seconds relative to the alert, sorted, shape (N,)
        svc: Service index of each line, shape (N,)
        tpl: Template index of each line, shape (N,)
        templates: Template strings
        window: (start, end) of the log window in seconds relative to the alert
        metrics: Per service and metric kind, ``[z, direction, first_t, rel_change]``:
            the largest robust deviation after the alert in IQR units, +1 / -1 for up /
            down, the first time (s after the alert) the metric left its normal range
            (None if never), and the relative change of the median. Empty when absent.
    """

    id: str
    system: str
    services: list[str]
    root_service: str
    fault_type: str
    rep: int
    t: np.ndarray
    svc: np.ndarray
    tpl: np.ndarray
    templates: list[str]
    window: tuple[float, float] = (0.0, 0.0)
    meta: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    _stats: Optional[TemplateStats] = field(default=None, repr=False, compare=False)
    _tpl_sev: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.root_service not in self.services:
            raise ValueError(f"{self.id}: root {self.root_service!r} not in services")
        if not (self.t.shape == self.svc.shape == self.tpl.shape):
            raise ValueError(f"{self.id}: line arrays differ in shape")
        if self.t.size and (np.any(np.diff(self.t) < 0)):
            raise ValueError(f"{self.id}: lines are not sorted by time")
        if self.window == (0.0, 0.0) and self.t.size:
            self.window = (float(self.t[0]), float(self.t[-1]))

    @property
    def n_lines(self) -> int:
        return int(self.t.size)

    @property
    def n_services(self) -> int:
        return len(self.services)

    @property
    def template_severity(self) -> np.ndarray:
        if self._tpl_sev is None:
            self._tpl_sev = np.array(
                [severity(t) for t in self.templates], dtype=np.int8
            )
        return self._tpl_sev

    def stats(self) -> TemplateStats:
        """Per (service, template) counts (cached)."""
        if self._stats is None:
            n_tpl = max(len(self.templates), 1)
            key = self.svc.astype(np.int64) * n_tpl + self.tpl.astype(np.int64)
            uniq, inverse = np.unique(key, return_inverse=True)
            post_mask = self.t >= 0
            post = np.bincount(inverse[post_mask], minlength=uniq.size)
            pre = np.bincount(inverse[~post_mask], minlength=uniq.size)
            first = np.full(uniq.size, np.nan)
            if post_mask.any():
                # Lines are time-sorted, so the first occurrence of each key is its min
                idx_post = np.flatnonzero(post_mask)
                keys_post = inverse[idx_post]
                firsts, where = np.unique(keys_post, return_index=True)
                first[firsts] = self.t[idx_post[where]]
            tpl_idx = (uniq % n_tpl).astype(np.int32)
            self._stats = TemplateStats(
                svc=(uniq // n_tpl).astype(np.int32),
                tpl=tpl_idx,
                pre=pre.astype(np.int64),
                post=post.astype(np.int64),
                first_post=first,
                sev=self.template_severity[tpl_idx] if len(tpl_idx) else tpl_idx,
            )
        return self._stats

    def service_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """Lines per service before and after the alert, each of shape (S,)."""
        post = self.t >= 0
        n = self.n_services
        return (
            np.bincount(self.svc[~post], minlength=n),
            np.bincount(self.svc[post], minlength=n),
        )

    def metric_rows(
        self, service: str, min_z: float = 3.0, top: int = 3
    ) -> list[tuple[str, list[Any]]]:
        """The ``top`` metric kinds of ``service`` that left their normal range."""
        rows = [
            (k, v) for k, v in self.metrics.get(service, {}).items() if v[0] >= min_z
        ]
        rows.sort(key=lambda kv: -kv[1][0])
        return rows[:top]

    def line_text(self, i: int) -> str:
        """One templated line as the naive baselines show it."""
        return f"{self.t[i]:+.0f}s {self.services[self.svc[i]]} {self.templates[self.tpl[i]]}"
