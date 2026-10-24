"""A small synthetic incident generator for tests and the smoke path (no download).

Services write background templates at steady rates; at the alert the root-cause service
starts new error templates and its rate shifts, and a few of its metrics leave their
normal range. Real data (``rcaeval``) is what the experiment reports; this only checks the
code. Fault types leave different signatures: ``delay`` shifts latency, ``cpu`` the cpu
metric, ``loss`` writes connection-reset errors.
"""

import numpy as np

from .incident import Incident
from .rcaeval import FAULT_TYPES

_BACKGROUND = (
    "INFO handled request method=GET",
    "INFO cache hit for key <*>",
    "DEBUG heartbeat ok",
    "INFO connection opened to db",
)
_FAULT_LINES = {
    "cpu": "WARN worker busy, request queued",
    "mem": "ERROR java.lang.OutOfMemoryError heap space",
    "disk": "ERROR write failed: no space left on device",
    "delay": "WARN upstream call timed out after lat_seconds",
    "loss": "ERROR connection reset by peer HTTP503",
    "socket": "ERROR too many open files: socket exhausted",
}
_FAULT_METRIC = {
    "cpu": "cpu",
    "mem": "mem",
    "disk": "diskio",
    "delay": "latency-90",
    "loss": "error",
    "socket": "socket",
}


def synthetic_incident(
    seed: int,
    n_services: int = 6,
    lines_per_service: int = 400,
    system: str = "synthetic",
    rep: int = 1,
    noise_errors: bool = True,
) -> Incident:
    rng = np.random.default_rng(seed)
    services = [f"svc{i}" for i in range(n_services)]
    root = services[int(rng.integers(n_services))]
    fault = FAULT_TYPES[int(rng.integers(len(FAULT_TYPES)))]
    templates = list(_BACKGROUND) + [
        _FAULT_LINES[fault],
        "ERROR flaky dependency noise",
    ]
    fault_tpl = len(_BACKGROUND)
    noise_tpl = fault_tpl + 1
    t_all, s_all, p_all = [], [], []
    for s in range(n_services):
        n = lines_per_service * (2 if services[s] == root else 1)
        t = np.sort(rng.uniform(-300, 300, size=n))
        tpl = rng.integers(0, len(_BACKGROUND), size=n)
        if services[s] == root:
            err = (t >= 0) & (rng.random(n) < 0.35)
            tpl = np.where(err, fault_tpl, tpl)
        if (
            noise_errors
        ):  # every service has a constant error on both sides of the alert
            noisy = rng.random(n) < 0.05
            tpl = np.where(noisy & (tpl != fault_tpl), noise_tpl, tpl)
        t_all.append(t)
        s_all.append(np.full(n, s))
        p_all.append(tpl)
    t = np.concatenate(t_all)
    order = np.argsort(t, kind="stable")
    metrics: dict[str, dict[str, list]] = {}
    for name in services:
        kinds = {
            k: [round(float(rng.uniform(0.2, 2.5)), 2), 1, None, 0.0]
            for k in ("cpu", "mem")
        }
        if name == root:
            kinds[_FAULT_METRIC[fault]] = [
                round(float(rng.uniform(30, 90)), 2),
                1,
                12.0,
                4.0,
            ]
        metrics[name] = kinds
    return Incident(
        id=f"{system}/syn{seed}/{rep}",
        system=system,
        services=services,
        root_service=root,
        fault_type=fault,
        rep=rep,
        t=t[order].astype(np.float32),
        svc=np.concatenate(s_all)[order].astype(np.int16),
        tpl=np.concatenate(p_all)[order].astype(np.int32),
        templates=templates,
        metrics=metrics,
    )
