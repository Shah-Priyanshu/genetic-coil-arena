"""RCAEval RE2 (Pham et al., 2025): fetch, convert and load incidents with per-service logs.

RCAEval is a benchmark of microservice systems with injected faults; RE2 has multi-source
telemetry (metrics, logs, traces). Only the **logs** are used here: each case directory
``<root service>_<fault>/<rep>/`` has ``logs.csv`` (``timestamp`` in ns, ``container_name``,
``message``) and ``inject_time.txt``. The label is the injected service and fault type.
RE3 (systems ``RE3-SS``, ``RE3-OB``, ``RE3-TT``; same record) has the same layout with
code-level faults: cases are ``<service>_f<k>`` (``carts_f1``), the label is the root service
and a fault id that is specific to the service, and repetitions are numbered 1 to 4.
The alert time is taken to be the injection time, so every condition sees the same
"time zero" a real alert would give.

Source: Zenodo record 14590730, doi:10.5281/zenodo.14590730, CC-BY-4.0;
paper: Pham et al., "RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems
with Telemetry Data", WWW 2025 Companion, arXiv:2412.17015. Sock Shop (SS, 90 incidents)
and Online Boutique (OB) are used by default; Train Ticket (TT) is optional.

The zip files are downloaded to ``<cache>/raw`` (checked against Zenodo's MD5), then each
incident is converted once to a compressed ``.npz`` of templated lines. The cache is
``%LOCALAPPDATA%\\genetic_coil_arena\\rcaeval`` on Windows, ``~/.cache/genetic_coil_arena/
rcaeval`` elsewhere, or ``$GCA_RCAEVAL_DIR``. Nothing is written inside the repo.
"""

import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .incident import Incident
from .templating import make_template

RECORD = "14590730"
DOI = "10.5281/zenodo.14590730"
LICENSE = "CC-BY-4.0"
CITATION = (
    "Pham, L., Zhang, H., Ha, H., Salim, F., Zhang, X. (2025). RCAEval: A Benchmark for "
    "Root Cause Analysis of Microservice Systems with Telemetry Data. WWW '25 Companion. "
    "arXiv:2412.17015"
)
URL = "https://zenodo.org/api/records/" + RECORD + "/files/{name}/content"
# Zenodo's published MD5 and size of each RE2 archive (record 14590730)
ARCHIVES: dict[str, tuple[str, int, str]] = {
    "SS": ("RE2-SS.zip", 245629018, "bd747a8fc7c5be00c613e13fbf9dd74b"),
    "OB": ("RE2-OB.zip", 1191025569, "b9e23f8842c404b396ffd2becff15de4"),
    "TT": ("RE2-TT.zip", 2801345134, "a7fbcd1ada406067dcc50771ae398408"),
    # RE3 (code-level faults, same record). Keys are ``RE3-<system>`` so the RE2 incident
    # folders and their consumers are untouched.
    "RE3-SS": ("RE3-SS.zip", 101485651, "467964eddb2a4fef4d6486c4679e749b"),
    "RE3-OB": ("RE3-OB.zip", 190838064, "96947589084348f9d12cba370313458e"),
    "RE3-TT": ("RE3-TT.zip", 241715689, "ae1eb5906fb4d13a16a6a5e58faaf30b"),
}
DEFAULT_SYSTEMS = ("SS", "OB")
RE3_SYSTEMS = ("RE3-SS", "RE3-OB", "RE3-TT")
FAULT_TYPES = ("cpu", "mem", "disk", "delay", "loss", "socket")
SOURCE_FILE = "SOURCE.json"
CONVERT_VERSION = 3
DEV_REPS = (1,)

_CASE = re.compile(r"^(?P<service>.+)_(?P<fault>[a-z]+)$")
# RE3 injects code-level faults: ``carts_f1``, ``ts-route-service_f3_1``. The fault id is
# specific to the service (f1 of one service is not f1 of another).
_CASE_RE3 = re.compile(r"^(?P<service>.+?)_(?P<fault>f\d+(?:_\d+)?)$")


def cache_dir(override: Optional[Path] = None) -> Path:
    """The RCAEval cache root (``$GCA_RCAEVAL_DIR``, the per-user cache, or ``override``)."""
    if override is not None:
        return Path(override)
    env = os.environ.get("GCA_RCAEVAL_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "genetic_coil_arena" / "rcaeval"


def incident_dir(root: Path, system: str) -> Path:
    return Path(root) / "incidents" / system


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity check against the published checksum
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "genetic-coil-arena"})
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        with open(part, "wb") as out:
            while chunk := response.read(1 << 22):
                out.write(chunk)
    part.replace(dest)


def parse_case(name: str) -> tuple[str, str]:
    """``payment_mem`` -> (``payment``, ``mem``)."""
    match = _CASE.match(name) or _CASE_RE3.match(name)
    if not match:
        raise ValueError(f"unrecognised case name {name!r}")
    return match.group("service"), match.group("fault")


def dataset_of(system: str) -> str:
    """``RE2`` for ``SS``/``OB``/``TT``, ``RE3`` for ``RE3-*``."""
    return "RE3" if system.startswith("RE3-") else "RE2"


class NoRootLogsError(ValueError):
    """The injected service is in neither the logs nor the metrics."""


def metric_features(
    metrics_csv: Any, inject_time: float
) -> dict[str, dict[str, list[Any]]]:
    """Per service and kind, how far a metric moved after the alert (BARO-style scaling).

    Columns are ``<service>_<kind>`` (kinds: cpu, mem, socket, diskio, workload,
    latency-50, latency-90, error). The normal range is the median and IQR before the
    alert; ``z`` is the largest absolute deviation after it in IQR units, with a floor on
    the scale so flat counters do not explode.
    """
    reader = csv.reader(metrics_csv)
    header = next(reader)
    rows = [r for r in reader if len(r) == len(header)]
    data = np.full((len(rows), len(header)), np.nan)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            try:
                data[i, j] = float(cell)
            except ValueError:
                pass
    t = data[:, header.index("time")] - inject_time
    pre, post = t < 0, t >= 0
    out: dict[str, dict[str, list[Any]]] = {}
    for j, name in enumerate(header):
        if name == "time" or "_" not in name:
            continue
        service, kind = name.rsplit("_", 1)
        x = data[:, j]
        if not np.isfinite(x[pre]).any() or not np.isfinite(x[post]).any():
            continue
        base = x[pre][np.isfinite(x[pre])]
        med = float(np.median(base))
        q1, q3 = np.percentile(base, [25, 75])
        whole = x[np.isfinite(x)]
        spread = float(np.percentile(whole, 99) - np.percentile(whole, 1))
        scale = max(float(q3 - q1), 0.1 * abs(med), 0.02 * spread, 1e-9)
        dev = (x - med) / scale
        dpost = dev[post & np.isfinite(dev)]
        # a sustained shift, not a spike: the 75th percentile of the deviations after
        # the alert (the signed value of the same percentile gives the direction)
        z = float(np.percentile(np.abs(dpost), 75))
        direction = 1 if float(np.median(dpost)) >= 0 else -1
        out_of_range = np.flatnonzero(post & (np.abs(np.nan_to_num(dev)) > 3.0))
        first = float(t[out_of_range[0]]) if out_of_range.size else None
        post_med = float(np.nanmedian(x[post]))
        rel = (post_med - med) / (abs(med) + 1e-9)
        out.setdefault(service, {})[kind] = [
            round(min(z, 1000.0), 2),
            direction,
            None if first is None else round(first, 1),
            round(float(np.clip(rel, -100, 100)), 3),
        ]
    return out


def convert_case(
    logs_csv: io.TextIOBase,
    inject_time: float,
    root_service: str,
    fault_type: str,
    rep: int,
    system: str,
    case: str,
    metrics_csv: Any = None,
) -> Incident:
    """Template one case's ``logs.csv`` (and ``simple_metrics.csv``) into an :class:`Incident`.

    Raises:
        ValueError: The CSV lacks the expected columns, or the root service has no logs
    """
    reader = csv.DictReader(logs_csv)
    fields = reader.fieldnames or []
    for need in ("timestamp", "container_name", "message"):
        if need not in fields:
            raise ValueError(f"{case}: logs.csv has no {need!r} column ({fields})")
    stamps: list[float] = []
    svc_names: list[str] = []
    tpl_ids: list[int] = []
    templates: dict[str, int] = {}
    cache: dict[str, int] = {}
    for row in reader:
        try:
            ts = float(row["timestamp"]) / 1e9
        except (TypeError, ValueError):
            continue
        message = row["message"] or ""
        level = (row.get("level") or "").strip()
        if level:
            message = f"[{level.upper()}] {message}"
        tid = cache.get(message)
        if tid is None:
            template = make_template(message)
            tid = templates.setdefault(template, len(templates))
            if len(cache) < 400_000:
                cache[message] = tid
        stamps.append(ts - inject_time)
        svc_names.append(row["container_name"] or "?")
        tpl_ids.append(tid)
    order = np.argsort(np.asarray(stamps), kind="stable")
    t = np.asarray(stamps, dtype=np.float32)[order]
    metrics = metric_features(metrics_csv, inject_time) if metrics_csv else {}
    services = sorted(set(svc_names) | set(metrics))
    index = {name: i for i, name in enumerate(services)}
    svc = np.asarray([index[n] for n in svc_names], dtype=np.int16)[order]
    tpl = np.asarray(tpl_ids, dtype=np.int32)[order]
    if root_service not in index:
        raise NoRootLogsError(f"{case}: root service {root_service!r} is unknown")
    ordered = sorted(templates, key=templates.get)  # type: ignore[arg-type]
    return Incident(
        id=f"{system}/{case}/{rep}",
        system=system,
        services=services,
        root_service=root_service,
        fault_type=fault_type,
        rep=rep,
        t=t,
        svc=svc,
        tpl=tpl,
        templates=ordered,
        meta={"inject_time": inject_time, "raw_lines": int(t.size)},
        metrics=metrics,
    )


def save_incident(inc: Incident, path: Path) -> None:
    meta = {
        "id": inc.id,
        "system": inc.system,
        "services": inc.services,
        "root_service": inc.root_service,
        "fault_type": inc.fault_type,
        "rep": inc.rep,
        "templates": inc.templates,
        "window": list(inc.window),
        "meta": inc.meta,
        "metrics": inc.metrics,
        "version": CONVERT_VERSION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        t=inc.t,
        svc=inc.svc,
        tpl=inc.tpl,
        meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    )
    tmp.replace(path)


def load_file(path: Path) -> Incident:
    with np.load(path) as data:
        meta = json.loads(bytes(data["meta"]).decode("utf-8"))
        return Incident(
            id=meta["id"],
            system=meta["system"],
            services=meta["services"],
            root_service=meta["root_service"],
            fault_type=meta["fault_type"],
            rep=meta["rep"],
            t=data["t"],
            svc=data["svc"],
            tpl=data["tpl"],
            templates=meta["templates"],
            window=tuple(meta["window"]),
            meta=meta["meta"],
            metrics=meta.get("metrics", {}),
        )


def _incident_path(root: Path, system: str, case: str, rep: int) -> Path:
    return incident_dir(root, system) / f"{case}__{rep}.npz"


def list_incidents(
    root: Optional[Path] = None, systems: Optional[tuple[str, ...]] = None
) -> list[Path]:
    """Converted incident files, sorted; empty when nothing is cached."""
    base = cache_dir(root)
    found: list[Path] = []
    for system in systems or DEFAULT_SYSTEMS:
        folder = incident_dir(base, system)
        if folder.is_dir():
            found += sorted(folder.glob("*.npz"))
    return found


def load_incident(path: Path) -> Incident:
    return load_file(Path(path))


def split_of(incident: Incident) -> str:
    """``dev`` (repetition 1: prompt design, smoke runs) or ``test`` (the rest)."""
    return "dev" if incident.rep in DEV_REPS else "test"


def fetch(
    root: Optional[Path] = None,
    systems: tuple[str, ...] = DEFAULT_SYSTEMS,
    *,
    keep_raw: bool = True,
    timeout: float = 120.0,
    log: Callable[[str], None] = print,
) -> Path:
    """Download (if needed), verify and convert the chosen systems; write ``SOURCE.json``.

    ``systems`` are keys of :data:`ARCHIVES` (RE2: ``SS``, ``OB``, ``TT``; RE3: ``RE3-SS``,
    ``RE3-OB``, ``RE3-TT``).

    Re-running skips archives whose incidents are all converted.
    """
    base = cache_dir(root)
    raw = base / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Any] = {}
    for system in systems:
        name, size, md5 = ARCHIVES[system]
        zip_path = raw / name
        done = incident_dir(base, system) / "_done.json"
        if done.exists():
            sources[system] = json.loads(done.read_text(encoding="utf-8"))
            log(f"{system}: already converted ({sources[system]['n_incidents']})")
            continue
        if not zip_path.exists() or zip_path.stat().st_size != size:
            log(f"{system}: downloading {name} ({size / 1e6:.0f} MB)")
            _download(URL.format(name=name), zip_path, timeout)
        digest = _md5(zip_path)
        if digest != md5:
            raise ValueError(f"{name}: MD5 {digest} != published {md5}")
        sources[system] = _convert_archive(zip_path, base, system, size, md5, log)
        if not keep_raw:
            zip_path.unlink()
    previous: dict[str, Any] = {}
    source_path = base / SOURCE_FILE
    if source_path.exists():
        previous = json.loads(source_path.read_text(encoding="utf-8")).get(
            "systems", {}
        )
    previous.update(sources)
    source_path.write_text(
        json.dumps(
            {
                "dataset": "RCAEval RE2 and RE3 (logs and metrics)",
                "record": RECORD,
                "doi": DOI,
                "license": LICENSE,
                "citation": CITATION,
                "url": "https://zenodo.org/records/" + RECORD,
                "converter_version": CONVERT_VERSION,
                "systems": previous,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return base


def _convert_archive(
    zip_path: Path,
    base: Path,
    system: str,
    size: int,
    md5: str,
    log: Callable[[str], None],
) -> dict[str, Any]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    started = time.time()
    summary: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        cases = sorted(
            {
                tuple(n.split("/")[1:3])
                for n in names
                if n.endswith("/logs.csv")
                and n.count("/") == 3
                and n.split("/")[2].isdigit()
                and (_CASE.match(n.split("/")[1]) or _CASE_RE3.match(n.split("/")[1]))
            }
        )
        for k, (case, rep_s) in enumerate(cases):
            rep = int(rep_s)
            service, fault = parse_case(case)
            out = _incident_path(base, system, case, rep)
            prefix = zip_path.stem + "/" + case + "/" + rep_s + "/"
            inject = float(zf.read(prefix + "inject_time.txt").decode().strip())
            if not out.exists():
                with zf.open(prefix + "logs.csv") as raw_file:
                    text = io.TextIOWrapper(
                        raw_file, encoding="utf-8", errors="replace", newline=""
                    )
                    try:
                        with zf.open(prefix + "simple_metrics.csv") as mraw:
                            mtext = io.TextIOWrapper(
                                mraw, encoding="utf-8", errors="replace", newline=""
                            )
                            inc = convert_case(
                                text, inject, service, fault, rep, system, case, mtext
                            )
                    except NoRootLogsError:
                        skipped.append({"case": case, "rep": rep})
                        continue
                save_incident(inc, out)
            summary.append({"case": case, "rep": rep})
            if (k + 1) % 10 == 0:
                log(
                    f"{system}: {k + 1}/{len(cases)} incidents ({time.time() - started:.0f}s)"
                )
    info = {
        "archive": zip_path.name,
        "size": size,
        "md5": md5,
        "n_incidents": len(summary),
        "incidents": summary,
        "skipped_root_unknown": skipped,
    }
    marker = incident_dir(base, system) / "_done.json"
    marker.write_text(json.dumps(info), encoding="utf-8")
    return info
