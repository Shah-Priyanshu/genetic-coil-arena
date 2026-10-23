"""Standard result objects, so every experiment reports the same things the same way.

An ``ExperimentResult`` holds, for one task:

- performance of each system and each baseline across seeds (mean with a Student-t
  CI, and the IQM with a stratified bootstrap CI);
- the mandatory baselines it was required to include, and which ones are missing;
- Holm-corrected Mann-Whitney comparisons with the Vargha-Delaney A12 effect size;
- message-ablation deltas (intact minus intervened, paired by seed) with Holm-corrected
  Wilcoxon signed-rank p-values;
- communication metrics per system.

It serializes to JSON (lossless round trip; NaN is stored as null), to a flat CSV and
to a plain-text table.
"""

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .stats import iqm, mean_ci, stratified_bootstrap_ci

SYSTEM = "system"
BASELINE = "baseline"

PathLike = Union[str, Path]


def _nan_to_none(value: Any) -> Any:
    """Recursively make a structure strict-JSON safe (NaN/inf -> None, numpy -> python)."""
    if isinstance(value, dict):
        return {str(k): _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nan_to_none(v) for v in value]
    if isinstance(value, np.ndarray):
        return _nan_to_none(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return None if not math.isfinite(value) else value
    return value


def _none_to_nan(value: Any) -> float:
    return float("nan") if value is None else float(value)


@dataclass
class ConditionResult:
    """Performance of one condition (system or baseline) across seeds."""

    name: str
    role: str
    seeds: list[int]
    per_seed: list[float]
    mean: float
    ci: float
    iqm: float
    iqm_low: float
    iqm_high: float
    n_episodes: int = 0

    def __post_init__(self) -> None:
        # Per-seed artifacts, in seed order, kept in memory only (never serialized):
        # the system's outputs (``evaluate(keep_outputs=True)``) and the
        # ``RecordingHook`` of each seed (``evaluate(record=True)``)
        self.outputs: Optional[list[Any]] = None
        self.recordings: Optional[list[Any]] = None

    @classmethod
    def from_scores(
        cls,
        name: str,
        role: str,
        seeds: Sequence[int],
        per_seed: Sequence[float],
        n_episodes: int = 0,
        confidence: float = 0.95,
        n_boot: int = 2000,
        boot_seed: int = 0,
    ) -> "ConditionResult":
        """Summarize one score per seed."""
        scores = [float(v) for v in per_seed]
        if len(scores) != len(seeds):
            raise ValueError("Need exactly one score per seed")
        mean, half = mean_ci(scores, confidence)
        point, low, high = stratified_bootstrap_ci(
            scores, iqm, n_boot=n_boot, confidence=confidence, seed=boot_seed
        )
        return cls(
            name,
            role,
            [int(s) for s in seeds],
            scores,
            mean,
            half,
            point,
            low,
            high,
            int(n_episodes),
        )

    @property
    def low(self) -> float:
        return self.mean - self.ci

    @property
    def high(self) -> float:
        return self.mean + self.ci

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConditionResult":
        floats = ("mean", "ci", "iqm", "iqm_low", "iqm_high")
        kwargs = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        for key in floats:
            kwargs[key] = _none_to_nan(kwargs[key])
        kwargs["per_seed"] = [_none_to_nan(v) for v in kwargs["per_seed"]]
        return cls(**kwargs)


@dataclass
class AblationResult:
    """Effect of one message intervention on one system, paired by seed."""

    system: str
    intervention: str
    intact_mean: float
    ablated: ConditionResult
    delta_mean: float  # Intact minus ablated, oriented so positive = performance lost
    delta_ci: float
    p: float  # Paired test of intact vs ablated per-seed scores (see ``test``)
    p_holm: float
    a12: float  # P(intact beats ablated) per seed, oriented like delta
    listening: bool  # Delta CI lies entirely above zero
    test: str = "wilcoxon"  # Wilcoxon signed-rank on the per-seed pairs

    @property
    def delta_low(self) -> float:
        return self.delta_mean - self.delta_ci

    @property
    def delta_high(self) -> float:
        return self.delta_mean + self.delta_ci

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AblationResult":
        kwargs = dict(data)
        kwargs["ablated"] = ConditionResult.from_dict(data["ablated"])
        for key in ("intact_mean", "delta_mean", "delta_ci", "p", "p_holm", "a12"):
            kwargs[key] = _none_to_nan(kwargs[key])
        return cls(**kwargs)


@dataclass
class ExperimentResult:
    """Everything one experiment on one task reports."""

    task: dict[str, Any]
    seeds: list[int]
    conditions: list[ConditionResult] = field(default_factory=list)
    required_baselines: list[str] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    ablations: list[AblationResult] = field(default_factory=list)
    communication: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- Completeness -------------------------------------------------------------

    @property
    def baselines(self) -> list[ConditionResult]:
        return [c for c in self.conditions if c.role == BASELINE]

    @property
    def systems(self) -> list[ConditionResult]:
        return [c for c in self.conditions if c.role == SYSTEM]

    @property
    def missing_baselines(self) -> list[str]:
        present = {c.name for c in self.baselines}
        return [name for name in self.required_baselines if name not in present]

    @property
    def complete(self) -> bool:
        """A report counts only with all its required baselines (and at least one)."""
        return bool(self.required_baselines) and not self.missing_baselines

    def require_complete(self) -> "ExperimentResult":
        """Raise unless every mandatory baseline was run; returns self for chaining."""
        if not self.required_baselines:
            raise ValueError(
                f"Task {self.task.get('name')!r} has no mandatory baselines; "
                "register at least one before reporting"
            )
        if self.missing_baselines:
            raise ValueError(
                f"Report for task {self.task.get('name')!r} is missing "
                f"baselines: {self.missing_baselines}"
            )
        return self

    def condition(self, name: str) -> ConditionResult:
        for c in self.conditions:
            if c.name == name:
                return c
        raise KeyError(f"No condition named {name!r}")

    def add_communication(self, system: str, metrics: dict[str, float]) -> None:
        """Attach communication metrics (e.g. from ``communication_metrics``)."""
        self.communication.setdefault(system, {}).update(
            {k: float(v) for k, v in metrics.items()}
        )

    # ---- Serialization ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Strict-JSON-safe nested dict (NaN stored as None)."""
        data = asdict(self)
        data["complete"] = self.complete
        data["missing_baselines"] = self.missing_baselines
        return _nan_to_none(data)

    def to_json(self, path: Optional[PathLike] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, allow_nan=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentResult":
        def nanify(row: dict[str, Any]) -> dict[str, Any]:
            return {k: (float("nan") if v is None else v) for k, v in row.items()}

        return cls(
            task=dict(data["task"]),
            seeds=[int(s) for s in data["seeds"]],
            conditions=[
                ConditionResult.from_dict(c) for c in data.get("conditions", [])
            ],
            required_baselines=list(data.get("required_baselines", [])),
            comparisons=[nanify(c) for c in data.get("comparisons", [])],
            ablations=[AblationResult.from_dict(a) for a in data.get("ablations", [])],
            communication={
                k: nanify(v) for k, v in data.get("communication", {}).items()
            },
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, source: PathLike) -> "ExperimentResult":
        """Load from a JSON string or a path to a JSON file."""
        text = str(source)
        if not text.lstrip().startswith("{"):
            text = Path(source).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    def to_rows(self) -> list[dict[str, Any]]:
        """Flat rows (one per condition, ablation, comparison and metric) for CSV."""
        task = self.task.get("name", "")
        rows: list[dict[str, Any]] = []
        for c in self.conditions:
            rows.append(
                {
                    "task": task,
                    "section": c.role,
                    "name": c.name,
                    "n_seeds": len(c.per_seed),
                    "mean": c.mean,
                    "ci": c.ci,
                    "low": c.low,
                    "high": c.high,
                    "iqm": c.iqm,
                    "iqm_low": c.iqm_low,
                    "iqm_high": c.iqm_high,
                }
            )
        for a in self.ablations:
            rows.append(
                {
                    "task": task,
                    "section": "ablation",
                    "name": f"{a.system}|{a.intervention}",
                    "n_seeds": len(a.ablated.per_seed),
                    "mean": a.ablated.mean,
                    "ci": a.ablated.ci,
                    "low": a.ablated.low,
                    "high": a.ablated.high,
                    "iqm": a.ablated.iqm,
                    "iqm_low": a.ablated.iqm_low,
                    "iqm_high": a.ablated.iqm_high,
                    "delta": a.delta_mean,
                    "delta_low": a.delta_low,
                    "delta_high": a.delta_high,
                    "p": a.p,
                    "p_holm": a.p_holm,
                    "a12": a.a12,
                    "listening": a.listening,
                }
            )
        for comp in self.comparisons:
            rows.append(
                {
                    "task": task,
                    "section": "comparison",
                    "name": f"{comp['a']} vs {comp['b']}",
                    "delta": comp["mean_diff"],
                    "p": comp["p"],
                    "p_holm": comp["p_holm"],
                    "a12": comp["a12"],
                }
            )
        for system, metrics in self.communication.items():
            for metric, value in metrics.items():
                rows.append(
                    {
                        "task": task,
                        "section": "communication",
                        "name": f"{system}|{metric}",
                        "value": value,
                    }
                )
        return rows

    def to_csv(self, path: PathLike) -> Path:
        """Write ``to_rows()`` with a fixed header; empty cells where not applicable."""
        out = Path(path)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, restval="")
            writer.writeheader()
            for row in self.to_rows():
                writer.writerow({k: _csv_cell(v) for k, v in row.items()})
        return out

    def format_table(self, digits: int = 3) -> str:
        """Plain-text summary: performance, ablations, comparisons, communication."""
        return format_table(self, digits)

    def __str__(self) -> str:
        return self.format_table()


CSV_COLUMNS = [
    "task",
    "section",
    "name",
    "n_seeds",
    "mean",
    "ci",
    "low",
    "high",
    "iqm",
    "iqm_low",
    "iqm_high",
    "delta",
    "delta_low",
    "delta_high",
    "p",
    "p_holm",
    "a12",
    "listening",
    "value",
]


def _csv_cell(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths)) for r in rows]
    return [line, sep, *body]


def format_table(result: ExperimentResult, digits: int = 3) -> str:
    """Render an ``ExperimentResult`` as aligned plain-text tables."""
    f = lambda v: _fmt(v, digits)  # noqa: E731
    name = result.task.get("name", "?")
    lines = [f"Task: {name}  (seeds: {len(result.seeds)})"]
    if result.complete:
        lines.append(f"Baselines: {', '.join(result.required_baselines)}")
    else:
        missing = result.missing_baselines or ["<none registered>"]
        lines.append(f"INCOMPLETE: missing mandatory baselines {', '.join(missing)}")

    lines.append("")
    rows = [
        [
            c.name,
            c.role,
            str(len(c.per_seed)),
            f"{f(c.mean)} ± {f(c.ci)}",
            f"{f(c.iqm)} [{f(c.iqm_low)}, {f(c.iqm_high)}]",
        ]
        for c in result.conditions
    ]
    lines += _table(["condition", "role", "n", "mean ± 95% CI", "IQM [95% CI]"], rows)

    if result.ablations:
        lines.append("")
        rows = [
            [
                a.system,
                a.intervention,
                f(a.ablated.mean),
                f"{f(a.delta_mean)} [{f(a.delta_low)}, {f(a.delta_high)}]",
                f(a.p_holm),
                f(a.a12),
                f(a.listening),
            ]
            for a in result.ablations
        ]
        lines += _table(
            [
                "system",
                "intervention",
                "ablated mean",
                "drop [95% CI]",
                "p (Holm)",
                "A12",
                "listening",
            ],
            rows,
        )

    if result.comparisons:
        lines.append("")
        rows = [
            [
                f"{c['a']} vs {c['b']}",
                f(c["mean_diff"]),
                f(c["p"]),
                f(c["p_holm"]),
                f(c["a12"]),
            ]
            for c in result.comparisons
        ]
        lines += _table(["comparison", "mean diff", "p", "p (Holm)", "A12"], rows)

    if result.communication:
        lines.append("")
        metrics = sorted({m for v in result.communication.values() for m in v})
        rows = [
            [system, *(f(values.get(m)) for m in metrics)]
            for system, values in result.communication.items()
        ]
        lines += _table(["system", *metrics], rows)
    return "\n".join(lines)
