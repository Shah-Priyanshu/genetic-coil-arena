"""Per-run result cache, so an interrupted multi-run training job resumes.

Each finished training run is written at once to a cache folder as ``<key>.npz``
(parameter arrays) and ``<key>.json`` (the settings it was trained with and its JSON
summary), both via an atomic rename, the JSON last. A rerun asks ``load`` for each run
first: a hit with identical settings is reused and only the missing runs train, so a
restart loses only the runs that were in progress.

    hit = load(cache, key, settings)
    if hit is None:
        out = train(...)
        save(cache, key, settings, {"theta": out["theta"]}, summary)
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np


def finite(value: Any) -> Any:
    """Copy of a JSON-like structure with non-finite floats (inf, NaN) as None."""
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def default_cache_dir(stem: str) -> Path:
    """``$GCA_RUN_DIR/<stem>_runs`` when the GPU runner set it (that folder survives a
    rerun of the same job), else ``outputs/<stem>_runs`` (git-ignored)."""
    base = os.environ.get("GCA_RUN_DIR")
    return (Path(base) if base else Path("outputs")) / f"{stem}_runs"


def load(cache: Path, key: str, settings: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The cached run ``key`` if it was trained with exactly ``settings``, else None.

    Returns the saved summary plus its arrays, and ``cached=True``.
    """
    meta_path, arrays_path = cache / f"{key}.json", cache / f"{key}.npz"
    if not (meta_path.exists() and arrays_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta.get("settings") != finite(settings):
        return None
    with np.load(arrays_path) as arrays:
        loaded = {name: arrays[name] for name in arrays.files}
    return {**meta["summary"], **loaded, "cached": True}


def save(
    cache: Path,
    key: str,
    settings: dict[str, Any],
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> None:
    """Write run ``key`` (arrays first, then the JSON that marks it complete)."""
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / f"{key}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, cache / f"{key}.npz")
    meta = {"settings": finite(settings), "summary": finite(summary)}
    tmp_json = cache / f"{key}.tmp.json"
    tmp_json.write_text(json.dumps(meta, allow_nan=False), encoding="utf-8")
    os.replace(tmp_json, cache / f"{key}.json")
