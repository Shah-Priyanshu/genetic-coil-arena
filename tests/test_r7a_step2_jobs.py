"""The R7a step-2 job script end to end, with scripted models and fast fake executors.

Runs ``eval-lora``, ``eval-ollama``, ``eval-lora-rt`` and ``aggregate`` through ``main()``
so the file layout between jobs (``runs:`` globs, ``artifacts`` folders, adapters outside the
artifacts) and the report are exercised without Ollama, a GPU or the TEP data.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from genetic_coil_arena.experiments import r7a_sealed
from genetic_coil_arena.experiments.r7a_domains import build_domains, build_registry
from genetic_coil_arena.experiments.r7a_train_data import DEFAULTS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def job():
    spec = importlib.util.spec_from_file_location(
        "r7a_step2_script", ROOT / "scripts" / "r7a_step2.py"
    )
    sys.path.insert(0, str(ROOT / "scripts"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Oracle:
    """Replies like a perfect translator, except that it clarifies on every 7th spec item."""

    def __init__(self):
        self.by_text = {i.text: i for i in r7a_sealed.load_sealed()}
        self.summaries: dict[str, dict] = {}
        self.n = 0

    def reply_for(self, text: str) -> dict:
        if text in self.summaries:
            return {"refusal": "", "clarify": "", "params": self.summaries[text]}
        item = self.by_text.get(text)
        if item is None:
            return {"refusal": "unknown", "clarify": ""}
        self.n += 1
        if item.expected == "refuse":
            return {"refusal": "not possible", "clarify": ""}
        if item.expected == "clarify":
            return {"refusal": "", "clarify": "which one?"}
        if self.n % 7 == 0:
            return {"refusal": "", "clarify": "which one?"}
        return {"refusal": "", "clarify": "", "params": item.gold}

    def summary_for(self, prompt: str) -> str:
        trace = json.loads(prompt)
        p = trace["goal"]["params"]
        score = trace["outcome"]["score"]
        if "eaters" in p:
            names = " and ".join(p["eaters"])
            target = (
                "no food target"
                if p["food_target"] == 0
                else f"a food target of {p['food_target']}"
            )
            text = (
                f"The {names} snakes were ordered to forage at caution level {p['caution']} "
                f"with a step limit of {p['step_limit']} and {target}; the squad finished with mission score {score}."
            )
        else:
            nodes = " and ".join(p["active_nodes"])
            text = (
                f"The diagnosis ran with {p['byte_budget']} bytes per message and {p['target_far'] * 100:g}% "
                f"false alarms on the {nodes} nodes, reaching macro-F1 {score}."
            )
        self.summaries[text] = p
        return text


class FakeOllamaBackend:
    oracle = Oracle()

    def __init__(self, model, **kw):
        self.base_id = model

    def generate(self, prompts, **kw):
        out = []
        for prompt in prompts:
            if kw.get("system", "").startswith("You write the status report"):
                text = self.oracle.summary_for(prompt)
            else:
                m = re.search(r"Instruction: (.*)$", prompt, re.S)
                text = json.dumps(self.oracle.reply_for(m.group(1) if m else prompt))
            out.append(SimpleNamespace(text=text, latency_s=0.01, n_prompt_tokens=5))
        return out

    def unload(self):
        pass


class FakeServer:
    def __init__(self, host):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def require(self, model):
        pass

    def unload_all(self):
        pass


class FakeHF:
    base_id = "fake/base"
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    oracle = Oracle()

    def load_adapter(self, name, path):
        self.loaded = (name, str(path))

    def set_adapter(self, name):
        pass

    def reset_peak_memory(self):
        pass

    def peak_memory_mib(self):
        return 3500.0

    def generate(self, prompts, adapter_id, max_new_tokens):
        out = []
        for prompt in prompts:
            m = re.search(r"Request: (.*)\nReply:\n$", prompt, re.S)
            text = m.group(1) if m else prompt
            reply = self.oracle.reply_for(text)
            # the distilled model writes the compact reply: params without defaults
            if reply.get("params"):
                task = re.match(r"Task: (\S+)", prompt).group(1)
                dom = "squad" if task.startswith("squad") else "plant"
                reply = {
                    "params": {
                        k: v
                        for k, v in reply["params"].items()
                        if k not in DEFAULTS[dom] or v != DEFAULTS[dom][k]
                    }
                }
            out.append(
                SimpleNamespace(
                    text=json.dumps(reply), latency_s=0.01, n_prompt_tokens=5
                )
            )
        return out


class FastExec:
    """Deterministic fake executor: the score depends on the params only."""

    def __init__(self, domain):
        self.domain = domain

    def __call__(self, params):
        base = float(sum(len(str(v)) for v in params.values())) / 10
        per = [base + 0.1 * k for k in range(3)]
        if self.domain == "squad":
            return {
                "score": sum(per) / 3,
                "food": 1.0,
                "survivors": 1.0,
                "steps": 10.0,
                "per_seed": per,
            }
        return {"score": sum(per) / 3, "false_alarm_rate": 0.01, "detection_rate": 0.9,
                "bytes_per_node": 4.0, "per_seed": per}  # fmt: skip


@pytest.fixture
def patched(job, monkeypatch):
    FakeOllamaBackend.oracle = FakeHF.oracle = Oracle()
    monkeypatch.setattr(job, "OllamaServer", FakeServer)
    monkeypatch.setattr(job, "OllamaBackend", FakeOllamaBackend)
    monkeypatch.setattr(job, "open_hf", lambda args: FakeHF())

    def make_domains(args, execute):
        domains = build_domains(None, 1, 1)
        for name, dom in domains.items():
            dom.executor = FastExec(name)
        return build_registry(), domains

    monkeypatch.setattr(job, "make_domains", make_domains)
    monkeypatch.setattr(job, "VramMonitor", lambda: SimpleNamespace(
        available=False, read=lambda: None, start=lambda: None, stop=lambda: 0.0))  # fmt: skip

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["r7a_step2.py", *map(str, argv)])
        job.main()

    return run


def make_train_runs(root: Path):
    for name, score in (("r7a2-train-r8-n1000", 0.80), ("r7a2-train-r32-n4000", 0.91)):
        art = root / "runs" / name / "artifacts"
        art.mkdir(parents=True)
        (art / "dev.json").write_text(
            json.dumps(
                {
                    "name": name.replace("r7a2-train-", ""),
                    "config": {
                        "rank": 8,
                        "n": 1000,
                        "steps": 100,
                        "paraphrase_share": 0.5,
                    },
                    "train": {
                        "mean_last_loss": 0.01,
                        "wall_s": 60,
                        "peak_mib": 4000,
                        "final_loss": 0.01,
                    },
                    "dev": {
                        "score": score,
                        "refusal": {"precision": 0.9, "recall": 1.0},
                    },
                    "adapter_path": str(root / "store" / f"{name}.pt"),
                }
            ),
            encoding="utf-8",
        )


def test_the_whole_job_chain(patched, tmp_path, monkeypatch):
    make_train_runs(tmp_path)
    runs = tmp_path / "runs"
    for job_id in (
        "r7a2-eval-lora",
        "r7a2-eval-ollama",
        "r7a2-eval-lora-rt",
        "r7a2-zz-aggregate",
    ):
        (runs / job_id / "artifacts").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    def env_for(job_id):
        monkeypatch.setenv("GCA_RUN_DIR", str(runs / job_id / "artifacts"))

    env_for("r7a2-eval-lora")
    patched("eval-lora", "--limit", "60")
    lora_art = runs / "r7a2-eval-lora" / "artifacts"
    lora = json.loads((lora_art / "lora_forward.json").read_text(encoding="utf-8"))
    assert (
        json.loads((lora_art / "best.json").read_text(encoding="utf-8"))["name"]
        == "r32-n4000"
    )
    assert lora["smoke"] and lora["sealed_sha256"] == r7a_sealed.SEALED_SHA256
    assert lora["vram"]["peak_allocated_mib"] == 3500.0
    assert all("nl_view" not in r for r in lora["records"])

    env_for("r7a2-eval-ollama")
    patched("eval-ollama", "--limit", "60")
    oll_art = runs / "r7a2-eval-ollama" / "artifacts"
    ollama = json.loads((oll_art / "ollama_eval.json").read_text(encoding="utf-8"))
    assert (
        set(ollama["systems"]) == {"gemma", "o14", "regex"} and ollama["lora_included"]
    )
    summaries = json.loads((oll_art / "summaries.json").read_text(encoding="utf-8"))
    assert set(summaries) == {"gemma", "o14", "regex", "lora"}
    assert summaries["lora"] and all(v["summary"] for v in summaries["o14"].values())
    o14 = ollama["systems"]["o14"]
    assert (
        o14["reverse"]["all"]["round_trip"] > 0.9
        and o14["reverse"]["all"]["field_fidelity"] == 1.0
    )

    env_for("r7a2-eval-lora-rt")
    patched("eval-lora-rt")
    rt = json.loads(
        (runs / "r7a2-eval-lora-rt" / "artifacts" / "lora_rt.json").read_text(
            encoding="utf-8"
        )
    )
    assert rt["reverse"]["all"]["round_trip"] > 0.9

    env_for("r7a2-zz-aggregate")
    patched("aggregate", "--stem", "docs/results/r7a_step2")
    md = (tmp_path / "docs" / "results" / "r7a_step2.md").read_text(encoding="utf-8")
    report = json.loads(
        (tmp_path / "docs" / "results" / "r7a_step2.json").read_text(encoding="utf-8")
    )
    assert set(report["systems"]) == {"o14", "lora", "gemma", "regex"}
    assert "SMOKE RUN" in md and "Qwen3-1.7B + LoRA" in md and "r32-n4000" in md
    for key in ("o14", "lora"):
        assert report["systems"][key]["verdict"]["label"].startswith("SMOKE")
    ex = report["systems"]["lora"]["forward"]["exact"]["sealed"]
    assert 0.7 < ex["exact"] <= 1.0  # an oracle with a clarify slip every 7th spec item
    assert report["systems"]["lora"]["forward"]["clarify"]["asked"] > 0
    assert (runs / "r7a2-zz-aggregate" / "artifacts" / "r7a_step2.md").exists()


def test_train_refuses_a_pool_that_overlaps_the_sealed_set(job, tmp_path, monkeypatch):
    from genetic_coil_arena.experiments import r7a_train_data as td

    leak = td.Record(
        "leak",
        "squad",
        "spec",
        "paraphrase",
        "p",
        r7a_sealed.load_sealed()[0].text,
        "{}",
    )
    art = tmp_path / "pool"
    art.mkdir()
    (art / "pool.jsonl").write_text(leak.to_json() + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "r7a_step2.py",
            "--out",
            str(tmp_path / "o"),
            "train",
            "--pool",
            str(art),
            "--n",
            "10",
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        job.main()
