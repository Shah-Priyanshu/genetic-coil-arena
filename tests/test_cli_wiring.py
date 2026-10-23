"""Integration wiring: lazy package imports and the protocol/squad CLI commands."""

import json
import subprocess
import sys

from typer.testing import CliRunner

from genetic_coil_arena.cli.main import app


def test_domain_agnostic_packages_do_not_load_snake_or_numba():
    code = (
        "import sys\n"
        "import genetic_coil_arena.evaluation, genetic_coil_arena.protocol, "
        "genetic_coil_arena.runtime, genetic_coil_arena.learning\n"
        "loaded = [m for m in ('genetic_coil_arena.snake_core', 'numba') "
        "if m in sys.modules]\n"
        "print(','.join(loaded))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"loaded eagerly: {out.stdout.strip()}"


def test_top_level_names_still_resolve():
    import genetic_coil_arena as gca
    from genetic_coil_arena.evo.arena import EvolutionArena
    from genetic_coil_arena.snake_core.game import SnakeGame

    assert gca.SnakeGame is SnakeGame
    assert gca.EvolutionArena is EvolutionArena
    assert {"Config", "EvolutionArena", "SnakeGame", "SnakeAgent"} <= set(dir(gca))


def test_protocol_demo_writes_trace(tmp_path):
    trace = tmp_path / "trace.jsonl"
    result = CliRunner().invoke(app, ["protocol-demo", "--trace", str(trace)])
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert records


def test_squad_benchmark_small(tmp_path):
    out = tmp_path / "need_to_talk.md"
    result = CliRunner().invoke(
        app,
        ["squad-benchmark", "-n", "2", "-r", "3", "--no-throughput", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    report = out.read_text(encoding="utf-8")
    assert "| 3 | greedy + hand-coded centre (3 bits) |" in report
    assert "| 5 |" not in report  # Only the requested radius (oracle row is "full")
    assert "oracle" in report and "Throughput" not in report


def test_learned_protocol_tiny(tmp_path, monkeypatch):
    monkeypatch.delenv("GCA_RUN_DIR", raising=False)
    out = tmp_path / "lp.md"
    args = ["learned-protocol", "--scale", "smoke", "--generations", "2"]
    args += ["--eval-every", "1", "--eval-episodes", "2", "--entropy-bonus", "0.5"]
    result = CliRunner().invoke(app, args + ["--out", str(out)])
    assert result.exit_code == 0, result.output
    report = out.read_text(encoding="utf-8")
    assert "| learned-centre+learned-spec | system |" in report
    assert "--entropy-bonus 0.5 --entropy-anneal 0.5" in report  # reproduce line
    assert (tmp_path / "lp_params.npz").exists()
    assert (tmp_path / "lp_curves.csv").exists()
