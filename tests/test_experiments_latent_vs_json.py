"""Test the R9 experiment's pure logic with fake pipelines (no model, no torch)."""

import numpy as np
import pytest

from genetic_coil_arena.experiments.latent_vs_json import (
    Condition,
    Pipeline,
    Response,
    derangement,
    evaluate,
    measure_latency,
)
from genetic_coil_arena.experiments.records import canonical_text, make_records


class Oracle(Pipeline):
    """Answers correctly, except it corrupts the amount of every third record."""

    def __init__(self, shuffled=False):
        self.shuffled = shuffled
        self.calls = []

    def respond(
        self,
        records,
        *,
        ablation=None,
        rng=None,
        wire_dtype="bfloat16",
        through_protocol=False,
    ):
        self.calls.append((len(records), ablation, wire_dtype, through_protocol))
        outputs = []
        for i, rec in enumerate(records):
            gold = dict(rec.gold)
            if i % 3 == 0:
                gold["amount_cents"] += 1
            outputs.append(canonical_text(gold))
        n = len(records)
        return Response(
            outputs=outputs,
            timings={"sender": 0.5, "handoff": 0.25, "receiver": 1.0},
            message_bytes=[100] * n,
            message_tokens=[40] * n,
        )


def test_derangement_has_no_fixed_points():
    rng = np.random.default_rng(0)
    for n in (2, 3, 10, 64):
        for _ in range(20):
            perm = derangement(n, rng)
            assert sorted(perm) == list(range(n)) and not (perm == np.arange(n)).any()
    with pytest.raises(ValueError):
        derangement(1, rng)


def test_condition_names():
    assert Condition("json").name == "json"
    assert Condition("kv").name == "kv_prefix"
    assert Condition("single").name == "single"
    assert Condition("latent", k=4).name == "latent_k4"
    assert Condition("latent", k=16, layer=28, tag="L28").name == "latent_k16_L28"
    assert Condition("latent", k=4, pooling="attn", tag="attn").name == "latent_k4_attn"


def test_evaluate_scores_batches_and_sizes():
    splits = {s: make_records(10, s, 0) for s in ("id_test", "heldout_test")}
    pipe = Oracle()
    out = evaluate(pipe, splits, batch_size=4, keep_examples=2)
    for split in splits:
        summary = out[split]
        # records 0, 3 of each batch of 4 and record 0 of the batch of 2 are corrupted
        assert summary["n"] == 10 and summary["exact"] == pytest.approx(0.5)
        assert summary["fields"]["amount_cents"] == pytest.approx(0.5)
        assert summary["fields"]["vendor_canonical"] == 1.0 and summary["valid"] == 1.0
        assert summary["message_bytes"] == 100 and summary["message_tokens"] == 40
        assert len(summary["examples"]) == 2 and "stage_a" not in summary
    assert [c[0] for c in pipe.calls] == [4, 4, 2, 4, 4, 2]


def test_evaluate_passes_ablation_and_skips_a_lone_record_batch():
    splits = {"id_test": make_records(5, "id_test", 0)}
    pipe = Oracle()
    out = evaluate(pipe, splits, batch_size=4, ablation="shuffle", wire_dtype="int8")
    assert out["id_test"]["n"] == 4  # the trailing batch of 1 cannot be shuffled
    assert all(c[1:3] == ("shuffle", "int8") for c in pipe.calls)


def test_measure_latency_sums_stage_times_and_uses_the_protocol():
    pipe = Oracle()
    recs = make_records(4, "id_test", 0)
    out = measure_latency(pipe, recs, warmup=2)
    assert out["n"] == 4 and out["total_mean_s"] == pytest.approx(1.75)
    assert out["stage_mean_s"] == {"sender": 0.5, "handoff": 0.25, "receiver": 1.0}
    assert len(pipe.calls) == 6 and all(c[3] for c in pipe.calls)  # 2 warm-up + 4 timed
