"""R24: verdict and report rendering for the MaixCAM-compiled 448 brain."""

from genetic_coil_arena.experiments import r24_maixcam as r24

BINS = ("all", "0-8", "8-16", "16-32", "32+")


def _row(drop: float, false_rel: float, passes: bool, kill: bool = False) -> dict:
    test = {f"recall {b}": {"float": 0.3, "int8": 0.3 - drop} for b in BINS}
    test["false_per_tile"] = {"float": 0.26, "int8": 0.26 * (1 + false_rel)}
    return {"int8_tau": 0.5, "test": test, "recall_drop": drop, "false_rel": false_rel,
            "passes": passes, "kill": kill,
            "compile": {"cvimodel_bytes": 533736, "ion_mb": 2.47}}  # fmt: skip


def test_verdict_needs_both_seeds():
    ok = _row(0.01, 0.05, True)
    assert r24.verdict([ok]) == "NOT PASSED"
    assert r24.verdict([ok, ok]) == "PASS"
    assert r24.verdict([ok, _row(0.03, 0.0, False)]) == "NOT PASSED"
    assert r24.verdict([ok, _row(0.06, 0.0, False, kill=True)]) == "KILL"


def test_render_lists_seeds_and_memory():
    rep = {"verdict": "PASS", "models": {t: _row(0.01, 0.05, True) for t in r24.MODELS}}
    text = r24.render(rep)
    assert "| 0 | 0.500 |" in text and "| 1 | 0.500 |" in text
    assert "521 KB | 2.47 MB | yes" in text


def test_render_b_names_the_variant_and_levels():
    rep = {"verdict": "NOT PASSED", "variant": r24.HEATCAL,
           "models": {t: {**_row(0.03, 0.0, False), "val_levels": 4} for t in r24.MODELS}}  # fmt: skip
    text = r24.render(rep)
    assert text.startswith("# R24b:") and "0.56 on val: 4 / 4." in text
    assert r24.render({**rep, "variant": ""}).startswith("# R24:")
