"""G12: decoder, writer, verdict and report plumbing (synthetic data and untrained nets)."""

import numpy as np
import pytest

from genetic_coil_arena.experiments import r7d_legibility as r7d
from genetic_coil_arena.experiments.r7b_goal_conditioned import (
    BLIND,
    CENTRE_GOAL,
    HAND_GOAL,
    GoalNets,
    GoalTrainSettings,
)

VOCAB = 8


def _rec(eaters, caution, symbol_eat=3, symbol_no=5, steps=4):
    eaters = np.asarray(eaters, dtype=bool)
    b, n = eaters.shape
    symbols = np.where(eaters[:, None, :], symbol_eat, symbol_no).repeat(steps, axis=1)
    return {
        "symbols": symbols.astype(np.int64),
        "situation": np.zeros_like(symbols),
        "live": np.ones(symbols.shape, dtype=bool),
        "eaters": eaters,
        "caution": np.asarray(caution, dtype=np.int64),
    }


def test_symbol_counts_skip_dead_receivers():
    symbols = np.array([[[1, 2], [1, 2], [4, 2]]])  # (B=1, T=3, N=2)
    live = np.array([[[True, True], [True, False], [False, False]]])
    c = r7d.symbol_counts(symbols, live, VOCAB)
    assert c.shape == (1, 2, VOCAB)
    assert c[0, 0, 1] == 2 and c[0, 0, 4] == 0 and c[0, 1, 2] == 1


def test_decoder_reads_a_clean_code_and_never_returns_an_empty_set():
    eaters = [[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 1]] * 3
    rec = _rec(eaters, [0, 1, 2, 0] * 3)
    dec = r7d.fit_decoder(rec, VOCAB)
    scores = r7d.read_scores(dec, rec, VOCAB)
    assert scores["eaters_exact"] == 1.0 and scores["message_bal_acc"] == 1.0
    silent = _rec([[1, 0, 0]], [0], symbol_eat=0, symbol_no=0)
    pred = dec.eaters(r7d.symbol_counts(silent["symbols"], silent["live"], VOCAB))
    assert pred.sum() == 1


def test_shuffled_labels_break_the_decoder():
    rng = np.random.default_rng(0)
    eaters = rng.random((400, 3)) < 0.5
    eaters[~eaters.any(axis=1), 0] = True
    rec = _rec(eaters, rng.integers(0, 3, 400))
    true = r7d.fit_decoder(rec, VOCAB)
    null = r7d.fit_decoder(rec, VOCAB, shuffle=np.random.default_rng(1))
    gap = lambda d: abs(d.status_llr[3] - d.status_llr[5])  # noqa: E731
    assert gap(true) > 5 and gap(null) < 0.3


def test_writer_takes_the_most_frequent_symbol_ties_to_lowest():
    rec = _rec([[1, 0, 0]], [0], steps=2)
    rec["situation"][:] = 2
    dec = r7d.fit_decoder(rec, VOCAB)
    assert dec.writer[r7d.EAT, 2] == 3 and dec.writer[r7d.NO, 2] == 5
    assert dec.writer[r7d.EAT, 0] == 0  # unseen situation: all counts tie


def test_write_efficacy():
    intact = {
        "food": np.array([[10.0, 0.0], [12.0, 2.0]]),
        "eaters": np.array([[True, False], [True, False]]),
    }
    got = r7d.write_efficacy(
        {"eat": np.array([6.0, 4.0]), "no": np.array([1.0, 1.0])}, intact
    )
    assert got["efficacy"] == pytest.approx(4.0 / 10.0)


@pytest.mark.parametrize(
    "read, write, verdict",
    [
        ([0.95, 0.93], [0.6, 0.7], "PASS"),
        ([0.95, 0.93], [0.4, 0.4], "NOT PASSED"),
        ([0.5, 0.6], [0.9, 0.9], "KILL"),
        ([0.95, 0.95], [0.1, 0.2], "KILL"),
    ],
)
def test_verdict(read, write, verdict):
    assert r7d.g12_verdict(read, write)["verdict"] == verdict


SMALL = GoalTrainSettings(max_steps=40, spec_hidden=4, centre_hidden=4)
LS = r7d.LegibilitySettings(
    window=10, fit_per_goal=2, test_per_goal=2, n_random_writers=1
)


def _run(mode, run, best_val=20.0):
    theta = GoalNets.build(SMALL, mode).init(run)
    return {"mode": mode, "run": run, "best_val_mission": best_val, "theta": theta}


def test_hand_coded_centre_is_read_and_its_writer_is_its_own_code():
    row = r7d.evaluate_run((_run(HAND_GOAL, 0), SMALL.__dict__, LS))
    assert row["read"]["eaters_exact"] == 1.0
    assert set(row["writer"][r7d.NO]) == {0}
    assert all(v >= 1 for v in row["writer"][r7d.EAT])
    assert len(row["decoded_specs"]) == 6


def test_report_end_to_end():
    jobs = [
        _run(CENTRE_GOAL, 0),
        _run(CENTRE_GOAL, 1),
        _run(CENTRE_GOAL, 2, best_val=5.0),
        _run(BLIND, 0),
    ]
    rows = [r7d.evaluate_run((t, SMALL.__dict__, LS)) for t in jobs]
    rep = r7d.build_report(rows, SMALL.__dict__, LS)
    assert rep["g12"]["n"] == 2  # run 2 is below the validation flag
    assert rep["transfer"]["n"] == 2
    assert "_decoder" not in rep["runs"][0]
    md = r7d.render(rep)
    assert "G12 verdict" in md and "(a) blind-centre+learned-spec" in md
