"""R7a step 2: the sealed test set is intact, valid and independent of everything tuned on."""

import json
from collections import Counter

import pytest

from genetic_coil_arena.experiments import r7a_sealed
from genetic_coil_arena.experiments.r7a_domains import (
    COLOURS,
    PLANT_DIAGNOSE,
    SQUAD_MISSION,
    build_registry,
)
from genetic_coil_arena.experiments.r7a_evalset import few_shot_texts
from genetic_coil_arena.translator import OverlapIndex

MASKS = {
    **dict.fromkeys(COLOURS, "<C>"),
    **dict.fromkeys(
        (
            "reactor",
            "reactr",
            "separator",
            "sepparator",
            "stripper",
            "compressor",
            "utilities",
        ),
        "<K>",
    ),
}


@pytest.fixture(scope="module")
def sealed():
    return r7a_sealed.load_sealed()


def test_sealed_file_hash_is_pinned():
    # Changing the file (even a typo fix) changes this hash; that is the point.
    assert r7a_sealed.sealed_sha256() == r7a_sealed.SEALED_SHA256
    assert (
        r7a_sealed.SEALED_SHA256
        == "ca6ff07b5fb66ef21c288ba9a82d5e7763992fe2142b1a582baf597e542b5e01"
    )


def test_hash_ignores_checkout_line_endings(tmp_path):
    crlf = tmp_path / "x.jsonl"
    crlf.write_bytes(r7a_sealed.SEALED_PATH.read_bytes().replace(b"\n", b"\r\n"))
    assert r7a_sealed.sealed_sha256(crlf) == r7a_sealed.SEALED_SHA256


def test_loading_refuses_a_modified_file(tmp_path):
    edited = tmp_path / "x.jsonl"
    edited.write_text(
        r7a_sealed.SEALED_PATH.read_text(encoding="utf-8").replace("red", "rad", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pinned SHA-256"):
        r7a_sealed.load_sealed(path=edited)


def test_size_and_composition(sealed):
    assert len(sealed) >= 150
    kinds = Counter(i.expected for i in sealed)
    assert kinds["refuse"] / len(sealed) == pytest.approx(0.10, abs=0.04)
    assert kinds["clarify"] >= 6
    for domain in ("squad", "plant"):
        assert sum(1 for i in sealed if i.domain == domain) >= 75
    assert len({i.id for i in sealed}) == len(sealed)
    assert len({i.text for i in sealed}) == len(sealed)
    assert {i.split for i in sealed} == {"sealed"}


def test_styles_cover_what_the_brief_asks(sealed):
    styles = {i.style for i in sealed}
    for needed in (
        "typo",
        "words",
        "mixed_units",
        "indirect",
        "multi_sentence",
        "question",
        "negation",
        "self_correction",
    ):
        assert needed in styles, needed


def test_gold_specs_validate_through_the_registry(sealed):
    registry = build_registry()
    names = {"squad": SQUAD_MISSION, "plant": PLANT_DIAGNOSE}
    for item in sealed:
        if item.expected == "spec":
            registry.make(names[item.domain], item.gold)  # raises when invalid
        else:
            assert item.gold is None


def test_every_row_has_a_known_kind(sealed):
    rows = [
        json.loads(x)
        for x in r7a_sealed.SEALED_PATH.read_text(encoding="utf-8").splitlines()
        if x
    ]
    assert {r["kind"] for r in rows} == {"spec", "refuse", "clarify"}
    assert all(r["gold"] is None for r in rows if r["kind"] != "spec")


def test_sealed_texts_are_not_in_the_step1_set_dev_set_or_few_shots(sealed):
    texts = {i.text for i in sealed}
    old = {i.text for i in r7a_sealed.load_old_set()}
    dev = {i.text for i in r7a_sealed.load_dev()}
    assert not texts & old
    assert not texts & dev
    assert not texts & few_shot_texts()


def test_sealed_frames_are_far_from_the_step1_set_and_dev_set(sealed):
    sealed_texts = [i.text for i in sealed]
    index = OverlapIndex(sealed_texts, 4, MASKS)
    others = [i.text for i in r7a_sealed.load_old_set() + r7a_sealed.load_dev()]
    worst = max(index.max_containment(t)[0] for t in others)
    assert worst < 0.5, worst


def test_dev_set_is_a_fresh_draw(sealed):
    dev = r7a_sealed.load_dev()
    assert len(dev) >= 100
    assert {i.split for i in dev} == {"dev"}
    assert len({i.id for i in dev}) == len(dev)
    old = {i.text for i in r7a_sealed.load_old_set()}
    assert not {i.text for i in dev} & old
