"""Test the synthetic records pipeline: gold labels are right, splits are disjoint."""

import datetime
import json
import re

import pytest

from genetic_coil_arena.experiments.records import (
    CANON_FIELDS,
    HELDOUT_AMOUNT_FORMATS,
    HELDOUT_DATE_FORMATS,
    HELDOUT_VENDORS,
    RAW_FIELDS,
    TRAIN_AMOUNT_FORMATS,
    TRAIN_DATE_FORMATS,
    TRAIN_VENDORS,
    canonical_text,
    make_records,
    parse_fields,
    raw_json,
    score_fields,
    summarize_scores,
)

DATE_PARSERS = {
    "iso": lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"),
    "us_slash": lambda s: datetime.datetime.strptime(s, "%m/%d/%Y"),
    "dmy_long": lambda s: datetime.datetime.strptime(s, "%d %B %Y"),
    "mdy_short": lambda s: datetime.datetime.strptime(s, "%b %d, %Y"),
    "dot_dmy": lambda s: datetime.datetime.strptime(s, "%d.%m.%Y"),
    "dmy_abbr2": lambda s: datetime.datetime.strptime(s, "%d-%b-%y"),
    "mdy_ordinal": lambda s: datetime.datetime.strptime(
        re.sub(r"(\d)(st|nd|rd|th)", r"\1", s), "%B %d, %Y"
    ),
    "compact": lambda s: datetime.datetime.strptime(s, "%Y%m%d"),
}
SYMBOL_TO_CURRENCY = {"C$": "CAD", "A$": "AUD", "$": "USD", "€": "EUR", "£": "GBP"}


def parse_amount(text):
    """Independent amount reader: returns (cents, currency)."""
    currency = None
    rest = text
    for code in ("USD", "EUR", "GBP", "CAD", "AUD"):
        if code in rest:
            currency, rest = code, rest.replace(code, "")
    for symbol in ("C$", "A$", "$", "€", "£"):
        if symbol in rest:
            currency, rest = SYMBOL_TO_CURRENCY[symbol], rest.replace(symbol, "")
    rest = rest.strip()
    if re.search(r",\d\d$", rest):  # decimal comma: dots or spaces group thousands
        whole, frac = rest.rsplit(",", 1)
        whole = whole.replace(".", "").replace(" ", "")
    else:
        whole, frac = rest.rsplit(".", 1)
        whole = whole.replace(",", "")
    return int(whole) * 100 + int(frac), currency


@pytest.mark.parametrize("split", ["train", "id_test", "heldout_test"])
def test_gold_labels_match_what_is_written(split):
    for rec in make_records(300, split, seed=3):
        parse_date = DATE_PARSERS[rec.meta["date_format"]]
        assert parse_date(rec.raw["date"]).date().isoformat() == rec.gold["date_iso"]
        cents, currency = parse_amount(rec.raw["amount"])
        assert (cents, currency) == (rec.gold["amount_cents"], rec.gold["currency_iso"])
        assert rec.raw["invoice_id"].upper() == rec.gold["invoice_id"]
        for name in ("vendor", "date", "amount", "invoice_id"):
            assert rec.raw[name] in rec.text
        assert rec.gold["vendor_canonical"] == rec.meta["vendor"]
        assert set(rec.gold) == set(CANON_FIELDS) and set(rec.raw) == set(RAW_FIELDS)


def test_generation_is_reproducible_and_seeds_differ():
    a = make_records(20, "train", 7)
    assert [r.text for r in a] == [r.text for r in make_records(20, "train", 7)]
    assert [r.text for r in a] != [r.text for r in make_records(20, "train", 8)]
    assert [r.text for r in a] != [r.text for r in make_records(20, "id_test", 7)]


def test_heldout_split_uses_only_unseen_vendors_and_formats():
    assert not set(TRAIN_VENDORS) & set(HELDOUT_VENDORS)
    assert not set(TRAIN_DATE_FORMATS) & set(HELDOUT_DATE_FORMATS)
    assert not set(TRAIN_AMOUNT_FORMATS) & set(HELDOUT_AMOUNT_FORMATS)
    held = make_records(300, "heldout_test", 1)
    assert {r.meta["vendor"] for r in held} <= set(HELDOUT_VENDORS)
    assert {r.meta["date_format"] for r in held} == set(HELDOUT_DATE_FORMATS)
    assert {r.meta["amount_format"] for r in held} == set(HELDOUT_AMOUNT_FORMATS)
    for split in ("train", "id_test"):
        recs = make_records(300, split, 1)
        assert {r.meta["vendor"] for r in recs} <= set(TRAIN_VENDORS)
        assert {r.meta["date_format"] for r in recs} <= set(TRAIN_DATE_FORMATS)
        assert {r.meta["amount_format"] for r in recs} <= set(TRAIN_AMOUNT_FORMATS)


def test_day_first_dotted_dates_are_unambiguous():
    for rec in make_records(300, "heldout_test", 2):
        if rec.meta["date_format"] == "dot_dmy":
            assert int(rec.raw["date"][:2]) > 12


def test_snippets_contain_distractors():
    recs = make_records(100, "train", 4)
    assert all(
        len(re.findall(r"\d{4}-\d\d-\d\d|\d\d/\d\d/\d{4}|[A-Z][a-z]{2,8} \d", r.text))
        >= 1
        for r in recs
    )
    assert (
        sum(
            r.text.count("Subtotal")
            + r.text.count("Tax")
            + r.text.count("Shipping")
            + r.text.count("Deposit")
            > 0
            for r in recs
        )
        > 80
    )
    assert max(len(r.text) for r in recs) < 700


def test_vendor_aliases_vary_but_keep_the_canonical_recoverable():
    recs = make_records(400, "train", 5)
    aliases = {r.raw["vendor"] for r in recs}
    assert len(aliases) > len({r.meta["vendor"] for r in recs})
    exact = sum(r.raw["vendor"] == r.gold["vendor_canonical"] for r in recs)
    assert 0 < exact < len(recs)


def test_serialisation_is_compact_sorted_json():
    rec = make_records(1, "train", 0)[0]
    assert json.loads(raw_json(rec.raw)) == rec.raw
    assert list(json.loads(raw_json(rec.raw))) == sorted(RAW_FIELDS)
    text = canonical_text(rec.gold)
    assert json.loads(text) == rec.gold and ", " not in text and ": " not in text
    assert isinstance(json.loads(text)["amount_cents"], int)


class TestScoring:
    gold = {
        "vendor_canonical": "Blue Harbor Supply",
        "date_iso": "2024-03-14",
        "amount_cents": 123450,
        "currency_iso": "USD",
        "invoice_id": "INV-1",
    }

    def test_perfect_output(self):
        s = score_fields(canonical_text(self.gold), self.gold)
        assert s["exact"] and s["valid"] and all(s["fields"].values())

    def test_wrong_field_and_wrong_type(self):
        wrong = dict(self.gold, date_iso="2024-03-15", amount_cents="123450")
        s = score_fields(json.dumps(wrong), self.gold)
        assert not s["exact"] and s["valid"]
        assert not s["fields"]["date_iso"] and not s["fields"]["amount_cents"]
        assert s["fields"]["vendor_canonical"] and s["fields"]["invoice_id"]

    def test_malformed_json_still_scores_readable_fields(self):
        broken = canonical_text(self.gold)[:-12]  # cut off inside the vendor
        s = score_fields(broken, self.gold)
        valid, fields = parse_fields(broken, CANON_FIELDS)
        assert not valid and not s["exact"] and not s["valid"]
        assert s["fields"]["date_iso"] and s["fields"]["amount_cents"]
        assert not s["fields"]["vendor_canonical"]
        assert fields["currency_iso"] == "USD"

    def test_garbage_and_missing_keys(self):
        s = score_fields("not json at all", self.gold)
        assert not any(s["fields"].values()) and not s["valid"]
        partial = score_fields(json.dumps({"date_iso": "2024-03-14"}), self.gold)
        assert (
            partial["valid"] and not partial["exact"] and partial["fields"]["date_iso"]
        )

    def test_summary(self):
        good = score_fields(canonical_text(self.gold), self.gold)
        bad = score_fields("{}", self.gold)
        summary = summarize_scores([good, bad, good, good], CANON_FIELDS)
        assert (
            summary["n"] == 4 and summary["exact"] == 0.75 and summary["valid"] == 1.0
        )
        assert summary["fields"]["date_iso"] == 0.75
        with pytest.raises(ValueError):
            summarize_scores([], CANON_FIELDS)
