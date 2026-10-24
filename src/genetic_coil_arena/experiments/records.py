"""A synthetic two-stage records pipeline (the data behind R9).

Raw invoice-like snippets are generated programmatically, with a fixed seed and gold
labels at both stages:

- **Stage A, the extractor,** reads the snippet and reports the four fields *as
  written*: ``{vendor, date, amount, invoice_id}`` (``Record.raw``). The snippet has
  distractor dates, amounts and ids, so picking the right ones takes reading.
- **Stage B, the normalizer,** never sees the snippet. It must produce the canonical
  record ``{vendor_canonical, date_iso, amount_cents, currency_iso, invoice_id}``
  (``Record.gold``) from A's message alone.

Three splits:

- ``train``: the training vendors with the training date and amount formats;
- ``id_test``: fresh records of the same kind (in-distribution);
- ``heldout_test``: vendors never seen in training, written in date and amount formats
  never seen in training (``HELDOUT_DATE_FORMATS``, ``HELDOUT_AMOUNT_FORMATS``).

Held-out formats stay unambiguous by construction: the day-first dotted date only
uses days above 12 (so it cannot be read month-first), and decimal-comma amounts
always carry two decimals. Pure Python; no ML dependencies.

Conventions: slash dates are month first; ``$`` is USD, ``C$`` is CAD, ``A$`` is AUD;
two-digit years are 20xx; the canonical vendor is the Title Case name without a legal
suffix and with ``and`` for ``&``; the canonical invoice id is upper case.
"""

import datetime
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from ..protocol.envelope import canonical_json

RAW_FIELDS = ("vendor", "date", "amount", "invoice_id")
CANON_FIELDS = (
    "vendor_canonical",
    "date_iso",
    "amount_cents",
    "currency_iso",
    "invoice_id",
)
SPLITS = ("train", "id_test", "heldout_test")

TRAIN_VENDORS = (
    "Northwind Traders",
    "Blue Harbor Supply",
    "Summit Ridge Logistics",
    "Maple Leaf Printing",
    "Redwood Analytics",
    "Silver Birch Catering",
    "Ironclad Security",
    "Granite Peak Hardware",
    "Lakeside Dental Group",
    "Copperfield Textiles",
    "Harvest Moon Foods",
    "Pioneer Freight",
    "Cobalt Cloud Services",
    "Evergreen Landscaping",
    "Falcon Office Supplies",
    "Juniper Consulting",
    "Orchard Street Bakery",
    "Starlight Cinemas",
    "Tidewater Plumbing",
    "Vantage Point Marketing",
    "Willow Creek Farms",
    "Zenith Electronics",
    "Alpine Outfitters",
    "Beacon Insurance Brokers",
    "Cascade Water Works",
    "Dune Ridge Solar",
    "Eastgate Veterinary Clinic",
    "Foxglove Florists",
    "Golden Gate Couriers",
    "Hawthorne and Sons",
    "Indigo Print Works",
    "Jasper Auto Repair",
    "Keystone Staffing",
    "Lantern Hill Books",
    "Meridian Travel",
    "Nimbus Data Systems",
    "Oakmont Legal Services",
    "Paragon Pest Control",
    "Quarry Lane Stone",
    "Riverbend Pharmacy",
    "Sagebrush Energy",
    "Trailhead Sports",
    "Umber Paint Company",
    "Velvet Moon Salon",
    "Westbrook Engineering",
    "Yellowstone Cabinets",
    "Anchor Point Marine",
    "Bramble and Thorn",
)
HELDOUT_VENDORS = (
    "Quicksilver Printing",
    "Marigold Kitchens",
    "Stonebridge Capital",
    "Thistle Down Linens",
    "Ashford Medical Supply",
    "Coral Bay Diving",
    "Driftwood Furniture",
    "Emberlight Studios",
    "Fernhill Nurseries",
    "Glacier Bay Freight",
    "Highland Tool and Die",
    "Ivory Coast Imports",
    "Kestrel Aviation",
    "Lumen Optics",
    "Mulberry Lane Press",
    "Nightingale Care Services",
    "Opal Harbor Seafood",
    "Pinecrest Roofing",
    "Rosewood Interiors",
    "Sandpiper Beach Rentals",
    "Topaz Analytics",
    "Upland Seed Company",
    "Wren and Finch",
    "Zephyr Wind Power",
)

LEGAL_SUFFIXES = (
    "",
    "",
    "",
    " Inc.",
    " Inc",
    " LLC",
    " Ltd",
    " Ltd.",
    " Co.",
    " Corp.",
    " GmbH",
)
CURRENCIES = ("USD", "EUR", "GBP", "CAD", "AUD")
CURRENCY_WEIGHTS = (0.40, 0.20, 0.15, 0.13, 0.12)
SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "C$", "AUD": "A$"}
SIMPLE_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}  # the rest fall back to the code

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
FIRST_DAY = datetime.date(2019, 1, 1)
LAST_DAY = datetime.date(2027, 12, 31)

_KEYBOARD_NEIGHBOURS = {
    "a": "sq",
    "e": "wr",
    "i": "uo",
    "o": "ip",
    "u": "yi",
    "n": "bm",
    "m": "nk",
    "r": "et",
    "t": "ry",
    "s": "ad",
    "l": "k",
    "d": "sf",
}


def _ordinal(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return f"{day}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


# ----------------------------------------------------------------------- date formats


def _fmt_iso(d: datetime.date) -> str:
    return d.isoformat()


def _fmt_us_slash(d: datetime.date) -> str:
    return f"{d.month:02d}/{d.day:02d}/{d.year}"


def _fmt_dmy_long(d: datetime.date) -> str:
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def _fmt_mdy_short(d: datetime.date) -> str:
    return f"{MONTHS[d.month - 1][:3]} {d.day}, {d.year}"


def _fmt_dot_dmy(d: datetime.date) -> str:
    return f"{d.day:02d}.{d.month:02d}.{d.year}"


def _fmt_dmy_abbr2(d: datetime.date) -> str:
    return f"{d.day:02d}-{MONTHS[d.month - 1][:3]}-{d.year % 100:02d}"


def _fmt_mdy_ordinal(d: datetime.date) -> str:
    return f"{MONTHS[d.month - 1]} {_ordinal(d.day)}, {d.year}"


def _fmt_compact(d: datetime.date) -> str:
    return f"{d.year}{d.month:02d}{d.day:02d}"


DATE_FORMATS: dict[str, Callable[[datetime.date], str]] = {
    "iso": _fmt_iso,
    "us_slash": _fmt_us_slash,
    "dmy_long": _fmt_dmy_long,
    "mdy_short": _fmt_mdy_short,
    "dot_dmy": _fmt_dot_dmy,
    "dmy_abbr2": _fmt_dmy_abbr2,
    "mdy_ordinal": _fmt_mdy_ordinal,
    "compact": _fmt_compact,
}
TRAIN_DATE_FORMATS = ("iso", "us_slash", "dmy_long", "mdy_short")
HELDOUT_DATE_FORMATS = ("dot_dmy", "dmy_abbr2", "mdy_ordinal", "compact")
# Formats whose reading is only unambiguous when the day is above 12
_DAY_ABOVE_12 = frozenset({"dot_dmy"})


# --------------------------------------------------------------------- amount formats


def _grouped(cents: int, thousands: str, decimal: str) -> str:
    whole, frac = divmod(cents, 100)
    digits = f"{whole:,}".replace(",", thousands)
    return f"{digits}{decimal}{frac:02d}"


def _fmt_sym_prefix(cents: int, cur: str) -> str:
    return f"{SYMBOLS[cur]}{_grouped(cents, ',', '.')}"


def _fmt_sym_prefix_plain(cents: int, cur: str) -> str:
    return f"{SYMBOLS[cur]}{_grouped(cents, '', '.')}"


def _fmt_code_suffix(cents: int, cur: str) -> str:
    return f"{_grouped(cents, '', '.')} {cur}"


def _fmt_code_suffix_sep(cents: int, cur: str) -> str:
    return f"{_grouped(cents, ',', '.')} {cur}"


def _fmt_code_prefix(cents: int, cur: str) -> str:
    return f"{cur} {_grouped(cents, ',', '.')}"


def _fmt_eu_suffix(cents: int, cur: str) -> str:
    return f"{_grouped(cents, '.', ',')} {SIMPLE_SYMBOLS.get(cur, cur)}"


def _fmt_space_comma_code(cents: int, cur: str) -> str:
    return f"{_grouped(cents, ' ', ',')} {cur}"


def _fmt_sym_suffix_plain(cents: int, cur: str) -> str:
    return f"{_grouped(cents, '', '.')}{SIMPLE_SYMBOLS.get(cur, cur)}"


AMOUNT_FORMATS: dict[str, Callable[[int, str], str]] = {
    "sym_prefix": _fmt_sym_prefix,
    "sym_prefix_plain": _fmt_sym_prefix_plain,
    "code_suffix": _fmt_code_suffix,
    "code_suffix_sep": _fmt_code_suffix_sep,
    "code_prefix": _fmt_code_prefix,
    "eu_suffix": _fmt_eu_suffix,
    "space_comma_code": _fmt_space_comma_code,
    "sym_suffix_plain": _fmt_sym_suffix_plain,
}
TRAIN_AMOUNT_FORMATS = ("sym_prefix", "code_suffix", "code_suffix_sep", "code_prefix")
HELDOUT_AMOUNT_FORMATS = ("eu_suffix", "space_comma_code", "sym_suffix_plain")

# ------------------------------------------------------------------- snippet pieces

_VENDOR_LINES = (
    "{v}",
    "{v}",
    "From: {v}",
    "Billed by {v}",
    "Vendor: {v}",
    "{v} - INVOICE",
    "Remit to: {v}",
)
_ID_LINES = (
    "Invoice #: {x}",
    "Invoice No. {x}",
    "Inv# {x}",
    "Invoice ID {x}",
    "Ref: {x}",
)
_DATE_LINES = ("Date: {x}", "Invoice date: {x}", "Issued {x}", "Dated {x}")
_TOTAL_LINES = (
    "Total: {x}",
    "Amount due: {x}",
    "Balance due {x}",
    "Total due: {x}",
    "Grand total {x}",
)
_OTHER_DATE_LINES = (
    "Due: {x}",
    "Payment due {x}",
    "Ship date: {x}",
    "Order date: {x}",
)
_OTHER_AMOUNT_LINES = (
    "Subtotal: {x}",
    "Tax: {x}",
    "Shipping: {x}",
    "Deposit paid: {x}",
)


@dataclass(frozen=True)
class Record:
    """One generated record: the snippet and the gold labels of both stages.

    Attributes:
        text: Raw invoice-like snippet (what stage A reads)
        raw: Stage A's gold output, the four fields as written in the snippet
        gold: Stage B's gold output, the canonical record
        split: One of SPLITS
        meta: How it was generated (formats, vendor); for analysis only
    """

    text: str
    raw: dict[str, str]
    gold: dict[str, Any]
    split: str
    meta: dict[str, Any] = field(default_factory=dict)


def raw_json(raw: dict[str, str]) -> str:
    """Stage A's message as compact, key-sorted JSON text (the JSON condition's wire format)."""
    return canonical_json({name: raw[name] for name in RAW_FIELDS})


def canonical_text(gold: dict[str, Any]) -> str:
    """The canonical record as compact, key-sorted JSON text (stage B's target)."""
    return canonical_json({name: gold[name] for name in CANON_FIELDS})


def _typo(word: str, rng: random.Random) -> str:
    """One character-level typo in the interior of ``word`` (swap, drop, double, substitute)."""
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(("swap", "drop", "double", "sub"))
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    if kind == "drop":
        return word[:i] + word[i + 1 :]
    if kind == "double":
        return word[: i + 1] + word[i] + word[i + 1 :]
    neighbours = _KEYBOARD_NEIGHBOURS.get(word[i].lower(), "e")
    return word[:i] + rng.choice(neighbours) + word[i + 1 :]


def vendor_alias(name: str, rng: random.Random, typo_rate: float = 0.35) -> str:
    """How a vendor may be written in a snippet: case, legal suffix, ``&`` and typos.

    The canonical vendor (``name``) is recoverable from any alias.
    """
    words = name.split(" ")
    if "and" in words and rng.random() < 0.5:
        words = ["&" if w == "and" else w for w in words]
    if rng.random() < typo_rate:
        candidates = [i for i, w in enumerate(words) if len(w) >= 5]
        if candidates:
            i = rng.choice(candidates)
            words[i] = _typo(words[i], rng)
    alias = " ".join(words) + rng.choice(LEGAL_SUFFIXES)
    style = rng.random()
    if style < 0.25:
        return alias.upper()
    if style < 0.30:
        return alias.lower()
    return alias


def _invoice_id(rng: random.Random, year: int) -> str:
    kind = rng.randrange(5)
    if kind == 0:
        return f"INV-{year}-{rng.randrange(100000):05d}"
    if kind == 1:
        letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTVWXZ") for _ in range(3))
        return f"{letters}{rng.randrange(1000000):06d}"
    if kind == 2:
        return f"{rng.randrange(10000):04d}-{rng.randrange(1000):03d}"
    if kind == 3:
        return f"{rng.choice('ABCDEFGHJKMNPRSTW')}-{rng.randrange(100000):05d}"
    return f"INV/{rng.randrange(10000):04d}/{year % 100:02d}"


def _draw_date(rng: random.Random, fmt: str) -> datetime.date:
    span = (LAST_DAY - FIRST_DAY).days
    while True:
        day = FIRST_DAY + datetime.timedelta(days=rng.randrange(span + 1))
        if fmt not in _DAY_ABOVE_12 or day.day > 12:
            return day


def _draw_cents(rng: random.Random) -> int:
    """Amounts from a few dollars to about a hundred thousand, log-uniform."""
    cents = int(10 ** rng.uniform(2.7, 7.0))
    if rng.random() < 0.15:
        cents = cents // 100 * 100  # whole amounts
    return max(cents, 100)


def make_record(
    rng: random.Random,
    split: str,
    vendors: Sequence[str],
    date_formats: Sequence[str],
    amount_formats: Sequence[str],
) -> Record:
    """Generate one record with the given vendor pool and formats."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    vendor = rng.choice(list(vendors))
    date_fmt = rng.choice(list(date_formats))
    amount_fmt = rng.choice(list(amount_formats))
    currency = rng.choices(CURRENCIES, weights=CURRENCY_WEIGHTS)[0]
    cents = _draw_cents(rng)
    day = _draw_date(rng, date_fmt)

    fmt_date = DATE_FORMATS[date_fmt]
    fmt_amount = AMOUNT_FORMATS[amount_fmt]
    alias = vendor_alias(vendor, rng)
    date_text = fmt_date(day)
    amount_text = fmt_amount(cents, currency)
    inv_canonical = _invoice_id(rng, day.year)
    inv_text = inv_canonical.lower() if rng.random() < 0.15 else inv_canonical

    lines = [
        rng.choice(_ID_LINES).format(x=inv_text),
        rng.choice(_DATE_LINES).format(x=date_text),
        rng.choice(_TOTAL_LINES).format(x=amount_text),
    ]
    # Distractors: another date, other amounts, and numbers that look like ids
    other_day = day + datetime.timedelta(days=rng.randrange(7, 60))
    lines.append(rng.choice(_OTHER_DATE_LINES).format(x=fmt_date(other_day)))
    for _ in range(rng.randrange(1, 3)):
        other = max(100, int(cents * rng.uniform(0.05, 0.9)))
        lines.append(
            rng.choice(_OTHER_AMOUNT_LINES).format(x=fmt_amount(other, currency))
        )
    extras = (
        f"PO #: {rng.randrange(10000, 99999)}",
        f"PO-{day.year}-{rng.randrange(100000):05d}",
        f"Order No. {rng.randrange(10, 99)}-{rng.randrange(1000, 9999)}",
        f"Acct {rng.randrange(10**7, 10**8)}",
        f"Tel: ({rng.randrange(200, 999)}) {rng.randrange(200, 999)}-{rng.randrange(10000):04d}",
        f"Qty: {rng.randrange(1, 60)}",
        "Page 1 of 2",
    )
    lines.extend(rng.sample(extras, rng.randrange(1, 3)))
    rng.shuffle(lines)

    vendor_line = rng.choice(_VENDOR_LINES).format(v=alias)
    lines.insert(
        0 if rng.random() < 0.8 else rng.randrange(len(lines) + 1), vendor_line
    )
    text = ("\n" if rng.random() < 0.7 else " | ").join(lines)

    return Record(
        text=text,
        raw={
            "vendor": alias,
            "date": date_text,
            "amount": amount_text,
            "invoice_id": inv_text,
        },
        gold={
            "vendor_canonical": vendor,
            "date_iso": day.isoformat(),
            "amount_cents": cents,
            "currency_iso": currency,
            "invoice_id": inv_canonical.upper(),
        },
        split=split,
        meta={
            "vendor": vendor,
            "date_format": date_fmt,
            "amount_format": amount_fmt,
        },
    )


def make_records(n: int, split: str, seed: int) -> list[Record]:
    """Generate ``n`` records of a split, reproducibly.

    The generator is keyed on (split, seed), so every call replays and different
    splits or seeds draw independent streams.

    Args:
        n: Number of records
        split: One of SPLITS
        seed: Random seed

    Returns:
        List of ``n`` records
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "heldout_test":
        pools = (HELDOUT_VENDORS, HELDOUT_DATE_FORMATS, HELDOUT_AMOUNT_FORMATS)
    else:
        pools = (TRAIN_VENDORS, TRAIN_DATE_FORMATS, TRAIN_AMOUNT_FORMATS)
    rng = random.Random(f"records/{split}/{seed}")
    return [make_record(rng, split, *pools) for _ in range(n)]


# ------------------------------------------------------------------------------ scoring

_FIELD_PATTERNS = {
    name: re.compile(rf'"{name}"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+)')
    for name in dict.fromkeys(RAW_FIELDS + CANON_FIELDS)
}


def parse_fields(text: str, names: Sequence[str]) -> tuple[bool, dict[str, Any]]:
    """Read a model's JSON output; fall back to per-field regexes if it is malformed.

    Returns:
        ``(valid, fields)``. ``valid`` is true when the whole text is a JSON object;
        ``fields`` holds every field that could be read either way.
    """
    try:
        obj = json.loads(text.strip())
    except ValueError:
        obj = None
    if isinstance(obj, dict):
        return True, {name: obj[name] for name in names if name in obj}
    found: dict[str, Any] = {}
    for name in names:
        match = _FIELD_PATTERNS[name].search(text)
        if match:
            try:
                found[name] = json.loads(match.group(1))
            except ValueError:
                continue
    return False, found


def score_fields(
    text: str, gold: dict[str, Any], names: Sequence[str] = CANON_FIELDS
) -> dict[str, Any]:
    """Score one model output against gold.

    Returns:
        ``{"valid": bool, "fields": {name: bool}, "exact": bool}``. ``exact`` needs a
        valid JSON object and every field right (types included: ``amount_cents`` is an
        int).
    """
    valid, fields = parse_fields(text, names)
    ok = {
        name: name in fields
        and type(fields[name]) is type(gold[name])
        and fields[name] == gold[name]
        for name in names
    }
    return {"valid": valid, "fields": ok, "exact": valid and all(ok.values())}


def summarize_scores(
    scores: Sequence[dict[str, Any]], names: Sequence[str]
) -> dict[str, Any]:
    """Aggregate per-record scores into per-field and full-record accuracy."""
    n = len(scores)
    if n == 0:
        raise ValueError("no scores to summarize")
    return {
        "n": n,
        "exact": sum(s["exact"] for s in scores) / n,
        "valid": sum(s["valid"] for s in scores) / n,
        "fields": {name: sum(s["fields"][name] for s in scores) / n for name in names},
    }
