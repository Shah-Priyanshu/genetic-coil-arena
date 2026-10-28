"""R12c (goal G10b): the incident desk checks the translator's literals against the request.

A small translator sometimes writes a value the user never said: a year on "Nov 22", a month on
"the 19th", a bare day for "two days ago", a clamped "top 8". This check runs after the
translation and before validation and resolution. It compares the extracted literals with the
request text and does one of two things, never a third:

* removes what the text does not support (a year, a month), or
* takes the text's own value where the two disagree (a relative day from the desk clock, an
  explicit suspect count).

It never adds a parameter the model left out and never invents a value. Rules, in order
(``params`` is the translator's ``rca.investigate`` params, ``text`` the request):

``relative``
    ``at`` is present and the text says "yesterday", "the day before yesterday" or "N days
    ago" (digits or number words up to fourteen), and gives no explicit date (month name,
    ISO date or numeric date): the date part of ``at`` becomes the date that wording gives on
    the desk clock; a time part is kept.
``year``
    ``at`` starts with ``YYYY-`` and the text names no year (no four-digit 19xx/20xx outside a
    ticket or a clock time, no ``d/m/yy`` date) and uses no relative wording: the year is
    dropped (``YYYY-MM-DD`` becomes ``--MM-DD``).
``month``
    ``at`` is ``--MM-DD`` (after the year rule) and the text names no month (a month name or
    abbreviation, a misspelling of one, or a numeric ``d/m`` date), says nothing about a
    "month" and uses no relative wording: the month is dropped (``---DD``).
``top_k``
    The text states exactly one suspect count ("top 3", "three suspects", "5 most likely
    candidates"; digits or number words up to ten) and ``top_k`` differs or is missing: the
    text's count is used. A count outside the schema range then fails validation and the
    desk refuses, as the policy intends (no clamping).

Rules never touch ``ticket``, ``system``, ``exclude_services`` or ``want_fault_type`` and
never remove ``at`` as a whole. Domain code of the incident desk: no other package imports it.
"""

import copy
import dataclasses
import datetime as dt
import difflib
import re
from collections.abc import Mapping
from typing import Any, Optional

RULES = ("relative", "year", "month", "top_k")
RULES_TEXT = (
    "relative (a relative day in the text sets the date from the desk clock), "
    "year (a year the text does not name is dropped), "
    "month (a month the text does not name is dropped), "
    "top_k (the one suspect count the text states replaces a different top_k)."
)
NUMBER_WORDS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen".split()
    )
}
MONTH_NAMES = (
    "january february march april may june july august september october november "
    "december"
).split()
_NUM = r"(\d{1,2}|" + "|".join(NUMBER_WORDS) + ")"

_YEAR = re.compile(r"(?<![\w-])(?:19|20)\d\d(?![\w:])")
_NUMERIC_DATE_YEAR = re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b")
_NUMERIC_DATE = re.compile(r"(?<![\d:])\d{1,2}[/.]\d{1,2}(?![\d:])")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_RELATIVE = re.compile(
    r"\b(yesterday|today|tonight|tomorrow|ago|last|previous|earlier|the day before|"
    r"day before|this (?:morning|afternoon|evening|week|month|year)|the other day|"
    r"weeks?|months?)\b"
)
_DAY_BEFORE_YESTERDAY = re.compile(r"\bday before yesterday\b")
_YESTERDAY = re.compile(r"\byesterday\b")
_DAYS_AGO = re.compile(rf"\b{_NUM}\s+days?\s+ago\b")
_COUNT_NOUN = r"(?:suspects?|candidates?|culprits?|offenders?|suspected)"
_COUNT_MOD = r"(?:(?:top|most|likely|probable|main|prime|leading|likeliest)\s+)*"
_TOP_N = re.compile(rf"\btop[\s-]*{_NUM}\b(?!\s*[:.]\d)")
_N_NOUN = re.compile(rf"(?<![\w:./-]){_NUM}[\s-]+{_COUNT_MOD}{_COUNT_NOUN}\b")
_AT = re.compile(
    r"^(?:(\d{4})-(\d{2})-(\d{2})|--(\d{2})-(\d{2})|---(\d{2}))(T\d{2}:\d{2})?$"
)


@dataclasses.dataclass(frozen=True)
class CheckResult:
    """The checked params and the rules that changed something.

    Attributes:
        params: The params after the check (a copy; the input is never modified)
        rules: Names of the rules that fired, in order
        changes: ``field -> (before, after)`` for every changed field
    """

    params: dict[str, Any]
    rules: tuple[str, ...] = ()
    changes: Optional[dict[str, tuple[Any, Any]]] = None


def _number(token: str) -> int:
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


def names_month(low: str) -> bool:
    """Whether the (lower-cased) text names a month: a name, an abbreviation, a typo of one,
    a numeric ``d/m`` date, or the word "month". Leans towards yes (a false yes only keeps the
    translator's month)."""
    if _NUMERIC_DATE.search(low) or _NUMERIC_DATE_YEAR.search(low) or "month" in low:
        return True
    for word in re.findall(r"[a-z]{3,}", low):
        for month in MONTH_NAMES:
            if month.startswith(word):
                return True
            if difflib.SequenceMatcher(None, word, month).ratio() >= 0.75:
                return True
    return False


def names_year(text: str) -> bool:
    """Whether the text gives a year (``2024``, ``22/11/24``)."""
    return bool(_YEAR.search(text) or _NUMERIC_DATE_YEAR.search(text))


def relative_date(low: str, now: dt.datetime) -> Optional[dt.date]:
    """The date a relative wording names on the clock ``now``; None unless exactly one
    wording is present and the text gives no explicit date."""
    if _ISO_DATE.search(low) or names_month(low):
        return None
    found: set[int] = set()
    if _DAY_BEFORE_YESTERDAY.search(low):
        found.add(2)
    elif _YESTERDAY.search(low):
        found.add(1)
    for m in _DAYS_AGO.finditer(low):
        n = _number(m.group(1))
        if 1 <= n <= 14:
            found.add(n)
    if len(found) != 1:
        return None
    return now.date() - dt.timedelta(days=found.pop())


def stated_count(low: str) -> Optional[int]:
    """The one suspect count the text states, or None (none, or two different ones)."""
    counts = {_number(m.group(1)) for m in _TOP_N.finditer(low)}
    counts |= {_number(m.group(1)) for m in _N_NOUN.finditer(low)}
    return counts.pop() if len(counts) == 1 else None


def check_params(params: Mapping[str, Any], text: str, now: dt.datetime) -> CheckResult:
    """Apply the rules (see the module docstring) to ``params`` against ``text``.

    Args:
        params: The translator's ``rca.investigate`` params
        text: The user's request
        now: The desk clock (needed by the ``relative`` rule)
    """
    out = copy.deepcopy(dict(params))
    low = text.lower()
    fired: list[str] = []
    changes: dict[str, tuple[Any, Any]] = {}

    def set_field(rule: str, name: str, value: Any) -> None:
        before = changes[name][0] if name in changes else params.get(name)
        out[name] = value
        changes[name] = (before, value)
        if rule not in fired:
            fired.append(rule)

    at = out.get("at")
    m = _AT.match(str(at)) if at is not None else None
    if m is not None:
        time_part = m.group(7) or ""
        target = relative_date(low, now)
        if target is not None and f"{target.isoformat()}{time_part}" != at:
            set_field("relative", "at", f"{target.isoformat()}{time_part}")
            at, m = out["at"], _AT.match(out["at"])
        relative = bool(_RELATIVE.search(low))
        if m is not None and m.group(1) and not names_year(text) and not relative:
            set_field("year", "at", f"--{m.group(2)}-{m.group(3)}{time_part}")
            at, m = out["at"], _AT.match(out["at"])
        if m is not None and m.group(4) and not names_month(low) and not relative:
            set_field("month", "at", f"---{m.group(5)}{time_part}")
    count = stated_count(low)
    if count is not None and out.get("top_k") != count:
        set_field("top_k", "top_k", count)
    return CheckResult(out, tuple(fired), changes or None)


__all__ = ["CheckResult", "RULES", "check_params", "names_month", "stated_count"]
