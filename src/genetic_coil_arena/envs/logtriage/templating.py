"""Basic log templating: mask the variable parts of a line so equal events compare equal.

A regex pass, not a full parser such as Drain: timestamps, ids, hex strings, addresses,
thread names, ``key=value`` values and numbers become placeholders, HTTP status codes are
kept (they carry the signal), and long lines are cut. The same function templates every
condition's input, so no condition sees raw text the others do not.
"""

import re
from typing import Optional

MAX_TEMPLATE_CHARS = 200

_KEEP_KEYS = frozenset(
    {"method", "result", "level", "caller", "err", "error", "reason", "status", "state"}
)

_STATUS_JSON = re.compile(r'statusCode"?\s*[:=]\s*([1-5]\d\d)')
_BODY_JSON = re.compile(r'"body"\s*:\s*"([^"]{1,100})')
_JSON_BLOB = re.compile(r"\{[^{}]{60,}\}")
_STATUS_CTX = re.compile(
    r'(statusCode"?\s*[:=]\s*|\b(?:GET|POST|PUT|DELETE|PATCH|HEAD)\s+\S+\s+)([1-5]\d\d)\b'
)
_TS = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?")
_CLOCK = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b")
_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_SLEUTH = re.compile(r"\[([\w.-]+),[0-9a-f]{8,32},[0-9a-f]{8,16},(?:true|false)\]")
_THREAD = re.compile(r"\[[^\]\s]*(?:exec|thread|pool|worker|nio)[^\]\s]*\]")
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b")
_HEX = re.compile(r"\b(?=[0-9a-fA-F]*\d)(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{8,}\b")
_KEYVAL = re.compile(r"\b([A-Za-z_][\w.]*)=([^\s,;&\"'}\]]+)")
_NUMBER = re.compile(r"(?<![A-Za-z\d])\d+(?:\.\d+)?")
_SPACES = re.compile(r"\s+")
_DURATION = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?(ns|[µ�u]s|ms|s)(?![\w])")
_UNIT_MS = {"ns": 1e-6, "ms": 1.0, "s": 1000.0}

_SEVERE = re.compile(
    r"\bERROR\b|\bFATAL\b|\bCRITICAL\b|exception|\bpanic\b|HTTP5\d\d|out of memory|"
    r"\boom\b|no space left|segfault|stack ?trace",
    re.IGNORECASE,
)
_WARN = re.compile(
    r"\bWARN(?:ING)?\b|fail|timeout|timed out|refused|reset|unavailable|denied|"
    r"unable|cannot|can't|HTTP4\d\d|retry|disconnect|lost|broken|unreachable|"
    r"slow|too many|exceed",
    re.IGNORECASE,
)


def _bucket_ms(ms: float) -> str:
    """Decade bucket of a duration (words, so the digit mask leaves it alone)."""
    for bound, label in (
        (1.0, "lat_sub_ms"),
        (10.0, "lat_ms"),
        (100.0, "lat_tens_ms"),
        (1000.0, "lat_hundreds_ms"),
    ):
        if ms < bound:
            return label
    return "lat_seconds"


def _duration(match: "re.Match[str]") -> str:
    unit = match.group(2)
    scale = _UNIT_MS.get(unit, 1e-3)  # micro sign (possibly mangled) or "us"
    return f"<{_bucket_ms(float(match.group(1)) * scale)}>"


def _keyval(match: "re.Match[str]") -> str:
    key = match.group(1)
    if key.lower() in _KEEP_KEYS or match.group(2).startswith("<lat_"):
        return match.group(0)
    return f"{key}=<*>"


def make_template(message: str, max_chars: int = MAX_TEMPLATE_CHARS) -> str:
    """The template of one log message (see the module docstring)."""
    text = _SPACES.sub(" ", message.strip())
    status = _STATUS_JSON.search(text)
    body = _BODY_JSON.search(text)
    text = _JSON_BLOB.sub("{..}", text)
    if body:
        text += f" body: {body.group(1)}"
    if status:
        text += f" HTTP{status.group(1)}"
    text = _STATUS_CTX.sub(lambda m: f"{m.group(1)}HTTP{m.group(2)}", text)
    text = _TS.sub("<ts>", text)
    text = _DURATION.sub(_duration, text)
    text = _UUID.sub("<id>", text)
    text = _SLEUTH.sub(lambda m: f"[{m.group(1)},<trace>]", text)
    text = _THREAD.sub("[<thread>]", text)
    text = _IP.sub("<ip>", text)
    text = _CLOCK.sub("<time>", text)
    text = _HEX.sub("<hex>", text)
    text = _KEYVAL.sub(_keyval, text)
    text = _NUMBER.sub("<n>", text)
    return text[:max_chars]


def severity(template: str, level: Optional[str] = None) -> int:
    """0 plain, 1 warning-like, 2 error-like (from the template and an optional level)."""
    probe = f"{level} {template}" if level else template
    if _SEVERE.search(probe):
        return 2
    if _WARN.search(probe):
        return 1
    return 0
