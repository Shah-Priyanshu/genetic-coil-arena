"""Pieces shared by the distilled translator's training data, its inference and the leak checks.

A distilled translator is a small model that maps ``instruction -> policy reply`` (the JSON
of :mod:`.policy`). Training and inference must show it the same prompt and the same reply
format, so both build them here:

- :func:`format_prompt` wraps an instruction (the spec name selects the task);
- :func:`encode_reply` writes the compact reply the model is trained to produce. Params
  equal to their documented default are left out (the policy translator puts them back),
  which keeps replies to about 20 tokens;
- :func:`frame_ngrams`, :func:`containment` and :func:`max_containment` implement the
  n-gram overlap test that keeps evaluation items and their phrasing frames out of the
  training data.

Domain-agnostic: the caller passes the words to mask (colour names, node names, ...).
"""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty "
    "sixty seventy eighty ninety hundred thousand dozen score half quarter tenth "
    "hundredth thousandth twentieth".split()
)
_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?|[^\sa-z\d]")


def format_prompt(task: str, instruction: str) -> str:
    """The prompt a distilled translator sees: the task (spec name) and the instruction."""
    return f"Task: {task}\nRequest: {instruction}\nReply:\n"


def compact_params(
    params: Mapping[str, Any], defaults: Mapping[str, Any]
) -> dict[str, Any]:
    """Params without the entries that equal their default (sorted keys)."""
    return {
        k: params[k]
        for k in sorted(params)
        if k not in defaults or params[k] != defaults[k]
    }


def encode_reply(
    *,
    params: Optional[Mapping[str, Any]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
    refusal: str = "",
    clarify: str = "",
) -> str:
    """The compact JSON reply for one outcome: params, a refusal or a question."""
    if refusal:
        body: dict[str, Any] = {"refusal": refusal}
    elif clarify:
        body = {"clarify": clarify}
    else:
        body = {"params": compact_params(params or {}, defaults or {})}
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------- overlap


def frame_tokens(text: str, masks: Optional[Mapping[str, str]] = None) -> list[str]:
    """Lower-cased tokens with numbers (digits and number words) and masked words replaced.

    ``masks`` maps a lower-case word to its placeholder (e.g. ``{"red": "<C>"}``), so
    two texts that differ only in the colour or the number have the same frame.
    """
    masks = masks or {}
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok[0].isdigit() or tok in _NUMBER_WORDS:
            out.append("<N>")
        else:
            out.append(masks.get(tok, tok))
    # Collapse runs of number placeholders ("one hundred twenty" -> one <N>)
    collapsed: list[str] = []
    for tok in out:
        if tok == "<N>" and collapsed and collapsed[-1] == "<N>":
            continue
        collapsed.append(tok)
    return collapsed


def frame_ngrams(
    text: str, n: int = 4, masks: Optional[Mapping[str, str]] = None
) -> frozenset[tuple[str, ...]]:
    """The set of word n-grams of a text's frame (the whole frame if it is shorter)."""
    toks = frame_tokens(text, masks)
    if len(toks) <= n:
        return frozenset([tuple(toks)]) if toks else frozenset()
    return frozenset(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def containment(
    reference: frozenset[tuple[str, ...]], candidate: frozenset[tuple[str, ...]]
) -> float:
    """Share of the reference's n-grams that the candidate also has (0 for empty)."""
    return len(reference & candidate) / len(reference) if reference else 0.0


class OverlapIndex:
    """Finds the most similar reference text for candidate texts, by frame n-grams.

    Args:
        references: Texts to keep the candidates away from (e.g. the sealed test set)
        n: n-gram length in words
        masks: Placeholder map, see :func:`frame_tokens`
    """

    def __init__(
        self,
        references: Iterable[str],
        n: int = 4,
        masks: Optional[Mapping[str, str]] = None,
    ):
        self.n = n
        self.masks = dict(masks or {})
        self.references = list(references)
        self.grams = [frame_ngrams(t, n, self.masks) for t in self.references]
        self.exact = {" ".join(frame_tokens(t, self.masks)) for t in self.references}
        self._inverted: dict[tuple[str, ...], list[int]] = {}
        for i, grams in enumerate(self.grams):
            for g in grams:
                self._inverted.setdefault(g, []).append(i)

    def max_containment(self, candidate: str) -> tuple[float, int]:
        """``(score, index)`` of the reference with the largest share of its n-grams in
        ``candidate`` (score 1.0 if the frames are identical; ``(0.0, -1)`` if none)."""
        if " ".join(frame_tokens(candidate, self.masks)) in self.exact:
            toks = " ".join(frame_tokens(candidate, self.masks))
            for i, ref in enumerate(self.references):
                if " ".join(frame_tokens(ref, self.masks)) == toks:
                    return 1.0, i
        cand = frame_ngrams(candidate, self.n, self.masks)
        hits: dict[int, int] = {}
        for g in cand:
            for i in self._inverted.get(g, ()):
                hits[i] = hits.get(i, 0) + 1
        best, arg = 0.0, -1
        for i, h in hits.items():
            score = h / len(self.grams[i])
            if score > best:
                best, arg = score, i
        return best, arg


def max_containment(
    candidates: Sequence[str],
    references: Sequence[str],
    n: int = 4,
    masks: Optional[Mapping[str, str]] = None,
) -> list[float]:
    """Per candidate, the highest containment over the references."""
    index = OverlapIndex(references, n, masks)
    return [index.max_containment(c)[0] for c in candidates]
