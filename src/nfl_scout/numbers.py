"""Find every number in report prose and link it to the claim marker that follows it.

Deterministic (no LLM), so the same code scores drafts, final reports and saved runs. Football
vocabulary that isn't a statistic (downs, week ranges, seasons, "5+ rushers" definitions) is
masked out first. A number with no marker after it in its sentence is unsupported.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from nfl_scout.graph.state import Claim

Kind = Literal["percent", "number", "ordinal", "ratio"]
Status = Literal["ok", "value_mismatch", "unknown_claim", "unlinked"]

MARKER = re.compile(r"\[\s*([a-z]{1,3}\d+(?:\s*,\s*[a-z]{1,3}\d+)*)\s*\]", re.I)
# Adjacent marker groups ("[dc1][dc2]") read as one group.
MARKER_RUN = re.compile(rf"{MARKER.pattern}(?:\s*{MARKER.pattern})*", re.I)

_ORD = r"(?:1st|2nd|3rd|4th|first|second|third|fourth)"
MASKS = [
    MARKER,
    # Downs: "3rd-and-long", "1st and 2nd down", "third down", "3rd-and-8".
    re.compile(
        rf"\b{_ORD}(?:\s*(?:and|or|&)\s*{_ORD})*"
        r"(?:[\s-]+downs?\b|-and-\w+|\s+and\s+(?:short|long|medium|\d+)\b)",
        re.I,
    ),
    re.compile(r"\bweeks?\s+\d+(?:\s*(?:[-–]|to|through)\s*\d+)?", re.I),
    re.compile(r"\b(?:19|20)\d{2}\b"),  # seasons
    # Thresholds in definitions: "5+ rushers", "20-plus yards", "10 or more yards".
    re.compile(r"\d+(?:\+|-plus\b|\s+or\s+more\b)", re.I),
    re.compile(r"#\d+"),  # jersey numbers
]

_WORD_ORD = "first second third fourth fifth sixth seventh eighth ninth tenth".split()
_WORD_RATIO = {"twice": 2, "two times": 2, "three times": 3, "four times": 4, "half as": 0.5}
NUMBER = re.compile(
    r"(?P<ord>(?<![\w.,])\d+(?:st|nd|rd|th)\b)"
    rf"|(?P<wordord>\b(?:{'|'.join(_WORD_ORD)})\b"
    r"(?=[\s-]+(?:in\s+the\s+(?:league|nfl)|among|overall|best|worst|highest|lowest|most|fewest)))"
    rf"|(?P<wordratio>\b(?:{'|'.join(k.replace(' ', r'\s+') for k in _WORD_RATIO)})\b)"
    r"|(?P<num>(?<![\w.,])[-−+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?P<suffix>\s?%|\s+percent\b|x\b)?",
    re.I,
)
SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


@dataclass
class Mention:
    text: str
    value: float
    kind: Kind
    decimals: int
    start: int
    claim_id: str | None = None
    group: int | None = None  # index of the marker group that follows it in its sentence
    group_ids: tuple[str, ...] = ()


@dataclass
class NumberCheck:
    mention: Mention
    status: Status


def _mask(text: str) -> str:
    for pattern in MASKS:
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


def _mention(m: re.Match) -> Mention:
    raw = m.group(0)
    if m.group("ord"):
        return Mention(raw, float(re.match(r"\d+", raw).group(0)), "ordinal", 0, m.start())
    if m.group("wordord"):
        return Mention(raw, float(_WORD_ORD.index(raw.lower()) + 1), "ordinal", 0, m.start())
    if m.group("wordratio"):
        value = _WORD_RATIO[" ".join(raw.lower().split())]
        return Mention(raw, float(value), "ratio", 1 if value % 1 else 0, m.start())
    num = m.group("num")
    value = float(num.replace(",", "").replace("−", "-"))
    decimals = len(num.split(".")[1]) if "." in num else 0
    suffix = (m.group("suffix") or "").strip().lower()
    kind: Kind = "percent" if suffix in ("%", "percent") else "ratio" if suffix == "x" else "number"
    return Mention(raw, value, kind, decimals, m.start())


def extract_numbers(text: str) -> list[Mention]:
    """Every statistic-like number in `text`, in order, each linked to a claim id or None.

    A marker group links to the numbers just before it in the same sentence, pairing from the end
    ("41.7% and 30.3% [c1, c2]"); earlier unpaired numbers stay unlinked.
    """
    masked = _mask(text)
    events: list[tuple[int, str, object]] = [
        (m.start(), "num", _mention(m)) for m in NUMBER.finditer(masked)
    ]
    events += [
        (m.start(), "marker", re.findall(r"[a-z]{1,3}\d+", m.group(0), re.I))
        for m in MARKER_RUN.finditer(text)
    ]
    events += [(m.start(), "end", None) for m in SENTENCE_END.finditer(masked)]
    events.sort(key=lambda e: e[0])

    mentions: list[Mention] = []
    pending: list[Mention] = []
    for i, (_, kind, payload) in enumerate(events):
        if kind == "num":
            mentions.append(payload)
            pending.append(payload)
        elif kind == "marker":
            for mention in pending:
                mention.group, mention.group_ids = i, tuple(payload)
            for mention, cid in zip(reversed(pending), reversed(payload), strict=False):
                mention.claim_id = cid
            pending = []
        else:
            pending = []
    return mentions


def _same(stated: float | str, mention: Mention) -> bool:
    """Equal at the precision written in the prose. Signs are compared loosely because prose
    often carries them in words ("5.1 points below expected")."""
    try:
        stated = float(stated)
    except ValueError:
        return False
    return round(abs(stated), mention.decimals) == round(abs(mention.value), mention.decimals)


def _pair_by_value(group: list[Mention], by_id: dict[str, Claim]) -> None:
    """Re-link a marker group so each id goes to the number its claim states ("80.7% of 57
    snaps [c4]" links c4 to 80.7, not 57). Ids that match no number pair from the end."""
    free = list(group)
    unmatched = []
    for m in group:
        m.claim_id = None
    for cid in group[0].group_ids:
        claim = by_id.get(cid)
        hit = next((m for m in reversed(free) if claim and _same(claim.stated_value, m)), None)
        if hit:
            hit.claim_id = cid
            free.remove(hit)
        else:
            unmatched.append(cid)
    for m, cid in zip(reversed(free), reversed(unmatched), strict=False):
        m.claim_id = cid


def check_numbers(text: str, claims: Iterable[Claim]) -> list[NumberCheck]:
    """Status of every number in `text` against the claims its markers point to."""
    by_id = {c.claim_id: c for c in claims}
    mentions = extract_numbers(text)
    groups: dict[int, list[Mention]] = {}
    for m in mentions:
        if m.group is not None:
            groups.setdefault(m.group, []).append(m)
    for group in groups.values():
        _pair_by_value(group, by_id)

    results = []
    for m in mentions:
        if m.claim_id is None:
            status: Status = "unlinked"
        elif m.claim_id not in by_id:
            status = "unknown_claim"
        elif not _same(by_id[m.claim_id].stated_value, m):
            status = "value_mismatch"
        else:
            status = "ok"
        results.append(NumberCheck(m, status))
    return results
