"""Seeded-error set: inject known errors into clean drafts and measure what the checker catches.

Kinds (one per claim, at most one of each per draft):
- wrong_value: claim and prose both changed, so only re-running the tool can catch it
- wrong_team: the claim text names the other team
- wrong_split: offense <-> defense, or pass <-> run, in the claim text
- unlinked_number: a number added to the prose with no claim

Deterministic given the rng, so the set is regenerated from saved runs at scoring time.
"""

from __future__ import annotations

import random
import re
from typing import Any

import duckdb

from nfl_scout.checker import DEFENSE_ONLY, OFFENSE_ONLY, Checker, _close, check_sections
from nfl_scout.graph.state import Claim, Section, ToolCall
from nfl_scout.numbers import extract_numbers

KINDS = ("wrong_value", "wrong_team", "wrong_split", "unlinked_number")
SPLIT_SWAPS = {"offense": "defense", "defense": "offense", "pass": "run", "run": "pass"}


def _calls(tool_calls: list[ToolCall]) -> dict[str, ToolCall]:
    return {c.call_id: c for c in tool_calls if c.result is not None and not c.error}


def _numeric(claim: Claim, calls: dict[str, ToolCall]) -> bool:
    call = calls.get(claim.tool_call_id or "")
    if call is None or isinstance(claim.stated_value, str):
        return False
    return not isinstance(call.result.get(claim.measure, call.result["value"]), list)


def _split_word(claim: Claim, call: ToolCall) -> str | None:
    """A word in the claim text that states one of the call's splits, safe to swap."""
    side = call.args.get("side")
    side = (
        "offense" if call.tool in OFFENSE_ONLY else "defense" if call.tool in DEFENSE_ONLY else side
    )
    kind = call.args.get("play_kind")
    for word in (side, kind):
        if word in SPLIT_SWAPS and re.search(rf"\b{word}\b", claim.text, re.I):
            return word
    return None


def team_forms(con: duckdb.DuckDBPyConnection | None) -> dict[str, dict[str, str]]:
    """Each team's name in four forms (full name, city, nickname, abbreviation)."""
    if con is None:
        return {}
    out = {}
    for abbr, name, nick in con.execute(
        "SELECT team_abbr, team_name, team_nick FROM teams"
    ).fetchall():
        city = name[: -len(nick)].strip() if name.endswith(nick) else name
        out[abbr] = {"name": name, "city": city, "nick": nick, "abbr": abbr}
    return out


def _team_form(text: str, team: str, forms: dict[str, dict[str, str]]) -> str | None:
    """Which form of `team` the text uses, longest first."""
    names = forms.get(team, {"abbr": team})
    for form in ("name", "city", "nick", "abbr"):
        if form in names and re.search(rf"\b{re.escape(names[form])}\b", text):
            return form
    return None


def perturb(v: float, decimals: int, rng: random.Random) -> float:
    """A wrong value: the magnitude moves by about 20% (at least 5 units of the last decimal),
    and the sign is kept, since the checker compares magnitudes."""
    delta = max(abs(v) * 0.2, 5 * 10**-decimals)
    shrink = rng.random() < 0.5 and abs(v) - delta > 10**-decimals
    magnitude = abs(v) - delta if shrink else abs(v) + delta
    new = round(-magnitude if v < 0 else magnitude, decimals)
    # A trailing zero drops a decimal (-0.030 is -0.03), and the checker reads precision from
    # the value: keep moving until its own comparison calls the value wrong.
    while _close(new, v):
        magnitude += delta
        new = round(-magnitude if v < 0 else magnitude, decimals)
    return new


def _replace_word(text: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, text, count=1, flags=re.I)


def seed_errors(
    sections: dict[str, Section],
    tool_calls: list[ToolCall],
    rng: random.Random,
    kinds: list[str] | tuple[str, ...] = KINDS,
    con: duckdb.DuckDBPyConnection | None = None,
) -> tuple[dict[str, Section], list[dict[str, Any]]]:
    """Copies of `sections` with one error of each kind (where a claim allows it), and a
    manifest describing each injected error. With `con`, team names, cities and nicknames can
    be swapped too, not just abbreviations."""
    calls = _calls(tool_calls)
    forms = team_forms(con)
    seeded = {role: s.model_copy(deep=True) for role, s in sections.items()}
    teams = sorted({c.args["team"] for c in calls.values() if "team" in c.args})
    pool = [(role, c) for role, s in seeded.items() for c in s.claims if c.tool_call_id]
    rng.shuffle(pool)
    used: set[str] = set()
    manifest: list[dict[str, Any]] = []

    def pick(ok) -> tuple[str, Claim] | None:
        for role, claim in pool:
            if claim.claim_id not in used and ok(claim):
                used.add(claim.claim_id)
                return role, claim
        return None

    # Most constrained kinds first, so each finds a claim.
    for kind in sorted(kinds, key=lambda k: KINDS.index(k), reverse=True):
        if kind == "unlinked_number":
            role = rng.choice(sorted(seeded))
            text = f"{rng.uniform(20, 80):.1f}%"
            section = seeded[role]
            section.prose += f" Opponents also finish {text} of their red-zone trips."
            manifest.append({"kind": kind, "role": role, "claim_id": None, "seeded_text": text})
        elif kind == "wrong_split":
            hit = pick(lambda c: _split_word(c, calls[c.tool_call_id]) is not None)
            if hit:
                role, claim = hit
                word = _split_word(claim, calls[claim.tool_call_id])
                original = claim.text
                claim.text = _replace_word(claim.text, word, SPLIT_SWAPS[word])
                manifest.append(
                    {"kind": kind, "role": role, "claim_id": claim.claim_id,
                     "original": original, "seeded": claim.text}
                )  # fmt: skip
        elif kind == "wrong_team":

            def names_team(c: Claim) -> bool:
                team = calls[c.tool_call_id].args.get("team")
                return bool(team) and len(teams) > 1 and _team_form(c.text, team, forms) is not None

            hit = pick(names_team)
            if hit:
                role, claim = hit
                team = calls[claim.tool_call_id].args["team"]
                other = rng.choice([t for t in teams if t != team])
                form = _team_form(claim.text, team, forms)
                old_name = forms.get(team, {"abbr": team})[form]
                new_name = forms.get(other, {"abbr": other})[form]
                original = claim.text
                claim.text = re.sub(rf"\b{re.escape(old_name)}\b", new_name, claim.text, count=1)
                manifest.append(
                    {"kind": kind, "role": role, "claim_id": claim.claim_id,
                     "original": original, "seeded": claim.text}
                )  # fmt: skip
        elif kind == "wrong_value":

            def linked(c: Claim, role: str | None = None) -> bool:
                return _numeric(c, calls) and any(
                    m.claim_id == c.claim_id
                    for s in seeded.values()
                    for m in extract_numbers(s.prose)
                )

            hit = pick(linked)
            if hit:
                role, claim = hit
                section = seeded[role]
                mention = next(
                    m for m in extract_numbers(section.prose) if m.claim_id == claim.claim_id
                )
                d = mention.decimals
                new = perturb(float(claim.stated_value), d, rng)
                suffix = "%" if mention.text.endswith("%") else ""
                new_text = f"{new:.{d}f}{suffix}"
                section.prose = (
                    section.prose[: mention.start]
                    + new_text
                    + section.prose[mention.start + len(mention.text) :]
                )
                original = claim.stated_value
                claim.stated_value = new
                manifest.append(
                    {"kind": kind, "role": role, "claim_id": claim.claim_id,
                     "original": original, "seeded": new, "seeded_text": new_text}
                )  # fmt: skip
        else:
            raise ValueError(f"unknown error kind {kind!r}")
    return seeded, manifest


CAUGHT_BY = {"wrong_value": "value", "wrong_team": "wording", "wrong_split": "wording"}


def score_seeded(
    clean: dict[str, Section],
    seeded: dict[str, Section],
    manifest: list[dict[str, Any]],
    tool_calls: list[ToolCall],
    con: duckdb.DuckDBPyConnection | None = None,
    checker: Checker | None = None,
) -> dict[str, Any]:
    """Per kind: errors seeded and caught. False positives: failures on the seeded draft that
    neither the clean draft nor an injected error explains."""
    checker = checker or Checker(tool_calls, con)
    before = {(r.role, r.claim_id, r.check, r.detail if r.claim_id is None else "")
              for r in check_sections(clean, tool_calls, con, checker) if not r.ok}  # fmt: skip
    after = [r for r in check_sections(seeded, tool_calls, con, checker) if not r.ok]

    def relates(r, m) -> bool:
        """The failure is about this injected error (caused by it, not necessarily catching it)."""
        if m["kind"] == "unlinked_number":
            return r.check == "prose" and f"'{m['seeded_text']}'" in r.detail
        return r.claim_id == m["claim_id"]

    def catches(r, m) -> bool:
        return relates(r, m) and (m["kind"] == "unlinked_number" or r.check == CAUGHT_BY[m["kind"]])

    by_kind = {k: {"seeded": 0, "caught": 0} for k in {m["kind"] for m in manifest}}
    for m in manifest:
        by_kind[m["kind"]]["seeded"] += 1
        by_kind[m["kind"]]["caught"] += int(any(catches(r, m) for r in after))
    explained = {i for i, r in enumerate(after) if any(relates(r, m) for m in manifest)}
    false_positives = [
        {"role": r.role, "claim_id": r.claim_id, "check": r.check, "detail": r.detail}
        for i, r in enumerate(after)
        if i not in explained
        and (r.role, r.claim_id, r.check, r.detail if r.claim_id is None else "") not in before
    ]
    return {"by_kind": by_kind, "false_positives": false_positives}
