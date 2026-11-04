"""Fact-checker: deterministic checks of analyst sections against the tool-call trace.

1. value: re-run the cited call and compare with stated_value at the precision written (derived
   claims are recomputed from the claims they cite; list results are matched on the named player).
2. wording: the claim text must match the call's arguments (team, side, play kind, down, split,
   player, week range, season). Rule-based; an LLM check for free-text descriptors is a stretch.
3. prose: every number in the prose must link to one of the section's claims.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import duckdb

from nfl_scout.graph.state import Claim, Section, ToolCall
from nfl_scout.numbers import check_numbers
from nfl_scout.tools import TOOLS
from nfl_scout.tools.base import default_connection

Check = Literal["value", "wording", "prose"]

# Tools whose side is fixed (no `side` argument).
OFFENSE_ONLY = {
    "pass_rate",
    "pass_rate_over_expected",
    "average_depth_of_target",
    "run_direction_rate",
    "player_share",
    "usage_leaders",
    "ftn_play_rate",
}
DEFENSE_ONLY = {"blitz_rate", "average_pass_rushers", "average_box_defenders"}

DEFENSE_WORDS = re.compile(r"\bdefen[cs]e|\bdefensive", re.I)
OFFENSE_WORDS = re.compile(r"\boffen[cs]e|\boffensive", re.I)
PLAY_WORDS = {
    "pass": re.compile(r"\bpass|dropback|\bair\b|throw", re.I),
    "run": re.compile(r"\brun|\brush|carr(y|ies)", re.I),
}
SPLIT_WORDS = {
    "play_action": r"play[- ]action",
    "motion": r"motion",
    "screen": r"screen",
    "rpo": r"\brpo",
    "target": r"target",
    "rush": r"rush|carr(y|ies)",
    "left": r"\bleft\b",
    "right": r"\bright\b",
    "middle": r"middle|inside",
    "short": r"short",
    "medium": r"medium",
    "long": r"long",
}
ORDINALS = {1: ("1st", "first"), 2: ("2nd", "second"), 3: ("3rd", "third"), 4: ("4th", "fourth")}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


@dataclass
class CheckResult:
    role: str
    claim_id: str | None
    check: Check
    ok: bool
    detail: str = ""


def team_aliases(con: duckdb.DuckDBPyConnection) -> dict[str, set[str]]:
    """Ways a claim can name each team: abbreviation, nickname, city and full name."""
    out: dict[str, set[str]] = {}
    for abbr, name, nick in con.execute(
        "SELECT team_abbr, team_name, team_nick FROM teams"
    ).fetchall():
        city = name[: -len(nick)].strip() if name.endswith(nick) else name
        out[abbr] = {abbr, name.lower(), nick.lower(), city.lower()}
    return out


def _decimals(x: float) -> int:
    if float(x).is_integer():
        return 0
    return len(repr(float(x)).split(".")[1])


def _close(stated: float, actual: float) -> bool:
    """Equal at the precision stated. Signs are compared loosely, since prose often carries the
    sign in words ("5.1 points below expected")."""
    tol = 0.5 * 10 ** -_decimals(stated) + 1e-9
    return abs(abs(stated) - abs(actual)) <= tol


def _num(x: Any) -> float | None:
    try:
        return float(x) if not isinstance(x, bool) else None
    except (TypeError, ValueError):
        return None


def _same_result(a: dict[str, Any], b: dict[str, Any]) -> bool:
    def same(x: Any, y: Any) -> bool:
        if isinstance(x, float) or isinstance(y, float):
            return x is not None and y is not None and math.isclose(x, y, abs_tol=1e-9)
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(same(p, q) for p, q in zip(x, y, strict=True))
        if isinstance(x, dict) and isinstance(y, dict):
            return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
        return x == y

    return same(a.get("value"), b.get("value")) and same(a.get("n_plays"), b.get("n_plays"))


def _surname(name: str) -> str:
    """Last name from 'Marvin Mims Jr.', 'G.Holani' or 'A.St. Brown' (suffixes dropped)."""
    words = [w for w in re.split(r"[\s.]+", name.lower()) if w and w not in SUFFIXES]
    return words[-1] if words else name.lower()


def _mentions(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.I) is not None


def _check_list(claim: Claim, items: list[dict]) -> str | None:
    named = [it for it in items if _mentions(claim.text, _surname(str(it.get("player", ""))))]
    if not named:
        return f"names no player in the {len(items)}-player result"
    stated = claim.stated_value
    for it in named:
        for v in it.values():
            if isinstance(stated, str) and isinstance(v, str) and v.lower() == stated.lower():
                return None
            if _num(stated) is not None and _num(v) is not None and _close(_num(stated), _num(v)):
                return None
    players = ", ".join(str(it.get("player")) for it in named)
    return f"{stated} doesn't match the result for {players}"


class Checker:
    def __init__(
        self,
        tool_calls: Iterable[ToolCall],
        con: duckdb.DuckDBPyConnection | None = None,
    ):
        self.calls = {c.call_id: c for c in tool_calls}
        self.con = (con or default_connection()).cursor()
        self.aliases = team_aliases(self.con)
        self._reruns: dict[str, dict[str, Any] | str] = {}

    def rerun(self, call: ToolCall) -> dict[str, Any] | str:
        """The call's result, re-executed (cached per call), or an error message."""
        if call.call_id not in self._reruns:
            try:
                self._reruns[call.call_id] = (
                    TOOLS[call.tool].fn(con=self.con, **call.args).model_dump()
                )
            except (KeyError, ValueError, TypeError) as e:
                self._reruns[call.call_id] = f"re-run failed: {e}"
        return self._reruns[call.call_id]

    # --- 1. value ------------------------------------------------------------------------------

    def value(self, claim: Claim, claims: dict[str, Claim]) -> str | None:
        if claim.derived_from:
            return self._derived(claim, claims)
        call = self.calls.get(claim.tool_call_id or "")
        if call is None:
            return f"cites unknown call {claim.tool_call_id}"
        if call.error:
            return f"{call.call_id} returned an error: {call.error}"
        result = self.rerun(call)
        if isinstance(result, str):
            return result
        if call.result is None or not _same_result(result, call.result):
            return f"re-run of {call.call_id} doesn't reproduce the traced result"
        actual = result["n_plays"] if claim.measure == "n_plays" else result["value"]
        if isinstance(actual, list):
            return _check_list(claim, actual)
        stated = _num(claim.stated_value)
        if actual is None or stated is None:
            return f"{call.call_id} {call.tool} gives {actual!r}; stated {claim.stated_value!r}"
        if not _close(stated, float(actual)):
            n = result["n_plays"]
            return (
                f"{call.call_id} {call.tool} gives {round(float(actual), 3)} (n_plays {n}); "
                f"stated {claim.stated_value}"
            )
        return None

    def _derived(self, claim: Claim, claims: dict[str, Claim]) -> str | None:
        missing = [d for d in claim.derived_from or [] if d not in claims]
        if missing:
            return f"derived from unknown claims {missing}"
        vals = [_num(claims[d].stated_value) for d in claim.derived_from or []]
        if any(v is None for v in vals) or not vals:
            return "derived from non-numeric claims"
        if claim.op == "ratio":
            if len(vals) != 2 or vals[1] == 0:
                return "a ratio needs two claims and a non-zero denominator"
            computed = vals[0] / vals[1]
        elif claim.op == "difference":
            if len(vals) != 2:
                return "a difference needs two claims"
            computed = vals[0] - vals[1]
        else:
            computed = sum(vals)
        stated = _num(claim.stated_value)
        if stated is None or not _close(stated, computed):
            return f"{claim.op} of {claim.derived_from} is {round(computed, 3)}; stated {stated}"
        return None

    # --- 2. wording ----------------------------------------------------------------------------

    def wording(self, claim: Claim) -> list[str]:
        call = self.calls.get(claim.tool_call_id or "")
        if call is None:
            return []
        args = dict(call.args)
        tool = call.tool
        if tool == "league_rank":
            rerun = self.rerun(call)
            applied = rerun["query_meta"].get("metric_args", {}) if isinstance(rerun, dict) else {}
            tool = args["metric"]
            args = {"team": args["team"], **applied, **{k: args[k] for k in INJECTED_ARGS}}
        text = claim.text
        params = set(inspect.signature(TOOLS[tool].fn).parameters) if tool in TOOLS else set()
        player_named = "player" in args and _mentions(text, _surname(str(args["player"])))
        # A claim naming the player needn't name the team too, but must not name a wrong one.
        problems = [self._team(text, args.get("team"), required=not player_named)]
        if "side" in params or tool in OFFENSE_ONLY | DEFENSE_ONLY:
            problems.append(self._side(text, tool, args.get("side", "offense")))
        if "play_kind" in params:
            problems.append(self._play_kind(text, args.get("play_kind", "all")))
        if "down" in params:
            problems.append(self._down(text, args.get("down"), args.get("early_downs_only")))
        for key in ("feature", "direction", "kind", "distance"):
            if key in args and args[key] in SPLIT_WORDS:
                if not re.search(SPLIT_WORDS[args[key]], text, re.I):
                    label = str(args[key]).replace("_", " ")
                    problems.append(f"the call is for {label}, the text doesn't say so")
        # player_share's sample size is the team total, so it needn't name the player.
        team_total = tool == "player_share" and claim.measure == "n_plays"
        if "player" in args and not player_named and not team_total:
            problems.append(f"the call is for {args['player']}, the text doesn't name them")
        rerun = self.rerun(call)
        weeks = rerun["query_meta"].get("weeks") if isinstance(rerun, dict) else None
        problems.append(self._weeks(text, args, max(weeks) if weeks else None))
        return [p for p in problems if p]

    def _team(self, text: str, team: str | None, required: bool = True) -> str | None:
        if team is None:
            return None

        def named(abbr: str) -> bool:
            return any(
                re.search(rf"\b{re.escape(a)}\b", text) if a.isupper() else _mentions(text, a)
                for a in self.aliases.get(abbr, {abbr})
            )

        if named(team):
            return None
        others = sorted(t for t in self.aliases if t != team and named(t))
        if others:
            return f"the text names {', '.join(others)}, but the call is for {team}"
        return f"the text doesn't name {team}" if required else None

    @staticmethod
    def _side(text: str, tool: str, side: str) -> str | None:
        side = "offense" if tool in OFFENSE_ONLY else "defense" if tool in DEFENSE_ONLY else side
        says_def, says_off = DEFENSE_WORDS.search(text), OFFENSE_WORDS.search(text)
        if side == "offense" and says_def and not says_off:
            return "the text says defense, but the call measures the offense"
        if side == "defense" and says_off and not says_def:
            return "the text says offense, but the call measures the defense"
        return None

    @staticmethod
    def _play_kind(text: str, kind: str) -> str | None:
        says = {k for k, pat in PLAY_WORDS.items() if pat.search(text)}
        if kind == "all" and len(says) == 1:
            return f"the text says {says.pop()} plays, but the call covers all plays"
        if kind in PLAY_WORDS and kind not in says:
            return f"the call is for {kind} plays, the text doesn't say so"
        return None

    @staticmethod
    def _down(text: str, down: int | None, early: bool | None) -> str | None:
        said = {
            n
            for n, words in ORDINALS.items()
            if re.search(rf"\b(?:{'|'.join(words)})[\s-]+(?:down|and)", text, re.I)
        }
        if down is not None and down not in said:
            return f"the call is for {ORDINALS[down][0]} down, the text doesn't say so"
        if down is None and not early and said:
            return f"the text says {ORDINALS[min(said)][0]} down, but the call covers all downs"
        return None

    @staticmethod
    def _weeks(text: str, args: dict[str, Any], last_week: int | None) -> str | None:
        """Week ranges must end at the last week in the result (a player may have missed the
        latest game), or at the cutoff when the result doesn't list weeks."""
        last = last_week or args.get("through_week")
        m = re.search(r"\bweeks?\s+\d+\s*(?:[-–]|to|through)\s*(\d+)", text, re.I)
        if m and int(m.group(1)) != last:
            return f"the text says weeks through {m.group(1)}, the data runs through week {last}"
        for year in re.findall(r"\b(20\d\d)\b", text):
            if int(year) != args.get("season"):
                return f"the text says {year}, the call is for the {args.get('season')} season"
        return None


INJECTED_ARGS = ("season", "through_week")


def check_sections(
    sections: dict[str, Section],
    tool_calls: Iterable[ToolCall],
    con: duckdb.DuckDBPyConnection | None = None,
    checker: Checker | None = None,
) -> list[CheckResult]:
    """Run all three checks over every section. One value and one wording result per claim,
    plus one prose result per number that doesn't link to a matching claim. Pass a `checker`
    built on the same tool calls to reuse its cached re-runs."""
    checker = checker or Checker(tool_calls, con)
    claims = {c.claim_id: c for s in sections.values() for c in s.claims}
    results: list[CheckResult] = []
    for role, section in sections.items():
        for claim in section.claims:
            problem = checker.value(claim, claims)
            results.append(
                CheckResult(role, claim.claim_id, "value", problem is None, problem or "")
            )
            if claim.tool_call_id:
                problems = checker.wording(claim)
                results.append(
                    CheckResult(role, claim.claim_id, "wording", not problems, "; ".join(problems))
                )
        for nc in check_numbers(section.prose, section.claims):
            if nc.status != "ok":
                detail = f"{nc.mention.text!r} in the prose: {nc.status.replace('_', ' ')}"
                results.append(CheckResult(role, nc.mention.claim_id, "prose", False, detail))
    return results
