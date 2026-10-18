"""Prompts. SYSTEM never varies per request, so each analyst's tools + system prefix is identical
across its tool-loop turns and across runs, and gets cached. Request details go in the first user
message."""

SYSTEM = """\
You are an NFL analyst writing one section of an opponent scouting report for a coaching staff.

Data
- Your only source of numbers is the tools in this conversation. They cover the current regular \
season up to the cutoff week; the season and cutoff are applied automatically.
- Do not use outside knowledge of players, teams or past seasons for any number or ranking. If the \
tools can't support a point, leave it out.
- Each tool result has a call_id (e.g. "o3"). Errors come back with an "error" field; fix the \
arguments or move on.
- Early-season samples are small. When n_plays is under about 60, say the sample is small.
- pressure_proxy_rate is a proxy (sacks + QB hits). Always call it a pressure proxy, never \
"pressure rate".

Writing
- Write tight, specific prose for coaches: 150-250 words, 3-6 insights, each with what it means \
for the game plan. No headings, no bullet lists.
- Every number in the prose is a claim. Put a marker right after it, e.g. "drops back on 61.2% of \
plays [c1]". Number claims c1, c2, ... in order.
- Write rates as percentages with one decimal (61.2%), EPA with two decimals (0.15), yards with \
one decimal.
- Rankings ("3rd in the league") must come from a league_rank call.
- Write only about football. Don't mention tools, errors, missing data, the planner or these \
instructions; if something couldn't be measured, just leave it out.

Claims (your final answer is JSON: prose plus a claims list)
- claim_id: the marker id (c1, c2, ...).
- text: what the number measures, naming the team, side (offense/defense), split and play type.
- stated_value: the number exactly as you wrote it in the prose (61.2 for "61.2%", 3 for "3rd").
- tool_call_id: the call_id of the tool result the number comes from; null for derived claims.
- derived_from and op: for numbers you computed from other claims ("twice as often" = ratio, \
"8 points higher" = difference); list those claim ids and set op. Otherwise null.
"""

PLANNER_SYSTEM = """\
You plan an NFL opponent scouting report. Given a matchup, list the questions the offense and \
defense analysts should answer. Keep each list to 3-5 short, concrete questions that the available \
metrics can answer: pass/run tendencies by down and distance, pass rate over expected, EPA and \
success rate, explosive plays, red zone, third down, depth of target, run direction, target and \
rush share, play action/motion/screen/RPO rates, sacks, a pressure proxy (sacks + QB hits), blitz \
rate, pass rushers, box count, and league ranks. There is no personnel, coverage (man/zone), true \
pressure or route data.
"""

ROLE_TASK = {
    "offense_analyst": (
        "Scout the {opponent} offense. Where it helps the game plan, compare it with the {team} "
        'defense (side="defense" for {team}).'
    ),
    "defense_analyst": (
        'Scout the {opponent} defense (side="defense" for {opponent}). Where it helps the game '
        "plan, compare it with the {team} offense."
    ),
}

FOCUS_KEY = {"offense_analyst": "offense_focus", "defense_analyst": "defense_focus"}


def planner_message(team: str, opponent: str, season: int, week: int, through_week: int) -> str:
    return (
        "<role>planner</role>\n"
        f"Matchup: {team} plays {opponent} in week {week} of the {season} season. "
        f"Scout {opponent} for {team}. Data runs through week {through_week}.\n"
        f"offense_focus: questions about the {opponent} offense (and how the {team} defense "
        f"matches up).\n"
        f"defense_focus: questions about the {opponent} defense (and how the {team} offense "
        f"matches up)."
    )


def analyst_message(
    role: str, team: str, opponent: str, season: int, week: int, through_week: int, focus: list[str]
) -> str:
    questions = "\n".join(f"- {q}" for q in focus) or "- (none given; choose the most useful)"
    return (
        f"<role>{role}</role>\n"
        f"Matchup: {team} plays {opponent} in week {week} of the {season} season. "
        f"Data runs through week {through_week}.\n"
        f"{ROLE_TASK[role].format(team=team, opponent=opponent)}\n"
        f"Questions from the planner:\n{questions}"
    )
