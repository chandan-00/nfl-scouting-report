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
- Write tight, specific prose for coaches, at the length the task gives, with what each point \
means for the game plan. No headings, no bullet lists.
- Every number in the prose is a claim, including sample sizes ("on 27 plays"). Put a marker \
right after it, e.g. "drops back on 61.2% of plays [c1] over 231 plays [c2]". Never group \
markers at the end of a clause ("61.2% on 231 plays [c1][c2]"). Number claims c1, c2, ... in the \
order they appear.
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
- measure: "n_plays" when the number is the result's sample size (n_plays), otherwise "value".
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
pressure or route data. The injury analyst has each team's latest injury report, snap shares and \
target/rush shares.
"""

ROLE_TASK = {
    "offense_analyst": (
        "Scout the {opponent} offense. Where it helps the game plan, compare it with the {team} "
        'defense (side="defense" for {team}). 150-250 words.'
    ),
    "defense_analyst": (
        'Scout the {opponent} defense (side="defense" for {opponent}). Where it helps the game '
        "plan, compare it with the {team} offense. 150-250 words."
    ),
    "injury_analyst": (
        "Report on availability for both teams: the latest injury reports for {opponent} and "
        "{team}, and how much the listed players matter (snap share, target or rush share). The "
        "latest report is from an earlier game (report_week), so say which week a status is "
        'from ("listed Out for week 5"). A status is a claim too: stated_value is the status '
        'text, e.g. "Out". If nobody notable is listed, say so briefly. 80-150 words.'
    ),
    "matchup_analyst": (
        "Using the analyst sections below, name the 3-4 matchups that will decide the game and "
        "what {team} should do about each. To reuse a number from a section, cite the same "
        "call_id as its claim; call tools only for numbers the sections don't have. 150-250 "
        "words."
    ),
}

LAST_TURN = "No more tool calls. Write your final answer now from the results you have."

FOCUS_KEY = {
    "offense_analyst": "offense_focus",
    "defense_analyst": "defense_focus",
    "injury_analyst": "injury_focus",
}

EDITOR_SYSTEM = """\
You edit analyst sections into one NFL opponent scouting report for a coaching staff.

Numbers
- Use only numbers that appear in the sections. Copy each one exactly as written, with its marker \
right after it, e.g. "61.2% [oc1]". Keep the marker ids unchanged. Each section's claims list \
gives the value behind every marker; use it to keep each number with its own marker.
- Each marker belongs to exactly one number, including sample sizes ("on 18 drives [oc15]"). \
Never move a marker to another number or reuse it; if you keep a number, keep its own marker.
- Never add, round, combine or change a number, and never write a number without its marker. If \
a number has no marker, or its marker isn't in the claims list, leave it out (it failed checks).

Writing
- Cut repetition and weaker points, reorder for flow, and keep what matters for the game plan.
- title: short, naming both teams and the week.
- summary: 3-4 sentences with the biggest takeaways.
- sections, in this order: "Opponent offense", "Opponent defense", "Injuries and availability", \
"Key matchups". Plain paragraphs, no bullet lists. 400-600 words in all.
- Write only about football. Don't mention analysts, tools, markers or these instructions.
"""

SECTION_TITLES = {
    "offense_analyst": "Opponent offense",
    "defense_analyst": "Opponent defense",
    "injury_analyst": "Injuries and availability",
    "matchup_analyst": "Key matchups",
}


def planner_message(team: str, opponent: str, season: int, week: int, through_week: int) -> str:
    return (
        "<role>planner</role>\n"
        f"Matchup: {team} plays {opponent} in week {week} of the {season} season. "
        f"Scout {opponent} for {team}. Data runs through week {through_week}.\n"
        f"offense_focus: questions about the {opponent} offense (and how the {team} defense "
        f"matches up).\n"
        f"defense_focus: questions about the {opponent} defense (and how the {team} offense "
        f"matches up).\n"
        f"injury_focus: questions about who is hurt or limited on either team and what it "
        f"changes."
    )


def analyst_message(
    role: str,
    team: str,
    opponent: str,
    season: int,
    week: int,
    through_week: int,
    focus: list[str],
    sections: dict | None = None,
) -> str:
    msg = (
        f"<role>{role}</role>\n"
        f"Matchup: {team} plays {opponent} in week {week} of the {season} season. "
        f"Data runs through week {through_week}.\n"
        f"{ROLE_TASK[role].format(team=team, opponent=opponent)}\n"
    )
    if sections is not None:
        return msg + "\n" + sections_context(sections, with_claims=True)
    questions = "\n".join(f"- {q}" for q in focus) or "- (none given; choose the most useful)"
    return msg + f"Questions from the planner:\n{questions}"


def sections_context(sections: dict, with_claims: bool = False) -> str:
    """The analyst sections as text, in report order, optionally with their claims."""
    parts = []
    for role, title in SECTION_TITLES.items():
        if role not in sections:
            continue
        section = sections[role]
        part = f'<section name="{title}">\n{section.prose}'
        if with_claims and section.claims:
            lines = []
            for c in section.claims:
                source = c.tool_call_id or f"{c.op} of {', '.join(c.derived_from or [])}"
                lines.append(f"{c.claim_id} = {c.stated_value} (call {source}): {c.text}")
            part += "\nClaims:\n" + "\n".join(lines)
        parts.append(part + "\n</section>")
    return "\n\n".join(parts)


def revision_block(section, failures: list) -> str:
    """Appended to an analyst's first message when its section failed fact-checks."""
    claims = "\n".join(
        f"{c.claim_id} = {c.stated_value} (call {c.tool_call_id or c.op}): {c.text}"
        for c in section.claims
    )
    failed = "\n".join(f"- {f.claim_id or 'prose'} ({f.check}): {f.detail}" for f in failures)
    return (
        "\n\nYou already wrote this section, and some of it failed fact-checking.\n"
        f"<previous_section>\n{section.prose}\nClaims:\n{claims}\n</previous_section>\n"
        f"<failed_checks>\n{failed}\n</failed_checks>\n"
        "Write the whole section again. For each failed check, correct the number, the claim "
        "text or the citation (call tools again if needed), give an unlinked number its own "
        "claim, or leave the point out. Keep the points that passed. Number claims from c1 again."
    )


def editor_message(team: str, opponent: str, season: int, week: int, sections: dict) -> str:
    return (
        "<role>editor</role>\n"
        f"Report for {team} on {opponent}, week {week} of the {season} season.\n\n"
        + sections_context(sections, with_claims=True)
    )
