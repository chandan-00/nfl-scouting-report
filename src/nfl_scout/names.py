"""Team-name check: flag names that aren't how any NFL team is called.

Deterministic (no LLM), so it runs on analyst prose, final reports and saved runs alike. It flags
two kinds of names, found in the saved reports:
- a state used in place of a team's city ("Indiana" for Indianapolis). States that are a team's
  official name (Arizona, Carolina, Minnesota, Tennessee, Washington) are fine.
- a team's former name or city ("Oakland", "Redskins").

Near-miss spellings are not flagged: across ~450 saved reports an edit-distance check found only
player names ("Charles" vs Chargers, "Settle" vs Seattle).
"""

from __future__ import annotations

import re

TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAR": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

# Wrong name -> what to write instead.
STATES = {
    "Indiana": "Indianapolis",
    "Texas": "Houston or Dallas",
    "Florida": "Miami, Tampa Bay or Jacksonville",
    "Ohio": "Cincinnati or Cleveland",
    "California": "San Francisco or Los Angeles",
    "Missouri": "Kansas City",
    "Massachusetts": "New England",
    "Wisconsin": "Green Bay",
    "Maryland": "Baltimore",
    "Louisiana": "New Orleans",
    "Colorado": "Denver",
    "Georgia": "Atlanta",
    "Pennsylvania": "Philadelphia or Pittsburgh",
    "Michigan": "Detroit",
    "Nevada": "Las Vegas",
    "Illinois": "Chicago",
}
FORMER = {
    "Oakland": "Las Vegas",
    "San Diego": "Los Angeles (Chargers)",
    "St. Louis": "Los Angeles (Rams)",
    "Redskins": "Commanders",
    "Football Team": "Commanders",
}
# "Ohio State", "Texas A&M", "Georgia Tech": colleges, not teams.
_COLLEGE = r"(?!\s+(?:State|Tech|A&M)\b)"


def team_name_problems(text: str) -> list[str]:
    """One message per wrong name in the text, in a fixed order."""
    problems = []
    for wrong, right in STATES.items():
        if re.search(rf"\b{re.escape(wrong)}\b{_COLLEGE}", text):
            problems.append(f'"{wrong}" is a state, not a team name: write {right}')
    for wrong, right in FORMER.items():
        if re.search(rf"\b{re.escape(wrong)}\b", text):
            problems.append(f'"{wrong}" is a former team name: write {right}')
    return problems
