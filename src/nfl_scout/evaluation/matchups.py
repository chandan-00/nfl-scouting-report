"""Freeze the eval matchups: 2026 games from the eval weeks, balanced by week, one report per
game (a seeded coin flip picks which team the report is for). The file is versioned and never
overwritten, so every config is scored on the same matchups.

    uv run python -m nfl_scout.evaluation.matchups
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from nfl_scout.config import PROJECT_ROOT, load_settings
from nfl_scout.tools.base import default_connection

EVAL_DIR = PROJECT_ROOT / "eval"
SEED = 2026


@dataclass(frozen=True)
class Game:
    game_id: str
    week: int
    away: str
    home: str


def eval_games(con: duckdb.DuckDBPyConnection, season: int, weeks: list[int]) -> list[Game]:
    """Regular-season games in `weeks` that have been played (both scores set)."""
    rows = con.execute(
        """
        SELECT game_id, week, away_team, home_team FROM schedules
        WHERE season = ? AND game_type = 'REG' AND list_contains(?, week)
          AND away_score IS NOT NULL AND home_score IS NOT NULL
        ORDER BY week, game_id
        """,
        [season, weeks],
    ).fetchall()
    return [Game(*r) for r in rows]


def select_matchups(
    games: list[Game], per_week: int, n_hand_graded: int, seed: int = SEED
) -> list[dict]:
    """`per_week` games from each week, each scouted from one side, plus a hand-graded subset
    spread across the weeks."""
    rng = random.Random(seed)
    weeks = sorted({g.week for g in games})
    chosen: list[dict] = []
    for week in weeks:
        pool = sorted((g for g in games if g.week == week), key=lambda g: g.game_id)
        if len(pool) < per_week:
            raise ValueError(f"week {week} has {len(pool)} played games, need {per_week}")
        for g in sorted(rng.sample(pool, per_week), key=lambda g: g.game_id):
            team, opponent = (g.home, g.away) if rng.random() < 0.5 else (g.away, g.home)
            chosen.append(
                {
                    "game_id": g.game_id,
                    "week": week,
                    "team": team,
                    "opponent": opponent,
                    "through_week": week - 1,
                    "hand_graded": False,
                }
            )
    # Hand-graded subset: round-robin over the weeks so every week is represented.
    by_week = {w: rng.sample([m for m in chosen if m["week"] == w], per_week) for w in weeks}
    for i in range(n_hand_graded):
        by_week[weeks[i % len(weeks)]][i // len(weeks)]["hand_graded"] = True
    return chosen


def freeze(
    games: list[Game], path: Path, per_week: int, n_hand_graded: int, seed: int = SEED
) -> list[dict]:
    if path.exists():
        raise FileExistsError(f"{path} is frozen; write a new version instead")
    matchups = select_matchups(games, per_week, n_hand_graded, seed)
    record = {
        "version": path.stem.rsplit("_", 1)[-1],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "per_week": per_week,
        "matchups": matchups,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")
    return matchups


def load_matchups(version: str = "v1") -> list[dict]:
    return json.loads((EVAL_DIR / f"matchups_{version}.json").read_text())["matchups"]


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--hand-graded", type=int, default=10)
    a = ap.parse_args()
    weeks = settings.eval.weeks
    per_week = settings.eval.n_matchups // len(weeks)
    games = eval_games(default_connection(), settings.data.eval_season, weeks)
    path = EVAL_DIR / f"matchups_{a.version}.json"
    matchups = freeze(games, path, per_week, a.hand_graded)
    print(f"froze {len(matchups)} matchups to {path}")


if __name__ == "__main__":
    main()
