"""Audit which fields are populated for the eval season and write docs/data_coverage.md.

Run with:  uv run python -m nfl_scout.data.coverage

The report decides which metric functions (tools) can be built: a metric is only offered to the
agents if its source fields are populated for the eval season.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from nfl_scout.config import PROJECT_ROOT, load_settings

REPORT_PATH = PROJECT_ROOT / "docs" / "data_coverage.md"

# A field counts as available if at least this share of its relevant plays is non-null.
AVAILABLE_THRESHOLD = 0.95

# Play subsets that each field should be measured against (null on other plays is expected).
SUBSETS = {
    "scrimmage": "play_type IN ('pass', 'run')",
    "dropbacks": "play_type IN ('pass', 'run') AND qb_dropback = 1",
    "pass_attempts": "play_type = 'pass' AND pass_attempt = 1 AND sack = 0",
    "designed_runs": "play_type = 'run' AND qb_scramble = 0",
    # run_gap is left null on runs up the middle, so measure it on outside runs only.
    "outside_runs": "play_type = 'run' AND qb_scramble = 0 AND run_location IN ('left', 'right')",
}

# (field, subset) pairs checked in play-by-play.
PBP_FIELDS = [
    ("epa", "scrimmage"),
    ("yards_gained", "scrimmage"),
    ("success", "scrimmage"),
    ("down", "scrimmage"),
    ("ydstogo", "scrimmage"),
    ("yardline_100", "scrimmage"),
    ("shotgun", "scrimmage"),
    ("no_huddle", "scrimmage"),
    ("xpass", "scrimmage"),
    ("pass_oe", "scrimmage"),
    ("sack", "dropbacks"),
    ("qb_hit", "dropbacks"),
    ("air_yards", "pass_attempts"),
    ("pass_location", "pass_attempts"),
    ("receiver_player_id", "pass_attempts"),
    ("cpoe", "pass_attempts"),
    ("rusher_player_id", "designed_runs"),
    ("run_location", "designed_runs"),
    ("run_gap", "outside_runs"),
]

FTN_FIELDS = [
    "n_blitzers",
    "n_pass_rushers",
    "n_defense_box",
    "n_offense_backfield",
    "is_play_action",
    "is_motion",
    "is_screen_pass",
    "is_rpo",
    "is_no_huddle",
]

# Fields from nflverse participation data, which is not published for the eval season.
UNAVAILABLE = [
    ("offense_personnel / defense_personnel", "personnel groupings (11, 12, nickel, dime)"),
    ("offense_formation", "formation"),
    ("defense_coverage_type / defense_man_zone_type", "coverage shell, man vs. zone"),
    ("was_pressure", "true pressure rate (use sacks + QB hits as a proxy)"),
    ("time_to_throw", "time to throw"),
    ("route", "route run by each receiver"),
]


@dataclass
class Metric:
    name: str
    sources: list[str]  # fields that must be available ("ftn.<field>" for FTN charting)


PLANNED_METRICS = [
    Metric("Pass rate by down and distance", ["down", "ydstogo"]),
    Metric("EPA/play (pass, run) and allowed", ["epa"]),
    Metric("Success rate", ["success"]),
    Metric("Early-down pass rate / pass rate over expected", ["down", "pass_oe"]),
    Metric("Red-zone TD rate (off. and def.)", ["yardline_100"]),
    Metric("Explosive-play rate (20+ yd pass, 10+ yd run)", ["yards_gained"]),
    Metric("Target share by player", ["receiver_player_id"]),
    Metric("Rush share by player", ["rusher_player_id"]),
    Metric("Run direction (left/middle/right) and gap", ["run_location", "run_gap"]),
    Metric("Depth of target (air yards)", ["air_yards"]),
    Metric("Sack rate", ["sack"]),
    Metric("Pressure proxy (sacks + QB hits per dropback)", ["sack", "qb_hit"]),
    Metric("Blitz rate (5+ rushers) and rushers per dropback", ["ftn.n_blitzers", "ftn.n_pass_rushers"]),
    Metric("Defenders in the box", ["ftn.n_defense_box"]),
    Metric("Play-action / motion / screen / RPO rates", ["ftn.is_play_action", "ftn.is_motion",
                                                       "ftn.is_screen_pass", "ftn.is_rpo"]),
    Metric("Third-down conversion / stop rate", ["down"]),
    Metric("Personnel groupings", ["participation"]),
    Metric("Coverage type (man/zone)", ["participation"]),
]


def pbp_field_coverage(con: duckdb.DuckDBPyConnection, season: int) -> dict[str, tuple[float, int]]:
    """Share of non-null values for each PBP field over its relevant plays: field -> (share, n)."""
    out = {}
    for field, subset in PBP_FIELDS:
        share, n = con.execute(
            f"""
            SELECT avg(CASE WHEN "{field}" IS NOT NULL THEN 1.0 ELSE 0.0 END), count(*)
            FROM pbp
            WHERE season = ? AND season_type = 'REG' AND {SUBSETS[subset]}
            """,
            [season],
        ).fetchone()
        out[field] = (share or 0.0, n)
    return out


def ftn_coverage(con: duckdb.DuckDBPyConnection, season: int) -> tuple[float, int, dict[str, float]]:
    """Share of scrimmage plays with an FTN charting row, and non-null share of each FTN field."""
    joined = f"""
        WITH _joined AS (
            SELECT f.*, f.nflverse_play_id IS NOT NULL AS has_ftn
            FROM pbp p
            LEFT JOIN ftn_charting f
              ON f.nflverse_game_id = p.game_id
             AND f.nflverse_play_id = CAST(p.play_id AS INTEGER)
            WHERE p.season = $season AND p.season_type = 'REG' AND {SUBSETS["scrimmage"]}
        )
    """
    params = {"season": season}
    matched, n = con.execute(
        joined + "SELECT avg(has_ftn::INT), count(*) FROM _joined", params
    ).fetchone()
    fields = {
        f: con.execute(
            joined
            + f'SELECT avg(CASE WHEN "{f}" IS NOT NULL THEN 1.0 ELSE 0.0 END) '
            "FROM _joined WHERE has_ftn",
            params,
        ).fetchone()[0]
        or 0.0
        for f in FTN_FIELDS
    }
    return matched or 0.0, n, fields


def build_report() -> str:
    settings = load_settings()
    season = settings.data.eval_season
    db_path = settings.resolve(settings.data.duckdb_path)

    with duckdb.connect(str(db_path), read_only=True) as con:
        log = con.execute(
            "SELECT table_name, n_rows, latest_week_by_season, loaded_at FROM _ingest_log"
        ).fetchall()
        games = con.execute(
            """
            SELECT week, count(DISTINCT game_id) FROM pbp
            WHERE season = ? AND season_type = 'REG' GROUP BY week ORDER BY week
            """,
            [season],
        ).fetchall()
        scheduled = dict(
            con.execute(
                "SELECT week, count(*) FROM schedules WHERE season = ? AND game_type = 'REG' "
                "GROUP BY week",
                [season],
            ).fetchall()
        )
        pbp_cov = pbp_field_coverage(con, season)
        ftn_matched, ftn_n, ftn_cov = ftn_coverage(con, season)

    available = {f for f, (share, _) in pbp_cov.items() if share >= AVAILABLE_THRESHOLD}
    if ftn_matched >= AVAILABLE_THRESHOLD:
        available |= {f"ftn.{f}" for f, share in ftn_cov.items() if share >= AVAILABLE_THRESHOLD}

    lines = [
        f"# Data coverage ({season} season)",
        "",
        "Generated by `uv run python -m nfl_scout.data.coverage`. Re-run after each data refresh.",
        f"A field counts as available when at least {AVAILABLE_THRESHOLD:.0%} of its relevant plays "
        "are non-null.",
        "",
        "## Loaded tables",
        "",
        "| Table | Rows | Latest week by season | Loaded (UTC) |",
        "|---|---|---|---|",
        *(f"| {t} | {n:,} | {w or '-'} | {ts:%Y-%m-%d %H:%M} |" for t, n, w, ts in log),
        "",
        f"## {season} games with play-by-play",
        "",
        "| Week | Games with PBP | Games scheduled |",
        "|---|---|---|",
        *(f"| {wk} | {g} | {scheduled.get(wk, '-')} |" for wk, g in games),
        "",
        "## Play-by-play fields",
        "",
        "| Field | Measured over | Non-null | Plays |",
        "|---|---|---|---|",
        *(
            f"| `{f}` | {subset} | {pbp_cov[f][0]:.1%} | {pbp_cov[f][1]:,} |"
            for f, subset in PBP_FIELDS
        ),
        "",
        "## FTN charting",
        "",
        f"Scrimmage plays with an FTN row: **{ftn_matched:.1%}** of {ftn_n:,}.",
        "",
        "| Field | Non-null (matched plays) |",
        "|---|---|",
        *(f"| `{f}` | {share:.1%} |" for f, share in ftn_cov.items()),
        "",
        f"## Not available for {season}",
        "",
        "nflverse participation data is only published through 2025, so these are unavailable:",
        "",
        *(f"- `{f}`: {desc}" for f, desc in UNAVAILABLE),
        "",
        "## Planned metrics",
        "",
        "| Metric | Source fields | Buildable |",
        "|---|---|---|",
        *(
            f"| {m.name} | {', '.join(f'`{s}`' for s in m.sources)} | "
            f"{'yes' if all(s in available for s in m.sources) else '**no**'} |"
            for m in PLANNED_METRICS
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report())
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
