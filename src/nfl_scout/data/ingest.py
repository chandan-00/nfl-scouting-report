"""Download nflverse tables and write them to the project's DuckDB file.

Run with:  uv run python -m nfl_scout.data.ingest

Everything is loaded in full. The leakage guard (no plays after `through_week`) lives in the
tool functions, not here, so the database always holds all available data.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import duckdb
import nflreadpy as nfl
import polars as pl

from nfl_scout.config import load_settings

# DuckDB table name -> loader taking a list of seasons.
# Participation (personnel, coverage, pressure) is not published for 2026, so it isn't loaded.
SEASONAL_TABLES: dict[str, Callable[[list[int]], pl.DataFrame]] = {
    "pbp": lambda s: nfl.load_pbp(seasons=s),
    "schedules": lambda s: nfl.load_schedules(seasons=s),
    "rosters_weekly": lambda s: nfl.load_rosters_weekly(seasons=s),
    "injuries": lambda s: nfl.load_injuries(seasons=s),
    "snap_counts": lambda s: nfl.load_snap_counts(seasons=s),
    "ftn_charting": lambda s: nfl.load_ftn_charting(seasons=s),
}
STATIC_TABLES: dict[str, Callable[[], pl.DataFrame]] = {
    "teams": nfl.load_teams,
}


def write_table(con: duckdb.DuckDBPyConnection, name: str, df: pl.DataFrame) -> None:
    """Replace `name` with the contents of `df` and record the load in `_ingest_log`."""
    con.register("_incoming", df.to_arrow())
    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM _incoming')
    con.unregister("_incoming")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS _ingest_log (
            table_name VARCHAR, n_rows BIGINT, latest_week_by_season VARCHAR, loaded_at TIMESTAMP
        )
        """
    )
    con.execute("DELETE FROM _ingest_log WHERE table_name = ?", [name])
    con.execute(
        "INSERT INTO _ingest_log VALUES (?, ?, ?, ?)",
        [name, df.height, latest_weeks(df), datetime.now(UTC).replace(tzinfo=None)],
    )


def latest_weeks(df: pl.DataFrame) -> str | None:
    """Summarize the latest week per season, e.g. '2025: wk 22, 2026: wk 5'."""
    if not {"season", "week"} <= set(df.columns) or df.height == 0:
        return None
    rows = df.group_by("season").agg(pl.col("week").max()).sort("season").rows()
    return ", ".join(f"{season}: wk {week}" for season, week in rows)


def ingest(seasons: list[int] | None = None) -> dict[str, int]:
    """Load every table for `seasons` (default: settings.data.seasons). Returns row counts."""
    settings = load_settings()
    seasons = seasons or settings.data.seasons
    db_path = settings.resolve(settings.data.duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    with duckdb.connect(str(db_path)) as con:
        con.execute("DROP TABLE IF EXISTS _ingest_log")
        for name, loader in SEASONAL_TABLES.items():
            df = loader(seasons)
            write_table(con, name, df)
            counts[name] = df.height
            print(f"{name:16} {df.height:>8,} rows")
        for name, loader in STATIC_TABLES.items():
            df = loader()
            write_table(con, name, df)
            counts[name] = df.height
            print(f"{name:16} {df.height:>8,} rows")
    print(f"Wrote {db_path}")
    return counts


if __name__ == "__main__":
    ingest()
