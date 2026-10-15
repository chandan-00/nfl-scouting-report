# nfl-scouting-report

A multi-agent system that writes NFL opponent scouting reports from nflverse play-by-play data. Analyst agents draft the report, and an independent fact-checker re-runs every cited stat before the editor finalizes it. An evaluation harness measures draft and final claim accuracy, checks a sample of claims with independent SQL, and scores the fact-checker on planted errors.

> Work in progress. Results will be added here once the evaluation has run.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
cp .env.example .env   # add your ANTHROPIC_API_KEY
uv run pytest
```

## Data

```bash
uv run python -m nfl_scout.data.ingest     # download nflverse tables into data/nfl.duckdb
uv run python -m nfl_scout.data.coverage   # write docs/data_coverage.md
```

Re-run both after each week's games to refresh the in-season data. `docs/data_coverage.md` lists which fields are populated for the eval season, and therefore which metrics the agents can use.

## Configuration

`config/settings.yaml` sets the model and effort for each graph node (Claude Haiku 5.5 by default), the seasons to load, and the eval weeks. To run every node on one model for a comparison run, set `NFL_SCOUT_MODEL=haiku|sonnet|opus`.

## Layout

```
config/        settings.yaml
src/nfl_scout/ package code (data/: ingest + coverage audit)
docs/          generated reports (data coverage)
tests/         pytest suite
data/          local nflverse data and DuckDB file (git-ignored)
runs/          saved agent runs, scored offline (git-ignored)
```
