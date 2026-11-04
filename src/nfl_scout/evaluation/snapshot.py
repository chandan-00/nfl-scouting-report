"""The frozen DuckDB file eval runs are scored against.

Scoring re-runs every cited tool call. After a weekly data refresh, nflverse stat corrections
could change past weeks, and a correct claim would then look wrong. So scoring opens a copy of
the database taken when the eval runs were generated, recorded by hash in eval/snapshot.json,
and refuses to use it if the file has changed.

    uv run python -m nfl_scout.evaluation.snapshot data/snapshots/<file>.duckdb --note "..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from nfl_scout.config import PROJECT_ROOT
from nfl_scout.evaluation.matchups import EVAL_DIR

MANIFEST = EVAL_DIR / "snapshot.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def record_snapshot(db: Path, manifest: Path = MANIFEST, note: str = "") -> dict:
    record = {
        "path": _rel(db),
        "sha256": _sha256(db),
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": note,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=1) + "\n")
    return record


def snapshot_connection(manifest: Path = MANIFEST) -> duckdb.DuckDBPyConnection:
    """Read-only connection to the recorded snapshot, after checking its hash."""
    record = json.loads(manifest.read_text())
    path = Path(record["path"])
    path = path if path.is_absolute() else PROJECT_ROOT / path
    actual = _sha256(path)
    if actual != record["sha256"]:
        raise ValueError(f"{path} sha256 {actual[:12]} != recorded {record['sha256'][:12]}")
    return duckdb.connect(str(path), read_only=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db", type=Path)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    print(record_snapshot(a.db, note=a.note))


if __name__ == "__main__":
    main()
