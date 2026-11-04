"""Frozen DuckDB snapshot for scoring: recorded by hash, refused if the file changed."""

import duckdb
import pytest

from nfl_scout.evaluation.snapshot import record_snapshot, snapshot_connection


def make_db(path, value=1):
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE t AS SELECT ? AS x", [value])
    con.close()


def test_recorded_snapshot_opens_read_only(tmp_path):
    db = tmp_path / "snap.duckdb"
    make_db(db)
    manifest = tmp_path / "snapshot.json"
    record_snapshot(db, manifest, note="test")
    con = snapshot_connection(manifest)
    assert con.execute("SELECT x FROM t").fetchone() == (1,)
    with pytest.raises(duckdb.Error):
        con.execute("CREATE TABLE u (y INT)")


def test_a_changed_snapshot_is_refused(tmp_path):
    db = tmp_path / "snap.duckdb"
    make_db(db)
    manifest = tmp_path / "snapshot.json"
    record_snapshot(db, manifest, note="test")
    db.unlink()
    make_db(db, value=2)
    with pytest.raises(ValueError, match="sha256"):
        snapshot_connection(manifest)
