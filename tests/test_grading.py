"""Blind hand-grading packet: markers stripped, configs hidden behind shuffled labels."""

import csv
import json

import pytest

from nfl_scout.evaluation.grading import CRITERIA, build_packet, score_grades, strip_markers

MATCHUPS = [
    {"week": 4, "team": "BUF", "opponent": "NE", "hand_graded": True},
    {"week": 5, "team": "SEA", "opponent": "SF", "hand_graded": True},
    {"week": 5, "team": "DEN", "opponent": "KC", "hand_graded": False},
]
CONFIGS = ["full", "no_checker", "single"]


def write_runs(tmp_path, configs=CONFIGS):
    for c in configs:
        for m in MATCHUPS:
            d = tmp_path / c / "r1"
            d.mkdir(parents=True, exist_ok=True)
            name = f"2026_w{m['week']:02d}_{m['team']}_vs_{m['opponent']}.json"
            report = {
                "title": f"{m['opponent']} report",
                "summary": f"Passes 61.2% [oc1] of plays [oc2, oc3].",
                "sections": [{"heading": "Opponent offense", "body": "They rank 3rd [sc4]."}],
            }
            (d / name).write_text(json.dumps({"config": c, "report": report}))


def test_strip_markers():
    assert strip_markers("Passes 61.2% [oc1] of plays [oc2, oc3].") == "Passes 61.2% of plays."


def test_packet_hides_configs_and_shuffles_labels(tmp_path):
    write_runs(tmp_path)
    out = tmp_path / "grading"
    build_packet(MATCHUPS, CONFIGS, tmp_path, out, seed=1)
    packet = (out / "packet.md").read_text()
    assert "[oc1]" not in packet and "61.2%" in packet
    for c in CONFIGS:
        assert c not in packet  # the packet itself never names a config
    key = json.loads((out / "key.json").read_text())
    assert set(key) == {"2026_w04_BUF_vs_NE", "2026_w05_SEA_vs_SF"}  # hand-graded only
    assert all(sorted(v.values()) == sorted(CONFIGS) for v in key.values())
    orders = {tuple(v[label] for label in "ABC") for v in key.values()}
    assert len(orders) > 1 or len(key) == 1  # shuffled per matchup
    with (out / "grades.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6 and set(CRITERIA) <= set(rows[0])


def test_packet_refuses_missing_runs_and_existing_packets(tmp_path):
    write_runs(tmp_path, configs=["full", "single"])
    with pytest.raises(FileNotFoundError, match="no_checker"):
        build_packet(MATCHUPS, CONFIGS, tmp_path, tmp_path / "g", seed=1)
    write_runs(tmp_path)
    build_packet(MATCHUPS, CONFIGS, tmp_path, tmp_path / "g", seed=1)
    with pytest.raises(FileExistsError):
        build_packet(MATCHUPS, CONFIGS, tmp_path, tmp_path / "g", seed=1)


def test_score_grades_unblinds_into_per_config_means(tmp_path):
    write_runs(tmp_path)
    out = tmp_path / "grading"
    build_packet(MATCHUPS, CONFIGS, tmp_path, out, seed=1)
    key = json.loads((out / "key.json").read_text())
    with (out / "grades.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:  # grade the full graph 5, others 3; rank full first
        config = key[r["matchup"]][r["label"]]
        for c in CRITERIA:
            r[c] = "5" if config == "full" else "3"
        r["rank"] = "1" if config == "full" else ("2" if config == "single" else "3")
    with (out / "grades.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = score_grades(out)
    assert result["full"]["mean"]["relevance"] == 5.0
    assert result["single"]["mean_rank"] == 2.0
    assert result["full"]["wins"] == 2 and result["full"]["n"] == 2
