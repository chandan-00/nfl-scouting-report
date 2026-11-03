"""Frozen eval matchups: deterministic, balanced by week, one report per game."""

import json

import pytest

from nfl_scout.evaluation.matchups import Game, freeze, select_matchups


def games(weeks=(4, 5, 6, 7), per_week=14):
    teams = [f"T{i:02d}" for i in range(32)]
    out = []
    for w in weeks:
        for g in range(per_week):
            away, home = teams[(2 * g + w) % 32], teams[(2 * g + 1 + w) % 32]
            out.append(Game(game_id=f"2026_{w:02d}_{away}_{home}", week=w, away=away, home=home))
    return out


def test_selection_is_deterministic_and_balanced():
    a = select_matchups(games(), per_week=10, n_hand_graded=10, seed=7)
    b = select_matchups(games(), per_week=10, n_hand_graded=10, seed=7)
    assert a == b
    assert len(a) == 40
    assert {w: sum(m["week"] == w for m in a) for w in (4, 5, 6, 7)} == {4: 10, 5: 10, 6: 10, 7: 10}
    assert len({m["game_id"] for m in a}) == 40  # one report per game


def test_each_matchup_scouts_the_other_team_with_data_through_the_week_before():
    for m in select_matchups(games(), per_week=10, n_hand_graded=10, seed=7):
        away, home = m["game_id"].split("_")[2:]
        assert {m["team"], m["opponent"]} == {away, home}
        assert m["through_week"] == m["week"] - 1


def test_hand_graded_subset_spans_the_weeks():
    chosen = select_matchups(games(), per_week=10, n_hand_graded=10, seed=7)
    hand = [m for m in chosen if m["hand_graded"]]
    assert len(hand) == 10
    assert {m["week"] for m in hand} == {4, 5, 6, 7}


def test_a_short_week_is_an_error():
    with pytest.raises(ValueError, match="week 7"):
        short_week_7 = games((4, 5, 6), per_week=15) + games((7,), per_week=14)
        select_matchups(short_week_7, per_week=15, n_hand_graded=10, seed=7)


def test_freeze_refuses_to_overwrite(tmp_path):
    path = tmp_path / "matchups_v1.json"
    freeze(games(), path, per_week=10, n_hand_graded=10, seed=7)
    saved = json.loads(path.read_text())
    assert saved["version"] == "v1" and len(saved["matchups"]) == 40
    with pytest.raises(FileExistsError):
        freeze(games(), path, per_week=10, n_hand_graded=10, seed=7)
