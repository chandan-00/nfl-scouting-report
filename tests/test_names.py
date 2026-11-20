from nfl_scout.names import TEAM_NAMES, team_name_problems


def test_a_state_name_used_for_a_team_is_flagged_with_the_city():
    problems = team_name_problems("Indiana's run game leans right [oc5].")
    assert len(problems) == 1
    assert "Indiana" in problems[0] and "Indianapolis" in problems[0]


def test_correct_team_names_pass():
    text = (
        "Indianapolis leans on the run; the Colts and IND rank 22nd. Houston, the Texans, "
        "Arizona, Minnesota, Tennessee, Carolina, New England and Washington are fine."
    )
    assert team_name_problems(text) == []


def test_former_team_names_are_flagged():
    problems = team_name_problems("Oakland and the Redskins")
    assert any("Las Vegas" in p for p in problems)
    assert any("Commanders" in p for p in problems)


def test_college_names_are_not_flagged():
    assert team_name_problems("The rookie from Ohio State and Texas A&M") == []


def test_each_wrong_name_is_reported_once():
    assert len(team_name_problems("Indiana runs. Indiana passes.")) == 1


def test_every_current_team_has_a_full_name():
    assert len({v for k, v in TEAM_NAMES.items() if k != "LAR"}) == 32
    assert TEAM_NAMES["IND"] == "Indianapolis Colts"
