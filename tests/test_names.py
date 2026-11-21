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


# --- Venue -------------------------------------------------------------------------------------

from nfl_scout.names import venue_problems  # noqa: E402


def test_away_at_home_passes_in_any_naming():
    for text in ("IND at HOU Week 11", "Colts at Texans", "Indianapolis @ Houston",
                 "Indianapolis Colts at the Houston Texans"):  # fmt: skip
        assert venue_problems(text, home="HOU", away="IND") == [], text


def test_backwards_venue_is_flagged():
    problems = venue_problems("Green Bay at Dallas Week 6", home="GB", away="DAL")
    assert len(problems) == 1
    assert "DAL at GB" in problems[0]


def test_visits_and_hosts_name_the_home_team_too():
    assert venue_problems("Colts Visit Texans", home="HOU", away="IND") == []
    assert venue_problems("Houston visits Indianapolis", home="HOU", away="IND") != []
    assert venue_problems("Houston hosts the Colts", home="HOU", away="IND") == []
    assert venue_problems("Indianapolis hosts Houston", home="HOU", away="IND") != []


def test_at_without_two_teams_is_not_a_venue():
    text = "IND at 36.4% on third down; HOU at home is strong; the Colts at their best"
    assert venue_problems(text, home="HOU", away="IND") == []


def test_the_rams_two_codes_both_count():
    assert venue_problems("SF at LA", home="LA", away="SF") == []
    assert venue_problems("Rams at 49ers", home="LA", away="SF") != []


def test_team_codes_match_only_in_capitals():
    assert venue_problems("Atlanta at no point trailed", home="ATL", away="NO") == []
    assert venue_problems("ATL at NO", home="ATL", away="NO") != []
