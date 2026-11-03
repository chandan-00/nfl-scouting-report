"""Fact-checker: each claim re-checked against its tool call, its wording against the call's
arguments, and the prose against the claims. No LLM."""

import pytest
from conftest import KC_DEF_ROW, RUN, build_db

from nfl_scout.checker import check_sections, team_aliases
from nfl_scout.graph.llm_tools import ToolExecutor
from nfl_scout.graph.state import Claim, Section, ToolCall


def claim(cid, value, call="o1", text="KC offense pass rate", **kw):
    return Claim(claim_id=cid, text=text, stated_value=value, tool_call_id=call, **kw)


def section(prose, *claims, role="offense_analyst"):
    return {role: Section(role=role, prose=prose, claims=list(claims))}


@pytest.fixture
def db(con):
    # KC offense: 2 passes, 1 run. KC defense (DEN offense): 1 pass.
    build_db(con, [{"epa": 0.6}, {"epa": 0.2}, {**RUN, "epa": -0.2}, {**KC_DEF_ROW, "epa": 0.1}])
    return con


def run_calls(con, *calls, prefix="o"):
    ex = ToolExecutor(season=2026, through_week=1, prefix=prefix, con=con)
    for name, args in calls:
        ex.run(name, args)
    return ex.calls


def failures(results):
    return [(r.claim_id, r.check, r.detail) for r in results if not r.ok]


def test_correct_claims_pass_every_check(db):
    calls = run_calls(db, ("pass_rate", {"team": "KC"}))
    sections = section(
        "KC drops back on 66.7% of plays [c1] over 3 plays [c2].",
        claim("c1", 66.7),
        claim("c2", 3, text="KC offense sample size for pass rate", measure="n_plays"),
    )
    assert failures(check_sections(sections, calls, db)) == []


def test_wrong_value_fails_at_the_precision_written(db):
    calls = run_calls(db, ("pass_rate", {"team": "KC"}))
    results = check_sections(section("KC 66.0% [c1].", claim("c1", 66.0)), calls, db)
    [(cid, check, detail)] = failures(results)
    assert (cid, check) == ("c1", "value")
    assert "66.667" in detail


def test_unknown_or_failed_calls_fail(db):
    calls = run_calls(db, ("player_share", {"team": "KC", "player": "Nobody", "kind": "rush"}))
    sections = section(
        "A 5.0% [c1] B 6.0% [c2].",
        claim("c1", 5.0, text="KC rush share for Nobody"),
        claim("c2", 6.0, call="o9"),
    )
    found = failures(check_sections(sections, calls, db))
    assert [(c, k) for c, k, _ in found] == [("c1", "value"), ("c2", "value")]
    assert "error" in found[0][2] and "unknown call o9" in found[1][2]


def test_rerun_that_disagrees_with_the_trace_fails(db):
    # A trace value that the tool doesn't reproduce (e.g. results mixed up across threads).
    (call,) = run_calls(db, ("pass_rate", {"team": "KC"}))
    call.result["value"] = 50.0
    results = check_sections(section("KC 50.0% [c1].", claim("c1", 50.0)), [call], db)
    assert "re-run" in failures(results)[0][2]


def test_derived_claims_are_recomputed(db):
    calls = run_calls(db, ("success_rate", {"team": "KC", "play_kind": "pass"}))
    good = Claim(claim_id="c3", text="ratio", stated_value=2, derived_from=["c1", "c2"], op="ratio")
    bad = Claim(
        claim_id="c4", text="gap", stated_value=9, derived_from=["c1", "c2"], op="difference"
    )
    sections = section(
        "A 60.0% [c1] and 30.0% [c2], twice [c3], 9 points [c4].",
        claim("c1", 60.0, text="KC offense pass success rate", call="o1"),
        claim("c2", 30.0, text="KC offense pass success rate", call="o1"),
        good,
        bad,
    )
    derived = [f for f in failures(check_sections(sections, calls, db)) if f[0] in ("c3", "c4")]
    assert [(c, k) for c, k, _ in derived] == [("c4", "value")]


def test_list_results_are_matched_on_the_named_player(db):
    calls = run_calls(db, ("usage_leaders", {"team": "KC", "kind": "target"}))
    ok = claim("c1", 100.0, text="KC target share for W.One")
    wrong = claim("c2", 50.0, text="KC target share for W.One")
    nobody = claim("c3", 100.0, text="KC top target share")
    sections = section("A 100.0% [c1] B 50.0% [c2] C 100.0% [c3].", ok, wrong, nobody)
    found = failures(check_sections(sections, calls, db))
    assert [(c, k) for c, k, _ in found] == [("c2", "value"), ("c3", "value")]
    assert "names no player" in found[1][2]


@pytest.mark.parametrize(
    ("tool", "args", "text", "problem"),
    [
        ("epa_per_play", {"team": "KC"}, "DEN offense EPA per play", "names DEN"),
        ("epa_per_play", {"team": "KC"}, "Chiefs offense EPA per play", None),
        ("epa_per_play", {"team": "KC"}, "Kansas City EPA per play", None),
        ("epa_per_play", {"team": "KC"}, "KC defense EPA per play allowed", "defense"),
        ("epa_per_play", {"team": "KC", "side": "defense"}, "KC defense EPA allowed", None),
        ("pass_rate_over_expected", {"team": "KC"}, "KC defense PROE", "defense"),
        ("epa_per_play", {"team": "KC"}, "KC EPA per pass play", "pass"),
        ("epa_per_play", {"team": "KC", "play_kind": "run"}, "KC EPA per play", "run"),
        ("epa_per_play", {"team": "KC", "play_kind": "run"}, "KC EPA per rush", None),
        ("pass_rate", {"team": "KC", "down": 3}, "KC pass rate", "3rd down"),
        ("pass_rate", {"team": "KC"}, "KC pass rate on third down", "3rd down"),
        ("pass_rate", {"team": "KC", "down": 3}, "KC pass rate on 3rd down", None),
        ("epa_per_play", {"team": "KC"}, "KC EPA per play, weeks 1-5", "weeks"),
        ("epa_per_play", {"team": "KC"}, "KC EPA per play, 2025 season", "2025"),
        (
            "ftn_play_rate",
            {"team": "KC", "feature": "play_action"},
            "KC motion rate",
            "play action",
        ),
    ],
)
def test_wording_must_match_the_arguments(db, tool, args, text, problem):
    calls = run_calls(db, (tool, args))
    results = check_sections(section("X 1.0 [c1].", claim("c1", 1.0, text=text)), calls, db)
    wording = [r.detail for r in results if r.check == "wording" and not r.ok]
    if problem is None:
        assert wording == []
    else:
        assert len(wording) == 1 and problem in wording[0]


def test_player_claims_wording_false_positives_from_the_live_run(db):
    # Seen live (DEN vs SEA, week 6): all four were correct claims the first checker flagged.
    build_db(
        db,
        [{}, {**RUN}],
        snaps=[{"team": "KC", "player": "Wide One", "week": 1, "defense_pct": 0.8}],
    )
    calls = run_calls(
        db,
        ("snap_share", {"team": "KC", "player": "Wide One"}),
        ("player_share", {"team": "KC", "player": "W.One", "kind": "target"}),
    )
    sections = section(
        "A 90.0% [c1] in 1 game [c2], 100.0% of 1 target [c3].",
        # A defensive player's snap share; the result's weeks are what the text should match.
        claim("c1", 90.0, text="KC defense snap share of Wide One, weeks 1-1"),
        # Naming the player is enough; the team isn't needed too.
        claim("c2", 1, text="Games sampled for Wide One snap share", measure="n_plays"),
        # The sample size of a player share is the team total, so no player name is needed.
        claim("c3", 1, call="o2", text="KC team targets used as the sample", measure="n_plays"),
    )
    wording = [
        (r.claim_id, r.detail)
        for r in check_sections(sections, calls, db)
        if r.check == "wording" and not r.ok
    ]
    assert wording == []


def test_naming_the_player_doesnt_excuse_naming_the_wrong_team(db):
    # Found by the seeded-error set: "DEN ... of Wide One" for a KC call passed.
    build_db(db, [{}], snaps=[{"team": "KC", "player": "Wide One"}])
    calls = run_calls(db, ("snap_share", {"team": "KC", "player": "Wide One"}))
    text = "DEN snap share of Wide One"
    results = check_sections(section("A 90.0% [c1].", claim("c1", 90.0, text=text)), calls, db)
    wording = [r.detail for r in results if r.check == "wording" and not r.ok]
    assert len(wording) == 1 and "names DEN" in wording[0]


def test_league_rank_wording_uses_the_splits_actually_applied(db):
    # direction is ignored by epa_per_play, so a claim that names no direction is fine.
    calls = run_calls(
        db, ("league_rank", {"metric": "epa_per_play", "team": "KC", "direction": "left"})
    )
    results = check_sections(
        section("KC is 1st [c1].", claim("c1", 1, text="KC offense EPA rank")), calls, db
    )
    assert failures(results) == []


def test_prose_numbers_need_claims(db):
    calls = run_calls(db, ("pass_rate", {"team": "KC"}))
    sections = section("KC drops back on 66.7% [c1] on 3 plays, 70.0% [c9].", claim("c1", 66.7))
    found = failures(check_sections(sections, calls, db))
    assert [(c, k) for c, k, _ in found] == [(None, "prose"), ("c9", "prose")]


def test_team_aliases_cover_abbr_nick_and_city(db):
    aliases = team_aliases(db)
    assert {"KC", "chiefs", "kansas city", "kansas city chiefs"} <= aliases["KC"]


def test_matchup_claims_can_cite_other_roles_calls(db):
    calls = run_calls(db, ("pass_rate", {"team": "KC"}))
    sections = {
        **section("KC 66.7% [c1].", claim("c1", 66.7)),
        **section("KC passes 66.7% [mc1].", claim("mc1", 66.7), role="matchup_analyst"),
    }
    assert failures(check_sections(sections, calls, db)) == []


def test_results_say_which_role_failed(db):
    calls = [ToolCall(call_id="o1", node="x", tool="pass_rate", args={}, error="boom")]
    (r, *_) = check_sections(section("KC 1.0 [c1].", claim("c1", 1.0)), calls, db)
    assert r.role == "offense_analyst"
