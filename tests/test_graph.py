"""Graph plumbing with a scripted fake client: no API calls."""

import json
import threading
from types import SimpleNamespace as NS

import pytest
from conftest import KC_DEF_ROW, RUN, build_db

from nfl_scout.config import load_settings
from nfl_scout.graph.build import ROLE_TOOLS, build_graph
from nfl_scout.graph.llm_tools import ToolExecutor, tool_schemas
from nfl_scout.graph.state import Claim, Request, parse_section

TEAMS = ["DEN", "KC", "LAC", "LV"]


# --- Fake Anthropic client ----------------------------------------------------------------------


def _usage(inp=100, out=50, cache_read=0):
    return NS(
        input_tokens=inp,
        output_tokens=out,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=0,
    )


def text_response(payload: dict):
    return NS(
        content=[NS(type="text", text=json.dumps(payload))],
        stop_reason="end_turn",
        usage=_usage(),
    )


def tool_response(*calls):
    blocks = [
        NS(type="tool_use", id=f"toolu_{i}", name=n, input=a) for i, (n, a) in enumerate(calls)
    ]
    return NS(content=blocks, stop_reason="tool_use", usage=_usage(cache_read=40))


class FakeClient:
    """Returns scripted responses keyed by the node name found in the request."""

    def __init__(self, scripts: dict[str, list]):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.requests: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        first = kwargs["messages"][0]["content"]
        node = next(n for n in self.scripts if f"<role>{n}</role>" in first)
        return self.scripts[node].pop(0)

    def stream(self, **kwargs):
        response = self.create(**{**kwargs, "_streamed": True})
        return _Stream(response)


class _Stream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.response


# --- Tool schemas ----------------------------------------------------------------------------


def test_schemas_are_strict_and_hide_the_cutoff():
    schemas = {s["name"]: s for s in tool_schemas(TEAMS)}
    epa = schemas["epa_per_play"]
    assert epa["strict"] is True
    props = epa["input_schema"]["properties"]
    assert "season" not in props and "through_week" not in props
    assert props["team"]["enum"] == TEAMS
    assert props["side"]["enum"] == ["offense", "defense"]
    assert epa["input_schema"]["required"] == ["team"]
    assert epa["input_schema"]["additionalProperties"] is False


def test_league_rank_schema_lists_rankable_metrics_and_their_splits():
    rank = {s["name"]: s for s in tool_schemas(TEAMS)}["league_rank"]["input_schema"]
    assert "epa_per_play" in rank["properties"]["metric"]["enum"]
    assert "player_share" not in rank["properties"]["metric"]["enum"]
    assert {"side", "play_kind", "down", "feature"} <= set(rank["properties"])
    assert set(rank["required"]) == {"metric", "team"}


def test_league_rank_description_lists_splits_per_metric():
    rank = {s["name"]: s for s in tool_schemas(TEAMS)}["league_rank"]
    assert "epa_per_play: play_kind, side" in rank["description"]
    assert "average_depth_of_target: none" in rank["description"]


def test_league_rank_result_hides_the_team_value_from_the_model(con):
    build_db(con, [{"epa": 0.5}])
    ex = ToolExecutor(season=2026, through_week=1, prefix="o", con=con)
    payload = json.loads(ex.run("league_rank", {"metric": "epa_per_play", "team": "KC"})[0])
    assert "team_value" not in payload
    assert ex.calls[0].result["query_meta"]["team_value"] == 0.5


def test_schemas_are_deterministic():
    assert json.dumps(tool_schemas(TEAMS)) == json.dumps(tool_schemas(TEAMS))


# --- Executor ----------------------------------------------------------------------------------


def test_executor_injects_cutoff_and_numbers_calls(con):
    build_db(con, [{"epa": 0.5}, {"week": 3, "epa": 9.0}])
    ex = ToolExecutor(season=2026, through_week=2, prefix="o", con=con)
    content, is_error = ex.run("epa_per_play", {"team": "KC"})
    payload = json.loads(content)
    assert not is_error
    assert payload["call_id"] == "o1"
    assert payload["value"] == 0.5
    call = ex.calls[0]
    assert call.args == {"team": "KC", "season": 2026, "through_week": 2}
    assert call.result["value"] == 0.5
    assert ex.run("epa_per_play", {"team": "KC"})[0].count('"o2"') == 1


def test_executor_ignores_cutoff_args_from_the_model(con):
    build_db(con, [{"epa": 0.5}, {"week": 3, "epa": 9.0}])
    ex = ToolExecutor(season=2026, through_week=2, prefix="o", con=con)
    content, is_error = ex.run("epa_per_play", {"team": "KC", "through_week": 3})
    assert is_error
    assert "through_week" in content


def test_executor_returns_tool_errors_to_the_model(con):
    build_db(con, [{}])
    ex = ToolExecutor(season=2026, through_week=1, prefix="d", con=con)
    content, is_error = ex.run("player_share", {"team": "KC", "player": "Nobody", "kind": "rush"})
    assert is_error
    assert "not found" in content
    assert ex.calls[0].error


def test_executor_league_rank_drops_null_splits(con):
    build_db(con, [{"epa": 0.5}])
    ex = ToolExecutor(season=2026, through_week=1, prefix="o", con=con)
    content, is_error = ex.run(
        "league_rank", {"metric": "epa_per_play", "team": "KC", "side": "offense", "down": None}
    )
    assert not is_error
    assert json.loads(content)["value"] == 1


def test_parallel_executors_on_one_connection_get_their_own_results(con):
    # The analysts run in parallel threads; a shared DuckDB connection mixed up their results.
    build_db(con, [{"epa": 0.5}, {**RUN, "epa": 0.1}, {**KC_DEF_ROW, "epa": -0.3}])
    errors: list[str] = []

    def work(prefix, team, expected):
        ex = ToolExecutor(season=2026, through_week=1, prefix=prefix, con=con)
        for _ in range(150):
            content, is_error = ex.run("epa_per_play", {"team": team})
            value = json.loads(content).get("value")
            if is_error or value != expected:
                errors.append(content)

    threads = [
        threading.Thread(target=work, args=("o", "KC", 0.3)),
        threading.Thread(target=work, args=("d", "DEN", -0.3)),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []


# --- Sections and claims -----------------------------------------------------------------------


def test_parse_section_namespaces_claims_and_markers():
    raw = {
        "prose": "KC passes a lot [c1], twice as often as runs [c2].",
        "claims": [
            {"claim_id": "c1", "text": "KC pass rate", "stated_value": 66.7, "tool_call_id": "o1"},
            {
                "claim_id": "c2",
                "text": "twice as often",
                "stated_value": 2,
                "derived_from": ["c1"],
                "op": "ratio",
            },
        ],
    }
    section = parse_section("offense", raw)
    assert section.prose == "KC passes a lot [oc1], twice as often as runs [oc2]."
    assert [c.claim_id for c in section.claims] == ["oc1", "oc2"]
    assert section.claims[1].derived_from == ["oc1"]
    assert section.claims[0].stated_value == 66.7


def test_claim_needs_a_source():
    with pytest.raises(ValueError):
        Claim(claim_id="c1", text="x", stated_value=1.0)


# --- Graph -------------------------------------------------------------------------------------


def _claim(cid, value, call_id, measure="value", text="KC offense pass rate"):
    return {
        "claim_id": cid,
        "text": text,
        "stated_value": value,
        "tool_call_id": call_id,
        "derived_from": None,
        "op": None,
        "measure": measure,
    }


def _scripts():
    plan = {
        "offense_focus": ["How often do they pass?"],
        "defense_focus": ["Do they blitz?"],
        "injury_focus": ["Who is out?"],
    }
    offense = [
        tool_response(("pass_rate", {"team": "KC"})),
        text_response(
            {
                "prose": "KC drops back on 50.0% of plays [c1] on 2 plays [c2].",
                "claims": [_claim("c1", 50.0, "o1"), _claim("c2", 2, "o1", "n_plays")],
            }
        ),
    ]
    defense = [text_response({"prose": "Not enough data.", "claims": []})]
    injury = [
        tool_response(("injury_report", {"team": "KC"})),
        text_response({"prose": "Nobody was listed.", "claims": []}),
    ]
    matchup = [
        text_response(
            {"prose": "KC throws 50.0% of the time [c1].", "claims": [_claim("c1", 50.0, "o1")]}
        )
    ]
    report = {
        "title": "KC scouting report",
        "summary": "KC drops back on 50.0% of plays [oc1].",
        "sections": [
            {"heading": "Opponent offense", "body": "KC passes on 50.0% [oc1] of 2 plays."},
            {"heading": "Key matchups", "body": "KC throws 50.0% of the time [mc1]."},
        ],
    }
    return {
        "planner": [text_response(plan)],
        "offense_analyst": offense,
        "defense_analyst": defense,
        "injury_analyst": injury,
        "matchup_analyst": matchup,
        "editor": [text_response(report)],
    }


def _run(con, client):
    return build_graph(client=client, con=con, teams=TEAMS).invoke(
        {"request": Request(team="DEN", opponent="KC", season=2026, week=2)}
    )


def test_graph_runs_planner_then_parallel_analysts(con):
    build_db(con, [{}, {**RUN}, {**KC_DEF_ROW}])
    out = _run(con, FakeClient(_scripts()))

    assert out["plan"]["offense_focus"] == ["How often do they pass?"]
    offense = out["sections"]["offense_analyst"]
    assert [c.claim_id for c in offense.claims] == ["oc1", "oc2"]
    assert offense.claims[1].measure == "n_plays"
    assert out["sections"]["defense_analyst"].claims == []
    # The tool value lives in the trace, not in the section.
    calls = {c.call_id: c for c in out["tool_calls"]}
    assert set(calls) == {"o1", "i1"}
    assert (calls["o1"].args["through_week"], calls["o1"].result["value"]) == (1, 50.0)
    nodes = {u.node for u in out["usage"]}
    assert nodes == {
        "planner",
        "offense_analyst",
        "defense_analyst",
        "injury_analyst",
        "matchup_analyst",
        "editor",
    }


def test_matchup_analyst_sees_every_section_and_can_cite_their_calls(con):
    build_db(con, [{}, {**RUN}])
    client = FakeClient(_scripts())
    out = _run(con, client)

    (first,) = [
        r["messages"][0]["content"]
        for r in client.requests
        if "<role>matchup_analyst</role>" in r["messages"][0]["content"]
    ]
    assert "KC drops back on 50.0% of plays [oc1]" in first
    assert "Nobody was listed." in first and "Not enough data." in first
    assert "oc1 = 50.0 (call o1" in first
    (claim,) = out["sections"]["matchup_analyst"].claims
    assert (claim.claim_id, claim.tool_call_id) == ("mc1", "o1")


def test_editor_report_keeps_markers_and_numbers_are_checked(con):
    build_db(con, [{}, {**RUN}])
    out = _run(con, FakeClient(_scripts()))

    report = out["report"]
    assert report.title == "KC scouting report"
    assert "## Key matchups" in report.markdown()
    statuses = [(c.mention.text, c.mention.claim_id, c.status) for c in out["number_checks"]]
    assert statuses == [
        ("50.0%", "oc1", "ok"),
        ("50.0%", "oc1", "ok"),
        ("2", None, "unlinked"),  # the editor dropped the sample-size marker
        ("50.0%", "mc1", "ok"),
    ]


def test_each_analyst_keeps_a_stable_cached_prefix(con):
    build_db(con, [{}, {**RUN}])
    client = FakeClient(_scripts())
    _run(con, client)
    analyst = [r for r in client.requests if r.get("tools")]
    by_node: dict[str, set] = {}
    for r in analyst:
        node = r["messages"][0]["content"].split("</role>")[0].removeprefix("<role>")
        by_node.setdefault(node, set()).add(json.dumps(r["tools"]))
    assert all(len(v) == 1 for v in by_node.values())
    assert len({json.dumps(r["system"]) for r in analyst}) == 1
    assert all(r["tool_choice"] == {"type": "auto"} for r in analyst)
    assert all(r["thinking"] == {"type": "adaptive"} for r in client.requests)


def test_role_tools_fit_the_strict_tool_limit():
    for role, names in ROLE_TOOLS.items():
        schemas = tool_schemas(TEAMS, names)
        assert [s["name"] for s in schemas] == sorted(names)
        assert len(schemas) <= 20, role  # API limit on strict tools per request
    assert "blitz_rate" not in ROLE_TOOLS["offense_analyst"]
    assert "pass_rate" not in ROLE_TOOLS["defense_analyst"]
    assert "injury_report" in ROLE_TOOLS["injury_analyst"]


def test_through_week_is_week_minus_one():
    assert Request(team="DEN", opponent="KC", season=2026, week=6).through_week == 5


def test_analyst_is_made_to_answer_on_its_last_turn(con):
    # A model that keeps calling tools gets tool_choice "none" on the final turn and must answer.
    from nfl_scout.graph import nodes

    build_db(con, [{}])
    loop = [tool_response(("epa_per_play", {"team": "KC"})) for _ in range(nodes.MAX_TURNS - 1)]
    final = text_response({"prose": "Done.", "claims": []})
    client = FakeClient({"offense_analyst": [*loop, final]})
    llm = nodes.LLM(client, load_settings())
    tools = tool_schemas(TEAMS, ROLE_TOOLS["offense_analyst"])
    state = {"request": Request(team="DEN", opponent="KC", season=2026, week=2)}
    out = nodes.analyst(state, "offense_analyst", llm, tools, con)

    assert out["sections"]["offense_analyst"].prose == "Done."
    assert [r["tool_choice"]["type"] for r in client.requests][-2:] == ["auto", "none"]
    last_user = client.requests[-1]["messages"][-1]["content"]
    assert last_user[-1]["type"] == "text"


# --- Fact-checker and revise loop ------------------------------------------------------------


def _offense(value):
    return text_response(
        {
            "prose": f"KC drops back on {value}% of plays [c1].",
            "claims": [_claim("c1", value, "o1")],
        }
    )


def test_failed_claim_goes_back_to_its_analyst_and_draft_is_kept(con):
    build_db(con, [{}, {**RUN}])
    scripts = _scripts()
    # First answer says 60.0 (the tool says 50.0); the revision fixes it.
    scripts["offense_analyst"] = [tool_response(("pass_rate", {"team": "KC"})), _offense(60.0)]
    scripts["offense_analyst"] += [_offense(50.0)]
    client = FakeClient(scripts)
    out = _run(con, client)

    assert out["draft"]["offense_analyst"].claims[0].stated_value == 60.0
    assert out["sections"]["offense_analyst"].claims[0].stated_value == 50.0
    assert out["checks_run"] == 2 and out["dropped"] == []
    (failed,) = out["check_log"]
    assert (failed["round"], failed["role"], failed["claim_id"], failed["check"]) == (
        1,
        "offense_analyst",
        "oc1",
        "value",
    )
    revision = [
        r["messages"][0]["content"]
        for r in client.requests
        if "<role>offense_analyst</role>" in r["messages"][0]["content"]
    ][-1]
    assert "<failed_checks>" in revision and "stated 60.0" in revision
    # The reviser's own tool calls keep numbering after the first pass (no clash with o1).
    assert len({c.call_id for c in out["tool_calls"]}) == len(out["tool_calls"])


def test_claims_still_failing_after_the_last_loop_are_dropped(con):
    build_db(con, [{}, {**RUN}])
    scripts = _scripts()
    scripts["offense_analyst"] = [tool_response(("pass_rate", {"team": "KC"}))]
    scripts["offense_analyst"] += [_offense(60.0)] * 3  # draft + 2 revisions, all wrong
    client = FakeClient(scripts)
    out = _run(con, client)

    assert out["checks_run"] == 3
    assert [d["claim_id"] for d in out["dropped"]] == ["oc1"]
    assert out["sections"]["offense_analyst"].claims == []
    assert {e["round"] for e in out["check_log"]} == {1, 2, 3}
    editor = next(
        r["messages"][0]["content"]
        for r in client.requests
        if "<role>editor</role>" in r["messages"][0]["content"]
    )
    assert "oc1 = " not in editor  # the dropped claim isn't offered to the editor


def test_run_is_saved_with_draft_final_and_cost(con, tmp_path):
    from nfl_scout.runs import load_run, save_run

    build_db(con, [{}, {**RUN}])
    out = _run(con, FakeClient(_scripts()))
    path = save_run(out, "full", load_settings(), runs_dir=tmp_path)

    assert path == tmp_path / "full" / "2026_w02_DEN_vs_KC.json"
    run = load_run(path)
    assert run["config"] == "full"
    assert run["draft"]["offense_analyst"]["claims"][0]["claim_id"] == "oc1"
    assert run["report"]["title"] == "KC scouting report"
    assert run["tool_calls"][0]["call_id"] in {"o1", "i1"}
    assert run["cost_usd"]["total"] > 0
    assert run["number_checks"][0]["status"] == "ok"


# --- Baselines -----------------------------------------------------------------------------


def _single_scripts():
    report = {
        "title": "KC scouting report",
        "summary": "KC drops back on 50.0% of plays [c1].",
        "sections": [{"heading": "Opponent offense", "body": "Over 2 plays [c2]."}],
        "claims": [_claim("c1", 50.0, "s1"), _claim("c2", 2, "s1", "n_plays")],
    }
    return {
        "single_agent": [
            tool_response(("pass_rate", {"team": "KC"})),
            NS(content=[NS(type="text", text="Done.")], stop_reason="end_turn", usage=_usage()),
            text_response(report),
        ]
    }


def test_single_agent_writes_the_whole_report_in_the_same_claim_format(con):
    build_db(con, [{}, {**RUN}])
    client = FakeClient(_single_scripts())
    out = build_graph(client=client, con=con, teams=TEAMS, config="single").invoke(
        {"request": Request(team="DEN", opponent="KC", season=2026, week=2)}
    )

    section = out["sections"]["single_agent"]
    assert [c.claim_id for c in section.claims] == ["sc1", "sc2"]
    assert "[sc1]" in out["report"].summary and "[sc2]" in out["report"].sections[0].body
    assert out["draft"]["single_agent"].claims == section.claims
    assert [c.status for c in out["number_checks"]] == ["ok", "ok"]
    assert {u.node for u in out["usage"]} == {"single_agent"}
    # Tool turns carry no output schema (20 strict tools + a schema is too large a grammar);
    # the last call forbids tools and asks for the structured report.
    assert ["format" in r["output_config"] for r in client.requests] == [False, False, True]
    assert client.requests[-1]["tool_choice"] == {"type": "none"}
    # The whole report is long: the final call streams with a larger output budget.
    assert client.requests[-1]["_streamed"] and client.requests[-1]["max_tokens"] > 16000
    assert not any(r.get("_streamed") for r in client.requests[:-1])
    assert len({json.dumps(r["tools"]) for r in client.requests}) == 1  # cached prefix holds
    assert len(client.requests[0]["tools"]) == 20


def test_no_checker_config_goes_straight_from_draft_to_editor(con):
    build_db(con, [{}, {**RUN}])
    out = build_graph(
        client=FakeClient(_scripts()), con=con, teams=TEAMS, config="no_checker"
    ).invoke({"request": Request(team="DEN", opponent="KC", season=2026, week=2)})
    assert "checks_run" not in out or out["checks_run"] == 0
    assert out["draft"]["offense_analyst"].claims[0].claim_id == "oc1"
    assert out["report"].title == "KC scouting report"


def test_a_claim_without_a_source_is_set_aside_not_fatal():
    # Seen in 8/120 single-agent eval runs: the run crashed instead of being scored.
    from nfl_scout.numbers import check_numbers

    raw = {
        "prose": "KC passes 61.2% [c1] and ranks 3rd [c2].",
        "claims": [
            {"claim_id": "c1", "text": "KC pass rate", "stated_value": 61.2,
             "tool_call_id": "o1", "derived_from": None, "op": None, "measure": "value"},
            {"claim_id": "c2", "text": "KC rank", "stated_value": 3,
             "tool_call_id": None, "derived_from": None, "op": None, "measure": "value"},
        ],
    }  # fmt: skip
    section = parse_section("offense_analyst", raw)
    assert [c.claim_id for c in section.claims] == ["oc1"]
    assert [u["claim_id"] for u in section.unsourced] == ["oc2"]
    statuses = [c.status for c in check_numbers(section.prose, section.claims)]
    assert statuses == ["ok", "unknown_claim"]  # the number counts as unsupported


def test_single_agent_report_survives_an_unsourced_claim():
    from nfl_scout.graph.state import parse_single

    raw = {
        "title": "t",
        "summary": "KC passes 61.2% [c1].",
        "sections": [],
        "claims": [
            {"claim_id": "c1", "text": "KC pass rate", "stated_value": 61.2,
             "tool_call_id": None, "derived_from": None, "op": None, "measure": "value"},
        ],
    }  # fmt: skip
    section, report = parse_single(raw)
    assert section.claims == [] and section.unsourced[0]["claim_id"] == "sc1"
    assert "[sc1]" in report.summary
