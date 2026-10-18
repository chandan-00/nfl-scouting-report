"""Graph plumbing with a scripted fake client: no API calls."""

import json
import threading
from types import SimpleNamespace as NS

import pytest
from conftest import KC_DEF_ROW, RUN, build_db

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


def _scripts():
    plan = {"offense_focus": ["How often do they pass?"], "defense_focus": ["Do they blitz?"]}
    offense = [
        tool_response(("pass_rate", {"team": "KC"})),
        text_response(
            {
                "prose": "KC drops back on 50% of plays [c1].",
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "KC dropback rate",
                        "stated_value": 50.0,
                        "tool_call_id": "o1",
                        "derived_from": None,
                        "op": None,
                    }
                ],
            }
        ),
    ]
    defense = [text_response({"prose": "Not enough data.", "claims": []})]
    return {
        "planner": [text_response(plan)],
        "offense_analyst": offense,
        "defense_analyst": defense,
    }


def test_graph_runs_planner_then_both_analysts(con):
    build_db(con, [{}, {**RUN}, {**KC_DEF_ROW}])
    client = FakeClient(_scripts())
    graph = build_graph(client=client, con=con, teams=TEAMS)
    out = graph.invoke({"request": Request(team="DEN", opponent="KC", season=2026, week=2)})

    assert out["plan"]["offense_focus"] == ["How often do they pass?"]
    offense = out["sections"]["offense_analyst"]
    assert offense.claims[0].claim_id == "oc1"
    assert offense.claims[0].tool_call_id == "o1"
    assert out["sections"]["defense_analyst"].claims == []
    # The tool value lives in the trace, not in the section.
    (call,) = out["tool_calls"]
    assert (call.call_id, call.args["through_week"], call.result["value"]) == ("o1", 1, 50.0)
    assert {u.node for u in out["usage"]} == {"planner", "offense_analyst", "defense_analyst"}


def test_each_analyst_keeps_a_stable_cached_prefix(con):
    build_db(con, [{}, {**RUN}])
    client = FakeClient(_scripts())
    build_graph(client=client, con=con, teams=TEAMS).invoke(
        {"request": Request(team="DEN", opponent="KC", season=2026, week=2)}
    )
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


def test_through_week_is_week_minus_one():
    assert Request(team="DEN", opponent="KC", season=2026, week=6).through_week == 5
