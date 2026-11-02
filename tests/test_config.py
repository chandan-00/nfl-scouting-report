import pytest

from nfl_scout.config import MODEL_OVERRIDE_ENV, load_settings


@pytest.fixture
def settings():
    return load_settings()


def test_every_node_defaults_to_haiku(settings, monkeypatch):
    monkeypatch.delenv(MODEL_OVERRIDE_ENV, raising=False)
    for node in settings.models.nodes:
        assert settings.node_model(node).model == "claude-haiku-5-5"


def test_override_switches_model_but_keeps_effort(settings, monkeypatch):
    monkeypatch.setenv(MODEL_OVERRIDE_ENV, "opus")
    node = settings.node_model("matchup_analyst")
    assert node.model == "claude-opus-5-5"
    assert node.effort == settings.models.nodes["matchup_analyst"].effort


def test_unknown_override_is_rejected(settings, monkeypatch):
    monkeypatch.setenv(MODEL_OVERRIDE_ENV, "gpt")
    with pytest.raises(ValueError):
        settings.node_model("planner")


def test_unlisted_node_uses_default(settings, monkeypatch):
    monkeypatch.delenv(MODEL_OVERRIDE_ENV, raising=False)
    node = settings.node_model("not_a_node")
    assert node.model == settings.models.aliases[settings.models.default]
    assert node.effort == settings.models.default_effort


def test_eval_season_is_loaded(settings):
    assert settings.data.eval_season in settings.data.seasons


def test_usage_cost_uses_each_token_type(settings):
    from nfl_scout.costs import usage_cost
    from nfl_scout.graph.state import Usage

    u = Usage(
        node="planner",
        model="claude-haiku-5-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        latency_s=1.0,
    )
    # Haiku 5.5: $0.10 in, $0.50 out, $0.01 cache read, $0.125 cache write (5 min), per MTok.
    assert usage_cost(u, settings) == pytest.approx(0.10 + 0.50 + 0.01 + 0.125)


def test_every_model_alias_has_a_price(settings):
    assert set(settings.models.prices) == set(settings.models.aliases)
