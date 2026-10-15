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
