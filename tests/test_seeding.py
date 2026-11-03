"""Seeded errors: each kind is injected into clean drafts, and the checker's catch rate and
false positives are measured against the manifest."""

import random

import pytest
from conftest import KC_DEF_ROW, RUN, build_db

from nfl_scout.evaluation.seeding import KINDS, score_seeded, seed_errors
from nfl_scout.graph.llm_tools import ToolExecutor
from nfl_scout.graph.state import Claim, Section


@pytest.fixture
def clean(con):
    build_db(con, [{"epa": 0.6}, {"epa": 0.2}, {**RUN, "epa": -0.2}, {**KC_DEF_ROW, "epa": 0.1}])
    ex = ToolExecutor(season=2026, through_week=1, prefix="o", con=con)
    ex.run("pass_rate", {"team": "KC"})
    ex.run("epa_per_play", {"team": "KC", "play_kind": "pass"})
    ex.run("epa_per_play", {"team": "DEN", "side": "defense"})
    claims = [
        Claim(claim_id="oc1", text="KC offense pass rate", stated_value=66.7, tool_call_id="o1"),
        Claim(claim_id="oc2", text="KC offense EPA per pass play", stated_value=0.4,
              tool_call_id="o2"),
        Claim(claim_id="oc3", text="DEN defense EPA allowed", stated_value=0.1, tool_call_id="o3"),
        Claim(claim_id="oc4", text="KC offense plays in the sample", stated_value=3,
              tool_call_id="o1", measure="n_plays"),
    ]  # fmt: skip
    prose = (
        "KC drops back on 66.7% of plays [oc1] over 3 plays [oc4]. Its passes average 0.40 EPA "
        "[oc2]. DEN's defense allows 0.10 EPA per play [oc3]."
    )
    sections = {"offense_analyst": Section(role="offense_analyst", prose=prose, claims=claims)}
    return sections, ex.calls, con


def test_seeding_is_deterministic(clean):
    sections, calls, _ = clean
    a = seed_errors(sections, calls, random.Random(1))
    b = seed_errors(sections, calls, random.Random(1))
    assert a[1] == b[1]
    assert a[0]["offense_analyst"].prose == b[0]["offense_analyst"].prose


def test_every_kind_is_seeded_once_on_a_different_claim(clean):
    sections, calls, _ = clean
    _, manifest = seed_errors(sections, calls, random.Random(1))
    assert sorted(m["kind"] for m in manifest) == sorted(KINDS)
    ids = [m["claim_id"] for m in manifest if m["claim_id"]]
    assert len(ids) == len(set(ids))


def test_wrong_value_changes_claim_and_prose_together(clean):
    sections, calls, _ = clean
    seeded, manifest = seed_errors(sections, calls, random.Random(1), kinds=["wrong_value"])
    (m,) = manifest
    claim = next(c for c in seeded["offense_analyst"].claims if c.claim_id == m["claim_id"])
    assert claim.stated_value == m["seeded"] != m["original"]
    assert m["seeded_text"] in seeded["offense_analyst"].prose


def test_checker_catches_every_seeded_kind_with_no_false_positives(clean):
    sections, calls, con = clean
    seeded, manifest = seed_errors(sections, calls, random.Random(3))
    score = score_seeded(sections, seeded, manifest, calls, con)
    assert {k: v["caught"] for k, v in score["by_kind"].items()} == {k: 1 for k in KINDS}
    assert score["false_positives"] == []


def test_a_missed_error_is_reported_as_missed(clean, monkeypatch):
    sections, calls, con = clean
    seeded, manifest = seed_errors(sections, calls, random.Random(3), kinds=["wrong_team"])
    import nfl_scout.evaluation.seeding as seeding

    monkeypatch.setattr(seeding, "check_sections", lambda *a, **k: [])
    score = score_seeded(sections, seeded, manifest, calls, con)
    assert score["by_kind"]["wrong_team"] == {"seeded": 1, "caught": 0}


def test_wrong_team_swaps_names_in_the_same_form(clean):
    # Claims often say "Chiefs" or "Kansas City" rather than "KC"; swap like for like.
    sections, calls, con = clean
    renamed = {
        "offense_analyst": sections["offense_analyst"].model_copy(
            update={
                "claims": [
                    c.model_copy(update={"text": c.text.replace("KC", "Kansas City")})
                    for c in sections["offense_analyst"].claims
                    if c.claim_id in ("oc1", "oc2")
                ]
            }
        )
    }
    seeded, manifest = seed_errors(renamed, calls, random.Random(0), kinds=["wrong_team"], con=con)
    (m,) = manifest
    assert "Denver" in m["seeded"] and "Kansas City" not in m["seeded"]


@pytest.mark.parametrize("v", [-0.025, 0.025, 61.2, -5.1, 3.0, 0.0])
def test_wrong_value_changes_magnitude_beyond_tolerance_and_keeps_sign(v):
    # The checker compares magnitudes (prose carries sign in words), so a seeded sign flip near
    # zero (-0.025 -> 0.03) would be an uncatchable error, not a checker miss.
    from nfl_scout.evaluation.seeding import perturb

    for seed in range(50):
        new = perturb(v, decimals=2 if abs(v) < 1 else 1, rng=random.Random(seed))
        assert abs(abs(new) - abs(v)) > 0.5 * 10 ** -(2 if abs(v) < 1 else 1)
        assert new * v >= 0


def test_wrong_value_is_wrong_at_the_precision_the_checker_reads():
    # -0.025 -> -0.030 is stored as -0.03, which the checker reads at 2 decimals: within
    # tolerance of -0.0252. The seeded value must fail the checker's own comparison.
    from nfl_scout.checker import _close
    from nfl_scout.evaluation.seeding import perturb

    for seed in range(200):
        new = perturb(-0.025, decimals=3, rng=random.Random(seed))
        assert not _close(new, -0.02519)
