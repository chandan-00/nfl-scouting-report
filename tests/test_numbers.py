"""Number extractor: every number in report prose, linked to the claim marker that follows it."""

from nfl_scout.graph.state import Claim
from nfl_scout.numbers import check_numbers, extract_numbers


def values(text):
    return [(m.text, m.value, m.kind) for m in extract_numbers(text)]


def test_extracts_percents_decimals_and_negatives():
    assert values("Drops back on 61.2% of plays [oc1] at -0.14 EPA [oc2] and 6.5 men [oc3].") == [
        ("61.2%", 61.2, "percent"),
        ("-0.14", -0.14, "number"),
        ("6.5", 6.5, "number"),
    ]


def test_extracts_rankings_and_ratios():
    text = "SEA is 3rd in the league [oc1], first in the NFL [oc2], and runs twice as often [oc3]."
    assert values(text) == [
        ("3rd", 3, "ordinal"),
        ("first", 1, "ordinal"),
        ("twice", 2, "ratio"),
    ]
    assert values("Blitzes three times as often [dc1], 1.5x the rate [dc2].") == [
        ("three times", 3, "ratio"),
        ("1.5x", 1.5, "ratio"),
    ]


def test_ignores_downs_weeks_years_thresholds_and_markers():
    text = (
        "On 3rd-and-long and 1st and 2nd down, in weeks 1-5 of 2026 and week 4, "
        "third down, 5+ rushers, 20+ yard plays, the first half [oc12]."
    )
    assert values(text) == []


def test_ranges_are_not_negative_numbers():
    assert values("Gains 6-7 yards [oc1].") == [("6", 6, "number"), ("7", 7, "number")]


def test_links_each_number_to_the_next_marker_in_its_sentence():
    text = (
        "Opponents pass with 40.4% success [dc6] against 34.5% on runs [dc5]. "
        "It is only 11 drives. Box count is 6.5 [dc9]."
    )
    links = [(m.text, m.claim_id) for m in extract_numbers(text)]
    assert links == [("40.4%", "dc6"), ("34.5%", "dc5"), ("11", None), ("6.5", "dc9")]


def test_multi_marker_groups_pair_from_the_end():
    text = "Runs right 41.7% and left 30.3% [oc19, oc20]; total 72.0 and 9 [oc21]."
    links = [(m.text, m.claim_id) for m in extract_numbers(text)]
    assert links == [("41.7%", "oc19"), ("30.3%", "oc20"), ("72.0", None), ("9", "oc21")]


def _claim(cid, value, measure="value"):
    return Claim(claim_id=cid, text="t", stated_value=value, tool_call_id="o1", measure=measure)


def test_check_flags_unlinked_unknown_and_changed_numbers():
    text = (
        "Passes 61.2% [oc1], runs 38.0% [oc2], EPA 0.10 [oc9] on 27 plays. "
        "PROE is 5.1 points below expected [oc3]."
    )
    claims = [_claim("oc1", 61.2), _claim("oc2", 38.8), _claim("oc3", -5.1)]
    statuses = [(r.mention.text, r.status) for r in check_numbers(text, claims)]
    assert statuses == [
        ("61.2%", "ok"),
        ("38.0%", "value_mismatch"),  # the editor changed the number
        ("0.10", "unknown_claim"),
        ("27", "unlinked"),
        ("5.1", "ok"),  # sign carried by the words ("below")
    ]


def test_check_matches_at_the_precision_written():
    claims = [_claim("oc1", 85.18518), _claim("oc2", 3)]
    results = check_numbers("Passes 85.2% [oc1], 3rd in the league [oc2].", claims)
    assert [r.status for r in results] == ["ok", "ok"]


def test_adjacent_marker_groups_merge_and_plus_wording_is_a_threshold():
    text = "It allows -0.09 EPA on 183 dropbacks [dc1][dc2], and 4.9% go 20-plus yards [dc4]."
    links = [(m.text, m.claim_id) for m in extract_numbers(text)]
    assert links == [("-0.09", "dc1"), ("183", "dc2"), ("4.9%", "dc4")]


def test_check_pairs_one_marker_with_the_number_it_states():
    # One marker after two numbers: the marker belongs to 80.7, so 57 is the unsupported one.
    results = check_numbers("Drops back on 80.7% of 57 snaps [oc4].", [_claim("oc4", 80.7)])
    assert [(r.mention.text, r.mention.claim_id, r.status) for r in results] == [
        ("80.7%", "oc4", "ok"),
        ("57", None, "unlinked"),
    ]
