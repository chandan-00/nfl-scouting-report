"""Deterministic metric tools over DuckDB. Every tool takes team, season, through_week."""

from nfl_scout.tools.base import TOOLS, ToolResult, ToolSpec
from nfl_scout.tools.defense import average_box_defenders, average_pass_rushers, blitz_rate
from nfl_scout.tools.efficiency import (
    epa_per_play,
    explosive_play_rate,
    pressure_proxy_rate,
    red_zone_td_rate,
    sack_rate,
    success_rate,
    third_down_conversion_rate,
)
from nfl_scout.tools.league import league_rank
from nfl_scout.tools.offense import (
    average_depth_of_target,
    ftn_play_rate,
    pass_rate,
    pass_rate_over_expected,
    player_share,
    run_direction_rate,
    usage_leaders,
)
from nfl_scout.tools.roster import injury_report, snap_share

__all__ = [
    "TOOLS",
    "ToolResult",
    "ToolSpec",
    "average_box_defenders",
    "average_depth_of_target",
    "average_pass_rushers",
    "blitz_rate",
    "epa_per_play",
    "explosive_play_rate",
    "ftn_play_rate",
    "injury_report",
    "league_rank",
    "pass_rate",
    "pass_rate_over_expected",
    "player_share",
    "pressure_proxy_rate",
    "red_zone_td_rate",
    "run_direction_rate",
    "sack_rate",
    "snap_share",
    "success_rate",
    "third_down_conversion_rate",
    "usage_leaders",
]
