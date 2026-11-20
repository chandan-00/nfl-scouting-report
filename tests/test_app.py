"""The demo app, run headlessly."""

import sys
import types
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_survives_a_stale_helper_module(monkeypatch):
    """Streamlit Cloud pulls new commits into a running process, where the old app_render is
    still imported. The app must load the current one, not the cached one."""
    import nfl_scout.app_render as current

    stale = types.ModuleType("nfl_scout.app_render")
    stale.__file__ = current.__file__
    stale.__spec__ = current.__spec__
    stale.annotate_report = current.annotate_report  # an older copy without cost_rows
    monkeypatch.setitem(sys.modules, "nfl_scout.app_render", stale)

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert list(at.dataframe[-1].value.columns) == ["node", "cost_usd"]
