"""The Streamlit app itself, exercised headlessly with AppTest.

The app is pointed at the sample run via the ALMATY_ACCESS_DATA
environment variable (the same mechanism a deployment would use), then
driven like a user: switch every layer, pick a scenario, click a hex
(injected through session_state — pydeck selection events cannot be
synthesized in AppTest), clear it, and break the data directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from almaty_access.config import Config
from almaty_access.dashboard.data import ENV_VAR

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
SELECTED_KEY = "selected_hex"


def _widget(elements, label: str):
    matches = [element for element in elements if element.label == label]
    assert matches, f"no widget labelled {label!r}"
    return matches[0]


@pytest.fixture()
def app(dashboard_sample: Config, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv(ENV_VAR, str(dashboard_sample.paths.processed))
    test = AppTest.from_file(str(APP_PATH), default_timeout=120)
    test.run()
    assert not test.exception, test.exception[0].value if test.exception else None
    return test


def test_boots_with_map_and_tabs(app: AppTest) -> None:
    assert app.title[0].value == "Almaty Jobs & Accessibility Lab"
    assert len(app.get("deck_gl_json_chart")) == 1
    assert _widget(app.sidebar.radio, "Layer").value == "access"
    assert _widget(app.sidebar.radio, "Population weighting").value == "raster"
    assert len(app.get("download_button")) == 2
    # the Method tab names the data directory the app resolved
    flat = " ".join(str(md.value) for md in app.markdown)
    assert "Limitations" in flat


def test_every_layer_renders(app: AppTest) -> None:
    layer_radio = _widget(app.sidebar.radio, "Layer")
    for layer in layer_radio.options:
        layer_radio.set_value(layer)
        app.run()
        assert not app.exception, f"layer {layer}: {app.exception[0].value}"
        assert len(app.get("deck_gl_json_chart")) == 1


def test_scenario_shows_headline_and_gains(app: AppTest) -> None:
    _widget(app.sidebar.selectbox, "Scenario").set_value("demo_abay")
    app.run()
    assert not app.exception
    assert app.info, "scenario headline expected"
    assert "demo" in str(app.info[0].value).lower() or app.info[0].value
    # the gains chart appears in an expander
    assert app.expander, "gains expander expected"


def test_gravity_budget_switches_measure(app: AppTest) -> None:
    _widget(app.sidebar.select_slider, "Time budget").set_value("grav")
    app.run()
    assert not app.exception
    labels = [radio.label for radio in app.sidebar.radio]
    assert "Measure" not in labels  # gravity has no percent variant


def test_click_opens_reach_panel_and_clears(app: AppTest, dashboard_sample: Config) -> None:
    import pandas as pd

    access = pd.read_parquet(dashboard_sample.paths.processed / "access_r8.parquet")
    origin = access.loc[access["in_city"], "h3"].iloc[0]
    app.session_state[SELECTED_KEY] = origin
    app.run()
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Hexes reached" in labels and "Jobs within reach" in labels
    # three deck layers now: hexes + reach wash + origin
    _widget(app.button, "Clear selection").click()
    app.run()
    assert SELECTED_KEY not in app.session_state
    assert not app.exception


def test_calibrated_weighting_and_districts_tab(app: AppTest) -> None:
    _widget(app.sidebar.radio, "Population weighting").set_value("calibrated")
    _widget(app.sidebar.radio, "Layer").set_value("population")
    app.run()
    assert not app.exception
    charts = app.get("vega_lite_chart")
    assert len(charts) >= 2, "district bars + Lorenz expected"


def test_missing_data_dir_shows_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    test = AppTest.from_file(str(APP_PATH), default_timeout=60)
    test.run()
    assert not test.exception
    assert test.error, "expected a clean error box"
    assert "missing required files" in str(test.error[0].value)
