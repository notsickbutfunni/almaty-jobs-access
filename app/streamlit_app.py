"""Almaty Jobs & Accessibility Lab — the interactive dashboard (Phase 7).

Run locally with::

    streamlit run app/streamlit_app.py

A thin UI over ``almaty_access.dashboard``: every data decision (which
directory to read, layer tables, colors, reach sets, chart specs) lives
in the package and is unit-tested; this file only lays out widgets.
Data resolution order: $ALMATY_ACCESS_DATA, the committed data/app/
bundle, data/processed/, the sample (see dashboard/data.py).
"""

from __future__ import annotations

import math
from pathlib import Path

import pydeck as pdk
import streamlit as st

from almaty_access.dashboard import charts, downloads, layers, palette
from almaty_access.dashboard.data import Tables, load_tables, resolve_data_dir
from almaty_access.dashboard.layers import GRAVITY, LayerState
from almaty_access.dashboard.reach import reach_summary, reachable

ROOT = Path(__file__).resolve().parents[1]
MAP_KEY = "map"
SELECTED_KEY = "selected_hex"
BASELINE_LABEL = "Baseline (no scenario)"
GRAVITY_LABEL = "gravity-weighted (no cutoff)"

st.set_page_config(
    page_title="Almaty Jobs & Accessibility",
    page_icon="🧭",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading tables…")
def cached_tables(data_dir: str) -> Tables:
    """One shared read-only Tables object per data directory."""
    return load_tables(Path(data_dir))


def picked_hex_from_map_state() -> str | None:
    """The hex id of the last map click, if any (parsed BEFORE the map
    is rebuilt, so the highlight appears in the same rerun)."""
    state = st.session_state.get(MAP_KEY)
    selection = getattr(state, "selection", None)
    objects = getattr(selection, "objects", None) or {}
    picked = objects.get("hexes") or []
    if picked and isinstance(picked[0], dict) and picked[0].get("h3"):
        return str(picked[0]["h3"])
    return None


def legend_html(layer: layers.MapLayer) -> str:
    """Categorical chips, or the ramp gradient with its bounds."""
    if layer.legend:
        chips = "".join(
            f'<span style="display:inline-flex;align-items:center;margin-right:14px;">'
            f'<span style="width:12px;height:12px;border-radius:3px;background:{color};'
            f'border:1px solid {palette.AXIS};display:inline-block;margin-right:5px;"></span>'
            f'<span style="color:{palette.INK_2};font-size:13px;">{label}</span></span>'
            for label, color in layer.legend
        )
        return f"<div>{chips}</div>"
    if layer.ramp and layer.ramp_range is not None:
        low, high = layer.ramp_range
        if not (math.isfinite(low) and math.isfinite(high)):
            return ""  # nothing painted (e.g. a scenario with no gaining hex)
        stops = ", ".join(reversed(layer.ramp) if layer.ramp_reversed else layer.ramp)
        return (
            f'<div style="max-width:380px;">'
            f'<div style="height:10px;border-radius:5px;'
            f'background:linear-gradient(to right, {stops});"></div>'
            f'<div style="display:flex;justify-content:space-between;'
            f'color:{palette.MUTED};font-size:12px;">'
            f"<span>{low:,.1f}</span><span>{high:,.1f}</span></div></div>"
        )
    return ""


def sidebar_state(tables: Tables, catalog: layers.Catalog) -> LayerState:
    """All sidebar widgets; returns the (always-valid) layer selection."""
    st.sidebar.title("Almaty access")
    layer_options = [
        name for name in layers.LAYERS if name != "scenario_delta" or catalog.scenarios
    ]
    layer = st.sidebar.radio(
        "Layer",
        layer_options,
        format_func=lambda key: layers.LAYER_TITLES[key],
    )

    scenario: str | None = None
    if catalog.scenarios and layer in ("access", "scenario_delta"):
        chosen = st.sidebar.selectbox(
            "Scenario",
            [BASELINE_LABEL, *catalog.scenarios],
            index=1 if layer == "scenario_delta" else 0,
            format_func=lambda name: (
                name if name == BASELINE_LABEL else tables.scenario_title(name).split(":")[0]
            ),
        )
        if chosen != BASELINE_LABEL:
            scenario = chosen
        elif layer == "scenario_delta":
            scenario = catalog.scenarios[0]
            st.sidebar.caption(f"Showing {tables.scenario_title(scenario).split(':')[0]}.")

    opportunity = "jobs"
    if layer == "access":
        opportunity = st.sidebar.selectbox(
            "Opportunity",
            catalog.opportunities,
            format_func=str.capitalize,
        )

    if layer in ("lisa", "deserts"):
        mode = st.sidebar.selectbox("Mode", catalog.lisa_modes)
    elif layer in ("access", "scenario_delta") and opportunity == "jobs":
        modes = catalog.modes
        if layer == "scenario_delta" and scenario is not None:
            # Deltas only exist for the modes the scenario changes.
            scenario_modes = tables.scenario_modes(scenario)
            modes = tuple(m for m in catalog.modes if m in scenario_modes) or catalog.modes
        default = modes.index("transit") if "transit" in modes else 0
        mode = st.sidebar.selectbox("Mode", modes, index=default)
    else:
        mode = "walk"
        if layer == "access":
            st.sidebar.caption("Facility counts exist for walking only (Phase 4).")

    threshold: int | str = 45
    if layer in ("access", "scenario_delta", "gap"):
        if layer == "gap":
            options: list[int | str] = list(catalog.gap_thresholds)
        elif opportunity == "jobs":
            options = [*catalog.thresholds, GRAVITY]
        else:
            options = list(catalog.thresholds)
        threshold = st.sidebar.select_slider(
            "Time budget",
            options=options,
            value=45 if 45 in options else options[0],
            format_func=lambda t: GRAVITY_LABEL if t == GRAVITY else f"{t} min",
        )

    metric = "accpct"
    if layer in ("access", "scenario_delta") and opportunity == "jobs":
        if threshold == GRAVITY:
            metric = "acc"
        else:
            labels = {"accpct": "% of city jobs", "acc": "number of jobs"}
            metric = st.sidebar.radio(
                "Measure", list(labels), format_func=labels.get, horizontal=True
            )
    elif opportunity != "jobs":
        metric = "acc"

    weighting = st.sidebar.radio(
        "Population weighting",
        catalog.weightings,
        format_func=str.capitalize,
        horizontal=True,
        help=(
            "Raster = Kontur population as-is; calibrated = rescaled to "
            "official district shares of employed residents (D-058). The two "
            "can flip equity verdicts."
        ),
    )

    uncertainty = False
    if layer == "jobs":
        uncertainty = st.sidebar.toggle(
            "Show model uncertainty (P90 - P10)",
            help="Spread of the bootstrapped random-forest job estimate.",
        )

    return LayerState(
        layer=layer,
        mode=mode,
        threshold=threshold,
        opportunity=opportunity,
        metric=metric,
        weighting=weighting,
        scenario=scenario,
        uncertainty=uncertainty,
    )


def deck_for(map_layer: layers.MapLayer, tables: Tables, state: LayerState) -> pdk.Deck:
    """The pydeck Deck: the colored layer plus the click-to-reach overlay."""
    columns = [
        "h3",
        "fill",
        "display",
        "district_label",
        "population_label",
        "jobs_label",
        "modes_label",
    ]
    deck_layers = [
        pdk.Layer(
            "H3HexagonLayer",
            id="hexes",
            data=map_layer.table[columns].to_dict("records"),
            get_hexagon="h3",
            get_fill_color="fill",
            stroked=False,
            extruded=False,
            pickable=True,
        )
    ]
    origin = st.session_state.get(SELECTED_KEY)
    if origin:
        reach_mode = state.mode if state.mode in tables.tt else "transit"
        budget = state.threshold if isinstance(state.threshold, int) else 45
        reached = reachable(tables.tt[reach_mode], origin, budget)
        wash = palette.hex_to_rgb(palette.AQUA)
        deck_layers.append(
            pdk.Layer(
                "H3HexagonLayer",
                id="reach",
                data=[{"h3": value} for value in reached["h3"]],
                get_hexagon="h3",
                get_fill_color=[*wash, 90],
                stroked=False,
                extruded=False,
                pickable=False,
            )
        )
        deck_layers.append(
            pdk.Layer(
                "H3HexagonLayer",
                id="origin",
                data=[{"h3": origin}],
                get_hexagon="h3",
                get_fill_color=[*palette.hex_to_rgb(palette.ORANGE), 235],
                stroked=False,
                extruded=False,
                pickable=False,
            )
        )
    tooltip_html = (
        "<b>{district_label}</b><br/>"
        "residents: {population_label} · estimated jobs: {jobs_label}<br/>"
        f"{map_layer.value_name}: {{display}}<br/>"
        "<span style='font-size:11px'>% of jobs in 45 min — {modes_label}</span>"
    )
    return pdk.Deck(
        layers=deck_layers,
        initial_view_state=pdk.ViewState(latitude=43.238, longitude=76.89, zoom=9.6),
        map_provider="carto",
        map_style="light",
        tooltip={"html": tooltip_html},
    )


def reach_panel(tables: Tables, state: LayerState, origin: str) -> None:
    """The small isochrone summary next to the map."""
    reach_mode = state.mode if state.mode in tables.tt else "transit"
    budget = state.threshold if isinstance(state.threshold, int) else 45
    try:
        summary = reach_summary(tables, origin, reach_mode, budget)
    except ValueError:
        st.session_state.pop(SELECTED_KEY, None)
        return
    st.markdown(f"**Reachable from the selected hex** · {reach_mode}, {budget} min")
    one, two = st.columns(2)
    one.metric("Hexes reached", f"{summary.n_reachable:,}")
    two.metric("Jobs within reach", f"{summary.jobs_sum:,.0f}")
    if state.scenario:
        st.caption(
            "The highlight shows the BASELINE network — scenario travel "
            "times are not stored per pair (Phase 6 keeps deltas only)."
        )
    if not summary.facilities_walk.empty:
        st.caption(f"Facilities within {budget} min walk of this hex:")
        st.dataframe(summary.facilities_walk, hide_index=True, height=200)
    if not summary.nearest_walk.empty:
        st.caption("Nearest facility (walking minutes):")
        st.dataframe(summary.nearest_walk.sort_values("minutes"), hide_index=True, height=240)
    if st.button("Clear selection"):
        st.session_state.pop(SELECTED_KEY, None)
        st.rerun()


def map_tab(tables: Tables, state: LayerState) -> None:
    map_layer = layers.build_layer(tables, state)
    st.subheader(map_layer.title)
    if state.scenario:
        entry = tables.scenarios_report["scenarios"][state.scenario]
        st.info(f"**{entry['title']}** — {entry['headline']}")
    picked = picked_hex_from_map_state()
    if picked:
        st.session_state[SELECTED_KEY] = picked
    show_panel = st.session_state.get(SELECTED_KEY) is not None
    map_col, side_col = st.columns([7, 3]) if show_panel else (st.container(), None)
    with map_col:
        st.pydeck_chart(
            deck_for(map_layer, tables, state),
            key=MAP_KEY,
            on_select="rerun",
            selection_mode="single-object",
            height=560,
        )
        st.markdown(legend_html(map_layer), unsafe_allow_html=True)
        st.caption(map_layer.caption + " Click a hex to see what it reaches.")
    if side_col is not None:
        with side_col:
            reach_panel(tables, state, st.session_state[SELECTED_KEY])
    if state.scenario:
        with st.expander("Who gains from this scenario"):
            st.altair_chart(
                charts.gains_bars(tables.gains, state.scenario),
                width="stretch",
            )
            entry = tables.scenarios_report["scenarios"][state.scenario]
            st.caption(entry["note"])
    numeric = map_layer.table[["h3", "district", "population", "jobs_rf", "value"]].rename(
        columns={"value": map_layer.value_name}
    )
    one, two = st.columns(2)
    one.download_button(
        "Download this layer (CSV)",
        data=downloads.csv_bytes(numeric),
        file_name=f"almaty_{state.layer}.csv",
        mime="text/csv",
    )
    two.download_button(
        "Download all access measures (Parquet)",
        data=downloads.parquet_bytes(tables.access),
        file_name="access_r8.parquet",
        mime="application/octet-stream",
    )


def districts_tab(tables: Tables, state: LayerState) -> None:
    measures = list(dict.fromkeys(tables.districts["measure"]))
    measure = st.selectbox("Measure", measures, format_func=charts.measure_label)
    left, right = st.columns([5, 4])
    with left:
        st.altair_chart(
            charts.district_bars(tables.districts, measure, state.weighting),
            width="stretch",
        )
    with right:
        st.altair_chart(
            charts.lorenz_curve(tables.lorenz, tables.summary, measure, state.weighting),
            width="content",
        )
    st.caption(
        "Inhabited city hexes, weighted by residents under the "
        f"{state.weighting} population weighting. The Lorenz curve sorts "
        "residents from worst to best access; the further it sags below the "
        "dashed equality line, the more unequal the access (Gini 0 = equal)."
    )


def method_tab(tables: Tables, data_dir: Path) -> None:
    headline = tables.equity_report.get("headline", {})
    if headline.get("sentence"):
        st.markdown(f"**The one-sentence result.** {headline['sentence']}")
    st.markdown(
        """
#### What you are looking at
Jobs are estimated per H3 hexagon (resolution 8, ~0.7 km²) by
disaggregating official district payroll-job totals with a
building-floor-area baseline and a random forest; district totals are
conserved exactly. Travel times come from open street and transit data;
access is counted four ways:

- **Cumulative**: jobs (or facilities) reachable within 15/30/45/60 minutes.
- **Gravity**: every job counts, discounted by `exp(-beta*t)` with a
  20-minute half-life.
- **2SFCA**: facilities per 1,000 residents with both supply and demand
  within a walking catchment.
- **Gini / Lorenz**: how equally access is spread over residents.

#### Data sources (all open)
OpenStreetMap via Geofabrik (streets, transit route relations, POIs,
buildings) · Kontur population raster · Overture buildings ·
Copernicus GLO-30 elevation · stat.gov.kz Taldau (district payroll
jobs, employed residents).
"""
    )
    st.markdown("#### Limitations")
    model_quality = ""
    rf_metrics = tables.model_report.get("lodo_metrics_rf") or {}
    if isinstance(rf_metrics, dict) and rf_metrics.get("mape") is not None:
        model_quality = f" Leave-one-district-out MAPE of the job model: {rf_metrics['mape']:.0%}."
    st.markdown(
        "- **Jobs are modelled, not counted.** Hex values are estimates; only "
        "district totals are official." + model_quality
    )
    for caveat in tables.equity_report.get("caveats", []):
        st.markdown(f"- {caveat}")
    for caveat in tables.scenarios_report.get("caveats", []):
        st.markdown(f"- {caveat}")
    repo = "https://github.com/notsickbutfunni/almaty-jobs-access"
    st.markdown(
        f"#### Reproduce it\nCode, decision log and learning guide: [{repo}]({repo}). "
        f"This app is reading `{data_dir}`."
    )


def main() -> None:
    try:
        data_dir = resolve_data_dir(ROOT)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
        return
    tables = cached_tables(str(data_dir))
    catalog = layers.build_catalog(tables)
    state = sidebar_state(tables, catalog)
    st.sidebar.caption(
        "Transit is approximate and a lower bound (OSM routes, assumed "
        "headways). Car is free-flow, an upper bound."
    )

    st.title("Almaty Jobs & Accessibility Lab")
    st.caption(
        "Where the jobs are and who can reach them — open data on an H3 grid, "
        "four travel modes, two population weightings, three what-if scenarios."
    )
    tab_map, tab_districts, tab_method = st.tabs(["Map", "Districts", "Method & data"])
    with tab_map:
        map_tab(tables, state)
    with tab_districts:
        districts_tab(tables, state)
    with tab_method:
        method_tab(tables, data_dir)


main()
