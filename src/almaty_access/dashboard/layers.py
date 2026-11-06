"""Turn a UI selection into a colored hex table for the map.

Every layer the sidebar offers is one pure function of the loaded
``Tables`` and a frozen ``LayerState``; the Streamlit app only renders
what comes back. Values stay numeric in the table (downloads reuse it),
display strings are precomputed for the tooltip, and colors follow the
chart system in ``palette`` (magnitude = blue ramp, classes = fixed
validated hues).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from almaty_access.dashboard import palette
from almaty_access.dashboard.data import MODES, Tables

THRESHOLDS: tuple[int, ...] = (15, 30, 45, 60)
GRAVITY = "grav"

#: Facility opportunities exist for walking only (Phase 4 scope).
WALK_OPPORTUNITIES: tuple[str, ...] = (
    "schools",
    "kindergartens",
    "clinics",
    "grocery",
    "parks",
)

LAYERS: tuple[str, ...] = (
    "access",
    "jobs",
    "population",
    "gap",
    "lisa",
    "deserts",
    "scenario_delta",
)

LAYER_TITLES: dict[str, str] = {
    "access": "Accessibility",
    "jobs": "Estimated jobs",
    "population": "Residents",
    "gap": "Transit vs car gap",
    "lisa": "LISA clusters",
    "deserts": "Access deserts",
    "scenario_delta": "Scenario gain",
}

_LISA_RE = re.compile(r"^lisa_accpct_jobs_([a-z]+)_(\d+)$")
_GAP_RE = re.compile(r"^modegap_transit_car_(\d+)$")
_ACC_OPP_RE = re.compile(r"^acc_([a-z]+)_walk_15$")

DESERT_LABEL = "Access desert"
OTHER_LABEL = "Inhabited, not a desert"


@dataclass(frozen=True)
class LayerState:
    """One sidebar selection. Invalid combinations raise ``ValueError``."""

    layer: str = "access"
    mode: str = "transit"
    threshold: int | str = 45  # 15 / 30 / 45 / 60, or "grav"
    opportunity: str = "jobs"
    metric: str = "accpct"  # "acc" (count) | "accpct" (% of city jobs)
    weighting: str = "raster"
    scenario: str | None = None
    uncertainty: bool = False  # jobs layer: show the P90-P10 spread


@dataclass(frozen=True)
class MapLayer:
    """What the map needs: rows, colors, and words."""

    table: pd.DataFrame  # h3 + numeric value + display/label columns
    fill: np.ndarray  # uint8 (n, 4), aligned with table rows
    title: str
    caption: str
    value_name: str
    legend: tuple[tuple[str, str], ...] = ()  # categorical (label, hex)
    ramp: tuple[str, ...] = ()  # sequential hexes, light -> dark
    ramp_range: tuple[float, float] | None = None
    ramp_reversed: bool = False


@dataclass(frozen=True)
class Catalog:
    """What the loaded data actually supports (drives the sidebar)."""

    modes: tuple[str, ...]
    thresholds: tuple[int, ...]
    opportunities: tuple[str, ...]
    weightings: tuple[str, ...]
    scenarios: tuple[str, ...]
    lisa_modes: tuple[str, ...]
    gap_thresholds: tuple[int, ...]


def build_catalog(tables: Tables) -> Catalog:
    """Derive the sidebar's options from the columns that exist."""
    access_cols = set(tables.access.columns)
    thresholds = tuple(t for t in THRESHOLDS if f"acc_jobs_transit_{t}" in access_cols)
    opportunities = (
        "jobs",
        *(opp for opp in WALK_OPPORTUNITIES if f"acc_{opp}_walk_15" in access_cols),
    )
    raster = tables.equity["raster"]
    lisa_mode_set = {match.group(1) for col in raster.columns if (match := _LISA_RE.fullmatch(col))}
    lisa_modes = tuple(mode for mode in MODES if mode in lisa_mode_set)
    gap_thresholds = tuple(
        sorted(int(match.group(1)) for col in raster.columns if (match := _GAP_RE.fullmatch(col)))
    )
    return Catalog(
        modes=MODES,
        thresholds=thresholds,
        opportunities=opportunities,
        weightings=tuple(tables.equity),
        scenarios=tables.scenario_names,
        lisa_modes=lisa_modes,
        gap_thresholds=gap_thresholds,
    )


def lisa_column(equity: pd.DataFrame, mode: str) -> str:
    """The LISA column computed for ``mode`` (Phase 5 fixed one per mode)."""
    for col in equity.columns:
        match = _LISA_RE.fullmatch(col)
        if match and match.group(1) == mode:
            return col
    raise ValueError(f"no LISA column for mode '{mode}'")


def access_column(state: LayerState) -> str:
    """The access_r8 column a selection points at (also used for scenarios)."""
    if state.opportunity == "jobs":
        if state.threshold == GRAVITY:
            if state.metric != "acc":
                raise ValueError("gravity access has no percent variant; use metric='acc'")
            return f"acc_jobs_{state.mode}_{GRAVITY}"
        return f"{state.metric}_jobs_{state.mode}_{state.threshold}"
    if state.mode != "walk":
        raise ValueError(f"'{state.opportunity}' counts exist for walking only (Phase 4)")
    if state.threshold == GRAVITY or state.metric != "acc":
        raise ValueError(f"'{state.opportunity}' has plain counts only (no percent / gravity)")
    return f"acc_{state.opportunity}_walk_{state.threshold}"


def threshold_label(threshold: int | str) -> str:
    return "gravity-weighted" if threshold == GRAVITY else f"{threshold} min"


def _fmt(value: Any, spec: str, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return format(value, spec) + suffix


def _base_frame(tables: Tables, weighting: str) -> pd.DataFrame:
    """City hexes with the shared tooltip context columns."""
    base = tables.access.loc[tables.access["in_city"], ["h3", "district", "in_city"]].copy()
    population = tables.equity[weighting].set_index("h3")["population"]
    base["population"] = base["h3"].map(population)
    base["jobs_rf"] = base["h3"].map(tables.jobs.set_index("h3")["jobs_rf"])
    pct45 = {mode: tables.access.set_index("h3")[f"accpct_jobs_{mode}_45"] for mode in MODES}
    base["district_label"] = base["district"].fillna("outside districts").str.capitalize()
    base["population_label"] = [_fmt(v, ",.0f") for v in base["population"]]
    base["jobs_label"] = [_fmt(v, ",.0f") for v in base["jobs_rf"]]
    base["modes_label"] = [
        " · ".join(f"{mode} {_fmt(pct45[mode].get(h), '.1f', '%')}" for mode in MODES)
        for h in base["h3"]
    ]
    return base


def _sequential(
    base: pd.DataFrame,
    values: pd.Series,
    *,
    title: str,
    caption: str,
    value_name: str,
    spec: str,
    suffix: str = "",
    reverse: bool = False,
) -> MapLayer:
    table = base.copy()
    table["value"] = np.asarray(values, dtype=float)
    table = table[np.isfinite(table["value"])].reset_index(drop=True)
    fill, bounds = palette.sequential_rgba(table["value"], reverse=reverse)
    table["fill"] = [row.tolist() for row in fill]
    table["display"] = [_fmt(v, spec, suffix) for v in table["value"]]
    return MapLayer(
        table=table,
        fill=fill,
        title=title,
        caption=caption,
        value_name=value_name,
        ramp=palette.SEQ_BLUE,
        ramp_range=bounds,
        ramp_reversed=reverse,
    )


def _categorical(
    base: pd.DataFrame,
    labels: pd.Series,
    colors: dict[str, str],
    *,
    title: str,
    caption: str,
    value_name: str,
    legend_order: tuple[str, ...],
) -> MapLayer:
    table = base.copy()
    table["value"] = labels.to_numpy()
    table = table[table["value"].notna()].reset_index(drop=True)
    fill = palette.categorical_rgba(table["value"], colors)
    table["fill"] = [row.tolist() for row in fill]
    table["display"] = table["value"]
    present = set(table["value"])
    legend = tuple((label, colors[label]) for label in legend_order if label in present)
    return MapLayer(
        table=table,
        fill=fill,
        title=title,
        caption=caption,
        value_name=value_name,
        legend=legend,
    )


def build_layer(tables: Tables, state: LayerState) -> MapLayer:
    """Dispatch a sidebar selection to its layer builder."""
    if state.layer not in LAYERS:
        raise ValueError(f"unknown layer '{state.layer}'; expected one of {LAYERS}")
    if state.weighting not in tables.equity:
        raise ValueError(f"unknown weighting '{state.weighting}'")
    base = _base_frame(tables, state.weighting)
    builder = {
        "access": _access_layer,
        "jobs": _jobs_layer,
        "population": _population_layer,
        "gap": _gap_layer,
        "lisa": _lisa_layer,
        "deserts": _deserts_layer,
        "scenario_delta": _scenario_delta_layer,
    }[state.layer]
    return builder(tables, state, base)


def _scenario_frame(tables: Tables, state: LayerState) -> pd.DataFrame:
    if not state.scenario:
        raise ValueError("this view needs a scenario selected")
    if state.scenario not in tables.scenarios:
        raise ValueError(f"unknown scenario '{state.scenario}'")
    if state.mode not in tables.scenario_modes(state.scenario):
        modes = ", ".join(tables.scenario_modes(state.scenario))
        raise ValueError(f"scenario '{state.scenario}' changes only: {modes}")
    return tables.scenarios[state.scenario]


def _access_layer(tables: Tables, state: LayerState, base: pd.DataFrame) -> MapLayer:
    column = access_column(state)
    source, variant = tables.access, ""
    caveat = {
        "transit": " Transit is approximate and a lower bound (OSM routes, assumed headways).",
        "car": " Car times are free-flow: an upper bound.",
    }.get(state.mode, "")
    scenario_applies = (
        state.scenario is not None
        and state.opportunity == "jobs"
        and state.scenario in tables.scenarios
        and state.mode in tables.scenario_modes(state.scenario)
    )
    if scenario_applies:
        source, variant = tables.scenarios[state.scenario], "_scen"
    what = "jobs" if state.opportunity == "jobs" else state.opportunity
    pct = state.opportunity == "jobs" and state.metric == "accpct"
    title = (
        f"{'Share of city jobs' if pct else what.capitalize()} reachable · "
        f"{state.mode} · {threshold_label(state.threshold)}"
    )
    if scenario_applies:
        title += f" · with {tables.scenario_title(state.scenario).split(':')[0]}"
    caption = "City hexes only; color scale clipped at the 2nd-98th percentile." + caveat
    if state.scenario is not None and not scenario_applies and state.opportunity == "jobs":
        caption += " The selected scenario does not change this mode: baseline shown."
    return _sequential(
        base,
        base["h3"].map(source.set_index("h3")[column + variant]),
        title=title,
        caption=caption,
        value_name=title.split(" · ")[0],
        spec=",.1f" if pct else ",.0f",
        suffix="%" if pct else "",
    )


def _jobs_layer(tables: Tables, state: LayerState, base: pd.DataFrame) -> MapLayer:
    jobs = tables.jobs.set_index("h3")
    if state.uncertainty:
        values = base["h3"].map(jobs["jobs_rf_p90"] - jobs["jobs_rf_p10"])
        title = "Job estimate uncertainty (P90 - P10)"
        caption = (
            "Spread of the bootstrapped random-forest estimate: dark hexes are "
            "where the model is least sure. City hexes only."
        )
    else:
        values = base["h3"].map(jobs["jobs_rf"])
        title = "Estimated jobs per hex (random forest)"
        caption = (
            "Dasymetric estimate; district totals match official payroll jobs "
            "exactly. City hexes only; scale clipped at P2-P98."
        )
    return _sequential(base, values, title=title, caption=caption, value_name="Jobs", spec=",.0f")


def _population_layer(tables: Tables, state: LayerState, base: pd.DataFrame) -> MapLayer:
    note = (
        "Kontur raster counts"
        if state.weighting == "raster"
        else "raster rescaled to official district shares of employed residents"
    )
    return _sequential(
        base,
        base["population"],
        title=f"Residents per hex ({state.weighting} weighting)",
        caption=f"Population weighting: {note}. City hexes only.",
        value_name="Residents",
        spec=",.0f",
    )


def _gap_layer(tables: Tables, state: LayerState, base: pd.DataFrame) -> MapLayer:
    threshold = 45 if state.threshold == GRAVITY else int(state.threshold)
    column = f"modegap_transit_car_{threshold}"
    equity = tables.equity[state.weighting]
    if column not in equity.columns:
        raise ValueError(f"no mode-gap column for {threshold} min")
    return _sequential(
        base,
        base["h3"].map(equity.set_index("h3")[column]),
        title=f"Transit vs car · share of car-reachable jobs · {threshold} min",
        caption=(
            "Jobs reachable by transit as a share of jobs reachable by car; "
            "DARK = transit lags most. Divides a lower bound (transit) by an "
            "upper bound (car), so it overstates car dependence."
        ),
        value_name="Transit / car",
        spec=".0%",
        reverse=True,
    )


def _lisa_layer(tables: Tables, state: LayerState, base: pd.DataFrame) -> MapLayer:
    equity = tables.equity[state.weighting]
    column = lisa_column(equity, state.mode)
    threshold = column.rsplit("_", 1)[1]
    codes = base["h3"].map(equity.set_index("h3")[column])
    labels = codes.map(palette.LISA_LABELS)
    colors = {palette.LISA_LABELS[code]: color for code, color in palette.LISA_COLORS.items()}
    code_order = ("HH", "LL", "HL", "LH", "ns", "island")
    order = tuple(dict.fromkeys(palette.LISA_LABELS[code] for code in code_order))
    return _categorical(
        base,
        labels,
        colors,
        title=f"Local clusters of job access · {state.mode} · {threshold} min",
        caption=(
            "Local Moran's I on the share of jobs reachable "
            f"({state.weighting} weighting), tie-aware permutation p ≤ 0.05 "
            "with FDR control. Spatial outliers (HL / LH) share one color."
        ),
        value_name="Cluster",
        legend_order=order,
    )


def _deserts_layer(tables: Tables, state: LayerState, base: pd.DataFrame) -> MapLayer:
    equity = tables.equity[state.weighting]
    column = lisa_column(equity, state.mode).replace("lisa_", "desert_")
    threshold = column.rsplit("_", 1)[1]
    indexed = equity.set_index("h3")
    flags = base["h3"].map(indexed[column])
    labels = flags.map({True: DESERT_LABEL, False: OTHER_LABEL})
    # The desert verdict only exists for inhabited hexes; leave the rest blank.
    labels = labels.where(base["h3"].map(indexed["inhabited"]).fillna(False).astype(bool))
    return _categorical(
        base,
        labels,
        {DESERT_LABEL: palette.RED, OTHER_LABEL: palette.NEUTRAL},
        title=f"Access deserts · {state.mode} · {threshold} min",
        caption=(
            "Inhabited hexes in the person-weighted bottom of the access "
            f"distribution under the {state.weighting} weighting (D-055). "
            "Uninhabited hexes are unpainted."
        ),
        value_name="Status",
        legend_order=(DESERT_LABEL, OTHER_LABEL),
    )


def _scenario_delta_layer(tables: Tables, state: LayerState, base: pd.DataFrame) -> MapLayer:
    frame = _scenario_frame(tables, state)
    column = access_column(
        LayerState(
            layer="access",
            mode=state.mode,
            threshold=state.threshold,
            opportunity="jobs",
            metric=state.metric,
        )
    )
    pct = state.metric == "accpct" and state.threshold != GRAVITY
    title = (
        f"Gain with {tables.scenario_title(state.scenario).split(':')[0]} · "
        f"{state.mode} · {threshold_label(state.threshold)}"
    )
    deltas = base["h3"].map(frame.set_index("h3")[column + "_delta"])
    # Paint only the hexes that actually gain: a zero painted pale blue
    # would still read as "changed".
    return _sequential(
        base,
        deltas.where(deltas > 0),
        title=title,
        caption=(
            "Hexes with no gain are unpainted; deltas are never negative by "
            "construction. Illustrative alignment, not an official design; "
            "transit deltas inherit the lower-bound caveat."
        ),
        value_name="Gain",
        spec="+,.1f" if pct else "+,.0f",
        suffix=" pp" if pct else "",
    )
