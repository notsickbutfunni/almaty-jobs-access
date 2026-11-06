"""Altair chart specs for the district panel (pure, no Streamlit).

Charts follow the project chart system: one blue for magnitude, the
fixed weighting pair (light blue = raster, blue = calibrated), ink-token
text, recessive grid, a legend whenever two series share a plot.
"""

from __future__ import annotations

import re

import altair as alt
import pandas as pd

from almaty_access.dashboard import palette

_JOB_RE = re.compile(r"^(acc|accpct)_jobs_([a-z]+)_(\d+|grav)$")
_SFCA_RE = re.compile(r"^sfca_(\w+)_walk$")

BAR_HEIGHT = 26


def measure_label(measure: str) -> str:
    """Human name for a stored measure column."""
    job = _JOB_RE.fullmatch(measure)
    if job:
        metric, mode, threshold = job.groups()
        what = "Share of city jobs" if metric == "accpct" else "Jobs"
        when = "gravity-weighted" if threshold == "grav" else f"{threshold} min"
        unit = " (%)" if metric == "accpct" else ""
        return f"{what} reachable · {mode} · {when}{unit}"
    sfca = _SFCA_RE.fullmatch(measure)
    if sfca:
        return f"{sfca.group(1).capitalize()} per 1,000 residents (2SFCA, walk)"
    return measure


def resolution_for(frame: pd.DataFrame, measure: str, weighting: str) -> int:
    """The H3 resolution a long table stores for this measure.

    Job measures exist at r8 (walk at both), the walk-catchment 2SFCA
    measures at r9 only — prefer the analysis resolution when available.
    """
    rows = frame.loc[(frame["measure"] == measure) & (frame["weights"] == weighting), "res"]
    if rows.empty:
        raise ValueError(f"no rows for {measure} / {weighting}")
    levels = sorted({int(value) for value in rows})
    return 8 if 8 in levels else levels[0]


def _styled(chart: alt.Chart | alt.LayerChart) -> alt.Chart | alt.LayerChart:
    """Recessive chrome, chart-system text colors."""
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            labelColor=palette.INK_2,
            titleColor=palette.INK_2,
            gridColor=palette.GRID_LINE,
            domainColor=palette.AXIS,
            tickColor=palette.AXIS,
        )
        .configure_legend(labelColor=palette.INK_2, titleColor=palette.INK_2)
        .configure_title(color=palette.INK, fontSize=13, anchor="start")
    )


def district_bars(
    districts: pd.DataFrame, measure: str, weighting: str, *, res: int | None = None
) -> alt.LayerChart:
    """Mean access by district, one blue bar each, value-labelled."""
    if res is None:
        res = resolution_for(districts, measure, weighting)
    rows = districts.loc[
        (districts["measure"] == measure)
        & (districts["weights"] == weighting)
        & (districts["res"] == res),
        ["district", "mean", "population"],
    ].copy()
    if rows.empty:
        raise ValueError(f"no district rows for {measure} / {weighting} / r{res}")
    rows["district"] = rows["district"].str.capitalize()
    rows["label"] = rows["mean"].map(lambda v: f"{v:,.1f}")
    base = alt.Chart(rows).encode(
        y=alt.Y("district:N", sort="-x", title=None),
        x=alt.X("mean:Q", title=measure_label(measure)),
    )
    bars = base.mark_bar(color=palette.BLUE, cornerRadiusEnd=4, height=BAR_HEIGHT - 10)
    text = base.mark_text(align="left", dx=4, color=palette.INK).encode(text="label:N")
    chart = (bars + text).properties(
        height=BAR_HEIGHT * len(rows),
        title=f"District means · {weighting} weighting · r{res} hexes",
    )
    return _styled(chart)


def lorenz_curve(
    lorenz: pd.DataFrame,
    summary: pd.DataFrame,
    measure: str,
    weighting: str,
    *,
    res: int | None = None,
) -> alt.LayerChart:
    """Lorenz curve for one measure and weighting, Gini in the title."""
    if res is None:
        res = resolution_for(lorenz, measure, weighting)
    rows = lorenz.loc[
        (lorenz["measure"] == measure) & (lorenz["weights"] == weighting) & (lorenz["res"] == res)
    ]
    if rows.empty:
        raise ValueError(f"no Lorenz rows for {measure} / {weighting} / r{res}")
    gini_rows = summary.loc[
        (summary["measure"] == measure)
        & (summary["weights"] == weighting)
        & (summary["res"] == res),
        "gini",
    ]
    gini = float(gini_rows.iloc[0]) if not gini_rows.empty else float("nan")
    axis_x = alt.X(
        "pop_share:Q",
        title="Share of residents (poorest access first)",
        axis=alt.Axis(format=".0%"),
        scale=alt.Scale(domain=[0, 1]),
    )
    axis_y = alt.Y(
        "access_share:Q",
        title="Share of total access",
        axis=alt.Axis(format=".0%"),
        scale=alt.Scale(domain=[0, 1]),
    )
    curve = alt.Chart(rows).mark_line(color=palette.BLUE, strokeWidth=2).encode(axis_x, axis_y)
    equality = (
        alt.Chart(pd.DataFrame({"pop_share": [0.0, 1.0], "access_share": [0.0, 1.0]}))
        .mark_line(color=palette.AXIS, strokeDash=[4, 3])
        .encode(axis_x, axis_y)
    )
    chart = (equality + curve).properties(
        width=340,
        height=340,
        title=f"Lorenz curve · Gini {gini:.3f} · {weighting} weighting",
    )
    return _styled(chart)


def gains_bars(gains: pd.DataFrame, scenario: str, *, res: int = 8) -> alt.LayerChart:
    """Residents gaining more than each jobs cutoff, both weightings."""
    rows = gains.loc[(gains["scenario"] == scenario) & (gains["res"] == res)].copy()
    if rows.empty:
        raise ValueError(f"no gains rows for scenario '{scenario}'")
    measure = str(rows["measure"].iloc[0])

    def _cutoff_label(value: float) -> str:
        return f"> {value / 1000:g}k jobs" if value >= 1000 else f"> {value:g} jobs"

    rows["cutoff"] = rows["cutoff_jobs"].map(_cutoff_label)
    rows["label"] = rows["residents_gaining"].map(lambda v: f"{v:,.0f}")
    weightings = [w for w in palette.WEIGHTING_COLORS if w in set(rows["weights"])]
    colors = [palette.WEIGHTING_COLORS[w] for w in weightings]
    base = alt.Chart(rows).encode(
        x=alt.X("cutoff:N", sort=None, title="Gain cutoff", axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("weights:N", sort=weightings),
        y=alt.Y("residents_gaining:Q", title="Residents gaining"),
        color=alt.Color(
            "weights:N",
            title="Population weighting",
            scale=alt.Scale(domain=weightings, range=colors),
        ),
    )
    bars = base.mark_bar(cornerRadiusEnd=4)
    # Text wears ink, never the series color (alt.value beats the encoding).
    text = base.mark_text(dy=-6, fontSize=10).encode(text="label:N", color=alt.value(palette.INK))
    chart = (bars + text).properties(
        height=260,
        title=f"Who gains · {measure_label(measure)}",
    )
    return _styled(chart)
