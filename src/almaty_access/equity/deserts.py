"""Access deserts and bivariate classes: where many people meet little access.

Both tools cross population with accessibility at hex level, and both
ask the access question about PEOPLE rather than land:

- A **desert** is a dense hex (population above the hex quantile, README:
  75th percentile) whose access is at or below the population-weighted
  access quantile (25th) — the level under which the worst-served
  quarter of residents live.
- The **bivariate classes** sort every inhabited hex into a 3x3 grid:
  population tercile across hexes x access tercile across residents.

Why not the README's literal reading (both percentiles across hexes)?
The hex-level access percentile is set by the many near-empty outskirt
cells, far below anything a dense hex experiences: for the four r8
headline job measures the literal rule flags at most one hex. Its
count is reported next to every result so the difference stays visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from almaty_access.equity.inequality import bottom_share_mean, weighted_quantile

#: When the worst-served quarter of residents still reaches, ON AVERAGE,
#: this share of what the median resident reaches, the measure has no
#: meaningful low tail (free-flow car access: 98.5%) and the relative rule
#: would only flag noise — no deserts are reported for it. The mean of
#: the quarter is used, not its upper edge: when a quarter of residents
#: share the median value the edge says nothing about the tail below.
LOW_SPREAD_RATIO = 0.95


def access_deserts(
    frame: pd.DataFrame,
    measure: str,
    *,
    pop_quantile: float,
    access_quantile: float,
) -> tuple[pd.DataFrame, dict]:
    """Desert hexes for one measure, ranked by residents affected.

    ``frame`` holds the inhabited hexes: ``h3``, ``district``,
    ``population`` (the weights in force) and the measure; an optional
    ``point_mass`` column is carried into the result. Returns the ranked
    table and an info dict:

    - ``access_threshold`` — the population-weighted quantile used;
      ``access_threshold_hex_quantile`` — the README-literal one;
    - ``n_hexes_literal_rule`` — hexes the literal rule would flag, with
      the SAME ``<=`` comparison, so the two counts differ only by the
      weighting of the quantile;
    - ``pop_share_at_or_below_access_threshold`` — exceeds the quantile
      when ties pile up at the threshold (a zero-inflated measure);
    - ``low_spread`` — True when the measure has no meaningful low tail
      (see ``LOW_SPREAD_RATIO``); the desert list is then empty.
    """
    population = frame["population"]
    access = frame[measure]
    weights = population.to_numpy()
    values = access.to_numpy()
    pop_threshold = float(population.quantile(pop_quantile))
    access_threshold = weighted_quantile(values, weights, access_quantile)
    literal_threshold = float(access.quantile(access_quantile))
    median = weighted_quantile(values, weights, 0.5)
    worst_mean = bottom_share_mean(values, weights, access_quantile)
    spread_ratio = worst_mean / median if median > 0 else 0.0
    low_spread = bool(spread_ratio >= LOW_SPREAD_RATIO)

    dense = population > pop_threshold
    mask = dense & (access <= access_threshold) & (not low_spread)
    n_literal = int((dense & (access <= literal_threshold)).sum())

    columns = ["h3", "district", "population", measure]
    if "point_mass" in frame.columns:
        columns.append("point_mass")
    deserts = (
        frame.loc[mask, columns]
        .rename(columns={measure: "value"})
        .sort_values(["population", "h3"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if "point_mass" not in deserts.columns:
        deserts["point_mass"] = False
    deserts.insert(0, "rank", np.arange(1, len(deserts) + 1))

    total = float(population.sum())
    affected = float(deserts["population"].sum())
    at_or_below = float(population[access <= access_threshold].sum())
    by_district = (
        deserts.groupby("district")["population"].sum().sort_values(ascending=False).round(0)
    )
    info = {
        "pop_threshold": round(pop_threshold, 1),
        "access_threshold": round(float(access_threshold), 4),
        "access_threshold_hex_quantile": round(literal_threshold, 4),
        "pop_share_at_or_below_access_threshold": (
            round(at_or_below / total, 4) if total > 0 else None
        ),
        "spread_ratio": round(float(spread_ratio), 4),
        "low_spread": low_spread,
        "n_hexes": len(deserts),
        "population": round(affected, 0),
        "pop_share": round(affected / total, 4) if total > 0 else None,
        "n_hexes_literal_rule": n_literal,
        "n_point_mass_hexes": int(deserts["point_mass"].sum()),
        "population_by_district": {str(k): float(v) for k, v in by_district.items()},
    }
    return deserts, info


def tercile_classes(values: pd.Series, weights: pd.Series | None = None) -> pd.Series:
    """Classes 1 (low) to 3 (high) by the 1/3 and 2/3 quantiles.

    Without weights the quantiles run across hexes; with weights across
    residents. Ties always share a class: a rank-based split would hand
    hexes with IDENTICAL values different classes, which is arbitrary —
    and common for zero-inflated measures. The price is unequal class
    sizes (all zeros land together in class 1), which is the honest
    picture.
    """
    if weights is None:
        lower, upper = values.quantile(1 / 3), values.quantile(2 / 3)
    else:
        lower = weighted_quantile(values.to_numpy(), weights.to_numpy(), 1 / 3)
        upper = weighted_quantile(values.to_numpy(), weights.to_numpy(), 2 / 3)
    return (1 + (values > lower).astype(int) + (values > upper).astype(int)).astype("int64")


def bivariate_classes(population: pd.Series, access: pd.Series) -> pd.Series:
    """3x3 class labels ``"{population tercile}-{access tercile}"``.

    Population terciles run across hexes (density is a property of the
    hex); access terciles across RESIDENTS, for the same reason the
    desert rule weights its access quantile: ``"3-1"`` then means "a
    dense hex among the worst-served third of residents", not "below a
    cut set by empty outskirts". ``"1-3"`` is few people with plenty.
    """
    pop_class = tercile_classes(population)
    access_class = tercile_classes(access, population)
    return (pop_class.astype(str) + "-" + access_class.astype(str)).astype("string")
