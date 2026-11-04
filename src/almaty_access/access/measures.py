"""Accessibility measures over long travel-time tables (README §7 Phase 4.3).

All functions take the long matrix (``origin_h3, dest_h3, minutes``) plus
hex-indexed opportunity values and return origin-hex-indexed results.
Because the matrices only store pairs within the cutoff, every measure
is implicitly truncated there — documented, and harmless for gravity
(a job 60 min away at half-life 20 keeps 12.5% weight; farther ones
would add almost nothing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Per-1,000-residents scaling of the 2SFCA ratio (raw facilities per
#: person are unreadably small numbers).
SFCA_PER = 1000.0


def cumulative(tt: pd.DataFrame, values: pd.Series, thresholds: tuple[int, ...]) -> pd.DataFrame:
    """Opportunities reachable within each threshold: A_i(T) = sum O_j [t_ij <= T].

    Returns a frame indexed by origin hex with one column per threshold
    (int column names); only origins present in the matrix appear.
    """
    dest_value = values.reindex(tt["dest_h3"]).fillna(0.0).to_numpy()
    minutes = tt["minutes"].to_numpy()
    out: dict[int, pd.Series] = {}
    for threshold in thresholds:
        mask = minutes <= threshold
        out[threshold] = (
            pd.Series(dest_value[mask]).groupby(pd.Series(tt["origin_h3"].to_numpy()[mask])).sum()
        )
    frame = pd.DataFrame(out)
    frame.index.name = "origin_h3"
    return frame


def gravity(tt: pd.DataFrame, values: pd.Series, half_life_min: float) -> pd.Series:
    """Hansen gravity access: A_i = sum O_j exp(-beta t_ij), beta = ln2 / half-life."""
    beta = np.log(2.0) / half_life_min
    dest_value = values.reindex(tt["dest_h3"]).fillna(0.0).to_numpy()
    weighted = dest_value * np.exp(-beta * tt["minutes"].to_numpy())
    result = pd.Series(weighted).groupby(pd.Series(tt["origin_h3"].to_numpy())).sum()
    result.index.name = "origin_h3"
    return result


def two_step_fca(
    tt: pd.DataFrame,
    supply: pd.Series,
    population: pd.Series,
    catchment_min: float,
) -> pd.Series:
    """Two-step floating catchment area, per 1,000 residents.

    Step 1 — each facility hex j gets a supply-to-demand ratio
    R_j = S_j / sum(P_i over hexes i that reach j within the catchment):
    many residents competing for the same clinics shrink R_j.
    Step 2 — each origin hex i sums the ratios it can reach:
    A_i = sum(R_j over j within the catchment). Facility hexes that no
    population reaches keep R_j = 0 rather than dividing by zero.
    """
    within = tt.loc[tt["minutes"].to_numpy() <= catchment_min, ["origin_h3", "dest_h3"]]

    origin_pop = population.reindex(within["origin_h3"]).fillna(0.0).to_numpy()
    demand = pd.Series(origin_pop).groupby(pd.Series(within["dest_h3"].to_numpy())).sum()
    supply_j = supply.reindex(demand.index).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pd.Series(
            np.where(demand.to_numpy() > 0, supply_j.to_numpy() / demand.to_numpy(), 0.0),
            index=demand.index,
        )

    reach_ratio = ratio.reindex(within["dest_h3"]).fillna(0.0).to_numpy()
    result = (
        pd.Series(reach_ratio).groupby(pd.Series(within["origin_h3"].to_numpy())).sum() * SFCA_PER
    )
    result.index.name = "origin_h3"
    return result
