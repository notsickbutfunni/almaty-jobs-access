"""Click-to-reach: what one hex reaches within a time budget.

Works off the stored r8 travel-time matrices (README keeps the
click-to-reach view at the analysis resolution for hosting reasons), on
the BASELINE network only — Phase 6 stored scenario deltas, not scenario
matrices, and the app says so whenever a scenario is active.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from almaty_access.dashboard.data import Tables

_NEAR_RE = re.compile(r"^near_(\w+)_walk_min$")
_FACILITY_RE = re.compile(r"^acc_([a-z]+)_walk_(\d+)$")


@dataclass(frozen=True)
class ReachSummary:
    """The small panel shown after clicking a hex."""

    origin: str
    mode: str
    minutes: int
    n_reachable: int
    jobs_sum: float  # jobs_rf summed over the reachable hexes
    jobs_stored: float | None  # the stored acc_jobs_* value, as a cross-check
    facilities_walk: pd.DataFrame  # facility, count (walk, same budget)
    nearest_walk: pd.DataFrame  # facility, minutes


def reachable(tt: pd.DataFrame, origin: str, minutes: float) -> pd.DataFrame:
    """Hexes reachable from ``origin`` within ``minutes`` (self included).

    The stored matrices keep pairs up to 60 minutes, so any threshold the
    sidebar offers is answerable exactly.
    """
    hits = tt.loc[(tt["origin_h3"] == origin) & (tt["minutes"] <= minutes)]
    frame = hits[["dest_h3", "minutes"]].rename(columns={"dest_h3": "h3"})
    return frame.sort_values("minutes").reset_index(drop=True)


def reach_summary(tables: Tables, origin: str, mode: str, minutes: int) -> ReachSummary:
    """Reachable-set totals plus the origin's stored facility numbers."""
    if mode not in tables.tt:
        raise ValueError(f"no travel-time matrix for mode '{mode}'")
    access = tables.access.set_index("h3")
    if origin not in access.index:
        raise ValueError(f"hex {origin} is not in the analysis grid")
    reached = reachable(tables.tt[mode], origin, minutes)
    jobs = tables.jobs.set_index("h3")["jobs_rf"]
    jobs_sum = float(jobs.reindex(reached["h3"]).fillna(0.0).sum())
    stored_col = f"acc_jobs_{mode}_{minutes}"
    row = access.loc[origin]
    stored = float(row[stored_col]) if stored_col in access.columns else None
    facilities = [
        {"facility": match.group(1), "count": int(row[col])}
        for col in access.columns
        if (match := _FACILITY_RE.fullmatch(col))
        and int(match.group(2)) == minutes
        and pd.notna(row[col])
    ]
    nearest = [
        {"facility": match.group(1), "minutes": float(row[col])}
        for col in access.columns
        if (match := _NEAR_RE.fullmatch(col)) and pd.notna(row[col])
    ]
    return ReachSummary(
        origin=origin,
        mode=mode,
        minutes=minutes,
        n_reachable=len(reached),
        jobs_sum=jobs_sum,
        jobs_stored=stored,
        facilities_walk=pd.DataFrame(facilities, columns=["facility", "count"]),
        nearest_walk=pd.DataFrame(nearest, columns=["facility", "minutes"]),
    )
