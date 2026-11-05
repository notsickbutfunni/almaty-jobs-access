"""Equity tables: districts, nearest-facility shares, mode gap, gradient.

Small population-weighted aggregations over the inhabited hexes. Every
function takes a frame with a ``population`` column and returns plain
tables or dicts ready for parquet / JSON.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from almaty_access.equity.inequality import weighted_quantile

#: ``district`` value of the city-wide rows in the nearest-facility table.
CITY_SCOPE = "ALL"


def district_table(frame: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Population-weighted mean and median of a measure per district.

    Columns: ``district, population, mean, median, rank`` — rank 1 is
    the best-served district by weighted mean.
    """
    rows = []
    for district, group in frame.dropna(subset=["district"]).groupby("district", sort=True):
        weights = group["population"].to_numpy()
        values = group[measure].to_numpy()
        rows.append(
            {
                "district": str(district),
                "population": float(weights.sum()),
                "mean": float((values * weights).sum() / weights.sum()),
                "median": weighted_quantile(values, weights, 0.5),
            }
        )
    table = pd.DataFrame(rows, columns=["district", "population", "mean", "median"])
    table["rank"] = table["mean"].rank(ascending=False, method="min").astype("int64")
    return table.sort_values(["rank", "district"]).reset_index(drop=True)


def below_cutoff_shares(
    frame: pd.DataFrame, measure: str, cutoffs: tuple[float, ...]
) -> dict[str, float]:
    """Share of residents whose value is strictly below each cutoff."""
    population = frame["population"].to_numpy()
    values = frame[measure].to_numpy()
    total = population.sum()
    return {
        f"{cutoff:g}": round(float(population[values < cutoff].sum() / total), 4)
        for cutoff in cutoffs
    }


def nearest_tables(
    frame: pd.DataFrame, thresholds: tuple[int, ...]
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Residents within T walking minutes of the nearest facility.

    Scans every ``near_{facility}_walk_min`` column, for the whole city
    (``district == "ALL"``) and per district. A missing time (off-network
    hex or unreachable facility) counts as BEYOND every threshold —
    dropping those residents would hide the worst served. Returns the
    long threshold table and a city-wide per-facility summary with the
    weighted median / p90 over residents that have a time at all.
    """
    population = frame["population"].to_numpy()
    total = population.sum()
    # Plain strings with "" for a missing district: comparing an array
    # holding pd.NA would yield NA instead of a boolean mask.
    districts = frame["district"].astype("string").fillna("").to_numpy(dtype=str)
    scopes: list[tuple[str, np.ndarray]] = [(CITY_SCOPE, np.ones(len(frame), dtype=bool))]
    scopes += [(name, districts == name) for name in sorted(set(districts) - {""})]

    rows = []
    summary: dict[str, dict] = {}
    for column in sorted(c for c in frame.columns if c.startswith("near_")):
        facility = column.removeprefix("near_").removesuffix("_walk_min")
        minutes = frame[column].to_numpy(dtype="float64")
        reachable = np.isfinite(minutes)
        for scope, members in scopes:
            scope_total = population[members].sum()
            for threshold in thresholds:
                within = members & reachable & (minutes <= threshold)
                rows.append(
                    {
                        "district": scope,
                        "facility": facility,
                        "threshold_min": int(threshold),
                        "pop_share_within": round(float(population[within].sum() / scope_total), 4),
                    }
                )
        summary[facility] = {
            "median_min": round(weighted_quantile(minutes, population, 0.5), 1),
            "p90_min": round(weighted_quantile(minutes, population, 0.9), 1),
            "pop_share_unreachable": round(float(population[~reachable].sum() / total), 4),
        }
    table = pd.DataFrame(
        rows, columns=["district", "facility", "threshold_min", "pop_share_within"]
    )
    return table, summary


def mode_gap(table: pd.DataFrame, thresholds: tuple[int, ...]) -> pd.DataFrame:
    """Transit / car job-access ratio per hex and threshold.

    ``modegap_transit_car_{T}`` = jobs within T minutes by transit over
    jobs within T minutes by car. Near 1: transit competes; near 0: the
    car reaches far more — car dependence. NaN where the car reaches
    nothing (the ratio is undefined, not zero).
    """
    out = pd.DataFrame(index=table.index)
    for threshold in thresholds:
        transit = table[f"acc_jobs_transit_{threshold}"].to_numpy(dtype="float64")
        car = table[f"acc_jobs_car_{threshold}"].to_numpy(dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(car > 0, transit / car, np.nan)
        out[f"modegap_transit_car_{threshold}"] = np.round(ratio, 4)
    return out


def center_gradient(
    frame: pd.DataFrame, measures: list[str], bands_km: tuple[float, ...]
) -> list[dict]:
    """Population and weighted mean access by distance-to-centre band.

    ``frame`` needs ``center_km``. Bands are ``(previous edge, edge]``
    plus one open band beyond the last edge.
    """
    edges = [0.0, *bands_km, np.inf]
    total = float(frame["population"].sum())
    rows = []
    distance = frame["center_km"]
    for low, high in itertools.pairwise(edges):
        # The first band is closed at zero so the centre hex itself counts.
        above_low = (distance >= low) if low == 0.0 else (distance > low)
        band = frame.loc[above_low & (distance <= high)]
        label = f"{low:g}-{high:g} km" if np.isfinite(high) else f">{low:g} km"
        population = float(band["population"].sum())
        row: dict = {
            "band": label,
            "n_hexes": len(band),
            "population": round(population, 0),
            "pop_share": round(population / total, 4) if total > 0 else None,
        }
        for measure in measures:
            if population > 0:
                weights = band["population"].to_numpy()
                row[measure] = round(
                    float((band[measure].to_numpy() * weights).sum() / population), 4
                )
            else:
                row[measure] = None
        rows.append(row)
    return rows
