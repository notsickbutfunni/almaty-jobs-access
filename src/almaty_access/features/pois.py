"""POI features: counts per category, density, diversity and shares.

Category diversity uses Shannon entropy: H = -Σ p·ln p over the POI
categories present in a hex. H = 0 means a single-purpose hex; higher
values mean a mixed-use one — mixed-use areas are where service jobs
cluster, which is why the README asks for it.
"""

from __future__ import annotations

from collections.abc import Sequence

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

#: Office-type jobs (README: "share of office/commercial POIs").
OFFICE_CATEGORIES: tuple[str, ...] = ("office", "finance")

#: Consumer commerce.
COMMERCIAL_CATEGORIES: tuple[str, ...] = (
    "grocery",
    "market",
    "food_service",
    "retail_other",
    "fuel_auto",
    "hotel",
)


def poi_cells(pois: gpd.GeoDataFrame, res: int) -> pd.Series:
    """The H3 cell containing each POI (expects EPSG:4326 points)."""
    return pd.Series(
        [
            h3.latlng_to_cell(lat, lon, res)
            for lat, lon in zip(pois.geometry.y, pois.geometry.x, strict=True)
        ],
        index=pois.index,
        name="h3",
    )


def shannon_entropy(counts: pd.DataFrame) -> pd.Series:
    """Row-wise Shannon entropy of category counts (0 for empty rows)."""
    totals = counts.sum(axis=1).to_numpy()[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts.to_numpy() / totals
        terms = np.where(p > 0, p * np.log(p), 0.0)
    entropy = -np.nansum(terms, axis=1)
    return pd.Series(entropy, index=counts.index)


def poi_features(
    pois: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, res: int, categories: Sequence[str]
) -> pd.DataFrame:
    """Per-hex POI block (``poi_`` columns), aligned to the grid."""
    index = pd.Index(grid["h3"], name="h3")
    frame = pd.DataFrame({"h3": poi_cells(pois, res), "category": pois["category"]})
    counts = (
        frame.groupby(["h3", "category"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=index, columns=list(categories))
        .fillna(0)
        .astype(int)
    )

    out = counts.add_prefix("poi_cnt_")
    total = counts.sum(axis=1)
    out["poi_cnt_total"] = total
    out["poi_density_km2"] = total / h3.average_hexagon_area(res, unit="km^2")
    out["poi_entropy"] = shannon_entropy(counts)
    denominator = total.to_numpy().astype(float)
    for name, group in (("office", OFFICE_CATEGORIES), ("commercial", COMMERCIAL_CATEGORIES)):
        present = [c for c in group if c in counts.columns]
        share = np.divide(
            counts[present].sum(axis=1).to_numpy(),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        out[f"poi_share_{name}"] = share
    return out
