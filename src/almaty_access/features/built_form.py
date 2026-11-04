"""Built-form features: how much building mass does each hex hold?

Floor area is the backbone of the Phase 3 job model (jobs live in
non-residential floor space), so mass must land in the right hex. A
building that straddles a hex border is split **area-weighted**: each
hex receives footprint and GFA in proportion to the geometric overlap,
computed in the metric CRS. Mass outside the grid domain is genuinely
outside and is dropped, not re-normalized in.

Counts and floor statistics are *per building* (indivisible), so they
go to the hex that contains the building's representative point.
"""

from __future__ import annotations

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import shapely

from almaty_access.schemas import BUILDING_FUNCTIONS

#: Structural-zero fills: a hex without buildings has zero mass and, by
#: convention, zero floor statistics (documented in the learning guide).
_ZERO_FILL = 0.0


def split_mass(
    buildings: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame, value_column: str
) -> pd.DataFrame:
    """Area-weighted share of each building's footprint and ``value_column``
    per hex. Both frames must be in the same metric CRS; ``hexes`` is
    indexed by cell id. Returns one row per (building, hex) pair with
    columns ``h3``, ``part_m2``, ``value`` and the building's index.
    """
    pairs = gpd.sjoin(buildings[["geometry"]], hexes[["geometry"]], predicate="intersects")
    if pairs.empty:
        return pd.DataFrame(columns=["h3", "part_m2", "value", "building"])
    parts = shapely.intersection(
        pairs.geometry.to_numpy(), hexes.geometry.loc[pairs["index_right"]].to_numpy()
    )
    part_area = shapely.area(parts)
    footprint = pairs.geometry.area.to_numpy()
    weight = np.divide(part_area, footprint, out=np.zeros_like(part_area), where=footprint > 0)
    return pd.DataFrame(
        {
            "h3": pairs["index_right"].to_numpy(),
            "part_m2": part_area,
            "value": buildings[value_column].loc[pairs.index].to_numpy() * weight,
            "building": pairs.index.to_numpy(),
        }
    )


def built_form_features(
    buildings: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    res: int,
    metric_crs: str,
    large_gfa_m2: float,
) -> pd.DataFrame:
    """Per-hex built-form block (``bf_`` columns), aligned to the grid."""
    index = pd.Index(grid["h3"], name="h3")
    # rename_axis(None): sjoin would otherwise name its join column after
    # the right index ("h3") instead of the "index_right" we read below.
    hexes = grid.set_index("h3")[["geometry"]].to_crs(metric_crs).rename_axis(None)
    blds = buildings.to_crs(metric_crs)

    mass = split_mass(blds, hexes, "gfa_m2")
    mass["function"] = blds["function"].loc[mass["building"]].to_numpy()
    mass["is_large"] = (blds["gfa_m2"] > large_gfa_m2).loc[mass["building"]].to_numpy()

    totals = mass.groupby("h3")[["part_m2", "value"]].sum()
    by_function = mass.pivot_table(
        index="h3", columns="function", values="value", aggfunc="sum", fill_value=0.0
    )
    large = mass.loc[mass["is_large"]].groupby("h3")["value"].sum()

    # Counts and floor statistics: indivisible, assigned by representative point.
    points = buildings.geometry.representative_point()
    cells = pd.Series(
        [h3.latlng_to_cell(lat, lon, res) for lat, lon in zip(points.y, points.x, strict=True)],
        index=buildings.index,
        name="h3",
    )
    per_building = pd.DataFrame({"h3": cells, "levels": buildings["levels"]})
    counts = per_building.groupby("h3").size()
    levels = per_building.groupby("h3")["levels"].agg(["mean", "max"])

    out = pd.DataFrame(index=index)
    out["bf_bldg_cnt"] = counts.reindex(index).fillna(0).astype(int)
    out["bf_footprint_m2"] = totals["part_m2"].reindex(index).fillna(_ZERO_FILL)
    out["bf_gfa_m2"] = totals["value"].reindex(index).fillna(_ZERO_FILL)
    for function in BUILDING_FUNCTIONS:
        column = by_function[function] if function in by_function else pd.Series(dtype=float)
        out[f"bf_gfa_{function}_m2"] = column.reindex(index).fillna(_ZERO_FILL)
    out["bf_gfa_large_m2"] = large.reindex(index).fillna(_ZERO_FILL)
    gfa = out["bf_gfa_m2"].to_numpy()
    out["bf_share_large_gfa"] = np.divide(
        out["bf_gfa_large_m2"].to_numpy(), gfa, out=np.zeros_like(gfa), where=gfa > 0
    )
    out["bf_levels_mean"] = levels["mean"].reindex(index).fillna(_ZERO_FILL)
    out["bf_levels_max"] = levels["max"].reindex(index).fillna(_ZERO_FILL)
    return out
