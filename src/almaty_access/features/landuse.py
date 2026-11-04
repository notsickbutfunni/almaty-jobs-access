"""Land-use features: share of each hex covered by grouped classes.

Raw OSM landuse polygons (plus ``leisure=park``) are grouped into four
job-relevant classes. Within a group the polygons are dissolved first,
so overlapping same-class mapping cannot push a share above 1. ACROSS
groups a precedence applies: specific classes bite out of blanket ones
(coarse OSM ``landuse=residential`` polygons routinely blanket whole
industrial zones — without precedence the review measured per-hex share
sums up to 1.95). One square metre therefore belongs to at most one
class, shares sum to <= 1, and unmapped land fills the rest.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely import make_valid

#: landuse=<value> -> group. Values not listed belong to no group.
LANDUSE_GROUPS: dict[str, str] = {
    "industrial": "industrial",
    "railway": "industrial",
    "quarry": "industrial",
    "depot": "industrial",
    "commercial": "commercial",
    "retail": "commercial",
    "residential": "residential",
    "grass": "green",
    "forest": "green",
    "meadow": "green",
    "orchard": "green",
    "vineyard": "green",
    "village_green": "green",
    "recreation_ground": "green",
    "cemetery": "green",
    "flowerbed": "green",
    "greenfield": "green",
}

GROUPS: tuple[str, ...] = ("industrial", "commercial", "residential", "green")

#: Earlier wins where classes overlap: the specific (industrial zone,
#: shopping area, park) takes the ground from the blanket residential.
GROUP_PRECEDENCE: tuple[str, ...] = ("industrial", "commercial", "green", "residential")


def grouped_landuse(landuse: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Landuse polygons grouped, dissolved and made mutually exclusive."""
    frame = landuse.copy()
    frame["group"] = frame["landuse"].map(LANDUSE_GROUPS)
    frame.loc[frame.get("leisure").eq("park").fillna(False), "group"] = "green"
    frame = frame.dropna(subset=["group"])
    if frame.empty:
        return gpd.GeoDataFrame({"group": []}, geometry=[], crs=landuse.crs)
    frame.geometry = make_valid(frame.geometry.to_numpy())
    dissolved = frame[["group", "geometry"]].dissolve(by="group")

    taken = None
    rows: list[dict] = []
    for group in GROUP_PRECEDENCE:
        if group not in dissolved.index:
            continue
        geometry = dissolved.loc[group, "geometry"]
        if taken is not None:
            geometry = geometry.difference(taken)
        if not geometry.is_empty:
            rows.append({"group": group, "geometry": geometry})
        taken = geometry if taken is None else taken.union(geometry)
    if not rows:
        return gpd.GeoDataFrame({"group": []}, geometry=[], crs=landuse.crs)
    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=landuse.crs)
    return result.explode(ignore_index=True)


def landuse_features(
    landuse: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, metric_crs: str
) -> pd.DataFrame:
    """Per-hex land-use block (``lu_share_*`` columns), aligned to the grid."""
    index = pd.Index(grid["h3"], name="h3")
    out = pd.DataFrame(index=index)
    hexes = grid.set_index("h3")[["geometry"]].to_crs(metric_crs).rename_axis(None)
    hex_area = hexes.geometry.area

    parts = grouped_landuse(landuse)
    if parts.empty:
        for group in GROUPS:
            out[f"lu_share_{group}"] = 0.0
        return out
    parts = parts.to_crs(metric_crs)

    pairs = gpd.sjoin(parts, hexes[["geometry"]], predicate="intersects")
    overlap = (
        pairs.geometry.intersection(hexes.geometry.loc[pairs["index_right"]], align=False)
        .area.groupby([pairs["index_right"], pairs["group"]])
        .sum()
        .unstack(fill_value=0.0)
    )
    for group in GROUPS:
        column = overlap[group] if group in overlap.columns else pd.Series(dtype=float)
        share = (column.reindex(index).fillna(0.0) / hex_area.reindex(index)).clip(upper=1.0)
        out[f"lu_share_{group}"] = share
    return out
