"""Transit features: stop supply and rapid-transit proximity.

``transit_stop`` POIs approximate physical stops (platforms, bus stops,
rail/tram stations; stop_position and subway entrances were excluded in
Phase 1 to avoid double counting). Metro stations are identified by
``railway=station`` + ``station=subway``; heavy-rail stations by
``railway=station`` without the subway mark. Routes-per-hex needs OSM
route relations, which Phase 4 extracts for its GTFS fallback — noted
in DECISIONS.md, not fabricated here.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import h3
import pandas as pd

logger = logging.getLogger(__name__)


def metro_stations(pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """POIs that are metro stations (railway=station + station=subway)."""
    railway = pois.get("railway", pd.Series(pd.NA, index=pois.index)).astype("string")
    station = pois.get("station", pd.Series(pd.NA, index=pois.index)).astype("string")
    return pois.loc[railway.eq("station").fillna(False) & station.eq("subway").fillna(False)]


def rail_stations(pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """POIs that are railway stations of any kind (metro included)."""
    railway = pois.get("railway", pd.Series(pd.NA, index=pois.index)).astype("string")
    return pois.loc[railway.eq("station").fillna(False)]


def _distance_to_nearest(
    centroids: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    index: pd.Index,
    fallback_m: float,
    what: str,
) -> pd.Series:
    if targets.empty:
        logger.warning("no %s in the study area; using fallback distance", what)
        return pd.Series(fallback_m, index=index)
    nearest = gpd.sjoin_nearest(centroids, targets[["geometry"]], distance_col="distance_m")
    distance = nearest["distance_m"].groupby(level=0).min()
    return distance.reindex(index).fillna(fallback_m)


def transit_features(
    pois: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    res: int,
    metric_crs: str,
    fallback_distance_m: float,
) -> pd.DataFrame:
    """Per-hex transit block (``tr_`` columns), aligned to the grid."""
    index = pd.Index(grid["h3"], name="h3")
    out = pd.DataFrame(index=index)
    hexes = grid.set_index("h3")[["geometry"]].to_crs(metric_crs)
    centroids = gpd.GeoDataFrame(geometry=hexes.geometry.centroid, index=hexes.index)

    stops = pois.loc[pois["category"].eq("transit_stop")]
    cells = pd.Series(
        [
            h3.latlng_to_cell(lat, lon, res)
            for lat, lon in zip(stops.geometry.y, stops.geometry.x, strict=True)
        ]
    )
    counts = cells.value_counts()
    out["tr_stop_cnt"] = counts.reindex(index).fillna(0).astype(int)
    out["tr_stop_density_km2"] = out["tr_stop_cnt"] / h3.average_hexagon_area(res, unit="km^2")

    stops_metric = stops.to_crs(metric_crs)
    metro_metric = metro_stations(pois).to_crs(metric_crs)
    rail_metric = rail_stations(pois).to_crs(metric_crs)
    out["tr_dist_stop_m"] = _distance_to_nearest(
        centroids, stops_metric, index, fallback_distance_m, "transit stops"
    )
    out["tr_dist_metro_m"] = _distance_to_nearest(
        centroids, metro_metric, index, fallback_distance_m, "metro stations"
    )
    out["tr_dist_rail_m"] = _distance_to_nearest(
        centroids, rail_metric, index, fallback_distance_m, "railway stations"
    )
    return out
