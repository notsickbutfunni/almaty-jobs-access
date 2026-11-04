"""Centrality features: distance to the centre and to job sub-centres.

Sub-centres are found by clustering business POIs with DBSCAN in the
metric CRS (README: "nearest major sub-center (found by clustering
POIs)"). DBSCAN needs no cluster count up front and ignores noise
points, which fits an unknown polycentric structure. The cluster whose
centroid lies nearest the configured historic centre IS the centre, so
it is excluded; the remaining clusters are ranked by POI count.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

from almaty_access.config import Config
from almaty_access.features.pois import COMMERCIAL_CATEGORIES, OFFICE_CATEGORIES

logger = logging.getLogger(__name__)

#: POI categories that mark job concentrations.
BUSINESS_CATEGORIES: tuple[str, ...] = OFFICE_CATEGORIES + COMMERCIAL_CATEGORIES


def center_point(config: Config) -> gpd.GeoSeries:
    """The configured historic centre as a one-point GeoSeries (metric)."""
    point = gpd.GeoSeries(
        [Point(config.features.center_lon, config.features.center_lat)],
        crs=config.crs.storage,
    )
    return point.to_crs(config.crs.metric)


def find_subcenters(pois: gpd.GeoDataFrame, config: Config) -> gpd.GeoDataFrame:
    """Job sub-centres as points (storage CRS, schemas.PROCESSED_SUBCENTERS)."""
    params = config.features.subcenters
    business = pois.loc[pois["category"].isin(BUSINESS_CATEGORIES)].to_crs(config.crs.metric)
    empty = gpd.GeoDataFrame(
        {"subcenter_id": pd.Series(dtype=int), "n_pois": pd.Series(dtype=int)},
        geometry=gpd.GeoSeries([], crs=config.crs.storage),
    )
    if len(business) < params.min_samples:
        logger.warning("too few business POIs (%d) to cluster sub-centres", len(business))
        return empty

    coordinates = np.column_stack([business.geometry.x, business.geometry.y])
    labels = DBSCAN(eps=params.eps_m, min_samples=params.min_samples).fit_predict(coordinates)
    clustered = pd.DataFrame({"label": labels, "x": coordinates[:, 0], "y": coordinates[:, 1]}).loc[
        labels >= 0
    ]
    if clustered.empty:
        logger.warning("DBSCAN found no business-POI clusters")
        return empty

    summary = clustered.groupby("label").agg(
        n_pois=("label", "size"), x=("x", "mean"), y=("y", "mean")
    )
    center = center_point(config).iloc[0]
    distance_to_center = np.hypot(summary["x"] - center.x, summary["y"] - center.y)
    summary = summary.drop(index=distance_to_center.idxmin())  # that cluster IS the centre
    summary = summary.sort_values("n_pois", ascending=False).head(
        config.features.subcenters.max_centers
    )
    if summary.empty:
        logger.info("only the centre cluster was found; no sub-centres")
        return empty

    result = gpd.GeoDataFrame(
        {
            "subcenter_id": range(1, len(summary) + 1),
            "n_pois": summary["n_pois"].to_numpy(),
        },
        geometry=gpd.points_from_xy(summary["x"], summary["y"], crs=config.crs.metric),
    )
    logger.info("sub-centres: %s", [int(n) for n in result["n_pois"]])
    return result.to_crs(config.crs.storage)


def centrality_features(
    grid: gpd.GeoDataFrame, subcenters: gpd.GeoDataFrame, config: Config
) -> pd.DataFrame:
    """Per-hex centrality block (``cent_`` columns), aligned to the grid.

    With no sub-centres found, the centre stands in for the nearest
    sub-centre (the monocentric degenerate case), keeping the column
    finite and the schema stable.
    """
    index = pd.Index(grid["h3"], name="h3")
    centroids = grid.set_index("h3").to_crs(config.crs.metric).geometry.centroid
    center = center_point(config).iloc[0]
    out = pd.DataFrame(index=index)
    out["cent_dist_center_m"] = centroids.distance(center).to_numpy()
    if subcenters.empty:
        out["cent_dist_subcenter_m"] = out["cent_dist_center_m"]
        return out
    points = subcenters.to_crs(config.crs.metric).geometry
    distances = np.column_stack([centroids.distance(point) for point in points])
    out["cent_dist_subcenter_m"] = distances.min(axis=1)
    return out
