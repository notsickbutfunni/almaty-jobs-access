"""Opportunity layers: what accessibility counts (README §7 Phase 4.3).

Six opportunity types: jobs (the Phase 3 surface — the central ``jobs_rf``
estimate), schools, kindergartens, clinics (clinics + hospitals), grocery
shops and parks over the configured minimum area. Metro stations join as
a nearest-facility target only. POIs come from the categorized interim
layer; parks from the raw land-use polygons (their representative point
stands in as the entrance, an approximation noted in DECISIONS).
"""

from __future__ import annotations

import geopandas as gpd
import h3
import numpy as np
import pandas as pd

from almaty_access.config import Config
from almaty_access.io import read_geoparquet

#: Cumulative / 2SFCA service types -> interim POI categories.
OPPORTUNITY_POI_CATEGORIES: dict[str, tuple[str, ...]] = {
    "schools": ("education_school",),
    "kindergartens": ("education_kindergarten",),
    "clinics": ("health_clinic", "health_hospital"),
    "grocery": ("grocery",),
}

#: Service types measured with cumulative counts (walk mode).
SERVICE_TYPES = ("schools", "kindergartens", "clinics", "grocery", "parks")

#: Capacity-constrained services that get the 2SFCA treatment.
SFCA_TYPES = ("schools", "kindergartens", "clinics")

#: Nearest-facility targets (singular names used in column names).
NEAREST_TYPES: dict[str, str] = {
    "school": "schools",
    "kindergarten": "kindergartens",
    "clinic": "clinics",
    "grocery": "grocery",
    "park": "parks",
    "metro": "metro",
}


def load_opportunity_points(config: Config) -> dict[str, gpd.GeoDataFrame]:
    """Point layers (EPSG:4326) per opportunity type, plus metro stations."""
    pois = read_geoparquet(config.paths.interim / "pois.parquet")
    points: dict[str, gpd.GeoDataFrame] = {}
    for kind, categories in OPPORTUNITY_POI_CATEGORIES.items():
        points[kind] = pois.loc[pois["category"].isin(categories), ["geometry"]].reset_index(
            drop=True
        )

    landuse = read_geoparquet(config.paths.raw / "osm_landuse.parquet")
    parks = landuse.loc[landuse["leisure"] == "park"]
    if len(parks):
        metric = parks.to_crs(config.crs.metric)
        large = metric.loc[metric.geometry.area >= config.access.park_min_m2]
        centers = large.geometry.representative_point().to_crs(config.crs.storage)
        points["parks"] = gpd.GeoDataFrame(geometry=centers.reset_index(drop=True))
    else:
        points["parks"] = gpd.GeoDataFrame(geometry=[], crs=config.crs.storage)

    metro = pois.loc[(pois["railway"] == "station") & (pois["station"] == "subway")]
    points["metro"] = metro[["geometry"]].reset_index(drop=True)
    return points


def _hex_of_points(points: gpd.GeoDataFrame, res: int) -> pd.Series:
    geometry = points.geometry
    return pd.Series(
        [h3.latlng_to_cell(lat, lon, res) for lat, lon in zip(geometry.y, geometry.x, strict=True)],
        dtype="string",
    )


def opportunity_table(
    config: Config,
    points: dict[str, gpd.GeoDataFrame],
    res: int,
    grid_h3: pd.Series,
) -> pd.DataFrame:
    """Per-hex opportunity counts aligned to the grid (missing -> 0).

    Columns: ``jobs`` (Phase 3 central estimate) and a count per service
    type. Indexed by ``h3``.
    """
    jobs = pd.read_parquet(
        config.paths.processed / f"jobs_r{res}.parquet", columns=["h3", "jobs_rf"]
    )
    table = pd.DataFrame(index=pd.Index(grid_h3, name="h3"))
    table["jobs"] = jobs.set_index("h3")["jobs_rf"].reindex(table.index).fillna(0.0)
    for kind in SERVICE_TYPES:
        cells = _hex_of_points(points[kind], res)
        counts = cells.value_counts()
        table[kind] = counts.reindex(table.index).fillna(0).astype("float64")
    return table


def metric_points_xy(points: gpd.GeoDataFrame, metric_crs: str) -> np.ndarray:
    """Facility coordinates in the metric CRS, shape (n, 2)."""
    if not len(points):
        return np.empty((0, 2))
    metric = points.geometry.to_crs(metric_crs)
    return np.column_stack([metric.x.to_numpy(), metric.y.to_numpy()])
