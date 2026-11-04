"""The study grid: H3 hexes covering the buffered boundary.

A hex belongs to the grid when its centroid lies inside the buffered
city boundary (intersected with ``area.bbox`` when one is set — the
sample dataset covers only its bbox, so its grid must too). Each hex is
assigned to the district with the largest area overlap; hexes in the
buffer ring overlap no district and get ``district = NA`` — they exist
so that k-ring context and, later, accessibility are not cut off at the
city edge (README §4).
"""

from __future__ import annotations

import logging

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from almaty_access.config import Config
from almaty_access.prepare.population import cell_polygon

logger = logging.getLogger(__name__)

GRID_FILE_TEMPLATE = "grid_r{res}.parquet"


def study_domain(config: Config, buffer_polygon: BaseGeometry) -> BaseGeometry:
    """The polygon the grid must cover (buffered city ∩ optional bbox)."""
    if config.area.bbox is None:
        return buffer_polygon
    domain = buffer_polygon.intersection(shapely.box(*config.area.bbox))
    if domain.is_empty:
        raise ValueError("config error: 'area.bbox' does not intersect the buffered boundary")
    return domain


def study_cells(domain: BaseGeometry, res: int) -> list[str]:
    """H3 cells (as strings) whose centroid falls inside ``domain``."""
    shape = h3.geo_to_h3shape(domain.__geo_interface__)
    cells = h3.h3shape_to_cells(shape, res)  # centroid containment is the default
    if not cells:
        raise ValueError(f"study area contains no H3 cells at resolution {res}")
    return sorted(cells)


def assign_districts(
    grid: gpd.GeoDataFrame, districts: gpd.GeoDataFrame, metric_crs: str
) -> pd.DataFrame:
    """Largest-overlap district per hex (NA when a hex touches none).

    Returns a frame aligned to ``grid.index`` with columns ``district``
    (string) and ``district_share`` — the assigned district's share of
    the hex area, so border hexes are visibly split (README asks for an
    area-weighted assignment).
    """
    hexes = grid.to_crs(metric_crs)
    polys = districts.to_crs(metric_crs)
    pairs = gpd.sjoin(
        hexes[["geometry"]], polys[["district", "geometry"]], predicate="intersects"
    ).rename(columns={"index_right": "district_index"})
    if pairs.empty:
        return pd.DataFrame(
            {
                "district": pd.Series(pd.NA, index=grid.index, dtype="string"),
                "district_share": 0.0,
            }
        )
    overlap = shapely.area(
        shapely.intersection(
            pairs.geometry.to_numpy(), polys.geometry.loc[pairs["district_index"]].to_numpy()
        )
    )
    pairs = pd.DataFrame(
        {
            "district": pairs["district"].to_numpy(),
            "overlap_m2": overlap,
            "hex_m2": pairs.geometry.area.to_numpy(),
        },
        index=pairs.index,
    )
    best = (
        pairs.sort_values(["overlap_m2", "district"], ascending=[False, True])
        .groupby(level=0)
        .head(1)
    )
    result = pd.DataFrame(
        {
            "district": pd.Series(best["district"], dtype="string"),
            # clip: hex ⊂ district gives area ratios like 1.0000000000000002
            "district_share": (best["overlap_m2"] / best["hex_m2"]).clip(upper=1.0),
        }
    ).reindex(grid.index)
    result["district_share"] = result["district_share"].fillna(0.0)
    return result


def build_grid(
    config: Config,
    res: int,
    buffer_polygon: BaseGeometry,
    districts: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """The full study grid for one resolution (schemas.PROCESSED_GRID)."""
    domain = study_domain(config, buffer_polygon)
    cells = study_cells(domain, res)
    grid = gpd.GeoDataFrame(
        {"h3": pd.Series(cells, dtype="string"), "res": np.int16(res)},
        geometry=[cell_polygon(cell) for cell in cells],
        crs=config.crs.storage,
    )
    assignment = assign_districts(grid, districts, config.crs.metric)
    grid["district"] = assignment["district"]
    grid["district_share"] = assignment["district_share"]
    grid["in_city"] = grid["district"].notna()
    logger.info(
        "grid r%d: %d cells, %d in a district, %d buffer-ring",
        res,
        len(grid),
        int(grid["in_city"].sum()),
        int((~grid["in_city"]).sum()),
    )
    return grid[["h3", "res", "district", "district_share", "in_city", "geometry"]]
