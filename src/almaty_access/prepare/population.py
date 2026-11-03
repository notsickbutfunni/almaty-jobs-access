"""Population per hex: clean resolution 8, redistribute to resolution 9.

Kontur population comes natively as H3 resolution-8 hexes. Resolution 9
is derived with a small dasymetric step (README §7 Phase 1): each r8
hex's population is split over its 7 r9 children proportionally to the
**residential floor area** (GFA) in each child; children with no
residential GFA in a hex that has none anywhere get an equal split.
Totals are conserved exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from almaty_access.config import Config
from almaty_access.io import ensure_built, read_geoparquet, write_geoparquet
from almaty_access.prepare.boundary import BUFFER_FILE
from almaty_access.prepare.buildings import OUTPUT_FILE as BUILDINGS_FILE

logger = logging.getLogger(__name__)

RAW_FILE = "population_r8_raw.parquet"
R8_FILE = "population_r8.parquet"
R9_FILE = "population_r9.parquet"


def cell_polygon(cell: str) -> Polygon:
    """Shapely polygon (lon/lat order) for an H3 cell boundary."""
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(cell)])


def residential_gfa_by_cell(buildings: gpd.GeoDataFrame, resolution: int) -> pd.Series:
    """Sum of residential GFA per H3 cell (building assigned by centroid)."""
    residential = buildings[buildings["function"] == "residential"]
    if len(residential) == 0:
        return pd.Series(dtype=float)
    points = residential.geometry.representative_point()
    cells = [
        h3.latlng_to_cell(lat, lng, resolution)
        for lng, lat in zip(points.x.to_numpy(), points.y.to_numpy(), strict=True)
    ]
    return pd.Series(residential["gfa_m2"].to_numpy(), index=cells).groupby(level=0).sum()


def redistribute_to_children(
    population_r8: pd.DataFrame, gfa_by_r9: pd.Series, fine_resolution: int
) -> pd.DataFrame:
    """Split each r8 hex's population over its r9 children by residential GFA."""
    rows: list[tuple[str, float]] = []
    equal_split_hexes = 0
    for cell, population in zip(
        population_r8["h3"].to_numpy(), population_r8["population"].to_numpy(), strict=True
    ):
        children = list(h3.cell_to_children(cell, fine_resolution))
        weights = np.array([float(gfa_by_r9.get(child, 0.0)) for child in children])
        total = weights.sum()
        if total > 0:
            shares = weights / total
        else:
            equal_split_hexes += 1
            shares = np.full(len(children), 1.0 / len(children))
        rows.extend(
            (child, float(population) * share)
            for child, share in zip(children, shares, strict=True)
        )
    if equal_split_hexes:
        logger.info(
            "population r9: %d r8 hexes had no residential GFA, split equally",
            equal_split_hexes,
        )
    return pd.DataFrame(rows, columns=["h3", "population"])


def prepare_population(config: Config, *, force: bool = False) -> dict[str, Path]:
    """Clip r8 population to the study buffer and derive the r9 layer."""
    raw_path = config.paths.raw / RAW_FILE
    buildings_path = config.paths.interim / BUILDINGS_FILE
    buffer_path = config.paths.interim / BUFFER_FILE
    out_r8 = config.paths.interim / R8_FILE
    out_r9 = config.paths.interim / R9_FILE

    def build() -> None:
        boundary = read_geoparquet(buffer_path).geometry.iloc[0]
        raw = read_geoparquet(raw_path)
        buildings = read_geoparquet(buildings_path)

        r8 = raw[raw.geometry.intersects(boundary)].copy()
        r8["h3"] = r8["h3"].astype(str)
        r8["population"] = pd.to_numeric(r8["population"], errors="coerce").fillna(0.0)
        r8 = r8[["h3", "population", "geometry"]]
        logger.info("population r8: %d hexes, %.0f residents", len(r8), r8["population"].sum())

        gfa_by_r9 = residential_gfa_by_cell(buildings, config.h3.fine_resolution)
        r9_table = redistribute_to_children(
            pd.DataFrame(r8.drop(columns="geometry")), gfa_by_r9, config.h3.fine_resolution
        )
        if not np.isclose(r9_table["population"].sum(), r8["population"].sum()):
            raise RuntimeError("population redistribution did not conserve the total")
        r9 = gpd.GeoDataFrame(
            r9_table,
            geometry=[cell_polygon(cell) for cell in r9_table["h3"]],
            crs=config.crs.storage,
        )

        write_geoparquet(r8, out_r8)
        write_geoparquet(r9, out_r9)

    ensure_built(
        [out_r8, out_r9],
        [raw_path, buildings_path, buffer_path, config.source_path],
        build,
        force=force,
    )
    return {"r8": out_r8, "r9": out_r9}
