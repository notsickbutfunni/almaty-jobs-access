"""Terrain features: mean elevation and slope per hex.

Every valid raster pixel is assigned to its H3 cell (pixel centre →
``latlng_to_cell``) and aggregated — a zonal mean without rasterizing
hex polygons, exact to half a pixel (~15 m, far below hex size). The
integer H3 API is used because 3.7 M string cell ids would be slow and
heavy; keys become strings only after the groupby.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pyproj
import rasterio
from h3.api import basic_int as h3_int

logger = logging.getLogger(__name__)


def _pixel_cells(
    path: Path, res: int, storage_crs: str
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """(valid pixel values, their H3 cells as ints, raster shape)."""
    with rasterio.open(path) as src:
        values = src.read(1, masked=True).filled(np.nan)
        transform = src.transform
        crs = src.crs
    valid = np.isfinite(values)
    rows, cols = np.nonzero(valid)
    # Pixel centres via the affine transform (offset 0.5), vectorized.
    xs = transform.c + transform.a * (cols + 0.5) + transform.b * (rows + 0.5)
    ys = transform.f + transform.d * (cols + 0.5) + transform.e * (rows + 0.5)
    to_storage = pyproj.Transformer.from_crs(crs, storage_crs, always_xy=True)
    lons, lats = to_storage.transform(xs, ys)
    cells = np.fromiter(
        (h3_int.latlng_to_cell(lat, lon, res) for lat, lon in zip(lats, lons, strict=True)),
        dtype=np.uint64,
        count=len(lats),
    )
    return values[valid], cells, values.shape


def terrain_features(
    elevation_path: Path,
    slope_path: Path,
    grid: gpd.GeoDataFrame,
    res: int,
    storage_crs: str,
    metric_crs: str,
) -> pd.DataFrame:
    """Per-hex terrain block (``ter_`` columns), aligned to the grid.

    Hexes without any valid pixel (possible at the raster edge) borrow
    the value of the nearest hex that has one — logged, never NaN.
    """
    index = pd.Index(grid["h3"], name="h3")
    elevation, elevation_cells, shape_e = _pixel_cells(elevation_path, res, storage_crs)
    slope, slope_cells, shape_s = _pixel_cells(slope_path, res, storage_crs)
    if shape_e != shape_s:
        raise ValueError(
            f"elevation raster {shape_e} and slope raster {shape_s} shapes differ; "
            "re-run `almaty-access prepare --only dem`"
        )

    by_elevation = pd.Series(elevation).groupby(elevation_cells).mean()
    slope_series = pd.Series(slope).groupby(slope_cells)
    stats = pd.DataFrame(
        {
            "ter_elev_mean_m": by_elevation,
            "ter_slope_mean_deg": slope_series.mean(),
            "ter_slope_max_deg": slope_series.max(),
        }
    )
    stats.index = pd.Index([h3.int_to_str(cell) for cell in stats.index], name="h3")
    out = stats.reindex(index)

    # Any missing column counts: the slope raster's valid mask is smaller
    # than the DEM's (the gradient eats an edge ring), so a hex can have
    # elevation but no slope pixels — it must borrow too.
    missing = out.isna().any(axis=1)
    if missing.any():
        logger.info("terrain: %d hexes without pixels borrow the nearest value", missing.sum())
        centroids = grid.set_index("h3").to_crs(metric_crs).geometry.centroid
        have = gpd.GeoDataFrame(out.loc[~missing], geometry=centroids.loc[~missing])
        need = gpd.GeoDataFrame(geometry=centroids.loc[missing])
        borrowed = gpd.sjoin_nearest(need, have)
        borrowed = borrowed[~borrowed.index.duplicated()]
        for column in out.columns:
            # fillna, not assignment: a hex missing only slope keeps its
            # own valid elevation and borrows just the missing column.
            out.loc[missing, column] = out.loc[missing, column].fillna(borrowed[column])
    return out
