"""Tests for node elevations and edge grades."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from almaty_access.prepare.network import MAX_ABS_GRADE, attach_elevation, edge_grades

METRIC = "EPSG:32643"


def test_edge_grades_directed_and_clipped() -> None:
    nodes = gpd.GeoDataFrame(
        {"node_id": [1, 2, 3], "elevation_m": [100.0, 110.0, 200.0]},
        geometry=[Point(0, 0), Point(0, 1), Point(0, 2)],
        crs="EPSG:4326",
    )
    edges = pd.DataFrame(
        {
            "u": [1, 2, 1, 1],
            "v": [2, 1, 3, 1],
            "length_m": [100.0, 100.0, 50.0, 0.0],
        }
    )
    grade = edge_grades(edges, nodes)
    assert grade[0] == 0.1  # uphill
    assert grade[1] == -0.1  # downhill
    assert grade[2] == MAX_ABS_GRADE  # 100 m rise over 50 m -> clipped
    assert grade[3] == 0.0  # zero length guarded


def test_attach_elevation_samples_dem(tmp_path: Path) -> None:
    # A constant 555 m raster in the metric CRS around a point in Almaty.
    node = gpd.GeoDataFrame({"node_id": [1]}, geometry=[Point(76.90, 43.24)], crs="EPSG:4326")
    center = node.to_crs(METRIC).geometry.iloc[0]
    transform = from_origin(center.x - 500, center.y + 500, 10, 10)
    dem_path = tmp_path / "dem.tif"
    profile = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
        "count": 1,
        "dtype": "float32",
        "crs": METRIC,
        "transform": transform,
    }
    with rasterio.open(dem_path, "w", **profile) as dst:
        dst.write(np.full((100, 100), 555.0, dtype="float32"), 1)

    result = attach_elevation(node, dem_path, METRIC)
    assert result["elevation_m"].iloc[0] == 555.0


def test_attach_elevation_outside_dem_is_nan(tmp_path: Path) -> None:
    node = gpd.GeoDataFrame({"node_id": [1]}, geometry=[Point(60.0, 50.0)], crs="EPSG:4326")
    dem_path = tmp_path / "dem.tif"
    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "float32",
        "crs": METRIC,
        "transform": from_origin(700000, 4790000, 10, 10),
        "nodata": -9999.0,
    }
    with rasterio.open(dem_path, "w", **profile) as dst:
        dst.write(np.full((10, 10), 500.0, dtype="float32"), 1)
    result = attach_elevation(node, dem_path, METRIC)
    assert np.isnan(result["elevation_m"].iloc[0])
