"""Tests for the r8 -> r9 dasymetric population redistribution."""

from __future__ import annotations

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import box

from almaty_access.prepare.population import (
    cell_polygon,
    redistribute_to_children,
    residential_gfa_by_cell,
)

PARENT = h3.latlng_to_cell(43.24, 76.90, 8)


def test_cell_polygon_is_lon_lat() -> None:
    polygon = cell_polygon(PARENT)
    centroid = polygon.centroid
    assert abs(centroid.x - 76.90) < 0.02  # x must be longitude
    assert abs(centroid.y - 43.24) < 0.02


def test_residential_gfa_by_cell_counts_only_residential() -> None:
    lat, lng = h3.cell_to_latlng(PARENT)
    square = box(lng, lat, lng + 0.0002, lat + 0.0002)
    buildings = gpd.GeoDataFrame(
        {"function": ["residential", "residential", "commercial"], "gfa_m2": [100.0, 50.0, 999.0]},
        geometry=[square, square, square],
        crs="EPSG:4326",
    )
    gfa = residential_gfa_by_cell(buildings, 9)
    assert len(gfa) == 1
    assert gfa.iloc[0] == 150.0


def test_redistribute_proportional_and_conserving() -> None:
    children = list(h3.cell_to_children(PARENT, 9))
    gfa = pd.Series({children[0]: 300.0, children[1]: 100.0})
    r8 = pd.DataFrame({"h3": [PARENT], "population": [400.0]})
    r9 = redistribute_to_children(r8, gfa, 9)
    assert len(r9) == len(children)
    by_cell = r9.set_index("h3")["population"]
    assert by_cell[children[0]] == 300.0
    assert by_cell[children[1]] == 100.0
    assert by_cell[children[2]] == 0.0
    assert np.isclose(r9["population"].sum(), 400.0)


def test_redistribute_equal_split_without_gfa() -> None:
    children = list(h3.cell_to_children(PARENT, 9))
    r8 = pd.DataFrame({"h3": [PARENT], "population": [70.0]})
    r9 = redistribute_to_children(r8, pd.Series(dtype=float), 9)
    assert np.allclose(r9["population"], 70.0 / len(children))
    assert np.isclose(r9["population"].sum(), 70.0)
