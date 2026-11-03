"""Tests for the OSM-vs-Overture completeness table."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point, box

from almaty_access.prepare.completeness import WEAK_OSM_SHARE, completeness_table

CRS = "EPSG:4326"


def test_completeness_counts_shares_and_flags() -> None:
    districts = gpd.GeoDataFrame(
        {"district": ["a", "b", "empty"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs=CRS,
    )
    buildings = gpd.GeoDataFrame(
        {"src": ["osm", "osm", "osm", "overture", "osm", "overture", "overture", "overture"]},
        geometry=[
            Point(0.2, 0.5),
            Point(0.4, 0.5),
            Point(0.6, 0.5),
            Point(0.8, 0.5),  # district a: 3 osm, 1 overture
            Point(1.2, 0.5),
            Point(1.4, 0.5),
            Point(1.6, 0.5),
            Point(1.8, 0.5),  # district b: 1 osm, 3 overture
        ],
        crs=CRS,
    )
    pois = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5), Point(0.6, 0.6)], crs=CRS)

    table = completeness_table(buildings, pois, districts).set_index("district")

    assert table.loc["a", "buildings_osm"] == 3
    assert table.loc["a", "buildings_overture"] == 1
    assert table.loc["a", "osm_share"] == 0.75
    assert table.loc["a", "pois"] == 2
    assert not table.loc["a", "weak_osm_coverage"]

    assert table.loc["b", "osm_share"] == 0.25
    assert table.loc["b", "osm_share"] < WEAK_OSM_SHARE
    assert table.loc["b", "weak_osm_coverage"]
    assert table.loc["b", "pois"] == 0

    # A district with no buildings at all must stay in the table, flagged.
    assert table.loc["empty", "buildings_osm"] == 0
    assert table.loc["empty", "weak_osm_coverage"]
