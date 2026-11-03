"""Tests for the building merge, function classification and floor logic."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

from almaty_access.prepare.buildings import (
    classify_function,
    estimate_levels,
    merge_osm_overture,
    parse_number,
)

CRS = "EPSG:4326"
METRIC = "EPSG:32643"


def _square(lon: float, lat: float, size: float = 0.0002):
    return box(lon, lat, lon + size, lat + size)


def _osm_frame(geoms, **overrides) -> gpd.GeoDataFrame:
    n = len(geoms)
    data = {
        "osm_id": [f"w{i}" for i in range(n)],
        "building": ["yes"] * n,
        "building_levels": [None] * n,
        "height": [None] * n,
        "name": [None] * n,
        "amenity": [None] * n,
        "shop": [None] * n,
        "office": [None] * n,
    }
    data.update(overrides)
    return gpd.GeoDataFrame(pd.DataFrame(data), geometry=list(geoms), crs=CRS)


def _overture_frame(geoms, **overrides) -> gpd.GeoDataFrame:
    n = len(geoms)
    data = {
        "overture_id": [f"o{i}" for i in range(n)],
        "subtype": [None] * n,
        "class": [None] * n,
        "height": [None] * n,
        "num_floors": [None] * n,
    }
    data.update(overrides)
    return gpd.GeoDataFrame(pd.DataFrame(data), geometry=list(geoms), crs=CRS)


def test_parse_number() -> None:
    assert parse_number("3") == 3.0
    assert parse_number("2,5") == 2.5
    assert parse_number("12 m") == 12.0
    assert parse_number(4) == 4.0
    assert np.isnan(parse_number("yes"))
    assert np.isnan(parse_number(None))


def test_merge_drops_high_iou_overture_keeps_osm() -> None:
    osm = _osm_frame([_square(76.90, 43.24)], building=["apartments"])
    overture = _overture_frame(
        [_square(76.90, 43.24), _square(76.92, 43.24)],  # identical + disjoint
        subtype=["residential", "commercial"],
    )
    merged = merge_osm_overture(osm, overture, iou_threshold=0.5, metric_crs=METRIC)
    assert len(merged) == 2  # 1 OSM + 1 non-duplicate Overture
    assert merged["src"].tolist() == ["osm", "overture"]
    assert merged.loc[merged["src"] == "osm", "building_tag"].iloc[0] == "apartments"
    assert merged.loc[merged["src"] == "overture", "building_tag"].iloc[0] == "commercial"


def test_merge_keeps_partial_overlap() -> None:
    # Half-overlapping square: IoU = 0.5A / 1.5A = 1/3 < 0.5 -> kept.
    osm = _osm_frame([_square(76.90, 43.24)])
    overture = _overture_frame([_square(76.9001, 43.24)])
    merged = merge_osm_overture(osm, overture, iou_threshold=0.5, metric_crs=METRIC)
    assert len(merged) == 2


def test_merge_normalizes_overture_attributes() -> None:
    osm = _osm_frame([_square(76.90, 43.24)])
    overture = _overture_frame(
        [_square(76.92, 43.24), _square(76.94, 43.24)],
        subtype=[None, "residential"],
        **{"class": ["warehouse", "apartments"]},
        height=[9.0, None],
        num_floors=[None, 5],
    )
    merged = merge_osm_overture(osm, overture, iou_threshold=0.5, metric_crs=METRIC)
    ovt = merged[merged["src"] == "overture"].reset_index(drop=True)
    assert ovt.loc[0, "building_tag"] == "warehouse"  # class fills missing subtype
    assert ovt.loc[1, "building_tag"] == "residential"
    assert ovt.loc[0, "height_num"] == 9.0
    assert ovt.loc[1, "levels_num"] == 5.0


def test_merge_best_iou_across_multiple_osm_neighbors() -> None:
    # One Overture footprint intersecting TWO OSM squares: a sliver overlap
    # with one (low IoU) and near-identity with the other (high IoU). The
    # max IoU decides, so it must be dropped.
    osm = _osm_frame([_square(76.90, 43.24), _square(76.9002, 43.24)])
    overture = _overture_frame([_square(76.9002, 43.24)])
    merged = merge_osm_overture(osm, overture, iou_threshold=0.5, metric_crs=METRIC)
    assert (merged["src"] == "overture").sum() == 0


def test_classify_function_priority_ladder() -> None:
    geoms = [_square(76.90 + i * 0.001, 43.24) for i in range(6)]
    buildings = gpd.GeoDataFrame(
        pd.DataFrame(
            {
                "building_tag": ["apartments", "warehouse", "yes", None, None, None],
                "amenity": [None, None, None, None, None, None],
                "shop": [None, None, "bakery", None, None, None],
                "office": [None] * 6,
            }
        ),
        geometry=geoms,
        crs=CRS,
    )

    # One food POI inside building 3, two public POIs inside building 4.
    def center(i: int) -> Point:
        return Point(76.90 + i * 0.001 + 0.0001, 43.2401)

    pois = gpd.GeoDataFrame(
        {"category": ["food_service", "education_school", "health_clinic"]},
        geometry=[center(3), center(4), center(4)],
        crs=CRS,
    )
    # Building 5 sits on residential landuse.
    landuse = gpd.GeoDataFrame(
        {"landuse": ["residential"]},
        geometry=[box(76.9049, 43.239, 76.906, 43.241)],
        crs=CRS,
    )
    function = classify_function(buildings, pois, landuse)
    assert function.tolist() == [
        "residential",  # own tag
        "industrial",  # own tag
        "commercial",  # own shop tag
        "commercial",  # food POI inside
        "public",  # public POIs win
        "residential",  # landuse fallback
    ]


def test_classify_poi_vote_tie_rules() -> None:
    """Any industrial POI wins; public must STRICTLY beat commercial."""
    geoms = [_square(76.90, 43.24), _square(76.902, 43.24)]
    buildings = gpd.GeoDataFrame(
        pd.DataFrame(
            {
                "building_tag": [None, None],
                "amenity": [None, None],
                "shop": [None, None],
                "office": [None, None],
            }
        ),
        geometry=geoms,
        crs=CRS,
    )
    pois = gpd.GeoDataFrame(
        {
            "category": [
                "industrial",  # building 0: industrial beats the others
                "education_school",
                "office",
                "education_school",  # building 1: 1 public vs 1 commercial -> tie
                "office",
            ]
        },
        geometry=[
            Point(76.9001, 43.2401),
            Point(76.90005, 43.24005),
            Point(76.90015, 43.24015),
            Point(76.9021, 43.2401),
            Point(76.90205, 43.24005),
        ],
        crs=CRS,
    )
    empty_landuse = gpd.GeoDataFrame({"landuse": []}, geometry=[], crs=CRS)
    function = classify_function(buildings, pois, empty_landuse)
    assert function.tolist() == ["industrial", "commercial"]


def test_classify_function_unknown_when_no_signal() -> None:
    buildings = gpd.GeoDataFrame(
        pd.DataFrame({"building_tag": [None], "amenity": [None], "shop": [None], "office": [None]}),
        geometry=[_square(76.90, 43.24)],
        crs=CRS,
    )
    empty_pois = gpd.GeoDataFrame({"category": []}, geometry=[], crs=CRS)
    empty_landuse = gpd.GeoDataFrame({"landuse": []}, geometry=[], crs=CRS)
    function = classify_function(buildings, empty_pois, empty_landuse)
    assert function.tolist() == ["unknown"]


def test_estimate_levels_sources_and_floor() -> None:
    buildings = pd.DataFrame(
        {
            "levels_num": [3.0, np.nan, np.nan, np.nan],
            "height_num": [np.nan, 15.0, np.nan, 1.5],
        }
    )
    function = pd.Series(["commercial", "commercial", "residential", "industrial"])
    levels, source = estimate_levels(
        buildings,
        function,
        m_per_level=3.0,
        default_levels={"residential": 2.0, "commercial": 2.0, "industrial": 1.0},
    )
    assert levels.tolist() == [3.0, 5.0, 2.0, 1.0]  # 1.5 m / 3 -> 0.5 -> clipped to 1
    assert source.tolist() == ["tag", "height", "default", "height"]
