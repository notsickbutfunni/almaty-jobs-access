"""Tests for the study grid (cells, domain, district assignment)."""

from __future__ import annotations

import dataclasses

import geopandas as gpd
import h3
import pandas as pd
import pytest
from shapely.geometry import box

from almaty_access.config import AreaConfig, Config
from almaty_access.features.grid import assign_districts, study_cells, study_domain
from almaty_access.prepare.population import cell_polygon

CRS = "EPSG:4326"
METRIC = "EPSG:32643"

CENTER_CELL = h3.latlng_to_cell(43.24, 76.93, 8)


def test_study_cells_centroid_containment() -> None:
    polygon = cell_polygon(CENTER_CELL)
    cells = study_cells(polygon, 8)
    assert CENTER_CELL in cells
    for cell in cells:
        lat, lon = h3.cell_to_latlng(cell)
        assert polygon.buffer(1e-9).contains(gpd.points_from_xy([lon], [lat])[0])


def test_study_cells_finer_resolution_has_more_cells() -> None:
    polygon = cell_polygon(CENTER_CELL)
    assert len(study_cells(polygon, 9)) > len(study_cells(polygon, 8))


def test_study_domain_without_bbox_is_the_buffer(sample_config: Config) -> None:
    buffer_polygon = box(76.8, 43.1, 77.1, 43.4)
    config = dataclasses.replace(
        sample_config, area=AreaConfig(place=sample_config.area.place, bbox=None)
    )
    assert study_domain(config, buffer_polygon).equals(buffer_polygon)


def test_study_domain_intersects_bbox(sample_config: Config) -> None:
    buffer_polygon = box(76.8, 43.1, 77.1, 43.4)
    domain = study_domain(sample_config, buffer_polygon)  # sample config has a bbox
    assert domain.within(buffer_polygon.buffer(1e-12))
    assert domain.within(box(*sample_config.area.bbox).buffer(1e-12))


def test_study_domain_disjoint_bbox_raises(sample_config: Config) -> None:
    buffer_polygon = box(0, 0, 1, 1)  # nowhere near the sample bbox
    with pytest.raises(ValueError, match="does not intersect"):
        study_domain(sample_config, buffer_polygon)


def test_assign_districts_largest_overlap_wins() -> None:
    hex_polygon = cell_polygon(CENTER_CELL)
    far_cell = h3.latlng_to_cell(43.40, 77.05, 8)
    grid = gpd.GeoDataFrame(
        {"h3": pd.Series([CENTER_CELL, far_cell], dtype="string")},
        geometry=[hex_polygon, cell_polygon(far_cell)],
        crs=CRS,
    )
    min_x, min_y, max_x, max_y = hex_polygon.bounds
    split_x = min_x + 0.65 * (max_x - min_x)
    districts = gpd.GeoDataFrame(
        {"district": ["west", "east"]},
        geometry=[
            box(min_x - 0.01, min_y - 0.01, split_x, max_y + 0.01),
            box(split_x, min_y - 0.01, max_x + 0.01, max_y + 0.01),
        ],
        crs=CRS,
    )

    result = assign_districts(grid, districts, METRIC)

    assert result.loc[0, "district"] == "west"
    assert 0.5 < result.loc[0, "district_share"] < 0.9
    # The far-away hex touches no district: NA, share 0.
    assert pd.isna(result.loc[1, "district"])
    assert result.loc[1, "district_share"] == 0.0
