"""Tests for boundary preparation: validity, slivers, districts, buffer."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, box

from almaty_access.config import Config
from almaty_access.io import read_geoparquet, write_geoparquet
from almaty_access.prepare.boundary import RAW_FILE, prepare_boundary


def _toy_raw_boundary(config: Config, *, drop_first_district: bool = False) -> gpd.GeoDataFrame:
    """8 district squares in a 4x2 grid + the city polygon covering them.

    District 0 gets a far-away sliver part (~800 m^2) that must be dropped.
    """
    rows = []
    for index, district in enumerate(config.districts):
        if drop_first_district and index == 0:
            continue
        x0 = 76.80 + (index % 4) * 0.02
        y0 = 43.20 + (index // 4) * 0.02
        geom = box(x0, y0, x0 + 0.02, y0 + 0.02)
        if index == 0:
            sliver = box(76.70, 43.10, 76.7003, 43.1003)
            geom = MultiPolygon([geom, sliver])
        rows.append(
            {
                "name": district.osm_name,
                "admin_level": "6",
                "level": "district",
                "geometry": geom,
            }
        )
    rows.append(
        {
            "name": "Almaty",
            "admin_level": "4",
            "level": "city",
            "geometry": box(76.80, 43.20, 76.88, 43.24),
        }
    )
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs=config.crs.storage)


def test_prepare_boundary_outputs(tmp_config: Config) -> None:
    write_geoparquet(_toy_raw_boundary(tmp_config), tmp_config.paths.raw / RAW_FILE)
    out = prepare_boundary(tmp_config)

    city = read_geoparquet(out["city"])
    districts = read_geoparquet(out["districts"])
    buffer = read_geoparquet(out["buffer"])

    assert len(city) == 1
    assert len(districts) == len(tmp_config.districts) == 8
    assert districts["area_km2"].gt(0).all()
    assert set(districts["district"]) == {d.key for d in tmp_config.districts}

    # The sliver part of district 0 was dropped -> a single polygon remains.
    first_key = tmp_config.districts[0].key
    geom = districts.loc[districts["district"] == first_key].geometry.iloc[0]
    n_parts = 1 if geom.geom_type == "Polygon" else len(geom.geoms)
    assert n_parts == 1

    # The buffer must strictly contain the city.
    metric = tmp_config.crs.metric
    city_area = city.to_crs(metric).geometry.area.iloc[0]
    buffer_area = buffer.to_crs(metric).geometry.area.iloc[0]
    assert buffer_area > city_area


def test_prepare_boundary_missing_district_raises(tmp_config: Config) -> None:
    raw = _toy_raw_boundary(tmp_config, drop_first_district=True)
    write_geoparquet(raw, tmp_config.paths.raw / RAW_FILE)
    with pytest.raises(ValueError, match="districts not found"):
        prepare_boundary(tmp_config)
