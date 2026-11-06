"""Offline tests for the download package.

No network: the PBF extractor runs against the committed sample
osm_extract.osm.pbf, the area helpers against toy boundary files, and
the Overpass endpoint rotation against stub operations.
"""

from __future__ import annotations

import dataclasses

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from almaty_access import schemas
from almaty_access.config import Config, PathsConfig
from almaty_access.download.area import study_bbox, study_inputs, study_polygon
from almaty_access.download.dem import tile_names
from almaty_access.download.osm import _flatten, _with_overpass_rotation, extract_osm_layers
from almaty_access.io import read_geoparquet, write_geoparquet
from almaty_access.prepare.boundary import RAW_FILE as BOUNDARY_RAW_FILE


def test_extractor_on_committed_sample_pbf(sample_config: Config, tmp_path) -> None:
    """The pyosmium extractor reproduces the three raw layers offline."""
    paths = PathsConfig(
        root=sample_config.paths.root,
        raw=tmp_path / "raw",
        interim=tmp_path / "interim",
        processed=tmp_path / "processed",
        sample=sample_config.paths.sample,
        figures=tmp_path / "figures",
        app=tmp_path / "app",
        webmap=tmp_path / "webmap",
    )
    config = dataclasses.replace(sample_config, paths=paths)
    config.paths.raw.mkdir(parents=True)
    # Reuse the committed PBF as this run's input.
    pbf_bytes = (sample_config.paths.raw / "osm_extract.osm.pbf").read_bytes()
    (config.paths.raw / "osm_extract.osm.pbf").write_bytes(pbf_bytes)

    outputs = extract_osm_layers(config)

    buildings = read_geoparquet(outputs["buildings"])
    pois = read_geoparquet(outputs["pois"])
    landuse = read_geoparquet(outputs["landuse"])
    assert len(buildings) > 500
    assert len(pois) > 1000
    assert len(landuse) > 100
    assert set(schemas.RAW_OSM_BUILDINGS) - {"geometry"} <= set(buildings.columns)
    assert set(schemas.RAW_OSM_POIS) - {"geometry"} <= set(pois.columns)
    # Everything extracted must sit in/near the sample bbox (padded filter).
    min_lon, min_lat, max_lon, max_lat = config.area.bbox
    bounds = pois.total_bounds
    assert bounds[0] > min_lon - 0.05 and bounds[2] < max_lon + 0.05
    assert bounds[1] > min_lat - 0.05 and bounds[3] < max_lat + 0.05
    assert buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()


def test_study_polygon_prefers_bbox(sample_config: Config) -> None:
    polygon = study_polygon(sample_config)
    assert polygon.bounds == pytest.approx(sample_config.area.bbox)
    assert study_inputs(sample_config) == []  # bbox mode needs no boundary file


def test_study_polygon_place_mode_buffers_city(tmp_config: Config) -> None:
    config = dataclasses.replace(tmp_config, area=dataclasses.replace(tmp_config.area, bbox=None))
    city = gpd.GeoDataFrame(
        pd.DataFrame(
            {
                "name": ["Almaty"],
                "name_ru": [None],
                "name_kk": [None],
                "admin_level": ["4"],
                "level": ["city"],
            }
        ),
        geometry=[box(76.85, 43.18, 77.0, 43.3)],
        crs=config.crs.storage,
    )
    write_geoparquet(city, config.paths.raw / BOUNDARY_RAW_FILE)

    polygon = study_polygon(config)
    bbox = study_bbox(config, pad_degrees=0.0)
    assert bbox[0] < 76.85 and bbox[2] > 77.0  # buffered beyond the city
    assert polygon.contains(box(76.85, 43.18, 77.0, 43.3))
    assert study_inputs(config) == [config.paths.raw / BOUNDARY_RAW_FILE]


def test_dem_tile_names_cover_bbox() -> None:
    names = tile_names((76.7, 43.0, 77.2, 43.5))
    assert names == [
        "Copernicus_DSM_COG_10_N43_00_E076_00_DEM",
        "Copernicus_DSM_COG_10_N43_00_E077_00_DEM",
    ]
    # Crossing a degree boundary pulls in the extra row of tiles.
    names = tile_names((76.7, 42.9, 77.2, 43.5))
    assert len(names) == 4


def test_flatten_joins_osm_lists() -> None:
    assert _flatten(["residential", "service"]) == "residential;service"
    assert _flatten("primary") == "primary"


def test_overpass_rotation_falls_through_and_raises() -> None:
    class FakeSettings:
        requests_timeout = 0
        overpass_url = ""
        overpass_rate_limit = False

    class FakeOx:
        settings = FakeSettings()

    calls: list[str] = []

    def flaky() -> str:
        calls.append(FakeOx.settings.overpass_url)
        if len(calls) < 2:
            raise TimeoutError("busy")
        return "ok"

    assert _with_overpass_rotation(FakeOx, "test op", flaky) == "ok"
    assert len(calls) == 2
    assert calls[0] != calls[1]  # a different endpoint was tried

    def always_fails() -> str:
        raise TimeoutError("down")

    with pytest.raises(RuntimeError, match="all Overpass endpoints failed"):
        _with_overpass_rotation(FakeOx, "test op", always_fails)
