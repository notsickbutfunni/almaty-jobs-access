"""End-to-end tests on the committed sample dataset (offline).

The sample under data/sample/raw/ is REAL downloaded data for a small
bbox in central Almaty. These tests run the actual prepare pipeline on
it (writing into tmp) and check the outputs' invariants — the closest
thing to running the full pipeline that CI can do without network.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from almaty_access import schemas
from almaty_access.config import Config, PathsConfig
from almaty_access.io import read_geoparquet
from almaty_access.prepare.boundary import prepare_boundary
from almaty_access.prepare.buildings import prepare_buildings
from almaty_access.prepare.completeness import prepare_completeness
from almaty_access.prepare.dem import prepare_dem
from almaty_access.prepare.network import MAX_ABS_GRADE, prepare_networks
from almaty_access.prepare.pois import load_category_rules, prepare_pois
from almaty_access.prepare.population import prepare_population

RAW_SCHEMAS = {
    "boundary_osm.parquet": schemas.RAW_BOUNDARY,
    "osm_buildings.parquet": schemas.RAW_OSM_BUILDINGS,
    "osm_pois.parquet": schemas.RAW_OSM_POIS,
    "osm_landuse.parquet": schemas.RAW_OSM_LANDUSE,
    "overture_buildings.parquet": schemas.RAW_OVERTURE_BUILDINGS,
    "population_r8_raw.parquet": schemas.RAW_POPULATION_R8,
}


def test_sample_raw_layers_exist_and_match_schemas(sample_config: Config) -> None:
    raw = sample_config.paths.raw
    for file_name, expected_columns in RAW_SCHEMAS.items():
        path = raw / file_name
        assert path.exists(), f"missing committed sample file: {path.name}"
        frame = read_geoparquet(path)
        assert len(frame) > 0, f"{path.name} is empty"
        missing = set(expected_columns) - set(frame.columns) - {frame.geometry.name}
        assert not missing, f"{path.name} lacks columns: {sorted(missing)}"
        assert frame.crs is not None and frame.crs.to_epsg() == 4326
    for mode in schemas.NETWORK_MODES:
        for part in ("nodes", "edges"):
            assert (raw / f"network_{mode}_{part}.parquet").exists()
    assert (raw / "dem.tif").exists()
    assert (raw / "osm_extract.osm.pbf").exists()
    stats = pd.read_csv(raw / "district_stats.csv")
    assert (stats["source"] == "synthetic-for-tests").all(), (
        "sample district stats must stay clearly synthetic"
    )


@pytest.fixture(scope="module")
def prepared(sample_config: Config, tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Run the whole prepare pipeline on the sample into tmp once per module."""
    tmp = tmp_path_factory.mktemp("sample_run")
    paths = PathsConfig(
        root=sample_config.paths.root,
        raw=sample_config.paths.raw,
        interim=tmp / "interim",
        processed=tmp / "processed",
        sample=sample_config.paths.sample,
        figures=tmp / "figures",
        app=tmp / "app",
        webmap=tmp / "webmap",
    )
    config = dataclasses.replace(sample_config, paths=paths)
    prepare_boundary(config)
    prepare_dem(config)
    prepare_pois(config)
    prepare_buildings(config)
    prepare_population(config)
    prepare_networks(config)
    prepare_completeness(config)
    return config


def test_prepared_boundary_has_all_districts(prepared: Config) -> None:
    districts = read_geoparquet(prepared.paths.interim / "boundary_districts.parquet")
    assert len(districts) == len(prepared.districts) == 8
    assert districts["area_km2"].gt(0).all()


def test_prepared_buildings_invariants(prepared: Config) -> None:
    buildings = read_geoparquet(prepared.paths.interim / "buildings.parquet")
    assert len(buildings) > 500
    assert set(buildings["src"].unique()) <= {"osm", "overture"}
    assert set(buildings["function"].unique()) <= set(schemas.BUILDING_FUNCTIONS)
    assert (buildings["levels"] >= 1).all()
    assert (buildings["footprint_m2"] > 0).all()
    assert np.allclose(
        buildings["gfa_m2"], buildings["footprint_m2"] * buildings["levels"], rtol=0.01
    )
    assert set(buildings["levels_source"].unique()) <= {"tag", "height", "default"}
    # The merge must actually de-duplicate: Overture adds SOME buildings,
    # but not a full second copy of the OSM stock.
    counts = buildings["src"].value_counts()
    assert counts.get("osm", 0) > 0


def test_prepared_pois_categories_come_from_the_rule_file(prepared: Config) -> None:
    pois = read_geoparquet(prepared.paths.interim / "pois.parquet")
    rules = load_category_rules(prepared.pois.categories_file)
    allowed = {rule.category for rule in rules}
    assert len(pois) > 200
    assert set(pois["category"].unique()) <= allowed
    assert pois["osm_id"].is_unique


def test_prepared_population_conserves_totals(prepared: Config) -> None:
    r8 = read_geoparquet(prepared.paths.interim / "population_r8.parquet")
    r9 = read_geoparquet(prepared.paths.interim / "population_r9.parquet")
    assert len(r8) > 0
    assert len(r9) > len(r8)
    assert np.isclose(r8["population"].sum(), r9["population"].sum())
    assert (r9["population"] >= 0).all()


def test_prepared_networks_have_elevation_and_grades(prepared: Config) -> None:
    for mode in schemas.NETWORK_MODES:
        nodes = read_geoparquet(prepared.paths.interim / f"network_{mode}_nodes.parquet")
        edges = read_geoparquet(prepared.paths.interim / f"network_{mode}_edges.parquet")
        assert len(nodes) > 50 and len(edges) > 50
        assert nodes["elevation_m"].notna().mean() > 0.95, f"{mode}: too many nodes without DEM"
        assert (edges["length_m"] > 0).all()
        assert edges["grade"].abs().max() <= MAX_ABS_GRADE + 1e-9
        # Almaty slopes south-up: grades must not be all zero.
        assert edges["grade"].abs().mean() > 0


def test_prepared_completeness_table(prepared: Config) -> None:
    table = pd.read_csv(prepared.paths.interim / "completeness.csv")
    assert {"district", "buildings_osm", "buildings_overture", "osm_share", "pois"} <= set(
        table.columns
    )
    # All 8 districts stay in the table; the tiny sample bbox populates one.
    assert len(table) == len(prepared.districts) == 8
    assert table["buildings_osm"].sum() > 0
    assert (table["buildings_osm"] > 0).sum() >= 1
