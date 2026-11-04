"""End-to-end Phase 2 test: prepare + features on the committed sample.

Runs the real prepare pipeline and the real feature build into tmp and
checks the contracts later phases rely on: stable schema across
resolutions, no NaN/inf, shares in [0, 1], positive distances, and a
report that explains every transform and drop.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from almaty_access import schemas
from almaty_access.config import Config, PathsConfig
from almaty_access.features.build import build_features
from almaty_access.io import read_geoparquet
from almaty_access.prepare.boundary import prepare_boundary
from almaty_access.prepare.buildings import prepare_buildings
from almaty_access.prepare.dem import prepare_dem
from almaty_access.prepare.network import prepare_networks
from almaty_access.prepare.pois import prepare_pois
from almaty_access.prepare.population import prepare_population


@pytest.fixture(scope="module")
def built(sample_config: Config, tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Run prepare + features on the sample into tmp once per module."""
    tmp = tmp_path_factory.mktemp("features_run")
    paths = PathsConfig(
        root=sample_config.paths.root,
        raw=sample_config.paths.raw,
        interim=tmp / "interim",
        processed=tmp / "processed",
        sample=sample_config.paths.sample,
        figures=tmp / "figures",
    )
    config = dataclasses.replace(sample_config, paths=paths)
    prepare_boundary(config)
    prepare_dem(config)
    prepare_pois(config)
    prepare_buildings(config)
    prepare_population(config)
    prepare_networks(config)
    build_features(config)
    return config


def _load(config: Config, res: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = read_geoparquet(config.paths.processed / f"grid_r{res}.parquet")
    features = pd.read_parquet(config.paths.processed / f"features_r{res}.parquet")
    return grid, features


def test_all_outputs_exist(built: Config) -> None:
    processed = built.paths.processed
    for name in (
        "grid_r8.parquet",
        "grid_r9.parquet",
        "features_r8.parquet",
        "features_r9.parquet",
        "subcenters.parquet",
        "features_report.json",
    ):
        assert (processed / name).exists(), f"missing output: {name}"


def test_grid_schema_and_district_assignment(built: Config) -> None:
    valid_districts = {district.key for district in built.districts}
    for res in (8, 9):
        grid, _ = _load(built, res)
        assert list(grid.columns) == list(schemas.PROCESSED_GRID)
        assert grid["h3"].is_unique
        assert (grid["res"] == res).all()
        assigned = grid.loc[grid["in_city"]]
        # The sample bbox lies inside the city, so every cell is assigned.
        assert len(assigned) == len(grid)
        assert set(assigned["district"]) <= valid_districts
        assert ((assigned["district_share"] > 0) & (assigned["district_share"] <= 1)).all()


def test_feature_tables_share_schema_and_are_finite(built: Config) -> None:
    _, features_r8 = _load(built, 8)
    _, features_r9 = _load(built, 9)
    assert list(features_r8.columns) == list(features_r9.columns)
    assert len(features_r9) > len(features_r8)

    feature_columns = [
        column for column in features_r8.columns if column not in schemas.FEATURE_ID_COLUMNS
    ]
    assert len(feature_columns) >= 50, "README §7 Phase 2 asks for ~50-100 features"
    for frame in (features_r8, features_r9):
        values = frame[feature_columns].to_numpy(dtype=float)
        assert np.isfinite(values).all(), "feature tables must be NaN- and inf-free"
        prefixes = tuple(schemas.FEATURE_PREFIXES)
        assert all(column.startswith(prefixes) for column in feature_columns)


def test_features_align_with_grid(built: Config) -> None:
    for res in (8, 9):
        grid, features = _load(built, res)
        assert set(features["h3"]) == set(grid["h3"])
        assert features["h3"].is_unique


def test_feature_value_sanity(built: Config) -> None:
    _, features = _load(built, 9)
    share_columns = [c for c in features.columns if "_share_" in c or c.startswith("lu_share")]
    for column in share_columns:
        assert features[column].between(0, 1).all(), f"{column} out of [0, 1]"
    for column in (
        "tr_dist_stop_m",
        "tr_dist_metro_m",
        "net_dist_major_m",
        "cent_dist_center_m",
        "cent_dist_subcenter_m",
    ):
        if column in features.columns:  # pruning may drop near-duplicates
            assert (features[column] >= 0).all()
    gfa_columns = [c for c in features.columns if c.startswith("bf_gfa_")]
    assert features[gfa_columns].to_numpy().min() >= 0
    # The sample covers a dense central area: it must contain real mass.
    assert features.filter(like="poi_cnt").to_numpy().sum() > 0
    if "pop_residents" in features.columns:
        assert features["pop_residents"].sum() > 0
    # Land-use precedence: a square metre belongs to at most one class.
    lu_columns = [c for c in features.columns if c.startswith("lu_share_")]
    assert (features[lu_columns].sum(axis=1) <= 1.001).all()
    # The sample bbox holds no motorway/trunk/primary roads, so
    # net_dist_major_m must be the documented fallback: one constant,
    # clearly positive value (a regression returning 0 would claim
    # every hex sits on a major road).
    assert features["net_dist_major_m"].nunique() == 1
    assert features["net_dist_major_m"].iloc[0] > 1000


def test_built_mass_matches_building_layer(built: Config) -> None:
    buildings = read_geoparquet(built.paths.interim / "buildings.parquet")
    _, features = _load(built, 9)
    gfa_by_function = [f"bf_gfa_{f}_m2" for f in schemas.BUILDING_FUNCTIONS]
    table_total = features[[c for c in gfa_by_function if c in features.columns]].sum().sum()
    layer_total = buildings["gfa_m2"].sum()
    # Footprint parts outside the grid are genuinely dropped, nothing
    # more. The tiny sample has a high edge-to-area ratio (extraction
    # keeps buildings whose FIRST vertex is in the bbox, so many poke
    # out of the 28-cell grid), hence the loose lower bound.
    assert table_total <= layer_total * 1.001
    assert table_total >= layer_total * 0.5


def test_report_documents_transforms_and_pruning(built: Config) -> None:
    report = json.loads((built.paths.processed / "features_report.json").read_text("utf-8"))
    assert report["resolutions"] == [8, 9]
    assert report["n_cells"]["r9"] > report["n_cells"]["r8"] > 0
    # The sample grid is tiny, so pruning must be skipped and say why.
    assert report["pruning_skipped_reason"]
    assert report["near_duplicates_dropped"] == []
    assert report["high_corr_kept"] == []
    _, features = _load(built, 8)
    assert set(report["log1p_columns"]) <= set(features.columns)
    assert set(report["protected_features"]) <= set(features.columns)
    # log1p targets counts and areas only — never shares, floor
    # statistics or ratios (review finding: over-broad prefixes).
    for column in report["log1p_columns"]:
        assert "share" not in column and "levels" not in column and "per_1k" not in column
    assert report["random_seed"] == built.random_seed


def test_subcenters_schema(built: Config) -> None:
    subcenters = read_geoparquet(built.paths.processed / "subcenters.parquet")
    expected = [c for c in schemas.PROCESSED_SUBCENTERS if c != "geometry"]
    assert [c for c in subcenters.columns if c != "geometry"] == expected


def test_feature_build_is_cached(built: Config) -> None:
    report_path = built.paths.processed / "features_report.json"
    before = report_path.stat().st_mtime_ns
    build_features(built)  # second call: everything fresh, must not rebuild
    assert report_path.stat().st_mtime_ns == before
