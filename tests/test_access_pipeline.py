"""End-to-end Phase 4 test on the committed sample (offline).

Reads the session-wide sample run (prepare -> features -> model -> access,
see conftest.sample_pipeline) and checks schemas, invariants (threshold monotonicity, transit never
slower than walking, percentages bounded), the report contract and
caching. A second fixture empties the transit tables to pin down the
documented degraded path (no routes -> empty matrix, zero columns).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from almaty_access import schemas
from almaty_access.access.build import build_measures, build_times
from almaty_access.config import Config
from almaty_access.prepare.transit import LEGS_FILE, ROUTES_FILE, STOPS_FILE

MODES_R8 = ("walk", "bike", "transit", "car")


@pytest.fixture(scope="module")
def accessed(sample_pipeline: Config) -> Config:
    """The shared session-wide sample run (see conftest.sample_pipeline)."""
    return sample_pipeline


def _tt(config: Config, mode: str, res: int) -> pd.DataFrame:
    return pd.read_parquet(config.paths.processed / f"tt_{mode}_r{res}.parquet")


def test_transit_interim_schemas_and_headway_provenance(accessed: Config) -> None:
    """The three interim transit files honour their schema constants, and
    the OSM interval tag wins over the config assumption where present
    (the committed sample contains bus 12 with interval=16 min)."""
    stops = gpd.read_parquet(accessed.paths.interim / STOPS_FILE)
    routes = pd.read_parquet(accessed.paths.interim / ROUTES_FILE)
    legs = pd.read_parquet(accessed.paths.interim / LEGS_FILE)
    assert tuple(stops.columns) == schemas.INTERIM_TRANSIT_STOPS
    assert tuple(routes.columns) == schemas.INTERIM_TRANSIT_ROUTES
    assert tuple(legs.columns) == schemas.INTERIM_TRANSIT_LEGS
    assert set(routes["headway_source"]) <= {"interval_tag", "assumed"}
    tagged = routes.loc[routes["headway_source"] == "interval_tag"]
    assert (tagged["headway_min"] == 16.0).any()
    defaults = routes.loc[routes["headway_source"] == "assumed"]
    assert len(defaults) and (defaults["headway_min"] == 10.0).any()  # bus default


def test_matrix_schema_and_bounds(accessed: Config) -> None:
    cutoff = max(accessed.travel.thresholds_min)
    for mode, res in [(m, 8) for m in MODES_R8] + [("walk", 9)]:
        table = _tt(accessed, mode, res)
        assert list(table.columns) == list(schemas.PROCESSED_TT)
        assert len(table), f"tt_{mode}_r{res} is empty"
        assert (table["minutes"] >= 0).all()
        assert (table["minutes"] <= cutoff).all()
        assert not table.duplicated(["origin_h3", "dest_h3"]).any()


def test_transit_never_slower_than_walk(accessed: Config) -> None:
    """The transit graph contains the walk graph, so on every shared pair
    transit time <= walk time (it can always just keep walking)."""
    walk = _tt(accessed, "walk", 8).set_index(["origin_h3", "dest_h3"])["minutes"]
    transit = _tt(accessed, "transit", 8).set_index(["origin_h3", "dest_h3"])["minutes"]
    joined = pd.concat([walk, transit], axis=1, keys=["walk", "transit"]).dropna()
    assert len(joined)
    assert (joined["transit"] <= joined["walk"] + 0.05).all()


def test_access_tables_schema(accessed: Config) -> None:
    for res in (8, 9):
        table = pd.read_parquet(accessed.paths.processed / f"access_r{res}.parquet")
        assert list(table.columns[:4]) == list(schemas.ACCESS_ID_COLUMNS)
        assert table["h3"].is_unique
        assert (table["res"] == res).all()
        for column in table.columns[4:]:
            assert column.startswith(schemas.ACCESS_PREFIXES), column
        grid = pd.read_parquet(accessed.paths.processed / f"grid_r{res}.parquet")
        assert len(table) == len(grid)
    r9 = pd.read_parquet(accessed.paths.processed / "access_r9.parquet")
    assert not any(c.endswith(("_bike_30", "_car_30", "_transit_30")) for c in r9.columns)


def test_thresholds_monotone_and_pct_bounded(accessed: Config) -> None:
    table = pd.read_parquet(accessed.paths.processed / "access_r8.parquet")
    thresholds = accessed.travel.thresholds_min
    for mode in MODES_R8:
        for earlier, later in itertools.pairwise(thresholds):
            assert (
                table[f"acc_jobs_{mode}_{earlier}"] <= table[f"acc_jobs_{mode}_{later}"] + 1e-6
            ).all(), mode
        for threshold in thresholds:
            pct = table[f"accpct_jobs_{mode}_{threshold}"]
            assert ((pct >= 0) & (pct <= 100.0 + 1e-9)).all()
        assert (table[f"acc_jobs_{mode}_grav"] <= table[f"acc_jobs_{mode}_60"] + 1e-6).all()


def test_service_measures_present_and_sane(accessed: Config) -> None:
    table = pd.read_parquet(accessed.paths.processed / "access_r9.parquet")
    assert (table["acc_grocery_walk_30"] > 0).any()  # 81 grocery POIs in the core
    for column in (c for c in table.columns if c.startswith("sfca_")):
        assert (table[column] >= 0).all()
    near = [c for c in table.columns if c.startswith("near_")]
    assert sorted(near) == [
        "near_clinic_walk_min",
        "near_grocery_walk_min",
        "near_kindergarten_walk_min",
        "near_metro_walk_min",
        "near_park_walk_min",
        "near_school_walk_min",
    ]
    for column in near:
        values = table[column].dropna()
        assert len(values) and (values >= 0).all()
    # The sample core contains metro stations: someone is close to one.
    assert table["near_metro_walk_min"].min() < 15


def test_transit_routes_cnt(accessed: Config) -> None:
    table = pd.read_parquet(accessed.paths.processed / "access_r9.parquet")
    assert (table["transit_routes_cnt"] >= 0).all()
    assert (table["transit_routes_cnt"] > 0).any()  # one trolleybus survives the bbox


def test_report_contract(accessed: Config) -> None:
    report = json.loads((accessed.paths.processed / "access_report.json").read_text("utf-8"))
    assert report["approximate_transit"] is True
    assert report["params"]["transit_assumed_headways_min"]["bus"] == 10.0
    assert report["transit_network"]["routes_kept"] >= 1
    assert report["total_jobs"]["r8"] > 0
    assert 0 <= report["coverage"]["r8"]["walk"]["share_reaching_jobs_30"] <= 1
    assert report["transit_network"]["routes_kept_headway_from_tag"] >= 1
    assert len(report["spot_checks"]) == 10
    for check in report["spot_checks"]:
        assert set(check["minutes"]) == set(MODES_R8)
    # Stratified sampling (D-051): the walk-matrix half guarantees pairs
    # the slow modes can actually be checked on.
    with_walk = sum(check["minutes"]["walk"] is not None for check in report["spot_checks"])
    assert with_walk >= 5
    sensitivity = report["gravity_sensitivity"]["spearman_vs"]
    assert set(sensitivity) == {"half_life_10", "half_life_30"}


def test_access_build_is_cached(accessed: Config) -> None:
    report_path = accessed.paths.processed / "access_report.json"
    before = report_path.stat().st_mtime_ns
    build_times(accessed)
    build_measures(accessed)
    assert report_path.stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# Degraded path: an area with no usable transit routes.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def no_transit(accessed: Config, tmp_path_factory: pytest.TempPathFactory) -> Config:
    """The same world with the transit tables emptied."""
    tmp = tmp_path_factory.mktemp("no_transit")
    shutil.copytree(accessed.paths.interim, tmp / "interim")
    shutil.copytree(accessed.paths.processed, tmp / "processed")
    config = dataclasses.replace(
        accessed,
        paths=dataclasses.replace(
            accessed.paths, interim=tmp / "interim", processed=tmp / "processed"
        ),
    )
    stops = gpd.read_parquet(config.paths.interim / STOPS_FILE)
    routes = pd.read_parquet(config.paths.interim / ROUTES_FILE)
    legs = pd.read_parquet(config.paths.interim / LEGS_FILE)
    stops.iloc[:0].to_parquet(config.paths.interim / STOPS_FILE, index=False)
    routes.iloc[:0].to_parquet(config.paths.interim / ROUTES_FILE, index=False)
    legs.iloc[:0].to_parquet(config.paths.interim / LEGS_FILE, index=False)
    build_times(config, force=True)
    build_measures(config, force=True)
    return config


def test_no_transit_degrades_explicitly(no_transit: Config) -> None:
    table = _tt(no_transit, "transit", 8)
    assert len(table) == 0
    stats = json.loads((no_transit.paths.processed / "tt_stats.json").read_text("utf-8"))
    assert stats["tt_transit_r8.parquet"]["skipped_reason"]
    access = pd.read_parquet(no_transit.paths.processed / "access_r8.parquet")
    assert (access["acc_jobs_transit_30"] == 0).all()
    assert (access["transit_routes_cnt"] == 0).all()
    # The other modes are untouched.
    assert (access["acc_jobs_walk_30"] > 0).any()
    assert np.isfinite(access["acc_jobs_car_60"]).all()
