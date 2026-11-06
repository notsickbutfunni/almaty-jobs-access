"""End-to-end Phase 6 test on the committed sample (offline).

Runs the scenarios stage on top of the session-wide sample run and checks
the core guarantee (an unchanged network reproduces the stored Phase 4
baseline exactly), the output schemas, delta signs, an independent
recomputation of the gains, that the scenario Gini-before equals the
equity report's Gini, caching, orphan cleanup and the report contract.
Anything that rewrites outputs runs on a COPY of the shared directories.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from almaty_access import schemas
from almaty_access.access.build import (
    ACCESS_FILE_TEMPLATE,
    TT_FILE_TEMPLATE,
    grid_points,
    street_graph,
)
from almaty_access.access.matrix import hex_times
from almaty_access.access.speed import tobler_walk_kmh
from almaty_access.access.transit import build_transit_graph
from almaty_access.config import Config
from almaty_access.equity.build import SUMMARY_FILE as EQUITY_SUMMARY_FILE
from almaty_access.equity.build import build_equity
from almaty_access.features.build import FEATURES_FILE_TEMPLATE
from almaty_access.io import read_geoparquet
from almaty_access.prepare.transit import LEGS_FILE, ROUTES_FILE, STOPS_FILE
from almaty_access.scenarios.build import (
    GAINS_FILE,
    REPORT_FILE,
    SCENARIO_FILE_TEMPLATE,
    SUMMARY_FILE,
    affected_columns,
    build_scenarios,
)
from almaty_access.scenarios.definition import Scenario
from almaty_access.scenarios.transit import apply_transit_scenario

RASTER, CALIBRATED = schemas.EQUITY_WEIGHTINGS


@pytest.fixture(scope="module")
def scenaried(sample_pipeline: Config) -> Config:
    build_scenarios(sample_pipeline)
    return sample_pipeline


@pytest.fixture()
def scratch(scenaried: Config, tmp_path: Path) -> Config:
    """A copy of the processed outputs plus a private scenario directory."""
    shutil.copytree(scenaried.paths.processed, tmp_path / "processed")
    scen_dir = tmp_path / "scenarios"
    shutil.copytree(scenaried.scenarios.dir, scen_dir)
    paths = dataclasses.replace(scenaried.paths, processed=tmp_path / "processed")
    return dataclasses.replace(
        scenaried,
        paths=paths,
        scenarios=dataclasses.replace(scenaried.scenarios, dir=scen_dir),
    )


def _hex_table(config: Config, name: str = "demo_abay") -> pd.DataFrame:
    res = config.h3.analysis_resolution
    return pd.read_parquet(
        config.paths.processed / SCENARIO_FILE_TEMPLATE.format(name=name, res=res)
    )


def test_unchanged_network_reproduces_stored_baseline(sample_pipeline: Config) -> None:
    """The consistency guarantee behind every delta: run the scenario code
    path with NOTHING drawn and the stored Phase 4 transit matrix comes
    back bit for bit."""
    config = sample_pipeline
    noop = Scenario(
        name="noop",
        path=Path("noop.geojson"),
        title="",
        note="",
        sources=(),
        lines=(),
        boosts=(),
    )
    stops = read_geoparquet(config.paths.interim / STOPS_FILE)
    routes = pd.read_parquet(config.paths.interim / ROUTES_FILE)
    legs = pd.read_parquet(config.paths.interim / LEGS_FILE)
    applied = apply_transit_scenario(noop, stops, routes, legs, config=config)
    assert len(applied.stops) == len(stops)
    assert applied.routes.equals(routes)
    assert applied.legs.equals(legs)
    assert applied.speeds_kmh == config.access.transit.speeds_kmh
    assert applied.circuity == config.access.transit.circuity

    walk_flat = config.travel.speeds_kmh["walk"]
    walk = street_graph(config, "walk", lambda e: tobler_walk_kmh(e["grade"].to_numpy(), walk_flat))
    graph = build_transit_graph(
        walk,
        applied.stops,
        applied.routes,
        applied.legs,
        walk_flat_kmh=walk_flat,
        speeds_kmh=applied.speeds_kmh,
        circuity=applied.circuity,
        metric_crs=config.crs.metric,
    )
    grid = grid_points(config, config.h3.analysis_resolution)
    recomputed, _ = hex_times(
        graph,
        grid,
        grid,
        walk_flat_kmh=walk_flat,
        cutoff_min=float(max(config.travel.thresholds_min)),
        snap_max_m=config.access.snap_max_m,
    )
    stored = pd.read_parquet(
        config.paths.processed
        / TT_FILE_TEMPLATE.format(mode="transit", res=config.h3.analysis_resolution)
    )
    order = ["origin_h3", "dest_h3"]
    pd.testing.assert_frame_equal(
        recomputed.sort_values(order).reset_index(drop=True),
        stored.sort_values(order).reset_index(drop=True),
        check_exact=True,
    )


def test_outputs_exist_with_documented_schema(scenaried: Config) -> None:
    processed = scenaried.paths.processed
    res = scenaried.h3.analysis_resolution
    for name in (
        SCENARIO_FILE_TEMPLATE.format(name="demo_abay", res=res),
        SUMMARY_FILE,
        GAINS_FILE,
        REPORT_FILE,
    ):
        assert (processed / name).exists(), name

    table = _hex_table(scenaried)
    thresholds = scenaried.travel.thresholds_min
    expected = list(schemas.SCENARIO_ID_COLUMNS) + [
        f"{column}{suffix}"
        for mode in ("transit", "bike")
        for column in affected_columns(mode, thresholds)
        for suffix in schemas.SCENARIO_VALUE_SUFFIXES
    ]
    assert list(table.columns) == expected
    access = pd.read_parquet(processed / ACCESS_FILE_TEMPLATE.format(res=res))
    assert table["h3"].tolist() == access["h3"].tolist()

    summary = pd.read_parquet(processed / SUMMARY_FILE)
    assert list(summary.columns) == list(schemas.PROCESSED_SCENARIOS_SUMMARY)
    assert set(summary["weights"]) <= set(schemas.EQUITY_WEIGHTINGS)
    assert set(summary["scenario"]) == {"demo_abay"}

    gains = pd.read_parquet(processed / GAINS_FILE)
    assert list(gains.columns) == list(schemas.PROCESSED_SCENARIOS_GAINS)
    assert sorted(set(gains["cutoff_jobs"])) == sorted(scenaried.scenarios.gain_cutoffs_jobs)


def test_baselines_match_access_table_and_deltas_behave(scenaried: Config) -> None:
    table = _hex_table(scenaried)
    res = scenaried.h3.analysis_resolution
    access = pd.read_parquet(scenaried.paths.processed / ACCESS_FILE_TEMPLATE.format(res=res))
    thresholds = scenaried.travel.thresholds_min
    columns = [
        column for mode in ("transit", "bike") for column in affected_columns(mode, thresholds)
    ]
    some_positive = False
    for column in columns:
        np.testing.assert_array_equal(
            table[f"{column}_base"].to_numpy(), access[column].to_numpy(), err_msg=column
        )
        delta = table[f"{column}_delta"].to_numpy()
        assert (delta >= 0).all(), column
        np.testing.assert_allclose(
            delta, table[f"{column}_scen"].to_numpy() - table[f"{column}_base"].to_numpy()
        )
        some_positive = some_positive or bool((delta > 0).any())
    assert some_positive  # the demo line/corridor must actually change something


def test_gains_recomputed_independently(scenaried: Config) -> None:
    table = _hex_table(scenaried)
    res = scenaried.h3.analysis_resolution
    population = (
        pd.read_parquet(
            scenaried.paths.processed / FEATURES_FILE_TEMPLATE.format(res=res),
            columns=["h3", "pop_residents"],
        )
        .set_index("h3")["pop_residents"]
        .reindex(pd.Index(table["h3"]))
        .fillna(0.0)
        .to_numpy()
    )
    inhabited = table["in_city"].to_numpy(dtype=bool) & (
        population >= scenaried.equity.min_population
    )
    gains = pd.read_parquet(scenaried.paths.processed / GAINS_FILE)
    raster = gains[(gains["weights"] == RASTER) & (gains["scenario"] == "demo_abay")]
    assert len(raster) == len(scenaried.scenarios.gain_cutoffs_jobs)
    (measure,) = set(raster["measure"])
    delta = table[f"{measure}_delta"].to_numpy()
    for row in raster.itertuples():
        expected = float(population[inhabited & (delta > row.cutoff_jobs)].sum())
        assert row.residents_gaining == pytest.approx(expected)
        assert row.share_gaining == pytest.approx(expected / population[inhabited].sum())
    # More residents can clear a lower bar: gains fall as the cutoff grows.
    ordered = raster.sort_values("cutoff_jobs")["residents_gaining"].to_numpy()
    assert (np.diff(ordered) <= 0).all()


def test_gini_before_matches_equity_report(scenaried: Config) -> None:
    build_equity(scenaried)  # cached after the equity test module, cheap here
    equity = pd.read_parquet(scenaried.paths.processed / EQUITY_SUMMARY_FILE)
    summary = pd.read_parquet(scenaried.paths.processed / SUMMARY_FILE)
    res = scenaried.h3.analysis_resolution
    measure = "accpct_jobs_transit_45"
    for weighting in (RASTER, CALIBRATED):
        ours = summary[(summary["weights"] == weighting) & (summary["measure"] == measure)]
        theirs = equity[
            (equity["res"] == res)
            & (equity["weights"] == weighting)
            & (equity["measure"] == measure)
        ]
        assert len(ours) == 1 and len(theirs) == 1, weighting
        assert ours["gini_base"].iloc[0] == pytest.approx(theirs["gini"].iloc[0], rel=1e-9), (
            weighting
        )
        assert ours["population"].iloc[0] == pytest.approx(theirs["population"].iloc[0], rel=1e-9)


def test_report_contract(scenaried: Config) -> None:
    report = json.loads((scenaried.paths.processed / REPORT_FILE).read_text(encoding="utf-8"))
    assert report["approximate_transit"] is True
    assert set(report["scenarios"]) == {"demo_abay"}
    entry = report["scenarios"]["demo_abay"]
    assert entry["modes"] == ["transit", "bike"]
    assert entry["headline_measure"] == "accpct_jobs_transit_45"
    assert "within 45 min by transit" in entry["headline"]
    assert any("lower-bound" in caveat for caveat in entry["caveats"])
    assert any("lower-bound" in caveat for caveat in report["caveats"])

    network = entry["network"]
    assert network["transit"]["n_new_routes"] == 1
    assert network["transit"]["n_new_stops"] >= 2
    assert network["bike"]["edges_boosted"] > 0  # the corridor matched real edges
    assert entry["matrix"]["transit"]["pairs"] > 0
    assert set(entry["results"]) == {RASTER, CALIBRATED}
    gini = entry["results"][RASTER]["measures"]["accpct_jobs_transit_45"]["gini"]
    assert len(gini) == 2  # before, after

    weightings = report["weightings"]
    assert weightings[CALIBRATED].get("synthetic") is True  # sample stats are fake
    assert "factors" in weightings[CALIBRATED]


def test_caching_orphans_and_rename(scratch: Config) -> None:
    processed = scratch.paths.processed
    res = scratch.h3.analysis_resolution
    demo_out = processed / SCENARIO_FILE_TEMPLATE.format(name="demo_abay", res=res)

    build_scenarios(scratch)  # the copied geojson is newer: rebuilds once
    first = demo_out.stat().st_mtime_ns
    build_scenarios(scratch)  # now everything is fresh
    assert demo_out.stat().st_mtime_ns == first

    # Renaming a scenario replaces its per-hex file and cleans the orphan.
    source = scratch.scenarios.dir / "demo_abay.geojson"
    renamed = scratch.scenarios.dir / "demo_renamed.geojson"
    source.rename(renamed)
    build_scenarios(scratch)
    assert not demo_out.exists()
    assert (processed / SCENARIO_FILE_TEMPLATE.format(name="demo_renamed", res=res)).exists()
    report = json.loads((processed / REPORT_FILE).read_text(encoding="utf-8"))
    assert set(report["scenarios"]) == {"demo_renamed"}


def test_force_rebuild_is_deterministic(scratch: Config) -> None:
    before = _hex_table(scratch)
    build_scenarios(scratch, force=True)
    after = _hex_table(scratch)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
