"""End-to-end Phase 3 test on the committed sample (offline).

The sample grid covers one district, so the pipeline must take the
documented degraded path: baseline rescaled to the synthetic official
total, random forest skipped with a recorded reason, rf columns
carrying the baseline — schema identical to the full-city run.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from almaty_access import schemas
from almaty_access.config import Config, DistrictConfig, PathsConfig, RandomForestConfig
from almaty_access.model.build import build_model
from almaty_access.model.target import load_district_targets


@pytest.fixture(scope="module")
def modeled(sample_pipeline: Config) -> Config:
    """The shared session-wide sample run (see conftest.sample_pipeline)."""
    return sample_pipeline


def test_outputs_exist_with_schema(modeled: Config) -> None:
    for res in (8, 9):
        jobs = pd.read_parquet(modeled.paths.processed / f"jobs_r{res}.parquet")
        assert list(jobs.columns) == list(schemas.PROCESSED_JOBS)
        assert jobs["h3"].is_unique
        assert (jobs["res"] == res).all()
    assert (modeled.paths.processed / "model_report.json").exists()


def test_totals_conserved_against_official(modeled: Config) -> None:
    targets = load_district_targets(modeled)
    report = json.loads((modeled.paths.processed / "model_report.json").read_text("utf-8"))
    year = report["target_year"]
    for res in (8, 9):
        jobs = pd.read_parquet(modeled.paths.processed / f"jobs_r{res}.parquet")
        for district in report["districts_modeled"][f"r{res}"]:
            official = float(
                targets.loc[
                    (targets["district"] == district) & (targets["year"] == year), "jobs"
                ].iloc[0]
            )
            mask = jobs["district"].eq(district).fillna(False)
            assert jobs.loc[mask, "jobs_baseline"].sum() == pytest.approx(official, rel=1e-6)
            assert jobs.loc[mask, "jobs_rf"].sum() == pytest.approx(official, rel=1e-6)


def test_sample_takes_the_documented_degraded_path(modeled: Config) -> None:
    report = json.loads((modeled.paths.processed / "model_report.json").read_text("utf-8"))
    # District sets are per resolution: the bbox clips almaly at r8,
    # but a bostandyk border sliver holds in-city r9 hexes.
    assert report["districts_modeled"]["r8"] == ["almaly"]
    assert "almaly" in report["districts_modeled"]["r9"]
    assert len(report["districts_without_hexes"]["r8"]) == 7
    assert report["rf_skipped_reason"]  # 1 district at the analysis res
    assert report["lodo"] == []
    assert report["rf_beats_baseline"] is None
    jobs = pd.read_parquet(modeled.paths.processed / "jobs_r9.parquet")
    # With the forest skipped the rf columns carry the baseline.
    pd.testing.assert_series_equal(jobs["jobs_rf"], jobs["jobs_baseline"], check_names=False)


def test_value_sanity(modeled: Config) -> None:
    for res in (8, 9):
        jobs = pd.read_parquet(modeled.paths.processed / f"jobs_r{res}.parquet")
        values = jobs[["jobs_baseline", "jobs_rf", "jobs_rf_p10", "jobs_rf_p90"]]
        assert (values >= 0).all().all()
        assert (jobs["jobs_rf_p10"] <= jobs["jobs_rf_p90"] + 1e-9).all()
        # Buffer hexes carry no official jobs.
        buffer_rows = jobs.loc[~jobs["in_city"]]
        if len(buffer_rows):
            assert (buffer_rows[values.columns] == 0).all().all()


def test_model_build_is_cached(modeled: Config) -> None:
    report_path = modeled.paths.processed / "model_report.json"
    before = report_path.stat().st_mtime_ns
    build_model(modeled)
    assert report_path.stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# Synthetic multi-district world: the full (non-degraded) forest path.
# The sample covers one district, so without this fixture a city-only
# crash would ship green (review finding). d5 exists ONLY at r9 — the
# exact edge the per-resolution district fix addresses.
# ---------------------------------------------------------------------------


def _synthetic_features(res: int, districts: list[str], hexes_per_district: int) -> pd.DataFrame:
    rng = np.random.default_rng(42 + res)
    rows = []
    for d_index, district in enumerate(districts):
        for h_index in range(hexes_per_district):
            gradient = (d_index + 1) * 1000.0
            commercial = gradient * (1 + 0.3 * rng.random())
            residential = 2000.0 + 500 * rng.random()
            rows.append(
                {
                    "h3": f"r{res}-{district}-{h_index}",
                    "res": res,
                    "district": district,
                    "in_city": True,
                    "bf_gfa_residential_m2": residential,
                    "bf_gfa_commercial_m2": commercial,
                    "bf_gfa_industrial_m2": gradient * 0.2,
                    "bf_gfa_public_m2": 300.0,
                    "bf_gfa_unknown_m2": 100.0,
                    "bf_gfa_m2": residential + commercial + gradient * 0.2 + 400.0,
                    "poi_cnt_total": float(d_index * 3 + h_index % 3),
                    "pop_residents": 500.0 + 100 * rng.random(),
                    "ter_slope_mean_deg": 2.0 + rng.random(),
                }
            )
    frame = pd.DataFrame(rows)
    frame["district"] = frame["district"].astype("string")
    return frame


@pytest.fixture(scope="module")
def synthetic_city(sample_config: Config, tmp_path_factory: pytest.TempPathFactory) -> Config:
    tmp = tmp_path_factory.mktemp("synthetic_city")
    districts = [f"d{i}" for i in range(6)]
    config = dataclasses.replace(
        sample_config,
        districts=tuple(DistrictConfig(key=key, osm_name=key) for key in districts),
        paths=PathsConfig(
            root=tmp,
            raw=tmp / "raw",
            interim=tmp / "interim",
            processed=tmp / "processed",
            sample=tmp / "sample",
            figures=tmp / "figures",
        ),
        model=dataclasses.replace(
            sample_config.model,
            target_year=2024,
            training_years=(2023, 2024),
            random_forest=RandomForestConfig(
                n_estimators=100, min_samples_leaf=2, max_features=0.5
            ),
        ),
    )

    # r8 lacks d5 entirely; r9 has all six districts.
    features_r8 = _synthetic_features(8, districts[:5], hexes_per_district=4)
    features_r9 = _synthetic_features(9, districts, hexes_per_district=6)
    # One completely empty (but in-city) r9 hex: the mask must zero it.
    empty = features_r9.index[0]
    zero_columns = [c for c in features_r9.columns if c.startswith(("bf_", "poi_", "pop_"))]
    features_r9.loc[empty, zero_columns] = 0.0
    # Two buffer hexes with no district.
    buffer_rows = features_r9.iloc[:2].copy()
    buffer_rows["h3"] = ["r9-buffer-0", "r9-buffer-1"]
    buffer_rows["district"] = pd.Series([pd.NA, pd.NA], dtype="string").to_numpy()
    buffer_rows["in_city"] = False
    features_r9 = pd.concat([features_r9, buffer_rows], ignore_index=True)

    config.paths.processed.mkdir(parents=True)
    features_r8.to_parquet(config.paths.processed / "features_r8.parquet", index=False)
    features_r9.to_parquet(config.paths.processed / "features_r9.parquet", index=False)

    rows = []
    for d_index, district in enumerate(districts):
        for year in (2023, 2024):
            for quarter in (3, 6, 9, 12):
                jobs = 10_000.0 * (d_index + 1) * (1.02 if year == 2024 else 1.0)
                rows.append((district, year, quarter, "payroll_jobs_located", jobs, "persons"))
    stats = pd.DataFrame(rows, columns=["district", "year", "month", "indicator", "value", "unit"])
    stats["source"] = "synthetic-for-tests"
    stats["notes"] = ""
    config.paths.raw.mkdir(parents=True)
    stats.to_csv(config.paths.raw / "district_stats.csv", index=False)

    build_model(config)
    return config


def test_synthetic_city_runs_the_full_forest_path(synthetic_city: Config) -> None:
    report = json.loads((synthetic_city.paths.processed / "model_report.json").read_text("utf-8"))
    assert report["rf_skipped_reason"] == ""
    assert report["districts_modeled"]["r8"] == [f"d{i}" for i in range(5)]
    assert report["districts_modeled"]["r9"] == [f"d{i}" for i in range(6)]
    assert report["districts_without_hexes"]["r8"] == ["d5"]
    assert report["districts_without_hexes"]["r9"] == []
    assert len(report["lodo"]) == 5
    assert report["permutation_importance_top15"]


def test_synthetic_city_conserves_totals_per_resolution(synthetic_city: Config) -> None:
    report = json.loads((synthetic_city.paths.processed / "model_report.json").read_text("utf-8"))
    for res in (8, 9):
        jobs = pd.read_parquet(synthetic_city.paths.processed / f"jobs_r{res}.parquet")
        for district in report["districts_modeled"][f"r{res}"]:
            mask = jobs["district"].eq(district).fillna(False)
            expected = 10_000.0 * (int(district[1]) + 1) * 1.02  # 2024 quarters are flat
            assert jobs.loc[mask, "jobs_rf"].sum() == pytest.approx(expected, rel=1e-6)
            assert jobs.loc[mask, "jobs_baseline"].sum() == pytest.approx(expected, rel=1e-6)


def test_synthetic_city_masks_empty_hex_and_zeroes_buffer(synthetic_city: Config) -> None:
    jobs = pd.read_parquet(synthetic_city.paths.processed / "jobs_r9.parquet")
    indexed = jobs.set_index("h3")
    assert indexed.loc["r9-d0-0", "jobs_rf"] == 0.0  # emptied hex, masked
    assert indexed.loc["r9-d0-0", "jobs_baseline"] == 0.0  # no GFA either
    for h3_id in ("r9-buffer-0", "r9-buffer-1"):
        assert (
            indexed.loc[h3_id, ["jobs_baseline", "jobs_rf", "jobs_rf_p10", "jobs_rf_p90"]] == 0
        ).all()
    assert (jobs["jobs_rf_p10"] <= jobs["jobs_rf_p90"] + 1e-9).all()
