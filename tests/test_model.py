"""Unit tests for the Phase 3 job model on constructed inputs."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from almaty_access.config import Config
from almaty_access.model.aggregate import apply_log1p, column_kind, district_design, hex_design
from almaty_access.model.baseline import baseline_weights, rescale_to_totals
from almaty_access.model.build import _rescaled_tree_surfaces
from almaty_access.model.forest import fit_forest, lodo_table, training_table, tree_weights
from almaty_access.model.target import load_district_targets, resolve_years


def _write_stats(path, rows) -> None:
    frame = pd.DataFrame(rows, columns=["district", "year", "month", "indicator", "value", "unit"])
    frame["source"] = "synthetic-for-tests"
    frame["notes"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


@pytest.fixture()
def stats_config(sample_config: Config, tmp_path) -> Config:
    paths = dataclasses.replace(sample_config.paths, raw=tmp_path / "raw")
    return dataclasses.replace(sample_config, paths=paths)


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------


def test_targets_average_quarters_and_resolve_years(stats_config: Config) -> None:
    rows = []
    for district in stats_config.districts:
        for quarter, value in ((3, 90.0), (6, 110.0), (9, 100.0), (12, 100.0)):
            rows.append((district.key, 2024, quarter, "payroll_jobs_located", value, "persons"))
    _write_stats(stats_config.paths.raw / "district_stats.csv", rows)

    targets = load_district_targets(stats_config)
    assert len(targets) == 8
    assert (targets["jobs"] == 100.0).all()  # mean of the four quarters

    target_year, training_years = resolve_years(targets, stats_config)
    # The config asks for 2024 and the data has it.
    assert target_year == 2024 and training_years == [2024]


def test_targets_fall_back_to_latest_complete_year(stats_config: Config) -> None:
    rows = []
    for district in stats_config.districts:
        rows.append((district.key, 2020, 12, "payroll_jobs_located", 50.0, "persons"))
    # 2023 exists but only for one district -> incomplete, unusable.
    rows.append(("almaly", 2023, 12, "payroll_jobs_located", 60.0, "persons"))
    _write_stats(stats_config.paths.raw / "district_stats.csv", rows)

    targets = load_district_targets(stats_config)
    target_year, training_years = resolve_years(targets, stats_config)
    assert target_year == 2020
    assert training_years == [2020]


def test_resolve_years_excludes_partially_reported_years(stats_config: Config) -> None:
    """A Q1-only current year must never masquerade as an annual mean."""
    rows = []
    for district in stats_config.districts:
        for quarter in (3, 6, 9, 12):
            rows.append((district.key, 2024, quarter, "payroll_jobs_located", 100.0, "persons"))
        rows.append((district.key, 2025, 3, "payroll_jobs_located", 120.0, "persons"))
    _write_stats(stats_config.paths.raw / "district_stats.csv", rows)

    config = dataclasses.replace(
        stats_config, model=dataclasses.replace(stats_config.model, target_year=2025)
    )
    targets = load_district_targets(config)
    target_year, training_years = resolve_years(targets, config)
    # 2025 covers every district but only one quarter -> excluded.
    assert target_year == 2024
    assert training_years == [2024]


def test_targets_missing_district_raises(stats_config: Config) -> None:
    rows = [("almaly", 2024, 12, "payroll_jobs_located", 100.0, "persons")]
    _write_stats(stats_config.paths.raw / "district_stats.csv", rows)
    with pytest.raises(ValueError, match="without target data"):
        load_district_targets(stats_config)


def test_targets_missing_indicator_raises(stats_config: Config) -> None:
    rows = [
        (d.key, 2024, 12, "employed_residents", 100.0, "persons") for d in stats_config.districts
    ]
    _write_stats(stats_config.paths.raw / "district_stats.csv", rows)
    with pytest.raises(ValueError, match="payroll_jobs_located"):
        load_district_targets(stats_config)


# --------------------------------------------------------------------------
# aggregation units
# --------------------------------------------------------------------------


def test_column_kind_rules() -> None:
    assert column_kind("bf_gfa_commercial_m2") == "mass"
    assert column_kind("poi_cnt_total") == "mass"
    assert column_kind("net_len_major_m") == "mass"
    assert column_kind("pop_residents") == "mass"
    assert column_kind("ctx_k1_sum_bf_gfa_m2") == "mass"
    assert column_kind("bf_share_large_gfa") == "mean"
    assert column_kind("bf_levels_mean") == "mean"
    assert column_kind("poi_entropy") == "mean"
    assert column_kind("poi_density_km2") == "mean"
    assert column_kind("tr_dist_metro_m") == "mean"
    assert column_kind("net_betweenness_max") == "mean"
    assert column_kind("cent_dist_center_m") == "mean"
    assert column_kind("ter_slope_mean_deg") == "mean"
    assert column_kind("pop_per_1k_nonres_gfa") == "mean"
    assert column_kind("ctx_k2_mean_poi_cnt_total") == "mean"
    with pytest.raises(ValueError, match="no aggregation rule"):
        column_kind("mystery_feature")


def _toy_features() -> pd.DataFrame:
    """Four r8 hexes: two in district x, one in y, one buffer hex."""
    return pd.DataFrame(
        {
            "h3": ["a", "b", "c", "d"],
            "res": [8, 8, 8, 8],
            "district": pd.Series(["x", "x", "y", pd.NA], dtype="string"),
            "in_city": [True, True, True, False],
            "bf_gfa_m2": [1000.0, 3000.0, 2000.0, 500.0],
            "ter_slope_mean_deg": [2.0, 4.0, 6.0, 8.0],
        }
    )


def test_hex_and_district_design_share_units(sample_config: Config) -> None:
    import h3 as h3lib

    features = _toy_features()
    hexes = hex_design(features)
    districts = district_design(
        features,
        dataclasses.replace(
            sample_config,
            districts=tuple(
                dataclasses.replace(d, key=k)
                for d, k in zip(sample_config.districts[:2], ["x", "y"], strict=False)
            ),
        ),
    )
    area = h3lib.average_hexagon_area(8, unit="km^2")
    # Mass becomes density at BOTH levels; the buffer hex is excluded.
    assert list(hexes.index) == ["a", "b", "c"]
    assert hexes.loc["a", "bf_gfa_m2"] == pytest.approx(1000.0 / area)
    assert districts.loc["x", "bf_gfa_m2"] == pytest.approx(4000.0 / (2 * area))
    # Intensities pass through / average.
    assert hexes.loc["c", "ter_slope_mean_deg"] == 6.0
    assert districts.loc["x", "ter_slope_mean_deg"] == pytest.approx(3.0)
    assert districts.loc["x", "area_km2"] == pytest.approx(2 * area)


def test_apply_log1p_touches_only_prefixed_columns() -> None:
    frame = pd.DataFrame({"bf_gfa_m2": [10.0], "ter_slope_mean_deg": [5.0], "area_km2": [2.0]})
    out = apply_log1p(frame, prefixes=("bf_",))
    assert out.loc[0, "bf_gfa_m2"] == pytest.approx(np.log1p(10.0))
    assert out.loc[0, "ter_slope_mean_deg"] == 5.0
    assert out.loc[0, "area_km2"] == 2.0  # never transformed


# --------------------------------------------------------------------------
# baseline + rescaling
# --------------------------------------------------------------------------


def test_baseline_weights_follow_coefficients(sample_config: Config) -> None:
    features = pd.DataFrame(
        {
            "h3": ["a", "b"],
            "bf_gfa_residential_m2": [1000.0, 0.0],
            "bf_gfa_commercial_m2": [0.0, 1000.0],
            "bf_gfa_industrial_m2": [0.0, 0.0],
            "bf_gfa_public_m2": [0.0, 0.0],
            "bf_gfa_unknown_m2": [0.0, 0.0],
        }
    )
    weights = baseline_weights(features, sample_config)
    coefficients = sample_config.model.baseline_jobs_per_100m2
    assert weights["a"] == pytest.approx(1000.0 * coefficients["residential"] / 100.0)
    assert weights["b"] == pytest.approx(1000.0 * coefficients["commercial"] / 100.0)


def test_rescale_conserves_totals_and_handles_edge_cases() -> None:
    weights = pd.Series([1.0, 3.0, 0.0, 0.0, 5.0], index=list("abcde"))
    district = pd.Series(["x", "x", "y", "y", pd.NA], index=list("abcde"), dtype="string")
    jobs = rescale_to_totals(weights, district, {"x": 100.0, "y": 60.0})
    assert jobs["a"] == pytest.approx(25.0)
    assert jobs["b"] == pytest.approx(75.0)
    # Zero-weight district: equal split, total conserved.
    assert jobs["c"] == jobs["d"] == pytest.approx(30.0)
    # Buffer hex (district NA): no official total covers it.
    assert jobs["e"] == 0.0
    assert jobs.sum() == pytest.approx(160.0)


def test_rescaled_tree_surfaces_conserve_every_tree() -> None:
    rng = np.random.default_rng(0)
    weights = np.exp(rng.normal(size=(7, 5)))  # 7 trees x 5 hexes
    districts = pd.Series(["x", "x", "y", "y", "y"])
    totals = {"x": 100.0, "y": 50.0}
    surfaces = _rescaled_tree_surfaces(weights, districts, totals)
    assert surfaces.shape == weights.shape
    np.testing.assert_allclose(surfaces[:, :2].sum(axis=1), 100.0)
    np.testing.assert_allclose(surfaces[:, 2:].sum(axis=1), 50.0)


def test_rescaled_tree_surfaces_stray_district_raises() -> None:
    weights = np.ones((2, 3))
    districts = pd.Series(["x", "x", "ghost"])
    with pytest.raises(ValueError, match="ghost"):
        _rescaled_tree_surfaces(weights, districts, {"x": 100.0})


def test_rescaled_tree_surfaces_zeroed_district_gets_equal_split() -> None:
    # The empty-hex mask can zero a whole district's weights in a tree.
    weights = np.array([[2.0, 2.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0]])
    districts = pd.Series(["x", "x", "y", "y"])
    surfaces = _rescaled_tree_surfaces(weights, districts, {"x": 100.0, "y": 60.0})
    np.testing.assert_allclose(surfaces[:, :2].sum(axis=1), 100.0)
    np.testing.assert_allclose(surfaces[:, 2:], 30.0)  # equal split of 60


# --------------------------------------------------------------------------
# forest + LODO
# --------------------------------------------------------------------------


def _planted_world(n_districts: int = 6, years: tuple[int, ...] = (2023, 2024)):
    """Districts whose log job density is a clean function of one feature."""
    rng = np.random.default_rng(3)
    design = pd.DataFrame(
        {
            "bf_gfa_m2": np.linspace(1e5, 6e5, n_districts),
            "ter_slope_mean_deg": rng.uniform(1, 9, n_districts),
            "area_km2": np.full(n_districts, 10.0),
        },
        index=[f"d{i}" for i in range(n_districts)],
    )
    density = design["bf_gfa_m2"] / 1e4  # jobs per km2, linear in GFA
    rows = []
    for year in years:
        for district in design.index:
            noise = rng.normal(1.0, 0.03)
            rows.append(
                {
                    "district": district,
                    "year": year,
                    "jobs": float(density[district] * 10.0 * noise),
                }
            )
    return design, pd.DataFrame(rows)


def test_training_table_one_row_per_district_year(sample_config: Config) -> None:
    design, targets = _planted_world()
    X, y, groups = training_table(design, targets, [2023, 2024])
    assert len(X) == len(y) == len(groups) == 12
    assert "area_km2" not in X.columns
    assert groups.value_counts().eq(2).all()


def test_forest_learns_planted_relation_and_lodo_runs(sample_config: Config) -> None:
    design, targets = _planted_world()
    X, y, _ = training_table(design, targets, [2023, 2024])
    forest = fit_forest(X, y, sample_config)
    weights = tree_weights(forest, design[list(X.columns)])
    assert weights.shape == (sample_config.model.random_forest.n_estimators, len(design))
    assert (weights > 0).all()
    # Densest district must get the largest weight (planted monotone).
    means = weights.mean(axis=0)
    assert means.argmax() == len(design) - 1

    baseline_unscaled = pd.Series(np.linspace(1, 2, len(design)), index=design.index)
    table = lodo_table(design, targets, 2024, [2023, 2024], baseline_unscaled, sample_config)
    assert set(table.columns) == {"district", "actual", "pred_baseline", "pred_rf"}
    assert len(table) == len(design)
    # On a clean planted world the forest should be in the right ballpark
    # for interior districts (trees cannot extrapolate at the extremes).
    interior = table.iloc[1:-1]
    assert ((interior["pred_rf"] - interior["actual"]).abs() / interior["actual"]).mean() < 0.5


def test_lodo_does_not_leak_the_held_out_district(sample_config: Config) -> None:
    """Poison probe: give one district an absurd (x100) target.

    Without leakage the model has never seen that district's target, so
    its LODO prediction must stay near the OTHER districts' scale; any
    leak of the held-out years into training would pull the prediction
    toward the x100 value.
    """
    design, targets = _planted_world()
    poisoned = targets.copy()
    victim = design.index[2]  # an interior district
    poisoned.loc[poisoned["district"] == victim, "jobs"] *= 100.0
    baseline_unscaled = pd.Series(np.linspace(1, 2, len(design)), index=design.index)
    table = lodo_table(design, poisoned, 2024, [2023, 2024], baseline_unscaled, sample_config)
    row = table.set_index("district").loc[victim]
    others = table.loc[table["district"] != victim, "actual"]
    assert row["actual"] > 40 * others.max()  # the poison is in place
    assert row["pred_rf"] < 10 * others.max()  # ... and was never seen


def test_forest_is_deterministic_with_seed(sample_config: Config) -> None:
    design, targets = _planted_world()
    X, y, _ = training_table(design, targets, [2023, 2024])
    first = fit_forest(X, y, sample_config).predict(X)
    second = fit_forest(X, y, sample_config).predict(X)
    # n_jobs=-1 reduces tree sums in nondeterministic order: identical
    # trees, float-epsilon differences in the mean.
    np.testing.assert_allclose(first, second, rtol=1e-12)
