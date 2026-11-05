"""End-to-end Phase 5 test on the committed sample (offline).

Builds the equity outputs on top of the session-wide sample run and
checks schemas, cross-table consistency, the two population weightings
(including an independent recomputation of the calibrated path and of
2SFCA with calibrated demand), that every per-hex layer sits on the
right hex, the documented degraded paths, the report contract,
determinism and caching. Anything that rewrites outputs runs on a COPY
of the shared directories.
"""

from __future__ import annotations

import dataclasses
import json
import shutil

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pytest

from almaty_access import schemas
from almaty_access.config import Config
from almaty_access.equity.build import (
    DESERTS_FILE,
    DISTRICTS_FILE,
    HEX_FILE_TEMPLATES,
    LORENZ_FILE,
    NEAREST_FILE,
    REPORT_FILE,
    SUMMARY_FILE,
    _Analysis,
    _caveats,
    build_equity,
)
from almaty_access.equity.deserts import bivariate_classes
from almaty_access.equity.inequality import gini, weighted_quantile
from almaty_access.equity.spatial import (
    MIN_HEXES_FOR_LISA,
    benjamini_hochberg,
    monte_carlo_margin,
)

RASTER, CALIBRATED = schemas.EQUITY_WEIGHTINGS
LISA_LABELS = {"HH", "LL", "HL", "LH", "ns", "island"}
LONG_TABLES = {
    SUMMARY_FILE: schemas.PROCESSED_EQUITY_SUMMARY,
    LORENZ_FILE: schemas.PROCESSED_EQUITY_LORENZ,
    DISTRICTS_FILE: schemas.PROCESSED_EQUITY_DISTRICTS,
    DESERTS_FILE: schemas.PROCESSED_EQUITY_DESERTS,
    NEAREST_FILE: schemas.PROCESSED_EQUITY_NEAREST,
}


@pytest.fixture(scope="module")
def equitied(sample_pipeline: Config) -> Config:
    build_equity(sample_pipeline)
    return sample_pipeline


@pytest.fixture()
def scratch(equitied: Config, tmp_path) -> Config:
    """A private copy of the run: the only place tests may force rebuilds."""
    shutil.copytree(equitied.paths.processed, tmp_path / "processed")
    paths = dataclasses.replace(equitied.paths, processed=tmp_path / "processed")
    return dataclasses.replace(equitied, paths=paths)


def _report(config: Config) -> dict:
    def reject(token: str) -> None:
        raise AssertionError(f"non-strict JSON token in the report: {token}")

    text = (config.paths.processed / REPORT_FILE).read_text("utf-8")
    return json.loads(text, parse_constant=reject)


def _hexes(config: Config, res: int, weighting: str = RASTER) -> pd.DataFrame:
    return pd.read_parquet(config.paths.processed / HEX_FILE_TEMPLATES[weighting].format(res=res))


def _raw(config: Config, res: int) -> pd.DataFrame:
    """Access values and raster residents, joined by h3 with independent code."""
    processed = config.paths.processed
    access = pd.read_parquet(processed / f"access_r{res}.parquet")
    features = pd.read_parquet(
        processed / f"features_r{res}.parquet", columns=["h3", "pop_residents"]
    )
    return access.merge(features, on="h3")


def _equity_mtimes(config: Config) -> dict[str, int]:
    return {
        path.name: path.stat().st_mtime_ns
        for path in sorted(config.paths.processed.glob("equity_*"))
    }


# ---------------------------------------------------------------------------
# Schemas and the domain
# ---------------------------------------------------------------------------


def test_hex_tables_schema_and_domain(equitied: Config) -> None:
    n_ids = len(schemas.EQUITY_ID_COLUMNS)
    for res in (8, 9):
        raw = _raw(equitied, res).set_index("h3")
        raster = _hexes(equitied, res, RASTER)
        calibrated = _hexes(equitied, res, CALIBRATED)
        for weighting, table in ((RASTER, raster), (CALIBRATED, calibrated)):
            assert tuple(table.columns[:n_ids]) == schemas.EQUITY_ID_COLUMNS
            assert table["h3"].is_unique and len(table) == len(raw)
            allowed = schemas.EQUITY_PREFIXES
            if weighting == CALIBRATED:
                allowed += schemas.EQUITY_CALIBRATED_VALUE_PREFIXES
            for column in table.columns[n_ids:]:
                assert column.startswith(allowed), (weighting, column)

        # The domain rule, row by row, on the RASTER residents - and the
        # same set of hexes under both weightings.
        residents = raw["pop_residents"].reindex(raster["h3"]).to_numpy()
        np.testing.assert_allclose(raster["population"].to_numpy(), residents)
        expected = raster["in_city"].to_numpy() & (residents >= equitied.equity.min_population)
        assert (raster["inhabited"].to_numpy() == expected).all()
        assert (calibrated["inhabited"].to_numpy() == expected).all()
        # Calibration moves residents between districts, never the total.
        inside = raster["inhabited"].to_numpy()
        assert calibrated.loc[inside, "population"].sum() == pytest.approx(
            raster.loc[inside, "population"].sum()
        )
    # The mode gap needs transit and car: analysis resolution only.
    assert any(c.startswith("modegap_") for c in _hexes(equitied, 8).columns)
    assert not any(c.startswith("modegap_") for c in _hexes(equitied, 9).columns)
    # r8 is the population raster's own resolution: no dasymetric point masses.
    assert not _hexes(equitied, 8)["point_mass"].any()


def test_long_tables_follow_their_schemas(equitied: Config) -> None:
    for name, columns in LONG_TABLES.items():
        table = pd.read_parquet(equitied.paths.processed / name)
        assert tuple(table.columns) == columns, name
        assert set(table["weights"]) <= set(schemas.EQUITY_WEIGHTINGS), name
    for name in (SUMMARY_FILE, LORENZ_FILE, DISTRICTS_FILE, NEAREST_FILE):
        table = pd.read_parquet(equitied.paths.processed / name)
        assert set(table["weights"]) == set(schemas.EQUITY_WEIGHTINGS), name


def test_headline_lists_are_resolution_specific(equitied: Config) -> None:
    lorenz = pd.read_parquet(equitied.paths.processed / LORENZ_FILE)
    by_res = lorenz.groupby("res")["measure"].unique().apply(set).to_dict()
    assert by_res[8] == set(equitied.equity.headline_analysis)
    assert by_res[9] == set(equitied.equity.headline_fine)


def test_summary_is_sane(equitied: Config) -> None:
    summary = pd.read_parquet(equitied.paths.processed / SUMMARY_FILE)
    assert set(summary["res"]) == {8, 9}
    assert summary["measure"].str.startswith(schemas.EQUITY_BENEFIT_PREFIXES).all()
    # Service measures are granularity noise on coarse hexes: jobs only at r8.
    coarse = summary.loc[summary["res"] == 8, "measure"]
    assert coarse.str.contains("_jobs_").all()
    assert summary.loc[summary["res"] == 9, "measure"].str.startswith("sfca_").any()
    defined = summary.dropna(subset=["gini"])
    assert ((defined["gini"] >= 0) & (defined["gini"] < 1)).all()
    assert (summary["palma"].dropna() >= 0).all()  # inf allowed: the bottom 40% hold nothing
    assert (summary["p10"] <= summary["median"] + 1e-9).all()
    assert (summary["median"] <= summary["p90"] + 1e-9).all()
    report = _report(equitied)
    for (res, weights), group in summary.groupby(["res", "weights"]):
        block = report["results"][weights][f"r{res}"]
        assert np.allclose(group["population"], block["population"], atol=1.0)


def test_lorenz_curves_are_valid(equitied: Config) -> None:
    lorenz = pd.read_parquet(equitied.paths.processed / LORENZ_FILE)
    for _, curve in lorenz.groupby(["res", "weights", "measure"]):
        shares = curve["access_share"].to_numpy()
        if np.isnan(shares).all():
            continue  # a measure that sums to zero on the sample
        assert shares[0] == pytest.approx(0.0) and shares[-1] == pytest.approx(1.0)
        assert (np.diff(shares) >= -1e-12).all()
        assert (shares <= curve["pop_share"].to_numpy() + 1e-12).all()


def test_district_table_covers_the_domain(equitied: Config) -> None:
    districts = pd.read_parquet(equitied.paths.processed / DISTRICTS_FILE)
    report = _report(equitied)
    for (res, weights, _), group in districts.groupby(["res", "weights", "measure"]):
        inhabited = report["results"][weights][f"r{res}"]["population"]
        assert group["population"].sum() == pytest.approx(inhabited, abs=1.0)
        assert group["rank"].min() == 1
        assert group["district"].is_unique


def test_desert_flags_match_the_desert_table(equitied: Config) -> None:
    deserts = pd.read_parquet(equitied.paths.processed / DESERTS_FILE)
    for res in (8, 9):
        for weighting in schemas.EQUITY_WEIGHTINGS:
            table = _hexes(equitied, res, weighting)
            for column in (c for c in table.columns if c.startswith("desert_")):
                measure = column.removeprefix("desert_")
                listed = deserts.loc[
                    (deserts["res"] == res)
                    & (deserts["weights"] == weighting)
                    & (deserts["measure"] == measure)
                ]
                assert set(table.loc[table[column], "h3"]) == set(listed["h3"])
                assert not table.loc[~table["inhabited"], column].any()
                if len(listed):
                    assert listed["rank"].tolist() == list(range(1, len(listed) + 1))
                    assert listed["population"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# Every per-hex layer sits on the right hex
# ---------------------------------------------------------------------------


def test_bivariate_classes_sit_on_the_right_hex(equitied: Config) -> None:
    """Recomputed from the access table BY H3, in a shuffled order."""
    stored = _hexes(equitied, 9).set_index("h3")
    raw = _raw(equitied, 9)
    domain = raw.loc[raw["in_city"] & (raw["pop_residents"] >= equitied.equity.min_population)]
    domain = domain.sample(frac=1.0, random_state=3).set_index("h3")
    for measure in equitied.equity.headline_fine:
        expected = bivariate_classes(domain["pop_residents"], domain[measure])
        got = stored.loc[domain.index, f"biv_{measure}"]
        assert (got.to_numpy() == expected.to_numpy()).all(), measure
        assert stored.loc[~stored["inhabited"], f"biv_{measure}"].isna().all()


def test_lisa_labels_sit_on_the_right_hex(equitied: Config) -> None:
    """A significant label must agree with the quadrant computed from the
    hex's OWN value and its neighbours' values - which no permutation
    affects, and which labels attached to the wrong hexes would violate."""
    stored = _hexes(equitied, 9).set_index("h3")
    raw = _raw(equitied, 9).set_index("h3")
    inhabited = set(stored.index[stored["inhabited"]])
    report = _report(equitied)["results"][RASTER]["r9"]["lisa"]
    checked = 0
    for measure in equitied.equity.headline_fine:
        if "skipped_reason" in report[measure]:
            continue
        values = raw.loc[sorted(inhabited), measure]
        deviation = values - values.mean()
        for cell in inhabited:
            label = stored.loc[cell, f"lisa_{measure}"]
            neighbours = [n for n in h3.grid_ring(cell, 1) if n in inhabited]
            if not neighbours:
                assert label == "island"
                assert np.isnan(stored.loc[cell, f"lisap_{measure}"])
                continue
            assert label != "island"
            if label == "ns":
                continue
            own, lag = deviation[cell], deviation[neighbours].mean()
            quadrant = ("H" if own > 0 else "L") + ("H" if lag > 0 else "L")
            assert label == quadrant, (measure, cell)
            assert stored.loc[cell, f"lisap_{measure}"] < equitied.equity.lisa_alpha
            checked += 1
    assert checked > 0  # the sample does produce significant clusters


def test_lisa_degrades_at_r8_and_runs_at_r9(equitied: Config) -> None:
    report = _report(equitied)["results"][RASTER]
    coarse, fine = _hexes(equitied, 8), _hexes(equitied, 9)
    assert coarse["inhabited"].sum() < MIN_HEXES_FOR_LISA <= fine["inhabited"].sum()

    for measure in equitied.equity.headline_analysis:
        assert "skipped_reason" in report["r8"]["lisa"][measure]
        assert coarse[f"lisa_{measure}"].isna().all()

    ran = 0
    for measure in equitied.equity.headline_fine:
        block = report["r9"]["lisa"][measure]
        labels = fine.loc[fine["inhabited"], f"lisa_{measure}"]
        if "skipped_reason" in block:  # e.g. a measure constant on the sample
            assert labels.isna().all()
            continue
        ran += 1
        assert set(labels) <= LISA_LABELS
        assert fine.loc[~fine["inhabited"], f"lisa_{measure}"].isna().all()
        assert labels.value_counts().to_dict() == block["clusters"]
        assert -1 <= block["global_moran"]["I"] <= 1
        # The FDR count is the Benjamini-Hochberg survivors among the
        # significant cells - or null when the permutations cannot resolve it.
        p = fine.loc[fine["inhabited"], f"lisap_{measure}"].to_numpy()
        alpha = equitied.equity.lisa_alpha
        survivors = int((benjamini_hochberg(p, alpha) & (p < alpha)).sum())
        fdr = block["fdr"]
        assert fdr["tests"] == int(np.isfinite(p).sum())
        if fdr["significant"] is None:
            assert survivors == 0 and 0 < fdr["n_at_floor"] < fdr["n_needed"]
        else:
            assert fdr["significant"] == survivors
        # Hexes within Monte Carlo error of alpha, counted from the file.
        assert block["borderline"] == monte_carlo_margin(
            p, alpha, equitied.equity.lisa_permutations
        )
    assert ran >= 1


# ---------------------------------------------------------------------------
# The calibrated weighting, recomputed independently
# ---------------------------------------------------------------------------


def test_calibrated_weights_are_wired_through(equitied: Config) -> None:
    """With two districts at r9 the calibration is not a no-op: rebuild
    the weights from the reported factors and require the calibrated
    file, summary and comparison to carry exactly their consequences."""
    report = _report(equitied)
    calibration = report["calibration"]
    factors = calibration["district_factors"]["r9"]
    assert len(factors) == 2 and min(factors.values()) < 1 < max(factors.values())
    shares = calibration["district_shares"]["r9"]
    for name, factor in factors.items():
        assert factor == pytest.approx(shares[name]["target"] / shares[name][RASTER], rel=2e-2)

    raster = _hexes(equitied, 9, RASTER).set_index("h3")
    calibrated = _hexes(equitied, 9, CALIBRATED).set_index("h3")
    inside = raster.index[raster["inhabited"]]
    expected = raster.loc[inside, "population"] * raster.loc[inside, "district"].map(factors)
    np.testing.assert_allclose(
        calibrated.loc[inside, "population"].to_numpy(), expected.to_numpy(), rtol=1e-3
    )

    measure = "accpct_jobs_walk_30"
    values = _raw(equitied, 9).set_index("h3").loc[inside, measure].to_numpy()
    weights = calibrated.loc[inside, "population"].to_numpy()
    comparison = report["comparison"]["r9"][measure]
    assert comparison["gini"][CALIBRATED] == pytest.approx(gini(values, weights), abs=1e-4)
    assert comparison["median"][CALIBRATED] == pytest.approx(
        weighted_quantile(values, weights, 0.5), abs=1e-4
    )
    assert comparison["mean"][CALIBRATED] == pytest.approx(
        float((values * weights).sum() / weights.sum()), abs=1e-4
    )
    assert comparison["mean"][CALIBRATED] != comparison["mean"][RASTER]
    summary = pd.read_parquet(equitied.paths.processed / SUMMARY_FILE)
    row = summary.loc[
        (summary["res"] == 9) & (summary["weights"] == CALIBRATED) & (summary["measure"] == measure)
    ].iloc[0]
    assert row["gini"] == pytest.approx(gini(values, weights))


def test_two_step_fca_is_recomputed_with_calibrated_demand(equitied: Config) -> None:
    """Residents are 2SFCA's demand term: under the calibrated weighting
    the VALUES change, and the measure still conserves supply - the
    resident-weighted mean over all hexes is the same under both."""
    raw = _raw(equitied, 9).set_index("h3")
    raster = _hexes(equitied, 9, RASTER).set_index("h3")
    calibrated = _hexes(equitied, 9, CALIBRATED).set_index("h3")
    changed = 0
    for column in (c for c in calibrated.columns if c.startswith("sfca_")):
        original = raw[column].reindex(calibrated.index).to_numpy()
        recomputed = calibrated[column].to_numpy()
        if original.sum() == 0:
            continue
        changed += int(not np.allclose(original, recomputed))
        supply_raster = float((original * raster["population"].to_numpy()).sum())
        supply_calibrated = float((recomputed * calibrated["population"].to_numpy()).sum())
        assert supply_calibrated == pytest.approx(supply_raster, rel=2e-2)
    assert changed >= 1


def test_calibrated_two_step_fca_matches_an_independent_recomputation(
    equitied: Config,
) -> None:
    """Rebuild 2SFCA from the walk matrix, the POI layer and the calibrated
    residents with plain groupby code - not the pipeline's functions - and
    require the calibrated file's values. (Supply conservation alone would
    also be satisfied by dividing the raster values by the district factor.)"""
    processed = equitied.paths.processed
    calibrated = _hexes(equitied, 9, CALIBRATED).set_index("h3")
    residents = calibrated["population"]
    walk = pd.read_parquet(processed / "tt_walk_r9.parquet")
    walk = walk.loc[walk["minutes"] <= equitied.access.sfca_catchment_min]
    pois = gpd.read_parquet(equitied.paths.interim / "pois.parquet")
    categories = {
        "schools": ["education_school"],
        "kindergartens": ["education_kindergarten"],
        "clinics": ["health_clinic", "health_hospital"],
    }
    checked = 0
    for service, wanted in categories.items():
        points = pois.loc[pois["category"].isin(wanted)].geometry
        cells = [h3.latlng_to_cell(point.y, point.x, 9) for point in points]
        supply = pd.Series(cells).value_counts()
        demand = (
            walk.assign(pop=walk["origin_h3"].map(residents).fillna(0.0))
            .groupby("dest_h3")["pop"]
            .sum()
        )
        ratio = (supply.reindex(demand.index).fillna(0.0) / demand.where(demand > 0)).fillna(0.0)
        expected = (
            walk.assign(ratio=walk["dest_h3"].map(ratio).fillna(0.0))
            .groupby("origin_h3")["ratio"]
            .sum()
            * 1000.0
        )
        expected = expected.reindex(calibrated.index).fillna(0.0)
        stored = calibrated[f"sfca_{service}_walk"]
        np.testing.assert_allclose(stored.to_numpy(), expected.to_numpy(), atol=2e-4)
        checked += int(stored.sum() > 0)
    assert checked >= 1


def test_lisa_locations_are_weight_free_exactly_where_the_values_are(
    equitied: Config,
) -> None:
    """For job measures the two per-hex files must carry IDENTICAL LISA
    labels and p-values (the statistic never sees the weights). For 2SFCA
    the values are recomputed, so the hot spots may move - and the
    report's overlap counts must be what the two files actually contain."""
    comparison = _report(equitied)["comparison"]
    for res, measures in (
        (8, equitied.equity.headline_analysis),
        (9, equitied.equity.headline_fine),
    ):
        raster = _hexes(equitied, res, RASTER).set_index("h3")
        calibrated = _hexes(equitied, res, CALIBRATED).set_index("h3").reindex(raster.index)
        for measure in measures:
            label, p_value = f"lisa_{measure}", f"lisap_{measure}"
            if measure.startswith("accpct_jobs_"):
                pd.testing.assert_series_equal(raster[label], calibrated[label])
                pd.testing.assert_series_equal(raster[p_value], calibrated[p_value])
            overlap = comparison[f"r{res}"][measure].get("hot_spot_hexes")
            if overlap is None:
                continue  # LISA skipped for this measure
            hot_raster = set(raster.index[raster[label].fillna("") == "HH"])
            hot_calibrated = set(calibrated.index[calibrated[label].fillna("") == "HH"])
            assert overlap == {
                RASTER: len(hot_raster),
                CALIBRATED: len(hot_calibrated),
                "both": len(hot_raster & hot_calibrated),
            }


def test_mode_gap_is_recomputed_from_the_access_table(equitied: Config) -> None:
    raw = _raw(equitied, 8).set_index("h3")
    table = _hexes(equitied, 8).set_index("h3")
    report = _report(equitied)["results"][RASTER]["r8"]["mode_gap"]
    assert set(report) == {f"modegap_transit_car_{t}" for t in equitied.travel.thresholds_min}
    inside = table.index[table["inhabited"]]
    weights = table.loc[inside, "population"].to_numpy()
    for threshold in equitied.travel.thresholds_min:
        transit = raw.loc[table.index, f"acc_jobs_transit_{threshold}"].to_numpy()
        car = raw.loc[table.index, f"acc_jobs_car_{threshold}"].to_numpy()
        stored = table[f"modegap_transit_car_{threshold}"].to_numpy()
        defined = car > 0
        np.testing.assert_allclose(stored[defined], (transit / car)[defined], atol=1e-4)
        assert np.isnan(stored[~defined]).all()
        block = report[f"modegap_transit_car_{threshold}"]
        ratio = table.loc[inside, f"modegap_transit_car_{threshold}"].to_numpy()
        finite = np.isfinite(ratio)
        assert block["median"] == pytest.approx(
            weighted_quantile(ratio[finite], weights[finite], 0.5), abs=1e-4
        )
        same = np.isclose(
            raw.loc[inside, f"acc_jobs_transit_{threshold}"],
            raw.loc[inside, f"acc_jobs_walk_{threshold}"],
        )
        assert block["pop_share_transit_equals_walk"] == pytest.approx(
            weights[same].sum() / weights.sum(), abs=1e-4
        )


def test_nearest_table(equitied: Config) -> None:
    nearest = pd.read_parquet(equitied.paths.processed / NEAREST_FILE)
    assert ((nearest["pop_share_within"] >= 0) & (nearest["pop_share_within"] <= 1)).all()
    assert "ALL" in set(nearest["district"])
    assert set(nearest["threshold_min"]) == set(equitied.equity.nearest_thresholds_min)
    for _, group in nearest.groupby(["res", "weights", "district", "facility"]):
        ordered = group.sort_values("threshold_min")["pop_share_within"]
        assert ordered.is_monotonic_increasing  # a longer walk reaches at least as many


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_report_contract(equitied: Config) -> None:
    report = _report(equitied)  # strict JSON, or _report raises
    first = equitied.equity.headline_analysis[0]
    headline = report["headline"]
    assert headline["measure"] == first and headline["res"] == equitied.h3.analysis_resolution
    assert "of residents" in headline["sentence"]
    assert "lower bound" in headline["sentence"]  # the transit caveat travels with it
    # The headline share is the same number three independent code paths report.
    cut = f"{equitied.equity.headline_cutoff_pct:g}"
    assert headline["pop_share"] == report["results"][RASTER]["r8"]["below_cutoffs"][first][cut]
    assert headline["pop_share"] == report["comparison"]["r8"][first]["below_cutoffs"][cut][RASTER]
    assert (
        headline["pop_share_calibrated"]
        == report["comparison"]["r8"][first]["below_cutoffs"][cut][CALIBRATED]
    )
    shares = [
        report["results"][RASTER]["r8"]["below_cutoffs"][first][f"{c:g}"]
        for c in equitied.equity.job_share_cutoffs_pct
    ]
    assert shares == sorted(shares)  # a higher cutoff catches at least as many

    assert report["companions"] and all("walk" in line for line in report["companions"])
    assert any("LOWER BOUND" in caveat for caveat in report["caveats"])
    assert report["params"]["lisa_p_value"].startswith("tie-aware")
    assert set(report["results"]) == set(schemas.EQUITY_WEIGHTINGS)
    for weighting in schemas.EQUITY_WEIGHTINGS:
        for res in ("r8", "r9"):
            block = report["results"][weighting][res]
            assert block["inhabited_hexes"] > 0
            assert block["nearest"]["clinic"]["median_min"] >= 0
        coverage = report["results"][weighting]["r8"]["transit_coverage"]
        assert coverage["pop_share_transit_equals_walk"].keys() == {
            str(t) for t in equitied.travel.thresholds_min
        }
        assert coverage["headline_walk_only"]["measure"] == "accpct_jobs_walk_45"


def test_sample_statistics_are_labelled_synthetic(equitied: Config) -> None:
    """The committed sample ships SYNTHETIC district statistics; the
    report must say so wherever it describes the calibration - and must
    never use the word 'official' about any source."""
    report = _report(equitied)
    assert report["calibration"]["synthetic"] is True
    assert report["calibration"]["source"] == "synthetic-for-tests"
    assert "SYNTHETIC test numbers" in report["weightings"][CALIBRATED]
    assert any("SYNTHETIC test numbers" in caveat for caveat in report["caveats"])
    assert "official" not in json.dumps(report).lower()


def test_caveats_are_generated_from_the_data(equitied: Config) -> None:
    """A caveat is a claim about THIS dataset. The sample's 2SFCA is not
    zero-inflated (every inhabited hex reaches a facility), so that
    caveat must not appear here although the city report carries it."""
    report = _report(equitied)
    zero_shares = [
        entry["zero_share_hexes"]
        for name, entry in report["results"][RASTER]["r9"]["lisa"].items()
        if name.startswith("sfca_") and "zero_share_hexes" in entry
    ]
    assert zero_shares and max(zero_shares) <= 0.5
    assert not any("zero-inflated" in caveat for caveat in report["caveats"])
    # ... while the transit-coverage caveat quotes the measured shares of
    # BOTH weightings (one number when the two print the same).
    low, high = sorted(
        report["results"][weighting]["r8"]["transit_coverage"]["pop_share_transit_equals_walk"][
            "45"
        ]
        for weighting in schemas.EQUITY_WEIGHTINGS
    )
    quoted = f"{low:.0%}" if f"{low:.0%}" == f"{high:.0%}" else f"{low:.0%} and {high:.0%}"
    assert any(f"{quoted} at 45 min" in caveat for caveat in report["caveats"])


def test_transit_coverage_caveat_quotes_both_weightings(equitied: Config) -> None:
    """The caveat is a sentence about residents, so it must not quote the
    raster alone: where the weightings print differently both figures
    appear, where they agree (or only one run exists) a single one."""
    domain = pd.DataFrame({"h3": ["a", "b"], "transit_routes_cnt": [2, 0]})

    def analysis(weighting: str, shares: dict[str, float]) -> _Analysis:
        block = {"transit_coverage": {"pop_share_transit_equals_walk": shares}}
        return _Analysis(weighting=weighting, res=8, domain=domain, hexes=domain, block=block)

    analyses = {
        (RASTER, 8): analysis(RASTER, {"15": 1.0, "45": 0.5}),
        (CALIBRATED, 8): analysis(CALIBRATED, {"15": 1.0, "45": 0.43}),
    }
    caveats = _caveats(equitied, analyses, None, {})
    assert "1 of 2 inhabited r8 hexes host a modelled stop" in caveats[0]
    line = next(caveat for caveat in caveats if "transit = walk" in caveat)
    assert "100% at 15 min, 43% and 50% at 45 min" in line

    raster_only = _caveats(equitied, {(RASTER, 8): analyses[(RASTER, 8)]}, None, {})
    line = next(caveat for caveat in raster_only if "transit = walk" in caveat)
    assert "100% at 15 min, 50% at 45 min" in line
    # No residence series at all: said plainly, with no calibration claim.
    assert "no residence-based series was available" in raster_only[-1]


# ---------------------------------------------------------------------------
# Caching, determinism, failure modes (rebuilds only on a private copy)
# ---------------------------------------------------------------------------


def test_equity_build_is_cached(equitied: Config) -> None:
    before = _equity_mtimes(equitied)
    build_equity(equitied)  # everything fresh: nothing may be rewritten
    assert _equity_mtimes(equitied) == before


def test_equity_is_deterministic(equitied: Config, scratch: Config) -> None:
    build_equity(scratch, force=True)  # seeded: identical to the shared run
    for res in (8, 9):
        for weighting in schemas.EQUITY_WEIGHTINGS:
            pd.testing.assert_frame_equal(
                _hexes(equitied, res, weighting), _hexes(scratch, res, weighting)
            )
    assert _report(equitied) == _report(scratch)


def test_unknown_headline_measure_fails_before_writing_anything(scratch: Config) -> None:
    before = _equity_mtimes(scratch)
    broken = dataclasses.replace(
        scratch,
        equity=dataclasses.replace(scratch.equity, headline_fine=("accpct_jobs_transit_45",)),
    )
    # A transit measure exists at r8 but NOT in the fine table.
    with pytest.raises(ValueError, match=r"headline_measures\.fine.*access_r9"):
        build_equity(broken, force=True)
    assert _equity_mtimes(scratch) == before  # no output was touched


def test_without_a_resident_series_only_the_raster_run_exists(scratch: Config, tmp_path) -> None:
    """No employed-residents rows: the calibrated run is skipped with a
    recorded reason instead of inventing weights."""
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(scratch.paths.raw / "osm_landuse.parquet", raw / "osm_landuse.parquet")
    stats = pd.read_csv(scratch.paths.raw / "district_stats.csv")
    stats.loc[stats["indicator"] != "employed_residents"].to_csv(
        raw / "district_stats.csv", index=False
    )
    # The scratch copy still holds the calibrated files of the earlier
    # run: they must not survive next to a report that says "skipped".
    assert list(scratch.paths.processed.glob("equity_calibrated_*"))
    config = dataclasses.replace(scratch, paths=dataclasses.replace(scratch.paths, raw=raw))
    build_equity(config)
    report = _report(config)
    assert "skipped_reason" in report["calibration"]
    assert set(report["results"]) == {RASTER}
    assert "comparison" not in report
    assert report["headline"]["pop_share_calibrated"] is None
    assert not list(config.paths.processed.glob("equity_calibrated_*"))
    for name in LONG_TABLES:
        table = pd.read_parquet(config.paths.processed / name)
        assert set(table["weights"]) <= {RASTER}


def test_a_missing_statistics_file_also_skips_the_calibrated_run(scratch: Config, tmp_path) -> None:
    raw = tmp_path / "raw_without_stats"
    raw.mkdir()
    shutil.copy(scratch.paths.raw / "osm_landuse.parquet", raw / "osm_landuse.parquet")
    config = dataclasses.replace(scratch, paths=dataclasses.replace(scratch.paths, raw=raw))
    build_equity(config)
    assert "skipped_reason" in _report(config)["calibration"]
    assert not list(config.paths.processed.glob("equity_calibrated_*"))


def test_drifted_two_step_fca_stops_the_build(scratch: Config) -> None:
    """The calibrated 2SFCA is only trusted because the same code first
    reproduces the stored raster values. If the access table and the
    recomputation ever disagree, the build must stop - not ship both."""
    path = scratch.paths.processed / "access_r9.parquet"
    access = pd.read_parquet(path)
    access["sfca_schools_walk"] = access["sfca_schools_walk"] + 1.0
    access.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="cannot reproduce sfca_schools_walk"):
        build_equity(scratch, force=True)
