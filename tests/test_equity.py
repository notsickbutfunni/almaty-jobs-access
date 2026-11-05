"""Unit tests for the Phase 5 building blocks (offline, toy data).

Inequality measures are checked against hand-computed values and against
brute-force replication (a hex of 3 residents must behave exactly like
three one-resident hexes); the spatial statistics against planted
clusters on real H3 cells, the project's own Phase 2 Moran's I, and the
two tie traps the library falls into on zero-inflated data.
"""

from __future__ import annotations

import dataclasses

import h3
import numpy as np
import pandas as pd
import pytest

from almaty_access.config import Config
from almaty_access.equity.build import (
    _Analysis,
    _companions,
    _headline,
    _jsonable,
    _lisa_block,
    _share_text,
)
from almaty_access.equity.deserts import access_deserts, bivariate_classes, tercile_classes
from almaty_access.equity.inequality import (
    bottom_share_mean,
    distribution_summary,
    gini,
    lorenz_curve,
    palma,
    weighted_quantile,
)
from almaty_access.equity.spatial import (
    benjamini_hochberg,
    fdr_resolution,
    fdr_summary,
    global_moran,
    hex_weights,
    local_moran,
    monte_carlo_margin,
    tie_aware_p,
)
from almaty_access.equity.tables import (
    CITY_SCOPE,
    below_cutoff_shares,
    center_gradient,
    district_table,
    mode_gap,
    nearest_tables,
)
from almaty_access.equity.weights import (
    apply_factors,
    calibrated_weights,
    district_resident_shares,
)
from almaty_access.features.spatial import morans_i

CENTER = h3.latlng_to_cell(43.2586, 76.9532, 9)  # Panfilov park, Almaty


# ---------------------------------------------------------------------------
# Weighted quantiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_weighted_quantile_matches_replication(q: float) -> None:
    rng = np.random.default_rng(7)
    values = rng.integers(0, 50, size=40).astype(float)
    weights = rng.integers(1, 9, size=40)
    expanded = np.repeat(values, weights)
    expected = np.quantile(expanded, q, method="inverted_cdf")
    assert weighted_quantile(values, weights, q) == pytest.approx(expected)


def test_weighted_quantile_ignores_zero_weight_and_nan() -> None:
    values = np.array([1.0, 100.0, np.nan, 3.0])
    weights = np.array([1.0, 0.0, 5.0, 1.0])
    assert weighted_quantile(values, weights, 1.0) == 3.0  # 100 has no residents


def test_weighted_quantile_survives_float_drift_at_a_boundary() -> None:
    """Five weights of 0.3 at q = 0.2: the first hex holds exactly a fifth
    of the weight, but 0.2 * 1.5 evaluates to 0.30000000000000004, a hair
    ABOVE the running sum 0.3. Without the tolerance the answer skips to
    the second value."""
    values = np.arange(1.0, 6.0)
    weights = np.full(5, 0.3)
    assert 0.2 * weights.sum() > np.cumsum(weights)[0]  # the drift this test is about
    assert weighted_quantile(values, weights, 0.2) == 1.0
    # ... and the harmless cases stay right.
    assert weighted_quantile(np.arange(1.0, 11.0), np.full(10, 0.1), 0.8) == 8.0
    assert weighted_quantile(np.arange(1.0, 11.0), np.full(10, 0.1), 0.5) == 5.0


def test_bottom_share_mean_splits_the_straddling_hex() -> None:
    """Worst-served quarter of 1,000 residents: all 200 at 0.1 plus 50 of
    the residents at 5.0 -> (20 + 250) / 250."""
    values = np.array([0.1, 5.0, 9.0])
    weights = np.array([200.0, 500.0, 300.0])
    assert bottom_share_mean(values, weights, 0.25) == pytest.approx(1.08)
    assert bottom_share_mean(np.zeros(3), np.ones(3), 0.25) == 0.0


def test_weighted_quantile_empty_and_invalid() -> None:
    assert np.isnan(weighted_quantile(np.array([1.0]), np.array([0.0]), 0.5))
    with pytest.raises(ValueError, match="quantile"):
        weighted_quantile(np.array([1.0]), np.array([1.0]), 1.5)


# ---------------------------------------------------------------------------
# Lorenz / Gini / Palma
# ---------------------------------------------------------------------------


def test_gini_of_equal_access_is_zero() -> None:
    assert gini(np.full(5, 7.0), np.array([1, 20, 3, 400, 5])) == pytest.approx(0.0)


def test_gini_two_groups_exact() -> None:
    """Half the residents hold everything: Lorenz (0,0)-(0.5,0)-(1,1), G = 0.5."""
    assert gini(np.array([0.0, 1.0]), np.array([50.0, 50.0])) == pytest.approx(0.5)


def test_gini_weights_equal_row_replication() -> None:
    """A hex of w residents == w one-resident hexes == the textbook
    mean-absolute-difference Gini on the expanded population."""
    rng = np.random.default_rng(11)
    values = rng.gamma(2.0, 3.0, size=25)
    weights = rng.integers(1, 7, size=25)
    expanded = np.repeat(values, weights)
    by_weights = gini(values, weights)
    by_replication = gini(expanded, np.ones_like(expanded))
    mean_abs_diff = np.abs(expanded[:, None] - expanded[None, :]).mean()
    textbook = mean_abs_diff / (2 * expanded.mean())
    assert by_weights == pytest.approx(by_replication)
    assert by_weights == pytest.approx(textbook)


def test_gini_is_scale_invariant() -> None:
    values = np.array([1.0, 4.0, 9.0, 16.0])
    weights = np.array([3.0, 1.0, 2.0, 5.0])
    assert gini(values, weights) == pytest.approx(gini(values * 1000, weights))


def test_gini_undefined_when_total_is_zero() -> None:
    assert np.isnan(gini(np.zeros(4), np.ones(4)))


def test_negative_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        gini(np.array([1.0, -2.0]), np.array([1.0, 1.0]))


def test_palma_exact_on_ten_equal_groups() -> None:
    """Values 1..10, equal populations: top 10% hold 10, bottom 40% hold
    1+2+3+4 = 10 of the 55 total -> Palma = 1."""
    assert palma(np.arange(1.0, 11.0), np.ones(10)) == pytest.approx(1.0)


def test_palma_equality_and_extremes() -> None:
    assert palma(np.full(6, 3.0), np.ones(6)) == pytest.approx(0.25)  # 10% / 40%
    assert palma(np.array([0.0, 0.0, 5.0]), np.array([40.0, 40.0, 20.0])) == float("inf")
    assert np.isnan(palma(np.zeros(3), np.ones(3)))


def test_palma_splits_a_hex_across_the_cutoffs() -> None:
    """Two hexes of 50 residents with access 1 and 3 (total mass 200):
    the bottom 40% are 40 residents of the first hex (40/200), the top
    10% are 10 residents of the second (30/200) -> 0.15 / 0.20."""
    assert palma(np.array([1.0, 3.0]), np.array([50.0, 50.0])) == pytest.approx(0.75)


def test_lorenz_curve_shape() -> None:
    rng = np.random.default_rng(3)
    values = rng.gamma(1.5, 2.0, size=60)
    weights = rng.integers(1, 30, size=60)
    pop_share, access_share = lorenz_curve(values, weights, n_points=51)
    assert pop_share[0] == 0.0 and pop_share[-1] == 1.0
    assert access_share[0] == pytest.approx(0.0) and access_share[-1] == pytest.approx(1.0)
    assert (np.diff(access_share) >= -1e-12).all()  # monotone
    assert (np.diff(access_share, 2) >= -1e-9).all()  # convex
    assert (access_share <= pop_share + 1e-12).all()  # never above equality


def test_lorenz_curve_of_nothing_is_nan() -> None:
    _, access_share = lorenz_curve(np.zeros(3), np.ones(3))
    assert np.isnan(access_share).all()


def test_distribution_summary_fields() -> None:
    values = np.array([0.0, 0.0, 10.0, 20.0])
    weights = np.array([10.0, 30.0, 40.0, 20.0])
    summary = distribution_summary(values, weights)
    assert summary["n_hexes"] == 4
    assert summary["population"] == 100.0
    assert summary["mean"] == pytest.approx(8.0)
    assert summary["median"] == 10.0  # cumulative share reaches 0.5 inside the third hex
    assert summary["pop_share_zero"] == pytest.approx(0.4)
    assert 0 < summary["gini"] < 1


# ---------------------------------------------------------------------------
# Spatial weights, Moran, LISA
# ---------------------------------------------------------------------------


def test_hex_weights_row_standardized_with_island() -> None:
    disk = sorted(h3.grid_disk(CENTER, 2))
    far = h3.latlng_to_cell(43.40, 77.20, 9)  # nowhere near the disk
    weights, islands = hex_weights([*disk, far])
    assert islands.sum() == 1 and islands[-1]
    assert weights.cardinalities[CENTER] == 6
    for cell in disk:
        assert sum(weights[cell].values()) == pytest.approx(1.0)
        for neighbour in weights.neighbors[cell]:
            assert cell in weights.neighbors[neighbour]  # symmetric contiguity


def test_tie_aware_p_counts_float_ties() -> None:
    """300 permutations tie with the observed maximum but differ in the
    last bits: they must count as 'as extreme' (p = 0.602), where a bare
    >= would give the absurd 0.002."""
    observed = np.array([1.0])
    tied = np.full(300, 1.0 - 1e-15)
    lower = np.linspace(-1.0, 0.5, 699)
    simulated = np.concatenate([tied, lower])[None, :]
    assert (simulated >= observed[:, None]).sum() == 0  # the trap
    assert tie_aware_p(observed, simulated)[0] == pytest.approx(2 * 301 / 1000)


def test_tie_aware_p_extreme_without_ties() -> None:
    simulated = np.linspace(0.0, 1.0, 999)[None, :]
    assert tie_aware_p(np.array([5.0]), simulated)[0] == pytest.approx(2 / 1000)
    assert tie_aware_p(np.array([0.5]), simulated)[0] == pytest.approx(1.0)  # dead centre


def test_benjamini_hochberg_reference_case() -> None:
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.6, np.nan])
    mask = benjamini_hochberg(p, alpha=0.05)
    # m = 5 valid tests: 0.001 <= 0.01 and 0.008 <= 0.02 pass, 0.039 > 0.03 stops it.
    assert mask.tolist() == [True, True, False, False, False, False]
    assert not benjamini_hochberg(np.array([0.9, 0.8]), 0.05).any()


def test_fdr_resolution_says_when_permutations_cannot_resolve() -> None:
    """9,999 permutations: the floor is 0.0002, and BH over 3,324 tests
    admits floor-level cells only when at least 14 of them exist."""
    p = np.concatenate([np.full(8, 0.0002), np.full(3316, 0.5), [np.nan]])
    resolution = fdr_resolution(p, alpha=0.05, permutations=9999)
    assert resolution == {"tests": 3324, "min_p": 0.0002, "n_at_floor": 8, "n_needed": 14}
    assert not benjamini_hochberg(p, 0.05).any()  # zero - but only for lack of resolution


def test_fdr_summary_reports_null_only_when_unresolved() -> None:
    """8 floor-level hot spots among 3,324 tests need 14 to be admitted:
    the survivor counts are None, not 0. With no floor-level cell at all
    a zero is a real zero; with enough of them the counts are counts."""
    p = np.concatenate([np.full(8, 0.0002), np.full(3316, 0.5)])
    labels = np.array(["HH"] * 8 + ["ns"] * 3316, dtype=object)

    def frame(p_values: np.ndarray, tags: np.ndarray) -> pd.DataFrame:
        survives = benjamini_hochberg(p_values, 0.05) & (p_values < 0.05)
        return pd.DataFrame({"label": tags, "p": p_values, "fdr": survives})

    unresolved = fdr_summary(frame(p, labels), alpha=0.05, permutations=9999)
    assert unresolved["significant"] is None
    assert unresolved["hot_spots"] is None and unresolved["cold_spots"] is None
    assert (unresolved["n_at_floor"], unresolved["n_needed"]) == (8, 14)

    nothing = fdr_summary(frame(np.full(3324, 0.5), labels), alpha=0.05, permutations=9999)
    assert nothing["significant"] == 0 and nothing["n_at_floor"] == 0

    many_p = np.concatenate([np.full(20, 0.0002), np.full(3304, 0.5)])
    many_labels = np.array(["HH"] * 12 + ["LL"] * 8 + ["ns"] * 3304, dtype=object)
    resolved = fdr_summary(frame(many_p, many_labels), alpha=0.05, permutations=9999)
    assert (resolved["significant"], resolved["hot_spots"], resolved["cold_spots"]) == (20, 12, 8)


def test_monte_carlo_margin_counts_cells_near_alpha() -> None:
    """9,999 permutations: the two-sided p-value near 0.05 has a standard
    error of 2 * sqrt(0.025 * 0.975 / 10,000) = 0.0031, so the band of
    two standard errors is 0.0438 .. 0.0562 - and wider with fewer."""
    p = np.array([0.0002, 0.0436, 0.0438, 0.0498, 0.05, 0.0562, 0.0564, 0.9, np.nan])
    margin = monte_carlo_margin(p, alpha=0.05, permutations=9999)
    assert margin == {"p_band": [0.0438, 0.0562], "significant": 2, "not_significant": 2}
    assert monte_carlo_margin(p, alpha=0.05, permutations=999)["p_band"] == [0.0303, 0.0697]
    nothing = monte_carlo_margin(np.array([np.nan, 0.0002, 0.9]), alpha=0.05, permutations=999)
    assert (nothing["significant"], nothing["not_significant"]) == (0, 0)


def _planted_field() -> tuple[list[str], np.ndarray, str, str]:
    """169 real H3 cells: noise + a hot blob in one corner, a cold one opposite."""
    cells = sorted(h3.grid_disk(CENTER, 7))
    rng = np.random.default_rng(5)
    values = rng.normal(10.0, 0.5, size=len(cells))
    ring = h3.grid_ring(CENTER, 5)
    hot_center, cold_center = ring[0], ring[len(ring) // 2]
    index = {cell: i for i, cell in enumerate(cells)}
    for cell in h3.grid_disk(hot_center, 2):
        if cell in index:
            values[index[cell]] += 8.0
    for cell in h3.grid_disk(cold_center, 2):
        if cell in index:
            values[index[cell]] -= 8.0
    return cells, values, hot_center, cold_center


def test_local_moran_recovers_planted_clusters() -> None:
    cells, values, hot_center, cold_center = _planted_field()
    weights, islands = hex_weights(cells)
    result = local_moran(values, weights, islands, permutations=999, seed=1, alpha=0.05)
    labels = dict(zip(cells, result["label"], strict=True))
    assert labels[hot_center] == "HH"
    assert labels[cold_center] == "LL"
    assert labels[CENTER] == "ns"  # plain noise far from both blobs
    assert set(result["label"]) <= {"HH", "LL", "HL", "LH", "ns"}
    assert result.loc[result["fdr"], "p"].max() < 0.05  # FDR is a subset
    expected = benjamini_hochberg(result["p"].to_numpy(), 0.05) & (result["p"] < 0.05)
    assert result["fdr"].tolist() == expected.tolist()


def test_local_moran_is_seeded() -> None:
    cells, values, _, _ = _planted_field()
    weights, islands = hex_weights(cells)
    first = local_moran(values, weights, islands, permutations=199, seed=3, alpha=0.05)
    second = local_moran(values, weights, islands, permutations=199, seed=3, alpha=0.05)
    pd.testing.assert_frame_equal(first, second)


def test_zero_inflated_field_flags_the_hot_blob_and_no_cold_spots() -> None:
    """The regression test for both library tie traps: with 96% zeros,
    the provision blob must be a significant HH (esda's own two-sided
    p-value cannot get there) while a zero among zeros must NOT become
    a 'cold spot' (esda's one-sided tails say p = 0.001)."""
    cells = sorted(h3.grid_disk(CENTER, 7))
    values = np.zeros(len(cells))
    index = {cell: i for i, cell in enumerate(cells)}
    for cell in h3.grid_disk(CENTER, 1):
        values[index[cell]] = 5.0
    weights, islands = hex_weights(cells)
    result = local_moran(values, weights, islands, permutations=999, seed=1, alpha=0.05)
    labels = dict(zip(cells, result["label"], strict=True))
    assert labels[CENTER] == "HH"
    assert "LL" not in set(result["label"])
    far_zero = h3.grid_ring(CENTER, 6)[0]
    assert labels[far_zero] == "ns"


def test_islands_never_cluster() -> None:
    cells, values, _, _ = _planted_field()
    far = h3.latlng_to_cell(43.40, 77.20, 9)
    weights, islands = hex_weights([*cells, far])
    result = local_moran(
        np.append(values, 99.0), weights, islands, permutations=199, seed=1, alpha=0.05
    )
    assert result["label"].iloc[-1] == "island"
    assert np.isnan(result["p"].iloc[-1])
    assert not result["fdr"].iloc[-1]


def test_global_moran_agrees_with_the_phase2_implementation() -> None:
    """Two independent implementations (esda here, the project's own
    numpy version from Phase 2) must give the same statistic."""
    cells, values, _, _ = _planted_field()
    weights, _ = hex_weights(cells)
    ours = morans_i(pd.Series(values, index=cells), n_perm=19, seed=0)
    theirs = global_moran(values, weights, permutations=99, seed=0)
    assert theirs["I"] == pytest.approx(ours.i, abs=1e-4)  # report rounds to 4 dp
    assert theirs["I"] > 0.5 and theirs["p_sim"] <= 0.05


def test_global_moran_is_seeded() -> None:
    cells, values, _, _ = _planted_field()
    weights, _ = hex_weights(cells)
    assert global_moran(values, weights, permutations=99, seed=4) == global_moran(
        values, weights, permutations=99, seed=4
    )


def test_lisa_cache_never_serves_another_weighting_values(sample_config: Config) -> None:
    """The two weightings share one LISA cache. A measure whose VALUES
    differ between them (2SFCA with calibrated demand) must be recomputed;
    only identical values may be served from the cache."""
    cells = sorted(h3.grid_disk(CENTER, 4))
    values = np.random.default_rng(2).gamma(2.0, 1.0, size=len(cells))
    first = pd.DataFrame({"h3": cells, "population": 10.0, "sfca_x_walk": values})
    second = first.assign(sfca_x_walk=values[::-1].copy())
    cache: dict = {}
    _lisa_block(sample_config, 9, first, ["sfca_x_walk"], cache)
    frames, report = _lisa_block(sample_config, 9, second, ["sfca_x_walk"], cache)
    fresh_frames, fresh_report = _lisa_block(sample_config, 9, second, ["sfca_x_walk"], {})
    pd.testing.assert_frame_equal(frames["sfca_x_walk"], fresh_frames["sfca_x_walk"])
    assert report == fresh_report
    # Same values under other weights: the cached frame itself comes back,
    # and only the resident shares are recomputed.
    heavier = second.assign(population=np.where(np.arange(len(cells)) < 30, 50.0, 10.0))
    again, again_report = _lisa_block(sample_config, 9, heavier, ["sfca_x_walk"], cache)
    assert again["sfca_x_walk"] is frames["sfca_x_walk"]
    assert again_report["sfca_x_walk"]["clusters"] == report["sfca_x_walk"]["clusters"]


# ---------------------------------------------------------------------------
# Deserts and bivariate classes
# ---------------------------------------------------------------------------


def _desert_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "h3": list("abcdefgh"),
            "district": ["north"] * 4 + ["south"] * 4,
            "population": [10.0, 20.0, 30.0, 40.0, 500.0, 600.0, 700.0, 800.0],
            "access": [9.0, 8.0, 7.0, 6.0, 1.0, 5.0, 0.5, 4.0],
        }
    )


def test_access_deserts_rule_ranking_and_info() -> None:
    frame = _desert_frame()
    deserts, info = access_deserts(frame, "access", pop_quantile=0.75, access_quantile=0.25)
    # Dense hexes: population > hex Q75 (= 625) -> g, h. Worst-served
    # quarter of RESIDENTS: hex g alone (access 0.5) holds 700 / 2700 =
    # 26% of them, so the weighted access quantile is 0.5 -> only g.
    assert info["pop_threshold"] == pytest.approx(625.0)
    assert info["access_threshold"] == pytest.approx(0.5)
    assert deserts["h3"].tolist() == ["g"]
    assert deserts["rank"].tolist() == [1]
    assert info["population"] == 700.0
    assert info["pop_share"] == pytest.approx(700 / 2700, abs=1e-4)
    assert info["population_by_district"] == {"south": 700.0}
    # The README-literal rule (hex quantile of access = 3.25) agrees here.
    assert info["access_threshold_hex_quantile"] == pytest.approx(3.25)
    assert info["n_hexes_literal_rule"] == 1
    assert info["low_spread"] is False
    assert not deserts["point_mass"].any()  # column present even when not supplied


def test_access_deserts_ties_at_zero_are_reported() -> None:
    """A zero-inflated measure: the weighted quantile is 0, every dense
    zero hex is a desert, and the info says how many residents share
    the bottom value instead of pretending it is a clean quarter."""
    frame = pd.DataFrame(
        {
            "h3": list("abcd"),
            "district": ["x"] * 4,
            "population": [100.0, 100.0, 900.0, 950.0],
            "access": [0.0, 2.0, 0.0, 3.0],
        }
    )
    deserts, info = access_deserts(frame, "access", pop_quantile=0.5, access_quantile=0.25)
    assert info["access_threshold"] == 0.0
    assert deserts["h3"].tolist() == ["c"]
    assert info["pop_share_at_or_below_access_threshold"] == pytest.approx(1000 / 2050, abs=1e-4)
    # Both quantiles are 0 here, so with the SAME comparison the literal
    # rule flags the same hex: the two rules differ only where the
    # weighting moves the threshold, never through the operator.
    assert info["access_threshold_hex_quantile"] == 0.0
    assert info["n_hexes_literal_rule"] == 1


def test_access_deserts_skip_a_measure_without_a_low_tail() -> None:
    """Free-flow car access: the worst-served quarter still reaches 99%
    of the median. A relative rule would flag a tenth of residents for
    nothing, so no deserts are reported and the info says why."""
    frame = pd.DataFrame(
        {
            "h3": list("abcdef"),
            "district": ["x"] * 6,
            "population": [10.0, 20.0, 500.0, 600.0, 700.0, 800.0],
            "access": [96.0, 96.5, 96.6, 97.0, 97.2, 97.4],
        }
    )
    deserts, info = access_deserts(frame, "access", pop_quantile=0.75, access_quantile=0.25)
    assert info["low_spread"] is True
    assert info["spread_ratio"] > 0.95
    assert len(deserts) == 0 and info["n_hexes"] == 0


def test_low_spread_guard_looks_at_the_tail_not_at_its_upper_edge() -> None:
    """Half the residents share the median value, so the 25th percentile
    EQUALS the median - yet a fifth of residents sit far below in one
    dense hex. Judging the edge would switch deserts off; judging what
    the worst-served quarter reaches on average keeps them."""
    frame = pd.DataFrame(
        {
            "h3": [f"h{i}" for i in range(14)],
            "district": ["x"] * 14,
            "population": [200.0] + [50.0] * 10 + [100.0] * 3,
            "access": [0.1] + [5.0] * 10 + [9.0] * 3,
        }
    )
    deserts, info = access_deserts(frame, "access", pop_quantile=0.75, access_quantile=0.25)
    assert info["access_threshold"] == 5.0  # the quarter's upper edge is the median
    assert info["spread_ratio"] == pytest.approx(1.08 / 5.0, abs=1e-4)
    assert info["low_spread"] is False
    assert deserts["h3"].tolist() == ["h0"]


def test_access_deserts_carry_the_point_mass_flag() -> None:
    frame = _desert_frame().assign(point_mass=[False] * 6 + [True, False])
    deserts, info = access_deserts(frame, "access", pop_quantile=0.75, access_quantile=0.25)
    assert deserts["point_mass"].tolist() == [True]
    assert info["n_point_mass_hexes"] == 1


def test_tercile_classes_can_run_across_residents() -> None:
    """One hex holding most residents: across hexes its value is the top
    third, across residents it IS the bottom third."""
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    weights = pd.Series([900.0, 40.0, 30.0, 30.0])
    assert tercile_classes(values).tolist() == [1, 1, 2, 3]
    assert tercile_classes(values, weights).tolist() == [1, 3, 3, 3]


def test_tercile_classes_keep_ties_together() -> None:
    classes = tercile_classes(pd.Series([0.0, 0.0, 0.0, 0.0, 1.0, 2.0]))
    assert classes.iloc[:4].nunique() == 1  # identical values, identical class
    assert classes.tolist() == [1, 1, 1, 1, 3, 3]


def test_bivariate_classes_labels() -> None:
    population = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    access = pd.Series([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    classes = bivariate_classes(population, access)
    assert classes.iloc[0] == "1-3"  # few people, plenty of access
    assert classes.iloc[-1] == "3-1"  # many people, little access
    assert classes.str.fullmatch(r"[123]-[123]").all()


def test_bivariate_access_axis_is_weighted_by_residents() -> None:
    """The access terciles follow residents (as the desert rule does): a
    dense hex with the second-lowest value is still among the
    worst-served third of PEOPLE."""
    population = pd.Series([10.0, 900.0, 20.0, 30.0, 40.0, 50.0])
    access = pd.Series([0.1, 0.2, 5.0, 6.0, 7.0, 8.0])
    classes = bivariate_classes(population, access)
    assert classes.iloc[1] == "3-1"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def test_district_table_weighted_and_ranked() -> None:
    frame = pd.DataFrame(
        {
            "district": ["a", "a", "b", "b", None],
            "population": [100.0, 300.0, 50.0, 50.0, 999.0],
            "access": [10.0, 20.0, 40.0, 60.0, 0.0],
        }
    )
    table = district_table(frame, "access")
    assert table["district"].tolist() == ["b", "a"]  # rank order
    assert table["rank"].tolist() == [1, 2]
    by_district = table.set_index("district")
    assert by_district.loc["a", "mean"] == pytest.approx(17.5)  # (1000 + 6000) / 400
    assert by_district.loc["a", "median"] == 20.0
    assert by_district.loc["b", "population"] == 100.0


def test_below_cutoff_shares() -> None:
    frame = pd.DataFrame({"population": [10.0, 30.0, 60.0], "access": [0.5, 4.0, 12.0]})
    shares = below_cutoff_shares(frame, "access", (1.0, 5.0, 25.0))
    assert shares == {"1": 0.1, "5": 0.4, "25": 1.0}


def test_nearest_tables_count_missing_as_beyond() -> None:
    frame = pd.DataFrame(
        {
            "district": ["a", "a", "b", pd.NA],
            "population": [100.0, 100.0, 100.0, 100.0],
            "near_clinic_walk_min": [5.0, 12.0, np.nan, 30.0],
        }
    )
    table, summary = nearest_tables(frame, (10, 15))
    city = table.loc[table["district"] == CITY_SCOPE].set_index("threshold_min")
    assert city.loc[10, "pop_share_within"] == pytest.approx(0.25)
    assert city.loc[15, "pop_share_within"] == pytest.approx(0.50)  # NaN is beyond
    district_b = table.loc[table["district"] == "b"]
    assert (district_b["pop_share_within"] == 0.0).all()
    assert set(table["district"]) == {CITY_SCOPE, "a", "b"}  # NA district: city rows only
    assert summary["clinic"]["pop_share_unreachable"] == pytest.approx(0.25)
    assert summary["clinic"]["median_min"] == 12.0


def test_mode_gap_ratio_and_undefined() -> None:
    table = pd.DataFrame(
        {
            "acc_jobs_transit_30": [50.0, 0.0, 10.0],
            "acc_jobs_car_30": [100.0, 200.0, 0.0],
        }
    )
    gaps = mode_gap(table, (30,))
    ratio = gaps["modegap_transit_car_30"]
    assert ratio.iloc[0] == pytest.approx(0.5)
    assert ratio.iloc[1] == 0.0  # the car reaches jobs, transit none
    assert np.isnan(ratio.iloc[2])  # no car access: undefined, not zero


def test_center_gradient_bands() -> None:
    frame = pd.DataFrame(
        {
            "center_km": [0.0, 1.0, 3.0, 9.0],
            "population": [10.0, 30.0, 40.0, 20.0],
            "access": [100.0, 60.0, 20.0, 5.0],
        }
    )
    rows = center_gradient(frame, ["access"], (2.5, 5.0))
    assert [row["band"] for row in rows] == ["0-2.5 km", "2.5-5 km", ">5 km"]
    assert rows[0]["population"] == 40.0  # the centre hex at distance 0 is included
    assert rows[0]["access"] == pytest.approx(70.0)  # (1000 + 1800) / 40
    assert sum(row["pop_share"] for row in rows) == pytest.approx(1.0)
    assert rows[2]["n_hexes"] == 1


# ---------------------------------------------------------------------------
# Population-weight calibration
# ---------------------------------------------------------------------------


def test_calibrated_weights_follow_official_shares_and_keep_the_total() -> None:
    population = pd.Series([100.0, 300.0, 600.0, 50.0])
    district = pd.Series(["a", "a", "b", pd.NA], dtype="string")
    shares = pd.Series({"a": 0.7, "b": 0.3, "c": 0.0})  # c has no hexes here
    weights, factors = calibrated_weights(population, district, shares)
    # Raster shares over the present districts: a 0.4, b 0.6 -> factors 1.75 and 0.5.
    assert factors == pytest.approx({"a": 1.75, "b": 0.5})
    # The hex without a district keeps its raster value.
    assert weights.tolist() == pytest.approx([175.0, 525.0, 300.0, 50.0])
    assert weights[:3].sum() == pytest.approx(population[:3].sum())  # total preserved
    assert weights[:2].sum() / weights[:3].sum() == pytest.approx(0.7)


def test_calibrated_weights_single_district_is_a_no_op() -> None:
    population = pd.Series([10.0, 20.0])
    district = pd.Series(["a", "a"], dtype="string")
    weights, factors = calibrated_weights(population, district, pd.Series({"a": 0.2, "b": 0.8}))
    assert factors == {"a": 1.0}
    assert weights.tolist() == [10.0, 20.0]


def test_calibrated_weights_ignore_districts_without_official_data() -> None:
    population = pd.Series([10.0, 30.0])
    district = pd.Series(["a", "zzz"], dtype="string")
    weights, factors = calibrated_weights(population, district, pd.Series({"a": 1.0}))
    assert factors["zzz"] == 1.0 and weights[1] == 30.0


def _stats_config(sample_config: Config, tmp_path, rows: list[tuple]) -> Config:
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = pd.DataFrame(rows, columns=["district", "year", "month", "indicator", "value"])
    frame["unit"] = "persons"
    frame["source"] = "synthetic-for-tests"
    frame["notes"] = ""
    frame.to_csv(raw / "district_stats.csv", index=False)
    return dataclasses.replace(
        sample_config, paths=dataclasses.replace(sample_config.paths, raw=raw)
    )


def test_district_resident_shares_use_the_latest_complete_year(
    sample_config: Config, tmp_path
) -> None:
    keys = [district.key for district in sample_config.districts]
    rows = [(key, 2023, 12, "employed_residents", 100.0 * (i + 1)) for i, key in enumerate(keys)]
    rows += [(key, 2024, 12, "employed_residents", 50.0) for key in keys[:-1]]  # one missing
    config = _stats_config(sample_config, tmp_path, rows)
    result = district_resident_shares(config)
    assert result.year == 2023  # 2024 does not cover every district
    assert result.shares.sum() == pytest.approx(1.0)
    assert result.shares[keys[-1]] == pytest.approx(8 / 36)
    # The sample's fake numbers are the one status the project can vouch for.
    assert result.synthetic is True and result.source == "synthetic-for-tests"
    assert "SYNTHETIC test numbers" in result.method


def test_district_resident_shares_absent_indicator(sample_config: Config, tmp_path) -> None:
    rows = [("almaly", 2024, 12, "payroll_jobs_located", 1000.0)]
    assert district_resident_shares(_stats_config(sample_config, tmp_path, rows)) is None


def test_district_resident_shares_quote_the_source_without_certifying_it(
    sample_config: Config, tmp_path
) -> None:
    """A hand-collected file is neither 'official' nor 'synthetic' as far
    as the code can know: the source is quoted verbatim and that is all."""
    keys = [district.key for district in sample_config.districts]
    rows = [(key, 2025, 12, "employed_residents", 100.0) for key in keys]
    config = _stats_config(sample_config, tmp_path, rows)
    stats_path = config.paths.raw / "district_stats.csv"
    frame = pd.read_csv(stats_path)
    frame["source"] = "stat.gov.kz bulletin, collected by hand"
    frame.to_csv(stats_path, index=False)
    result = district_resident_shares(config)
    assert result.synthetic is False
    assert "stat.gov.kz bulletin, collected by hand" in result.method
    assert "2025" in result.method and "SYNTHETIC" not in result.method
    assert "official" not in result.method.lower()


def test_district_resident_shares_survive_a_file_without_a_source_column(
    sample_config: Config, tmp_path
) -> None:
    """`source` is an optional column of the statistics file."""
    keys = [district.key for district in sample_config.districts]
    rows = [(key, 2025, 12, "employed_residents", 100.0) for key in keys]
    config = _stats_config(sample_config, tmp_path, rows)
    stats_path = config.paths.raw / "district_stats.csv"
    pd.read_csv(stats_path).drop(columns=["source"]).to_csv(stats_path, index=False)
    result = district_resident_shares(config)
    assert result.source == "unknown" and result.synthetic is False


def test_district_resident_shares_missing_file(sample_config: Config, tmp_path) -> None:
    config = dataclasses.replace(
        sample_config, paths=dataclasses.replace(sample_config.paths, raw=tmp_path / "nope")
    )
    assert district_resident_shares(config) is None


def test_apply_factors_reaches_every_hex() -> None:
    population = pd.Series([10.0, 20.0, 30.0])
    district = pd.Series(["a", pd.NA, "b"], dtype="string")
    assert apply_factors(population, district, {"a": 2.0}).tolist() == [20.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# Report wording: the sentence the README asks for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "calibrated", "expected"),
    [
        (0.6155, None, "62%"),
        (0.6155, 0.5148, "Between 51% and 62%"),
        (0.5148, 0.6155, "Between 51% and 62%"),  # order does not matter
        (0.2150, 0.2241, "22%"),  # 0.9 pp apart but both print as 22%
        (0.2249, 0.2251, "Between 22% and 23%"),  # 0.02 pp apart across a boundary
    ],
)
def test_share_text(raw: float, calibrated: float | None, expected: str) -> None:
    """The choice between one number and a range is made on what is
    PRINTED, so a range can never show two identical ends."""
    assert _share_text(raw, calibrated) == expected


def test_jsonable_keeps_infinity_meaningful() -> None:
    data = {"a": np.float64("nan"), "b": float("inf"), "c": np.int64(3), "d": [np.bool_(True)]}
    assert _jsonable(data) == {"a": None, "b": "inf", "c": 3, "d": [True]}


def _headline_world(sample_config: Config) -> tuple[Config, _Analysis, _Analysis]:
    """Six hexes in four districts. On the raster the affected residents
    rank b > a > c > d; calibrated they rank d > a > c > b - so b is in
    the raster top three only, d in the calibrated one only."""
    domain = pd.DataFrame(
        {
            "h3": list("uvwxyz"),
            "district": pd.array(["a", "a", "b", "b", "c", "d"], dtype="string"),
            "accpct_jobs_transit_45": [50.0, 1.0, 2.0, 3.0, 4.0, 4.5],
            "population": [560.0, 110.0, 150.0, 50.0, 90.0, 40.0],
        }
    )
    calibrated = domain.assign(population=[560.0, 100.0, 15.0, 5.0, 80.0, 240.0])
    config = dataclasses.replace(
        sample_config, equity=dataclasses.replace(sample_config.equity, headline_cutoff_pct=5.0)
    )

    def analysis(frame: pd.DataFrame, weighting: str) -> _Analysis:
        return _Analysis(weighting=weighting, res=8, domain=frame, hexes=frame, block={})

    return config, analysis(domain, "raster"), analysis(calibrated, "calibrated")


def test_headline_states_a_range_and_names_only_stable_districts(
    sample_config: Config,
) -> None:
    config, raster, calibrated = _headline_world(sample_config)
    headline = _headline(config, raster, calibrated, "accpct_jobs_transit_45")
    # Below the 5% cutoff: every hex but u. Raster 440 of 1000, calibrated 440 of 1000.
    assert headline["pop_share"] == pytest.approx(0.44)
    assert headline["pop_share_calibrated"] == pytest.approx(0.44)
    assert headline["sentence"].startswith("44% of residents can reach fewer than 5% of city jobs")
    assert "45 min by transit (approximate network, a lower bound)" in headline["sentence"]
    # Raster order: b (200), a (110), c (90), d (40). Calibrated: d (240), a (100), c (80), b (20).
    assert list(headline["top_districts_raster"]) == ["b", "a", "c"]
    assert list(headline["top_districts_calibrated"]) == ["d", "a", "c"]
    # Only districts in BOTH top threes are named: b and d each appear in one.
    assert headline["districts_named"] == ["a", "c"]
    assert headline["sentence"].endswith("concentrated in A and C.")
    assert headline["district_rates"]["a"] == pytest.approx(110 / 670, abs=1e-4)
    # The share is the same number the cutoff table reports for this frame.
    shares = below_cutoff_shares(raster.domain, "accpct_jobs_transit_45", (5.0,))
    assert headline["pop_share"] == shares["5"] == 0.44


def test_headline_hedges_only_when_the_weightings_print_differently(
    sample_config: Config,
) -> None:
    config, raster, calibrated = _headline_world(sample_config)
    shifted = dataclasses.replace(
        calibrated,
        domain=calibrated.domain.assign(population=[900.0, 20.0, 30.0, 10.0, 25.0, 15.0]),
    )
    headline = _headline(config, raster, shifted, "accpct_jobs_transit_45")
    assert headline["pop_share_calibrated"] == pytest.approx(0.10)
    assert headline["sentence"].startswith(
        "Between 10% and 44% of residents - depending on how the population raster is "
        "calibrated - can reach fewer than 5% of city jobs"
    )
    single = _headline(config, raster, None, "accpct_jobs_transit_45")
    assert single["pop_share_calibrated"] is None
    assert single["district_rates_calibrated"] is None
    assert single["sentence"].startswith("44% of residents can reach")
    assert single["districts_named"] == ["b", "a", "c"]


def test_headline_uses_each_weighting_own_values(sample_config: Config) -> None:
    """A measure that contains residents (2SFCA) has DIFFERENT values in
    the calibrated run: the calibrated share must come from those, not
    from the raster values paired with calibrated weights."""
    config, raster, calibrated = _headline_world(sample_config)
    moved = dataclasses.replace(
        calibrated,
        domain=calibrated.domain.assign(accpct_jobs_transit_45=[50.0, 60.0, 2.0, 3.0, 70.0, 80.0]),
    )
    headline = _headline(config, raster, moved, "accpct_jobs_transit_45")
    assert headline["pop_share"] == pytest.approx(0.44)
    assert headline["pop_share_calibrated"] == pytest.approx(0.02)  # only district b: 15 + 5
    assert headline["district_rates_calibrated"]["a"] == 0.0
    assert headline["district_rates"]["a"] == pytest.approx(110 / 670, abs=1e-4)


def test_headline_falls_back_for_a_non_job_measure(sample_config: Config) -> None:
    config, raster, _ = _headline_world(sample_config)
    renamed = dataclasses.replace(
        raster, domain=raster.domain.rename(columns={"accpct_jobs_transit_45": "sfca_x_walk"})
    )
    headline = _headline(config, renamed, None, "sfca_x_walk")
    assert headline["sentence"].startswith("44% of residents have sfca_x_walk below 5")


def test_companions_state_ranges_per_facility(sample_config: Config) -> None:
    def analysis(shares: dict[str, float], weighting: str) -> _Analysis:
        nearest = pd.DataFrame(
            {
                "district": "ALL",
                "facility": list(shares),
                "threshold_min": 15,
                "pop_share_within": list(shares.values()),
            }
        )
        empty = pd.DataFrame()
        return _Analysis(
            weighting=weighting, res=9, domain=empty, hexes=empty, block={}, nearest=nearest
        )

    raster = analysis({"clinic": 0.59, "park": 0.216}, "raster")
    calibrated = analysis({"clinic": 0.65, "park": 0.221}, "calibrated")
    lines = _companions(sample_config, raster, calibrated)
    assert lines[0].startswith("Between 59% and 65% of residents live within a 15-minute walk")
    assert "raster vs calibrated residents" in lines[0]
    assert lines[1].startswith("22% of residents") and "raster vs" not in lines[1]
