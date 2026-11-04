"""Unit tests for the Phase 2 feature blocks on constructed inputs.

Each test builds the smallest input that pins the behaviour down:
known geometry in known cells, so expected values are computable by
hand. The sample-pipeline test covers the full assembly.
"""

from __future__ import annotations

import dataclasses
import math

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from almaty_access.config import Config, SubcentersConfig
from almaty_access.features.build import _assert_finite, log1p_columns, near_duplicate_drops
from almaty_access.features.built_form import built_form_features
from almaty_access.features.centrality import find_subcenters
from almaty_access.features.context import kring_features
from almaty_access.features.landuse import landuse_features
from almaty_access.features.network import (
    highway_group,
    network_features,
    node_betweenness,
    node_degrees,
    undirected_edges,
)
from almaty_access.features.pois import poi_features, shannon_entropy
from almaty_access.features.population import population_features
from almaty_access.features.spatial import morans_i
from almaty_access.features.transit import _distance_to_nearest, metro_stations, rail_stations
from almaty_access.prepare.population import cell_polygon

CRS = "EPSG:4326"
METRIC = "EPSG:32643"

CELL_A = h3.latlng_to_cell(43.24, 76.93, 9)
CELL_B = h3.grid_ring(CELL_A, 1)[0]


def make_grid(cells: list[str]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"h3": pd.Series(cells, dtype="string")},
        geometry=[cell_polygon(cell) for cell in cells],
        crs=CRS,
    )


def to_wgs(geometry, crs: str = METRIC) -> gpd.GeoSeries:
    return gpd.GeoSeries(geometry, crs=crs).to_crs(CRS)


def metric_center(cell: str) -> tuple[float, float]:
    lat, lon = h3.cell_to_latlng(cell)
    point = gpd.GeoSeries([Point(lon, lat)], crs=CRS).to_crs(METRIC).iloc[0]
    return point.x, point.y


# --------------------------------------------------------------------------
# built form
# --------------------------------------------------------------------------


def test_built_form_mass_is_conserved_and_split() -> None:
    grid = make_grid([CELL_A, CELL_B])
    ax, ay = metric_center(CELL_A)
    bx, by = metric_center(CELL_B)
    inside = box(ax - 10, ay - 10, ax + 10, ay + 10)  # 400 m2, fully in A
    straddling = LineString([(ax, ay), (bx, by)]).buffer(5, cap_style=2)  # crosses the border
    buildings = gpd.GeoDataFrame(
        {
            "function": ["commercial", "industrial"],
            "levels": [5.0, 1.0],
            "gfa_m2": [2000.0, 1000.0],
        },
        geometry=to_wgs([inside, straddling]),
        crs=CRS,
    )

    out = built_form_features(buildings, grid, 9, METRIC, large_gfa_m2=1500.0)

    assert out.loc[CELL_A, "bf_gfa_commercial_m2"] == pytest.approx(2000.0, rel=1e-3)
    # The straddling building's GFA lands in both cells and sums to 1000.
    industrial = out["bf_gfa_industrial_m2"]
    assert industrial.loc[CELL_A] > 0 and industrial.loc[CELL_B] > 0
    assert industrial.sum() == pytest.approx(1000.0, rel=1e-3)
    assert out["bf_gfa_m2"].sum() == pytest.approx(3000.0, rel=1e-3)
    # Counts are indivisible: 2 buildings in total, assigned to one hex each.
    assert out["bf_bldg_cnt"].sum() == 2
    # Only the 2000 m2 building exceeds the large threshold.
    assert out["bf_gfa_large_m2"].sum() == pytest.approx(2000.0, rel=1e-3)
    assert 0 < out.loc[CELL_A, "bf_share_large_gfa"] <= 1


def test_built_form_empty_hex_is_zero_not_nan() -> None:
    grid = make_grid([CELL_A, CELL_B])
    ax, ay = metric_center(CELL_A)
    buildings = gpd.GeoDataFrame(
        {"function": ["public"], "levels": [2.0], "gfa_m2": [500.0]},
        geometry=to_wgs([box(ax - 5, ay - 5, ax + 5, ay + 5)]),
        crs=CRS,
    )
    out = built_form_features(buildings, grid, 9, METRIC, large_gfa_m2=2000.0)
    row = out.loc[CELL_B]
    assert (row == 0).all()
    assert not out.isna().any().any()


# --------------------------------------------------------------------------
# POIs
# --------------------------------------------------------------------------


def test_shannon_entropy_known_values() -> None:
    counts = pd.DataFrame({"a": [2, 4, 0], "b": [2, 0, 0]})
    entropy = shannon_entropy(counts)
    assert entropy.iloc[0] == pytest.approx(math.log(2))  # 50/50 mix
    assert entropy.iloc[1] == 0.0  # single category
    assert entropy.iloc[2] == 0.0  # empty hex


def test_poi_features_counts_and_shares() -> None:
    grid = make_grid([CELL_A, CELL_B])
    lat_a, lon_a = h3.cell_to_latlng(CELL_A)
    pois = gpd.GeoDataFrame(
        {"category": pd.Series(["office", "grocery", "office"], dtype="string")},
        geometry=[Point(lon_a, lat_a), Point(lon_a, lat_a), Point(lon_a, lat_a)],
        crs=CRS,
    )
    out = poi_features(pois, grid, 9, ["office", "grocery", "finance"])
    assert out.loc[CELL_A, "poi_cnt_office"] == 2
    assert out.loc[CELL_A, "poi_cnt_total"] == 3
    assert out.loc[CELL_B, "poi_cnt_total"] == 0
    assert out.loc[CELL_A, "poi_share_office"] == pytest.approx(2 / 3)
    assert out.loc[CELL_A, "poi_share_commercial"] == pytest.approx(1 / 3)
    assert out.loc[CELL_B, "poi_share_office"] == 0.0
    assert out.loc[CELL_A, "poi_entropy"] > 0


# --------------------------------------------------------------------------
# land use
# --------------------------------------------------------------------------


def test_landuse_shares_partial_cover_and_park() -> None:
    grid = make_grid([CELL_A, CELL_B])
    hex_a = cell_polygon(CELL_A)
    min_x, min_y, max_x, max_y = hex_a.bounds
    # Clip to the hexagon: the raw half-bbox would spill into the
    # interlocking neighbour hex and legitimately give it a share too.
    west_half = box(min_x - 0.001, min_y - 0.001, (min_x + max_x) / 2, max_y + 0.001).intersection(
        hex_a
    )
    landuse = gpd.GeoDataFrame(
        {"landuse": ["industrial", None], "leisure": [None, "park"]},
        geometry=[west_half, cell_polygon(CELL_B)],
        crs=CRS,
    )
    out = landuse_features(landuse, grid, METRIC)
    assert 0.3 < out.loc[CELL_A, "lu_share_industrial"] < 0.7
    assert out.loc[CELL_B, "lu_share_green"] == pytest.approx(1.0, abs=0.05)
    assert out.loc[CELL_B, "lu_share_industrial"] == pytest.approx(0.0, abs=1e-9)
    assert ((out >= 0) & (out <= 1)).all().all()


def test_landuse_precedence_specific_beats_blanket() -> None:
    """A residential blanket over an industrial zone must not double-count."""
    grid = make_grid([CELL_A])
    hex_a = cell_polygon(CELL_A)
    min_x, min_y, max_x, max_y = hex_a.bounds
    west_half = box(min_x - 0.001, min_y - 0.001, (min_x + max_x) / 2, max_y + 0.001).intersection(
        hex_a
    )
    landuse = gpd.GeoDataFrame(
        {"landuse": ["residential", "industrial"], "leisure": [None, None]},
        geometry=[hex_a, west_half],  # residential covers the WHOLE hex
        crs=CRS,
    )
    out = landuse_features(landuse, grid, METRIC)
    industrial = out.loc[CELL_A, "lu_share_industrial"]
    residential = out.loc[CELL_A, "lu_share_residential"]
    assert 0.3 < industrial < 0.7
    assert residential == pytest.approx(1.0 - industrial, abs=0.02)
    assert industrial + residential <= 1.001


def test_landuse_overlapping_same_group_does_not_double_count() -> None:
    grid = make_grid([CELL_A])
    polygon = cell_polygon(CELL_A)
    landuse = gpd.GeoDataFrame(
        {"landuse": ["residential", "residential"], "leisure": [None, None]},
        geometry=[polygon, polygon],  # the same area mapped twice
        crs=CRS,
    )
    out = landuse_features(landuse, grid, METRIC)
    assert out.loc[CELL_A, "lu_share_residential"] == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------


def test_highway_group_mapping() -> None:
    values = pd.Series(["primary", "residential;service", "footway", None])
    groups = highway_group(values)
    assert list(groups) == ["major", "local", "other", "other"]


def test_node_degrees_counts_distinct_neighbours() -> None:
    edges = pd.DataFrame({"u": [1, 2, 1, 1], "v": [2, 1, 3, 4]})
    degrees = node_degrees(edges)
    assert degrees[1] == 3
    assert degrees[2] == 1


def test_node_betweenness_middle_of_path_is_highest() -> None:
    edges = pd.DataFrame({"u": [1, 2, 3], "v": [2, 3, 4], "length_m": [100.0, 100.0, 100.0]})
    betweenness = node_betweenness(edges, k=4, seed=42)
    assert betweenness[2] > betweenness[1]
    assert betweenness[3] > betweenness[4]


def test_node_degrees_ignore_self_loops() -> None:
    edges = pd.DataFrame({"u": [1, 1, 1], "v": [1, 2, 3]})  # self-loop + 2 neighbours
    assert node_degrees(edges)[1] == 2


def test_undirected_edges_drops_reverse_twins_keeps_parallels() -> None:
    geometry = [LineString([(0, 0), (1, 0)])] * 4
    edges = gpd.GeoDataFrame(
        {
            # (1,2)/(2,1): a two-way street stored as directed twins.
            # (1,3)/(3,1) with different lengths: a dual carriageway.
            "u": [1, 2, 1, 3],
            "v": [2, 1, 3, 1],
            "key": [0, 0, 0, 0],
            "length_m": [100.0, 100.0, 70.0, 80.0],
        },
        geometry=geometry,
        crs=METRIC,
    )
    result = undirected_edges(edges)
    assert len(result) == 3
    assert (result["length_m"] == [100.0, 70.0, 80.0]).all()


def test_two_way_street_is_counted_once_in_length_features() -> None:
    grid = make_grid([CELL_A])
    ax, ay = metric_center(CELL_A)
    line = LineString([(ax - 50, ay), (ax + 50, ay)])
    edges = gpd.GeoDataFrame(
        {
            "u": [1, 2],
            "v": [2, 1],
            "key": [0, 0],
            "length_m": [100.0, 100.0],
            "highway": pd.Series(["residential", "residential"], dtype="string"),
        },
        geometry=[line, line.reverse()],
        crs=METRIC,
    ).to_crs(CRS)
    points = gpd.GeoSeries([Point(ax - 50, ay), Point(ax + 50, ay)], crs=METRIC).to_crs(CRS)
    nodes = gpd.GeoDataFrame(
        {"node_id": [1, 2], "x": points.x, "y": points.y, "elevation_m": [700.0, 700.0]},
        geometry=points,
        crs=CRS,
    )
    out = network_features(
        nodes, edges, edges, grid, 9, METRIC, pd.Series(dtype=float), fallback_distance_m=9999.0
    )
    assert out.loc[CELL_A, "net_len_local_m"] == pytest.approx(100.0, rel=0.01)
    assert out.loc[CELL_A, "net_len_walk_m"] == pytest.approx(100.0, rel=0.01)
    # No major roads in this toy network: the documented fallback applies.
    assert out.loc[CELL_A, "net_dist_major_m"] == 9999.0


# --------------------------------------------------------------------------
# transit
# --------------------------------------------------------------------------


def test_distance_fallback_when_no_targets_exist() -> None:
    centroids = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(100, 0)], crs=METRIC)
    empty = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=METRIC))
    index = pd.Index(["a", "b"], name="h3")
    centroids.index = index
    distances = _distance_to_nearest(centroids, empty, index, fallback_m=2812.1, what="metro")
    assert (distances == 2812.1).all()


def test_metro_and_rail_station_filters() -> None:
    pois = gpd.GeoDataFrame(
        {
            "railway": pd.Series(["station", "station", "halt", None], dtype="string"),
            "station": pd.Series(["subway", None, None, "subway"], dtype="string"),
        },
        geometry=[Point(76.9, 43.2)] * 4,
        crs=CRS,
    )
    assert len(metro_stations(pois)) == 1  # railway=station AND station=subway
    assert len(rail_stations(pois)) == 2  # any railway=station


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------


def test_population_ratio_uses_denominator_floor() -> None:
    grid = make_grid([CELL_A, CELL_B])
    population = gpd.GeoDataFrame(
        {"h3": pd.Series([CELL_A], dtype="string"), "population": [100.0]},
        geometry=[cell_polygon(CELL_A)],
        crs=CRS,
    )
    built = pd.DataFrame(
        {
            "bf_gfa_commercial_m2": [0.0, 2000.0],
            "bf_gfa_industrial_m2": [0.0, 0.0],
            "bf_gfa_public_m2": [0.0, 0.0],
        },
        index=pd.Index([CELL_A, CELL_B], name="h3"),
    )
    out = population_features(population, built, grid, 9)
    # No workspace -> the 100 m2 floor kicks in: 100 / 100 * 1000 = 1000.
    assert out.loc[CELL_A, "pop_per_1k_nonres_gfa"] == pytest.approx(1000.0)
    assert out.loc[CELL_B, "pop_residents"] == 0.0
    assert out.loc[CELL_B, "pop_per_1k_nonres_gfa"] == 0.0
    assert np.isfinite(out.to_numpy()).all()


# --------------------------------------------------------------------------
# sub-centres
# --------------------------------------------------------------------------


def test_find_subcenters_excludes_the_centre_cluster(sample_config: Config) -> None:
    features = dataclasses.replace(
        sample_config.features,
        subcenters=SubcentersConfig(eps_m=300.0, min_samples=10, max_centers=3),
    )
    config = dataclasses.replace(sample_config, features=features)
    rng = np.random.default_rng(1)

    def cluster(lon: float, lat: float, n: int) -> list[Point]:
        return [
            Point(lon + dx, lat + dy)
            for dx, dy in rng.normal(0, 0.0005, size=(n, 2))  # ~50 m jitter
        ]

    center_lon, center_lat = config.features.center_lon, config.features.center_lat
    points = cluster(center_lon, center_lat, 30) + cluster(center_lon + 0.03, center_lat, 20)
    pois = gpd.GeoDataFrame(
        {"category": pd.Series(["office"] * len(points), dtype="string")},
        geometry=points,
        crs=CRS,
    )

    subcenters = find_subcenters(pois, config)

    assert len(subcenters) == 1  # the centre cluster itself is excluded
    assert subcenters.iloc[0]["n_pois"] == 20
    assert subcenters.crs.to_epsg() == 4326
    assert abs(subcenters.geometry.iloc[0].x - (center_lon + 0.03)) < 0.01


# --------------------------------------------------------------------------
# k-ring context
# --------------------------------------------------------------------------


def test_kring_sums_spread_a_point_value() -> None:
    cells = sorted(h3.grid_disk(CELL_A, 1))
    features = pd.DataFrame({"v": 0.0}, index=pd.Index(cells, name="h3"))
    features.loc[CELL_A, "v"] = 7.0

    out = kring_features(features, ["v"], ks=[1])

    # Every cell's 1-disk contains the centre, so every sum sees the 7.
    assert (out["ctx_k1_sum_v"] == 7.0).all()
    disk_size = len(set(h3.grid_disk(CELL_A, 1)) & set(cells))
    assert out.loc[CELL_A, "ctx_k1_mean_v"] == pytest.approx(7.0 / disk_size)


def test_kring_unknown_column_raises() -> None:
    features = pd.DataFrame({"v": [1.0]}, index=pd.Index([CELL_A], name="h3"))
    with pytest.raises(ValueError, match="context_columns"):
        kring_features(features, ["nope"], ks=[1])


# --------------------------------------------------------------------------
# near-duplicate pruning + transforms
# --------------------------------------------------------------------------


def test_near_duplicate_drops_later_column_of_a_pair() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(100)
    table = pd.DataFrame({"zz_a": a, "zz_b": 2 * a, "zz_c": rng.random(100)})
    drops, kept_pairs = near_duplicate_drops(
        table, prefixes=("none_",), threshold=0.95, protected=frozenset()
    )
    assert [item["dropped"] for item in drops] == ["zz_b"]
    assert drops[0]["kept"] == "zz_a"
    assert drops[0]["r"] == pytest.approx(1.0)
    assert kept_pairs == []


def test_near_duplicate_drops_never_remove_protected_features() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(100)
    table = pd.DataFrame({"zz_a": a, "zz_prot": 2 * a, "zz_core": 3 * a})
    protected = frozenset({"zz_prot", "zz_core"})
    drops, kept_pairs = near_duplicate_drops(
        table, prefixes=("none_",), threshold=0.95, protected=protected
    )
    # The unprotected zz_a loses to the protected zz_prot even though it
    # comes first; the two protected columns survive as a recorded pair.
    assert [item["dropped"] for item in drops] == ["zz_a"]
    assert drops[0]["kept"] == "zz_prot"
    assert {pair["a"] for pair in kept_pairs} == {"zz_prot"}
    assert {pair["b"] for pair in kept_pairs} == {"zz_core"}


def test_near_duplicate_drops_correlate_on_the_log1p_view() -> None:
    rng = np.random.default_rng(7)
    u = rng.uniform(0, 8, 400)
    x = np.expm1(u)  # heavy-tailed; log1p(x) == u exactly
    y = np.expm1(np.clip(u + rng.normal(0, 0.5, 400), 0, None))  # counts stay >= 0
    table = pd.DataFrame({"poi_cnt_x": x, "poi_cnt_y": y})
    raw_r = abs(table.corr().iloc[0, 1])
    log_r = abs(np.corrcoef(np.log1p(x), np.log1p(y))[0, 1])
    assert log_r > 0.95 > raw_r, "test construction: views must disagree"
    drops, _ = near_duplicate_drops(
        table, prefixes=("poi_cnt_",), threshold=0.95, protected=frozenset()
    )
    assert [item["dropped"] for item in drops] == ["poi_cnt_y"]


def test_near_duplicate_drops_skip_log1p_for_negative_columns() -> None:
    rng = np.random.default_rng(1)
    a = rng.random(50)
    table = pd.DataFrame({"poi_cnt_a": a, "poi_cnt_neg": -a})  # log1p impossible
    drops, _ = near_duplicate_drops(
        table, prefixes=("poi_cnt_",), threshold=0.95, protected=frozenset()
    )
    assert [item["dropped"] for item in drops] == ["poi_cnt_neg"]  # |r| = 1 raw


def test_assert_finite_raises_naming_the_column() -> None:
    _assert_finite(pd.DataFrame({"x": [1.0, 2.0]}), 8)  # clean table passes
    with pytest.raises(RuntimeError, match="x"):
        _assert_finite(pd.DataFrame({"x": [1.0, np.nan]}), 8)
    with pytest.raises(RuntimeError, match="x"):
        _assert_finite(pd.DataFrame({"x": [1.0, np.inf]}), 8)


def test_log1p_columns_expand_prefixes() -> None:
    table = pd.DataFrame({"poi_cnt_a": [1], "ter_elev_mean_m": [700], "pop_residents": [5]})
    assert log1p_columns(table, ("poi_cnt_", "pop_")) == ["poi_cnt_a", "pop_residents"]


# --------------------------------------------------------------------------
# Moran's I
# --------------------------------------------------------------------------


def test_morans_i_gradient_clusters_noise_does_not() -> None:
    cells = sorted(h3.grid_disk(CELL_A, 3))
    index = pd.Index(cells, name="h3")
    latitudes = pd.Series([h3.cell_to_latlng(cell)[0] for cell in cells], index=index)

    smooth = morans_i(latitudes, n_perm=199, seed=42)
    assert smooth.i > 0.3
    assert smooth.p_sim < 0.05

    rng = np.random.default_rng(42)
    noise = pd.Series(rng.normal(size=len(cells)), index=index)
    random = morans_i(noise, n_perm=199, seed=42)
    assert abs(random.i) < 0.3
    assert random.p_sim > 0.05


def test_morans_i_constant_is_nan_and_nan_raises() -> None:
    cells = sorted(h3.grid_disk(CELL_A, 2))
    index = pd.Index(cells, name="h3")
    constant = morans_i(pd.Series(1.0, index=index))
    assert math.isnan(constant.i)
    with pytest.raises(ValueError, match="NaN"):
        morans_i(pd.Series(np.nan, index=index))
