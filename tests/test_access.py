"""Unit tests for the Phase 4 building blocks (all offline, toy data).

The travel-time math is checked against hand-computed values: speeds on
known grades, Dijkstra on a four-node line, boarding/transfer costs on a
two-stop bus, and the accessibility measures on tiny matrices where the
right answer fits on paper.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.sparse.csgraph import dijkstra

from almaty_access.access.graph import ModeGraph, build_graph, snap
from almaty_access.access.matrix import hex_times, nearest_node_minutes
from almaty_access.access.measures import SFCA_PER, cumulative, gravity, two_step_fca
from almaty_access.access.speed import (
    BIKE_MAX_FACTOR,
    BIKE_MIN_FACTOR,
    bike_kmh,
    car_kmh,
    seconds,
    tobler_walk_kmh,
)
from almaty_access.access.transit import ALIGHT_S, build_transit_graph
from almaty_access.prepare.transit import _RawRoute, _segments, parse_interval

METRIC = "EPSG:32643"


# ---------------------------------------------------------------------------
# Speeds
# ---------------------------------------------------------------------------


def test_tobler_flat_equals_configured_speed() -> None:
    assert tobler_walk_kmh(np.array([0.0]), 4.8)[0] == pytest.approx(4.8)


def test_tobler_shape() -> None:
    """Fastest on a gentle -5% descent, slower uphill than downhill."""
    grades = np.array([-0.05, 0.0, 0.10, -0.10])
    speed = tobler_walk_kmh(grades, 4.8)
    assert speed[0] > speed[1]  # -5% beats flat
    assert speed[2] < speed[1]  # +10% is slow
    assert speed[2] < speed[3]  # +10% is slower than -10%
    assert (speed > 0).all()


def test_bike_speed_linear_with_caps() -> None:
    flat = 15.0
    speed = bike_kmh(np.array([0.0, 0.05, -0.05, 0.30, -0.30]), flat)
    assert speed[0] == pytest.approx(flat)
    assert speed[1] == pytest.approx(flat - 5.0)  # -1 km/h per 1% climb
    assert speed[2] == pytest.approx(flat + 2.5)  # +0.5 km/h per 1% descent
    assert speed[3] == pytest.approx(BIKE_MIN_FACTOR * flat)  # floored
    assert speed[4] == pytest.approx(BIKE_MAX_FACTOR * flat)  # capped


def test_car_speed_by_class_with_link_and_default() -> None:
    speeds = {"primary": 60.0, "residential": 30.0, "default": 25.0}
    result = car_kmh(np.array(["primary", "primary_link", "footway", "residential"]), speeds)
    assert result.tolist() == [60.0, 60.0, 25.0, 30.0]


def test_seconds() -> None:
    assert seconds(np.array([1000.0]), 3.6)[0] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Graph construction and snapping
# ---------------------------------------------------------------------------


def _line_nodes(coords: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"node_id": np.arange(1, len(coords) + 1, dtype="int64")},
        geometry=gpd.points_from_xy([c[0] for c in coords], [c[1] for c in coords]),
        crs=METRIC,
    )


def _edges(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "u": [r[0] for r in rows],
            "v": [r[1] for r in rows],
            "key": 0,
            "length_m": [r[2] for r in rows],
        }
    )


def _line_world() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """1 - 2 - 3 - 4 bidirectional, 100 m hops."""
    nodes = _line_nodes([(0, 0), (100, 0), (200, 0), (300, 0)])
    rows = [(u, v, 100.0) for u, v in ((1, 2), (2, 3), (3, 4))]
    rows += [(v, u, d) for u, v, d in rows]
    return nodes, _edges(rows)


def _line_walk_graph() -> ModeGraph:
    """The line world walked at 6 km/h: 60 s per hop."""
    nodes, edges = _line_world()
    return build_graph(nodes, edges, seconds(edges["length_m"].to_numpy(), 6.0), METRIC)


@pytest.fixture()
def line_graph() -> ModeGraph:
    return _line_walk_graph()


def test_build_graph_times(line_graph: ModeGraph) -> None:
    times = dijkstra(line_graph.csr, directed=True, indices=0)
    assert times[3] == pytest.approx(180.0)


def test_parallel_edges_keep_fastest() -> None:
    nodes = _line_nodes([(0, 0), (100, 0)])
    edges = _edges([(1, 2, 100.0), (1, 2, 100.0)])
    graph = build_graph(nodes, edges, np.array([50.0, 30.0]), METRIC)
    assert graph.csr[0, 1] == pytest.approx(30.0)  # min, not 80 (the COO-sum trap)


def test_self_loops_dropped() -> None:
    nodes = _line_nodes([(0, 0), (100, 0)])
    edges = _edges([(1, 1, 5.0), (1, 2, 100.0), (2, 1, 100.0)])
    graph = build_graph(nodes, edges, np.array([1.0, 60.0, 60.0]), METRIC)
    assert graph.csr[0, 0] == 0.0


def test_snap_ignores_disconnected_island() -> None:
    """A node outside the main strongly connected component never captures
    origins, even when it is the geometrically nearest node."""
    nodes = _line_nodes([(0, 0), (100, 0), (200, 0), (310, 0)])
    rows = [(1, 2, 100.0), (2, 1, 100.0), (2, 3, 100.0), (3, 2, 100.0)]  # 4 is isolated
    edges = _edges(rows)
    graph = build_graph(nodes, edges, np.full(len(edges), 60.0), METRIC)
    assert not graph.snappable[3]
    position, dist = snap(graph, np.array([[305.0, 0.0]]))
    assert position[0] == 2  # node 3, not the isolated node 4
    assert dist[0] == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# Transit graph
# ---------------------------------------------------------------------------


def _toy_transit(
    headway_min: float, bus_circuity: float = 1.0, legs: pd.DataFrame | None = None
) -> ModeGraph:
    """Line walk graph + one bus S1(node1) -> S2(node4), 300 m leg."""
    walk = _line_walk_graph()
    stops = gpd.GeoDataFrame(
        {"stop_id": pd.array([101, 102], dtype="int64"), "name": ["a", "b"]},
        geometry=gpd.points_from_xy([0.0, 300.0], [0.0, 0.0]),
        crs=METRIC,
    )
    routes = pd.DataFrame(
        {
            "route_id": [7],
            "mode": ["bus"],
            "ref": ["X"],
            "name": ["x"],
            "headway_min": [headway_min],
            "headway_source": ["assumed"],
            "n_stops": [2],
        }
    )
    if legs is None:
        legs = pd.DataFrame(
            {
                "route_id": [7],
                "seg": [0],
                "seq": [0],
                "from_stop": [101],
                "to_stop": [102],
                "dist_m": [300.0],
            }
        )
    return build_transit_graph(
        walk,
        stops,
        routes,
        legs,
        walk_flat_kmh=6.0,
        speeds_kmh={"bus": 10.8, "trolleybus": 15.0, "subway": 32.0},
        circuity={"bus": bus_circuity, "trolleybus": 1.0, "subway": 1.0},
        metric_crs=METRIC,
    )


def test_transit_rides_when_frequent() -> None:
    """Headway 0.5 min: wait 15 s + ride 100 s beats the 180 s walk."""
    graph = _toy_transit(headway_min=0.5)
    times = dijkstra(graph.csr, directed=True, indices=0)
    # connector (>=1 s) + board 15 s + ride 100 s + alight + connector back
    expected = 1.0 + 15.0 + 100.0 + ALIGHT_S + 1.0
    assert times[3] == pytest.approx(expected)


def test_circuity_multiplies_ride_distance() -> None:
    """Bus circuity 1.5: the 300 m leg rides as 450 m -> 150 s, not 100 s."""
    graph = _toy_transit(headway_min=0.5, bus_circuity=1.5)
    times = dijkstra(graph.csr, directed=True, indices=0)
    expected = 1.0 + 15.0 + 150.0 + ALIGHT_S + 1.0
    assert times[3] == pytest.approx(expected)


def test_repeated_leg_keeps_fastest() -> None:
    """The same (route, from, to) hop listed twice (looping route) must
    keep the fastest ride — naive COO assembly would SUM them."""
    legs = pd.DataFrame(
        {
            "route_id": [7, 7],
            "seg": [0, 1],
            "seq": [0, 0],
            "from_stop": [101, 101],
            "to_stop": [102, 102],
            "dist_m": [300.0, 600.0],
        }
    )
    graph = _toy_transit(headway_min=0.5, legs=legs)
    times = dijkstra(graph.csr, directed=True, indices=0)
    expected = 1.0 + 15.0 + 100.0 + ALIGHT_S + 1.0  # the 300 m hop, not 300+600
    assert times[3] == pytest.approx(expected)


def test_transfer_pays_a_second_expected_wait() -> None:
    """Two routes meeting at a middle stop: the transfer must cost a
    fresh half-headway — the defining semantic of the headway model."""
    nodes = _line_nodes([(0, 0), (5000, 0), (10000, 0)])
    rows = [(u, v, 5000.0) for u, v in ((1, 2), (2, 3))]
    rows += [(v, u, d) for u, v, d in rows]
    edges = _edges(rows)
    walk = build_graph(nodes, edges, seconds(edges["length_m"].to_numpy(), 6.0), METRIC)
    stops = gpd.GeoDataFrame(
        {"stop_id": pd.array([101, 102, 103], dtype="int64"), "name": ["a", "b", "c"]},
        geometry=gpd.points_from_xy([0.0, 5000.0, 10000.0], [0.0, 0.0, 0.0]),
        crs=METRIC,
    )
    routes = pd.DataFrame(
        {
            "route_id": [1, 2],
            "mode": ["bus", "bus"],
            "ref": ["A", "B"],
            "name": ["a", "b"],
            "headway_min": [10.0, 10.0],
            "headway_source": ["assumed", "assumed"],
            "n_stops": [2, 2],
        }
    )
    legs = pd.DataFrame(
        {
            "route_id": [1, 2],
            "seg": [0, 0],
            "seq": [0, 0],
            "from_stop": [101, 102],
            "to_stop": [102, 103],
            "dist_m": [5000.0, 5000.0],
        }
    )
    graph = build_transit_graph(
        walk,
        stops,
        routes,
        legs,
        walk_flat_kmh=6.0,
        speeds_kmh={"bus": 30.0, "trolleybus": 15.0, "subway": 32.0},
        circuity={"bus": 1.0, "trolleybus": 1.0, "subway": 1.0},
        metric_crs=METRIC,
    )
    times = dijkstra(graph.csr, directed=True, indices=0)
    wait, ride = 300.0, 600.0  # headway/2 in seconds; 5 km at 30 km/h
    assert times[1] == pytest.approx(1.0 + wait + ride + ALIGHT_S + 1.0)  # one route
    # Via the transfer at stop 102: the SECOND boarding pays its own wait.
    expected = 1.0 + wait + ride + ALIGHT_S + wait + ride + ALIGHT_S + 1.0
    assert times[2] == pytest.approx(expected)
    assert times[2] < 2 * 5000.0 / (6.0 / 3.6)  # and it beats walking


def test_transit_walks_when_waiting_dominates() -> None:
    """Headway 10 min: the 300 s expected wait loses to walking 180 s."""
    graph = _toy_transit(headway_min=10.0)
    times = dijkstra(graph.csr, directed=True, indices=0)
    assert times[3] == pytest.approx(180.0)


def test_transit_legs_are_directed() -> None:
    """The bus runs 1 -> 4 only; the reverse trip has to walk."""
    graph = _toy_transit(headway_min=0.5)
    times = dijkstra(graph.csr, directed=True, indices=3)
    assert times[0] == pytest.approx(180.0)


def test_transit_graph_never_slower_than_walking() -> None:
    """The walk graph is a subgraph, so adding transit can only help."""
    graph = _toy_transit(headway_min=0.5)
    walk = _line_walk_graph()
    walk_times = dijkstra(walk.csr, directed=True, indices=0)
    transit_times = dijkstra(graph.csr, directed=True, indices=0)[: walk.n_nodes]
    assert (transit_times <= walk_times + 1e-9).all()


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------


def test_hex_times_exact_and_filtered(line_graph: ModeGraph) -> None:
    hexes = pd.DataFrame({"h3": ["a", "d"], "x": [0.0, 300.0], "y": [0.0, 0.0]})
    table, stats = hex_times(
        line_graph, hexes, hexes, walk_flat_kmh=6.0, cutoff_min=60.0, snap_max_m=2000.0
    )
    indexed = table.set_index(["origin_h3", "dest_h3"])["minutes"]
    assert indexed[("a", "d")] == pytest.approx(3.0)  # 180 s, zero snap
    assert indexed[("a", "a")] == pytest.approx(0.0)
    assert stats["pairs"] == 4
    short, _ = hex_times(
        line_graph, hexes, hexes, walk_flat_kmh=6.0, cutoff_min=2.0, snap_max_m=2000.0
    )
    pairs = set(zip(short["origin_h3"], short["dest_h3"], strict=True))
    assert ("a", "d") not in pairs and ("a", "a") in pairs


def test_hex_times_snap_added_as_walk_time(line_graph: ModeGraph) -> None:
    """A 100 m off-network origin adds 100 m / 6 km/h = 1 min to times."""
    hexes = pd.DataFrame({"h3": ["a", "d"], "x": [0.0, 300.0], "y": [-100.0, 0.0]})
    table, _ = hex_times(
        line_graph, hexes, hexes, walk_flat_kmh=6.0, cutoff_min=60.0, snap_max_m=2000.0
    )
    indexed = table.set_index(["origin_h3", "dest_h3"])["minutes"]
    assert indexed[("a", "d")] == pytest.approx(4.0)  # 1 snap + 3 graph
    # Intrazonal rule (D-048): the self-pair costs ONE snap leg — a
    # traveller inside the hex does not walk to the network and back.
    assert indexed[("a", "a")] == pytest.approx(1.0)


def test_hex_times_drops_offnetwork_hexes(line_graph: ModeGraph) -> None:
    hexes = pd.DataFrame({"h3": ["a", "far"], "x": [0.0, 9000.0], "y": [0.0, 0.0]})
    table, stats = hex_times(
        line_graph, hexes, hexes, walk_flat_kmh=6.0, cutoff_min=60.0, snap_max_m=2000.0
    )
    assert stats["origins_offnetwork"] == 1
    assert set(table["origin_h3"]) == {"a"}


def test_nearest_node_minutes_exact(line_graph: ModeGraph) -> None:
    """Facility at node 4 with a 60 m snap: node 1 is 180 s + 36 s away."""
    minutes = nearest_node_minutes(
        line_graph,
        np.array([3]),
        np.array([60.0]),
        walk_flat_kmh=6.0,
    )
    assert minutes[0] == pytest.approx((180.0 + 36.0) / 60.0)
    assert minutes[3] == pytest.approx(36.0 / 60.0)


def test_nearest_node_minutes_no_facilities(line_graph: ModeGraph) -> None:
    minutes = nearest_node_minutes(line_graph, np.array([]), np.array([]), walk_flat_kmh=6.0)
    assert np.isinf(minutes).all()


def test_nearest_two_facilities_on_one_node_keep_min_snap(line_graph: ModeGraph) -> None:
    """Two facilities snapping to the same node: the super-source arc must
    carry the SMALLER snap — naive COO assembly would sum 60 + 120 m."""
    minutes = nearest_node_minutes(
        line_graph, np.array([3, 3]), np.array([60.0, 120.0]), walk_flat_kmh=6.0
    )
    assert minutes[0] == pytest.approx((180.0 + 36.0) / 60.0)


def test_multi_batch_dijkstra_matches_single_batch(
    line_graph: ModeGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forcing BATCH_ORIGINS=2 over four origins (two batches) must give
    byte-identical pairs to the single-batch run."""
    from almaty_access.access import matrix as matrix_module

    hexes = pd.DataFrame(
        {"h3": ["a", "b", "c", "d"], "x": [0.0, 100.0, 200.0, 300.0], "y": [0.0] * 4}
    )
    single, _ = hex_times(
        line_graph, hexes, hexes, walk_flat_kmh=6.0, cutoff_min=60.0, snap_max_m=2000.0
    )
    monkeypatch.setattr(matrix_module, "BATCH_ORIGINS", 2)
    batched, _ = hex_times(
        line_graph, hexes, hexes, walk_flat_kmh=6.0, cutoff_min=60.0, snap_max_m=2000.0
    )
    order = ["origin_h3", "dest_h3"]
    pd.testing.assert_frame_equal(
        single.sort_values(order).reset_index(drop=True),
        batched.sort_values(order).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Measures (hand-computed)
# ---------------------------------------------------------------------------


@pytest.fixture()
def toy_tt() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin_h3": ["o1", "o1", "o2"],
            "dest_h3": ["d1", "d2", "d1"],
            "minutes": [10.0, 25.0, 40.0],
        }
    )


def test_cumulative_exact(toy_tt: pd.DataFrame) -> None:
    values = pd.Series({"d1": 100.0, "d2": 50.0})
    result = cumulative(toy_tt, values, (15, 30, 45))
    assert result.loc["o1", 15] == pytest.approx(100.0)
    assert result.loc["o1", 30] == pytest.approx(150.0)
    assert result.loc["o2", 45] == pytest.approx(100.0)
    # o2 reaches nothing within 15 min: absent from that threshold's
    # groupby, NaN after the frame union; callers reindex + fillna(0).
    assert np.isnan(result.loc["o2", 15])


def test_cumulative_unknown_destination_counts_zero(toy_tt: pd.DataFrame) -> None:
    result = cumulative(toy_tt, pd.Series({"d1": 100.0}), (45,))
    assert result.loc["o1", 45] == pytest.approx(100.0)  # d2 contributes 0


def test_gravity_exact(toy_tt: pd.DataFrame) -> None:
    values = pd.Series({"d1": 100.0, "d2": 50.0})
    result = gravity(toy_tt, values, half_life_min=10.0)
    expected_o1 = 100.0 * 2 ** (-1.0) + 50.0 * 2 ** (-2.5)
    assert result["o1"] == pytest.approx(expected_o1)
    assert result["o2"] == pytest.approx(100.0 * 2 ** (-4.0))


def test_two_step_fca_exact() -> None:
    tt = pd.DataFrame(
        {
            "origin_h3": ["i1", "i2", "i2", "i1"],
            "dest_h3": ["f1", "f1", "f2", "f2"],
            "minutes": [10.0, 12.0, 14.0, 20.0],  # last pair beyond catchment
        }
    )
    supply = pd.Series({"f1": 2.0, "f2": 1.0})
    population = pd.Series({"i1": 100.0, "i2": 300.0})
    result = two_step_fca(tt, supply, population, catchment_min=15.0)
    assert result["i1"] == pytest.approx(2.0 / 400.0 * SFCA_PER)
    assert result["i2"] == pytest.approx((2.0 / 400.0 + 1.0 / 300.0) * SFCA_PER)


def test_two_step_fca_zero_demand_keeps_zero_ratio() -> None:
    tt = pd.DataFrame({"origin_h3": ["i1"], "dest_h3": ["f1"], "minutes": [5.0]})
    result = two_step_fca(tt, pd.Series({"f1": 3.0}), pd.Series({"i1": 0.0}), catchment_min=15.0)
    assert result["i1"] == 0.0  # no division by zero, no inf


# ---------------------------------------------------------------------------
# Transit preparation helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10", 10.0),
        ("00:15", 15.0),
        ("01:00", 60.0),
        ("00:10:30", 10.5),
        ("", None),
        (None, None),
        ("abc", None),
        ("300", None),  # outside the plausible headway range
        ("00:00:01", None),
    ],
)
def test_parse_interval(value: str | None, expected: float | None) -> None:
    assert parse_interval(value) == expected


def _route(refs: list[int]) -> _RawRoute:
    return _RawRoute(relation_id=1, mode="bus", ref="x", name="x", interval=None, stop_refs=refs)


def test_segments_break_at_unusable_stop() -> None:
    xy = {1: (0.0, 0.0), 2: (500.0, 0.0), 3: (1000.0, 0.0), 4: (1500.0, 0.0)}
    legs, too_long = _segments(_route([1, 2, 9, 3, 4]), usable=set(xy), xy=xy)
    assert too_long == 0
    assert [(leg["from_stop"], leg["to_stop"], leg["seg"]) for leg in legs] == [
        (1, 2, 0),
        (3, 4, 1),
    ]
    assert legs[0]["dist_m"] == pytest.approx(500.0)


def test_segments_dedupe_same_halt() -> None:
    """A stop position and its platform 20 m apart are one halt."""
    xy = {1: (0.0, 0.0), 2: (20.0, 0.0), 3: (500.0, 0.0)}
    legs, _ = _segments(_route([1, 2, 3]), usable=set(xy), xy=xy)
    assert len(legs) == 1
    assert (legs[0]["from_stop"], legs[0]["to_stop"]) == (1, 3)


def test_segments_break_at_absurd_leg() -> None:
    xy = {1: (0.0, 0.0), 2: (500.0, 0.0), 3: (50_000.0, 0.0), 4: (50_500.0, 0.0)}
    legs, too_long = _segments(_route([1, 2, 3, 4]), usable=set(xy), xy=xy)
    assert too_long == 1
    assert [(leg["from_stop"], leg["to_stop"]) for leg in legs] == [(1, 2), (3, 4)]
