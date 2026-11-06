"""Unit tests for the Phase 6 scenario engine on toy data (offline).

Geometry is constructed in the metric CRS (exact distances) and converted
to storage degrees for the GeoJSON files and tables, mirroring how real
scenario files are drawn.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, Point

from almaty_access import schemas
from almaty_access.access.graph import build_graph
from almaty_access.access.transit import build_transit_graph
from almaty_access.config import Config
from almaty_access.scenarios.bike import boosted_edge_seconds
from almaty_access.scenarios.build import _num, headline_measure, job_measures
from almaty_access.scenarios.definition import (
    BikeBoost,
    load_scenario,
    load_scenarios,
)
from almaty_access.scenarios.transit import apply_transit_scenario

TO_DEG = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True).transform
TO_M = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True).transform

#: Metric anchor somewhere inside the study area (near central Almaty).
ANCHOR = TO_M(76.90, 43.22)


def _deg(dx: float, dy: float) -> tuple[float, float]:
    """Lon/lat of a point ``(dx, dy)`` metres from the anchor."""
    lon, lat = TO_DEG(ANCHOR[0] + dx, ANCHOR[1] + dy)
    return float(lon), float(lat)


def _line_coords(*offsets: tuple[float, float]) -> list[list[float]]:
    return [list(_deg(dx, dy)) for dx, dy in offsets]


def _feature(kind: str, geometry: dict, **props) -> dict:
    return {"type": "Feature", "properties": {"kind": kind, **props}, "geometry": geometry}


def _line(offsets, **props) -> dict:
    return _feature("line", {"type": "LineString", "coordinates": _line_coords(*offsets)}, **props)


def _stop(offset, name: str, line: str) -> dict:
    return _feature(
        "stop", {"type": "Point", "coordinates": list(_deg(*offset))}, name=name, line=line
    )


def _boost(offsets, **props) -> dict:
    return _feature(
        "bike_boost", {"type": "LineString", "coordinates": _line_coords(*offsets)}, **props
    )


@pytest.fixture()
def scen_config(repo_config: Config, tmp_path: Path) -> Config:
    """Repo config whose scenario directory is an empty tmp folder."""
    directory = tmp_path / "scen"
    directory.mkdir()
    return dataclasses.replace(
        repo_config, scenarios=dataclasses.replace(repo_config.scenarios, dir=directory)
    )


def _write(config: Config, name: str, features: list[dict], properties: dict | None = None) -> Path:
    payload: dict = {"type": "FeatureCollection", "features": features}
    if properties is not None:
        payload["properties"] = properties
    path = config.scenarios.dir / f"{name}.geojson"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Definition parsing and validation
# ---------------------------------------------------------------------------


def test_unknown_kind_rejected(scen_config: Config) -> None:
    path = _write(
        scen_config, "bad", [_feature("station", {"type": "Point", "coordinates": [76.9, 43.22]})]
    )
    with pytest.raises(ValueError, match="kind"):
        load_scenario(path, scen_config)


def test_unknown_property_rejected(scen_config: Config) -> None:
    path = _write(
        scen_config, "typo", [_line([(0, 0), (2000, 0)], name="x", mode="bus", headway_mins=5)]
    )
    with pytest.raises(ValueError, match=r"unknown properties.*headway_mins"):
        load_scenario(path, scen_config)


def test_unknown_mode_rejected(scen_config: Config) -> None:
    path = _write(scen_config, "tram", [_line([(0, 0), (2000, 0)], name="x", mode="tram")])
    with pytest.raises(ValueError, match="mode"):
        load_scenario(path, scen_config)


def test_stop_naming_missing_line_rejected(scen_config: Config) -> None:
    path = _write(
        scen_config,
        "orphan",
        [_line([(0, 0), (2000, 0)], name="x", mode="bus"), _stop((500, 0), "s", "ghost")],
    )
    with pytest.raises(ValueError, match="unknown line 'ghost'"):
        load_scenario(path, scen_config)


def test_stop_far_from_line_rejected(scen_config: Config) -> None:
    path = _write(
        scen_config,
        "far",
        [_line([(0, 0), (2000, 0)], name="x", mode="bus"), _stop((500, 1000), "s", "x")],
    )
    with pytest.raises(ValueError, match="away from line"):
        load_scenario(path, scen_config)


def test_extension_forbids_speed_and_headway(scen_config: Config) -> None:
    path = _write(
        scen_config,
        "ext",
        [_line([(0, 0), (2000, 0)], name="x", mode="subway", extends_route_ref="1", speed_kmh=40)],
    )
    with pytest.raises(ValueError, match="not allowed on an extension"):
        load_scenario(path, scen_config)


def test_speed_bound_rejected(scen_config: Config) -> None:
    path = _write(
        scen_config, "fast", [_line([(0, 0), (2000, 0)], name="x", mode="bus", speed_kmh=500)]
    )
    with pytest.raises(ValueError, match="speed_kmh"):
        load_scenario(path, scen_config)


def test_boost_requires_speed(scen_config: Config) -> None:
    path = _write(scen_config, "slow", [_boost([(0, 0), (2000, 0)], name="b")])
    with pytest.raises(ValueError, match=r"speed_kmh.*required"):
        load_scenario(path, scen_config)


def test_single_stop_line_rejected(scen_config: Config) -> None:
    path = _write(
        scen_config,
        "short",
        [_line([(0, 0), (2000, 0)], name="x", mode="bus"), _stop((500, 0), "only", "x")],
    )
    with pytest.raises(ValueError, match="at least 2"):
        load_scenario(path, scen_config)


def test_bad_file_name_rejected(scen_config: Config) -> None:
    path = scen_config.scenarios.dir / "Bad Name.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="file name"):
        load_scenarios(scen_config)


def test_empty_directory_rejected(scen_config: Config) -> None:
    with pytest.raises(ValueError, match="no scenario files"):
        load_scenarios(scen_config)


def test_stops_ordered_along_line_not_by_feature_order(scen_config: Config) -> None:
    path = _write(
        scen_config,
        "order",
        [
            _line([(0, 0), (3000, 0)], name="x", mode="bus"),
            _stop((2000, 10), "far", "x"),
            _stop((100, -10), "near", "x"),
            _stop((1000, 0), "mid", "x"),
        ],
    )
    scenario = load_scenario(path, scen_config)
    stations = scenario.lines[0].stations
    assert [s.name for s in stations] == ["near", "mid", "far"]
    assert [round(s.s_m, -1) for s in stations] == [100, 1000, 2000]


def test_synthesized_stops_follow_spacing(scen_config: Config) -> None:
    path = _write(
        scen_config,
        "spacing",
        [_line([(0, 0), (3000, 0)], name="x", mode="bus", stop_spacing_m=1000)],
    )
    scenario = load_scenario(path, scen_config)
    stations = scenario.lines[0].stations
    assert len(stations) == 4
    assert all(s.synthesized for s in stations)
    np.testing.assert_allclose([s.s_m for s in stations], [0, 1000, 2000, 3000], atol=1.0)
    # Resolved defaults from config: assumed bus speed and headway.
    assert scenario.lines[0].speed_kmh == scen_config.access.transit.speeds_kmh["bus"]
    assert scenario.lines[0].headway_min == scen_config.access.transit.headways_min["bus"]


def test_committed_city_scenarios_parse(repo_config: Config) -> None:
    scenarios = {s.name: s for s in load_scenarios(repo_config)}
    assert set(scenarios) == {"bike_west", "brt_raiymbek", "metro_extension"}

    metro = scenarios["metro_extension"]
    assert metro.modes == ("transit",)
    (line,) = metro.lines
    assert line.extends_route_ref == "1" and line.mode == "subway"
    assert next(s.name for s in line.stations) == "Калқаман"
    assert len(line.stations) == 3 and not any(s.synthesized for s in line.stations)
    assert metro.sources  # provenance travels with the file

    brt = scenarios["brt_raiymbek"]
    (line,) = brt.lines
    assert line.speed_kmh == 22 and line.headway_min == 5
    assert all(s.synthesized for s in line.stations)
    assert len(line.stations) == round(line.length_m / 800) + 1

    bike = scenarios["bike_west"]
    assert bike.modes == ("bike",)
    assert len(bike.boosts) == 3
    assert all(b.speed_kmh == 18 for b in bike.boosts)


def test_committed_sample_scenario_parses(sample_config: Config) -> None:
    (demo,) = load_scenarios(sample_config)
    assert demo.name == "demo_abay"
    assert demo.modes == ("transit", "bike")


# ---------------------------------------------------------------------------
# Transit application (toy baseline tables)
# ---------------------------------------------------------------------------


def _toy_transit() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Two directed subway relations ref '1': A -> B and B -> A (3 km)."""
    stops = gpd.GeoDataFrame(
        {"stop_id": pd.array([1, 2], dtype="int64"), "name": ["A", "B"]},
        geometry=[Point(*_deg(0, 0)), Point(*_deg(3000, 0))],
        crs="EPSG:4326",
    )
    routes = pd.DataFrame(
        [
            {
                "route_id": 10,
                "mode": "subway",
                "ref": "1",
                "name": "A>B",
                "headway_min": 7.0,
                "headway_source": "assumed",
                "n_stops": 2,
            },
            {
                "route_id": 11,
                "mode": "subway",
                "ref": "1",
                "name": "B>A",
                "headway_min": 7.0,
                "headway_source": "assumed",
                "n_stops": 2,
            },
        ],
        columns=list(schemas.INTERIM_TRANSIT_ROUTES),
    )
    legs = pd.DataFrame(
        [
            {"route_id": 10, "seg": 0, "seq": 0, "from_stop": 1, "to_stop": 2, "dist_m": 3000.0},
            {"route_id": 11, "seg": 0, "seq": 0, "from_stop": 2, "to_stop": 1, "dist_m": 3000.0},
        ],
        columns=list(schemas.INTERIM_TRANSIT_LEGS),
    )
    return stops, routes, legs


def test_extension_attaches_to_both_directions(scen_config: Config) -> None:
    stops, routes, legs = _toy_transit()
    path = _write(
        scen_config,
        "ext",
        [
            _line([(3000, 0), (5000, 0)], name="x", mode="subway", extends_route_ref="1"),
            _stop((4000, 0), "C", "x"),
            _stop((5000, 0), "D", "x"),
        ],
    )
    scenario = load_scenario(path, scen_config)
    applied = apply_transit_scenario(scenario, stops, routes, legs, config=scen_config)

    assert len(applied.routes) == 2  # extension adds no route
    assert applied.stats["n_new_stops"] == 2
    new = applied.legs.iloc[2:]
    assert len(new) == 4  # B->C->D appended, D->C->B prepended
    forward = new[new["route_id"] == 10].sort_values("seq")
    assert forward["seq"].tolist() == [1, 2]
    backward = new[new["route_id"] == 11].sort_values("seq")
    assert backward["seq"].tolist() == [-2, -1]
    # Hops run outward on route 10 and inward on route 11.
    ids = applied.stops.set_index("name")["stop_id"]
    assert forward["from_stop"].tolist() == [2, ids["C"]]
    assert forward["to_stop"].tolist() == [ids["C"], ids["D"]]
    assert backward["from_stop"].tolist() == [ids["D"], ids["C"]]
    assert backward["to_stop"].tolist() == [ids["C"], 2]
    # 1 km hops stored pre-divided by the subway circuity.
    circuity = scen_config.access.transit.circuity["subway"]
    np.testing.assert_allclose(new["dist_m"], 1000.0 / circuity, atol=2.0)
    assert (applied.stops["stop_id"].iloc[2:] >= schemas.SCENARIO_ID_BASE).all()


def test_extension_merges_redrawn_terminus(scen_config: Config) -> None:
    stops, routes, legs = _toy_transit()
    path = _write(
        scen_config,
        "merge",
        [
            _line([(3000, 0), (5000, 0)], name="x", mode="subway", extends_route_ref="1"),
            _stop((3010, 0), "B again", "x"),  # within TERMINUS_MERGE_M of stop B
            _stop((5000, 0), "D", "x"),
        ],
    )
    scenario = load_scenario(path, scen_config)
    applied = apply_transit_scenario(scenario, stops, routes, legs, config=scen_config)
    assert applied.stats["n_new_stops"] == 1  # only D
    assert "B again" not in set(applied.stops["name"])
    forward = applied.legs.iloc[2:].query("route_id == 10")
    assert len(forward) == 1
    np.testing.assert_allclose(
        forward["dist_m"], 2000.0 / scen_config.access.transit.circuity["subway"], atol=2.0
    )


def test_extension_without_matching_ref_fails(scen_config: Config) -> None:
    stops, routes, legs = _toy_transit()
    path = _write(
        scen_config,
        "noref",
        [_line([(3000, 0), (5000, 0)], name="x", mode="subway", extends_route_ref="9")],
    )
    scenario = load_scenario(path, scen_config)
    with pytest.raises(ValueError, match="refs present"):
        apply_transit_scenario(scenario, stops, routes, legs, config=scen_config)


def test_extension_too_far_from_terminals_fails(scen_config: Config) -> None:
    stops, routes, legs = _toy_transit()
    path = _write(
        scen_config,
        "faraway",
        [_line([(4500, 0), (6000, 0)], name="x", mode="subway", extends_route_ref="1")],
    )
    scenario = load_scenario(path, scen_config)
    with pytest.raises(ValueError, match="attach_max_m"):
        apply_transit_scenario(scenario, stops, routes, legs, config=scen_config)


def test_new_line_rides_both_ways_under_its_own_speed(scen_config: Config) -> None:
    stops, routes, legs = _toy_transit()
    path = _write(
        scen_config,
        "newline",
        [
            _line(
                [(0, 2000), (4000, 2000)],
                name="express",
                mode="bus",
                speed_kmh=30,
                headway_min=4,
                stop_spacing_m=2000,
            ),
        ],
    )
    scenario = load_scenario(path, scen_config)
    applied = apply_transit_scenario(scenario, stops, routes, legs, config=scen_config)
    added = applied.routes.iloc[2]
    assert added["mode"] == "line:express"
    assert added["headway_min"] == 4.0
    assert added["headway_source"] == "scenario"
    assert applied.speeds_kmh["line:express"] == 30.0
    assert applied.circuity["line:express"] == 1.0
    new_legs = applied.legs.iloc[2:]
    assert len(new_legs) == 4  # 2 hops x 2 directions
    pairs = set(zip(new_legs["from_stop"], new_legs["to_stop"], strict=True))
    assert len(pairs) == 4  # every directed hop distinct
    np.testing.assert_allclose(new_legs["dist_m"], 2000.0, atol=2.0)


def test_extension_charges_no_second_wait(scen_config: Config) -> None:
    """An extension rides through; the same line as a separate route pays
    one more expected wait (half the headway) plus the alight/connector
    hops of a transfer — the physical difference the design encodes."""
    stops, routes, legs = _toy_transit()
    nodes = gpd.GeoDataFrame(
        {"node_id": pd.array([100, 101, 102], dtype="int64")},
        geometry=[Point(*_deg(0, 0)), Point(*_deg(3000, 0)), Point(*_deg(5000, 0))],
        crs="EPSG:4326",
    )
    edges = pd.DataFrame(
        {
            "u": [100, 101, 101, 102],
            "v": [101, 100, 102, 101],
            "length_m": [3000.0] * 2 + [2000.0] * 2,
        }
    )
    walk_seconds = edges["length_m"].to_numpy() / (4.8 / 3.6)
    walk = build_graph(nodes, edges, walk_seconds, scen_config.crs.metric)

    def total_seconds(features: list[dict], name: str) -> float:
        path = _write(scen_config, name, features)
        scenario = load_scenario(path, scen_config)
        applied = apply_transit_scenario(scenario, stops, routes, legs, config=scen_config)
        graph = build_transit_graph(
            walk,
            applied.stops,
            applied.routes,
            applied.legs,
            walk_flat_kmh=4.8,
            speeds_kmh=applied.speeds_kmh,
            circuity=applied.circuity,
            metric_crs=scen_config.crs.metric,
        )
        times = dijkstra(graph.csr, directed=True, indices=[0])
        return float(times[0, 2])  # walk node at the far end

    extension = [
        _line([(3000, 0), (5000, 0)], name="x", mode="subway", extends_route_ref="1"),
        _stop((5000, 0), "D", "x"),
    ]
    separate = [
        _line([(3000, 0), (5000, 0)], name="x", mode="subway"),  # synthesized end stops
    ]
    ride_through = total_seconds(extension, "ext")
    with_transfer = total_seconds(separate, "sep")
    extra_wait = scen_config.access.transit.headways_min["subway"] / 2.0 * 60.0
    assert with_transfer > ride_through
    assert with_transfer - ride_through == pytest.approx(extra_wait, abs=10.0)


# ---------------------------------------------------------------------------
# Bike boosts
# ---------------------------------------------------------------------------


def _metric_line(*offsets: tuple[float, float]) -> LineString:
    return LineString([(ANCHOR[0] + dx, ANCHOR[1] + dy) for dx, dy in offsets])


def _toy_bike_edges() -> gpd.GeoDataFrame:
    along = LineString([_deg(0, 0), _deg(1000, 0)])
    crossing = LineString([_deg(500, -300), _deg(500, 300)])
    far = LineString([_deg(0, 5000), _deg(1000, 5000)])
    return gpd.GeoDataFrame(
        {
            "u": [1, 3, 5],
            "v": [2, 4, 6],
            "length_m": [1000.0, 600.0, 1000.0],
            "grade": [0.0, 0.0, 0.0],
        },
        geometry=[along, crossing, far],
        crs="EPSG:4326",
    )


def test_boost_matches_corridor_edges_only(repo_config: Config) -> None:
    edges = _toy_bike_edges()
    boost = BikeBoost(
        name="corridor",
        speed_kmh=18.0,
        buffer_m=30.0,
        geometry_m=_metric_line((0, 0), (1000, 0)),
        length_m=1000.0,
    )
    seconds_, stats = boosted_edge_seconds(
        edges, [boost], flat_kmh=15.0, metric_crs=repo_config.crs.metric
    )
    assert stats["edges_boosted"] == 1
    assert seconds_[0] == pytest.approx(1000.0 / (18.0 / 3.6))
    assert seconds_[1] == pytest.approx(600.0 / (15.0 / 3.6))  # crossing street untouched
    assert seconds_[2] == pytest.approx(1000.0 / (15.0 / 3.6))


def test_boost_never_slows_an_edge(repo_config: Config) -> None:
    edges = _toy_bike_edges()
    slow = BikeBoost(
        name="slow",
        speed_kmh=10.0,
        buffer_m=30.0,
        geometry_m=_metric_line((0, 0), (1000, 0)),
        length_m=1000.0,
    )
    seconds_, stats = boosted_edge_seconds(
        edges, [slow], flat_kmh=15.0, metric_crs=repo_config.crs.metric
    )
    assert stats["edges_boosted"] == 1  # matched, but kept at the faster baseline
    assert seconds_[0] == pytest.approx(1000.0 / (15.0 / 3.6))


def test_overlapping_boosts_keep_the_fastest(repo_config: Config) -> None:
    edges = _toy_bike_edges()
    line = _metric_line((0, 0), (1000, 0))
    boosts = [
        BikeBoost(name="a", speed_kmh=17.0, buffer_m=30.0, geometry_m=line, length_m=1000.0),
        BikeBoost(name="b", speed_kmh=20.0, buffer_m=30.0, geometry_m=line, length_m=1000.0),
    ]
    seconds_, _ = boosted_edge_seconds(
        edges, boosts, flat_kmh=15.0, metric_crs=repo_config.crs.metric
    )
    assert seconds_[0] == pytest.approx(1000.0 / (20.0 / 3.6))


# ---------------------------------------------------------------------------
# Measures and headline plumbing
# ---------------------------------------------------------------------------


def test_job_measures_match_hand_computation() -> None:
    tt = pd.DataFrame(
        {
            "origin_h3": ["O", "O", "O", "D1", "D2"],
            "dest_h3": ["O", "D1", "D2", "D1", "D2"],
            "minutes": [0.5, 10.0, 20.0, 0.5, 0.5],
        }
    )
    index = pd.Index(["O", "D1", "D2"], name="h3")
    jobs = pd.Series([0.0, 100.0, 50.0], index=index)
    out = job_measures(tt, jobs, (15, 30), 20.0, index, total_jobs=150.0, mode="transit")
    np.testing.assert_allclose(out["acc_jobs_transit_15"], [100.0, 100.0, 50.0])
    np.testing.assert_allclose(out["acc_jobs_transit_30"], [150.0, 100.0, 50.0])
    np.testing.assert_allclose(out["accpct_jobs_transit_30"], [100.0, 66.67, 33.33])
    expected_grav = 100 * 2 ** (-10 / 20) + 50 * 2 ** (-20 / 20)
    assert out["acc_jobs_transit_grav"][0] == pytest.approx(expected_grav, abs=0.1)


def test_headline_measure_reuses_equity_config(repo_config: Config) -> None:
    assert headline_measure(repo_config, "transit") == ("accpct_jobs_transit_45", 45)
    assert headline_measure(repo_config, "bike") == ("accpct_jobs_bike_45", 45)
    stripped = dataclasses.replace(
        repo_config,
        equity=dataclasses.replace(repo_config.equity, headline_analysis=("accpct_jobs_walk_30",)),
    )
    with pytest.raises(ValueError, match="accpct_jobs_transit"):
        headline_measure(stripped, "transit")


def test_scenario_id_base_clears_osm_ids() -> None:
    """Synthetic ids must stay disjoint from real OSM node/relation ids."""
    assert schemas.SCENARIO_ID_BASE >= 10**12  # OSM node ids are ~1.2e10 in 2026


def test_report_numbers_stay_strict_json() -> None:
    """A Gini over an all-zero distribution is NaN; the report must carry
    None (valid strict JSON), never NaN."""
    assert _num(float("nan"), 4) is None
    assert _num(float("inf"), 4) is None
    assert _num(0.123456, 4) == 0.1235
