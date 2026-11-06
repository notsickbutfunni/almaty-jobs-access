"""Merge scenario transit lines into the baseline stop/route/leg tables.

The Phase 4 transit graph is built from three plain tables (stops,
routes, legs), so a scenario is applied by TABLE EDITING, not graph
surgery: a new line becomes one new route whose legs run both ways along
the drawn alignment; an extension appends (or prepends) legs to every
existing route it attaches to, so riders stay on the same vehicle and no
second boarding wait is charged. ``build_transit_graph`` then treats the
merged tables exactly like the baseline ones — the scenario pipeline
reuses Phase 4 code instead of reimplementing it.

Distances: scenario legs know their TRUE path length (measured along the
drawn alignment), but the graph build multiplies every leg by the per-
mode circuity factor that compensates OSM's straight-line legs. New
lines therefore ride under their own mode key with circuity 1.0, and
extension legs (which must keep the extended route's mode) are stored
PRE-DIVIDED by that mode's circuity, so the multiplication restores the
drawn length. The merged tables live only in memory — nothing written to
disk carries an adjusted distance.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from almaty_access.config import Config
from almaty_access.scenarios.definition import Scenario, Station, TransitLine
from almaty_access.schemas import SCENARIO_ID_BASE

logger = logging.getLogger(__name__)

#: A drawn station this close (metres) to the stop an extension attaches
#: at IS that stop redrawn, not a new station — it is merged away.
TERMINUS_MERGE_M = 50.0


@dataclass
class TransitApplication:
    """Baseline transit tables with one scenario merged in."""

    stops: gpd.GeoDataFrame
    routes: pd.DataFrame
    legs: pd.DataFrame
    speeds_kmh: dict[str, float]
    """Config's assumed speeds plus one entry per new scenario line."""
    circuity: dict[str, float]
    """Config's circuity plus 1.0 per new line (drawn length is exact)."""
    stats: dict
    """Per-line network statistics for the report."""


def apply_transit_scenario(
    scenario: Scenario,
    stops: gpd.GeoDataFrame,
    routes: pd.DataFrame,
    legs: pd.DataFrame,
    *,
    config: Config,
) -> TransitApplication:
    """Merge every transit line of ``scenario`` into the baseline tables."""
    speeds = dict(config.access.transit.speeds_kmh)
    circuity = dict(config.access.transit.circuity)

    metric = stops.geometry.to_crs(config.crs.metric)
    stop_xy = pd.DataFrame(
        {"x": metric.x.to_numpy(), "y": metric.y.to_numpy()},
        index=stops["stop_id"].to_numpy(),
    )

    # One id space for everything a scenario adds, allocated in a single
    # pass: stations first (per line, in order), then the line's own
    # route id when it is a new route.
    next_id = SCENARIO_ID_BASE
    station_ids: dict[tuple[str, str], int] = {}
    station_of_id: dict[int, Station] = {}
    route_id_of_line: dict[str, int] = {}
    for line in scenario.lines:
        for station in line.stations:
            station_ids[(line.name, station.name)] = next_id
            station_of_id[next_id] = station
            next_id += 1
        if line.extends_route_ref is None:
            route_id_of_line[line.name] = next_id
            next_id += 1

    new_route_rows: list[dict] = []
    new_leg_rows: list[dict] = []
    line_stats: list[dict] = []
    for line in scenario.lines:
        ids = {station.name: station_ids[(line.name, station.name)] for station in line.stations}
        if line.extends_route_ref is None:
            route_id = route_id_of_line[line.name]
            key = f"line:{line.name}"
            speeds[key] = float(line.speed_kmh)
            circuity[key] = 1.0
            new_route_rows.append(
                {
                    "route_id": route_id,
                    "mode": key,
                    "ref": "",
                    "name": line.name,
                    "headway_min": float(line.headway_min),
                    "headway_source": "scenario",
                    "n_stops": len(line.stations),
                }
            )
            new_leg_rows.extend(_two_way_legs(line, ids, route_id))
            line_stats.append(
                {
                    "line": line.name,
                    "kind": "new_route",
                    "mode": line.mode,
                    "route_id": route_id,
                    "speed_kmh": line.speed_kmh,
                    "headway_min": line.headway_min,
                    "length_km": round(line.length_m / 1000.0, 2),
                    "stations": [station.name for station in line.stations],
                }
            )
        else:
            attached, rows = _extend_routes(
                line,
                ids,
                routes,
                legs,
                stop_xy,
                attach_max_m=config.scenarios.attach_max_m,
                mode_circuity=circuity[line.mode],
            )
            new_leg_rows.extend(rows)
            line_stats.append(
                {
                    "line": line.name,
                    "kind": "extension",
                    "mode": line.mode,
                    "extends_route_ref": line.extends_route_ref,
                    "attachments": attached,
                    "length_km": round(line.length_m / 1000.0, 2),
                    "stations": [station.name for station in line.stations],
                }
            )

    # Only stations an actual leg references become stop rows: a station
    # merged into the terminus must not leave an orphan stop node behind.
    used = {row["from_stop"] for row in new_leg_rows} | {row["to_stop"] for row in new_leg_rows}
    station_rows = [
        {
            "stop_id": stop_id,
            "name": station.name,
            "geometry": Point(station.lon, station.lat),
        }
        for stop_id, station in station_of_id.items()
        if stop_id in used
    ]

    merged_stops = stops[["stop_id", "name", "geometry"]]
    if station_rows:
        addition = gpd.GeoDataFrame(station_rows, geometry="geometry", crs=stops.crs)
        merged_stops = pd.concat([merged_stops, addition], ignore_index=True)
        merged_stops = gpd.GeoDataFrame(merged_stops, geometry="geometry", crs=stops.crs)

    merged_routes = routes
    if new_route_rows:
        merged_routes = pd.concat(
            [routes, pd.DataFrame(new_route_rows, columns=list(routes.columns))],
            ignore_index=True,
        )
    merged_legs = legs
    if new_leg_rows:
        merged_legs = pd.concat(
            [legs, pd.DataFrame(new_leg_rows, columns=list(legs.columns))], ignore_index=True
        )

    logger.info(
        "scenario %s: +%d stops, +%d routes, +%d legs",
        scenario.name,
        len(station_rows),
        len(new_route_rows),
        len(new_leg_rows),
    )
    return TransitApplication(
        stops=merged_stops,
        routes=merged_routes,
        legs=merged_legs,
        speeds_kmh=speeds,
        circuity=circuity,
        stats={
            "lines": line_stats,
            "n_new_stops": len(station_rows),
            "n_new_routes": len(new_route_rows),
            "n_new_legs": len(new_leg_rows),
        },
    )


def _two_way_legs(line: TransitLine, ids: dict[str, int], route_id: int) -> list[dict]:
    """Directed legs of a new line: drawn order and back (one service).

    One route with legs in both directions models one bidirectional
    service: every (route, stop) pair gets one virtual node, so boarding
    pays the expected wait once and may then ride either way — the same
    cost separate per-direction relations would charge.
    """
    rows: list[dict] = []
    seq = 0
    for a, b in itertools.pairwise(line.stations):
        dist = round(float(b.s_m - a.s_m), 1)
        for from_station, to_station in ((a, b), (b, a)):
            rows.append(
                {
                    "route_id": route_id,
                    "seg": 0,
                    "seq": seq,
                    "from_stop": ids[from_station.name],
                    "to_stop": ids[to_station.name],
                    "dist_m": dist,
                }
            )
            seq += 1
    return rows


def _extend_routes(
    line: TransitLine,
    ids: dict[str, int],
    routes: pd.DataFrame,
    legs: pd.DataFrame,
    stop_xy: pd.DataFrame,
    *,
    attach_max_m: float,
    mode_circuity: float,
) -> tuple[list[dict], list[dict]]:
    """Attach an extension line to every matching existing route.

    A route matches on mode and ``ref``. It attaches when its first or
    last stop lies within ``attach_max_m`` of one END of the drawn line;
    the extension's legs then continue outward from that stop — appended
    after a route that ENDS there, prepended (reversed) before a route
    that STARTS there. A two-relation service (one relation per
    direction, like the metro) is thereby extended in both directions.
    """
    candidates = routes.loc[
        (routes["mode"] == line.mode) & (routes["ref"].astype(str) == line.extends_route_ref)
    ]
    if candidates.empty:
        available = sorted(
            routes.loc[routes["mode"] == line.mode, "ref"].astype(str).unique().tolist()
        )
        raise ValueError(
            f"scenario line {line.name!r}: no existing {line.mode} route has "
            f"ref={line.extends_route_ref!r}; refs present: {available or 'none'}"
        )

    ends = {"start": np.asarray(line.start_xy), "end": np.asarray(line.end_xy)}
    attached: list[dict] = []
    new_legs: list[dict] = []
    for route_id, route_legs in legs[legs["route_id"].isin(candidates["route_id"])].groupby(
        "route_id"
    ):
        ordered = route_legs.sort_values(["seg", "seq"])
        first_stop = int(ordered.iloc[0]["from_stop"])
        last_stop = int(ordered.iloc[-1]["to_stop"])
        options: list[tuple[float, str, int, bool]] = []
        for terminal, append in ((last_stop, True), (first_stop, False)):
            if terminal not in stop_xy.index:
                continue
            txy = stop_xy.loc[terminal, ["x", "y"]].to_numpy(dtype="float64")
            for end_name, end_xy in ends.items():
                options.append((float(np.hypot(*(txy - end_xy))), end_name, terminal, append))
        if not options:
            continue
        gap, end_name, terminal, append = min(options, key=lambda option: option[0])
        if gap > attach_max_m:
            continue

        terminal_xy = stop_xy.loc[terminal, ["x", "y"]].to_numpy(dtype="float64")
        outward = _outward_stations(line, end_name, terminal_xy)
        if not outward:
            raise ValueError(
                f"scenario line {line.name!r}: every drawn station merges into the "
                f"terminus of route {route_id} — nothing to extend with"
            )
        rows = _extension_legs(
            route_id=int(route_id),
            terminal=terminal,
            outward=outward,
            ids=ids,
            append=append,
            seg=int(ordered.iloc[-1]["seg"] if append else ordered.iloc[0]["seg"]),
            seq_after=int(ordered.iloc[-1]["seq"]) if append else None,
            mode_circuity=mode_circuity,
        )
        new_legs.extend(rows)
        attached.append(
            {
                "route_id": int(route_id),
                "attached_at": "last_stop" if append else "first_stop",
                "gap_m": round(gap, 1),
                "n_new_legs": len(rows),
            }
        )

    if not attached:
        raise ValueError(
            f"scenario line {line.name!r}: no terminal of a matching route lies within "
            f"{attach_max_m:.0f} m of the drawn line's ends — check the alignment "
            "or raise scenarios.attach_max_m"
        )
    return attached, new_legs


def _outward_stations(
    line: TransitLine, end_name: str, terminal_xy: np.ndarray
) -> list[tuple[Station, float]]:
    """Stations ordered outward from the attaching end, with hop lengths.

    Returns ``(station, dist_m)`` pairs. Rides start where the attached
    terminal PROJECTS onto the drawn line: the first hop runs from that
    projection to the first kept station along the line, plus the
    terminal's own (usually negligible) offset from the line. A station
    within ``TERMINUS_MERGE_M`` of the terminal is the terminus redrawn;
    a station behind the projection would ride backward — both are
    skipped.
    """
    terminal = Point(float(terminal_xy[0]), float(terminal_xy[1]))
    offset = float(line.geometry_m.distance(terminal))
    s_terminal = float(line.geometry_m.project(terminal))

    def outward_s(s_m: float) -> float:
        return s_m if end_name == "start" else line.length_m - s_m

    start = outward_s(s_terminal)
    kept: list[tuple[Station, float]] = []
    for station in sorted(line.stations, key=lambda st: outward_s(st.s_m)):
        position = outward_s(station.s_m)
        to_terminal = float(np.hypot(station.x - terminal.x, station.y - terminal.y))
        if to_terminal <= TERMINUS_MERGE_M or position <= start:
            continue
        kept.append((station, position))

    result: list[tuple[Station, float]] = []
    previous = start
    for index, (station, position) in enumerate(kept):
        result.append((station, position - previous + (offset if index == 0 else 0.0)))
        previous = position
    return result


def _extension_legs(
    *,
    route_id: int,
    terminal: int,
    outward: list[tuple[Station, float]],
    ids: dict[str, int],
    append: bool,
    seg: int,
    seq_after: int | None,
    mode_circuity: float,
) -> list[dict]:
    """Leg rows continuing an existing route outward from its terminal.

    Lengths are divided by the route mode's circuity so that the graph
    build's multiplication restores the measured along-line distance
    (see the module docstring).
    """
    hops: list[tuple[int, int, float]] = []
    previous = terminal
    for station, dist in outward:
        hops.append((previous, ids[station.name], dist / mode_circuity))
        previous = ids[station.name]
    if not append:
        hops = [(to_stop, from_stop, dist) for from_stop, to_stop, dist in reversed(hops)]
    rows = []
    for offset, (from_stop, to_stop, dist) in enumerate(hops):
        seq = (seq_after + 1 + offset) if seq_after is not None else (-len(hops) + offset)
        rows.append(
            {
                "route_id": route_id,
                "seg": seg,
                "seq": seq,
                "from_stop": from_stop,
                "to_stop": to_stop,
                "dist_m": round(dist, 1),
            }
        )
    return rows
