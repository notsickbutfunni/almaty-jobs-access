"""Headway-based transit graph: walk network + boarding + in-vehicle arcs.

The classic frequency-based model (no timetable, GTFS does not exist for
Almaty): every route-stop gets a *virtual* node; boarding it costs half
the route's headway (the expected wait of a random arrival), riding
between consecutive stops costs leg distance x circuity / assumed mode
speed, alighting is ~free. Because the walk network is part of the same
graph, Dijkstra discovers access, egress and transfer walks on its own,
and every additional boarding pays a new expected wait — which is
exactly what a transfer costs in a frequency-based world.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

from almaty_access.access.graph import ModeGraph, snap
from almaty_access.access.speed import seconds

#: Alighting arc, seconds. Kept strictly positive: a zero-weight edge in
#: a scipy sparse graph is indistinguishable from "no edge".
ALIGHT_S = 1.0

#: Lower bound on connector/boarding arcs for the same reason.
MIN_ARC_S = 1.0


def build_transit_graph(
    walk: ModeGraph,
    stops: gpd.GeoDataFrame,
    routes: pd.DataFrame,
    legs: pd.DataFrame,
    *,
    walk_flat_kmh: float,
    speeds_kmh: dict[str, float],
    circuity: dict[str, float],
    metric_crs: str,
) -> ModeGraph:
    """Union of the walking graph and the approximate transit network.

    Node layout: ``[walk nodes | stop nodes | virtual (route, stop) nodes]``.
    Only walk nodes stay snappable — travellers enter and leave the
    system on foot.
    """
    n_walk = walk.n_nodes
    stop_ids = stops["stop_id"].to_numpy()
    stop_pos = pd.Series(n_walk + np.arange(len(stop_ids)), index=stop_ids)
    points = stops.geometry.to_crs(metric_crs)
    stop_xy = np.column_stack([points.x.to_numpy(), points.y.to_numpy()])

    # Virtual nodes: one per (route, stop) pair that any leg touches.
    pairs = pd.concat(
        [
            legs[["route_id", "from_stop"]].rename(columns={"from_stop": "stop_id"}),
            legs[["route_id", "to_stop"]].rename(columns={"to_stop": "stop_id"}),
        ]
    ).drop_duplicates()
    pairs = pairs.reset_index(drop=True)
    first_virtual = n_walk + len(stop_ids)
    pairs["pos"] = first_virtual + np.arange(len(pairs))
    virtual_pos = pairs.set_index(["route_id", "stop_id"])["pos"]

    walk_coo = walk.csr.tocoo()
    rows: list[np.ndarray] = [walk_coo.row]
    cols: list[np.ndarray] = [walk_coo.col]
    data: list[np.ndarray] = [walk_coo.data]

    def add(u: np.ndarray, v: np.ndarray, t: np.ndarray) -> None:
        rows.append(np.asarray(u, dtype="int64"))
        cols.append(np.asarray(v, dtype="int64"))
        data.append(np.maximum(np.asarray(t, dtype="float64"), MIN_ARC_S))

    if len(stop_ids):
        # Stop <-> nearest walkable street node, walked in a straight line.
        walk_node, dist = snap(walk, stop_xy)
        connector_s = seconds(dist, walk_flat_kmh)
        stop_positions = stop_pos.to_numpy()
        add(stop_positions, walk_node, connector_s)
        add(walk_node, stop_positions, connector_s)

        # Boarding: expected wait = half the route's headway. Alighting
        # returns to the street-side stop node almost for free.
        by_route = routes.set_index("route_id")
        pair_stop_pos = stop_pos.reindex(pairs["stop_id"]).to_numpy()
        headway_min = by_route["headway_min"].reindex(pairs["route_id"]).to_numpy()
        add(pair_stop_pos, pairs["pos"].to_numpy(), headway_min / 2.0 * 60.0)
        add(pairs["pos"].to_numpy(), pair_stop_pos, np.full(len(pairs), ALIGHT_S))

        # In-vehicle legs, directed in travel order. A pair repeated by
        # a looping route keeps its fastest hop: duplicate COO entries
        # would otherwise be SUMMED into a bogus doubled time.
        hops = legs.groupby(["route_id", "from_stop", "to_stop"], as_index=False)["dist_m"].min()
        mode = by_route["mode"].reindex(hops["route_id"]).to_numpy()
        leg_kmh = np.array([speeds_kmh[m] for m in mode], dtype="float64")
        leg_circuity = np.array([circuity[m] for m in mode], dtype="float64")
        ride_s = seconds(hops["dist_m"].to_numpy() * leg_circuity, leg_kmh)
        from_pos = virtual_pos.reindex(
            pd.MultiIndex.from_frame(hops[["route_id", "from_stop"]])
        ).to_numpy()
        to_pos = virtual_pos.reindex(
            pd.MultiIndex.from_frame(hops[["route_id", "to_stop"]])
        ).to_numpy()
        add(from_pos, to_pos, ride_s)

    n_total = first_virtual + len(pairs)
    csr = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_total, n_total),
    ).tocsr()

    xy = np.vstack([walk.xy, stop_xy, np.full((len(pairs), 2), np.nan)])
    snappable = np.concatenate([walk.snappable, np.zeros(n_total - n_walk, dtype=bool)])
    node_ids = np.concatenate([walk.node_ids, stop_ids, -1 - np.arange(len(pairs), dtype="int64")])
    return ModeGraph(csr=csr, xy=xy, snappable=snappable, node_ids=node_ids)
