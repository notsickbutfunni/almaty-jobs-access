"""Street-network features: density, hierarchy, centrality, access.

Intersection density and road length by class describe local street
structure; sampled betweenness centrality (how often a node lies on
shortest paths between random node pairs) finds the through-movement
corridors where retail and services concentrate. Betweenness runs on
the full buffered drive network so values near the city edge are not
biased by a cut-off graph.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import h3
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

#: highway=<first value> -> hierarchy group for length-by-class features.
#: No 'service' class: osmnx network_type='drive' (Phase 1) excludes
#: highway=service/track entirely, so such a column would be
#: structurally zero — a dead feature the Phase 2 review caught.
HIGHWAY_GROUPS: dict[str, str] = {
    "motorway": "major",
    "motorway_link": "major",
    "trunk": "major",
    "trunk_link": "major",
    "primary": "major",
    "primary_link": "major",
    "secondary": "secondary",
    "secondary_link": "secondary",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary",
    "residential": "local",
    "living_street": "local",
    "unclassified": "local",
}

GROUP_ORDER: tuple[str, ...] = ("major", "secondary", "tertiary", "local", "other")

#: A node is an intersection when it joins at least 3 distinct neighbours.
MIN_INTERSECTION_DEGREE = 3


def highway_group(values: pd.Series) -> pd.Series:
    """Map the (possibly ``;``-joined) highway tag to a hierarchy group."""
    first = values.astype("string").str.split(";").str[0]
    return first.map(HIGHWAY_GROUPS).fillna("other")


def node_degrees(edges: pd.DataFrame) -> pd.Series:
    """Distinct-neighbour count per node on the undirected street graph."""
    forward = edges[["u", "v"]]
    backward = forward.rename(columns={"u": "v", "v": "u"})
    links = pd.concat([forward, backward], ignore_index=True).drop_duplicates()
    links = links.loc[links["u"] != links["v"]]  # a self-loop is not a neighbour
    return links.groupby("u")["v"].nunique()


def undirected_edges(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One row per physical street segment.

    The interim layers store osmnx DIRECTED edges: a two-way street is
    u->v AND v->u with the same geometry, so summing raw rows counts it
    twice (measured: drive lengths inflated 1.21x on major roads up to
    1.98x on local ones — enough to distort the class mix). Keeping
    ``length_m`` in the key spares genuinely distinct parallel edges
    (dual carriageways) between the same node pair.
    """
    key = pd.DataFrame(
        {
            "a": edges[["u", "v"]].min(axis=1),
            "b": edges[["u", "v"]].max(axis=1),
            "k": edges["key"],
            "len": edges["length_m"].round(1),
        }
    )
    return edges.loc[~key.duplicated().to_numpy()]


def node_betweenness(edges: pd.DataFrame, k: int, seed: int) -> pd.Series:
    """Sampled betweenness centrality per node (normalized, weighted).

    ``k`` source nodes are sampled (capped at the node count); the
    result is an approximation whose sampling error shrinks with k —
    the exact computation is O(V·E) and needless here (README: sampled).
    """
    graph = nx.Graph()
    for u, v, length in edges[["u", "v", "length_m"]].itertuples(index=False):
        if graph.has_edge(u, v):
            graph[u][v]["length_m"] = min(graph[u][v]["length_m"], length)
        else:
            graph.add_edge(u, v, length_m=length)
    k_effective = min(k, graph.number_of_nodes())
    logger.info(
        "betweenness: %d nodes, %d edges, k=%d", len(graph), graph.number_of_edges(), k_effective
    )
    values = nx.betweenness_centrality(graph, k=k_effective, weight="length_m", seed=seed)
    return pd.Series(values, name="betweenness")


def _length_by_cell(
    edges: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame, by: pd.Series | None = None
) -> pd.DataFrame | pd.Series:
    """Metres of line per hex (optionally split by a per-edge group)."""
    pairs = gpd.sjoin(edges[["geometry"]], hexes[["geometry"]], predicate="intersects")
    if pairs.empty:
        empty = pd.Series(dtype=float)
        return empty if by is None else pd.DataFrame()
    lengths = pairs.geometry.intersection(
        hexes.geometry.loc[pairs["index_right"]], align=False
    ).length
    if by is None:
        return lengths.groupby(pairs["index_right"]).sum()
    groups = by.loc[pairs.index]
    return lengths.groupby([pairs["index_right"], groups]).sum().unstack(fill_value=0.0)


def network_features(
    drive_nodes: gpd.GeoDataFrame,
    drive_edges: gpd.GeoDataFrame,
    walk_edges: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    res: int,
    metric_crs: str,
    betweenness: pd.Series,
    fallback_distance_m: float,
) -> pd.DataFrame:
    """Per-hex network block (``net_`` columns), aligned to the grid."""
    index = pd.Index(grid["h3"], name="h3")
    out = pd.DataFrame(index=index)
    hexes = grid.set_index("h3")[["geometry"]].to_crs(metric_crs).rename_axis(None)
    hex_area_km2 = h3.average_hexagon_area(res, unit="km^2")

    # Intersections and betweenness live on nodes -> point-in-cell.
    degrees = node_degrees(drive_edges)
    cells = pd.Series(
        [
            h3.latlng_to_cell(lat, lon, res)
            for lat, lon in zip(drive_nodes["y"], drive_nodes["x"], strict=True)
        ],
        index=drive_nodes["node_id"].to_numpy(),
        name="h3",
    )
    node_frame = pd.DataFrame(
        {
            "h3": cells,
            "degree": degrees.reindex(cells.index).fillna(0),
            "betweenness": betweenness.reindex(cells.index).fillna(0.0),
        }
    )
    intersections = node_frame.loc[node_frame["degree"] >= MIN_INTERSECTION_DEGREE]
    out["net_intersection_cnt"] = (
        intersections.groupby("h3").size().reindex(index).fillna(0).astype(int)
    )
    out["net_intersection_density_km2"] = out["net_intersection_cnt"] / hex_area_km2
    grouped = node_frame.groupby("h3")["betweenness"]
    out["net_betweenness_mean"] = grouped.mean().reindex(index).fillna(0.0)
    out["net_betweenness_max"] = grouped.max().reindex(index).fillna(0.0)

    # Road length by hierarchy class, exact geometric split per hex.
    # Lengths need ONE row per physical segment (undirected); degrees and
    # betweenness above already collapse direction themselves.
    drive_undirected = undirected_edges(drive_edges)
    drive_metric = drive_undirected.to_crs(metric_crs)
    by_class = _length_by_cell(drive_metric, hexes, by=highway_group(drive_undirected["highway"]))
    for group in GROUP_ORDER:
        column = by_class[group] if group in getattr(by_class, "columns", ()) else None
        values = column.reindex(index).fillna(0.0) if column is not None else 0.0
        out[f"net_len_{group}_m"] = values
    out["net_len_drive_m"] = out[[f"net_len_{g}_m" for g in GROUP_ORDER]].sum(axis=1)
    walk_lengths = _length_by_cell(undirected_edges(walk_edges).to_crs(metric_crs), hexes)
    out["net_len_walk_m"] = walk_lengths.reindex(index).fillna(0.0)

    # Distance from the hex centroid to the nearest major road.
    major = drive_metric.loc[highway_group(drive_undirected["highway"]).eq("major").to_numpy()]
    centroids = gpd.GeoDataFrame(geometry=hexes.geometry.centroid, index=hexes.index)
    if major.empty:
        logger.warning("no major roads in the study area; using fallback distance")
        out["net_dist_major_m"] = fallback_distance_m
    else:
        nearest = gpd.sjoin_nearest(centroids, major[["geometry"]], distance_col="distance_m")
        distance = nearest["distance_m"].groupby(level=0).min()
        out["net_dist_major_m"] = distance.reindex(index).fillna(fallback_distance_m)
    return out
