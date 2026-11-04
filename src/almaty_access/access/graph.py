"""Directed time-weighted graphs for scipy shortest paths (Phase 4).

A :class:`ModeGraph` wraps a mode's network as a CSR matrix of edge
traversal times in seconds, plus the metric node coordinates and a
``snappable`` mask. Snapping is restricted to the largest strongly
connected component: a centroid snapped onto a disconnected island
(a park loop, a gated service yard) would appear unable to reach the
whole city, which is a data artifact, not a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ModeGraph:
    """A routable network: times in seconds on a directed sparse graph."""

    csr: sparse.csr_matrix
    """Directed edge times, seconds; shape (n_nodes, n_nodes)."""
    xy: np.ndarray
    """Metric-CRS node coordinates, shape (n_nodes, 2); NaN for virtual
    transit nodes, which are never snapped to."""
    snappable: np.ndarray
    """True for street nodes in the largest strongly connected component."""
    node_ids: np.ndarray
    """Original node ids (OSM for street nodes, synthetic negatives for
    virtual transit nodes); for inspection only."""

    @property
    def n_nodes(self) -> int:
        return self.csr.shape[0]


def metric_xy(nodes: gpd.GeoDataFrame, metric_crs: str) -> np.ndarray:
    """Node coordinates in the metric CRS, shape (n, 2)."""
    points = nodes.geometry.to_crs(metric_crs)
    return np.column_stack([points.x.to_numpy(), points.y.to_numpy()])


def build_graph(
    nodes: gpd.GeoDataFrame,
    edges: pd.DataFrame,
    edge_seconds: np.ndarray,
    metric_crs: str,
) -> ModeGraph:
    """Assemble a :class:`ModeGraph` from prepared node/edge tables.

    Parallel edges (same u, v with different keys) keep only the FASTEST
    time: building a COO matrix naively would SUM duplicate entries,
    silently inflating travel times. Self-loops are dropped.
    """
    node_ids = nodes["node_id"].to_numpy()
    position = pd.Series(np.arange(len(node_ids)), index=node_ids)

    table = pd.DataFrame(
        {
            "u": position.reindex(edges["u"]).to_numpy(),
            "v": position.reindex(edges["v"]).to_numpy(),
            "seconds": np.asarray(edge_seconds, dtype="float64"),
        }
    )
    known = table.dropna(subset=["u", "v"])
    known = known.loc[known["u"] != known["v"]]
    fastest = known.groupby(["u", "v"], as_index=False)["seconds"].min()

    n = len(node_ids)
    csr = sparse.coo_matrix(
        (
            fastest["seconds"].to_numpy(),
            (fastest["u"].to_numpy(dtype="int64"), fastest["v"].to_numpy(dtype="int64")),
        ),
        shape=(n, n),
    ).tocsr()

    _, labels = connected_components(csr, directed=True, connection="strong")
    counts = np.bincount(labels)
    snappable = labels == int(np.argmax(counts))
    return ModeGraph(
        csr=csr,
        xy=metric_xy(nodes, metric_crs),
        snappable=snappable,
        node_ids=node_ids,
    )


def snap(graph: ModeGraph, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest snappable node for each metric point.

    Returns ``(positions, dist_m)``: indices into the graph's node set
    and the straight-line snapping distance, which callers add to travel
    times as walking (README §7 Phase 4.2).
    """
    candidates = np.flatnonzero(graph.snappable)
    if not len(candidates):
        raise ValueError("graph has no snappable nodes")
    tree = cKDTree(graph.xy[candidates])
    dist, index = tree.query(np.asarray(points_xy, dtype="float64"))
    return candidates[index], dist
