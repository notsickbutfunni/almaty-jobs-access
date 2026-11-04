"""Hex-to-hex travel times via batched scipy Dijkstra (Phase 4).

README §7 Phase 4.2: convert the graph to a sparse matrix and run
``scipy.sparse.csgraph.dijkstra`` in origin batches with ``limit`` set to
the largest threshold — orders of magnitude faster than NetworkX loops.
Hex centroids snap to the nearest network node; the snapping distance is
walked at the flat walking speed and added to every time through that
hex. Only pairs within the cutoff are kept (long format).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from almaty_access.access.graph import ModeGraph, snap

logger = logging.getLogger(__name__)

#: Origins per Dijkstra call: bounds the dense (batch x n_nodes) result
#: to a few hundred MB on the full city walk graph.
BATCH_ORIGINS = 512

#: Nearest-facility searches stop here; beyond it the answer is reported
#: as missing rather than a number nobody should trust.
NEAREST_LIMIT_MIN = 180.0


def _snap_hexes(
    graph: ModeGraph,
    hexes: pd.DataFrame,
    walk_flat_kmh: float,
    snap_max_m: float,
) -> tuple[pd.DataFrame, int, np.ndarray]:
    """Snap hex centroids; drop off-network hexes beyond ``snap_max_m``."""
    nodes, dist = snap(graph, hexes[["x", "y"]].to_numpy())
    kept = dist <= snap_max_m
    frame = pd.DataFrame(
        {
            "h3": hexes["h3"].to_numpy()[kept],
            "node": nodes[kept],
            "snap_min": dist[kept] / (walk_flat_kmh * 1000.0 / 60.0),
        }
    )
    return frame, int((~kept).sum()), dist


def hex_times(
    graph: ModeGraph,
    origins: pd.DataFrame,
    dests: pd.DataFrame,
    *,
    walk_flat_kmh: float,
    cutoff_min: float,
    snap_max_m: float,
) -> tuple[pd.DataFrame, dict]:
    """Long travel-time table between hex centroids, pairs <= cutoff.

    ``origins`` and ``dests`` carry columns ``h3``, ``x``, ``y`` (metric).
    Returns ``(table, stats)`` where the table has columns
    ``origin_h3, dest_h3, minutes`` and stats documents snapping.
    Self-pairs (origin == dest) cost one snap leg — the intrazonal rule,
    see D-048.
    """
    o_frame, o_dropped, o_dist = _snap_hexes(graph, origins, walk_flat_kmh, snap_max_m)
    d_frame, d_dropped, _ = _snap_hexes(graph, dests, walk_flat_kmh, snap_max_m)

    dest_nodes = np.unique(d_frame["node"].to_numpy())
    dest_col = pd.Series(np.arange(len(dest_nodes)), index=dest_nodes)
    d_cols = dest_col.reindex(d_frame["node"]).to_numpy(dtype="int64")
    d_h3 = d_frame["h3"].to_numpy()
    d_snap = d_frame["snap_min"].to_numpy()

    origin_nodes = np.unique(o_frame["node"].to_numpy())
    row_of_node = pd.Series(np.arange(len(origin_nodes)), index=origin_nodes)
    self_col: dict[str, int] = {h3_id: col for col, h3_id in enumerate(d_h3)}

    o_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []
    m_parts: list[np.ndarray] = []
    for start in range(0, len(origin_nodes), BATCH_ORIGINS):
        batch = origin_nodes[start : start + BATCH_ORIGINS]
        seconds = dijkstra(graph.csr, directed=True, indices=batch, limit=cutoff_min * 60.0)
        minutes = seconds[:, dest_nodes] / 60.0

        # origin_nodes is sorted, so the hexes whose node falls in the
        # batch's value range are exactly the batch's hexes.
        in_batch = o_frame.loc[(o_frame["node"] >= batch[0]) & (o_frame["node"] <= batch[-1])]
        for h3_id, node, snap_min in in_batch.itertuples(index=False):
            row = minutes[int(row_of_node[node]) - start]
            total = row[d_cols] + snap_min + d_snap
            # Intrazonal rule: a traveller starting inside the hex does
            # not walk out to the network and back, so the self-pair
            # costs ONE snap leg, not two (D-048) — otherwise a hex far
            # from the network could not "reach" its own opportunities.
            own = self_col.get(h3_id)
            if own is not None:
                total[own] = snap_min
            keep = total <= cutoff_min
            if not keep.any():
                continue
            o_parts.append(np.repeat(h3_id, int(keep.sum())))
            d_parts.append(d_h3[keep])
            m_parts.append(np.round(total[keep], 2))

    if o_parts:
        table = pd.DataFrame(
            {
                "origin_h3": np.concatenate(o_parts),
                "dest_h3": np.concatenate(d_parts),
                "minutes": np.concatenate(m_parts),
            }
        )
    else:
        table = pd.DataFrame({"origin_h3": [], "dest_h3": [], "minutes": []}).astype(
            {"origin_h3": "string", "dest_h3": "string", "minutes": "float64"}
        )
    stats = {
        "origins_offnetwork": o_dropped,
        "dests_offnetwork": d_dropped,
        "snap_m_median": round(float(np.median(o_dist)), 1) if len(o_dist) else None,
        "snap_m_p95": round(float(np.quantile(o_dist, 0.95)), 1) if len(o_dist) else None,
        "pairs": len(table),
    }
    return table, stats


def nearest_node_minutes(
    graph: ModeGraph,
    facility_nodes: np.ndarray,
    facility_snap_m: np.ndarray,
    *,
    walk_flat_kmh: float,
) -> np.ndarray:
    """Minutes from EVERY graph node to its nearest facility.

    One Dijkstra from a virtual super-source on the TRANSPOSED graph:
    distances on the reversed graph from a source connected to each
    facility node (arc weight = that facility's own snapping walk) equal
    ``min_j (snap_j + t(node -> facility_j))`` on the forward graph —
    exact, and independent of the number of facilities. Unreached nodes
    come back as ``inf``.
    """
    n = graph.n_nodes
    if not len(facility_nodes):
        return np.full(n, np.inf)
    snap_s = facility_snap_m / (walk_flat_kmh * 1000.0 / 3600.0)
    weights = pd.Series(np.maximum(snap_s, 1.0)).groupby(facility_nodes).min()

    reversed_graph = graph.csr.T.tocoo()
    rows = np.concatenate([reversed_graph.row, np.full(len(weights), n, dtype="int64")])
    cols = np.concatenate([reversed_graph.col, weights.index.to_numpy(dtype="int64")])
    data = np.concatenate([reversed_graph.data, weights.to_numpy(dtype="float64")])
    augmented = sparse.coo_matrix((data, (rows, cols)), shape=(n + 1, n + 1)).tocsr()

    seconds = dijkstra(
        augmented, directed=True, indices=n, limit=NEAREST_LIMIT_MIN * 60.0, min_only=False
    )
    return seconds[:n] / 60.0
