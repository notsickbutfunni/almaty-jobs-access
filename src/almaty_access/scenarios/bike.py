"""Bike corridors: higher flat speed on edges along the drawn streets.

The README's bike scenario is "a bike speed boost on selected streets":
dedicated infrastructure mostly removes conflicts and stops, which the
model expresses as a HIGHER FLAT SPEED on the matched edges. The grade
handling stays untouched (a protected lane does not flatten a hill), and
an edge is never slowed down: where the baseline already rides faster —
say a steep descent at the capped downhill bonus of a higher flat speed —
the baseline time is kept.

Edge matching: an edge belongs to a corridor when at least half of its
length lies inside the drawn line's buffer. Plain intersection would
also catch every side street CROSSING the corridor, which only touches
the buffer for a few metres.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import geopandas as gpd
import numpy as np

from almaty_access.access.speed import bike_kmh, seconds
from almaty_access.scenarios.definition import BikeBoost

logger = logging.getLogger(__name__)

#: Minimum share of an edge's length inside the corridor buffer: below
#: it the edge merely crosses or grazes the corridor.
MIN_OVERLAP = 0.5


def boosted_edge_seconds(
    edges: gpd.GeoDataFrame,
    boosts: Iterable[BikeBoost],
    *,
    flat_kmh: float,
    metric_crs: str,
) -> tuple[np.ndarray, dict]:
    """Per-edge traversal seconds with the corridors applied.

    Parameters mirror the baseline bike graph: ``edges`` is the prepared
    bike network (``length_m``, ``grade``, geometry), ``flat_kmh`` the
    configured flat bike speed. Returns the edge seconds and matching
    statistics. Where corridors overlap, the fastest one wins.
    """
    length = edges["length_m"].to_numpy(dtype="float64")
    grade = edges["grade"].to_numpy(dtype="float64")
    result = seconds(length, bike_kmh(grade, flat_kmh))

    geometry = edges.geometry.to_crs(metric_crs)
    boosted_any = np.zeros(len(edges), dtype=bool)
    per_boost: list[dict] = []
    for boost in boosts:
        zone = boost.geometry_m.buffer(boost.buffer_m)
        candidates = geometry.sindex.query(zone, predicate="intersects")
        matched = np.zeros(len(edges), dtype=bool)
        if len(candidates):
            inside = geometry.iloc[candidates].intersection(zone).length.to_numpy()
            total = geometry.iloc[candidates].length.to_numpy()
            share = np.divide(inside, total, out=np.zeros_like(inside), where=total > 0)
            matched[candidates[share >= MIN_OVERLAP]] = True
        if matched.any():
            faster = seconds(length[matched], bike_kmh(grade[matched], boost.speed_kmh))
            result[matched] = np.minimum(result[matched], faster)
        boosted_any |= matched
        per_boost.append(
            {
                "corridor": boost.name,
                "speed_kmh": boost.speed_kmh,
                "buffer_m": boost.buffer_m,
                "drawn_km": round(boost.length_m / 1000.0, 2),
                "edges_matched": int(matched.sum()),
                "edge_km_matched": round(float(length[matched].sum()) / 1000.0, 2),
            }
        )
        if not matched.any():
            logger.warning(
                "bike corridor %r matched no edges — is it drawn over streets "
                "the bike network contains?",
                boost.name,
            )

    stats = {
        "corridors": per_boost,
        "edges_boosted": int(boosted_any.sum()),
        "edge_km_boosted": round(float(length[boosted_any].sum()) / 1000.0, 2),
    }
    return result, stats
