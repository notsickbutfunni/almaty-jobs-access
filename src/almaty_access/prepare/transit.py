"""Approximate transit network from OSM route relations (Phase 4).

No public GTFS exists for Almaty (DECISIONS D-016), so the README's
fallback applies: extract ``route=bus|trolleybus|subway`` relations from
the already-downloaded PBF and turn their ordered stop members into a
stop/leg table. Travel times are NOT computed here — the access stage
combines these legs with the assumed speeds and headways from config,
and every output is labelled *approximate transit access*.

What counts as a stop: PTv2 relations order their members in travel
direction, tagging each halt as a ``stop`` (position on the road) and/or
``platform`` (where passengers wait). Whichever list a relation fills in
more completely is used — never both, which would traverse every halt
twice. Relations with neither — route ways only — cannot yield
stop-to-stop legs and are excluded and counted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from almaty_access import schemas
from almaty_access.config import Config
from almaty_access.download.osm import PBF_FILE
from almaty_access.io import ensure_built, read_geoparquet, write_geoparquet
from almaty_access.prepare.boundary import BUFFER_FILE
from almaty_access.schemas import TRANSIT_MODES

logger = logging.getLogger(__name__)

STOPS_FILE = "transit_stops.parquet"
ROUTES_FILE = "transit_routes.parquet"
LEGS_FILE = "transit_legs.parquet"
EXTRACT_REPORT_FILE = "transit_extract.json"

#: Consecutive stops closer than this are one physical halt mapped twice
#: (stop position + platform); the duplicate is dropped.
STOP_DEDUP_M = 40.0

#: A leg longer than this inside the study area is a data error (missing
#: stops in the relation); the sequence is broken there instead.
MAX_LEG_M = 20_000.0

#: Parsed ``interval`` tags outside this range (minutes) are ignored.
HEADWAY_RANGE_MIN = (1.0, 120.0)


@dataclass
class _RawRoute:
    relation_id: int
    mode: str
    ref: str
    name: str
    interval: str | None
    stop_refs: list[int]


def _is_stop_role(role: str) -> bool:
    return role.startswith(("stop", "platform"))


def parse_interval(value: str | None) -> float | None:
    """OSM ``interval`` tag -> headway in minutes, or None.

    Accepts plain minutes ("15"), "HH:MM" and "HH:MM:SS". Values outside
    ``HEADWAY_RANGE_MIN`` (e.g. interval=1 second, interval=0) are noise.
    """
    if not value:
        return None
    text = value.strip()
    try:
        if ":" in text:
            parts = [float(p) for p in text.split(":")]
            if len(parts) == 2:
                minutes = parts[0] * 60 + parts[1]
            elif len(parts) == 3:
                minutes = parts[0] * 60 + parts[1] + parts[2] / 60
            else:
                return None
        else:
            minutes = float(text)
    except ValueError:
        return None
    low, high = HEADWAY_RANGE_MIN
    return minutes if low <= minutes <= high else None


def extract_routes(pbf_path: Path) -> tuple[list[_RawRoute], gpd.GeoDataFrame, int]:
    """Read route relations and their stop nodes from the PBF.

    Returns ``(routes, stop_points, n_without_stops)``: every bus /
    trolleybus / subway relation that has stop-usable node members (in
    member order), the located stop nodes (EPSG:4326), and how many
    relations had no usable stop members at all.
    """
    import osmium

    routes: list[_RawRoute] = []
    without_stops = 0

    class _Relations(osmium.SimpleHandler):
        def relation(self, r) -> None:
            nonlocal without_stops
            mode = r.tags.get("route")
            if mode not in TRANSIT_MODES:
                return
            stop_nodes = [m.ref for m in r.members if m.type == "n" and m.role.startswith("stop")]
            platform_nodes = [
                m.ref for m in r.members if m.type == "n" and m.role.startswith("platform")
            ]
            # Use whichever member list is RICHER. Many Almaty routes map
            # every platform but only a handful of stop positions (one
            # route: 56 platforms vs 13 stops), so preferring stops would
            # throw most of the sequence away. Never mix the two lists:
            # a relation listing both would traverse every halt twice.
            refs = max(stop_nodes, platform_nodes, key=len)
            if len(refs) < 2:
                without_stops += 1
                return
            routes.append(
                _RawRoute(
                    relation_id=r.id,
                    mode=mode,
                    ref=r.tags.get("ref", ""),
                    name=r.tags.get("name", ""),
                    interval=r.tags.get("interval"),
                    stop_refs=refs,
                )
            )

    _Relations().apply_file(str(pbf_path))

    needed = {ref for route in routes for ref in route.stop_refs}
    locations: dict[int, tuple[float, float]] = {}
    names: dict[int, str] = {}

    class _Nodes(osmium.SimpleHandler):
        def node(self, n) -> None:
            if n.id in needed:
                locations[n.id] = (n.location.lon, n.location.lat)
                name = n.tags.get("name")
                if name:
                    names[n.id] = name

    _Nodes().apply_file(str(pbf_path))

    stop_ids = sorted(locations)
    points = gpd.GeoDataFrame(
        {
            "stop_id": pd.array(stop_ids, dtype="int64"),
            "name": [names.get(i, "") for i in stop_ids],
        },
        geometry=[shapely.Point(*locations[i]) for i in stop_ids],
        crs="EPSG:4326",
    )
    return routes, points, without_stops


def _segments(
    route: _RawRoute, usable: set[int], xy: dict[int, tuple[float, float]]
) -> tuple[list[dict], int]:
    """Legs of one route: consecutive usable stops, broken at gaps.

    A stop outside the study area (or without a location) breaks the
    sequence — a leg must never bridge the gap it leaves. Consecutive
    stops closer than ``STOP_DEDUP_M`` are one halt mapped twice.
    """
    legs: list[dict] = []
    too_long = 0
    seg = 0
    seq = 0
    previous: int | None = None
    for ref in route.stop_refs:
        if ref not in usable:
            if previous is not None:
                seg += 1
            previous = None
            continue
        if previous is None:
            previous = ref
            continue
        if ref == previous:
            continue
        ax, ay = xy[previous]
        bx, by = xy[ref]
        dist = float(np.hypot(bx - ax, by - ay))
        if dist < STOP_DEDUP_M:
            continue  # same halt, keep the first node
        if dist > MAX_LEG_M:
            too_long += 1
            seg += 1
            previous = ref
            continue
        legs.append(
            {
                "route_id": route.relation_id,
                "seg": seg,
                "seq": seq,
                "from_stop": previous,
                "to_stop": ref,
                "dist_m": round(dist, 1),
            }
        )
        seq += 1
        previous = ref
    return legs, too_long


def prepare_transit(config: Config, *, force: bool = False) -> dict[str, Path]:
    """Extract, clip and tabulate the approximate transit network."""
    pbf_path = config.paths.raw / PBF_FILE
    buffer_path = config.paths.interim / BUFFER_FILE
    outputs = {
        "stops": config.paths.interim / STOPS_FILE,
        "routes": config.paths.interim / ROUTES_FILE,
        "legs": config.paths.interim / LEGS_FILE,
        "report": config.paths.interim / EXTRACT_REPORT_FILE,
    }

    def build() -> None:
        routes, points, without_stops = extract_routes(pbf_path)
        logger.info(
            "PBF: %d route relations with usable stops, %d without (excluded)",
            len(routes),
            without_stops,
        )

        # Clip stops to the same area the grid uses: the buffered city
        # boundary, intersected with area.bbox when one is set (sample).
        area = read_geoparquet(buffer_path).geometry.iloc[0]
        if config.area.bbox is not None:
            area = area.intersection(shapely.box(*config.area.bbox))
        inside = points[points.geometry.within(area)].copy()

        metric = inside.to_crs(config.crs.metric)
        xy = {
            int(stop): (float(point.x), float(point.y))
            for stop, point in zip(metric["stop_id"], metric.geometry, strict=True)
        }
        usable = set(xy)

        all_legs: list[dict] = []
        kept_routes: list[dict] = []
        outside_area = 0
        legs_too_long = 0
        for route in routes:
            legs, too_long = _segments(route, usable, xy)
            legs_too_long += too_long
            if not legs:
                outside_area += 1
                continue
            headway = parse_interval(route.interval)
            headway_source = "interval_tag" if headway is not None else "assumed"
            if headway is None:
                headway = config.access.transit.headways_min[route.mode]
            stops_kept = {leg["from_stop"] for leg in legs} | {leg["to_stop"] for leg in legs}
            kept_routes.append(
                {
                    "route_id": route.relation_id,
                    "mode": route.mode,
                    "ref": route.ref,
                    "name": route.name,
                    "headway_min": float(headway),
                    "headway_source": headway_source,
                    "n_stops": len(stops_kept),
                }
            )
            all_legs.extend(legs)

        routes_frame = pd.DataFrame(kept_routes, columns=list(schemas.INTERIM_TRANSIT_ROUTES))
        legs_frame = pd.DataFrame(all_legs, columns=list(schemas.INTERIM_TRANSIT_LEGS))
        used = set(legs_frame["from_stop"]) | set(legs_frame["to_stop"])
        stops_frame = inside[inside["stop_id"].isin(used)].reset_index(drop=True)

        config.paths.interim.mkdir(parents=True, exist_ok=True)
        write_geoparquet(stops_frame, outputs["stops"])
        routes_frame.to_parquet(outputs["routes"], index=False)
        legs_frame.to_parquet(outputs["legs"], index=False)
        extract_report = {
            "relations_with_stops": len(routes),
            "relations_without_stops": without_stops,
            "routes_kept": len(routes_frame),
            "routes_outside_area": outside_area,
            "routes_kept_by_mode": {
                mode: int((routes_frame["mode"] == mode).sum()) for mode in TRANSIT_MODES
            },
            "stops_kept": len(stops_frame),
            "legs": len(legs_frame),
            "legs_dropped_too_long": legs_too_long,
            # Scope matters here: the first key counts over EVERY extracted
            # relation in the PBF (country-wide for the city pipeline); only
            # the second describes the kept network the graph is built from.
            "relations_with_parseable_interval": int(
                sum(parse_interval(r.interval) is not None for r in routes)
            ),
            "routes_kept_headway_from_tag": int(
                (routes_frame["headway_source"] == "interval_tag").sum()
            ),
        }
        outputs["report"].write_text(json.dumps(extract_report, indent=2), encoding="utf-8")
        logger.info(
            "transit: kept %d routes (%s), %d stops, %d legs",
            len(routes_frame),
            ", ".join(f"{m}={extract_report['routes_kept_by_mode'][m]}" for m in TRANSIT_MODES),
            len(stops_frame),
            len(legs_frame),
        )

    ensure_built(
        list(outputs.values()),
        [pbf_path, buffer_path, config.source_path],
        build,
        force=force,
    )
    return outputs
