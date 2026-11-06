"""Phase 4 orchestrator: travel-time matrices and accessibility tables.

Two cached steps, runnable separately through ``almaty-access access``:

``times``
    Builds one time-weighted graph per mode (walk with Tobler slopes,
    bike with the linear up/down adjustment, car free-flow by road
    class, transit as the headway-based walk+ride union), then batched
    Dijkstra from every grid hex: ``tt_{mode}_r{res}.parquet`` long
    matrices plus ``nearest_r{res}.parquet`` (minutes to the nearest
    facility of each kind, walking) and ``tt_stats.json``.

``measures``
    Turns matrices into per-hex accessibility: cumulative opportunities,
    gravity (Hansen) access to jobs, normalized percentages, 2SFCA for
    capacity-constrained services and routes-per-hex. Writes
    ``access_r{8,9}.parquet`` and ``access_report.json`` (assumed-
    parameter echo, snap/coverage statistics, gravity sensitivity and
    ten seeded spot-check OD pairs for manual comparison).

Resolution policy (DECISIONS D-045): walk runs at r8 and r9 (fine maps),
bike / transit / car at r8 only — an r9 bike or car matrix would be
storage-heavy while adding nothing the r8 one does not say.
"""

from __future__ import annotations

import json
import logging

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from scipy import stats

from almaty_access.access.graph import ModeGraph, build_graph, snap
from almaty_access.access.matrix import hex_times, nearest_node_minutes
from almaty_access.access.measures import cumulative, gravity, two_step_fca
from almaty_access.access.opportunities import (
    NEAREST_TYPES,
    SERVICE_TYPES,
    SFCA_TYPES,
    load_opportunity_points,
    metric_points_xy,
    opportunity_table,
)
from almaty_access.access.speed import bike_kmh, car_kmh, seconds, tobler_walk_kmh
from almaty_access.access.transit import build_transit_graph
from almaty_access.config import Config
from almaty_access.features.build import FEATURES_FILE_TEMPLATE
from almaty_access.features.grid import GRID_FILE_TEMPLATE
from almaty_access.io import ensure_built, read_geoparquet
from almaty_access.model.build import JOBS_FILE_TEMPLATE
from almaty_access.prepare.network import interim_network_files
from almaty_access.prepare.transit import (
    EXTRACT_REPORT_FILE,
    LEGS_FILE,
    ROUTES_FILE,
    STOPS_FILE,
)

logger = logging.getLogger(__name__)

TT_FILE_TEMPLATE = "tt_{mode}_r{res}.parquet"
NEAREST_FILE_TEMPLATE = "nearest_r{res}.parquet"
TT_STATS_FILE = "tt_stats.json"
ACCESS_FILE_TEMPLATE = "access_r{res}.parquet"
REPORT_FILE = "access_report.json"

#: Which H3 resolutions each mode is computed at (see module docstring).
MODE_RESOLUTIONS: dict[str, tuple[int, ...]] = {
    "walk": (8, 9),
    "bike": (8,),
    "transit": (8,),
    "car": (8,),
}

#: Gravity sensitivity: the report compares rankings at these half-lives
#: against the configured one (README: "test sensitivity" for beta).
SENSITIVITY_HALF_LIVES = (10.0, 30.0)

N_SPOT_CHECKS = 10


def _resolutions(config: Config) -> tuple[int, int]:
    return (config.h3.analysis_resolution, config.h3.fine_resolution)


def grid_points(config: Config, res: int) -> pd.DataFrame:
    """Grid hexes with metric centroid coordinates: h3, x, y, district, in_city."""
    grid = read_geoparquet(config.paths.processed / GRID_FILE_TEMPLATE.format(res=res))
    lat_lon = [h3.cell_to_latlng(cell) for cell in grid["h3"]]
    points = gpd.GeoSeries(
        gpd.points_from_xy([p[1] for p in lat_lon], [p[0] for p in lat_lon]),
        crs=config.crs.storage,
    ).to_crs(config.crs.metric)
    return pd.DataFrame(
        {
            "h3": grid["h3"].to_numpy(),
            "x": points.x.to_numpy(),
            "y": points.y.to_numpy(),
            "district": grid["district"].to_numpy(),
            "in_city": grid["in_city"].to_numpy(),
        }
    )


def street_graph(config: Config, mode: str, edge_kmh) -> ModeGraph:
    """Load a prepared network and weight it with ``edge_kmh(edges)``."""
    files = interim_network_files(config, mode)
    nodes = read_geoparquet(files["nodes"])
    edges = read_geoparquet(files["edges"])
    kmh = edge_kmh(edges)
    return build_graph(nodes, edges, seconds(edges["length_m"].to_numpy(), kmh), config.crs.metric)


def _load_graphs(config: Config) -> tuple[dict[str, ModeGraph], bool]:
    """All four mode graphs, plus whether any transit routes exist."""
    walk_flat = config.travel.speeds_kmh["walk"]
    bike_flat = config.travel.speeds_kmh["bike"]
    graphs: dict[str, ModeGraph] = {}
    graphs["walk"] = street_graph(
        config, "walk", lambda e: tobler_walk_kmh(e["grade"].to_numpy(), walk_flat)
    )
    graphs["bike"] = street_graph(
        config, "bike", lambda e: bike_kmh(e["grade"].to_numpy(), bike_flat)
    )
    graphs["car"] = street_graph(
        config, "drive", lambda e: car_kmh(e["highway"].to_numpy(), config.access.car_speeds_kmh)
    )

    stops = read_geoparquet(config.paths.interim / STOPS_FILE)
    routes = pd.read_parquet(config.paths.interim / ROUTES_FILE)
    legs = pd.read_parquet(config.paths.interim / LEGS_FILE)
    graphs["transit"] = build_transit_graph(
        graphs["walk"],
        stops,
        routes,
        legs,
        walk_flat_kmh=walk_flat,
        speeds_kmh=config.access.transit.speeds_kmh,
        circuity=config.access.transit.circuity,
        metric_crs=config.crs.metric,
    )
    return graphs, len(legs) > 0


def _empty_tt() -> pd.DataFrame:
    return pd.DataFrame({"origin_h3": [], "dest_h3": [], "minutes": []}).astype(
        {"origin_h3": "string", "dest_h3": "string", "minutes": "float64"}
    )


def build_times(config: Config, *, force: bool = False) -> None:
    """Travel-time matrices + nearest-facility minutes (cached step)."""
    processed = config.paths.processed
    resolutions = _resolutions(config)
    outputs = [
        *(
            processed / TT_FILE_TEMPLATE.format(mode=mode, res=res)
            for mode, mode_res in MODE_RESOLUTIONS.items()
            for res in mode_res
        ),
        *(processed / NEAREST_FILE_TEMPLATE.format(res=res) for res in resolutions),
        processed / TT_STATS_FILE,
    ]
    network_inputs = [
        path
        for mode in ("walk", "bike", "drive")
        for path in interim_network_files(config, mode).values()
    ]
    inputs = [
        *network_inputs,
        *(processed / GRID_FILE_TEMPLATE.format(res=res) for res in resolutions),
        config.paths.interim / STOPS_FILE,
        config.paths.interim / ROUTES_FILE,
        config.paths.interim / LEGS_FILE,
        config.paths.interim / "pois.parquet",
        config.paths.raw / "osm_landuse.parquet",
        config.source_path,
    ]

    def build() -> None:
        cutoff = float(max(config.travel.thresholds_min))
        walk_flat = config.travel.speeds_kmh["walk"]
        graphs, transit_has_routes = _load_graphs(config)
        grids = {res: grid_points(config, res) for res in resolutions}
        stats_out: dict[str, object] = {}

        for mode, mode_res in MODE_RESOLUTIONS.items():
            for res in mode_res:
                name = TT_FILE_TEMPLATE.format(mode=mode, res=res)
                if mode == "transit" and not transit_has_routes:
                    _empty_tt().to_parquet(processed / name, index=False)
                    stats_out[name] = {"pairs": 0, "skipped_reason": "no transit routes in area"}
                    logger.warning("transit: no routes in the area, %s is empty", name)
                    continue
                table, tt_stats = hex_times(
                    graphs[mode],
                    grids[res],
                    grids[res],
                    walk_flat_kmh=walk_flat,
                    cutoff_min=cutoff,
                    snap_max_m=config.access.snap_max_m,
                )
                table.to_parquet(processed / name, index=False)
                stats_out[name] = tt_stats
                logger.info("%s: %s pairs <= %.0f min", name, f"{len(table):,}", cutoff)

        # Nearest facilities: one virtual-source Dijkstra per kind on the
        # walk graph, composed per resolution with each hex's own snap.
        points = load_opportunity_points(config)
        walk = graphs["walk"]
        node_minutes = {
            kind: nearest_node_minutes(
                walk,
                *_facility_nodes(walk, metric_points_xy(points[source], config.crs.metric)),
                walk_flat_kmh=walk_flat,
            )
            for kind, source in NEAREST_TYPES.items()
        }
        for res in resolutions:
            grid = grids[res]
            nodes, dist = snap(walk, grid[["x", "y"]].to_numpy())
            snap_min = dist / (walk_flat * 1000.0 / 60.0)
            offnet = dist > config.access.snap_max_m
            frame = pd.DataFrame({"h3": grid["h3"]})
            for kind, minutes in node_minutes.items():
                values = snap_min + minutes[nodes]
                values[offnet | ~np.isfinite(values)] = np.nan
                frame[f"near_{kind}_walk_min"] = np.round(values, 2)
            frame.to_parquet(processed / NEAREST_FILE_TEMPLATE.format(res=res), index=False)

        (processed / TT_STATS_FILE).write_text(json.dumps(stats_out, indent=2), encoding="utf-8")

    ensure_built(outputs, inputs, build, force=force)


def _facility_nodes(graph: ModeGraph, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(xy):
        return np.array([], dtype="int64"), np.array([])
    return snap(graph, xy)


def _tt_path(config: Config, mode: str, res: int):
    return config.paths.processed / TT_FILE_TEMPLATE.format(mode=mode, res=res)


def build_measures(config: Config, *, force: bool = False) -> None:
    """Accessibility tables + report from the stored matrices (cached step)."""
    processed = config.paths.processed
    resolutions = _resolutions(config)
    outputs = [
        *(processed / ACCESS_FILE_TEMPLATE.format(res=res) for res in resolutions),
        processed / REPORT_FILE,
    ]
    inputs = [
        *(
            _tt_path(config, mode, res)
            for mode, mode_res in MODE_RESOLUTIONS.items()
            for res in mode_res
        ),
        *(processed / NEAREST_FILE_TEMPLATE.format(res=res) for res in resolutions),
        *(processed / GRID_FILE_TEMPLATE.format(res=res) for res in resolutions),
        *(processed / JOBS_FILE_TEMPLATE.format(res=res) for res in resolutions),
        *(processed / FEATURES_FILE_TEMPLATE.format(res=res) for res in resolutions),
        processed / TT_STATS_FILE,
        config.paths.interim / "pois.parquet",
        config.paths.raw / "osm_landuse.parquet",
        config.paths.interim / STOPS_FILE,
        config.paths.interim / LEGS_FILE,
        config.paths.interim / ROUTES_FILE,
        config.paths.interim / EXTRACT_REPORT_FILE,
        config.source_path,
    ]

    def build() -> None:
        thresholds = config.travel.thresholds_min
        half_life = config.access.gravity_half_life_min
        points = load_opportunity_points(config)
        report: dict[str, object] = {
            "approximate_transit": True,
            "params": {
                "cutoff_min": float(max(thresholds)),
                "thresholds_min": list(thresholds),
                "speeds_kmh": dict(sorted(config.travel.speeds_kmh.items())),
                "transit_assumed_speeds_kmh": config.access.transit.speeds_kmh,
                "transit_assumed_headways_min": config.access.transit.headways_min,
                "transit_circuity": config.access.transit.circuity,
                "gravity_half_life_min": half_life,
                "sfca_catchment_min": config.access.sfca_catchment_min,
                "park_min_m2": config.access.park_min_m2,
                "snap_max_m": config.access.snap_max_m,
                "car_speeds_kmh": config.access.car_speeds_kmh,
            },
            "transit_network": json.loads(
                (config.paths.interim / EXTRACT_REPORT_FILE).read_text("utf-8")
            ),
            "matrices": json.loads((processed / TT_STATS_FILE).read_text("utf-8")),
        }

        totals: dict[str, float] = {}
        coverage: dict[str, object] = {}
        for res in resolutions:
            grid = read_geoparquet(config.paths.processed / GRID_FILE_TEMPLATE.format(res=res))
            table = grid[["h3", "res", "district", "in_city"]].copy()
            index = pd.Index(table["h3"], name="h3")
            opportunities = opportunity_table(config, points, res, table["h3"])
            population = pd.read_parquet(
                processed / FEATURES_FILE_TEMPLATE.format(res=res),
                columns=["h3", "pop_residents"],
            ).set_index("h3")["pop_residents"]
            total_jobs = float(opportunities.loc[table.loc[table["in_city"], "h3"], "jobs"].sum())
            totals[f"r{res}"] = round(total_jobs, 1)

            modes = [mode for mode, mode_res in MODE_RESOLUTIONS.items() if res in mode_res]
            tt_by_mode = {mode: pd.read_parquet(_tt_path(config, mode, res)) for mode in modes}

            for mode in modes:
                tt = tt_by_mode[mode]
                cum = cumulative(tt, opportunities["jobs"], thresholds)
                for threshold in thresholds:
                    values = cum[threshold].reindex(index).fillna(0.0)
                    table[f"acc_jobs_{mode}_{threshold}"] = np.round(values.to_numpy(), 1)
                    table[f"accpct_jobs_{mode}_{threshold}"] = np.round(
                        values.to_numpy() / total_jobs * 100.0, 2
                    )
                grav = gravity(tt, opportunities["jobs"], half_life).reindex(index).fillna(0.0)
                table[f"acc_jobs_{mode}_grav"] = np.round(grav.to_numpy(), 1)

            walk_tt = tt_by_mode["walk"]
            for service in SERVICE_TYPES:
                cum = cumulative(walk_tt, opportunities[service], thresholds)
                for threshold in thresholds:
                    table[f"acc_{service}_walk_{threshold}"] = (
                        cum[threshold].reindex(index).fillna(0.0).to_numpy()
                    )
            for service in SFCA_TYPES:
                sfca = two_step_fca(
                    walk_tt,
                    opportunities[service],
                    population,
                    float(config.access.sfca_catchment_min),
                )
                table[f"sfca_{service}_walk"] = np.round(
                    sfca.reindex(index).fillna(0.0).to_numpy(), 4
                )

            nearest = pd.read_parquet(processed / NEAREST_FILE_TEMPLATE.format(res=res))
            table = table.merge(nearest, on="h3", how="left")

            table["transit_routes_cnt"] = (
                _routes_per_hex(config, res).reindex(index).fillna(0).to_numpy(dtype="int64")
            )

            table.to_parquet(processed / ACCESS_FILE_TEMPLATE.format(res=res), index=False)
            logger.info("access_r%d: %d hexes x %d columns", res, len(table), len(table.columns))

            in_city = table["in_city"].to_numpy()
            coverage[f"r{res}"] = {
                mode: {
                    "share_reaching_jobs_30": round(
                        float((table.loc[in_city, f"acc_jobs_{mode}_30"] > 0).mean()), 4
                    ),
                    "median_jobs_45": round(
                        float(table.loc[in_city, f"acc_jobs_{mode}_45"].median()), 1
                    ),
                }
                for mode in modes
            }

            if res == config.h3.analysis_resolution:
                report["gravity_sensitivity"] = _gravity_sensitivity(
                    tt_by_mode["walk"], opportunities["jobs"], half_life, index
                )
                report["spot_checks"] = _spot_checks(config, table, tt_by_mode)

        report["total_jobs"] = totals
        report["coverage"] = coverage
        (processed / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")

    ensure_built(outputs, inputs, build, force=force)


def _routes_per_hex(config: Config, res: int) -> pd.Series:
    """Distinct transit routes with a stop in each hex (deferred from Phase 2)."""
    stops = read_geoparquet(config.paths.interim / STOPS_FILE)
    legs = pd.read_parquet(config.paths.interim / LEGS_FILE)
    if not len(stops) or not len(legs):
        return pd.Series(dtype="int64")
    cells = pd.Series(
        [h3.latlng_to_cell(point.y, point.x, res) for point in stops.geometry],
        index=stops["stop_id"].to_numpy(),
        name="h3",
    )
    pairs = pd.concat(
        [
            legs[["route_id", "from_stop"]].rename(columns={"from_stop": "stop_id"}),
            legs[["route_id", "to_stop"]].rename(columns={"to_stop": "stop_id"}),
        ]
    ).drop_duplicates()
    pairs["h3"] = cells.reindex(pairs["stop_id"]).to_numpy()
    return pairs.groupby("h3")["route_id"].nunique()


def _gravity_sensitivity(
    walk_tt: pd.DataFrame, jobs: pd.Series, half_life: float, index: pd.Index
) -> dict:
    """Does the configured half-life drive the RANKING? (README §4.3.2)."""
    central = gravity(walk_tt, jobs, half_life).reindex(index).fillna(0.0)
    out: dict[str, object] = {"half_life_min": half_life, "spearman_vs": {}}
    for other in SENSITIVITY_HALF_LIVES:
        variant = gravity(walk_tt, jobs, other).reindex(index).fillna(0.0)
        rho = stats.spearmanr(central.to_numpy(), variant.to_numpy()).statistic
        out["spearman_vs"][f"half_life_{other:g}"] = round(float(rho), 4)
    return out


def _spot_checks(
    config: Config, table: pd.DataFrame, tt_by_mode: dict[str, pd.DataFrame]
) -> list[dict]:
    """Ten seeded OD pairs for manual comparison against a routing app.

    Stratified (D-051): half the pairs are drawn from the WALK matrix, so
    the slow modes that most need human checking carry actual minutes —
    a uniform draw across a 30 km city leaves 9 of 10 pairs beyond the
    walking cutoff. The other half are unconstrained city pairs (good
    for car and transit at city scale).
    """
    rng = np.random.default_rng(config.random_seed)
    city_hexes = table.loc[table["in_city"], "h3"].to_numpy()
    if len(city_hexes) < 2:
        return []
    city_set = set(city_hexes)
    indexed = {
        mode: tt.set_index(["origin_h3", "dest_h3"])["minutes"] for mode, tt in tt_by_mode.items()
    }

    walk = tt_by_mode["walk"]
    walkable = walk.loc[
        (walk["origin_h3"] != walk["dest_h3"])
        & walk["origin_h3"].isin(city_set)
        & walk["dest_h3"].isin(city_set)
    ]
    n_near = min(N_SPOT_CHECKS // 2, len(walkable))
    pairs: list[tuple[str, str]] = []
    if n_near:
        picked = walkable.iloc[rng.choice(len(walkable), size=n_near, replace=False)]
        pairs.extend(zip(picked["origin_h3"], picked["dest_h3"], strict=True))
    while len(pairs) < N_SPOT_CHECKS:
        origin, dest = rng.choice(city_hexes, size=2, replace=False)
        pairs.append((str(origin), str(dest)))

    checks: list[dict] = []
    for origin, dest in pairs:
        o_lat, o_lon = h3.cell_to_latlng(origin)
        d_lat, d_lon = h3.cell_to_latlng(dest)
        minutes = {}
        for mode, series in indexed.items():
            value = series.get((origin, dest))
            minutes[mode] = None if value is None or pd.isna(value) else round(float(value), 1)
        checks.append(
            {
                "origin_h3": str(origin),
                "dest_h3": str(dest),
                "origin_latlng": [round(o_lat, 5), round(o_lon, 5)],
                "dest_latlng": [round(d_lat, 5), round(d_lon, 5)],
                "minutes": minutes,
            }
        )
    return checks
